"""GUI widgets: piano roll, waveform, results panel, live indicator, error banner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal

from ..core.figures import figures_summary, short_label
from ..core.accuracy import AccuracyHistory
from ..core.continuity import coherence_report
from ..core.intonation import IN_TUNE_CENTS
from ..core.i18n import note_display_pair, tr, tr_chord_label, tr_figure, tr_key_label
from ..core.music import midi_to_name

# Confidence color ramp (red -> green)
LOW_COLOR = (220, 80, 80)
HIGH_COLOR = (80, 220, 120)
SILENCE_COLOR = (120, 120, 130)
MATCH_COLOR = (80, 210, 120)
MISMATCH_COLOR = (220, 80, 80)
AMBER_COLOR = (230, 180, 80)
CENTER_CENTS = IN_TUNE_CENTS   # in tune (green) — the scale intonation.py owns
NEAR_CENTS = 25.0     # <= 25 cents: close (amber); beyond = red


def cents_deviation(midi_float: float) -> float:
    """Deviation of a continuous pitch from its nearest piano note, in cents."""
    return (midi_float - round(midi_float)) * 100.0


def tuning_color(cents: float | None) -> tuple[int, int, int]:
    """Green/amber/red for a cents deviation — the tuning meter's palette.

    Shared by the TuningMeter needle and the live piano key, so both
    feedbacks read the same scale.
    """
    if cents is None:
        return MISMATCH_COLOR
    a = abs(cents)
    if a <= CENTER_CENTS:
        return MATCH_COLOR
    if a <= NEAR_CENTS:
        return AMBER_COLOR
    return MISMATCH_COLOR

# Colors per note figure (keys match the figure names produced by figures.py)
FIGURE_COLORS = {
    "ronde": (240, 240, 160),
    "blanche": (120, 220, 220),
    "noire": (120, 180, 255),
    "croche": (170, 140, 255),
    "double-croche": (255, 140, 200),
    "triolet": (255, 190, 120),
    "pointée": (200, 200, 200),
}
UNCLASSIFIED_COLOR = (120, 120, 130)


def figure_color(figure: str | None) -> tuple:
    """Color for a figure name (fall back to a generic gray when unknown)."""
    if not figure:
        return UNCLASSIFIED_COLOR + (200,)
    if figure.startswith("triolet"):
        key = "triolet"
    elif figure.endswith("pointée"):
        key = "pointée"
    else:
        key = figure
    return FIGURE_COLORS.get(key, UNCLASSIFIED_COLOR) + (210,)


class PitchAxis(pg.AxisItem):
    """Left axis of the roll: note names (C4, D4…) instead of MIDI numbers.

    The pitch of a note is what a musician reads, and the keyboard below the
    roll says the same thing — raw numbers only force a conversion the ear
    has already made.
    """

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        return [midi_to_name(int(round(v))) for v in values]


class PianoRoll(pg.PlotWidget):
    """Piano roll: rectangles = notes, curve = fine pitch.

    In "notation" mode each note is colored by its figure and carries a
    short label (R, B, N, C, DC, 3N, trailing dot = dotted).
    A playhead line tracks playback; double-click seeks (seek_requested).

    A SELECTABLE roll (the results page) also selects on the graph:
    press a bar to select that note, drag across neighbours to extend
    the selection, click empty space to clear. The gesture EMITS
    `note_selection` (indices, [] = cleared) — the notes table remains
    the one selection owner, this widget only renders what it says.
    """

    seek_requested = pg.QtCore.pyqtSignal(float)
    note_selection = pg.QtCore.pyqtSignal(list)  # selected bar indices; [] = cleared

    MIDI_MIN, MIDI_MAX = 21, 108
    LIVE_SPAN = 10.0      # seconds of history kept visible during live capture
    LIVE_Y_MIN = 14.0     # minimum visible pitch band, in semitones

    def __init__(self, parent=None, *, selectable: bool = False):
        super().__init__(parent, axisItems={"left": PitchAxis(orientation="left")})
        self.setBackground(BG_APP)
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.12)
        self.set_axis_labels()   # titled by the window's retranslate()
        self._bar_items: list = []
        self._labels: list = []
        self._diff_items: list = []  # live-vs-final outlines (outline_note)
        self._curve = None
        self._notation_mode = False
        self._last_notes: list = []
        self._rendered_figures: list = []
        self._selectable = selectable
        self._selection: list[int] = []
        self._drag_anchor: int | None = None  # gesture started on this bar
        self._press_pos: tuple[float, float] | None = None  # empty press: click vs pan
        self._pan_moved = False
        self._live_span = self.LIVE_SPAN
        self._live_preview = None
        self._live_preview_midi: int | None = None
        self._live_band: tuple[float, float] | None = None
        self.setLimits(xMin=0)
        # playhead: an orange vertical line + a glowing marker on the notes
        self._play_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen((255, 150, 60), width=1.6)
        )
        self._play_line.hide()
        self.addItem(self._play_line)
        self._play_marker = pg.TextItem("▶", color=(255, 150, 60), anchor=(0.5, 1.0))
        self._play_marker.hide()
        self.addItem(self._play_marker)
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_axis_labels(self, note: str = "", time: str = "") -> None:
        """Axis titles in the current language, and in seconds on the bottom.

        The pitch axis prints note NAMES (`PitchAxis`); these two titles are
        the only other text in the plot, so they follow the language like
        every label — and `tr()` is only safe after `set_language`, hence a
        `retranslate()` call rather than a default at construction.
        """
        self.getPlotItem().setLabel("left", note)
        self.getPlotItem().setLabel("bottom", time, units="s")

    def _move_playhead(self, t: float) -> None:
        """Place the playhead line + marker on the timeline and show them."""
        self._play_line.setPos(t)
        self._play_marker.setPos(t, 109.0)
        self._play_line.show()
        self._play_marker.show()

    def set_playhead(self, t: float, follow: bool = True) -> None:
        """Move the playhead; with `follow`, scroll the view along with it.

        `follow=False` (a seek) moves the playhead only: scrolling there too
        would yank the view away from what the user just clicked.
        """
        self._move_playhead(t)
        vb = self.getPlotItem().vb
        (x0, x1), _ = vb.viewRange()
        if follow and not (x0 <= t <= x1):
            span = x1 - x0
            vb.setXRange(max(0.0, t - span * 0.1), t + span * 0.9, padding=0)

    def hide_playhead(self) -> None:
        self._play_line.hide()
        self._play_marker.hide()

    # --------------------------------------------------------------- live
    def begin_live(self, span: float | None = None) -> None:
        """Reset the roll for a live take: empty timeline, fixed-width window."""
        self._live_span = span or self.LIVE_SPAN
        self.set_live_preview(None)
        self._clear_notes()
        self._last_notes = []
        self._move_playhead(0.0)
        self.getPlotItem().vb.setXRange(0.0, self._live_span, padding=0)
        self._live_band = None
        self.set_live_pitch_range([48, 84])  # default frame, refined while singing

    def set_live_cursor(self, t: float) -> None:
        """Live capture: scroll a fixed-width window so the cursor stays visible."""
        self._move_playhead(t)
        span = self._live_span
        x1 = t + span * 0.12  # a little look-ahead so the cursor is not at the edge
        self.getPlotItem().vb.setXRange(max(0.0, x1 - span), x1, padding=0)

    def set_live_pitch_range(self, midis) -> None:
        """Frame the live view on the pitches actually being sung.

        Y auto-range would follow every octave error of the live pitch
        cloud and squash the roll into a sliver; a stable band around the
        segmented notes keeps the follow readable.
        """
        vals = [int(m) for m in midis]
        if not vals:
            return
        lo, hi = float(min(vals) - 4), float(max(vals) + 4)
        if hi - lo < self.LIVE_Y_MIN:
            mid = (lo + hi) / 2
            lo, hi = mid - self.LIVE_Y_MIN / 2, mid + self.LIVE_Y_MIN / 2
        band = (max(float(self.MIDI_MIN), lo), min(float(self.MIDI_MAX), hi))
        if band == self._live_band:
            return  # nothing moved: leave the view alone
        self._live_band = band
        self.getPlotItem().vb.setYRange(band[0], band[1], padding=0)

    def set_live_preview(self, note) -> None:
        """Draw the note being sung right now (created once, then updated).

        Notes only enter the segmenter's list when they end, so without this
        preview a long note stays invisible until it is over.
        """
        if note is None:
            if self._live_preview is not None:
                self._live_preview.hide()
            return
        if self._live_preview is None:
            self._live_preview = pg.BarGraphItem(
                x0=[note.start],
                x1=[note.end],
                y0=[note.midi - 0.4],
                y1=[note.midi + 0.4],
                brush=(255, 200, 110, 170),
                pen=pg.mkPen(None),
            )
            self.addItem(self._live_preview)
            self._live_preview_midi = note.midi
        elif note.midi != self._live_preview_midi:
            self._live_preview.setOpts(y0=[note.midi - 0.4], y1=[note.midi + 0.4])
            self._live_preview_midi = note.midi
        self._live_preview.setOpts(x0=[note.start], x1=[note.end])
        self._live_preview.show()

    def _on_click(self, event) -> None:
        """Click in the plot seeks the playhead via the double-click hook."""
        if event.double():
            pos = event.scenePos()
            if self.sceneBoundingRect().contains(pos):
                t = self.getPlotItem().vb.mapSceneToView(pos).x()
                if t >= 0:
                    self.seek_requested.emit(float(t))

    # ------------------------------------------------------- note selection
    def _note_at(self, t: float, midi: float, tol: float = 0.6) -> int | None:
        """Index of the note under (time, pitch); nearest pitch on ties.

        A bar spans midi ±0.4, so `tol` is the pick margin in semitones:
        the event path widens it to a few SCREEN pixels at the current
        zoom (zoomed out, a bar shrinks below its own half-height), while
        the default keeps hit-testing exact.
        """
        best: int | None = None
        best_d: float | None = None
        for i, n in enumerate(self._last_notes):
            if not (n.start <= t <= n.end):
                continue
            d = abs(midi - n.midi)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = i, d
        return best

    def _hit(self, event) -> int | None:
        """The note under the event, tolerance scaled to the view's zoom."""
        vb = self.getPlotItem().vb
        data = vb.mapSceneToView(self.mapToScene(event.position().toPoint()))
        (_x0, _x1), (y0, y1) = vb.viewRange()
        per_px = (y1 - y0) / max(1, vb.height())
        return self._note_at(float(data.x()), float(data.y()), tol=max(0.6, per_px * 8))

    def set_selection(self, indices) -> None:
        """Highlight the selected bars — the notes TABLE's selection, rendered.

        The page calls this when the table's selection changes; a roll
        gesture instead EMITS `note_selection` so the table applies it —
        one selection (the table), two views, no mirrored state here.

        On a non-selectable roll (the live page has no table) nothing is
        ever highlighted: `selectable` gates BOTH directions.
        """
        if not self._selectable:
            return
        self._selection = sorted({int(i) for i in indices})
        self._render_selection()

    def set_selectable(self, enabled: bool) -> None:
        """Turn graph selection on or off (the live page's take lifecycle).

        The results page builds its roll selectable and never changes it;
        the live roll starts frozen out and becomes selectable only when a
        capture has STOPPED — the closed notes are then explorable, and
        starting the next take clears whatever was picked.
        """
        if enabled == self._selectable:
            return
        self._selectable = enabled
        self._selection = []
        self._drag_anchor = None
        self._press_pos = None
        self._render_selection()  # any highlight goes back to the own colors

    def _render_selection(self) -> None:
        """Rebrush only the bars whose state changed (a click on a big
        take must not repaint 3000 pictures) — the RAW color goes into
        `opts["brush"]`, exactly as `_add_note` stores it.
        """
        chosen = set(self._selection)
        for i, item in enumerate(self._bar_items):
            if i >= len(self._last_notes):
                break  # defensive: bars and notes are built together
            want = ACCENT if i in chosen else self._color_for(self._last_notes[i])
            if item.opts.get("brush") != want:
                item.setOpts(brush=want)

    def _emit_selection(self, indices: list[int]) -> None:
        """Publish a gesture's selection (silent when nothing changed)."""
        if indices == self._selection:
            return
        self._selection = indices
        self._render_selection()
        self.note_selection.emit(list(indices))

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Press on a bar selects it (a drag then extends it to a range).

        A press on EMPTY space stays pyqtgraph's own gesture — the view
        keeps panning the timeline — and its click (no movement) clears
        the selection. On a non-selectable roll (the live page) every
        event passes through untouched.
        """
        if self._selectable and event.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            idx = self._hit(event)
            if idx is not None:
                self._press_pos = None
                self._drag_anchor = idx
                self._emit_selection([idx])
                event.accept()
                return
            self._press_pos = (event.position().x(), event.position().y())
            self._pan_moved = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._drag_anchor is not None:
            idx = self._hit(event)
            if idx is not None:  # over a gap: keep the range built so far
                lo, hi = sorted((self._drag_anchor, idx))
                self._emit_selection(list(range(lo, hi + 1)))
            event.accept()
            return
        if self._press_pos is not None:
            x0, y0 = self._press_pos
            if abs(event.position().x() - x0) + abs(event.position().y() - y0) > 4:
                self._pan_moved = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            if self._drag_anchor is not None:
                self._drag_anchor = None
                event.accept()
                return
            clicked_empty = self._press_pos is not None and not self._pan_moved
            self._press_pos = None
            if clicked_empty:
                self._emit_selection([])  # a click on empty space clears
        super().mouseReleaseEvent(event)

    def set_notation_mode(self, enabled: bool) -> None:
        if enabled == self._notation_mode:
            return  # idempotent: live mode re-checks on every note-on
        self._notation_mode = enabled
        if self._last_notes:
            self.set_notes(self._last_notes, force=True)  # recolor every bar

    def set_notes(self, notes, force: bool = False) -> None:
        notes = list(notes)
        # fast path for the live scroll: the list only grows, so append the
        # new bars instead of rebuilding every Qt item on each note event.
        # Figures are labeled in place after the fact, so the shared prefix
        # must be unchanged AND still carry the figures already rendered.
        n_old = len(self._last_notes)
        figures = [n.figure for n in notes]
        if (
            not force
            and n_old
            and len(notes) >= n_old
            and all(a is b for a, b in zip(self._last_notes, notes[:n_old]))
            and figures[:n_old] == self._rendered_figures
        ):
            for n in notes[n_old:]:
                self._add_note(n)
            self._last_notes = notes
            self._rendered_figures = figures
            return
        self._last_notes = notes
        self._clear_notes()
        for n in notes:
            self._add_note(n)
        self._rendered_figures = figures
        self._render_selection()  # a rebuild keeps the table's selection visible

    def _color_for(self, n) -> tuple:
        """The bar's own color: figure in notation mode, confidence ramp else."""
        return figure_color(n.figure) if self._notation_mode else self._color_for_conf(n.confidence)

    def _add_note(self, n) -> None:
        item = pg.BarGraphItem(
            x0=[n.start],
            x1=[n.end],
            y0=[n.midi - 0.4],
            y1=[n.midi + 0.4],
            brush=self._color_for(n),
            pen=pg.mkPen(None),
        )
        self.addItem(item)
        self._bar_items.append(item)
        # label only when the bar is wide enough to carry it
        if self._notation_mode and n.figure and (n.end - n.start) > 0.12:
            lbl = pg.TextItem(short_label(n.figure), color=(235, 235, 235))
            lbl.setPos((n.start + n.end) / 2 - 0.03, n.midi + 0.45)
            self.addItem(lbl)
            self._labels.append(lbl)

    def set_pitch_curve(self, track) -> None:
        if self._curve is not None:
            self.removeItem(self._curve)
            self._curve = None
        if track is None:
            return
        with np.errstate(invalid="ignore"):
            midi = 69.0 + 12.0 * np.log2(np.asarray(track.hz, dtype=float) / 440.0)
        self._curve = self.plot(
            np.asarray(track.times, dtype=float),
            midi,
            pen=pg.mkPen((255, 210, 120, 160), width=1.0),
            connect="finite",  # NaN gaps break the line instead of plunging
        )

    def _color_for_conf(self, conf: float) -> tuple:
        """Interpolate the confidence ramp; near-silence stays gray."""
        if conf <= 0.05:
            return SILENCE_COLOR + (180,)
        conf = min(1.0, max(0.0, conf))
        r = int(LOW_COLOR[0] + (HIGH_COLOR[0] - LOW_COLOR[0]) * conf)
        g = int(LOW_COLOR[1] + (HIGH_COLOR[1] - LOW_COLOR[1]) * conf)
        b = int(LOW_COLOR[2] + (HIGH_COLOR[2] - LOW_COLOR[2]) * conf)
        return (r, g, b, 220)

    def outline_note(self, start: float, end: float, midi: int, color) -> None:
        """Draw an OUTLINE over one note's band (the live-vs-final diff mark).

        The final analysis replaced this note (pitch moved, or onset beyond
        the grid tolerance): the outline shows the singer WHERE, without
        repainting the final bars — the roll reads "here is what I corrected".
        Marks are cleared by `_clear_notes` (next `set_notes` rebuild) and by
        `begin_live` (a new take starts with a clean roll).
        """
        # `color` may be a hex string (the palette's ACCENT) or an RGB tuple:
        # mkPen accepts both, tuple() would SHATTER a string into chars and
        # the mark silently never drew (the try/except in the caller ate it)
        pen = pg.mkPen(color, width=2)
        item = pg.LinearRegionItem(
            values=[start, max(end, start + 0.02)],
            orientation="vertical",
            brush=None,
            pen=pen,
            movable=False,
        )
        item.setZValue(5)
        self.addItem(item)
        self._diff_items.append(item)

    def _clear_notes(self) -> None:
        for it in self._bar_items:
            self.removeItem(it)
        for it in self._labels:
            self.removeItem(it)
        for it in self._diff_items:
            self.removeItem(it)
        self._bar_items = []
        self._labels = []
        self._diff_items = []
        self._rendered_figures = []


class WaveformView(pg.PlotWidget):
    """Waveform overview with a playback/analysis position marker."""

    MAX_POINTS = 50_000  # plenty at screen resolution

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(BG_APP)
        self.getPlotItem().hideAxis("left")
        self.getPlotItem().setLabel("bottom", "Temps", units="s")
        self._curves: list = []  # min and max envelope curves
        self._line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 170, 80, 220), width=1.2))
        self.addItem(self._line)

    def set_signal(self, y: np.ndarray, sr: int) -> None:
        """Draw the overview as a min/max envelope.

        Taking every Nth sample (the obvious decimation) drops the peaks and
        makes a long take look flat; per-bucket min/max keeps the real
        dynamics at any length. The two curves read as the filled band of
        the usual waveform overview.
        """
        y = np.asarray(y, dtype=float)
        step = max(1, len(y) // self.MAX_POINTS)
        if step > 1:
            n_buckets = len(y) // step
            buckets = y[: n_buckets * step].reshape(n_buckets, step)
            t = (np.arange(n_buckets) + 0.5) * step / sr
            lo, hi = buckets.min(axis=1), buckets.max(axis=1)
        else:
            t, lo, hi = np.arange(len(y)) / sr, y, y
        for curve in self._curves:
            self.removeItem(curve)
        pen = pg.mkPen((140, 200, 255, 200), width=0.8)
        self._curves = [self.plot(t, lo, pen=pen), self.plot(t, hi, pen=pen)]
        self.setXRange(0, len(y) / sr, padding=0)

    def set_position(self, t: float) -> None:
        self._line.setPos(t)


class ResultsPanel(pg.LayoutWidget):
    """Side panel: tempo, key, chords, stats + per-note and per-segment tables.

    All labels come from i18n.tr() and refresh via retranslate().
    """

    # The value rows, top to bottom — the i18n keys themselves. Slots
    # (_values/_labels indexes) share the key names, so one literal serves
    # both, and the tr() key guard sees every key here statically.
    SLOT_KEYS = (
        "key source",
        "key duration",
        "key tempo",
        "key meter",
        "key key",
        "key key score",
        "key alternatives",
        "key chord",
        "key notes count",
        "key measures count",
        "key figures",
        "key tuning offset",
        "key coherence",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._title = pg.QtWidgets.QLabel("")
        self._title.setStyleSheet("font-size: 14pt; color: #e6ebf2; padding-bottom: 6px;")
        self.addWidget(self._title, row=0, col=0, colspan=2)
        # empty state: a centered invitation shown before any analysis
        self._waiting = pg.QtWidgets.QLabel("")
        self._waiting.setAlignment(pg.QtCore.Qt.AlignmentFlag.AlignCenter)
        self._waiting.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10pt; font-style: italic;"
            f"background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 12px;"
        )
        self._waiting.setVisible(False)
        self.addWidget(self._waiting, row=len(self.SLOT_KEYS) + 3, col=0, colspan=2)
        self._labels: dict[str, pg.QtWidgets.QLabel] = {}
        self._values: dict[str, pg.QtWidgets.QLabel] = {}
        for i, slot in enumerate(self.SLOT_KEYS, start=1):
            kl = pg.QtWidgets.QLabel("")
            vl = pg.QtWidgets.QLabel("—")
            kl.setTextInteractionFlags(pg.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            vl.setTextInteractionFlags(pg.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            vl.setWordWrap(True)
            self.addWidget(kl, row=i, col=0)
            self.addWidget(vl, row=i, col=1)
            self._labels[slot] = kl
            self._values[slot] = vl

        # Per-note table: position (measure:beat) + figure
        r = len(self.SLOT_KEYS) + 1
        self._notes_table = self._make_table(4)
        self._lbl_notes = pg.QtWidgets.QLabel("")
        self.addWidget(self._lbl_notes, row=r, col=0)
        self.addWidget(self._notes_table, row=r + 1, col=0, colspan=2)
        # the live-vs-final explainer, one line per cause (a live take
        # only): hidden for a file analysis, where nothing moved
        self._lbl_explain = pg.QtWidgets.QLabel("")
        self._lbl_explain.setWordWrap(True)
        self._lbl_explain.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 8pt; font-style: italic;"
            f"background: {BG_CARD}; border: 1px solid {BORDER};"
            "border-radius: 8px; padding: 8px;"
        )
        self._lbl_explain.setVisible(False)
        self.addWidget(self._lbl_explain, row=r + 2, col=0, colspan=2)
        self._has_results: bool = False

        # Per-segment chord table
        r2 = r + 4
        self._seg_table = self._make_table(2)
        self._lbl_segs = pg.QtWidgets.QLabel("")
        self.addWidget(self._lbl_segs, row=r2, col=0)
        self.addWidget(self._seg_table, row=r2 + 1, col=0, colspan=2)
        self.retranslate()

    @staticmethod
    def _make_table(ncols: int) -> pg.QtWidgets.QTableWidget:
        table = pg.QtWidgets.QTableWidget(0, ncols)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(pg.QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setMaximumHeight(140)
        return table

    def retranslate(self) -> None:
        """Refresh every label in the current language (labels only).

        One literal line per row ON PURPOSE: the tr() key guard
        (tests/test_docs_guard.py) only sees literal arguments, so a typo'd
        key fails THERE instead of reaching the user as a raw label — a
        loop over SLOT_KEYS would be an invisible dynamic call.
        """
        self._title.setText(f"<b>{tr('key results title')}</b>")
        self._waiting.setText(tr("ui panel waiting"))
        self._waiting.setVisible(self._has_results is False)
        self._labels["key source"].setText(f"<b>{tr('key source')}</b>")
        self._labels["key duration"].setText(f"<b>{tr('key duration')}</b>")
        self._labels["key tempo"].setText(f"<b>{tr('key tempo')}</b>")
        self._labels["key meter"].setText(f"<b>{tr('key meter')}</b>")
        self._labels["key key"].setText(f"<b>{tr('key key')}</b>")
        self._labels["key key score"].setText(f"<b>{tr('key key score')}</b>")
        self._labels["key alternatives"].setText(f"<b>{tr('key alternatives')}</b>")
        self._labels["key chord"].setText(f"<b>{tr('key chord')}</b>")
        self._labels["key notes count"].setText(f"<b>{tr('key notes count')}</b>")
        self._labels["key measures count"].setText(f"<b>{tr('key measures count')}</b>")
        self._labels["key figures"].setText(f"<b>{tr('key figures')}</b>")
        self._labels["key tuning offset"].setText(f"<b>{tr('key tuning offset')}</b>")
        self._labels["key coherence"].setText(f"<b>{tr('key coherence')}</b>")
        self._lbl_notes.setText(f"<b>{tr('key notes table')}</b>")
        self._lbl_segs.setText(f"<b>{tr('key chord segments')}</b>")
        self._notes_table.setHorizontalHeaderLabels(
            ["#", f"{tr('key meter')}:{tr('key time col')}", tr("key note col"), tr("key figure col")]
        )
        self._seg_table.setHorizontalHeaderLabels([tr("key span col"), tr("key chord")])
        self._notes_table.resizeColumnsToContents()
        self._seg_table.resizeColumnsToContents()

    def set_results(self, res) -> None:
        self._has_results = res is not None
        self._waiting.setVisible(not self._has_results)
        if res is None:
            for vl in self._values.values():
                vl.setText("—")
            return
        self._values["key source"].setText(Path(res.source).name)
        self._values["key duration"].setText(f"{res.duration_s:.2f} s")
        self._values["key tempo"].setText(f"{res.tempo.bpm:.1f} BPM" if res.tempo else "—")
        self._values["key meter"].setText(
            res.meter.label if getattr(res, "meter", None) else tr("key default meter")
        )
        if getattr(res, "note_positions", None) and res.tempo:
            self._values["key measures count"].setText(
                str(getattr(res, "num_measures", 0) or len(res.note_positions))
            )
        else:
            self._values["key measures count"].setText("—")
        # display translates the stored canonical labels ('C major' ->
        # 'C majeur'), like the staff header already does
        self._values["key key"].setText(
            tr_key_label(res.key.best.label) if res.key else "—"
        )
        self._values["key key score"].setText(f"{res.key.best.score:.2f}" if res.key else "—")
        alts = (
            ", ".join(tr_key_label(a.label) for a in res.key.alternatives)
            if res.key and res.key.alternatives
            else "—"
        )
        self._values["key alternatives"].setText(alts)

        # Global chord + runner-ups (labels translated at display, like keys)
        chord = getattr(res, "chord", None)
        if chord is not None:
            txt = tr_chord_label(chord.best.label)
            if chord.alternatives:
                txt += "  (" + ", ".join(tr_chord_label(c.label) for c in chord.alternatives) + ")"
            self._values["key chord"].setText(txt)
        else:
            self._values["key chord"].setText("—")

        self._values["key notes count"].setText(str(len(res.notes)))
        # The GLOBAL tuning offset (cents) the analysis absorbed, with a
        # one-line intonation advice for the singer. |offset| <= 10 c is
        # dead-on (a professional reads ±5-10); 10-25 c is a mild pull to
        # correct at the next take; beyond 25 c the take sits noticeably
        # off — and beyond the ±35 st*100 cap it was treated as a
        # transposition, not intonation, so nothing was absorbed.
        cents = getattr(res, "tuning_offset_cents", None)
        if cents is None:
            cents = round(100.0 * getattr(res, "tuning_offset_st", 0.0), 1)
        if cents is None:
            self._values["key tuning offset"].setText("—")
        else:
            c = float(cents)
            if abs(c) <= 10.0:
                advice = tr("ui tuning ok")
            elif abs(c) <= 25.0:
                advice = tr("ui tuning slightly off").format(
                    dir=tr("ui tuning sharp") if c > 0 else tr("ui tuning flat"))
            else:
                advice = tr("ui tuning off").format(
                    dir=tr("ui tuning sharp") if c > 0 else tr("ui tuning flat"))
            self._values["key tuning offset"].setText(
                f"{c:+.1f} c — {advice}"
            )
        figs = figures_summary(res.notes)
        self._values["key figures"].setText(
            ", ".join(f"{tr_figure(k)} ×{v}" for k, v in figs.items()) if figs else "—"
        )

        # The take's coherence verdict, where the singer looks: the same
        # `coherence_report` the JSON block carries (same evidence, same
        # thresholds), read here so the verdict needs no script and no JSON
        # diving. A file analysis without evidence shows why it is silent.
        observation = getattr(res, "observation", None)
        if observation is None:
            self._values["key coherence"].setText(
                tr("ui coherence manual")
                if getattr(res, "manual_pitch_corrections", False)
                else tr("ui coherence na")
            )
        else:
            block = coherence_report(observation, res.notes)
            checked = block["checked"]
            n_suspects = len(block["suspects"])
            if n_suspects:
                self._values["key coherence"].setText(
                    tr("ui coherence suspects").format(n=n_suspects, checked=checked)
                )
            else:
                self._values["key coherence"].setText(
                    tr("ui coherence clean").format(checked=checked)
                )

        # Per-note table (note names follow the UI language)
        positions = getattr(res, "note_positions", None) or []
        self._notes_table.setRowCount(len(res.notes))
        for i, n in enumerate(res.notes):
            pos = positions[i] if i < len(positions) else None
            vals = [
                str(i + 1),
                f"{pos[0]}:{pos[1]:g}" if pos else "—",
                note_display_pair(n.midi),
                tr_figure(n.figure) or "—",
            ]
            for c, v in enumerate(vals):
                self._notes_table.setItem(i, c, pg.QtWidgets.QTableWidgetItem(v))
        self._notes_table.resizeColumnsToContents()

        # The live/final explainer: WHY the final score differs from what
        # the singer watched, one honest line per cause (numbers included).
        # A file analysis has no live reading and stays silent.
        from ..core.explain import explain as explain_block

        if getattr(res, "manual_pitch_corrections", False):
            self._lbl_explain.setText(tr("ui coherence live pending"))
            self._lbl_explain.setVisible(True)
        else:
            lines = explain_block(getattr(res, "live_vs_final", None) or {})
            if lines:
                self._lbl_explain.setText("<br>".join(lines))
                self._lbl_explain.setVisible(True)
            else:
                self._lbl_explain.setVisible(False)

        # Per-measure chord progression (falls back to legacy 2 s segments)
        prog = getattr(res, "chord_progression", None)
        if prog:
            self._seg_table.setRowCount(len(prog))
            for r, (m0, m1, c) in enumerate(prog):
                span_txt = f"m. {m0}" if m1 == m0 + 1 else f"m. {m0}–{m1 - 1}"
                self._seg_table.setItem(r, 0, pg.QtWidgets.QTableWidgetItem(span_txt))
                self._seg_table.setItem(r, 1, pg.QtWidgets.QTableWidgetItem(tr_chord_label(c.label)))
        else:
            segs = getattr(res, "chord_segments", None) or []
            self._seg_table.setRowCount(len(segs))
            for r, (t0, t1, c) in enumerate(segs):
                self._seg_table.setItem(r, 0, pg.QtWidgets.QTableWidgetItem(f"{t0:.1f} – {t1:.1f} s"))
                self._seg_table.setItem(r, 1, pg.QtWidgets.QTableWidgetItem(tr_chord_label(c.label)))
        self._seg_table.resizeColumnsToContents()


class PianoView(pg.QtWidgets.QWidget):
    """Piano keyboard showing which notes are played — or being sung.

    Black keys are drawn over the whites; a key lights up in the accent
    color while its note sounds. `set_notes` indexes the detected notes by
    pitch and `set_playhead` lights the keys sounding at time `t`; the note
    finder calls `set_pressed` to light the keys it is hearing right now.

    The range is a parameter, so the finder can show the 88 keys of a piano
    instead of the three octaves the analysis view needs.
    """

    MIN_MIDI, MAX_MIDI = 48, 84  # 3 octaves around the middle register
    WHITE_PCS = (0, 2, 4, 5, 7, 9, 11)
    SHARP_AFTER = (0, 2, 5, 7, 9)  # C, D, F, G, A are followed by a black key

    key_clicked = pyqtSignal(int)

    def __init__(self, parent=None, min_midi: int | None = None, max_midi: int | None = None):
        super().__init__(parent)
        self.min_midi = self.MIN_MIDI if min_midi is None else min_midi
        self.max_midi = self.MAX_MIDI if max_midi is None else max_midi
        self._whites = [
            m for m in range(self.min_midi, self.max_midi + 1) if m % 12 in self.WHITE_PCS
        ]
        self._spans: dict[int, list[tuple[float, float]]] = {}
        self._active: set[int] = set()
        self._active_colors: dict[int, tuple[int, int, int]] = {}
        self.setMinimumHeight(78)
        self.setCursor(pg.QtCore.Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Publish the key under the pointer, checking black keys first."""
        if event.button() != pg.QtCore.Qt.MouseButton.LeftButton:
            return
        midi = self._midi_at(event.position().x(), event.position().y())
        if midi is not None:
            self.key_clicked.emit(midi)

    def _midi_at(self, x: float, y: float) -> int | None:
        """Map a keyboard position to its key; black keys overlay whites."""
        if not 0 <= x < self.width() or not 0 <= y < self.height():
            return None
        if not self._whites:
            return None
        white_width = self.width() / len(self._whites)
        black_width = white_width * 0.62
        if y < self.height() * 0.62:
            for index in range(len(self._whites) - 1):
                base = self._white_midi(index)
                if self._has_black_after(base):
                    left = (index + 1) * white_width - black_width / 2
                    if left <= x < left + black_width:
                        return base + 1
        index = min(len(self._whites) - 1, int(x / white_width))
        return self._white_midi(index)

    def set_notes(self, notes) -> None:
        """Index the detected notes by pitch: {midi: [(start, end), ...]}"""
        self._spans = {}
        for n in notes:
            self._spans.setdefault(n.midi, []).append((n.start, n.end))
        self.update()

    def set_playhead(self, t: float) -> None:
        """Light the keys sounding at time `t` (follows the player)."""
        self._active = {
            m for m, spans in self._spans.items() if any(s <= t < e for s, e in spans)
        }
        self._active_colors = {}
        self.update()

    def set_pressed(self, midis=None, colors=None) -> None:
        """Light exactly these keys, optionally with per-key RGB colors."""
        self._active = set(midis or ())
        self._active_colors = dict(colors or {})
        self.update()

    def paintEvent(self, ev) -> None:  # noqa: N802 (Qt naming)
        from PyQt6.QtGui import QColor, QPainter, QPen

        p = QPainter(self)
        w, h = self.width(), self.height()
        n_white = len(self._whites)
        ww = w / n_white
        # label a white key with its own name whenever it is wide enough
        show_names = ww >= 16.0
        # white keys
        for i in range(n_white):
            midi = self._white_midi(i)
            x = i * ww
            key_on = midi in self._active
            p.fillRect(
                int(x), 0, int(ww) - 1, h,
                QColor(*self._active_colors[midi]) if midi in self._active_colors
                else QColor("#4da3d8") if key_on else QColor("#e9edf2"),
            )
            p.setPen(QPen(QColor("#39424d"), 1))
            p.drawRect(int(x), 0, int(ww) - 1, h - 1)
        # black keys (drawn over)
        bw, bh = ww * 0.62, h * 0.62
        for i in range(n_white - 1):
            midi = self._white_midi(i)
            if self._has_black_after(midi):
                x = (i + 1) * ww - bw / 2
                key_on = midi + 1 in self._active
                p.fillRect(
                    int(x), 0, int(bw), int(bh),
                    QColor(*self._active_colors[midi + 1]) if midi + 1 in self._active_colors
                else QColor("#2f81a8") if key_on else QColor("#151a20"),
                )
        # note names on the white keys (all of them when they fit, else the Cs)
        p.setPen(QColor("#6a7480"))
        f = p.font()
        f.setPointSize(7)
        p.setFont(f)
        for i in range(n_white):
            midi = self._white_midi(i)
            if show_names or midi % 12 == 0:
                p.drawText(int(i * ww) + 2, h - 4, midi_to_name(midi))
        p.end()

    # keyboard mapping helpers -----------------------------------------
    def _white_midi(self, idx: int) -> int:
        """MIDI note of the idx-th white key of this keyboard."""
        return self._whites[idx]

    def _has_black_after(self, midi: int) -> bool:
        return midi % 12 in self.SHARP_AFTER and midi + 1 <= self.max_midi


class ErrorBanner(pg.QtWidgets.QLabel):
    """Inline red banner for user-facing errors (no modal popups)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setStyleSheet(
            "background: #5c1a1a; color: #ffd7d7; border: 1px solid #a33;"
            "border-radius: 6px; padding: 8px; font-weight: bold;"
        )
        self.hide()

    def show_error(self, message: str) -> None:
        self.setText(f"⚠  {message}")
        self.show()

    def clear(self) -> None:
        super().setText("")
        self.hide()


class AccuracyPlot(pg.QtWidgets.QWidget):
    """Chronological cents deviation of the take, one point per note.

    The history line above says WHICH note drifted and by how much; this
    plot says WHEN in the take it drifted. It reuses the meter's ±50-cent
    frame and the shared green/amber/red scale, so needle, key and plot read
    as one instrument. A painter widget: a few dozen points, a full repaint
    per refresh is cheap and stateless.

    A point carries its note's onset time, and clicking it seeks the piano
    roll there (`pointClicked`) — the plot is not just a summary, it is a
    way back into the take: drift at 0:42, click, hear what happened.
    """

    pointClicked = pyqtSignal(float)  # onset time (s) of the clicked note

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(96)
        self._points: list[tuple[int, float]] = []
        self._times: list[float] = []  # onset of each point (same order)

    def set_points(self, points, times=None) -> None:
        """(midi, mean |cents|) in singing order; [] clears to the empty state.

        `times` (onsets in seconds, same length) enables the click-to-seek;
        without it the plot stays a plain summary.
        """
        self._points = list(points)
        self._times = list(times) if times is not None else []
        self.update()

    def _index_at(self, x: float) -> int | None:
        """Index of the point under x (same geometry as the paint), else None."""
        n = len(self._points)
        if not n or len(self._times) != n:
            return None
        pad = 16.0
        ww = (self.width() - 2 * pad) / n
        if ww <= 0:
            return None
        i = int((x - pad) // ww)
        return i if 0 <= i < n else None

    def mousePressEvent(self, ev) -> None:  # noqa: N802 (Qt naming)
        pos = ev.position().x() if hasattr(ev, "position") else float(ev.x())
        i = self._index_at(pos)
        if i is not None:
            self.pointClicked.emit(self._times[i])

    def paintEvent(self, ev) -> None:  # noqa: N802 (Qt naming)
        from PyQt6.QtCore import Qt, QPointF, QRectF
        from PyQt6.QtGui import QColor, QPainter, QPen

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        # card background, consistent with the hint cards
        p.setPen(QPen(QColor("#2c3540"), 1))
        p.setBrush(QColor("#1e242c"))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 10, 10)
        pad, top, bottom = 16.0, 10.0, 22.0  # bottom leaves room for names
        mid = (top + h - bottom) / 2
        span = (h - bottom - top) / 2
        # the meter's landmarks: ±25 guides and the ±10 in-tune zone
        p.setPen(QPen(QColor(230, 180, 80, 70), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(pad, mid - span * 0.5), QPointF(w - pad, mid - span * 0.5))
        p.drawLine(QPointF(pad, mid + span * 0.5), QPointF(w - pad, mid + span * 0.5))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(80, 210, 120, 26))
        p.drawRect(QRectF(pad, mid - span * 0.2, w - 2 * pad, span * 0.4))
        p.setPen(QPen(QColor("#3a4655"), 1))
        p.drawLine(QPointF(pad, mid), QPointF(w - pad, mid))
        n = len(self._points)
        if n:
            ww = (w - 2 * pad) / n
            f = p.font()
            f.setPointSize(7)
            for i, (midi, cents) in enumerate(self._points):
                c = max(-50.0, min(50.0, cents))
                x = pad + (i + 0.5) * ww
                y = mid - (c / 50.0) * span
                color = tuning_color(cents)
                p.setPen(QPen(QColor(*color), 0))
                p.setBrush(QColor(*color))
                p.drawEllipse(QPointF(x, y), 4.0, 4.0)
                if ww >= 34.0:  # room for a compact name under each point
                    p.setPen(QColor("#6a7480"))
                    p.setFont(f)
                    p.drawText(QPointF(x - ww / 2, h - 8), midi_to_name(midi))
        p.end()


class TuningMeter(pg.QtWidgets.QWidget):
    """Horizontal ±50-cent meter: how far the sung pitch is from the note.

    A painter widget, not a gauge of bars: an intonation meter reads like the
    one on a tuner — a centered needle, green inside ±10 cents (in tune by
    every practical standard), fading to amber then red as the error grows.
    Height is fixed; the parent layout stretches it in width.
    """

    CENTER_ST = CENTER_CENTS / 100.0
    NEAR_ST = NEAR_CENTS / 100.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self._cents: float | None = None

    def set_cents(self, cents: float | None) -> None:
        self._cents = None if cents is None or cents != cents else max(-50.0, min(50.0, cents))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        from PyQt6.QtCore import QPointF, QRectF
        from PyQt6.QtGui import QColor, QPainter, QPen

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        track = QRectF(6.0, h / 2 - 3.0, w - 12.0, 6.0)
        p.setPen(QPen(QColor("#2e3742"), 1.0))
        p.setBrush(QColor("#20262e"))
        p.drawRoundedRect(track, 3.0, 3.0)
        # center tick: the target (the nearest piano note)
        p.setPen(QPen(QColor("#9fb3c8"), 1.6))
        p.drawLine(QPointF(w / 2, h / 2 - 8), QPointF(w / 2, h / 2 + 8))
        c = self._cents
        if c is not None:
            x = w / 2 + (c / 50.0) * (w / 2 - 10.0)
            color = tuning_color(c)
            p.setPen(QPen(QColor(*color), 0))
            p.setBrush(QColor(*color))
            p.drawEllipse(QPointF(x, h / 2), 5.0, 5.0)
            p.end()
            return
        p.end()


class LiveIndicator(pg.QtWidgets.QWidget):
    """Real-time display: current note, frequency, level and tuning meter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = pg.QtWidgets.QHBoxLayout(self)
        left = pg.QtWidgets.QVBoxLayout()
        self.label_note = pg.QtWidgets.QLabel("—")
        self.label_note.setStyleSheet(
            f"font-size: 21pt; font-weight: bold; color: {NOTE_TEXT_COLOR};"
        )
        self.label_hz = pg.QtWidgets.QLabel("")
        self.label_rms = pg.QtWidgets.QLabel("")
        self.tuning = TuningMeter()
        self._rms = 0.0
        left.addWidget(self.label_note)
        left.addWidget(self.label_hz)
        left.addWidget(self.tuning)
        left.addWidget(self.label_rms)
        lay.addLayout(left, stretch=1)
        # the CONFIRMED-note counter, BIG: "is the recognition following me?"
        # is the singer's first question, and the status line was too small to
        # answer it — a glance answers now. Updated by the same live tick that
        # renders the roll (one owner of the live rendering).
        right = pg.QtWidgets.QVBoxLayout()
        self.label_note_count = pg.QtWidgets.QLabel("0")
        self.label_note_count.setAlignment(pg.QtCore.Qt.AlignmentFlag.AlignRight)
        self.label_note_count.setStyleSheet(
            f"font-size: 30pt; font-weight: bold; color: {ACCENT};"
        )
        self.label_count_unit = pg.QtWidgets.QLabel("")
        self.label_count_unit.setAlignment(pg.QtCore.Qt.AlignmentFlag.AlignRight)
        self.label_count_unit.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        right.addWidget(self.label_note_count)
        right.addWidget(self.label_count_unit)
        right.addStretch(1)
        lay.addLayout(right)
        self.retranslate()

    def set_note_count(self, count: int) -> None:
        """Show how many notes the take has CONFIRMED so far."""
        self.label_note_count.setText(str(count))

    def retranslate(self) -> None:
        """Refresh the idle level label without resetting live measurements."""
        self.update_rms(self._rms)
        self.label_count_unit.setText(tr("ui live notes label"))

    def update_note(
        self, midi_float: float | None, hz: float, cents: float | None = None
    ) -> None:
        """Show the note; `cents` overrides the derived deviation.

        The name comes from the note's median (stable); the meter can take
        the instantaneous deviation instead — while practicing, correcting
        the pitch mid-note must move the needle NOW, not drown in the
        note-long average.
        """
        if midi_float is None or hz != hz:  # NaN check
            self.label_note.setText("—")
            self.label_hz.setText("")
            self.tuning.set_cents(None)
            return
        nearest = int(round(midi_float))
        self.label_note.setText(note_display_pair(nearest))
        self.label_hz.setText(f"{hz:.1f} Hz")
        # the sung pitch vs its NEAREST piano note, in cents — the number a
        # singer works with when practicing intonation
        self.tuning.set_cents(
            cents_deviation(midi_float) if cents is None else cents
        )

    def update_rms(self, rms: float) -> None:
        self._rms = rms
        if rms <= 0:
            self.label_rms.setText(tr("ui level idle"))
            return
        bars = int(min(1.0, rms * 20) * 20)
        self.label_rms.setText(tr("key level") + " : " + "█" * bars + "░" * (20 - bars))


# Application-wide QSS: coherent dark palette, visible focus, calm motion.
ACCENT = "#4da3d8"  # single accent color for primary actions & progress
# The palette, read once by every inline style (cards, meters, plots): the
# tones must stay COHERENT across the app — a color family, not a grab bag.
BG_APP = "#12161d"        # the window's depth (plots sit exactly on it)
BG_CARD = "#1a202a"       # raised surfaces: group boxes, hint cards, results
BG_SUNKEN = "#0e1218"     # inputs and plots: recessed, slightly darker
BORDER = "#2c3644"        # hairline borders everywhere
BORDER_SOFT = "#242d39"   # inner separations (alternate rows, handles)
TEXT = "#e8edf4"          # primary text
TEXT_DIM = "#8b98a8"      # secondary text, hints
TEXT_FAINT = "#6a7686"    # disabled, idle placeholders
ACCENT_SOFT = "#274059"   # quiet accent wash (selections, hovered rows)
NOTE_TEXT_COLOR = "#7cd0ff"  # the recognized note, wherever it is written large
# The finder page's hint card (and any other page that wants the same
# "explaining, not acting" frame): grey rounded panel.
HINT_CARD_STYLE = (
    f"color: {TEXT_DIM}; font-size: 9pt;"
    f"background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px;"
    "padding: 10px 18px;"
)
# Font sizes in pt, never px: a px-sized font reports pointSize() == -1,
# and rich-text labels then hit QFont::setPointSize(-1) (console warning).
# pt also scales with the user's Windows DPI setting, px does not.
# The comments live OUTSIDE the QSS on purpose: Qt's stylesheet parser has
# no '#' line comment ('#' opens an id selector there) and one unrecognized
# token silently discards the WHOLE sheet — the app then renders in the
# system default theme with no error anywhere.
STYLESHEET = f"""
QMainWindow, QWidget {{ background: {BG_APP}; color: {TEXT}; font-size: 10pt; }}
QLabel {{ color: #cfd6dd; background: transparent; }}
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #2e3948, stop:1 #28313e);
    color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 8px 16px; font-weight: 500;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #38445a, stop:1 #2e3948);
    border-color: {ACCENT};
}}
QPushButton:pressed {{ background: {BG_SUNKEN}; }}
QPushButton:focus {{ border: 1px solid {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: #1c2229; }}
QPushButton:flat {{ border: none; background: transparent; }}
QCheckBox, QSpinBox {{ color: #cfd6dd; background: transparent; }}
QCheckBox:hover {{ color: {TEXT}; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {BORDER}; background: #1c2229;
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSpinBox {{
    background: {BG_SUNKEN}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 2px 6px;
}}
QSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; background: transparent; border: none; }}
QGroupBox {{
    background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    margin-top: 12px; padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; color: #9fb3c8; }}
QListWidget {{
    background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 4px;
}}
QListWidget::item {{ border-radius: 4px; padding: 2px; }}
QListWidget::item:hover {{ background: {ACCENT_SOFT}; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QTableWidget {{
    background: {BG_CARD}; alternate-background-color: {BG_SUNKEN};
    gridline-color: {BORDER_SOFT}; border: 1px solid {BORDER}; border-radius: 8px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QHeaderView::section {{
    background: #232c38; color: #9fb3c8; border: none; padding: 4px;
}}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #34404f; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #4a5a6e; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #34404f; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QProgressBar {{
    background: #1c2229; border: 1px solid {BORDER}; border-radius: 8px;
    text-align: center; color: #cfd6dd; height: 16px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {ACCENT}, stop:1 #7ccbf5);
    border-radius: 7px;
}}
QStatusBar {{ color: #9fb3c8; background: {BG_APP}; border-top: 1px solid {BORDER_SOFT}; }}
QToolTip {{
    background: #232c38; color: {TEXT}; border: 1px solid {BORDER};
    padding: 5px; border-radius: 6px;
}}
"""


def action_button_style(primary: bool = False) -> str:
    """Shared style for the big home actions (primary = accent-filled)."""
    if primary:
        return (
            f"QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            f"  stop:0 #5db2e2, stop:1 {ACCENT}); color: #10222f; font-size: 12pt;"
            "  padding: 10px 22px; border: none; border-radius: 12px;"
            "  font-weight: bold; }"
            "QPushButton:hover { background: #5db2e2; }"
            "QPushButton:pressed { background: #3d7ea6; }"
        )
    return (
        "QPushButton { font-size: 12pt; padding: 10px 22px; border-radius: 12px; }"
        f"QPushButton:hover {{ border-color: {ACCENT}; }}"
    )
