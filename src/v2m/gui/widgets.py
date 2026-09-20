"""GUI widgets: piano roll, waveform, results panel, live indicator, error banner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg

from ..core.figures import figures_summary, short_label
from ..core.i18n import note_display_pair, tr, tr_figure
from ..core.music import midi_to_name
from ..core.playback import NotePlayer

# Confidence color ramp (red -> green)
LOW_COLOR = (220, 80, 80)
HIGH_COLOR = (80, 220, 120)
SILENCE_COLOR = (120, 120, 130)

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
    """

    seek_requested = pg.QtCore.pyqtSignal(float)

    MIDI_MIN, MIDI_MAX = 21, 108
    LIVE_SPAN = 10.0      # seconds of history kept visible during live capture
    LIVE_Y_MIN = 14.0     # minimum visible pitch band, in semitones

    def __init__(self, parent=None):
        super().__init__(parent, axisItems={"left": PitchAxis(orientation="left")})
        self.setBackground("#12161c")
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.set_axis_labels()   # titled by the window's retranslate()
        self._bar_items: list = []
        self._labels: list = []
        self._curve = None
        self._notation_mode = False
        self._last_notes: list = []
        self._rendered_figures: list = []
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

    def _add_note(self, n) -> None:
        color = (
            figure_color(n.figure)
            if self._notation_mode
            else self._color_for_conf(n.confidence)
        )
        item = pg.BarGraphItem(
            x0=[n.start],
            x1=[n.end],
            y0=[n.midi - 0.4],
            y1=[n.midi + 0.4],
            brush=color,
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

    def _clear_notes(self) -> None:
        for it in self._bar_items:
            self.removeItem(it)
        for it in self._labels:
            self.removeItem(it)
        self._bar_items = []
        self._labels = []
        self._rendered_figures = []


class WaveformView(pg.PlotWidget):
    """Waveform overview with a playback/analysis position marker."""

    MAX_POINTS = 50_000  # plenty at screen resolution

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#12161c")
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

    KEYS = [
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
    ]
    # map the fixed value-slot name -> i18n key
    SLOT_KEYS = {
        "source": "key source",
        "duree": "key duration",
        "tempo": "key tempo",
        "meter": "key meter",
        "key": "key key",
        "score": "key key score",
        "alts": "key alternatives",
        "chord": "key chord",
        "nn": "key notes count",
        "nmeas": "key measures count",
        "figs": "key figures",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._title = pg.QtWidgets.QLabel("")
        self.addWidget(self._title, row=0, col=0, colspan=2)
        # empty state: a centered invitation shown before any analysis
        self._waiting = pg.QtWidgets.QLabel("")
        self._waiting.setAlignment(pg.QtCore.Qt.AlignmentFlag.AlignCenter)
        self._waiting.setStyleSheet("color: #6a7480; font-size: 10pt; font-style: italic;")
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
        self._has_results: bool = False

        # Per-segment chord table
        r2 = r + 2
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
        """Refresh every label in the current language (labels only)."""
        self._title.setText(f"<b>{tr('key results title')}</b>")
        self._waiting.setText(tr("ui panel waiting"))
        self._waiting.setVisible(self._has_results is False)
        for slot, key in self.SLOT_KEYS.items():
            self._labels[slot].setText(f"<b>{tr(key)}</b>")
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
        self._values["source"].setText(Path(res.source).name)
        self._values["duree"].setText(f"{res.duration_s:.2f} s")
        self._values["tempo"].setText(f"{res.tempo.bpm:.1f} BPM" if res.tempo else "—")
        self._values["meter"].setText(
            res.meter.label if getattr(res, "meter", None) else tr("key default meter")
        )
        if getattr(res, "note_positions", None) and res.tempo:
            self._values["nmeas"].setText(
                str(getattr(res, "num_measures", 0) or len(res.note_positions))
            )
        else:
            self._values["nmeas"].setText("—")
        self._values["key"].setText(res.key.best.label if res.key else "—")
        self._values["score"].setText(f"{res.key.best.score:.2f}" if res.key else "—")
        alts = (
            ", ".join(a.label for a in res.key.alternatives)
            if res.key and res.key.alternatives
            else "—"
        )
        self._values["alts"].setText(alts)

        # Global chord + runner-ups
        chord = getattr(res, "chord", None)
        if chord is not None:
            txt = chord.best.label
            if chord.alternatives:
                txt += "  (" + ", ".join(c.label for c in chord.alternatives) + ")"
            self._values["chord"].setText(txt)
        else:
            self._values["chord"].setText("—")

        self._values["nn"].setText(str(len(res.notes)))
        figs = figures_summary(res.notes)
        self._values["figs"].setText(
            ", ".join(f"{k} ×{v}" for k, v in figs.items()) if figs else "—"
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

        # Per-measure chord progression (falls back to legacy 2 s segments)
        prog = getattr(res, "chord_progression", None)
        if prog:
            self._seg_table.setRowCount(len(prog))
            for r, (m0, m1, c) in enumerate(prog):
                span_txt = f"m. {m0}" if m1 == m0 + 1 else f"m. {m0}–{m1 - 1}"
                self._seg_table.setItem(r, 0, pg.QtWidgets.QTableWidgetItem(span_txt))
                self._seg_table.setItem(r, 1, pg.QtWidgets.QTableWidgetItem(c.label))
        else:
            segs = getattr(res, "chord_segments", None) or []
            self._seg_table.setRowCount(len(segs))
            for r, (t0, t1, c) in enumerate(segs):
                self._seg_table.setItem(r, 0, pg.QtWidgets.QTableWidgetItem(f"{t0:.1f} – {t1:.1f} s"))
                self._seg_table.setItem(r, 1, pg.QtWidgets.QTableWidgetItem(c.label))
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

    def __init__(self, parent=None, min_midi: int | None = None, max_midi: int | None = None):
        super().__init__(parent)
        self.min_midi = self.MIN_MIDI if min_midi is None else min_midi
        self.max_midi = self.MAX_MIDI if max_midi is None else max_midi
        self._whites = [
            m for m in range(self.min_midi, self.max_midi + 1) if m % 12 in self.WHITE_PCS
        ]
        self._spans: dict[int, list[tuple[float, float]]] = {}
        self._active: set[int] = set()
        self.setMinimumHeight(78)

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
        self.update()

    def set_pressed(self, midis=None) -> None:
        """Light exactly these keys (the note finder's live pitch)."""
        self._active = set(midis or ())
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
                QColor("#4da3d8") if key_on else QColor("#e9edf2"),
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
                    QColor("#2f81a8") if key_on else QColor("#151a20"),
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
            "border-radius: 4px; padding: 8px; font-weight: bold;"
        )
        self.hide()

    def show_error(self, message: str) -> None:
        self.setText(f"⚠  {message}")
        self.show()

    def clear(self) -> None:
        super().setText("")
        self.hide()


class LiveIndicator(pg.QtWidgets.QWidget):
    """Real-time display: current note (UI-language format), frequency, level."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = pg.QtWidgets.QVBoxLayout(self)
        self.label_note = pg.QtWidgets.QLabel("—")
        self.label_note.setStyleSheet(
            f"font-size: 21pt; font-weight: bold; color: {NOTE_TEXT_COLOR};"
        )
        self.label_hz = pg.QtWidgets.QLabel("")
        self.label_rms = pg.QtWidgets.QLabel("niveau : —")
        lay.addWidget(self.label_note)
        lay.addWidget(self.label_hz)
        lay.addWidget(self.label_rms)

    def update_note(self, midi_float: float | None, hz: float) -> None:
        if midi_float is None or hz != hz:  # NaN check
            self.label_note.setText("—")
            self.label_hz.setText("")
            return
        self.label_note.setText(note_display_pair(int(round(midi_float))))
        self.label_hz.setText(f"{hz:.1f} Hz")

    def update_rms(self, rms: float) -> None:
        bars = int(min(1.0, rms * 20) * 20)
        self.label_rms.setText(tr("key level") + " : " + "█" * bars + "░" * (20 - bars))


# Application-wide QSS: coherent dark palette, visible focus, calm motion.
ACCENT = "#4da3d8"  # single accent color for primary actions & progress
NOTE_TEXT_COLOR = "#6cf"  # the recognized note, wherever it is written large

# Calm card for "here is what to do": the home steps and the note finder share
# it, so guidance reads the same wherever it appears.
HINT_CARD_STYLE = (
    "color: #8a97a5; font-size: 9pt;"
    "background: #20262e; border: 1px solid #2e3742; border-radius: 8px;"
    "padding: 10px 18px;"
)
STYLESHEET = """
# Font sizes in pt, never px: a px-sized font reports pointSize() == -1,
# and rich-text labels then hit QFont::setPointSize(-1) (console warning).
# pt also scales with the user's Windows DPI setting, px does not.
QMainWindow, QWidget { background: #1b2027; color: #e6e6e6; font-size: 10pt; }
QLabel { color: #cfd6dd; }
QPushButton {
    background: #2a3340; color: #e6e6e6; border: 1px solid #3a4655;
    border-radius: 6px; padding: 8px 14px;
}
QPushButton:hover { background: #34404f; border-color: #4a5a6e; }
QPushButton:pressed { background: #232b35; }
QPushButton:focus { border: 1px solid @ACCENT@; }
QPushButton:disabled { color: #6a7480; background: #232a33; }
QPushButton:flat { border: none; background: transparent; }
QCheckBox, QSpinBox { color: #cfd6dd; }
QCheckBox:hover { color: #e6e6e6; }
QSpinBox::up-button, QSpinBox::down-button { width: 16px; }
QGroupBox {
    border: 1px solid #3a4655; border-radius: 6px; margin-top: 10px;
    padding: 8px 6px 6px 6px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #9fb3c8; }
QTableWidget {
    background: #20262e; alternate-background-color: #232a33;
    gridline-color: #2e3742; border: 1px solid #3a4655; border-radius: 4px;
}
QHeaderView::section {
    background: #27303b; color: #9fb3c8; border: none; padding: 3px;
}
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: #34404f; border-radius: 3px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #4a5a6e; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 2px; }
QScrollBar::handle:horizontal { background: #34404f; border-radius: 3px; min-width: 24px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QProgressBar {
    background: #20262e; border: 1px solid #3a4655; border-radius: 5px;
    text-align: center; color: #cfd6dd;
}
QProgressBar::chunk { background: @ACCENT@; border-radius: 4px; }
QStatusBar { color: #9fb3c8; }
QToolTip {
    background: #27303b; color: #e6e6e6; border: 1px solid #3a4655;
    padding: 4px; border-radius: 3px;
}
""".replace("@ACCENT@", ACCENT)  # single source for the accent color
# NB: the token must not appear in real QSS — "#ACCENT" + a naive
# .replace("ACCENT", ...) produced "##4da3d8" (invalid QSS: Qt dropped the
# whole application stylesheet and the UI silently fell back to defaults).


def action_button_style(primary: bool = False) -> str:
    """Shared style for the two big home actions (primary = accent-filled)."""
    if primary:
        return (
            f"QPushButton {{ background: {ACCENT}; color: #10222f; font-size: 12pt;"
            "  padding: 10px 22px; border: none; font-weight: bold; }"
            f"QPushButton:hover {{ background: {ACCENT}; }}"
            "QPushButton:pressed { background: #3d7ea6; }"
        )
    return (
        "QPushButton { font-size: 12pt; padding: 10px 22px; }"
        "QPushButton:hover { border-color: #4da3d8; }"
    )
