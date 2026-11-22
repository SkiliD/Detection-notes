"""GUI widgets: piano roll, waveform, results panel, live indicator, error banner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QItemSelection, QItemSelectionModel, Qt, pyqtSignal

from ..core.accuracy import AccuracyHistory  # noqa: F401 — compatibility import
from ..core.continuity import coherence_report
from ..core.figures import figures_summary, short_label
from ..core.i18n import (
    format_number,
    note_display,
    note_display_pair,
    tr,
    tr_chord_label,
    tr_figure,
    tr_key_label,
)
from ..core.intonation import IN_TUNE_CENTS
from ..core.intonation import summarize as summarize_intonation


def intonation_lines(summary: dict) -> list[str]:
    """The intonation report as display lines (the shared rendering).

    The results panel and the live feedback both call this with a
    `intonation.summarize` dict — the same share, the same tendency
    wording, the same register leans — so the singer's take-so-far line
    and the per-take report can never disagree about the words or the
    scale. The caller supplies the evidence; this only translates it.
    """
    if summary["in_tune_share"] is None:
        return []
    lines = [tr("ui intonation share").format(share=round(100.0 * summary["in_tune_share"]))]
    tendency = summary["tendency"]
    if tendency == "even":
        direction = tr("ui intonation even")
    elif tendency == "sharp":
        direction = tr("ui intonation sharp").format(cents=summary["mean_cents"])
    elif tendency == "flat":
        direction = tr("ui intonation flat").format(cents=summary["mean_cents"])
    else:
        direction = None
    if direction is not None:
        lines.append(tr("ui intonation tendency").format(dir=direction))
    band_names = {
        "low": tr("ui register low"),
        "mid": tr("ui register mid"),
        "high": tr("ui register high"),
    }
    register = summary.get("by_register") or {}
    rows = [
        tr("ui register lean").format(band=band_names[band], cents=register[band]["mean_cents"])
        for band in ("low", "mid", "high")
        if register.get(band) is not None
    ]
    if rows:
        lines.append(tr("ui intonation registers").format(rows=" · ".join(rows)))
    # The same question asked of the take's two halves: a singer who starts
    # flat and ends sharp has a take-wide mean of zero, so the line above
    # says "steady" and only this one tells her where the drift lives.
    position = summary.get("by_position") or {}
    half_names = {"start": tr("ui position start"), "end": tr("ui position end")}
    rows = [
        tr("ui position lean").format(half=half_names[half], cents=position[half]["mean_cents"])
        for half in ("start", "end")
        if position.get(half) is not None
    ]
    if rows:
        lines.append(tr("ui intonation position").format(rows=" · ".join(rows)))
    return lines


def page_scroll(content: "pg.QtWidgets.QWidget") -> "pg.QtWidgets.QScrollArea":
    """Wrap a page's whole content in the app's ONE vertical scroll surface.

    A page that stacks cards (home, live, finder) must stay usable in a
    WINDOWED window, and a page's own minimum must not decide the window's
    height (a long unwrapped hint line can pin the window taller than a
    scaled laptop screen offers).

    The content keeps its own layout and stretches to fill the viewport
    when it fits (the live roll and staff must keep growing with the
    window); when it does not, this surface scrolls instead of clipping.
    The horizontal bar stays as a LAST resort rather than hidden: no
    control may become unreachable. The results page keeps its own two
    surfaces — its controls column scrolls independently of the roll.
    """
    area = pg.QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(pg.QtWidgets.QScrollArea.Shape.NoFrame)
    area.setWidget(content)
    return area


# Confidence color ramp (terracotta -> warm sand), in the palette's own
# warm family: a confident note reads as parchment, a weak one as a warning,
# and the SELECTED note turns copper — never the other way round.
LOW_COLOR = (198, 118, 104)
HIGH_COLOR = (216, 199, 168)
SILENCE_COLOR = (138, 152, 172)
MATCH_COLOR = (169, 200, 179)
MISMATCH_COLOR = (216, 135, 124)
AMBER_COLOR = (217, 184, 120)
CENTER_CENTS = IN_TUNE_CENTS  # in tune (green) — the scale intonation.py owns
NEAR_CENTS = 25.0  # <= 25 cents: close (amber); beyond = red


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


def tuning_advice(cents: float) -> str:
    """One-line intonation verdict for a cents deviation.

    The SAME scale `tuning_color` paints: on the note, watch it, or sit
    noticeably off. Shared so the report's global offset and the note
    finder's live reading can never disagree about what "in tune" means.
    """
    if abs(cents) <= CENTER_CENTS:
        return tr("ui tuning ok")
    direction = tr("ui tuning sharp") if cents > 0 else tr("ui tuning flat")
    if abs(cents) <= NEAR_CENTS:
        return tr("ui tuning slightly off").format(dir=direction)
    return tr("ui tuning off").format(dir=direction, cents=cents)


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

# The live-vs-final comparison overlay: the FINAL reading drawn BESIDE the
# captured one (a translucent wash where the two readings agree, the solid
# accent where the offline pass moved the note, so the eye lands on the
# differences first). The copper accent follows the application palette;
# opacity separates the captured and finalized readings.
COMPARE_FILL = (217, 167, 108, 70)
COMPARE_MOVED_FILL = (217, 167, 108, 190)

# The re-analysis diff overlay: the previous reading stays blue, while the
# changed and near-rounding marks use high-contrast warm colors.
GHOST_FILL = (90, 130, 200, 60)
GHOST_CHANGED_FILL = (230, 160, 60, 200)
GHOST_NEAR_FILL = (230, 190, 60, 160)


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


class NotesTable(pg.QtWidgets.QTableWidget):
    """The results notes table: the ONE selection owner, keyboard-driven.

    The table already owned the selection (the roll renders it); the
    keyboard makes the whole take reachable without the mouse: Up/Down step
    one note, Home/End jump to the first/last, Shift extends the range, and
    Space asks for the cursor note's audition. These keys and the roll's
    arrow keys write the SAME selection — the table stays the single owner,
    the roll a renderer of it.
    """

    audition_requested = pg.QtCore.pyqtSignal(int)  # the cursor row to play alone

    def __init__(self, ncols: int, parent=None) -> None:
        super().__init__(0, ncols, parent)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(pg.QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setMinimumHeight(78)
        self.setMaximumHeight(280)
        # focusable so the arrow keys steer the selection as soon as a row is
        # clicked (the same contract the roll carries)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Up/Down/Home/End move the selection; Space auditions the cursor."""
        key = event.key()
        if key == Qt.Key.Key_Space:
            row = self.currentRow()
            if row >= 0:
                self.audition_requested.emit(row)
                event.accept()
                return
        if self.rowCount() and key in (
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
            Qt.Key.Key_Home,
            Qt.Key.Key_End,
        ):
            self._move_cursor(
                key, extend=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            )
            event.accept()
            return
        super().keyPressEvent(event)

    def _move_cursor(self, key, extend: bool) -> None:
        """Move the selection cursor one row (Home/End jump to the ends).

        `setCurrentCell` carries the row highlight (SelectionBehavior is
        SelectRows); Shift then re-selects the ANCHOR-to-target range so the
        extended selection is contiguous, exactly like a drag on the roll.
        """
        rows = self.rowCount()
        column = self.currentColumn()
        if column < 0:
            column = 0
        # the range's FAR edge is the fixed anchor while extending (moving
        # right grows it forward, moving left grows it backward) — read
        # BEFORE the cursor moves, exactly like the roll's own extend
        selected = sorted({i.row() for i in self.selectedIndexes()})
        current = self.currentRow()
        if current < 0:
            current = (
                selected[0]
                if selected
                else (0 if key in (Qt.Key.Key_Down, Qt.Key.Key_Home, Qt.Key.Key_End) else rows - 1)
            )
        if key == Qt.Key.Key_Home:
            target = 0
        elif key == Qt.Key.Key_End:
            target = rows - 1
        elif key == Qt.Key.Key_Up:
            target = max(0, current - 1)
        else:
            target = min(rows - 1, current + 1)
        self.setCurrentCell(target, column)
        if extend and selected:
            edge = selected[0] if key in (Qt.Key.Key_Down, Qt.Key.Key_End) else selected[-1]
            lo, hi = sorted((edge, target))
            model = self.model()
            self.selectionModel().select(
                QItemSelection(model.index(lo, 0), model.index(hi, 0)),
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )


class PitchAxis(pg.AxisItem):
    """Left axis of the roll: both names of every labelled note row.

    The pitch of a note is what a musician reads, and the keyboard below the
    roll says the same thing — raw numbers only force a conversion the ear
    has already made. Both spellings travel together (`note_display_pair`:
    'Do4 / C4', current language first), so readers can recognize either
    notation directly on the ordinate.

    The labelled rows are OCTAVE-ANCHORED. pyqtgraph's automatic spacing marks
    arbitrary semitone multiples — a take spanning 60..72 came out labelled
    D4 / E4 / F#4 / G#4 / C5, correct rows nobody counts from.
    `LABEL_STEPS` walks 1, 2, 3, 4, 6 semitones then whole octaves: every step
    divides or is divided by the octave, so the C rows are always in the
    labelled set, and the walk stops as soon as the rows sit `MIN_LABEL_PX`
    apart. The second level keeps a finer grid line (`MIN_TICK_PX` at least)
    and carries no text.
    """

    OCTAVE = 12.0
    MIN_LABEL_PX = 34.0  # one named row needs this much room to stay readable
    MIN_TICK_PX = 6.0  # grid lines closer than this read as a smear
    LABEL_STEPS = (1.0, 2.0, 3.0, 4.0, 6.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0)

    # The labelled level, set by tickSpacing right before pyqtgraph asks for
    # the strings (it calls tickStrings once per level with that level's own
    # spacing): naming the fine grid too would stack a label between two
    # labels. None (a direct call) names whatever it is given.
    _labelled: float | None = None

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        if self._labelled is not None and spacing != self._labelled:
            return [""] * len(values)
        return [note_display_pair(int(round(v))) for v in values]

    def tickSpacing(self, minVal, maxVal, size):  # noqa: N802 (pyqtgraph API)
        """Rows `MIN_LABEL_PX` apart, on a grid that includes every C row."""
        span = abs(maxVal - minVal)
        if span == 0 or size <= 0:
            return []
        per_st = size / span  # pixels of one semitone
        self._labelled = next(
            (step for step in self.LABEL_STEPS if per_st * step >= self.MIN_LABEL_PX),
            self.LABEL_STEPS[-1],
        )
        minor = 1.0
        while minor * 2 <= self._labelled and per_st * minor < self.MIN_TICK_PX:
            minor *= 2.0
        if minor >= self._labelled:
            return [(self._labelled, 0.0)]  # the labels are already the grid
        return [(self._labelled, 0.0), (minor, 0.0)]


def ink_plot(widget: pg.PlotWidget) -> None:
    """Give a plot's hand-painted words the ACTIVE theme's inks.

    pyqtgraph paints its tick strings and axis titles itself, from the
    `foreground` config option, whose default ('d') is a fixed grey (#969696)
    whatever the palette — measured 2.62:1 on the light page and 2.41:1 on its
    sunken surface, the same blind spot the web roll's canvas words had, because
    no stylesheet reaches them. Every plot that draws words runs through this
    one function, so a view cannot answer the question differently.
    """
    item = widget.getPlotItem()
    for name in ("left", "bottom"):
        axis = item.getAxis(name)
        axis.setPen(pg.mkPen(BORDER))
        axis.setTextPen(pg.mkPen(TEXT_DIM))


class SecondAxis(pg.AxisItem):
    """The timeline's seconds, written in the window's language.

    pyqtgraph formats its own tick strings with `%.*f`, which writes a dot in
    every language: a French roll read "0.5 / 1.0" beside a French table
    reading "0,5 s / 1,0 s". The page's floor has ONE number formatter; this
    is the tick that calls it, so the painted seconds follow a language
    switch with every label around them.
    """

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        # The precision follows pyqtgraph's own spacing rule: whole seconds on
        # a wide view, then tenths, then hundredths as the view zooms in.
        digits = 0 if spacing >= 1.0 else (1 if spacing >= 0.1 else 2)
        return [format_number(v, digits) for v in values]


class PianoRoll(pg.PlotWidget):
    """Piano roll: rectangles = notes, curve = fine pitch.

    In "notation" mode each note is colored by its figure and carries a
    short label (R, B, N, C, DC, 3N, trailing dot = dotted).
    A playhead line tracks playback; double-click seeks (seek_requested).

    A SELECTABLE roll (the results page) also selects on the graph:
    a click picks one bar, Shift-click extends a contiguous range from
    the anchor, Ctrl/Cmd-click toggles one bar, and a click on empty
    space clears. The gesture EMITS `note_selection` (indices, [] =
    cleared) — the notes table remains the one selection owner, this
    widget only renders what it says.
    """

    seek_requested = pg.QtCore.pyqtSignal(float)
    note_selection = pg.QtCore.pyqtSignal(list)  # selected bar indices; [] = cleared
    audition_requested = pg.QtCore.pyqtSignal(int)  # the cursor note to play alone
    # one dragged note (index, start_s, end_s, midi) — emitted ONCE at
    # release, only when something moved; the window owns the edit
    note_dragged = pg.QtCore.pyqtSignal(int, float, float, int)

    MIDI_MIN, MIDI_MAX = 21, 108
    LIVE_SPAN = 10.0  # seconds of history kept visible during live capture
    LIVE_Y_MIN = 14.0  # minimum visible pitch band, in semitones

    def __init__(self, parent=None, *, selectable: bool = False):
        super().__init__(
            parent,
            axisItems={
                "left": PitchAxis(orientation="left"),
                "bottom": SecondAxis(orientation="bottom"),
            },
        )
        self.setBackground(BG_APP)
        ink_plot(self)
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.12)
        self.set_axis_labels()  # titled by the window's retranslate()
        self._bar_items: list = []
        self._labels: list = []
        self._compare_items: list = []  # the final reading beside the captured one
        self._ghost_items: list = []  # the previous reading (a re-analysis diff)
        self._curve = None
        self._notation_mode = False
        self._last_notes: list = []
        self._rendered_figures: list = []
        self._selectable = selectable
        self._editable = False
        self._selection: list[int] = []
        # an EDITABLE roll drags the pressed bar instead of extending the
        # selection: (index, mode, grab_t, grab_m, orig_start, orig_end,
        # orig_midi) — mode 'move' slides the whole bar, 'resize' drags
        # its right edge; pitch rides along in whole semitones
        self._note_drag: tuple | None = None
        # the note a Shift-click extends FROM (the last single-note choice,
        # made here or in the notes table)
        self._selection_anchor: int | None = None
        self._press_pos: tuple[float, float] | None = None  # empty press: click vs pan
        self._pan_moved = False
        self._live_span = self.LIVE_SPAN
        self._live_preview = None
        self._live_preview_midi: int | None = None
        self._live_band: tuple[float, float] | None = None
        self.setLimits(xMin=0)
        # arrow keys navigate the selection on a selectable roll, so the
        # roll must be able to take focus when a bar is pressed
        self.setFocusPolicy(pg.QtCore.Qt.FocusPolicy.StrongFocus)
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
        self.getPlotItem().setLabel("left", note, color=TEXT_DIM)
        self.getPlotItem().setLabel("bottom", time, units="s", color=TEXT_DIM)

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

        A SINGLE row chosen in the table becomes the anchor a later
        Shift-click on the graph extends from; a multi-note selection says
        nothing about which note the singer last picked, so it leaves the
        anchor alone.

        On a non-selectable roll (the live page has no table) nothing is
        ever highlighted: `selectable` gates BOTH directions.
        """
        if not self._selectable:
            return
        self._selection = sorted({int(i) for i in indices})
        if len(self._selection) == 1:
            self._selection_anchor = self._selection[0]
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
        self._selection_anchor = None
        self._press_pos = None
        self._note_drag = None
        self._render_selection()  # any highlight goes back to the own colors

    def set_editable(self, enabled: bool) -> None:
        """Turn bar-dragging on or off (an EDITABLE roll moves its notes).

        Only the results page's roll is editable: a drag there becomes ONE
        recorded edit through the window. The live review roll stays
        pick-only — its notes describe the CAPTURED take, the edit history
        belongs to the final reading.
        """
        self._editable = bool(enabled)
        if not self._editable:
            self._note_drag = None

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
            item.setOpts(brush=want, pen=self._pen_for(self._last_notes[i], i in chosen))

    def _emit_selection(self, indices: list[int]) -> None:
        """Publish a gesture's selection (silent when nothing changed)."""
        if indices == self._selection:
            return
        self._selection = indices
        self._render_selection()
        self.note_selection.emit(list(indices))

    def _move_selection(self, step: int, extend: bool = False) -> None:
        """Walk the selection note by note with the arrow keys.

        A plain arrow moves a SINGLE-note cursor to the next/previous note
        in time order (the ends hold, no wrap); with nothing selected the
        first arrow enters at the near end. Shift+arrow EXTENDS the current
        range by one note at the edge the arrow moves toward — the same
        contiguous range a drag builds. The roll stays a renderer: it EMITS
        `note_selection`, exactly like a gesture, so the table keeps owning
        the selection.
        """
        count = len(self._last_notes)
        if count == 0:
            return
        if not self._selection:
            entry = 0 if step > 0 else count - 1
            self._selection_anchor = entry
            self._emit_selection([entry])
            return
        edge = self._selection[-1] if step > 0 else self._selection[0]
        target = edge + step
        if target < 0 or target >= count:
            return  # the ends hold: navigating past a note never wraps
        if extend:
            lo, hi = sorted((self._selection[0], target))
            self._emit_selection(list(range(lo, hi + 1)))
            return
        self._selection_anchor = target
        self._emit_selection([target])

    def _selection_for_press(self, idx: int, modifiers) -> tuple[list[int], bool]:
        """The selection a press on `idx` means, and whether it was PLAIN.

        A plain click picks that note alone; Shift-click extends a
        CONTIGUOUS range from the anchor (the last single-note choice);
        Ctrl-click (Cmd on macOS) adds or removes that one note and leaves
        the rest alone. This is the selection convention every editor and
        list shares — dragging across bars no longer selects, because on
        this roll a drag means something else (moving a bar).

        `plain` is returned because ONLY a plain press may start a bar
        move: a modifier is a selection gesture, never an edit.
        """
        ctrl_mod = (
            pg.QtCore.Qt.KeyboardModifier.ControlModifier
            | pg.QtCore.Qt.KeyboardModifier.MetaModifier
        )
        ctrl = bool(modifiers & ctrl_mod)
        shift = bool(modifiers & pg.QtCore.Qt.KeyboardModifier.ShiftModifier)
        if ctrl:
            chosen = set(self._selection)
            chosen.symmetric_difference_update({idx})
            self._selection_anchor = idx
            return sorted(chosen), False
        anchor = self._selection_anchor if self._selection_anchor is not None else idx
        if shift:
            lo, hi = sorted((anchor, idx))
            return list(range(lo, hi + 1)), False
        self._selection_anchor = idx
        return [idx], True

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Press on a bar selects it; a plain drag on an editable roll moves it.

        A press on EMPTY space stays pyqtgraph's own gesture — the view
        keeps panning the timeline — and its click (no movement) clears
        the selection. On a non-selectable roll (the live page) every
        event passes through untouched. On an editable roll a PLAIN drag
        moves the bar; a press with Shift or Ctrl/Cmd selects instead and
        never starts a move.
        """
        if self._selectable and event.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            idx = self._hit(event)
            if idx is not None:
                self._press_pos = None
                self.setFocus()  # so the arrow keys steer the selection next
                selection, plain = self._selection_for_press(idx, event.modifiers())
                self._emit_selection(selection)
                if self._editable and plain:
                    t, m = self._view_at(event)
                    self._begin_note_drag(idx, t, m)
                event.accept()
                return
            self._press_pos = (event.position().x(), event.position().y())
            self._pan_moved = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._note_drag is not None:
            t, m = self._view_at(event)
            self._preview_note_drag(t, m)
            event.accept()
            return
        if self._press_pos is not None:
            x0, y0 = self._press_pos
            if abs(event.position().x() - x0) + abs(event.position().y() - y0) > 4:
                self._pan_moved = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            if self._note_drag is not None:
                t, m = self._view_at(event)
                self._end_note_drag(t, m)
                event.accept()
                return
            clicked_empty = self._press_pos is not None and not self._pan_moved
            self._press_pos = None
            if clicked_empty:
                self._emit_selection([])  # a click on empty space clears
        super().mouseReleaseEvent(event)

    # --------------------------------------------------------- note dragging
    def _view_at(self, event) -> tuple[float, float]:
        """The event's position in DATA coordinates (seconds, midi)."""
        vb = self.getPlotItem().vb
        data = vb.mapSceneToView(self.mapToScene(event.position().toPoint()))
        return float(data.x()), float(data.y())

    def _begin_note_drag(self, idx: int, t: float, m: float) -> None:
        """Arm the bar drag: MOVE anywhere on the bar, RESIZE its last quarter.

        The quarter is of the BAR's own duration (view-independent), so the
        grab zone stays honest at every zoom level.
        """
        n = self._last_notes[idx]
        mode = "resize" if t >= n.start + 0.75 * (n.end - n.start) else "move"
        self._note_drag = (
            idx,
            mode,
            float(t),
            float(m),
            float(n.start),
            float(n.end),
            int(n.midi),
            self._label_for_bar(idx),
        )

    def _label_for_bar(self, idx: int):
        """The figure label riding this bar, found by its stored position."""
        n = self._last_notes[idx]
        x, y = (n.start + n.end) / 2 - 0.03, n.midi + 0.45
        for lbl in self._labels:
            pos = lbl.pos()
            if abs(pos.x() - x) < 0.05 and abs(pos.y() - y) < 0.6:
                return lbl
        return None

    def _dragged_geometry(self, t: float, m: float) -> tuple[int, float, float, int]:
        """The clamped (index, start, end, midi) for the drag at (t, m).

        The clamp mirrors the core's `move_note` bounds exactly — inside
        the take, clear of the neighbours, positive duration, piano pitch —
        so a preview the edit would refuse never reaches the singer's eye.
        """
        # the drag state machine is bounded (`_note_drag` is armed on press
        # and cleared on release) — the unpacking asserts it for the type
        # checker without re-deriving the bounds
        assert self._note_drag is not None
        idx, mode, grab_t, grab_m, o_start, o_end, o_midi, _lbl = self._note_drag
        notes = self._last_notes
        dt = t - grab_t
        prev_end = notes[idx - 1].end if idx > 0 else 0.0
        next_start = notes[idx + 1].start if idx + 1 < len(notes) else float("inf")
        if mode == "resize":
            start = o_start
            end = min(max(o_end + dt, start + 0.02), max(next_start, start + 0.02))
        else:
            # a MOVE keeps the duration: the onset slides inside the gap
            # the neighbours leave, and a too-narrow gap means no move
            dur = o_end - o_start
            lo, hi = prev_end, next_start - dur
            if lo > hi:
                start, end = o_start, o_end
            else:
                start = min(max(o_start + dt, lo), hi)
                end = start + dur
        midi = int(min(108, max(21, o_midi + round(m - grab_m))))
        return idx, start, end, midi

    def _preview_note_drag(self, t: float, m: float) -> None:
        """Move the bar (and its figure label) with the pointer — a preview:
        the roll never touches the real notes, the window owns the edit."""
        assert self._note_drag is not None
        idx, _mode, _gt, _gm, o_start, o_end, o_midi, lbl = self._note_drag
        _idx, start, end, midi = self._dragged_geometry(t, m)
        item = self._bar_items[idx]
        item.setOpts(x0=[start], x1=[end], y0=[midi - 0.4], y1=[midi + 0.4])
        if lbl is not None:
            lbl.setPos((start + end) / 2 - 0.03, midi + 0.45)

    def _end_note_drag(self, t: float, m: float) -> None:
        """Release: put the bar back on the real note, publish ONE move.

        The window's edit re-renders the roll from the changed notes; if
        the release is a no-op (nothing moved) the restore alone undoes
        the preview, so the roll never disagrees with `_last_notes`.
        """
        assert self._note_drag is not None
        idx, _mode, _gt, _gm, o_start, o_end, o_midi, lbl = self._note_drag
        _idx, start, end, midi = self._dragged_geometry(t, m)
        self._bar_items[idx].setOpts(x0=[o_start], x1=[o_end], y0=[o_midi - 0.4], y1=[o_midi + 0.4])
        if lbl is not None:
            lbl.setPos((o_start + o_end) / 2 - 0.03, o_midi + 0.45)
        self._note_drag = None
        if (start, end, midi) != (o_start, o_end, o_midi):
            self.note_dragged.emit(idx, float(start), float(end), int(midi))

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Left/Right step the selection note by note (Shift extends).

        Only a selectable roll with notes reacts; every other key falls
        through to pyqtgraph's own handling.
        """
        if (
            self._selectable
            and self._last_notes
            and event.key()
            in (
                pg.QtCore.Qt.Key.Key_Left,
                pg.QtCore.Qt.Key.Key_Right,
            )
        ):
            self._move_selection(
                -1 if event.key() == pg.QtCore.Qt.Key.Key_Left else 1,
                extend=bool(event.modifiers() & pg.QtCore.Qt.KeyboardModifier.ShiftModifier),
            )
            event.accept()
            return
        if self._selectable and event.key() == pg.QtCore.Qt.Key.Key_Space and self._selection:
            # Space auditions the cursor note (the moving edge of the range)
            self.audition_requested.emit(self._selection[-1])
            event.accept()
            return
        super().keyPressEvent(event)

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
            and all(a is b for a, b in zip(self._last_notes, notes[:n_old], strict=False))
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

    def _pen_for(self, n, selected: bool):
        """The bar's outline: selection first, then the published doubt.

        ONE rule for the two writers (`_add_note` at build time and
        `_render_selection` on every selection change): a doubt cue set only
        at build time was erased by the first repaint. A real QPen, not
        `mkPen` — BarGraphItem normalises the latter back to NoPen (measured).
        """
        if selected:
            return pg.mkPen(TEXT, width=3)
        if getattr(n, "voice_doubt", False):
            return pg.QtGui.QPen(
                pg.QtGui.QColor(217, 167, 108, 220), 1.2, pg.QtCore.Qt.PenStyle.DashLine
            )
        return pg.mkPen(None)

    def _add_note(self, n) -> None:
        item = pg.BarGraphItem(
            x0=[n.start],
            x1=[n.end],
            y0=[n.midi - 0.4],
            y1=[n.midi + 0.4],
            brush=self._color_for(n),
            pen=self._pen_for(n, False),
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

    def set_comparison_notes(self, notes, moved=None) -> None:
        """Draw a SECOND reading BESIDE the captured one (live vs final).

        The live/final gap is shown side by side, not as an outline: every
        final note is drawn as a translucent bar at ITS OWN pitch, on the
        same timeline as the captured bars — a note the offline pass moved
        reads as two bars at two heights, in one glance. The notes the single
        matching policy (`core.compare.match`) flagged carry the solid ACCENT
        so the eye lands on them first; `moved` is the set of indices into
        `notes` (the final reading).

        The overlay belongs to the CURRENT roll: `_clear_notes` (a rebuild, a
        new take) drops it with the bars it annotates.
        """
        self.clear_comparison()  # idempotent: ONE overlay per reading
        chosen = set(moved or ())
        for i, n in enumerate(notes):
            is_moved = i in chosen
            item = pg.BarGraphItem(
                x0=[n.start],
                x1=[n.end],
                y0=[n.midi - 0.4],
                y1=[n.midi + 0.4],
                brush=pg.mkBrush(COMPARE_MOVED_FILL if is_moved else COMPARE_FILL),
                pen=pg.mkPen(ACCENT if is_moved else None),
            )
            item.setZValue(1)  # over the captured bars: a translucent tint
            self.addItem(item)
            self._compare_items.append(item)

    def clear_comparison(self) -> None:
        """Drop the other reading's overlay (a new reading replaces it)."""
        for it in self._compare_items:
            self.removeItem(it)
        self._compare_items = []

    def set_ghost_notes(self, ghost_notes, changed_afters=tuple(), near_afters=tuple()) -> None:
        """Draw the PREVIOUS reading as translucent ghost bars (a re-diff).

        `ghost_notes` carry (start, end, midi) snapshots of the OLD reading;
        `changed_afters`/`near_afters` are the AFTER-note indices whose
        pair is 'changed'/'near', accented so the eye lands on the
        differences first. The overlay belongs to the CURRENT roll:
        `_clear_notes` (a rebuild, a new take) drops it with the bars it
        annotates, exactly like the live/final comparison.
        """
        self.clear_ghost_notes()  # idempotent: ONE diff overlay per reading
        changed = set(changed_afters)
        near = set(near_afters)
        for g in ghost_notes:
            item = pg.BarGraphItem(
                x0=[g[0]],
                x1=[g[1]],
                y0=[g[2] - 0.4],
                y1=[g[2] + 0.4],
                brush=pg.mkBrush(GHOST_FILL),
                pen=pg.mkPen(None),
            )
            item.setZValue(1)
            self.addItem(item)
            self._ghost_items.append(item)
        for idx in changed | near:
            item = pg.BarGraphItem(
                x0=[self._last_notes[idx].start],
                x1=[self._last_notes[idx].end],
                y0=[self._last_notes[idx].midi - 0.4],
                y1=[self._last_notes[idx].midi + 0.4],
                brush=pg.mkBrush(GHOST_CHANGED_FILL if idx in changed else GHOST_NEAR_FILL),
                pen=pg.mkPen(None),
            )
            item.setZValue(2)
            self.addItem(item)
            self._ghost_items.append(item)

    def clear_ghost_notes(self) -> None:
        """Drop the previous-reading overlay (a new reading replaces it)."""
        for it in self._ghost_items:
            self.removeItem(it)
        self._ghost_items = []

    def _clear_notes(self) -> None:
        for it in self._bar_items:
            self.removeItem(it)
        for it in self._labels:
            self.removeItem(it)
        for it in self._compare_items:
            self.removeItem(it)
        for it in self._ghost_items:
            self.removeItem(it)
        self._bar_items = []
        self._labels = []
        self._note_drag = None  # the dragged bar no longer exists
        self._compare_items = []
        self._ghost_items = []
        self._rendered_figures = []


class WaveformView(pg.PlotWidget):
    """Waveform overview with a playback/analysis position marker."""

    MAX_POINTS = 50_000  # plenty at screen resolution

    def __init__(self, parent=None):
        super().__init__(parent, axisItems={"bottom": SecondAxis(orientation="bottom")})
        self.setBackground(BG_APP)
        ink_plot(self)
        self.getPlotItem().hideAxis("left")
        self.getPlotItem().setLabel("bottom", "Temps", units="s", color=TEXT_DIM)
        self._curves: list = []  # min and max envelope curves
        self._line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen((255, 170, 80, 220), width=1.2)
        )
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
        "key intonation",
        "key coherence",
        "key reading diff",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._title = pg.QtWidgets.QLabel("")
        self._title.setStyleSheet(f"font-size: 14pt; color: {TEXT}; padding-bottom: 6px;")
        self.addWidget(self._title, row=0, col=0, colspan=2)
        # empty state: a centered invitation shown before any analysis
        self._waiting = pg.QtWidgets.QLabel("")
        self._waiting.setAlignment(pg.QtCore.Qt.AlignmentFlag.AlignCenter)
        self._waiting.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10pt; font-style: italic;"
            f"background: {BG_SUNKEN}; border: none; border-radius: 4px; padding: 12px;"
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
        self._notes_table = NotesTable(5)
        self._lbl_notes = pg.QtWidgets.QLabel("")
        self.addWidget(self._lbl_notes, row=r, col=0)
        self.addWidget(self._notes_table, row=r + 1, col=0, colspan=2)
        # the live-vs-final explainer, one line per cause (a live take
        # only): hidden for a file analysis, where nothing moved
        self._lbl_explain = pg.QtWidgets.QLabel("")
        self._lbl_explain.setWordWrap(True)
        self._lbl_explain.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 8pt; font-style: italic;"
            f"background: {BG_SUNKEN}; border: none;"
            "border-radius: 4px; padding: 8px;"
        )
        self._lbl_explain.setVisible(False)
        self.addWidget(self._lbl_explain, row=r + 2, col=0, colspan=2)
        # The re-analysis DIFf detail (suspects shown with their diff label,
        # one line each): filled by `set_reading_diff`, hidden when the page
        # clears it (a new take, a manual edit, a failed re-analysis). The
        # plaintext rows keep the whole detail one selectable label — no
        # second table to keep in sync with the per-note one.
        self._lbl_diff = pg.QtWidgets.QLabel("")
        self._lbl_diff.setWordWrap(True)
        self._lbl_diff.setTextInteractionFlags(
            pg.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._lbl_diff.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 8pt;"
            f"background: {BG_SUNKEN}; border: none;"
            "border-radius: 4px; padding: 8px;"
        )
        self._lbl_diff.setVisible(False)
        self.addWidget(self._lbl_diff, row=r + 3, col=0, colspan=2)
        # the diff facts kept across a retranslate (the counts, the detail
        # rows, the suspects) — None = no diff shown
        self._diff_facts: dict | None = None
        self._diff_lines: tuple[str, ...] = ()
        self._diff_suspect_lines: tuple[str, ...] = ()
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
        table.setMinimumHeight(78)
        table.setMaximumHeight(280)
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
        self._labels["key intonation"].setText(f"<b>{tr('key intonation')}</b>")
        self._labels["key coherence"].setText(f"<b>{tr('key coherence')}</b>")
        self._labels["key reading diff"].setText(f"<b>{tr('key reading diff')}</b>")
        self._lbl_notes.setText(f"<b>{tr('key notes table')}</b>")
        self._lbl_segs.setText(f"<b>{tr('key chord segments')}</b>")
        self._notes_table.setHorizontalHeaderLabels(
            [
                "#",
                f"{tr('key meter')}:{tr('key time col')}",
                tr("key note col"),
                tr("key figure col"),
                tr("key voice col"),
            ]
        )
        self._seg_table.setHorizontalHeaderLabels([tr("key span col"), tr("key chord")])
        self._notes_table.resizeColumnsToContents()
        self._seg_table.resizeColumnsToContents()
        # the kept diff facts re-render in the active language (no recompute)
        self._render_diff_compact()

    def set_results(self, res) -> None:
        self._has_results = res is not None
        self._waiting.setVisible(not self._has_results)
        if res is None:
            for vl in self._values.values():
                vl.setText("—")
            return
        self._values["key source"].setText(Path(res.source).name)
        self._values["key duration"].setText(f"{format_number(res.duration_s, 2)} s")
        self._values["key tempo"].setText(
            f"{format_number(res.tempo.bpm, 1)} BPM" if res.tempo else "—"
        )
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
        self._values["key key"].setText(tr_key_label(res.key.best.label) if res.key else "—")
        self._values["key key score"].setText(
            format_number(res.key.best.score, 2) if res.key else "—"
        )
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
            self._values["key tuning offset"].setText(
                f"{format_number(c, 1, sign=True)} c — {tuning_advice(c)}"
            )
        # The take's per-note intonation report: the SAME `intonation.
        # summarize` the JSON block carries, through the SAME rendering the
        # live feedback shows while singing — one writer of the words.
        self._values["key intonation"].setText(
            "\n".join(intonation_lines(summarize_intonation(res.notes))) or "—"
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
                # the reading's own doubt, named in words: a note on thin
                # voicing is still a note (ADR 0005), and the table says so
                tr("ui doubt voice") if getattr(n, "voice_doubt", False) else "—",
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

    # ------------------------------------------------------------------
    # the before/after reading diff (a re-analysis's presentation; the
    # comparison object comes from core.reading_diff — page renders only)
    # ------------------------------------------------------------------
    def set_reading_diff(self, comparison) -> None:
        """Show one before/after comparison (a successful re-analysis).

        `comparison` is the core's `ReadingComparison`: the panel renders
        its counts and detail; it never re-derives. Rows are built once and
        retranslated from the same facts.
        """
        self._diff_facts = {
            "stable": comparison.count("stable"),
            "changed": comparison.count("changed"),
            "near": comparison.count("near"),
            "missing": len(comparison.missing),
            "added": len(comparison.added),
            "suspects": len(comparison.after_suspects),
        }
        rows: list[str] = []
        for pair in comparison.pairs:
            if pair.status == "stable":
                continue
            rows.append(
                tr("ui diff pair row").format(
                    i=pair.before.index + 1,
                    before=note_display_pair(pair.before.midi),
                    after=note_display_pair(pair.after.midi),
                    status=tr(f"diff status {pair.status}"),
                    reasons=", ".join(tr(f"diff reason {r}") for r in pair.reasons) or "—",
                )
            )
        for snap in comparison.missing:
            rows.append(
                tr("ui diff missing row").format(
                    i=snap.index + 1,
                    before=note_display_pair(snap.midi),
                    status=tr("diff status missing"),
                )
            )
        for snap in comparison.added:
            rows.append(
                tr("ui diff added row").format(
                    i=snap.index + 1,
                    after=note_display_pair(snap.midi),
                    status=tr("diff status added"),
                )
            )
        self._diff_lines = tuple(rows)
        suspect_rows: list[str] = []
        for row in comparison.after_suspects:
            tag = tr(f"diff suspect {row.status}") if row.status else tr("diff suspect unknown")
            suspects_line = tr("ui diff suspect row").format(
                midi=note_display_pair(row.midi),
                plateau=note_display_pair(row.plateau_midi),
                dur=format_number(row.plateau_s, 2),
                offset=f"{format_number(row.plateau_offset_st, 2, sign=True)} st",
                tag=tag,
                conf=(
                    tr("ui diff confidence").format(c=format_number(row.confidence, 2))
                    if row.confidence is not None
                    else ""
                ),
            )
            suspect_rows.append(suspects_line)
        for row in comparison.resolved_suspects:
            suspect_rows.append(
                tr("ui diff suspect resolved").format(
                    midi=note_display_pair(row.midi),
                    dur=format_number(row.duration_s, 2),
                )
            )
        self._diff_suspect_lines = tuple(suspect_rows)
        self._render_diff_compact()

    def clear_reading_diff(self) -> None:
        """Drop the before/after presentation (a new take, an edit, a failure)."""
        self._diff_facts = None
        self._diff_lines = ()
        self._diff_suspect_lines = ()
        self._render_diff_compact()

    def _render_diff_compact(self) -> None:
        """Render the count line and the detail from the kept facts.

        One literal tr() line ON PURPOSE (the tr() key guard only sees
        literal arguments); the kept rows are retranslated, never recomputed.
        """
        if self._diff_facts is None:
            self._values["key reading diff"].setText(tr("ui diff none"))
            self._lbl_diff.setVisible(False)
            return
        self._values["key reading diff"].setText(tr("ui diff compact").format(**self._diff_facts))
        detail = "<br>".join(self._diff_lines)
        suspects = "<br>".join(self._diff_suspect_lines)
        if suspects:
            header = tr("ui diff suspects header").format(n=len(self._diff_suspect_lines))
            suspects = f"<b>{header}</b><br>{suspects}"
        text = "<br>".join(part for part in (detail, suspects) if part)
        self._lbl_diff.setText(text)
        self._lbl_diff.setVisible(bool(text))

    def set_chord_progression(self, res) -> None:
        """Per-measure chord progression (falls back to legacy 2 s segments).

        Split from `set_results` so the diff renderer stays off the result's
        facts — the progression rows belong to the result, re-filled by it.
        """
        prog = getattr(res, "chord_progression", None)
        if prog:
            self._seg_table.setRowCount(len(prog))
            for r, (m0, m1, c) in enumerate(prog):
                span_txt = f"m. {m0}" if m1 == m0 + 1 else f"m. {m0}–{m1 - 1}"
                self._seg_table.setItem(r, 0, pg.QtWidgets.QTableWidgetItem(span_txt))
                self._seg_table.setItem(
                    r, 1, pg.QtWidgets.QTableWidgetItem(tr_chord_label(c.label))
                )
        else:
            segs = getattr(res, "chord_segments", None) or []
            self._seg_table.setRowCount(len(segs))
            for r, (t0, t1, c) in enumerate(segs):
                self._seg_table.setItem(
                    r,
                    0,
                    pg.QtWidgets.QTableWidgetItem(
                        f"{format_number(t0, 1)} – {format_number(t1, 1)} s"
                    ),
                )
                self._seg_table.setItem(
                    r, 1, pg.QtWidgets.QTableWidgetItem(tr_chord_label(c.label))
                )
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
        self._keyboard_cursor: int | None = None
        self.setMinimumHeight(78)
        self.setCursor(pg.QtCore.Qt.CursorShape.PointingHandCursor)

    def set_range(self, min_midi: int, max_midi: int) -> None:
        """Show exactly this span, keeping what is sounding.

        The finder chooses its span explicitly (a singing range rather than
        the whole piano, where the keys are too narrow to name). Changing it
        must not move a key under the pointer: the keyboard is redrawn in
        place, the lit keys stay lit, and a staged cursor that falls outside
        the new span is dropped rather than slid to an edge nobody chose.
        """
        self.min_midi = int(min_midi)
        self.max_midi = int(max_midi)
        self._whites = [
            m for m in range(self.min_midi, self.max_midi + 1) if m % 12 in self.WHITE_PCS
        ]
        cursor = self._keyboard_cursor
        if cursor is not None and not self.min_midi <= cursor <= self.max_midi:
            self._keyboard_cursor = None
        self.update()

    def set_keyboard_cursor(self, midi: int | None) -> None:
        """Highlight the piano key chosen from the keyboard."""
        self._keyboard_cursor = midi
        self.update()

    @property
    def keyboard_cursor(self) -> int | None:
        """The pitch staged for Enter — the highlighted key itself."""
        return self._keyboard_cursor

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Publish the key under the pointer, checking black keys first."""
        if event.button() != pg.QtCore.Qt.MouseButton.LeftButton:
            return
        midi = self._midi_at(event.position().x(), event.position().y())
        if midi is not None:
            self.key_clicked.emit(midi)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """The keyboard's route: arrows stage a key, Enter publishes it.

        One convention for every owner of a `PianoView`: an arrow with
        nothing staged yet starts from middle C (clamped into this
        keyboard's range), the arrows step one semitone, Enter emits
        `key_clicked` for the staged key. The results page installs an
        event filter that stages the SELECTED note first and consumes the
        same keys, so it keeps its own semantics; this route is what a
        keyboard with no selection (the finder's) falls back on.
        """
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            step = -1 if key in (Qt.Key.Key_Left, Qt.Key.Key_Down) else 1
            current = self._keyboard_cursor
            if current is None:
                current = min(self.max_midi, max(self.min_midi, 60))
            self.set_keyboard_cursor(min(self.max_midi, max(self.min_midi, current + step)))
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._keyboard_cursor is not None:
                self.key_clicked.emit(self._keyboard_cursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """The staged key belongs to the focused keyboard: blur clears it."""
        self.set_keyboard_cursor(None)
        super().focusOutEvent(event)

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
        self._active = {m for m, spans in self._spans.items() if any(s <= t < e for s, e in spans)}
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
                int(x),
                0,
                int(ww) - 1,
                h,
                QColor(*self._active_colors[midi])
                if midi in self._active_colors
                else QColor(ACCENT)
                if key_on
                else QColor(KEY_WHITE),
            )
            p.setPen(
                QPen(
                    QColor(ACCENT) if midi == self._keyboard_cursor else QColor(BORDER),
                    2 if midi == self._keyboard_cursor else 1,
                )
            )
            p.drawRect(int(x), 0, int(ww) - 1, h - 1)
        # black keys (drawn over)
        bw, bh = ww * 0.62, h * 0.62
        for i in range(n_white - 1):
            midi = self._white_midi(i)
            if self._has_black_after(midi):
                x = (i + 1) * ww - bw / 2
                key_on = midi + 1 in self._active
                p.fillRect(
                    int(x),
                    0,
                    int(bw),
                    int(bh),
                    QColor(*self._active_colors[midi + 1])
                    if midi + 1 in self._active_colors
                    else QColor(ACCENT_SOFT)
                    if key_on
                    else QColor(KEY_BLACK),
                )
                # an outline keeps the ebony keys an OBJECT on the dark
                # theme, where their fill is all but the page's own color
                p.setPen(
                    QPen(
                        QColor(ACCENT) if midi + 1 == self._keyboard_cursor else QColor(BORDER),
                        2 if midi + 1 == self._keyboard_cursor else 1,
                    )
                )
                p.drawRect(int(x), 0, int(bw), int(bh))
        # note names on the white keys (all of them when they fit, else the Cs)
        p.setPen(QColor(KEY_LABEL))
        f = p.font()
        f.setPointSize(7)
        p.setFont(f)
        for i in range(n_white):
            midi = self._white_midi(i)
            if show_names or midi % 12 == 0:
                p.drawText(int(i * ww) + 2, h - 4, note_display(midi))
        # one frame around the instrument: the keyboard is an object in the
        # page, not a band of stripes running edge to edge
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(pg.QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(0, 0, w - 1, h - 1)
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
            f"background: {ERROR_BG}; color: {ERROR_TEXT}; border: 1px solid {ERROR_BORDER};"
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
        from PyQt6.QtCore import QPointF, QRectF, Qt
        from PyQt6.QtGui import QColor, QPainter, QPen

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        # card background, consistent with the hint cards
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(QColor(BG_CARD))
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
        p.setPen(QPen(QColor(BORDER_SOFT), 1))
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
                    p.setPen(QColor(TEXT_DIM))
                    p.setFont(f)
                    p.drawText(QPointF(x - ww / 2, h - 8), note_display(midi))
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

    def sizeHint(self):  # noqa: N802 (Qt naming)
        """A gauge needs a readable span even where no layout stretches it.

        Centering a painter widget with no size hint gives it ZERO width
        (measured on the finder's answer card: the needle drew nothing).
        """
        from PyQt6.QtCore import QSize

        return QSize(340, self.height())

    def minimumSizeHint(self):  # noqa: N802 (Qt naming)
        from PyQt6.QtCore import QSize

        return QSize(180, self.height())

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
        p.setPen(QPen(QColor(BORDER), 1.0))
        p.setBrush(QColor(BG_CARD))
        p.drawRoundedRect(track, 3.0, 3.0)
        # center tick: the target (the nearest piano note)
        p.setPen(QPen(QColor(TEXT_DIM), 1.6))
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
        self.label_input_state = pg.QtWidgets.QLabel("")
        self.label_input_state.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        self.tuning = TuningMeter()
        self.label_tuning = pg.QtWidgets.QLabel("")
        self._tuning_cents: float | None = None
        self._rms = 0.0
        left.addWidget(self.label_note)
        left.addWidget(self.label_hz)
        left.addWidget(self.tuning)
        left.addWidget(self.label_tuning)
        left.addWidget(self.label_rms)
        left.addWidget(self.label_input_state)
        lay.addLayout(left, stretch=1)
        # the CONFIRMED-note counter, BIG: "is the recognition following me?"
        # is the singer's first question, and the status line was too small to
        # answer it — a glance answers now. Updated by the same live tick that
        # renders the roll (one owner of the live rendering).
        right = pg.QtWidgets.QVBoxLayout()
        self.label_note_count = pg.QtWidgets.QLabel("0")
        self.label_note_count.setAlignment(pg.QtCore.Qt.AlignmentFlag.AlignRight)
        self.label_note_count.setStyleSheet(f"font-size: 30pt; font-weight: bold; color: {ACCENT};")
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
        self.label_tuning.setText(
            tuning_advice(self._tuning_cents) if self._tuning_cents is not None else ""
        )

    def update_note(self, midi_float: float | None, hz: float, cents: float | None = None) -> None:
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
            self._tuning_cents = None
            self.label_tuning.clear()
            return
        nearest = int(round(midi_float))
        self.label_note.setText(note_display_pair(nearest))
        self.label_hz.setText(f"{format_number(hz, 1)} Hz")
        # the sung pitch vs its NEAREST piano note, in cents — the number a
        # singer works with when practicing intonation
        deviation = cents_deviation(midi_float) if cents is None else cents
        self._tuning_cents = deviation if np.isfinite(deviation) else None
        self.tuning.set_cents(self._tuning_cents)
        self.label_tuning.setText(
            tuning_advice(self._tuning_cents) if self._tuning_cents is not None else ""
        )

    def set_input_state(self, key: str, **values: object) -> None:
        """Show the captured input's published status beside its level."""
        self.label_input_state.setText(tr(key).format(**values))

    def set_theme(self) -> None:
        """Refresh the local styles that are assembled when this view is built."""
        self.label_note.setStyleSheet(
            f"font-size: 30pt; font-weight: bold; color: {NOTE_TEXT_COLOR};"
        )
        self.label_input_state.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        self.label_note_count.setStyleSheet(
            f"font-size: 30pt; font-weight: bold; color: {ACCENT};"
        )
        self.label_count_unit.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        self.tuning.update()
        self.update()

    def update_rms(self, rms: float) -> None:
        self._rms = rms
        if rms <= 0:
            self.label_rms.setText(tr("ui level idle"))
            return
        bars = int(min(1.0, rms * 20) * 20)
        self.label_rms.setText(tr("key level") + " : " + "█" * bars + "░" * (20 - bars))


# Typography carries most of the window's character: a humanist UI face for
# text and a serif for section titles, so the app reads like a printed score
# rather than a dashboard of cards. Families are resolved once against what
# the machine actually has (a wrist of candidates, one fallback), because a
# QSS naming a missing family silently exchanges it for the system default.
UI_FONT_CANDIDATES = (
    "Segoe UI",
    "Inter",
    "Aptos",
    "Helvetica Neue",
    "Noto Sans",
    "DejaVu Sans",
    "Arial",
)
DISPLAY_FONT_CANDIDATES = (
    "Georgia",
    "Iowan Old Style",
    "Palatino Linotype",
    "Charter",
    "Noto Serif",
    "DejaVu Serif",
    "Times New Roman",
)

THEMES = ("dark", "light")

# The primary plate's hover and pressed colours are hand-picked per theme (an
# ACCENT derivative read as flat), and they carry the page ink, so they are the
# one part of a button a palette swap can silently make unreadable: the light
# hover reached only 3.58:1 against its own label. Named here so the contrast
# test measures them instead of trusting a literal inside the style string.
PLATE_STATES = {
    "dark": {"hover": "#E6BC8C", "pressed": "#B8874F"},
    "light": {"hover": "#7E521F", "pressed": "#6E4718"},
}

# Application-wide QSS: night-blue surfaces, ivory text, copper accent.
# The three semantic verdict colors (in tune / close / off) keep their own
# scale — a note's intonation is MEASURED, not decorative.
ACCENT = "#D9A76C"
ERROR_BG = "#5c2522"
ERROR_TEXT = "#ffe5df"
ERROR_BORDER = "#E08D80"
BG_APP = "#111827"
BG_CARD = "#202B3A"
BG_SUNKEN = "#0C1421"
BORDER = "#4E6580"
BORDER_SOFT = "#2C3A4C"
TEXT = "#F4EFE6"
TEXT_DIM = "#B9C2CF"
TEXT_FAINT = "#97A3B4"
ACCENT_SOFT = "#3B3226"
NOTE_TEXT_COLOR = "#E9C99B"
KEY_WHITE = "#F1EADB"
KEY_BLACK = "#101924"
KEY_LABEL = "#5A5347"
STYLESHEET = ""
HINT_CARD_STYLE = ""


_ACTIVE_THEME = "dark"
_UI_FONT_FAMILY: str | None = None
_DISPLAY_FONT_FAMILY: str | None = None


def ui_font_family() -> str:
    """The first installed UI face, resolved once a font database exists."""
    global _UI_FONT_FAMILY
    if _UI_FONT_FAMILY is None:
        _UI_FONT_FAMILY = _installed_family(UI_FONT_CANDIDATES)
    return _UI_FONT_FAMILY or UI_FONT_CANDIDATES[-1]


def display_font_family() -> str:
    """The first installed serif for section titles."""
    global _DISPLAY_FONT_FAMILY
    if _DISPLAY_FONT_FAMILY is None:
        _DISPLAY_FONT_FAMILY = _installed_family(DISPLAY_FONT_CANDIDATES)
    return _DISPLAY_FONT_FAMILY or DISPLAY_FONT_CANDIDATES[-1]


def _installed_family(candidates: tuple[str, ...]) -> str | None:
    """The first candidate the font database knows, or None before Qt exists.

    `widgets` builds its stylesheet at import time, before any QApplication
    exists. Answering None there (rather than caching the fallback) keeps the
    real resolution for the first rebuild, which `set_theme` runs at startup.
    """
    from PyQt6.QtGui import QFontDatabase
    from PyQt6.QtWidgets import QApplication

    if QApplication.instance() is None:
        return None
    installed = set(QFontDatabase.families())
    return next((name for name in candidates if name in installed), None)


def display_title_style(size_pt: float = 13.0, color: str | None = None) -> str:
    """Style a page title in the serif face, at a display size."""
    return (
        f'font-family: "{display_font_family()}";'
        f" font-size: {size_pt:g}pt; color: {color or TEXT};"
    )


def theme_colors(name: str) -> dict[str, str]:
    """Return the semantic palette for one supported application theme."""
    if name not in THEMES:
        raise ValueError(f"unknown theme: {name}")
    if name == "light":
        return {
            "accent": "#8A5A24",
            "bg_app": "#F5F1E8",
            "bg_card": "#FFFCF4",
            "bg_sunken": "#F1E7D5",
            "border": "#B0844F",
            "border_soft": "#E3D3B8",
            "text": "#241E17",
            "text_dim": "#544B40",
            "text_faint": "#6C6152",
            "accent_soft": "#F0E2CC",
            "note_text": "#7A4E1C",
            # a piano keyboard is ivory and ebony in BOTH themes: what changes
            # is the paper behind it, never the keys themselves
            "key_white": "#FBF7EF",
            "key_black": "#1C2430",
            "key_label": "#6A5F50",
        }
    return {
        "accent": "#D9A76C",
        "bg_app": "#111827",
        "bg_card": "#202B3A",
        "bg_sunken": "#0D1218",
        "border": "#7A6749",
        "border_soft": "#3A332B",
        "text": "#F4EFE6",
        "text_dim": "#C9C2B3",
        "text_faint": "#A79E8C",
        "accent_soft": "#3B3226",
        "note_text": "#E9C99B",
        "key_white": "#F1EADB",
        "key_black": "#101924",
        "key_label": "#5A5347",
    }


def set_theme(name: str) -> None:
    """Select the color tokens used when Qt repaints the shared views."""
    global _ACTIVE_THEME
    if name not in THEMES:
        raise ValueError(f"unknown theme: {name}")
    _ACTIVE_THEME = name
    _apply_theme_tokens()


def current_theme() -> str:
    """Return the currently selected application-wide color theme."""
    return _ACTIVE_THEME


def _apply_theme_tokens() -> None:
    """Refresh shared tokens and hand-drawn view colors for the active theme."""
    global ACCENT, BG_APP, BG_CARD, BG_SUNKEN, BORDER, BORDER_SOFT
    global TEXT, TEXT_DIM, TEXT_FAINT, ACCENT_SOFT, NOTE_TEXT_COLOR
    global KEY_WHITE, KEY_BLACK, KEY_LABEL
    global GHOST_FILL, GHOST_CHANGED_FILL, GHOST_NEAR_FILL, STYLESHEET, HINT_CARD_STYLE
    from . import staff

    colors = theme_colors(_ACTIVE_THEME)
    ACCENT = colors["accent"]
    BG_APP = colors["bg_app"]
    BG_CARD = colors["bg_card"]
    BG_SUNKEN = colors["bg_sunken"]
    BORDER = colors["border"]
    BORDER_SOFT = colors["border_soft"]
    TEXT = colors["text"]
    TEXT_DIM = colors["text_dim"]
    TEXT_FAINT = colors["text_faint"]
    ACCENT_SOFT = colors["accent_soft"]
    NOTE_TEXT_COLOR = colors["note_text"]
    KEY_WHITE = colors["key_white"]
    KEY_BLACK = colors["key_black"]
    KEY_LABEL = colors["key_label"]
    if _ACTIVE_THEME == "light":
        GHOST_FILL = (90, 130, 200, 80)
        GHOST_CHANGED_FILL = (173, 105, 0, 210)
        GHOST_NEAR_FILL = (173, 130, 0, 180)
    else:
        GHOST_FILL = (90, 130, 200, 60)
        GHOST_CHANGED_FILL = (230, 160, 60, 200)
        GHOST_NEAR_FILL = (230, 190, 60, 160)
    staff.set_theme(_ACTIVE_THEME)
    HINT_CARD_STYLE = (
        f"color: {TEXT_DIM}; font-size: 9pt;"
        f"background: {BG_SUNKEN}; border: none; border-radius: 4px;"
        "padding: 8px 14px;"
    )
    STYLESHEET = _stylesheet()


def action_button_style(primary: bool = False) -> str:
    """Shared style for the take-off actions (primary = copper plate).

    The primary action is a FLAT copper plate, not a gradient pill: one
    remembered shape per screen instead of a ramp competing with the accent.
    """
    if primary:
        plate = PLATE_STATES[_ACTIVE_THEME]
        hover = plate["hover"]
        pressed = plate["pressed"]
        # A copper plate still needs an edge to sit on: the lit top and
        # grounded bottom lines are what make the press feel like one.
        hi = _blend(ACCENT, "#FFFFFF", 0.34)
        lo = _blend(ACCENT, "#000000", 0.24)
        return (
            f"QPushButton {{ background: {ACCENT}; color: {BG_APP};"
            "  font-size: 11pt; font-weight: 600; padding: 9px 22px;"
            "  border: 2px solid transparent;"
            f"  border-top-color: {hi}; border-bottom-color: {lo};"
            "  border-radius: 6px; }"
            f"QPushButton:hover {{ background: {hover}; border-color: {hover}; }}"
            f"QPushButton:pressed {{ background: {pressed};"
            f"  border-top-color: {lo}; border-bottom-color: {hi}; }}"
            # the ring is the plate's own ink: a copper ring on a copper plate
            # would disappear at exactly the moment focus matters
            f"QPushButton:focus {{ border: 2px dashed {BG_APP}; }}"
            f"QPushButton:disabled {{ background: transparent; color: {TEXT_FAINT};"
            f"  border: 1px solid {BORDER_SOFT}; }}"
        )
    edge_high = _blend(BORDER_SOFT, TEXT, 0.16)
    edge_low = _blend(BORDER_SOFT, BG_APP, 0.5)
    return (
        "QPushButton { font-size: 11pt; padding: 9px 22px;"
        f"  border: 1px solid {BORDER_SOFT}; border-top-color: {edge_high};"
        f"  border-bottom-color: {edge_low}; border-radius: 6px;"
        f"  background: {BG_CARD}; }}"
        f"QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT};"
        f"  background: {ACCENT_SOFT}; }}"
        f"QPushButton:pressed {{ background: {BG_SUNKEN};"
        f"  border-top-color: {edge_low}; border-bottom-color: {edge_high}; }}"
        f"QPushButton:focus {{ border: 2px dashed {ACCENT}; }}"
        f"QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER_SOFT};"
        f"  background: transparent; }}"
    )


def link_button_style() -> str:
    """Style a navigation or disclosure row as a quiet text link.

    Opening another file or unfolding the details is a way OUT of the
    current task, not a second action on it: a bordered button gave four
    full-width rows the same weight as the one the singer must press.
    """
    return (
        "QPushButton { background: transparent; border: none; padding: 5px 2px;"
        "  min-height: 24px;"
        f" color: {TEXT_DIM}; font-size: 10pt; }}"
        f"QPushButton:hover {{ color: {ACCENT}; }}"
        f"QPushButton:checked {{ color: {TEXT}; font-weight: 600; }}"
        f"QPushButton:focus {{ border: 2px dashed {ACCENT}; }}"
        f"QPushButton:disabled {{ color: {TEXT_FAINT}; }}"
    )


def quiet_button_style() -> str:
    """Shared style for the home page's secondary controls.

    Settings and follow-ups are not take-off actions: smaller type and a thin
    border keep them visibly secondary and let the whole row fit beside the
    appearance selector on a normal desktop window in both languages.
    """
    return (
        "QPushButton { font-size: 10pt; padding: 6px 14px; border-radius: 6px;"
        f" border: 1px solid {BORDER_SOFT}; background: transparent; color: {TEXT_DIM}; }}"
        f"QPushButton:hover {{ border-color: {ACCENT}; color: {TEXT};"
        f"  background: {ACCENT_SOFT}; }}"
        f"QPushButton:focus {{ border: 2px dashed {ACCENT}; }}"
        f"QPushButton:disabled {{ color: {TEXT_FAINT}; }}"
    )


def theme_stylesheet() -> str:
    """The current complete stylesheet, rebuilt from the active palette."""
    return STYLESHEET


def _blend(color: str, toward: str, weight: float) -> str:
    """Blend `color` toward `toward` by `weight` (0..1).

    The controls' bevel is DERIVED from the palette instead of a pair of
    hand-picked greys: fixed edges looked right in one theme and turned into
    two foreign lines in the other. Deriving them also keeps the sheet free of
    gradients, which the flat shell refuses.
    """
    from PyQt6.QtGui import QColor

    start = QColor(color)
    end = QColor(toward)
    return QColor(
        round(start.red() + (end.red() - start.red()) * weight),
        round(start.green() + (end.green() - start.green()) * weight),
        round(start.blue() + (end.blue() - start.blue()) * weight),
    ).name()


def _stylesheet() -> str:
    """Build the complete application stylesheet from the active tokens.

    The look is editorial rather than dashboard-like: no gradients and no
    rounded card around every block. Sections are separated by one hairline
    and titled in the serif face; controls keep a thin border and a 6 px
    corner, and gain their depth from a two-tone bevel (a lit top edge over a
    grounded bottom edge) that inverts when pressed. The accent is spent on the current
    choice and the primary action only, so the eye lands where the singer must
    act.
    """
    edge_high = _blend(BORDER_SOFT, TEXT, 0.16)
    edge_low = _blend(BORDER_SOFT, BG_APP, 0.5)
    return f"""
QMainWindow, QWidget {{
    background: {BG_APP}; color: {TEXT};
    font-family: "{ui_font_family()}"; font-size: 10pt;
}}
QLabel {{ color: {TEXT}; background: transparent; }}
/* Section rules: `#masthead` carries the weight of a page's top rule, and
   `#hairline` separates the blocks inside it. Named here so a theme switch
   repaints them with everything else. */
QFrame#masthead {{ background: {ACCENT}; border: none; }}
QFrame#hairline {{ background: {BORDER_SOFT}; border: none; }}
/* Buttons are TACTILE without a gradient: a lit top edge, a grounded bottom
   edge, a warm copper wash on hover. That bevel INVERTS on :pressed, so the
   control sinks under the click instead of only changing colour; without it a
   flat rectangle reads as a picture of a button. */
QPushButton {{
    background: {BG_CARD}; color: {TEXT};
    border: 1px solid {BORDER_SOFT};
    border-top-color: {edge_high};
    border-bottom-color: {edge_low};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; }}
QPushButton:pressed {{
    background: {BG_SUNKEN};
    border-top-color: {edge_low};
    border-bottom-color: {edge_high};
}}
QPushButton:checked {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; }}
QPushButton:focus {{ border: 2px dashed {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER_SOFT}; }}
QPushButton:flat {{ border: none; background: transparent; }}
QSlider::groove:horizontal {{ height: 3px; background: {BG_SUNKEN}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
    background: {ACCENT}; border: 1px solid {edge_high};
}}
QSlider::handle:horizontal:hover {{ background: {TEXT}; }}
QCheckBox, QSpinBox {{ color: {TEXT}; background: transparent; }}
/* The click target is the whole label, not the 14 px indicator — the page's
   file input has the same shape (a hidden field, its label is the target). */
QCheckBox {{ min-height: 24px; }}
/* The ring's 2 px are reserved in BOTH states: a border drawn only on :focus
   moved the slider's hint by 4 px (84x24 -> 88x28). 20 px of body plus the two
   2 px edges is the same 24 px target. */
QSlider {{ border: 2px solid transparent; min-height: 20px; }}
QCheckBox:hover {{ color: {TEXT}; }}
/* Every family that can take keyboard focus draws the SAME ring: 2 px of
   copper, dashed so it reads as focus rather than as the control's edge, and
   measured at 3:1 against the surface behind it (WCAG 1.4.11). A family that
   gains a focus rule here is one the Tab chain can never leave blind. */
QCheckBox:focus, QSlider:focus, QListWidget:focus, QTableWidget:focus {{
    border: 2px dashed {ACCENT};
}}
QComboBox {{ min-height: 26px; }}
QComboBox:focus {{ border: 2px dashed {ACCENT}; }}
QComboBox:disabled {{ color: {TEXT_FAINT}; }}
QLineEdit {{
    background: {BG_SUNKEN}; color: {TEXT};
    border: 1px solid {BORDER_SOFT}; border-radius: 3px; padding: 3px 6px;
}}
/* The ring pays for itself out of the padding (1 + 3 = 2 + 2), so the field
   keeps the same size hint: measured 137x25 in both states. */
QLineEdit:focus {{ border: 2px dashed {ACCENT}; padding: 2px 5px; }}
QLineEdit:disabled {{ color: {TEXT_FAINT}; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 2px;
    border: 1px solid {BORDER}; background: transparent;
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSpinBox {{
    background: {BG_SUNKEN}; border: 1px solid {BORDER_SOFT}; border-radius: 3px;
    padding: 2px 6px;
}}
QSpinBox:focus {{ border: 2px dashed {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; background: transparent; border: none; }}
QGroupBox {{
    background: transparent; border: none;
    border-top: 1px solid {BORDER_SOFT};
    margin-top: 12px; padding: 10px 0 2px 0;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 0; padding: 0 12px 0 0;
    font-family: "{display_font_family()}"; font-size: 12pt; color: {TEXT};
    background: transparent;
}}
QListWidget {{ background: transparent; border: none; padding: 2px 0; }}
QListWidget::item {{ border-radius: 3px; padding: 3px 6px; color: {TEXT_DIM}; }}
QListWidget::item:hover {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QTableWidget {{
    background: transparent; alternate-background-color: {BG_SUNKEN};
    gridline-color: transparent; border: none;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QTableWidget::item {{ border-bottom: 1px solid {BORDER_SOFT}; padding: 3px 6px; }}
QTableWidget::item:selected {{ border-bottom: 1px solid {ACCENT}; color: {TEXT}; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background: transparent; color: {TEXT_FAINT}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 4px 6px; font-size: 8pt;
}}
QScrollBar:vertical {{ background: transparent; width: 6px; margin: 2px 1px; }}
QScrollBar::handle:vertical {{ background: {BORDER_SOFT}; border-radius: 3px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 6px; margin: 1px 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_SOFT}; border-radius: 3px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QProgressBar {{
    background: {BG_SUNKEN}; border: none; border-radius: 2px;
    text-align: center; color: {TEXT_DIM}; height: 16px; font-size: 9pt;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
/* The drawn surfaces (the roll, the waveform) are INSTRUMENTS: a fine warm
   frame sets them apart from the paper they lie on, with the same lit top
   edge as a control so they belong to the same object. */
QGraphicsView {{
    border: 1px solid {BORDER_SOFT};
    border-top-color: {edge_high};
    background: {BG_SUNKEN};
}}
QStatusBar {{ color: {TEXT_FAINT}; background: {BG_APP}; border-top: 1px solid {BORDER}; }}
QToolTip {{
    background: {BG_CARD}; color: {TEXT}; border: 1px solid {BORDER_SOFT};
    border-radius: 6px; padding: 6px 8px;
}}
"""


set_theme("dark")
