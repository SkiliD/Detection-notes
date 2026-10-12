"""Staff view: a real (engraved-style) staff drawn with QPainter.

Why hand-drawn: the app must show notes on a staff *while singing*, so an
external engraver (MuseScore/LilyPond, used for the PDF export) is out of the
question — and no music font is guaranteed to be installed (the glyphs
U+1D11E and friends are missing on many systems). Everything is therefore
vector: clef, key signature, noteheads, stems, flags, dots, ledger lines and
barlines.

The musical decisions (spelling, durations, measure grid, accidentals) live
in `core/notation.py`; this module only turns them into pixels. Both modes
share one renderer: a *beat window* is drawn at a constant scale, either
paginated system by system (finished analysis) or scrolled (live take).
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ..core.i18n import tr, tr_figure, tr_key_label
from ..core.grid import BeatGrid
from ..core.music import sane_bpm
from ..core.notation import STAFF_LINES, StaffLayout, layout_staff

# Dark-theme palette, shared with the plot views.
BG = QColor("#12161c")
INK = QColor("#e6ebf2")          # noteheads, stems, barlines
STAFF_INK = QColor("#8f9aa6")    # staff lines, clef, ledger lines
SOFT_INK = QColor("#9fb3c8")     # header text, measure numbers
REST_INK = QColor("#c9d3dd")     # rests: readable, but dimmer than notes
BREATH_INK = QColor("#7fd4a0")   # the heard-breath "B" annotation
PREVIEW_INK = QColor("#ffc86e")  # the note being sung (piano roll amber)
CURSOR_INK = QColor("#ff9642")

S = 9.0            # staff line spacing, px
PX_PER_BEAT = 3.6  # horizontal room per quarter beat, in staff spacings
TOP_LINE_STEP = 38  # F5 sits on the top line
CLEF_X = 12.0
# G-clef geometry, in staff spaces from the top staff line (G line at 3.0).
CLEF_CENTER = (0.10, 3.05)   # centre of the belly spiral
CLEF_R0, CLEF_R1 = 1.05, 0.28
CLEF_TURNS = 1.9
LEFT_PAD = 8.0
RIGHT_PAD = 14.0
HEADER_H = 26.0


class StaffView(QWidget):
    """Monophonic treble staff. `live=True` scrolls instead of paginating."""

    def __init__(self, parent=None, *, live: bool = False):
        super().__init__(parent)
        self._live = live
        self._notes: list = []
        self._rests: list = []
        self._preview = None
        self._grid = BeatGrid(120.0)
        self._key = None
        self._positions = None
        self._k = 1.0  # engraving scale: grows into the room, shrinks if needed
        self.setMinimumHeight(104 if live else 132)
        self.setToolTip(tr("ui staff tooltip"))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def set_score(
        self, notes, *, grid: BeatGrid | None = None, key=None, positions=None,
        rests=None,
    ) -> None:
        """Show a finished analysis (audio file, or a closed live take)."""
        self._notes = list(notes or [])
        self._rests = list(rests or [])
        self._preview = None
        self._grid = grid or BeatGrid(None)
        self._key = key
        self._positions = positions
        self.update()

    def set_live(self, notes, preview=None, *, grid: BeatGrid | None = None) -> None:
        """Show a take in progress: the notes so far + the note being sung.

        The grid of a live take has no meter yet (the fallback 4/4 stands)
        and its tempo comes from the notes sung so far. The notes sit at
        their REAL onsets on that tempo (the grid path): chaining written
        durations from zero distorted the timeline — short notes with gaps
        collapsed into one measure, the measure count told the singer a
        different story than the seconds they actually sang.
        """
        self._notes = list(notes or [])
        self._rests = []  # no rest timing during the take: notes only
        self._preview = preview
        self._grid = grid or BeatGrid(None)
        self._key = None
        # real onsets, on the detected tempo: (measure, beat) per note
        self._positions = self._grid.positions(self._notes) if self._notes else None
        self.update()

    def clear(self) -> None:
        self.set_score([])

    def layout(self) -> StaffLayout:
        """Current layout (also read by the tests)."""
        return layout_staff(
            self._notes,
            grid=self._grid,
            key=self._key,
            positions=self._positions,
            rests=self._rests,
            preview=self._preview,
        )

    def _metrics(self, layout: StaffLayout) -> tuple[float, float, float]:
        """(px per beat, x of the first note slot, usable width) in staff units."""
        px_beat = S * PX_PER_BEAT
        x_notes = self._notes_x(layout)
        usable = max(px_beat, self.unit_width - x_notes - RIGHT_PAD)
        return px_beat, x_notes, usable

    @property
    def unit_width(self) -> float:
        """Widget width in staff units (the painter is scaled by `_k`)."""
        return self.width() / self._k

    def _prepare(self) -> StaffLayout:
        """Layout + engraving scale for the current widget size."""
        layout = self.layout()
        self._k = self._fit_scale(layout)
        return layout

    # Engraving scales tried from big to small: the first one whose systems
    # fit the widget wins, so a short score fills the view and a long take
    # shrinks instead of being clipped.
    SCALES = (1.6, 1.45, 1.3, 1.15, 1.0, 0.9, 0.8, 0.7, 0.6, 0.55)

    def _fit_scale(self, layout: StaffLayout) -> float:
        """Largest engraving scale whose systems fit the widget rectangle."""
        if not layout.notes:
            return 1.0
        if self._live:
            # one scrolling system: grow into the room available, capping the
            # zoom so the visible time window stays singable (~4 measures)
            room = max(60.0, self.height() - HEADER_H - 4.0)
            return max(1.0, min(1.8, room / (8.2 * S)))
        px_beat = S * PX_PER_BEAT
        for k in self.SCALES:
            room = self.height() / k - HEADER_H - 4.0
            usable = max(px_beat, self.width() / k - self._notes_x(layout) - RIGHT_PAD)
            per_system = max(1, int(usable // (layout.beats_per_measure * px_beat)))
            systems = max(1, -(-layout.measures // per_system))
            if systems * 8.2 * S <= room:
                return k
        return self.SCALES[-1]

    def visible_window(self) -> tuple[float, float]:
        """Beats currently drawn: the scrolling window in live mode."""
        layout = self._prepare()
        if not layout.notes:
            return (0.0, 0.0)
        if not self._live:
            return (0.0, layout.total_beats)
        px_beat, _, usable = self._metrics(layout)
        visible = usable / px_beat
        start = max(0.0, layout.total_beats - visible)
        return (start, start + visible)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: ANN001, N802
        layout = self._prepare()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), BG)
        p.scale(self._k, self._k)  # everything below works in staff units
        if not layout.notes:
            p.setPen(SOFT_INK)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, _EMPTY)
            p.end()
            return

        px_beat, x_notes, usable = self._metrics(layout)
        self._paint_header(p, layout)

        if self._live:
            start, end = self.visible_window()
            self._paint_window(
                p, layout, HEADER_H, x_notes, px_beat, start, end,
                first_system=False,
            )
            self._paint_cursor(p, layout, HEADER_H, x_notes, px_beat, start)
            p.end()
            return

        measure_px = layout.beats_per_measure * px_beat
        per_system = max(1, int(usable // measure_px))
        systems = max(1, -(-layout.measures // per_system))
        system_h = 4 * S + 4.2 * S
        for i in range(systems):
            first = i * per_system + 1
            last = min(layout.measures, first + per_system - 1)
            self._paint_window(
                p, layout, HEADER_H + i * system_h, x_notes, px_beat,
                (first - 1) * layout.beats_per_measure,
                last * layout.beats_per_measure,
                first_system=(i == 0),
            )
        p.end()

    def _notes_x(self, layout: StaffLayout) -> float:
        """Left edge of the first note slot (after clef + key + meter)."""
        return (
            CLEF_X
            + 2.6 * S                                   # clef
            + 0.95 * S * len(layout.signature)          # key signature
            + 2.4 * S                                   # time signature
        )

    def _paint_header(self, p: QPainter, layout: StaffLayout) -> None:
        """Tempo · meter · key above the staff (plain text: no music glyph)."""
        font = QFont()
        font.setPointSizeF(max(6.5, S * 0.82))
        p.setFont(font)
        p.setPen(SOFT_INK)
        parts = [f"{int(round(sane_bpm(self._grid.bpm)))} {tr('key bpm unit')}", layout.meter_label]
        if layout.key_tonic:
            parts.append(tr_key_label(f"{layout.key_tonic} {layout.key_mode or 'major'}"))
        p.drawText(QPointF(LEFT_PAD + 2.0, 14.0), "   ·   ".join(parts))

    def _paint_window(
        self, p, layout, staff_top, x_notes, px_beat, beat_from, beat_to,
        first_system: bool,
    ) -> None:
        """Draw one visible beat window: staff, clef, signature, bars, notes.

        `beat_from` is both the window start and the horizontal origin, so a
        later system starts again at the left edge (the notes of a system 2+
        must NOT be pushed off-screen).
        """
        scroll = beat_from
        bottom = staff_top + 4 * S

        p.setPen(QPen(STAFF_INK, 1.0))
        for step in STAFF_LINES:
            y = staff_top + (TOP_LINE_STEP - step) * (S / 2.0)
            p.drawLine(QPointF(CLEF_X, y), QPointF(self.unit_width - 6.0, y))

        _paint_clef(p, CLEF_X, staff_top)
        sig_x = CLEF_X + 2.6 * S
        for step, alter in layout.signature:
            _paint_accidental(p, alter, sig_x, staff_top + (TOP_LINE_STEP - step) * (S / 2.0))
            sig_x += 0.95 * S
        if first_system:
            num, _, den = layout.meter_label.partition("/")
            font = QFont()
            font.setPointSizeF(max(7.0, S * 1.05))
            font.setBold(True)
            p.setFont(font)
            p.setPen(SOFT_INK)
            p.drawText(QPointF(sig_x + 1.4 * S, staff_top + 1.9 * S), num)
            p.drawText(QPointF(sig_x + 1.4 * S, staff_top + 3.6 * S), den)

        # barlines on the measure grid inside the window
        measure = layout.beats_per_measure
        first_measure = int(scroll // measure) + 1
        k = first_measure
        while k * measure <= beat_to + 1e-6:
            x = x_notes + (k * measure - scroll) * px_beat
            if x_notes - 1.0 <= x <= self.unit_width + 4.0:
                _paint_barline(p, x, staff_top, bottom, final=(k == layout.measures))
            k += 1

        # measure number at the start of the window
        font = QFont()
        font.setPointSizeF(max(6.0, S * 0.68))
        p.setFont(font)
        p.setPen(SOFT_INK)
        p.drawText(QPointF(x_notes, staff_top - 1.15 * S), str(first_measure))

        for note in layout.notes:
            if not (beat_from - 1e-6 <= note.onset < beat_to - 1e-6):
                continue
            x = x_notes + (note.onset - scroll) * px_beat
            y = staff_top + (TOP_LINE_STEP - note.step) * (S / 2.0)
            _paint_note(p, x, y, staff_top, note)

        # rests: same window filter, drawn after the notes (they sit in the
        # written gaps, so nothing can overlap a notehead anyway)
        for rest in layout.rests:
            if not (beat_from - 1e-6 <= rest.onset < beat_to - 1e-6):
                continue
            x = x_notes + (rest.onset - scroll) * px_beat
            _paint_rest(p, x, staff_top, rest)

    def _paint_cursor(self, p, layout, staff_top, x_notes, px_beat, scroll) -> None:
        x = x_notes + (layout.total_beats - scroll) * px_beat
        p.setPen(QPen(CURSOR_INK, 1.4))
        p.drawLine(
            QPointF(x, staff_top - 1.3 * S), QPointF(x, staff_top + 5.3 * S)
        )


_EMPTY = "—"


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------
def _paint_barline(p: QPainter, x: float, top: float, bottom: float, final: bool) -> None:
    if final:  # thin + thick, the classic end-of-piece barline
        p.setPen(QPen(INK, 1.0))
        p.drawLine(QPointF(x - 3.0, top), QPointF(x - 3.0, bottom))
        p.setPen(QPen(INK, 3.0))
        p.drawLine(QPointF(x, top), QPointF(x, bottom))
        return
    p.setPen(QPen(STAFF_INK, 1.0))
    p.drawLine(QPointF(x, top), QPointF(x, bottom))


def _paint_clef(p: QPainter, x: float, staff_top: float) -> None:
    """Treble (G) clef: top curl, spine, spiral belly, tail curl + ball."""
    p.save()
    p.translate(x, staff_top)
    p.setPen(QPen(STAFF_INK, max(1.2, S * 0.16), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(_clef_path())
    p.setBrush(STAFF_INK)  # the little ball closing the tail
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QRectF(-0.80 * S, 4.26 * S, 0.19 * S, 0.19 * S))
    p.restore()


def _clef_path() -> QPainterPath:
    """G clef, in pixels, from staff-space coordinates.

    Origin = top staff line (F5), y grows downwards, the G line is 3 spaces
    below. The belly is a real exponential spiral (1.9 turns), which is what
    makes the shape read as a clef rather than as a scribble.
    """
    s = S
    path = QPainterPath()
    # 1) belly spiral, entered from above the G line
    theta_max = CLEF_TURNS * 2.0 * math.pi
    decay = math.log(CLEF_R0 / CLEF_R1) / theta_max
    cx, cy = CLEF_CENTER
    for i in range(161):
        theta = theta_max * i / 160.0
        r = CLEF_R0 * math.exp(-decay * theta)
        angle = -math.pi / 2.0 + theta
        x = (cx + r * math.cos(angle)) * s
        y = (cy + r * math.sin(angle)) * s
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    # 2) spine, from the top curl down to below the staff
    path.moveTo(0.42 * s, -0.48 * s)
    path.cubicTo(-0.10 * s, 0.70 * s, -0.10 * s, 2.60 * s, 0.06 * s, 4.25 * s)
    # 3) top curl (an open hook over the spine)
    path.moveTo(0.42 * s, -0.48 * s)
    path.cubicTo(0.95 * s, -0.95 * s, 0.75 * s, -1.55 * s, 0.20 * s, -1.60 * s)
    path.cubicTo(-0.30 * s, -1.65 * s, -0.55 * s, -1.15 * s, -0.30 * s, -0.75 * s)
    # 4) tail curl under the staff
    path.moveTo(0.06 * s, 4.25 * s)
    path.cubicTo(0.10 * s, 4.85 * s, -0.45 * s, 5.30 * s, -0.85 * s, 4.95 * s)
    path.cubicTo(-1.10 * s, 4.65 * s, -0.95 * s, 4.30 * s, -0.62 * s, 4.35 * s)
    return path


def _paint_accidental(p: QPainter, alter: int, x: float, y: float) -> None:
    """Sharp (+), natural (0) or flat (-) drawn with strokes; no music font.

    The natural must be its OWN glyph: alter==0 used to fall into the flat
    branch (the only else), so a key-signature letter sung natural was
    engraved with a FLAT — a wrong pitch on paper (F natural in G major
    printed as Fe). The two verticals + two slanted strokes of the natural
    borrow the sharp's geometry, narrowed to the same height as the flat.
    """
    p.save()
    p.setPen(QPen(INK, max(1.0, S * 0.13)))
    p.setBrush(Qt.BrushStyle.NoBrush)
    if alter > 0:
        for dx in (-0.17 * S, 0.19 * S):
            p.drawLine(QPointF(x + dx, y - 0.80 * S), QPointF(x + dx, y + 0.80 * S))
        for dy in (-0.24 * S, 0.30 * S):
            p.drawLine(
                QPointF(x - 0.46 * S, y + dy - 0.14 * S),
                QPointF(x + 0.50 * S, y + dy + 0.14 * S),
            )
    elif alter == 0:
        # natural: two verticals joined by two slanted crossbars — the
        # sharp's strokes, height-reduced so it fits where the flat sat
        for dx in (-0.17 * S, 0.19 * S):
            p.drawLine(QPointF(x + dx, y - 0.80 * S), QPointF(x + dx, y + 0.80 * S))
        p.drawLine(
            QPointF(x - 0.17 * S, y - 0.55 * S),
            QPointF(x + 0.19 * S, y - 0.30 * S),
        )
        p.drawLine(
            QPointF(x - 0.17 * S, y + 0.30 * S),
            QPointF(x + 0.19 * S, y + 0.55 * S),
        )
    else:
        p.drawLine(QPointF(x - 0.06 * S, y - 1.25 * S), QPointF(x - 0.06 * S, y + 0.50 * S))
        path = QPainterPath()
        path.moveTo(x - 0.06 * S, y + 0.05 * S)
        path.cubicTo(x + 0.72 * S, y - 0.40 * S, x + 0.72 * S, y + 0.60 * S, x - 0.06 * S, y + 0.50 * S)
        p.drawPath(path)
    p.restore()


def _paint_note(p: QPainter, x: float, y: float, staff_top: float, note) -> None:
    """Notehead + stem, flags, dot, triplet marker, accidental, ledger lines."""
    color = PREVIEW_INK if note.preview else INK
    written = note.written
    hollow = written.type_ in ("whole", "half")

    p.setPen(QPen(STAFF_INK, 1.0))
    _paint_ledger_lines(p, x, staff_top, note.step)

    if note.accidental:
        _paint_accidental(p, note.alter, x - 1.25 * S, y)

    # notehead (rotated ellipse, the engraved convention)
    p.save()
    p.translate(x, y)
    p.rotate(-20.0)
    w = (1.45 if written.type_ == "whole" else 1.18) * S
    p.setPen(QPen(color, max(1.2, S * 0.17)))
    p.setBrush(Qt.BrushStyle.NoBrush if hollow else color)
    p.drawEllipse(QRectF(-w / 2.0, -0.48 * S, w, 0.96 * S))
    p.restore()

    up = note.step < 34
    if written.type_ != "whole":
        dx = 0.52 * S * (1.0 if up else -1.0)
        tip = y - 3.4 * S if up else y + 3.4 * S
        p.setPen(QPen(color, max(1.2, S * 0.15)))
        p.drawLine(QPointF(x + dx, y), QPointF(x + dx, tip))
        flags = {"eighth": 1, "16th": 2, "32nd": 3}.get(written.type_, 0)
        for i in range(flags):
            path = _flag_path(x + dx, tip, up, i)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

    if written.dots:
        raised = S / 2.0 if note.step % 2 == 0 else 0.0
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(
            QRectF(x + 0.92 * S, y - raised - 0.20 * S, 0.38 * S, 0.38 * S)
        )

    if written.tuplet:
        font = QFont()
        font.setPointSizeF(max(6.0, S * 0.8))
        p.setFont(font)
        p.setPen(color)
        y3 = y - 4.1 * S if up else y + 4.5 * S
        p.drawText(QPointF(x - 0.25 * S, y3), str(written.tuplet[0]))


def _paint_rest(p: QPainter, x: float, staff_top: float, rest) -> None:
    """Vector rest glyphs: whole, half, quarter, eighth (no music font).

    Whole rest hangs FROM the 4th line, half rest sits ON the 3rd line,
    quarter and eighth rests center on the middle line. An unknown or
    dotted shape falls back to the quarter-rest zigzag + explicit dots.
    """
    shape = (rest.shape or "").removesuffix(" pointée")
    dots = rest.written.dots
    p.save()
    pen = QPen(REST_INK, max(1.1, S * 0.22))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    if shape == "pause":
        y = staff_top + (TOP_LINE_STEP - 36) * (S / 2.0)   # hangs from D5
        p.setBrush(REST_INK)
        p.drawRect(QRectF(x, y, 1.5 * S, 0.45 * S))
    elif shape == "demi-pause":
        y = staff_top + (TOP_LINE_STEP - 34) * (S / 2.0)   # sits on B4
        p.setBrush(REST_INK)
        p.drawRect(QRectF(x, y - 0.45 * S, 1.5 * S, 0.45 * S))
    elif shape == "demi-soupir":
        y = staff_top + 2.0 * S                            # middle line
        p.setBrush(REST_INK)
        p.drawEllipse(QRectF(x + 0.10 * S, y - 2.0 * S, 0.55 * S, 0.42 * S))
        p.drawLine(
            QPointF(x + 0.30 * S, y - 1.85 * S), QPointF(x - 0.38 * S, y + 0.75 * S)
        )
    else:  # "soupir" and every fallback (unknown figure, 16th): zigzag
        y = staff_top + 2.0 * S
        path = QPainterPath()
        path.moveTo(x - 0.42 * S, y - 2.05 * S)
        path.lineTo(x + 0.34 * S, y - 1.25 * S)
        path.lineTo(x - 0.16 * S, y - 0.45 * S)
        path.lineTo(x + 0.46 * S, y + 0.35 * S)
        path.cubicTo(
            x + 0.46 * S, y + 1.05 * S,
            x - 0.28 * S, y + 1.30 * S,
            x - 0.28 * S, y + 0.80 * S,
        )
        p.drawPath(path)
    if rest.shape:
        font = QFont()
        font.setPointSizeF(max(5.5, S * 0.65))
        p.setFont(font)
        p.setPen(REST_INK)
        p.drawText(QPointF(x - 0.45 * S, staff_top + 6.2 * S), tr_figure(rest.shape))
    if getattr(rest, "breath", False):
        # the heard air intake: a compact "B" ABOVE the rest — the same
        # annotation a vocal score prints where the singer takes air
        font = QFont()
        font.setPointSizeF(max(6.0, S * 0.8))
        font.setBold(True)
        p.setFont(font)
        p.setPen(BREATH_INK)
        p.drawText(QPointF(x - 0.25 * S, staff_top - 0.6 * S), "B")
    if dots:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(REST_INK)
        y_dot = staff_top + 1.8 * S
        for i in range(dots):
            p.drawEllipse(
                QRectF(x + 1.7 * S + i * 0.35 * S, y_dot, 0.30 * S, 0.30 * S)
            )
    p.restore()


def _paint_ledger_lines(p: QPainter, x: float, staff_top: float, step: int) -> None:
    """Short lines for pitches outside the five-line staff."""
    lo, hi = STAFF_LINES[0], STAFF_LINES[-1]
    if step < lo:
        candidates = range(lo - 2, step - 1, -2)
    elif step > hi:
        candidates = range(hi + 2, step + 1, 2)
    else:
        return
    for s in candidates:
        y = staff_top + (TOP_LINE_STEP - s) * (S / 2.0)
        p.drawLine(QPointF(x - 1.15 * S, y), QPointF(x + 1.15 * S, y))


def _flag_path(x: float, tip: float, up: bool, index: int) -> QPainterPath:
    """One flag at the stem tip; stacked downwards for 16ths, mirrored when down."""
    path = QPainterPath()
    sign = 1.0 if up else -1.0
    off = index * 0.60 * S * sign
    path.moveTo(x, tip + off)
    path.cubicTo(
        x + 0.90 * S, tip + (0.25 * S) * sign + off,
        x + 1.05 * S, tip + (0.95 * S) * sign + off,
        x + 0.30 * S, tip + (1.70 * S) * sign + off,
    )
    path.cubicTo(
        x + 0.70 * S, tip + (1.05 * S) * sign + off,
        x + 0.55 * S, tip + (0.60 * S) * sign + off,
        x, tip + (0.40 * S) * sign + off,
    )
    return path
