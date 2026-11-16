"""Tests for the PianoView keyboard, the PianoRoll live follow and the waveform."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtTest import QTest

from v2m.core.liveseg import LiveNote
from v2m.core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI
from v2m.core.notes import Note
from v2m.gui.widgets import (
    ACCENT,
    PianoRoll,
    PianoView,
    PitchAxis,
    WaveformView,
    figure_color,
)


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _notes():
    return [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),  # C4 (white)
        Note(start=0.5, end=1.0, midi=64, confidence=0.9),  # E4 (white)
        Note(start=1.0, end=1.5, midi=61, confidence=0.9),  # C#4 (black)
    ]


def test_set_notes_indexes_spans(qapp):
    pv = PianoView()
    pv.set_notes(_notes())
    assert set(pv._spans) == {60, 64, 61}
    assert pv._spans[60] == [(0.0, 0.5)]


def test_clicking_the_piano_emits_the_key_under_the_pointer(qapp):
    pv = PianoView(min_midi=60, max_midi=64)
    pv.resize(500, 100)
    pv.show()
    clicked = []
    pv.key_clicked.connect(clicked.append)

    QTest.mouseClick(pv, Qt.MouseButton.LeftButton, pos=QPoint(50, 80))
    assert clicked == [60]

    QTest.mouseClick(pv, Qt.MouseButton.LeftButton, pos=QPoint(130, 20))
    assert clicked[-1] == 61  # C# overlays the adjacent white keys


def test_playhead_lights_sounding_keys_only(qapp):
    pv = PianoView()
    pv.set_notes(_notes())
    pv.set_playhead(0.2)
    assert pv._active == {60}
    pv.set_playhead(0.7)
    assert pv._active == {64}
    pv.set_playhead(1.2)
    assert pv._active == {61}
    # after the last note: nothing lit
    pv.set_playhead(1.6)
    assert pv._active == set()


def test_playhead_without_notes_is_safe(qapp):
    pv = PianoView()
    pv.set_playhead(1.0)  # no crash, nothing lit
    assert pv._active == set()


def test_the_keyboard_range_is_a_parameter(qapp):
    """The note finder displays the 88 keys of a classical piano."""
    full = PianoView(min_midi=PIANO_MIN_MIDI, max_midi=PIANO_MAX_MIDI)
    assert len(full._whites) == 52  # 88 keys, 36 of them black
    assert full._white_midi(0) == 21 and full._white_midi(51) == 108
    assert not full._has_black_after(108)  # nothing after C8
    # the analysis view covers 48..84: 22 whites, C6 included (the 84 key
    # was invisible — a note at the top edge could never light)
    assert len(PianoView()._whites) == 22


def test_set_range_reshapes_the_keyboard_and_keeps_what_is_sounding(qapp):
    """The finder chooses its span in place: keys keep their light, and a
    staged key outside the new span is dropped rather than slid to an edge."""
    pv = PianoView(min_midi=48, max_midi=84)
    pv.set_pressed({60})
    pv.set_keyboard_cursor(72)
    pv.set_range(PIANO_MIN_MIDI, PIANO_MAX_MIDI)
    assert (pv.min_midi, pv.max_midi) == (21, 108)
    assert len(pv._whites) == 52 and pv._white_midi(0) == 21
    assert pv._active == {60}
    assert pv.keyboard_cursor == 72
    pv.set_range(36, 72)
    assert pv.keyboard_cursor == 72  # still inside
    pv.set_keyboard_cursor(24)  # below the new span
    pv.set_range(48, 84)
    assert pv.keyboard_cursor is None
    assert len(pv._whites) == 22


def test_every_black_key_of_the_piano_is_drawn(qapp):
    """Every black key is drawn: the rule must test the right pitch
    classes, all five black notes per octave can light."""
    pv = PianoView(min_midi=PIANO_MIN_MIDI, max_midi=PIANO_MAX_MIDI)
    drawn = {
        pv._white_midi(i) + 1
        for i in range(len(pv._whites))
        if pv._has_black_after(pv._white_midi(i))
    }
    black_pcs = (1, 3, 6, 8, 10)
    expected = {m for m in range(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1) if m % 12 in black_pcs}
    assert drawn == expected and len(drawn) == 36


def test_pressed_keys_are_lit_directly(qapp):
    """`set_pressed` lights the heard key, black keys included."""
    pv = PianoView(min_midi=PIANO_MIN_MIDI, max_midi=PIANO_MAX_MIDI)
    pv.set_pressed({63})  # D#4: a black key that used not to be drawn
    assert pv._active == {63}
    assert not pv.grab().isNull()
    pv.set_pressed()
    assert pv._active == set()


def test_the_pitch_axis_names_every_labelled_row_in_both_notations(qapp):
    """The ordinate carries both names of a note, current language first.

    This test MOVES the one that pinned the single spelling (`Do4` / `C4`
    per language): the axis now prints the same pair the audition readout
    and the note finder print, so a reader who learned one notation never
    has to translate the ordinate in their head.
    """
    from v2m.core.i18n import get_language, set_language

    axis = PitchAxis(orientation="left")
    previous = get_language()
    try:
        set_language("fr")
        assert axis.tickStrings([60, 69], 1.0, 12.0) == ["Do4 / C4", "La4 / A4"]
        set_language("en")
        assert axis.tickStrings([60, 69], 1.0, 12.0) == ["C4 / Do4", "A4 / La4"]
    finally:
        set_language(previous)


def test_the_pitch_axis_anchors_its_labelled_rows_on_the_octaves(qapp):
    """Row every step the axis paints, chosen from the pixels it has.

    pyqtgraph's automatic grid marks arbitrary semitone multiples (a
    60..72 take was labelled Ré4 / Mi4 / Fa#4 / Sol#4 / Do5) — correct
    names nobody counts from. Every step here divides or is divided by the
    octave, so the C rows are always in the labelled set, and the labels
    stay `MIN_LABEL_PX` apart; the finer level is a grid line, never a
    second column of text.
    """
    from math import ceil

    from v2m.core.i18n import note_display_pair

    axis = PitchAxis(orientation="left")
    cases = (
        # (range, axis pixels, expected labelled step)
        ((60.0, 72.0, 250.0), 2.0),  # a take: a named row every whole tone
        ((55.0, 79.0, 300.0), 3.0),
        ((21.0, 108.0, 300.0), 12.0),  # the whole piano: one name per octave
        ((57.0, 63.0, 40.0), 6.0),  # a cramped axis names Do4 alone
        ((61.0, 64.0, 250.0), 1.0),  # no C row in view: every row named
    )
    for (low, high, size), step in cases:
        levels = axis.tickSpacing(low, high, size)
        assert levels[0] == (step, 0.0), (low, high, size)
        assert axis.OCTAVE % step == 0 or step % axis.OCTAVE == 0, step
        labelled = axis.tickValues(low, high, size)[0][1]
        assert labelled, (low, high, size)
        assert all(low <= v <= high for v in labelled)
        assert axis.tickStrings(labelled, 1.0, step) == [
            note_display_pair(int(v)) for v in labelled
        ]
        # one name per C row the range contains and the step lands on
        first_c = ceil(low / axis.OCTAVE) * axis.OCTAVE
        expected_cs = [c for c in range(int(first_c), int(high) + 1, 12) if c % step == 0]
        assert [v for v in labelled if v % 12 == 0] == expected_cs
        # the labels fit: a single one cannot collide with anything
        per_px = size / (high - low)
        assert per_px * step >= axis.MIN_LABEL_PX or len(labelled) == 1
        for spacing, values in axis.tickValues(low, high, size)[1:]:
            assert per_px * spacing >= axis.MIN_TICK_PX, (spacing, per_px)
            assert axis.tickStrings(values, 1.0, spacing) == [""] * len(values)
    assert axis.tickSpacing(60.0, 60.0, 250.0) == []


# ----------------------------------------------------------------- piano roll


def _xrange(roll: PianoRoll) -> tuple[float, float]:
    (x0, x1), _ = roll.getPlotItem().vb.viewRange()
    return float(x0), float(x1)


def _yrange(roll: PianoRoll) -> tuple[float, float]:
    _, (y0, y1) = roll.getPlotItem().vb.viewRange()
    return float(y0), float(y1)


def test_live_pitch_band_frames_what_is_sung(qapp):
    """A stable band keeps an octave error from crushing the roll."""
    roll = PianoRoll()
    roll.begin_live()
    assert _yrange(roll)[1] - _yrange(roll)[0] >= roll.LIVE_Y_MIN
    roll.set_live_pitch_range([72])  # a single note: minimal centered band
    y0, y1 = _yrange(roll)
    assert y1 - y0 == pytest.approx(roll.LIVE_Y_MIN)
    assert y0 < 72 < y1
    roll.set_live_pitch_range([60, 64, 67])
    y0, y1 = _yrange(roll)
    assert y0 <= 60 and y1 >= 67
    roll.set_live_pitch_range([])  # nothing sung: the band does not move
    assert _yrange(roll) == pytest.approx((y0, y1))


def test_begin_live_empties_the_roll_and_fixes_the_window(qapp):
    roll = PianoRoll()
    roll.set_notes(_notes())
    assert roll._bar_items
    roll.begin_live()
    assert not roll._bar_items
    assert _xrange(roll) == pytest.approx((0.0, roll.LIVE_SPAN))


def test_the_comparison_draws_the_final_reading_beside_the_captured_one(qapp):
    """The live-vs-final comparison: ONE overlay bar per final note at its
    own pitch, the moved ones accented, replaced by a second reading and
    cleared by a rebuild or a new take."""
    from PyQt6.QtCore import Qt

    from v2m.core.notes import Note

    roll = PianoRoll()
    roll.set_notes(_notes())
    assert roll._compare_items == []

    final = [Note(0.6, 1.1, 64, 0.9), Note(1.2, 1.7, 62, 0.9)]
    roll.set_comparison_notes(final, moved={1})
    assert len(roll._compare_items) == 2  # the WHOLE final reading, not just moves
    # the moved note carries the solid accent edge, the agreeing one none
    assert roll._compare_items[1].opts["pen"].style() == Qt.PenStyle.SolidLine
    assert roll._compare_items[0].opts["pen"].style() == Qt.PenStyle.NoPen

    # a second reading REPLACES the first: never two overlays at once
    roll.set_comparison_notes(final[:1])
    assert len(roll._compare_items) == 1

    # a note rebuild (set_notes without the fast path) clears the overlay
    roll.set_notes(_notes(), force=True)
    assert roll._compare_items == []

    # ...and a new take starts from a clean roll
    roll.set_comparison_notes(final)
    assert len(roll._compare_items) == 2
    roll.begin_live()
    assert roll._compare_items == []


def test_the_live_final_overlay_can_be_hidden_and_restored(qapp):
    """Comparison toggle leaves captured bars intact and reuses the same final."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        page = win.page_live
        roll = page.piano_live
        roll.set_notes(_notes())
        final = [Note(0.6, 1.1, 64, 0.9)]

        page.set_comparison_notes(final, moved={0})
        assert len(roll._bar_items) == 3
        assert len(roll._compare_items) == 1

        page.set_comparison_visible(False)
        assert len(roll._bar_items) == 3
        assert roll._compare_items == []

        page.set_comparison_visible(True)
        assert len(roll._compare_items) == 1
    finally:
        win.close()


def test_live_indicator_counter_shows_confirmed_notes(qapp):
    """The big counter: "is it following me?" reads at a glance."""
    from v2m.gui.widgets import LiveIndicator

    ind = LiveIndicator()
    assert ind.label_note_count.text() == "0"
    assert ind.label_count_unit.text()  # the unit label is translated
    ind.set_note_count(12)
    assert ind.label_note_count.text() == "12"


def test_live_cursor_scrolls_a_constant_width_window(qapp):
    """The live cursor advances and the window follows without ever widening."""
    roll = PianoRoll()
    roll.begin_live()
    for t in (5.0, 40.0, 120.0):
        roll.set_live_cursor(t)
        x0, x1 = _xrange(roll)
        assert x0 <= t <= x1, f"cursor {t} outside window [{x0}, {x1}]"
        assert x1 - x0 <= roll.LIVE_SPAN + 1e-9  # never wider than the window
    # at start the window is shorter (no negative time)…
    roll.begin_live()
    roll.set_live_cursor(1.0)
    assert _xrange(roll)[0] == 0.0
    # …then it slides at constant width
    assert _xrange(roll)[1] - _xrange(roll)[0] < roll.LIVE_SPAN
    roll.set_live_cursor(120.0)
    x0, x1 = _xrange(roll)
    assert (x0, x1) == pytest.approx((120.0 - roll.LIVE_SPAN * 0.88, 120.0 + roll.LIVE_SPAN * 0.12))


def test_live_preview_follows_the_held_note(qapp):
    roll = PianoRoll()
    roll.set_live_preview(LiveNote(start=1.0, end=1.2, midi=64))
    preview = roll._live_preview
    assert preview is not None
    assert preview.opts["x1"] == [1.2]
    roll.set_live_preview(LiveNote(start=1.0, end=1.9, midi=64))  # held
    assert preview.opts["x1"] == [1.9]
    roll.set_live_preview(LiveNote(start=1.0, end=1.9, midi=67))  # note change
    assert preview.opts["y0"] == pytest.approx([66.6])
    roll.set_live_preview(None)
    assert not preview.isVisible()


def test_notation_toggle_recolors_existing_bars(qapp):
    """Toggling notation mode must repaint the already-placed bars."""
    roll = PianoRoll()
    notes = _notes()
    notes[0].figure = "noire"
    roll.set_notes(notes)
    assert roll._bar_items[0].opts["brush"] == roll._color_for_conf(0.9)
    roll.set_notation_mode(True)
    assert roll._bar_items[0].opts["brush"] == figure_color("noire")
    assert len(roll._labels) == 1  # 0.5 s: wide enough for the N label


def test_in_place_figure_labeling_refreshes_the_bars(qapp):
    """Live, label_notes mutates the notes after the fact: the bar must follow."""
    roll = PianoRoll()
    roll.set_notation_mode(True)
    notes = _notes()
    roll.set_notes(notes)
    assert roll._labels == []
    notes[0].figure = "croche"  # live labeling, in place
    roll.set_notes(notes)
    assert len(roll._labels) == 1
    assert roll._bar_items[0].opts["brush"] == figure_color("croche")


def test_notes_that_shrink_fall_back_to_a_full_rebuild(qapp):
    """The incremental shortcut must not keep ghost bars."""
    roll = PianoRoll()
    notes = _notes()
    roll.set_notes(notes)
    roll.set_notes(notes[:2])
    assert len(roll._bar_items) == 2


def test_seek_moves_the_playhead_without_yanking_the_view(qapp):
    """`on_seek` asks follow=False: a double-click must not yank the view
    away from the place just clicked."""
    roll = PianoRoll()
    roll.set_notes(_notes())
    roll.getPlotItem().vb.setXRange(0.0, 2.0, padding=0)
    roll.set_playhead(30.0, follow=False)
    assert _xrange(roll) == pytest.approx((0.0, 2.0))
    roll.set_playhead(30.0)  # during playback, the view follows the cursor
    assert _xrange(roll)[0] > 0.0


# ------------------------------------------------------- graph selection


def _ev(
    roll,
    ev_type,
    t: float,
    midi: float,
    held: Qt.MouseButton,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> QMouseEvent:
    """A real QMouseEvent aimed at a (time, pitch) data point.

    The position is expressed in VIEWPORT coordinates — what `_hit`
    maps back through `mapToScene`, exactly like an event delivered by
    QGraphicsView's viewport routing. `held` is the button mask (the
    button that caused a press/release is always the left one), and
    `modifiers` carries Shift/Ctrl the way a real press would.
    """
    vb = roll.getPlotItem().vb
    scene = vb.mapViewToScene(QPointF(t, midi))
    vp_point = roll.viewport().mapFrom(roll, roll.mapFromScene(scene))
    point = QPointF(vp_point)
    cause = (
        Qt.MouseButton.NoButton if ev_type == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton
    )
    return QMouseEvent(ev_type, point, scene, point, cause, held, modifiers)


def _selectable_roll() -> PianoRoll:
    roll = PianoRoll(selectable=True)
    roll.resize(640, 480)
    roll.set_notes(_notes())
    return roll


def test_pressing_a_bar_selects_and_emits_that_note(qapp):
    roll = _selectable_roll()
    got: list = []
    roll.note_selection.connect(got.append)

    roll.mousePressEvent(
        _ev(roll, QEvent.Type.MouseButtonPress, 0.25, 60, Qt.MouseButton.LeftButton)
    )
    roll.mouseReleaseEvent(
        _ev(roll, QEvent.Type.MouseButtonRelease, 0.25, 60, Qt.MouseButton.NoButton)
    )

    assert got == [[0]]  # one press = one note, one emission
    assert roll._selection == [0]
    assert roll._bar_items[0].opts["brush"] == ACCENT
    assert roll._bar_items[1].opts["brush"] == roll._color_for_conf(0.9)  # untouched


def _click(roll, t: float, midi: float, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    """One full left click at a data point, with the given modifiers."""
    roll.mousePressEvent(
        _ev(
            roll,
            QEvent.Type.MouseButtonPress,
            t,
            midi,
            Qt.MouseButton.LeftButton,
            modifiers,
        )
    )
    roll.mouseReleaseEvent(
        _ev(roll, QEvent.Type.MouseButtonRelease, t, midi, Qt.MouseButton.NoButton)
    )


def test_shift_click_extends_a_contiguous_range_from_the_anchor(qapp):
    """The list gesture every editor shares: click one, Shift-click a range."""
    roll = _selectable_roll()
    got: list = []
    roll.note_selection.connect(got.append)

    _click(roll, 0.25, 60)
    _click(roll, 1.25, 61, Qt.KeyboardModifier.ShiftModifier)  # anchor 0 -> note 2
    assert got == [[0], [0, 1, 2]]
    assert roll._selection == [0, 1, 2]

    _click(roll, 0.75, 64, Qt.KeyboardModifier.ShiftModifier)  # back to note 1
    assert roll._selection == [0, 1]

    # a new plain click moves the anchor: the next Shift-click starts here
    _click(roll, 1.25, 61)
    _click(roll, 0.25, 60, Qt.KeyboardModifier.ShiftModifier)
    assert roll._selection == [0, 1, 2]


def test_ctrl_click_toggles_one_note_without_clearing_the_rest(qapp):
    """Ctrl/Cmd-click adds and removes signs of attention, one bar at a time."""
    roll = _selectable_roll()
    got: list = []
    roll.note_selection.connect(got.append)

    _click(roll, 0.25, 60)
    _click(roll, 1.25, 61, Qt.KeyboardModifier.ControlModifier)
    assert got == [[0], [0, 2]]  # separated notes, not the range between them
    _click(roll, 0.75, 64, Qt.KeyboardModifier.ControlModifier)
    assert roll._selection == [0, 1, 2]
    _click(roll, 0.25, 60, Qt.KeyboardModifier.ControlModifier)
    assert roll._selection == [1, 2]
    assert roll._bar_items[0].opts["brush"] == roll._color_for_conf(0.9)

    # Cmd-click is the macOS spelling of the same gesture
    _click(roll, 0.25, 60, Qt.KeyboardModifier.MetaModifier)
    assert roll._selection == [0, 1, 2]


def test_dragging_across_bars_does_not_select_and_an_empty_click_clears(qapp):
    """A drag on this roll is not a selection gesture; an empty click clears."""
    roll = _selectable_roll()
    got: list = []
    roll.note_selection.connect(got.append)

    roll.mousePressEvent(
        _ev(roll, QEvent.Type.MouseButtonPress, 0.25, 60, Qt.MouseButton.LeftButton)
    )
    roll.mouseMoveEvent(_ev(roll, QEvent.Type.MouseMove, 1.25, 61, Qt.MouseButton.LeftButton))
    roll.mouseReleaseEvent(
        _ev(roll, QEvent.Type.MouseButtonRelease, 1.25, 61, Qt.MouseButton.NoButton)
    )
    assert got == [[0]]  # the drag added nothing
    assert roll._selection == [0]

    # a click on empty space (no pan) clears the selection
    _click(roll, 1.25, 70)
    assert got == [[0], []]
    assert roll._selection == []
    assert roll._bar_items[0].opts["brush"] == roll._color_for_conf(0.9)


def test_a_modifier_click_selects_without_starting_a_note_move(qapp):
    """Shift/Ctrl select on the editable roll; a move needs a PLAIN drag."""
    roll = _selectable_roll()
    roll.set_editable(True)
    moves: list = []
    roll.note_dragged.connect(lambda *args: moves.append(args))

    _click(roll, 0.25, 60)
    roll.mousePressEvent(
        _ev(
            roll,
            QEvent.Type.MouseButtonPress,
            0.75,
            64,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ShiftModifier,
        )
    )
    assert roll._note_drag is None
    roll.mouseMoveEvent(_ev(roll, QEvent.Type.MouseMove, 0.85, 66, Qt.MouseButton.LeftButton))
    roll.mouseReleaseEvent(
        _ev(roll, QEvent.Type.MouseButtonRelease, 0.85, 66, Qt.MouseButton.NoButton)
    )
    assert moves == []
    assert roll._selection == [0, 1]

    roll.mousePressEvent(  # a plain press still arms the bar move
        _ev(roll, QEvent.Type.MouseButtonPress, 0.25, 60, Qt.MouseButton.LeftButton)
    )
    assert roll._note_drag is not None


def test_a_non_selectable_roll_ignores_the_whole_gesture(qapp):
    """The live page's roll has no table: `selectable` gates both the
    gesture AND a pushed selection — a stray call can never highlight it."""
    roll = PianoRoll()
    roll.resize(640, 480)
    roll.set_notes(_notes())
    got: list = []
    roll.note_selection.connect(got.append)

    roll.mousePressEvent(
        _ev(roll, QEvent.Type.MouseButtonPress, 0.25, 60, Qt.MouseButton.LeftButton)
    )
    roll.mouseReleaseEvent(
        _ev(roll, QEvent.Type.MouseButtonRelease, 0.25, 60, Qt.MouseButton.NoButton)
    )
    assert got == []

    roll.set_selection([0])
    assert roll._selection == []
    assert roll._bar_items[0].opts["brush"] == roll._color_for_conf(0.9)


def test_a_rebuild_keeps_the_selection_visible(qapp):
    """set_notes rebuilds every bar (figure labeling, mode toggle): the
    highlight must survive it, since the TABLE still holds the selection."""
    roll = _selectable_roll()
    roll.set_selection([1])
    assert roll._bar_items[1].opts["brush"] == ACCENT

    roll.set_notes(_notes(), force=True)  # a full rebuild
    assert roll._bar_items[1].opts["brush"] == ACCENT
    assert roll._bar_items[0].opts["brush"] == roll._color_for_conf(0.9)

    roll.set_selection([])  # cleared: the bar's own color comes back
    assert roll._bar_items[1].opts["brush"] == roll._color_for_conf(0.9)


def test_set_selectable_gates_both_directions_and_clears_the_pick(qapp):
    """The live roll's take lifecycle: frozen out during a capture, open
    when the take stops — and starting the next take drops whatever was
    picked, from the highlight and from the gesture alike."""
    roll = PianoRoll(selectable=True)
    roll.resize(640, 480)
    roll.set_notes(_notes())
    roll.set_selection([1])
    assert roll._bar_items[1].opts["brush"] == ACCENT

    roll.set_selectable(False)  # the next take begins
    assert roll._selection == []
    assert roll._bar_items[1].opts["brush"] == roll._color_for_conf(0.9)
    roll.set_selection([0])  # both directions refused
    assert roll._selection == []

    roll.set_selectable(True)  # the take stopped: the notes open up
    roll.set_selection([0])
    assert roll._selection == [0]
    assert roll._bar_items[0].opts["brush"] == ACCENT


def _key_press(roll: PianoRoll, key, modifiers=Qt.KeyboardModifier.NoModifier):
    """A real QKeyEvent delivered to the roll's own key handler."""
    return QKeyEvent(QEvent.Type.KeyPress, key, modifiers)


def test_the_arrow_keys_step_the_selection_note_by_note(qapp):
    """Right/Left walk a single-note cursor through the notes in time
    order (the ends hold); Shift+arrow extends the range by one note."""
    roll = _selectable_roll()
    got: list = []
    roll.note_selection.connect(got.append)

    # nothing picked yet: the first arrow enters at the near end
    roll.keyPressEvent(_key_press(roll, Qt.Key.Key_Right))
    assert roll._selection == [0]
    roll.keyPressEvent(_key_press(roll, Qt.Key.Key_Right))
    assert roll._selection == [1]
    roll.keyPressEvent(_key_press(roll, Qt.Key.Key_Right))
    assert roll._selection == [2]
    roll.keyPressEvent(_key_press(roll, Qt.Key.Key_Right))  # the last note holds
    assert roll._selection == [2]
    roll.keyPressEvent(_key_press(roll, Qt.Key.Key_Left))
    assert roll._selection == [1]
    # the highlight follows the cursor, exactly like a gesture's selection
    assert roll._bar_items[1].opts["brush"] == ACCENT
    assert roll._bar_items[0].opts["brush"] == roll._color_for_conf(0.9)

    # Shift extends the current range by one note at the moving edge
    roll.keyPressEvent(_key_press(roll, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier))
    assert roll._selection == [1, 2]
    assert got[-1] == [1, 2]

    # Shift+Left walks back past the start edge, collapsing onto the note
    roll.set_selection([1])
    roll.keyPressEvent(_key_press(roll, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier))
    assert roll._selection == [0, 1]


def test_the_arrow_keys_enter_from_the_far_end_when_going_left(qapp):
    roll = _selectable_roll()
    roll.set_selection([])
    roll.keyPressEvent(_key_press(roll, Qt.Key.Key_Left))
    assert roll._selection == [2]


def test_a_non_selectable_roll_ignores_the_arrow_keys(qapp):
    """A frozen-out roll (a running take) never paints a selection from
    the keyboard either — the same gate as the mouse gesture."""
    roll = PianoRoll()
    roll.resize(640, 480)
    roll.set_notes(_notes())
    got: list = []
    roll.note_selection.connect(got.append)

    roll.keyPressEvent(_key_press(roll, Qt.Key.Key_Right))

    assert got == []
    assert roll._selection == []


# ---------------------------------------------------------------- waveform


def _envelope(wf: WaveformView) -> tuple[float, float, int]:
    """(min, max, points) actually handed to the plot (lo/hi curves)."""
    lo_x, lo = wf._curves[0].getData()
    _, hi = wf._curves[1].getData()
    return float(np.min(lo)), float(np.max(hi)), len(lo_x)


def test_waveform_envelope_keeps_the_true_peaks(qapp):
    """The envelope keeps the true peaks: a naive one-point-in-N decimation
    makes isolated peaks disappear and a long take look flat."""
    wf = WaveformView()
    y = np.zeros(600_000, dtype=np.float32)  # x12 decimation at display
    y[5] = 1.0  # peak never reached by y[::12]
    y[500_000] = -0.8
    wf.set_signal(y, 1000)

    lo, hi, n_points = _envelope(wf)
    assert n_points <= WaveformView.MAX_POINTS
    assert (lo, hi) == pytest.approx((-0.8, 1.0))


def test_waveform_short_signal_is_plotted_as_is(qapp):
    wf = WaveformView()
    y = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    wf.set_signal(y, 100)
    lo, hi, n_points = _envelope(wf)
    assert n_points == 100
    assert (lo, hi) == pytest.approx((-0.5, 0.5))


def test_paint_does_not_crash_offscreen(qapp):
    from PyQt6.QtWidgets import QWidget

    host = QWidget()
    pv = PianoView(host)
    pv.resize(400, 80)
    host.show()
    pv.set_notes(_notes())
    pv.set_playhead(0.2)
    pix = pv.grab()  # forces a paintEvent pass offscreen
    assert not pix.isNull()
    assert pix.width() / pix.devicePixelRatioF() == pytest.approx(400)
