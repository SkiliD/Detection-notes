"""Tests for the PianoView keyboard, the PianoRoll live follow and the waveform."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from v2m.core.liveseg import LiveNote
from v2m.core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI
from v2m.core.notes import Note
from v2m.gui.widgets import PianoRoll, PianoView, PitchAxis, WaveformView, figure_color


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _notes():
    return [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),   # C4 (white)
        Note(start=0.5, end=1.0, midi=64, confidence=0.9),   # E4 (white)
        Note(start=1.0, end=1.5, midi=61, confidence=0.9),   # C#4 (black)
    ]


def test_set_notes_indexes_spans(qapp):
    pv = PianoView()
    pv.set_notes(_notes())
    assert set(pv._spans) == {60, 64, 61}
    assert pv._spans[60] == [(0.0, 0.5)]


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
    assert len(full._whites) == 52         # 88 keys, 36 of them black
    assert full._white_midi(0) == 21 and full._white_midi(51) == 108
    assert not full._has_black_after(108)  # nothing after C8
    # the analysis view covers 48..84: 22 whites, C6 included (the 84 key
    # was invisible — a note at the top edge could never light)
    assert len(PianoView()._whites) == 22


def test_every_black_key_of_the_piano_is_drawn(qapp):
    """Regression: the keyboard drew only a C# and an F# per octave (the
    rule tested the wrong pitch classes), so D#, G# and A# did not exist
    — no other black note could ever light."""
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


def test_the_pitch_axis_names_notes_instead_of_numbers(qapp):
    """The roll's axis reads like a keyboard, not like MIDI numbers."""
    axis = PitchAxis(orientation="left")
    assert axis.tickStrings([21, 60, 61, 108], 1.0, 1.0) == ["A0", "C4", "C#4", "C8"]


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


def test_outline_note_marks_the_live_final_diff(qapp):
    """The live-vs-final mark: one outline per moved note, cleared by
    `begin_live` (a new take restarts from a clean roll)."""
    roll = PianoRoll()
    roll.set_notes(_notes())
    assert roll._diff_items == []
    roll.outline_note(0.6, 1.1, 64, (77, 163, 216))
    assert len(roll._diff_items) == 1
    # the REAL color is the palette's hex string: `tuple(color)` once
    # shattered it into 7 characters and the mark silently never drew
    from v2m.gui.widgets import ACCENT

    roll.outline_note(0.7, 1.0, 62, ACCENT)
    assert len(roll._diff_items) == 2
    # a note rebuild (set_notes without the fast path) also clears the marks
    roll.set_notes(_notes(), force=True)
    assert roll._diff_items == []
    # a new take too
    roll.outline_note(0.0, 0.5, 60, (77, 163, 216))
    assert len(roll._diff_items) == 1
    roll.begin_live()
    assert roll._diff_items == []


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


# ---------------------------------------------------------------- waveform


def _envelope(wf: WaveformView) -> tuple[float, float, int]:
    """(min, max, points) actually handed to the plot (lo/hi curves)."""
    lo_x, lo = wf._curves[0].getData()
    _, hi = wf._curves[1].getData()
    return float(np.min(lo)), float(np.max(hi)), len(lo_x)


def test_waveform_envelope_keeps_the_true_peaks(qapp):
    """Regression: the preview decimated one point in N, so isolated peaks
    disappeared and a long take looked flat."""
    wf = WaveformView()
    y = np.zeros(600_000, dtype=np.float32)  # x12 decimation at display
    y[5] = 1.0                               # peak never reached by y[::12]
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
    assert pix.width() == 400
