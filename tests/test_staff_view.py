"""StaffView tests: offscreen rendering, systems, live scrolling."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core.liveseg import LiveNote
from v2m.core.grid import BeatGrid, Meter
from v2m.core.notes import Note
from v2m.gui.staff import HEADER_H, S, StaffView


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _note(midi, start, beats=1.0, bpm=120.0, figure=None):
    return Note(
        start=start,
        end=start + beats * 60.0 / bpm,
        midi=midi,
        confidence=0.9,
        figure=figure,
    )


def _ink(widget, x0=None, y0=None, x1=None, y1=None, threshold=90) -> int:
    """Bright pixels in a region — the only way to 'see' the drawing here."""
    from PyQt6.QtGui import QImage

    img = widget.grab().toImage().convertToFormat(QImage.Format.Format_Grayscale8)
    x0 = 0 if x0 is None else x0
    y0 = 0 if y0 is None else y0
    x1 = img.width() if x1 is None else min(x1, img.width())
    y1 = img.height() if y1 is None else min(y1, img.height())
    count = 0
    for y in range(int(y0), int(y1)):
        for x in range(int(x0), int(x1)):
            if img.pixelColor(x, y).red() > threshold:
                count += 1
    return count


def test_empty_staff_paints_without_notes(qapp):
    view = StaffView()
    view.resize(400, 160)
    view.clear()
    assert not view.grab().isNull()
    assert view.layout().notes == []
    assert view.visible_window() == (0.0, 0.0)


def test_natural_accidental_draws_its_own_glyph(qapp):
    """The natural is its OWN glyph, not a disguised flat.

    alter==0 fell into the else (flat): a key letter sung natural was
    engraved with a FLAT — wrong pitch on paper. The natural has two
    verticals at x - 0.17S and x + 0.19S; the flat has one stem at
    x - 0.06S, so the two glyphs differ by the ink near the second
    vertical (same measures as the ASCII inspector).
    """
    from v2m.core.grid import BeatGrid

    view = StaffView()
    view.resize(420, 160)
    # F5 (step 38, top line) natural in G major: explicit accidental
    notes = [_note(77, 0.0)]          # F5, alter 0 -> natural engraved
    view.set_score(notes, grid=BeatGrid(120.0), key="G major")
    lay = view.layout()
    assert lay.notes[0].accidental and lay.notes[0].alter == 0

    img = view.grab().toImage()
    # accidental x = notes_x + 1st-note x - 1.25S ; second vertical at +0.19S
    from v2m.gui.staff import S as S_PX
    from v2m.gui.staff import TOP_LINE_STEP

    y = HEADER_H + (TOP_LINE_STEP - lay.notes[0].step) * (S_PX / 2.0)
    x2 = view._notes_x(lay) - 1.25 * S_PX + 0.19 * S_PX
    ink2 = _ink(view, x0=x2 - 2, y0=y - 0.9 * S_PX, x1=x2 + 2, y1=y + 0.9 * S_PX)
    # the flat's stem (the old glyph) leaves this column EMPTY
    assert ink2 > 0, "the natural has no 2nd vertical (flat glyph?)"


def test_score_needs_no_meter_or_key(qapp):
    """A take whose meter stayed undetected still renders (4/4 assumed)."""
    view = StaffView()
    view.resize(420, 160)
    view.set_score([_note(60, 0.0), _note(64, 0.5)], grid=BeatGrid(120.0), key=None)
    lay = view.layout()
    assert lay.beats_per_measure == 4.0 and lay.meter_label == "4/4"
    assert lay.fifths == 0
    assert not view.grab().isNull()


def test_every_system_draws_its_notes(qapp):
    """Every system draws its notes INSIDE the widget."""
    notes = [_note(60 + (i % 7), i * 0.5) for i in range(16)]   # 4 measures
    view = StaffView()
    view.resize(560, 260)
    view.set_score(notes, grid=BeatGrid(120.0, Meter(4, 4)), key="C major")
    lay = view.layout()
    assert lay.measures == 4

    px_beat = S * 3.6
    per_system = max(1, int((560 - view._notes_x(lay) - 14) // (4 * px_beat)))
    assert per_system == 3                      # two systems for four measures
    system_h = 4 * S + 4.2 * S
    for system in range(2):
        top = HEADER_H + system * system_h
        ink = _ink(view, x0=view._notes_x(lay) + 5, y0=top - 14, y1=top + 4 * S + 8)
        assert ink > 40, f"system {system + 1} empty (no note drawn)"


def test_live_view_scrolls_to_the_newest_notes(qapp):
    notes = [_note(60 + (i % 5), i * 0.5) for i in range(40)]   # 40 beats
    view = StaffView(live=True)
    view.resize(420, 160)
    view.set_live(notes, grid=BeatGrid(120.0))

    start, end = view.visible_window()
    lay = view.layout()
    assert end >= lay.total_beats - 1e-6        # the cursor is in view
    assert start > 0.0                          # the take has scrolled
    last = lay.notes[-1].onset
    assert start <= last <= end

    # the newest notes are inside the widget, the oldest ones are off-screen
    px_beat = S * 3.6
    x_notes = view._notes_x(lay)
    x_last = x_notes + (last - start) * px_beat
    assert 0 < x_last <= view.width()
    assert x_notes + (0.0 - start) * px_beat < 0
    assert _ink(view, x0=int(view.width() * 0.6)) > 20


def test_live_view_shows_the_note_being_sung(qapp):
    notes = [_note(60, 0.0, figure="noire")]
    view = StaffView(live=True)
    view.resize(400, 160)
    view.set_live(notes, LiveNote(start=0.5, end=0.8, midi=67), grid=BeatGrid(120.0))
    lay = view.layout()
    assert len(lay.notes) == 2
    assert lay.notes[-1].preview is True
    assert lay.notes[-1].step == 32             # G4
    assert not view.grab().isNull()


def test_live_clear_removes_everything(qapp):
    view = StaffView(live=True)
    view.resize(400, 160)
    view.set_live([_note(60, 0.0)], LiveNote(start=0.5, end=0.8, midi=64))
    assert view.layout().notes
    view.clear()
    assert view.layout().notes == []
    assert not view.grab().isNull()


# ----------------------------------------------------------------- silences


def test_score_engraves_the_breaths_between_phrases(qapp):
    """The core's rests engrave between the phrases: phrase 2 starts after
    the breath, not glued to the first note."""
    from v2m.core.rests import Rest

    notes = [
        _note(60, 0.0, beats=1.0, figure="noire"),
        _note(64, 2.5, beats=1.0, figure="noire"),
    ]
    rests = [Rest(start=0.5, end=2.5, figure="pause")]
    view = StaffView()
    view.resize(420, 160)
    view.set_score(notes, grid=BeatGrid(120.0), rests=rests)
    lay = view.layout()
    assert len(lay.rests) == 1
    assert lay.rests[0].shape == "pause"
    assert lay.notes[1].onset == 5.0     # 1 quarter + 4 beats of rest
    assert _ink(view, y0=HEADER_H + int(5.4 * S), y1=HEADER_H + int(7.0 * S)) > 0
    assert not view.grab().isNull()


def test_rests_without_notes_engrave_nothing(qapp):
    """Guard: no notes -> no rests, and no crash."""
    from v2m.core.rests import Rest

    view = StaffView()
    view.resize(420, 160)
    view.set_score([], grid=BeatGrid(120.0), rests=[Rest(start=0.0, end=2.0, figure="pause")])
    assert view.layout().rests == []
    assert not view.grab().isNull()
