"""StaffView tests: offscreen rendering, systems, live scrolling."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core.grid import BeatGrid, Meter
from v2m.core.liveseg import LiveNote
from v2m.core.notes import Note
from v2m.gui.staff import S, StaffView


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
    notes = [_note(77, 0.0)]  # F5, alter 0 -> natural engraved
    view.set_score(notes, grid=BeatGrid(120.0), key="G major")
    lay = view.layout()
    assert lay.notes[0].accidental and lay.notes[0].alter == 0

    # accidental x = notes_x + 1st-note x - 1.25S ; second vertical at +0.19S
    from v2m.gui.staff import TOP_LINE_STEP
    from v2m.gui.staff import S as S_PX

    # the systems are vertically centred: ask the view where it draws the
    # first one instead of assuming the header's own height
    y = view._score_geometry()[3] + (TOP_LINE_STEP - lay.notes[0].step) * (S_PX / 2.0)
    x2 = view._notes_x(lay) - 1.25 * S_PX + 0.19 * S_PX
    scale = view.devicePixelRatioF()
    ink2 = _ink(
        view,
        x0=(x2 - 2) * scale,
        y0=(y - 0.9 * S_PX) * scale,
        x1=(x2 + 2) * scale,
        y1=(y + 0.9 * S_PX) * scale,
    )
    # the flat's stem (the old glyph) leaves this column EMPTY
    assert ink2 > 0, "the natural has no 2nd vertical (flat glyph?)"


def test_score_needs_no_meter_or_key(qapp):
    """A take with undetected meter or key renders without inventing a key."""
    from v2m.core.i18n import tr

    view = StaffView()
    view.resize(420, 160)
    view.set_score([_note(60, 0.0), _note(64, 0.5)], grid=BeatGrid(120.0), key=None)
    lay = view.layout()
    assert lay.beats_per_measure == 4.0 and lay.meter_label == "4/4"
    assert lay.fifths == 0
    assert lay.key_tonic is None and lay.key_mode is None

    class HeaderPainter:
        def setFont(self, _font):
            pass

        def setPen(self, _pen):
            pass

        def drawText(self, _point, text):
            self.text = text

    painter = HeaderPainter()
    view._paint_header(painter, lay)
    assert "None" not in painter.text
    assert "major" not in painter.text
    assert tr("key bpm unit") in painter.text
    assert "4/4" in painter.text
    assert not view.grab().isNull()


def test_score_shows_a_detected_key_in_the_header(qapp):
    """A detected key remains visible when its label is given to the staff."""
    from v2m.core.i18n import get_language, set_language

    previous = get_language()
    try:
        set_language("en")
        view = StaffView()
        view.resize(420, 160)
        view.set_score([_note(60, 0.0)], grid=BeatGrid(120.0), key="G major")
        layout = view.layout()
        assert layout.key_tonic == "G"
        assert layout.key_mode == "major"

        class HeaderPainter:
            def setFont(self, _font):
                pass

            def setPen(self, _pen):
                pass

            def drawText(self, _point, text):
                self.text = text

        painter = HeaderPainter()
        view._paint_header(painter, layout)
        assert painter.text.endswith("G major")
        assert not view.grab().isNull()
    finally:
        set_language(previous)


def test_every_system_draws_its_notes(qapp):
    """Every system draws its notes INSIDE the widget."""
    notes = [_note(60 + (i % 7), i * 0.5) for i in range(16)]  # 4 measures
    view = StaffView()
    view.resize(560, 260)
    view.set_score(notes, grid=BeatGrid(120.0, Meter(4, 4)), key="C major")
    lay = view.layout()
    assert lay.measures == 4

    px_beat = S * 3.6
    per_system = max(1, int((560 - view._notes_x(lay) - 14) // (4 * px_beat)))
    assert per_system == 3  # two systems for four measures
    system_h = 4 * S + 4.2 * S
    first_top = view._score_geometry()[3]
    for system in range(2):
        top = first_top + system * system_h
        ink = _ink(view, x0=view._notes_x(lay) + 5, y0=top - 14, y1=top + 4 * S + 8)
        assert ink > 40, f"system {system + 1} empty (no note drawn)"


def test_live_view_scrolls_to_the_newest_notes(qapp):
    notes = [_note(60 + (i % 5), i * 0.5) for i in range(40)]  # 40 beats
    view = StaffView(live=True)
    view.resize(420, 160)
    view.set_live(notes, grid=BeatGrid(120.0))

    start, end = view.visible_window()
    lay = view.layout()
    assert end >= lay.total_beats - 1e-6  # the cursor is in view
    assert start > 0.0  # the take has scrolled
    last = lay.notes[-1].onset
    assert start <= last <= end

    # the newest notes are inside the widget, the oldest ones are off-screen
    px_beat = S * 3.6
    x_notes = view._notes_x(lay)
    x_last = x_notes + (last - start) * px_beat
    assert 0 < x_last <= view.width()
    assert x_notes + (0.0 - start) * px_beat < 0
    assert _ink(view, x0=int(view.width() * 0.6)) > 20


def test_a_heard_breath_rest_carries_its_annotation():
    """A rest marked `breath` arrives on the layout with the flag: the
    staff paints the "B" above it; an unmarked rest carries none."""
    from v2m.core.notation import layout_staff
    from v2m.core.rests import Rest

    grid = BeatGrid(120.0)
    notes = [
        Note(0.0, 0.5, 60, 0.9, figure="noire"),
        Note(1.0, 1.5, 62, 0.9, figure="noire"),
    ]
    marked = [Rest(start=0.5, end=1.0, figure="soupir", breath=True)]
    lay = layout_staff(notes, grid=grid, rests=marked)
    assert lay.rests[0].breath is True
    plain = [Rest(start=0.5, end=1.0, figure="soupir")]
    lay = layout_staff(notes, grid=grid, rests=plain)
    assert lay.rests[0].breath is False


def test_live_view_places_notes_at_their_real_onsets(qapp):
    """Live notes spread by the detected tempo: short notes with gaps used
    to chain by written duration and collapse into ONE measure."""
    notes = [LiveNote(i * 1.0, i * 1.0 + 0.1, 60 + i) for i in range(4)]
    view = StaffView(live=True)
    view.resize(500, 160)
    view.set_live(notes, grid=BeatGrid(120.0))  # 1.0 s = 2 quarter beats

    lay = view.layout()
    assert lay.measures == 2
    assert [round(n.onset, 2) for n in lay.notes] == [0.0, 2.0, 4.0, 6.0]
    assert [n.measure for n in lay.notes] == [1, 1, 2, 2]


def test_live_view_shows_the_note_being_sung(qapp):
    notes = [_note(60, 0.0, figure="noire")]
    view = StaffView(live=True)
    view.resize(400, 160)
    view.set_live(notes, LiveNote(start=0.5, end=0.8, midi=67), grid=BeatGrid(120.0))
    lay = view.layout()
    assert len(lay.notes) == 2
    assert lay.notes[-1].preview is True
    assert lay.notes[-1].step == 32  # G4
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
    assert lay.notes[1].onset == 5.0  # 1 quarter + 4 beats of rest
    origin = view._score_geometry()[3]
    assert _ink(view, y0=origin + int(5.4 * S), y1=origin + int(7.0 * S)) > 0
    assert not view.grab().isNull()


def test_rests_without_notes_engrave_nothing(qapp):
    """Guard: no notes -> no rests, and no crash."""
    from v2m.core.rests import Rest

    view = StaffView()
    view.resize(420, 160)
    view.set_score([], grid=BeatGrid(120.0), rests=[Rest(start=0.0, end=2.0, figure="pause")])
    assert view.layout().rests == []
    assert not view.grab().isNull()
