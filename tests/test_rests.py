"""Rests tests: breaths between phrases and rest figures.

`rests.py` had no direct test (it was covered indirectly by the vocal
accuracy, the Staff and the MusicXML). This file pins its contract: what
a gap becomes, and under which name.

Guards already carried by the code, tested here because they are the
module's contract: unordered or overlapping notes (the cursor never goes
back), a gap under the seam threshold, a take's end that is not a rest,
an absent tempo (no guessed figure).
"""

from v2m.core.grid import BeatGrid
from v2m.core.notation import layout_staff
from v2m.core.notes import Note
from v2m.core.rests import MIN_REST_S, Rest, classify_rest, detect_rests

GRID = BeatGrid(120.0)


def _note(midi, start, dur_s, figure=None):
    return Note(start=start, end=start + dur_s, midi=midi, confidence=0.9,
                figure=figure)


# ------------------------------------------------------------------- figures
def test_classify_rest_speaks_rest_not_note():
    """The classifier is the notes' one, the lexicon the rests': a 1-beat
    gap is a sung "noire", a written "soupir"."""
    assert classify_rest(1.0) == "soupir"
    assert classify_rest(2.0) == "demi-pause"
    assert classify_rest(4.0) == "pause"
    assert classify_rest(0.5) == "demi-soupir"


def test_classify_rest_never_labels_a_triplet():
    """Measured on chant.wav: a 1.17-beat breath came out as "triolet de
    blanche" (13.5% error) ahead of the quarter (16.9%), engraving an
    absurd silence. A rest is never under a bracket."""
    for beats in (0.33, 0.67, 1.17, 1.33, 2.67):
        fig = classify_rest(beats)
        assert fig is None or "triolet" not in fig, (beats, fig)


def test_classify_rest_rejects_a_non_positive_duration():
    assert classify_rest(0.0) is None
    assert classify_rest(-2.0) is None


def test_rest_figures_are_translatable():
    """Figures stored in French (the core's convention), translated for
    display: "demi-pause" must not come out as "demi-whole rest"."""
    from v2m.core.i18n import set_language, tr_figure

    set_language("en")
    try:
        assert tr_figure("demi-pause") == "half rest"
        assert tr_figure("quart de soupir") == "sixteenth rest"
    finally:
        set_language("fr")


# ----------------------------------------------------------------- detection
def test_lead_in_and_breaths_both_become_rests():
    notes = [_note(60, 0.41, 0.2), _note(62, 1.0, 0.2)]
    rests = detect_rests(notes, GRID)
    assert [(round(r.start, 2), round(r.end, 2)) for r in rests] == [
        (0.0, 0.41), (0.61, 1.0)
    ]


def test_the_tail_of_the_take_is_not_a_rest():
    """A take stops where the engineer cut, not where the song ends."""
    rests = detect_rests([_note(60, 0.0, 0.5), _note(62, 1.0, 0.5)], GRID)
    assert max(r.end for r in rests) <= 1.0


def test_lead_in_can_be_excluded():
    notes = [_note(60, 0.41, 0.2), _note(62, 1.0, 0.2)]
    rests = detect_rests(notes, GRID, include_lead_in=False)
    assert [(round(r.start, 2), round(r.end, 2)) for r in rests] == [(0.61, 1.0)]


def test_a_seam_is_not_a_breath():
    """Under the threshold, the gap is a segmentation seam (the
    interpolation filler works in that zone)."""
    notes = [_note(60, 0.0, 0.5), _note(62, 0.53, 0.5)]
    assert detect_rests(notes, GRID, include_lead_in=False) == []
    assert MIN_REST_S > 0.03


def test_unordered_and_overlapping_notes_never_go_backwards():
    notes = [
        _note(62, 1.0, 0.5),
        _note(60, 0.0, 0.8),   # overlaps the next one
        _note(64, 0.4, 0.4),
        _note(65, 2.0, 0.4),
    ]
    rests = detect_rests(notes, GRID)
    assert all(r.duration > 0 for r in rests)
    assert [r.start for r in rests] == sorted(r.start for r in rests)


def test_figure_follows_the_grid_tempo():
    notes = [_note(60, 0.0, 0.2), _note(62, 1.2, 0.2)]   # 1 s of breath
    at_120 = detect_rests(notes, GRID, include_lead_in=False)[0]
    at_60 = detect_rests(notes, BeatGrid(60.0), include_lead_in=False)[0]
    assert at_120.figure == "demi-pause"   # 2 beats at 120
    assert at_60.figure == "soupir"        # 1 beat at 60


def test_without_a_grid_the_rests_stay_unlabeled():
    """Without a Grid no figure is guessed: the Staff engraves from the
    real duration, which is honest."""
    notes = [_note(60, 0.0, 0.2), _note(62, 1.2, 0.2)]
    rests = detect_rests(notes)
    assert rests and all(r.figure is None for r in rests)


# -------------------------------------------------------- Staff and MusicXML
def test_layout_engraves_a_labeled_breath_in_the_gap():
    notes = [_note(60, 0.0, 0.5, figure="noire"), _note(62, 1.0, 0.5, figure="noire")]
    rests = [Rest(start=0.5, end=1.0, figure="soupir")]
    lay = layout_staff(notes, grid=GRID, rests=rests)
    assert len(lay.rests) == 1
    assert (lay.rests[0].shape, lay.rests[0].onset) == ("soupir", 1.0)


def test_a_missing_tempo_never_raises_mid_engraving():
    """Guard: layout and export both divide by the tempo, an absent tempo
    must not make a finished analysis raise."""
    from v2m.core.musicxml import build_score_xml

    notes = [_note(60, 0.0, 0.5, figure="noire"), _note(62, 1.0, 0.5, figure="noire")]
    for bpm in (None, 0):
        grid = BeatGrid(bpm)
        lay = layout_staff(notes, grid=grid, rests=[Rest(start=0.5, end=1.0)])
        assert len(lay.notes) == 2 and lay.total_beats > 0, bpm
        assert "<rest/>" in build_score_xml(notes, grid, None), bpm


def test_musicxml_writes_the_written_duration_not_a_wider_label():
    """A <type> contradicting the <duration> makes the file flimsy: the
    duration is what fills the measure, so a half rest detected inside a
    written beat stays a one-beat figure."""
    from v2m.core.musicxml import build_score_xml

    notes = [_note(60, 0.0, 0.5), _note(62, 1.0, 0.5)]
    rests = [Rest(start=0.5, end=1.0, figure="demi-pause")]
    xml = build_score_xml(notes, BeatGrid(120.0), None, snap=False, rests=rests)
    assert "<rest/>" in xml and "<type>quarter</type>" in xml
    assert "<type>half</type>" not in xml


def test_musicxml_fuses_a_silent_run_into_a_real_figure():
    """A two-beat breath is written as ONE half rest, not two filler
    quarters."""
    from v2m.core.musicxml import build_score_xml

    notes = [_note(60, 0.0, 0.5), _note(62, 1.5, 0.5)]
    rests = detect_rests(notes, GRID, include_lead_in=False)
    assert rests and rests[0].figure == "demi-pause"
    xml = build_score_xml(notes, GRID, None, snap=False, rests=rests)
    assert "<type>half</type>" in xml
