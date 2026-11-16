"""MusicXML export tests: valid XML, measures, key signature, ties, triplets, rests."""

import xml.etree.ElementTree as ET

import pytest
from scorexml import assert_measures_close, score_root, text_of

from v2m.core.grid import BeatGrid, Meter
from v2m.core.keys import KeyCandidate, KeyResult
from v2m.core.musicxml import (
    _TUPLET_STEPS,
    _TYPE_BY_STEPS,
    DIVISIONS,
    _key_fifths,
    build_score_xml,
    save_musicxml,
)
from v2m.core.notation import layout_staff
from v2m.core.notes import Note

BPM = 120.0
GRID = BeatGrid(BPM)
KEY_C = KeyResult(KeyCandidate("C", "major", 0.9), [])
KEY_AM = KeyResult(KeyCandidate("A", "minor", 0.8), [])
KEY_EB = KeyResult(KeyCandidate("Eb", "major", 0.8), [])


def _parse(xml: str) -> ET.Element:
    # the DOCTYPE points to an external DTD: strip it before parsing
    body = xml.split("?>", 1)[1]
    return ET.fromstring(body)


def test_valid_xml_and_structure():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    assert root.tag == "score-partwise"
    assert text_of(root.find("part-list/score-part/part-name")) == "Voix"
    attrs = root.find("part/measure/attributes")
    assert attrs is not None
    assert text_of(attrs.find("divisions")) == "24"
    assert text_of(attrs.find("key/fifths")) == "0"
    assert text_of(attrs.find("time/beats")) == "4"
    assert text_of(attrs.find("clef/sign")) == "G"


def test_key_fifths_mapping():
    assert _key_fifths("C", "major") == 0
    assert _key_fifths("G", "major") == 1
    assert _key_fifths("Eb", "major") == -3
    assert _key_fifths("A", "minor") == 0  # C major's relative
    assert _key_fifths("E", "minor") == 1


def test_long_note_tied_across_beats():
    # half note (2 beats) -> two tied notes
    notes = [
        Note(start=0.0, end=1.0, midi=60, confidence=0.9),
        Note(start=1.0, end=1.5, midi=60, confidence=0.9),
    ]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    notes_xml = root.findall("part/measure/note")
    tied = [t for n in notes_xml if (t := n.find("notations/tied")) is not None]
    assert len(tied) == 2
    types = sorted(t.get("type") or "" for t in tied)
    assert types == ["start", "stop"]


def test_triplet_group_rendered():
    # quarter triplet x3 (2 beats): 3 notes of figure "triolet de noire"
    td = (2 / 3) * 0.5
    notes = [Note(start=i * td, end=(i + 1) * td, midi=60 + i, confidence=0.9) for i in range(3)]
    for n in notes:
        n.figure = "triolet de noire"
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    tm = root.findall("part/measure/note/time-modification")
    assert len(tm) == 3
    assert text_of(tm[0].find("actual-notes")) == "3"
    assert text_of(tm[0].find("normal-notes")) == "2"
    tuplets = root.findall("part/measure/note/notations/tuplet")
    sides = [t.get("type") for t in tuplets]
    assert sides == ["start", "stop"]  # the middle has no bracket
    # the block totals 2 beats = 48 divisions -> 16 per note, then the two
    # free beats close the measure
    durations = [int(text_of(n.find("duration"))) for n in root.findall("part/measure/note")]
    assert durations[:3] == [16, 16, 16]
    assert sum(durations) == 96  # one 4/4 measure at 24 divisions per beat


def test_rests_fill_silences():
    # note on beat 5 (2.0-2.5 s): 2 measures (5 beats -> 4+1)
    notes = [Note(start=2.0, end=2.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    measures = root.findall("part/measure")
    assert len(measures) == 2
    # measure 1 entirely silent -> ONE whole rest (not 4 quarter rests);
    # measure 2: the note on beat 0, then 3 silent beats -> a dotted half
    m1_rests = [n for n in measures[0].findall("note") if n.find("rest") is not None]
    assert [(text_of(n.find("type")), len(n.findall("dot"))) for n in m1_rests] == [("whole", 0)]
    m2_rests = [n for n in measures[1].findall("note") if n.find("rest") is not None]
    assert [(text_of(n.find("type")), len(n.findall("dot"))) for n in m2_rests] == [("half", 1)]


# ----------------------------------------------------------------- silences


def test_leading_rest_uses_a_whole_rest_per_empty_measure():
    # note on beat 5: measure 1 is entirely silent -> a whole rest
    notes = [Note(start=2.0, end=2.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    m1 = root.findall("part/measure")[0]
    types = [text_of(n.find("type")) for n in m1.findall("note") if n.find("rest") is not None]
    assert types == ["whole"]


def test_breath_rest_between_phrases_is_a_quarter_rest():
    # 0.5 s of breath (1 beat at 120 BPM) -> quarter rest, not a generic silence
    from v2m.core.rests import Rest

    notes = [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),
        Note(start=1.0, end=1.5, midi=62, confidence=0.9),
        Note(start=2.0, end=2.5, midi=64, confidence=0.9),
    ]
    rests = [Rest(start=0.5, end=1.0), Rest(start=1.5, end=2.0)]
    xml = build_score_xml(notes, GRID, KEY_C, rests=rests)
    root = _parse(xml)
    m1 = root.findall("part/measure")[0]
    types = [text_of(n.find("type")) for n in m1.findall("note") if n.find("rest") is not None]
    assert types == ["quarter", "quarter"]


def test_unlabeled_rest_still_engraves_its_duration():
    # a figure-less rest (unknown tempo) is engraved from its duration
    from v2m.core.rests import Rest

    notes = [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),
        Note(start=1.5, end=2.0, midi=62, confidence=0.9),
    ]
    rests = [Rest(start=0.5, end=1.5)]  # 2 beats, unknown figure
    xml = build_score_xml(notes, GRID, KEY_C, rests=rests)
    root = _parse(xml)
    rest_notes = [n for n in root.findall("part/measure/note") if n.find("rest") is not None]
    assert len(rest_notes) == 1  # merged, not fragmented
    assert text_of(rest_notes[0].find("duration")) == "48"  # 2 beats = 48 divisions
    assert text_of(rest_notes[0].find("type")) == "half"  # the real value, not generic


# ------------------------------------------- what the takes never exercise
BEAT, SLOT = 0.5, 0.75  # 120 BPM: a quarter beat, a dotted beat
GRID_44 = BeatGrid(BPM)
GRID_68 = BeatGrid(BPM, Meter(6, 8))


def _note(midi: int, start: float, end: float, figure: str | None = None) -> Note:
    return Note(start=start, end=end, midi=midi, confidence=0.9, figure=figure)


# Every case must close and only state durations it can fill; the
# contract is checked by `scorexml.assert_measures_close`.
SYNTHETIC = {
    "4/4 plein": lambda: (GRID_44, [_note(60 + i, i * BEAT, (i + 1) * BEAT) for i in range(4)]),
    "4/4 note tenue": lambda: (GRID_44, [_note(60, 0.0, 2 * BEAT)]),
    "4/4 seizièmes": lambda: (
        GRID_44,
        [_note(60 + i, i * BEAT / 4, (i + 1) * BEAT / 4) for i in range(4)],
    ),
    "4/4 triolet de croches": lambda: (
        GRID_44,
        [_note(60 + i, i * BEAT / 3, (i + 1) * BEAT / 3, "triolet de croche") for i in range(3)],
    ),
    "6/8 plein": lambda: (GRID_68, [_note(60 + i, i * SLOT, (i + 1) * SLOT) for i in range(2)]),
    "6/8 vide": lambda: (GRID_68, []),
    "6/8 partiel": lambda: (GRID_68, [_note(60, 0.0, SLOT)]),
    "6/8 trois mesures": lambda: (
        GRID_68,
        [_note(60 + i, i * SLOT, (i + 1) * SLOT) for i in range(6)],
    ),
}


@pytest.mark.parametrize("label", sorted(SYNTHETIC))
def test_a_synthetic_score_closes_with_valid_types_and_snapped_durations(label):
    """Each synthetic meter closes, and snapped notes stay at least a sixteenth."""
    grid, notes = SYNTHETIC[label]()
    root = score_root(build_score_xml(notes, grid, KEY_C))
    assert_measures_close(root, label)
    for note in root.findall("part/measure/note"):
        assert int(text_of(note.find("duration"))) >= 6, label
        assert text_of(note.find("type")) != "32nd", label

    if label == "4/4 plein":
        measures = root.findall("part/measure")
        assert len(measures) == 1
        assert root.findall("part/measure/note/rest") == []


def test_the_closure_property_refuses_a_type_that_lies_about_its_duration():
    """An unexercised net proves nothing: three quarters and a half close
    (96 divisions), so only the type/duration coherence can reject the
    latter — and it must."""
    forged = (
        '<?xml version="1.0"?><score-partwise><part><measure number="1">'
        "<attributes><divisions>24</divisions><time><beats>4</beats>"
        "<beat-type>4</beat-type></time></attributes>"
        + "<note><rest/><duration>24</duration><type>quarter</type></note>" * 3
        + "<note><rest/><duration>24</duration><type>half</type></note>"
        + "</measure></part></score-partwise>"
    )
    with pytest.raises(AssertionError, match="half pour 24 divisions"):
        assert_measures_close(score_root(forged), "cas forgé")  # forged case label (data)


def test_save_musicxml_writes_file(tmp_path):
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    path = str(tmp_path / "out.musicxml")
    save_musicxml(notes, GRID, KEY_C, path)
    text = open(path, encoding="utf-8").read()
    assert text.startswith("<?xml")
    root = _parse(text)
    assert root.tag == "score-partwise"


def test_metronome_mark():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, BeatGrid(117.5), KEY_C)
    root = _parse(xml)
    per_minute = root.find("part/measure/direction/direction-type/metronome/per-minute")
    assert per_minute is not None and per_minute.text == "117.5"


# ------------------------------------- the durations table IS the catalogue
@pytest.mark.parametrize("steps", sorted(_TYPE_BY_STEPS))
def test_the_export_types_a_duration_the_way_the_grid_writes_it(steps):
    """One arbiter of the written value: the divisions table is the Grid's
    catalogue (`DRAWABLE`) in divisions. Only the triplets (the catalogue
    knows none) and the 32nd, shorter than its sixteenth floor, are
    stated by the export."""
    beats = steps / DIVISIONS
    written = GRID.written(GRID.seconds(beats))
    if abs(written.beats - beats) < 1e-9:
        assert _TYPE_BY_STEPS[steps] == (written.type_, written.dots)
    else:
        assert steps in _TUPLET_STEPS or steps == 3


def test_a_dotted_sixteenth_is_written_on_the_staff_but_snapped_in_the_file():
    """A dotted sixteenth is 9 divisions, which the grid can write; but the
    snap stays at the sixteenth (6 divisions), so the file receives an
    eighth where the staff — which does not snap — writes a dotted
    sixteenth. And never a triplet: 4 divisions only come out of a
    triplet block."""
    notes = [
        Note(start=0.0, end=GRID.seconds(0.375), midi=60, confidence=0.9),
        Note(start=GRID.seconds(0.625), end=GRID.seconds(1.0), midi=62, confidence=0.9),
    ]
    staff = layout_staff(notes, grid=GRID, key="C major")
    assert [(n.written.type_, n.written.dots) for n in staff.notes] == [("16th", 1)] * 2

    xml = build_score_xml(notes, GRID, KEY_C)
    sounding = [n for n in _parse(xml).findall("part/measure/note") if n.find("rest") is None]
    assert [text_of(n.find("type")) for n in sounding] == ["eighth", "eighth"]
    assert [int(text_of(n.find("duration"))) for n in sounding] == [12, 12]
    assert "<time-modification>" not in xml  # hence no triplet
    assert 0.375 * DIVISIONS in _TYPE_BY_STEPS  # 9: the value is expressible


def test_an_isolated_short_note_keeps_its_duration_and_rests_the_beat():
    """An isolated short note keeps its written duration: what remains of
    its beat becomes a rest, instead of being absorbed by an invented whole
    beat."""
    notes = [Note(start=0.0, end=GRID.seconds(0.5), midi=60, confidence=0.9)]
    written = [
        (
            text_of(n.find("type")),
            int(text_of(n.find("duration"))),
            "silence" if n.find("rest") is not None else "note",
        )
        for n in _parse(build_score_xml(notes, GRID, KEY_C)).findall("part/measure/note")
    ]
    # sung eighth + eighth rest + dotted half for the remainder
    assert written == [("eighth", 12, "note"), ("eighth", 12, "silence"), ("half", 72, "silence")]


def test_a_silent_compound_measure_closes_on_a_dotted_half_rest():
    """A 6/8 measure is three quarters (72 divisions), not six beats: a
    single dotted half rest fills it."""
    xml = build_score_xml([], BeatGrid(BPM, Meter(6, 8)), KEY_C)
    rests = [
        (text_of(n.find("type")), len(n.findall("dot")), int(text_of(n.find("duration"))))
        for n in _parse(xml).findall("part/measure/note")
    ]
    assert rests == [("half", 1, 72)]
