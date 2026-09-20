"""Tests export MusicXML : XML valide, mesures, armure, ties, triolets, silences."""

import xml.etree.ElementTree as ET

import pytest
from v2m.core.grid import BeatGrid, Meter
from v2m.core.keys import KeyCandidate, KeyResult
from v2m.core.musicxml import (
    DIVISIONS,
    _TUPLET_STEPS,
    _TYPE_BY_STEPS,
    build_score_xml,
    _key_fifths,
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
    # le DOCTYPE pointe vers une DTD externe : on le retire pour le parsing
    body = xml.split("?>", 1)[1]
    return ET.fromstring(body)


def test_valid_xml_and_structure():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    assert root.tag == "score-partwise"
    assert root.find("part-list/score-part/part-name").text == "Voix"
    attrs = root.find("part/measure/attributes")
    assert attrs.find("divisions").text == "12"
    assert attrs.find("key/fifths").text == "0"
    assert attrs.find("time/beats").text == "4"
    assert attrs.find("clef/sign").text == "G"


def test_measures_fill_total_duration():
    # 4 noires = 1 mesure 4/4
    notes = [Note(start=i * 0.5, end=(i + 1) * 0.5, midi=60 + i, confidence=0.9) for i in range(4)]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    measures = root.findall("part/measure")
    assert len(measures) == 1
    # 4 notes, pas de silences
    rest = root.findall("part/measure/note/rest")
    assert len(rest) == 0


def test_key_fifths_mapping():
    assert _key_fifths("C", "major") == 0
    assert _key_fifths("G", "major") == 1
    assert _key_fifths("Eb", "major") == -3
    assert _key_fifths("A", "minor") == 0      # relative de C major
    assert _key_fifths("Em", "minor") == 0 if False else _key_fifths("E", "minor") == 1


def test_long_note_tied_across_beats():
    # blanche (2 temps) -> deux notes liées
    notes = [Note(start=0.0, end=1.0, midi=60, confidence=0.9),
             Note(start=1.0, end=1.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    notes_xml = root.findall("part/measure/note")
    tied = [n for n in notes_xml if n.find("notations/tied") is not None]
    assert len(tied) == 2
    types = sorted(n.find("notations/tied").get("type") for n in tied)
    assert types == ["start", "stop"]


def test_triplet_group_rendered():
    # triolet de noire ×3 (2 temps) : 3 notes de figure "triolet de noire"
    td = (2 / 3) * 0.5
    notes = [Note(start=i * td, end=(i + 1) * td, midi=60 + i, confidence=0.9) for i in range(3)]
    for n in notes:
        n.figure = "triolet de noire"
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    tm = root.findall("part/measure/note/time-modification")
    assert len(tm) == 3
    assert tm[0].find("actual-notes").text == "3"
    assert tm[0].find("normal-notes").text == "2"
    tuplets = root.findall("part/measure/note/notations/tuplet")
    sides = [t.get("type") for t in tuplets]
    assert sides == ["start", "stop"]  # le milieu n'a pas de bracket
    # le bloc total = 2 temps = 24 divisions -> 8 par note, puis le temps
    # suivant (libre) est rendu en silence de 12 divisions
    durations = [int(n.find("duration").text) for n in root.findall("part/measure/note")]
    assert durations[:3] == [8, 8, 8]
    assert all(d % 8 == 0 or d == 12 for d in durations)


def test_rests_fill_silences():
    # note au 5e temps (2.0-2.5 s) : 2 mesures (5 temps -> 4+1)
    notes = [Note(start=2.0, end=2.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    measures = root.findall("part/measure")
    assert len(measures) == 2
    # mesure 1 entière silencieuse -> UNE pause entière (pas 4 soupirs) ;
    # mesure 2 : la note au beat 0, puis 3 temps silencieux -> une blanche pointée
    m1_rests = [n for n in measures[0].findall("note") if n.find("rest") is not None]
    assert [(n.find("type").text, len(n.findall("dot"))) for n in m1_rests] == [("whole", 0)]
    m2_rests = [n for n in measures[1].findall("note") if n.find("rest") is not None]
    assert [(n.find("type").text, len(n.findall("dot"))) for n in m2_rests] == [("half", 1)]


# ----------------------------------------------------------------- silences


def test_leading_rest_uses_a_whole_rest_per_empty_measure():
    # note au 5e temps : la mesure 1 entière est silencieuse -> une pause
    notes = [Note(start=2.0, end=2.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    m1 = root.findall("part/measure")[0]
    types = [n.find("type").text for n in m1.findall("note") if n.find("rest") is not None]
    assert types == ["whole"]


def test_breath_rest_between_phrases_is_a_quarter_rest():
    # 0,5 s de respiration (1 temps à 120 BPM) -> soupir, pas silence générique
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
    types = [n.find("type").text for n in m1.findall("note") if n.find("rest") is not None]
    assert types == ["quarter", "quarter"]


def test_unlabeled_rest_still_engraves_its_duration():
    # un silence sans figure (tempo inconnu) se grave d'après sa durée
    from v2m.core.rests import Rest

    notes = [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),
        Note(start=1.5, end=2.0, midi=62, confidence=0.9),
    ]
    rests = [Rest(start=0.5, end=1.5)]          # 2 temps, figure inconnue
    xml = build_score_xml(notes, GRID, KEY_C, rests=rests)
    root = _parse(xml)
    rest_notes = [n for n in root.findall("part/measure/note") if n.find("rest") is not None]
    assert len(rest_notes) == 1                           # fusionnée, pas fragmentée
    assert rest_notes[0].find("duration").text == "24"    # 2 temps = 24 divisions
    assert rest_notes[0].find("type").text == "half"      # vraie valeur, pas générique


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
    assert per_minute.text == "117.5"


# ------------------------------------- la table des durées EST le catalogue
@pytest.mark.parametrize("steps", sorted(_TYPE_BY_STEPS))
def test_the_export_types_a_duration_the_way_the_grid_writes_it(steps):
    """Un seul arbitre de la valeur écrite : la table des divisions est le
    catalogue de la Grille (`DRAWABLE`) en divisions. Seuls les triolets (le
    catalogue n'en connaît aucun) et le trente-deuxième, plus court que son
    plancher au seizième, sont énoncés par l'export."""
    beats = steps / DIVISIONS
    type_, written_beats, dots = GRID.written(GRID.seconds(beats))
    if abs(written_beats - beats) < 1e-9:
        assert _TYPE_BY_STEPS[steps] == (type_, dots)
    else:
        assert steps in _TUPLET_STEPS or steps == 1


def test_a_dotted_sixteenth_is_written_on_the_staff_but_cannot_be_exported():
    """Le seizième pointé (0,375 temps) : la portée l'écrit, l'export ne le
    peut pas — 0,375 × 12 = 4,5 divisions, et la valeur entière du <duration>
    est ce qui remplit la mesure. L'export ramène donc la note sur sa grille
    (une croche) et surtout NE FAIT PAS un triolet de croche, qui est ce que 4
    divisions signifient."""
    notes = [
        Note(start=0.0, end=GRID.seconds(0.375), midi=60, confidence=0.9),
        Note(start=GRID.seconds(0.625), end=GRID.seconds(1.0), midi=62, confidence=0.9),
    ]
    staff = layout_staff(notes, grid=GRID, key="C major")
    assert [(n.written.type_, n.written.dots) for n in staff.notes] == [("16th", 1)] * 2

    xml = build_score_xml(notes, GRID, KEY_C)
    sounding = [n for n in _parse(xml).findall("part/measure/note") if n.find("rest") is None]
    assert [n.find("type").text for n in sounding] == ["eighth", "eighth"]   # 6 divisions
    assert "<time-modification>" not in xml       # donc aucun triolet
    assert 0.375 * DIVISIONS not in _TYPE_BY_STEPS


def test_an_isolated_short_note_keeps_its_duration_and_rests_the_beat():
    """Une note courte isolée garde sa durée écrite : ce qui reste de son temps
    devient un silence, au lieu d'être absorbé par un temps entier inventé."""
    notes = [Note(start=0.0, end=GRID.seconds(0.5), midi=60, confidence=0.9)]
    written = [
        (n.find("type").text, int(n.find("duration").text),
         "silence" if n.find("rest") is not None else "note")
        for n in _parse(build_score_xml(notes, GRID, KEY_C)).findall("part/measure/note")
    ]
    # croche chantée + croche de silence + dotted half rest pour le reste
    assert written == [("eighth", 6, "note"), ("eighth", 6, "silence"), ("half", 36, "silence")]


def test_a_silent_compound_measure_fills_with_dotted_values():
    """Conséquence de la dérivation : la ronde pointée (6 temps = 72 divisions)
    entre dans la table, donc une mesure 6/8 vide se décompose en ronde pointée
    + blanche pointée (72 + 36), au lieu de deux rondes et une noire."""
    xml = build_score_xml([], BeatGrid(BPM, Meter(6, 8)), KEY_C)
    rests = [
        (n.find("type").text, len(n.findall("dot")), int(n.find("duration").text))
        for n in _parse(xml).findall("part/measure/note")
    ]
    assert rests == [("whole", 1, 72), ("half", 1, 36)]
