"""Tests export MusicXML : XML valide, mesures, armure, ties, triolets."""

import xml.etree.ElementTree as ET

from v2m.core.keys import KeyCandidate, KeyResult
from v2m.core.musicxml import build_score_xml, _key_fifths, save_musicxml
from v2m.core.notes import Note

BPM = 120.0
KEY_C = KeyResult(KeyCandidate("C", "major", 0.9), [])
KEY_AM = KeyResult(KeyCandidate("A", "minor", 0.8), [])
KEY_EB = KeyResult(KeyCandidate("Eb", "major", 0.8), [])


def _parse(xml: str) -> ET.Element:
    # le DOCTYPE pointe vers une DTD externe : on le retire pour le parsing
    body = xml.split("?>", 1)[1]
    return ET.fromstring(body)


def test_valid_xml_and_structure():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, BPM, KEY_C)
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
    xml = build_score_xml(notes, BPM, KEY_C)
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
    xml = build_score_xml(notes, BPM, KEY_C)
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
    xml = build_score_xml(notes, BPM, KEY_C)
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
    xml = build_score_xml(notes, BPM, KEY_C)
    root = _parse(xml)
    measures = root.findall("part/measure")
    assert len(measures) == 2
    # 4 temps de silence en mesure 1 + 3 temps de silence en mesure 2
    # (le beat 0 de la mesure 2 porte la note, beats 1-3 sont silencieux)
    rest_notes = root.findall("part/measure/note/rest")
    assert len(rest_notes) == 7


def test_save_musicxml_writes_file(tmp_path):
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    path = str(tmp_path / "out.musicxml")
    save_musicxml(notes, BPM, KEY_C, path)
    text = open(path, encoding="utf-8").read()
    assert text.startswith("<?xml")
    root = _parse(text)
    assert root.tag == "score-partwise"


def test_metronome_mark():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, 117.5, KEY_C)
    root = _parse(xml)
    per_minute = root.find("part/measure/direction/direction-type/metronome/per-minute")
    assert per_minute.text == "117.5"
