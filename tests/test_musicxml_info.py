"""Tests for the enriched MusicXML: harmonies, sol-fa lyrics, colors, credits."""

import xml.etree.ElementTree as ET

from v2m.core.chords import ChordCandidate
from v2m.core.keys import KeyCandidate, KeyResult
from v2m.core.musicxml import build_score_xml, _confidence_color
from v2m.core.notes import Note

BPM = 120.0
KEY_C = KeyResult(KeyCandidate("C", "major", 0.9), [])


def _parse(xml: str) -> ET.Element:
    body = xml.split("?>", 1)[1]
    return ET.fromstring(body)


def _notes_one_beat():
    return [Note(start=0.0, end=0.5, midi=60, confidence=0.95)]


def test_harmony_rendered_on_first_measure():
    prog = [(1, 2, ChordCandidate("C", "7", 0.8))]
    xml = build_score_xml(_notes_one_beat(), BPM, KEY_C, chord_progression=prog)
    root = _parse(xml)
    harms = root.findall("part/measure/harmony")
    assert len(harms) == 1
    assert harms[0].find("root/root-step").text == "C"
    assert harms[0].find("kind").text == "7"


def test_one_harmony_per_measure_span():
    # two measures of notes, one chord per measure
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9),
             Note(start=2.0, end=2.5, midi=67, confidence=0.9)]
    prog = [(1, 2, ChordCandidate("C", "7", 0.8)), (2, 3, ChordCandidate("G", "", 0.7))]
    root = _parse(build_score_xml(notes, BPM, KEY_C, chord_progression=prog))
    harms = root.findall("part/measure/harmony")
    assert len(harms) == 2
    assert [h.find("root/root-step").text for h in harms] == ["C", "G"]
    # sharp root supported (F# -> root-step F + alter 1)
    prog2 = [(1, 2, ChordCandidate("F#", "m", 0.7))]
    root2 = _parse(build_score_xml(_notes_one_beat(), BPM, KEY_C, chord_progression=prog2))
    h = root2.find("part/measure/harmony")
    assert h.find("root/root-step").text == "F"
    assert h.find("root/alter").text == "1"
    assert h.find("kind").text == "m"


def test_no_harmony_without_progression():
    root = _parse(build_score_xml(_notes_one_beat(), BPM, KEY_C))
    assert root.find("part/measure/harmony") is None


def test_solfa_lyrics_under_notes():
    root = _parse(build_score_xml(_notes_one_beat(), BPM, KEY_C, with_solfa_lyrics=True))
    lyric = root.find("part/measure/note/lyric/text")
    assert lyric is not None
    assert lyric.text == "Do4"
    # continuation notes of a tied long note carry no duplicate lyric
    long = [Note(start=0.0, end=1.0, midi=60, confidence=0.9)]
    root2 = _parse(build_score_xml(long, BPM, KEY_C, with_solfa_lyrics=True))
    lyrics = root2.findall("part/measure/note/lyric")
    assert len(lyrics) == 1


def test_note_color_reflects_confidence():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.4)]
    root = _parse(build_score_xml(notes, BPM, KEY_C, with_note_colors=True))
    assert root.find("part/measure/note").get("color") == _confidence_color(0.4)
    # very confident notes keep the default (no color attribute)
    sure = [Note(start=0.0, end=0.5, midi=60, confidence=0.99)]
    root2 = _parse(build_score_xml(sure, BPM, KEY_C, with_note_colors=True))
    assert root2.find("part/measure/note").get("color") is None


def test_credit_line_carries_all_detections():
    prog = [(1, 3, ChordCandidate("C", "", 0.8)), (3, 4, ChordCandidate("G", "", 0.7))]
    xml = build_score_xml(_notes_one_beat(), 117.5, KEY_C, chord_progression=prog)
    creator = _parse(xml).find("identification/creator")
    assert "117.5 BPM" in creator.text
    assert "C major" in creator.text
    assert "4/4" in creator.text
    assert "C / G" in creator.text


def test_new_options_off_disable_features():
    xml = build_score_xml(
        _notes_one_beat(), BPM, KEY_C,
        with_note_colors=False, with_solfa_lyrics=False,
    )
    root = _parse(xml)
    note = root.find("part/measure/note")
    assert note.get("color") is None
    assert note.find("lyric") is None
