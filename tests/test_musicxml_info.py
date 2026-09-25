"""Tests for the enriched MusicXML: harmonies, sol-fa lyrics, colors, credits."""

import xml.etree.ElementTree as ET

from scorexml import text_of
from v2m.core.chords import ChordCandidate
from v2m.core.grid import BeatGrid
from v2m.core.keys import KeyCandidate, KeyResult
from v2m.core.musicxml import build_score_xml, _confidence_color
from v2m.core.notes import Note

BPM = 120.0
GRID = BeatGrid(BPM)
KEY_C = KeyResult(KeyCandidate("C", "major", 0.9), [])


def _parse(xml: str) -> ET.Element:
    body = xml.split("?>", 1)[1]
    return ET.fromstring(body)


def _notes_one_beat():
    return [Note(start=0.0, end=0.5, midi=60, confidence=0.95)]


def test_harmony_rendered_on_first_measure():
    prog = [(1, 2, ChordCandidate("C", "7", 0.8))]
    xml = build_score_xml(_notes_one_beat(), GRID, KEY_C, chord_progression=prog)
    root = _parse(xml)
    harms = root.findall("part/measure/harmony")
    assert len(harms) == 1
    assert text_of(harms[0].find("root/root-step")) == "C"
    assert text_of(harms[0].find("kind")) == "7"


def test_one_harmony_per_measure_span():
    # two measures of notes, one chord per measure
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9),
             Note(start=2.0, end=2.5, midi=67, confidence=0.9)]
    prog = [(1, 2, ChordCandidate("C", "7", 0.8)), (2, 3, ChordCandidate("G", "", 0.7))]
    root = _parse(build_score_xml(notes, GRID, KEY_C, chord_progression=prog))
    harms = root.findall("part/measure/harmony")
    assert len(harms) == 2
    assert [text_of(h.find("root/root-step")) for h in harms] == ["C", "G"]
    # sharp root supported (F# -> root-step F + alter 1)
    prog2 = [(1, 2, ChordCandidate("F#", "m", 0.7))]
    root2 = _parse(build_score_xml(_notes_one_beat(), GRID, KEY_C, chord_progression=prog2))
    h = root2.find("part/measure/harmony")
    assert h is not None
    assert text_of(h.find("root/root-step")) == "F"
    assert text_of(h.find("root/alter")) == "1"
    assert text_of(h.find("kind")) == "m"


def test_no_harmony_without_progression():
    root = _parse(build_score_xml(_notes_one_beat(), GRID, KEY_C))
    assert root.find("part/measure/harmony") is None


def test_translated_chord_words_next_to_the_harmony():
    """Le chiffrage porte son NOM LISIBLE au-dessus de la portée : la
    traduction whole-label de i18n (« La mineur »), la même que la GUI et
    le CLI. Le <harmony> garde le symbole canonique (Am) — la structure
    harmonique du fichier reste anglaise ; les <words> sont un affichage.
    Sans progression : ni harmony ni words."""
    prog = [(1, 2, ChordCandidate("A", "m", 0.8))]
    root = _parse(build_score_xml(_notes_one_beat(), GRID, KEY_C, chord_progression=prog))
    words = root.findall("part/measure/direction/direction-type/words")
    assert len(words) == 1
    assert words[0].text == "La mineur"  # tr_chord_label("Am"), fr default
    # the canonical symbol travels in the harmony element beside it
    assert text_of(root.find("part/measure/harmony/kind")) == "m"
    root2 = _parse(build_score_xml(_notes_one_beat(), GRID, KEY_C))
    assert root2.find("part/measure/direction/direction-type/words") is None


def test_solfa_lyrics_under_notes():
    root = _parse(build_score_xml(_notes_one_beat(), GRID, KEY_C, with_solfa_lyrics=True))
    lyric = root.find("part/measure/note/lyric/text")
    assert lyric is not None
    assert lyric.text == "Do4"
    # continuation notes of a tied long note carry no duplicate lyric
    long = [Note(start=0.0, end=1.0, midi=60, confidence=0.9)]
    root2 = _parse(build_score_xml(long, GRID, KEY_C, with_solfa_lyrics=True))
    lyrics = root2.findall("part/measure/note/lyric")
    assert len(lyrics) == 1


def test_note_color_reflects_confidence():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.4)]
    root = _parse(build_score_xml(notes, GRID, KEY_C, with_note_colors=True))
    note = root.find("part/measure/note")
    assert note is not None
    assert note.get("color") == _confidence_color(0.4)
    # very confident notes keep the default (no color attribute)
    sure = [Note(start=0.0, end=0.5, midi=60, confidence=0.99)]
    root2 = _parse(build_score_xml(sure, GRID, KEY_C, with_note_colors=True))
    plain = root2.find("part/measure/note")
    assert plain is not None
    assert plain.get("color") is None


def test_credit_line_carries_all_detections():
    prog = [(1, 3, ChordCandidate("C", "", 0.8)), (3, 4, ChordCandidate("G", "", 0.7))]
    xml = build_score_xml(_notes_one_beat(), BeatGrid(117.5), KEY_C, chord_progression=prog)
    creator = _parse(xml).find("identification/creator")
    assert creator is not None and creator.text is not None
    assert "117.5 BPM" in creator.text
    assert "C major" in creator.text
    assert "4/4" in creator.text
    assert "C / G" in creator.text


def test_new_options_off_disable_features():
    xml = build_score_xml(
        _notes_one_beat(), GRID, KEY_C,
        with_note_colors=False, with_solfa_lyrics=False,
    )
    root = _parse(xml)
    note = root.find("part/measure/note")
    assert note is not None
    assert note.get("color") is None
    assert note.find("lyric") is None
