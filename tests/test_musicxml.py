"""Tests export MusicXML : XML valide, mesures, armure, ties, triolets, silences."""

import xml.etree.ElementTree as ET

import pytest
from scorexml import assert_measures_close, score_root, text_of

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
    assert text_of(root.find("part-list/score-part/part-name")) == "Voix"
    attrs = root.find("part/measure/attributes")
    assert attrs is not None
    assert text_of(attrs.find("divisions")) == "24"
    assert text_of(attrs.find("key/fifths")) == "0"
    assert text_of(attrs.find("time/beats")) == "4"
    assert text_of(attrs.find("clef/sign")) == "G"


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
    tied = [t for n in notes_xml if (t := n.find("notations/tied")) is not None]
    assert len(tied) == 2
    types = sorted(t.get("type") or "" for t in tied)
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
    assert text_of(tm[0].find("actual-notes")) == "3"
    assert text_of(tm[0].find("normal-notes")) == "2"
    tuplets = root.findall("part/measure/note/notations/tuplet")
    sides = [t.get("type") for t in tuplets]
    assert sides == ["start", "stop"]  # le milieu n'a pas de bracket
    # le bloc total = 2 temps = 48 divisions -> 16 par note, puis les deux
    # temps libres ferment la mesure
    durations = [int(text_of(n.find("duration"))) for n in root.findall("part/measure/note")]
    assert durations[:3] == [16, 16, 16]
    assert sum(durations) == 96  # une mesure 4/4 à 24 divisions par temps


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
    assert [(text_of(n.find("type")), len(n.findall("dot"))) for n in m1_rests] == [("whole", 0)]
    m2_rests = [n for n in measures[1].findall("note") if n.find("rest") is not None]
    assert [(text_of(n.find("type")), len(n.findall("dot"))) for n in m2_rests] == [("half", 1)]


# ----------------------------------------------------------------- silences


def test_leading_rest_uses_a_whole_rest_per_empty_measure():
    # note au 5e temps : la mesure 1 entière est silencieuse -> une pause
    notes = [Note(start=2.0, end=2.5, midi=60, confidence=0.9)]
    xml = build_score_xml(notes, GRID, KEY_C)
    root = _parse(xml)
    m1 = root.findall("part/measure")[0]
    types = [text_of(n.find("type")) for n in m1.findall("note") if n.find("rest") is not None]
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
    types = [text_of(n.find("type")) for n in m1.findall("note") if n.find("rest") is not None]
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
    assert text_of(rest_notes[0].find("duration")) == "48"    # 2 temps = 48 divisions
    assert text_of(rest_notes[0].find("type")) == "half"      # vraie valeur, pas générique


# ------------------------------------------- ce que les prises n'exercent pas
BEAT, SLOT = 0.5, 0.75          # 120 BPM : un temps noir, un temps pointé
GRID_44 = BeatGrid(BPM)
GRID_68 = BeatGrid(BPM, Meter(6, 8))


def _note(midi: int, start: float, end: float, figure: str | None = None) -> Note:
    return Note(start=start, end=end, midi=midi, confidence=0.9, figure=figure)


# Chaque cas doit se refermer et ne dire que des durées qu'il peut remplir ;
# le contrat est vérifié par `scorexml.assert_measures_close`.
SYNTHETIC = {
    "4/4 plein": lambda: (GRID_44, [_note(60 + i, i * BEAT, (i + 1) * BEAT) for i in range(4)]),
    "4/4 note tenue": lambda: (GRID_44, [_note(60, 0.0, 2 * BEAT)]),
    "4/4 seizièmes": lambda: (
        GRID_44, [_note(60 + i, i * BEAT / 4, (i + 1) * BEAT / 4) for i in range(4)]
    ),
    "4/4 triolet de croches": lambda: (
        GRID_44,
        [_note(60 + i, i * BEAT / 3, (i + 1) * BEAT / 3, "triolet de croche") for i in range(3)],
    ),
    "6/8 plein": lambda: (GRID_68, [_note(60 + i, i * SLOT, (i + 1) * SLOT) for i in range(2)]),
    "6/8 vide": lambda: (GRID_68, []),
    "6/8 partiel": lambda: (GRID_68, [_note(60, 0.0, SLOT)]),
    "6/8 trois mesures": lambda: (
        GRID_68, [_note(60 + i, i * SLOT, (i + 1) * SLOT) for i in range(6)]
    ),
}


@pytest.mark.parametrize("label", sorted(SYNTHETIC))
def test_a_synthetic_score_closes_and_its_types_do_not_lie(label):
    """Le contrat vaut aussi là où les deux prises ne vont pas : mesure 6/8 de
    deux temps pointés (36 divisions), triolet (deux tiers de durée), durées
    courtes. Un 6/8 gravé sur six temps au lieu de deux sortait trois fois trop
    long et ne se refermait pas."""
    grid, notes = SYNTHETIC[label]()
    assert_measures_close(score_root(build_score_xml(notes, grid, KEY_C)), label)


def test_the_snapped_score_writes_no_duration_below_a_sixteenth():
    """Le trente-deuxième (3 divisions à 24 par temps) est exprimable, mais le
    chemin arrimé n'y descend pas : les bornes tombent sur des quarts de temps,
    donc la plus courte valeur écrite est un seizième (6 divisions) et le type
    n'apparaît jamais."""
    for label, make in sorted(SYNTHETIC.items()):
        grid, notes = make()
        root = score_root(build_score_xml(notes, grid, KEY_C))
        for n in root.findall("part/measure/note"):
            assert int(text_of(n.find("duration"))) >= 6, label
            assert text_of(n.find("type")) != "32nd", label


def test_the_closure_property_refuses_a_type_that_lies_about_its_duration():
    """Un filet non éprouvé ne prouve rien : trois noires et une blanche se
    referment (96 divisions), donc seule la cohérence type/durée peut refuser
    cette dernière — et elle doit le faire."""
    forged = (
        '<?xml version="1.0"?><score-partwise><part><measure number="1">'
        "<attributes><divisions>24</divisions><time><beats>4</beats>"
        "<beat-type>4</beat-type></time></attributes>"
        + '<note><rest/><duration>24</duration><type>quarter</type></note>' * 3
        + '<note><rest/><duration>24</duration><type>half</type></note>'
        + "</measure></part></score-partwise>"
    )
    with pytest.raises(AssertionError, match="half pour 24 divisions"):
        assert_measures_close(score_root(forged), "cas forgé")


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


# ------------------------------------- la table des durées EST le catalogue
@pytest.mark.parametrize("steps", sorted(_TYPE_BY_STEPS))
def test_the_export_types_a_duration_the_way_the_grid_writes_it(steps):
    """Un seul arbitre de la valeur écrite : la table des divisions est le
    catalogue de la Grille (`DRAWABLE`) en divisions. Seuls les triolets (le
    catalogue n'en connaît aucun) et le trente-deuxième, plus court que son
    plancher au seizième, sont énoncés par l'export."""
    beats = steps / DIVISIONS
    written = GRID.written(GRID.seconds(beats))
    if abs(written.beats - beats) < 1e-9:
        assert _TYPE_BY_STEPS[steps] == (written.type_, written.dots)
    else:
        assert steps in _TUPLET_STEPS or steps == 3


def test_a_dotted_sixteenth_is_written_on_the_staff_but_snapped_in_the_file():
    """Le seizième pointé vaut 9 divisions, que la grille sait écrire ; mais
    l'arrimage, lui, reste au seizième (6 divisions), donc le fichier reçoit
    une croche là où la portée — qui n'arrime pas — écrit un seizième pointé.
    Et jamais un triolet : 4 divisions ne sortent que d'un bloc de triolets."""
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
    assert "<time-modification>" not in xml       # donc aucun triolet
    assert 0.375 * DIVISIONS in _TYPE_BY_STEPS    # 9 : la valeur est exprimable


def test_an_isolated_short_note_keeps_its_duration_and_rests_the_beat():
    """Une note courte isolée garde sa durée écrite : ce qui reste de son temps
    devient un silence, au lieu d'être absorbé par un temps entier inventé."""
    notes = [Note(start=0.0, end=GRID.seconds(0.5), midi=60, confidence=0.9)]
    written = [
        (text_of(n.find("type")), int(text_of(n.find("duration"))),
         "silence" if n.find("rest") is not None else "note")
        for n in _parse(build_score_xml(notes, GRID, KEY_C)).findall("part/measure/note")
    ]
    # croche chantée + croche de silence + blanche pointée pour le reste
    assert written == [("eighth", 12, "note"), ("eighth", 12, "silence"),
                       ("half", 72, "silence")]


def test_a_silent_compound_measure_closes_on_a_dotted_half_rest():
    """Une mesure 6/8 vaut trois noires (72 divisions), pas six temps : un seul
    silence de blanche pointée la remplit."""
    xml = build_score_xml([], BeatGrid(BPM, Meter(6, 8)), KEY_C)
    rests = [
        (text_of(n.find("type")), len(n.findall("dot")), int(text_of(n.find("duration"))))
        for n in _parse(xml).findall("part/measure/note")
    ]
    assert rests == [("half", 1, 72)]
