"""Tests meter.py : détection simple vs composée (6/8) et positions."""

from v2m.core.meter import DEFAULT_METER, Meter, detect_meter, note_positions
from v2m.core.notes import Note

BPM = 120.0
BEAT = 60.0 / BPM  # 0.5 s


def _notes_from_starts(starts, dur=BEAT * 0.8, midi=60):
    return [Note(start=t, end=t + dur, midi=midi, confidence=0.9) for t in starts]


def test_binary_grid_no_conclusion():
    """Grille binaire (attaques sur les temps et double-croches) : 2/4, 3/4
    et 4/4 sont indiscernables sans accents -> None (4/4 par défaut)."""
    starts = [i * BEAT for i in range(8)] + [i * BEAT + 0.375 for i in range(8)]
    assert detect_meter(_notes_from_starts(starts), BPM) is None


def test_detect_6_8_by_offbeats():
    """6/8 : attaques sur les temps pointés 1 et 2 (croches 1 et 4)."""
    eighth = 60.0 / BPM / 2
    starts = [m * 1.5 + e for m in range(3) for e in (0.0, 3 * eighth)]
    meter = detect_meter(_notes_from_starts(starts), BPM)
    assert meter is not None and meter.label == "6/8"
    assert meter.score >= 0.6


def test_6_8_beats_fall_between_binary_slots():
    """Le cas qui casse un rendu 4/4 : le 2e temps pointé du 6/8 tombe sur
    le « et » binaire (0,75 noire) — indétectable d'une grille simple."""
    eighth = 60.0 / BPM / 2
    second_slot = 3 * eighth  # 0,75 s = 1,5 noire
    assert abs((second_slot / (60.0 / BPM)) - 1.5) < 1e-9


def test_binary_only_returns_none():
    """Grille strictement binaire (temps et croches) : rien sur les temps
    pointés ternaires -> aucune conclusion."""
    eighth = BEAT / 2
    starts = [i * BEAT for i in range(8)] + [i * BEAT + eighth for i in range(8)]
    assert detect_meter(_notes_from_starts(starts), BPM) is None


def test_a_binary_4_4_line_is_not_6_8():
    """Régression mesurée : une ligne en 4/4 parfaitement régulière sortait
    en 6/8, ce qui décale toutes les barres de mesure.

    Ses attaques tombent pourtant souvent sur la grille ternaire (0,71 sur
    le motif de référence) — c'est pourquoi l'alignement seul ne décide pas :
    la grille simple doit être nettement moins bien expliquée (0,86).
    """
    starts = [i * BEAT + h * BEAT / 2 for i in range(4) for h in (0, 1)]
    meter = detect_meter(_notes_from_starts(starts), BPM)
    assert meter is None or meter.label != "6/8"


def test_too_few_notes_returns_none():
    notes = _notes_from_starts([0.0, BEAT, 2 * BEAT])
    assert detect_meter(notes, BPM) is None


def test_default_meter():
    assert DEFAULT_METER.label == "4/4"
    assert not DEFAULT_METER.is_compound
    assert DEFAULT_METER.measure_seconds(BPM) == 2.0


def test_note_positions():
    meter = Meter(3, 4)
    notes = _notes_from_starts([0.0, BEAT, 2 * BEAT, 3 * BEAT, 4 * BEAT])
    assert note_positions(notes, BPM, meter) == [
        (1, 1.0), (1, 2.0), (1, 3.0), (2, 1.0), (2, 2.0),
    ]


def test_note_positions_compound():
    meter = Meter(6, 8)
    eighth = 60.0 / BPM / 2
    notes = _notes_from_starts([0.0, 3 * eighth, 1.5, 1.5 + 3 * eighth])
    assert note_positions(notes, BPM, meter) == [
        (1, 1.0), (1, 2.0), (2, 1.0), (2, 2.0),
    ]


def test_compound_meter_length():
    """6/8 = 6 croches = 3 noires = 1,5 s à 120 BPM ; 2 temps pointés affichés."""
    meter = Meter(6, 8)
    assert meter.measure_quarters == 3.0
    assert meter.measure_seconds(BPM) == 1.5
    assert meter.n_slots == 2
    assert meter.slot_quarters == 1.5
    assert meter.is_compound
