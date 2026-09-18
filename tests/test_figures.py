"""Tests figures.py : classification ronde → double-croche, pointées, triolets."""

from v2m.core.figures import (
    classify_duration,
    figures_summary,
    label_notes,
    short_label,
)
from v2m.core.notes import Note

BPM = 120.0


def _beats(n: float) -> float:
    """Durée en secondes de n temps à 120 BPM (0.5 s par temps)."""
    return n * 60.0 / BPM


def test_simple_figures():
    # classify_duration prend des TEMPS (beats), pas des secondes
    assert classify_duration(4.0).name == "ronde"
    assert classify_duration(2.0).name == "blanche"
    assert classify_duration(1.0).name == "noire"
    assert classify_duration(0.5).name == "croche"
    assert classify_duration(0.25).name == "double-croche"


def test_dotted_figures():
    assert classify_duration(3.0).name == "blanche pointée"
    assert classify_duration(1.5).name == "noire pointée"
    assert classify_duration(0.75).name == "croche pointée"


def test_triplet_figures():
    assert classify_duration(2 / 3).name == "triolet de noire"
    assert classify_duration(1 / 3).name == "triolet de croche"
    assert classify_duration(4 / 3).name == "triolet de blanche"


def test_tolerance_rejects_offgrid():
    """Au-delà de la tolérance (±18 % pondéré), aucune classe."""
    # 4.8 temps : dans le trou entre la ronde (4, +18 %) et la ronde pointée (6, -18 %)
    assert classify_duration(4.8) is None
    # 10 temps : plus long que toute figure du catalogue
    assert classify_duration(10.0) is None


def test_imperfect_human_timing_classified():
    """Une note humaine à ±15 % reste bien étiquetée (pas déclassée en triolet)."""
    assert classify_duration(0.86).name == "noire"
    assert classify_duration(1.15).name == "noire"   # pénalité triolet : noire gagne
    assert classify_duration(0.53).name == "croche"  # idem croche vs triolet de noire


def test_triplet_wins_when_clearly_triplet():
    """Aux durées exactes de triolet, c'est bien le triolet qui gagne."""
    assert classify_duration(2 / 3).name == "triolet de noire"
    assert classify_duration(4 / 3).name == "triolet de blanche"
    assert classify_duration(1.3).name == "triolet de blanche"  # à 2,5 % du triolet


def test_label_notes_on_analysis_like_notes():
    notes = [
        Note(start=0.0, end=_beats(1), midi=60, confidence=0.9),      # noire
        Note(start=_beats(1), end=_beats(1) + _beats(1.5), midi=62, confidence=0.9),  # noire pointée
        Note(start=_beats(2.5), end=_beats(2.5) + _beats(2 / 3), midi=64, confidence=0.9),  # triolet noire
        Note(start=_beats(3.2), end=_beats(3.2) + _beats(0.5), midi=65, confidence=0.9),  # croche
        Note(start=_beats(4), end=_beats(4) + 0.13, midi=67, confidence=0.9),  # double-croche
    ]
    label_notes(notes, BPM)
    assert [n.figure for n in notes] == [
        "noire", "noire pointée", "triolet de noire", "croche", "double-croche",
    ]


def test_label_notes_zero_bpm_noop():
    notes = [Note(start=0, end=0.5, midi=60, confidence=0.9)]
    label_notes(notes, 0.0)
    assert notes[0].figure is None


def test_summary_and_short_labels():
    notes = [
        Note(start=0, end=_beats(1), midi=60, confidence=0.9),
        Note(start=_beats(1), end=_beats(2), midi=60, confidence=0.9),
        Note(start=_beats(2), end=_beats(3), midi=60, confidence=0.9),
        Note(start=_beats(3), end=_beats(4), midi=60, confidence=0.9),
    ]
    label_notes(notes, BPM)
    assert figures_summary(notes) == {"noire": 4}
    assert [short_label(f) for f in ("noire", "croche pointée", "triolet de noire", None)] == [
        "N", "C.", "3N", "",
    ]
