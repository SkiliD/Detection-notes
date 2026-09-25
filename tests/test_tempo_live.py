"""Tests estimate_bpm_live : BPM approximatif pendant l'enregistrement."""

import numpy as np

from v2m.core.notes import Note
from v2m.core.tempo import estimate_bpm_live


def _notes_at_beats(bpm: float, count: int):
    beat = 60.0 / bpm
    return [Note(start=i * beat, end=i * beat + beat * 0.8, midi=60, confidence=0.9)
            for i in range(count)]


def test_estimate_live_120():
    notes = _notes_at_beats(120.0, 8)
    bpm = estimate_bpm_live(notes)
    assert bpm is not None and abs(bpm - 120.0) <= 5.0


def test_estimate_live_folds_octave():
    # gaps de 1 s (60 BPM)... médiane -> 60 ; dans le range directement
    notes = _notes_at_beats(60.0, 6)
    bpm = estimate_bpm_live(notes)
    assert bpm is not None and 50.0 <= bpm <= 130.0


def test_estimate_live_too_few_notes():
    notes = _notes_at_beats(120.0, 3)
    assert estimate_bpm_live(notes) is None


def test_estimate_live_dotted_pulse_corrects_to_quarter():
    """Temps pointés avec alternance forte/faible : la médiane est un temps
    POINTÉ, le vrai tempo est ×1,5 — le repli ×2 ne répare jamais ce rapport.

    Cas 6/8 réel : des temps pointés à 100 BPM (1,5 noire, gaps de 0,9 s)
    donnaient 66,7 BPM publiés pendant toute la prise.
    """
    from v2m.core.notes import Note

    notes = [
        Note(start=i * 0.9, end=i * 0.9 + 0.75, midi=60, confidence=0.9,
             strength=(0.9 if i % 2 == 0 else 0.2))
        for i in range(8)
    ]
    bpm = estimate_bpm_live(notes)
    assert bpm is not None and abs(bpm - 100.0) <= 5.0


def test_estimate_live_flat_dotted_gaps_keep_octave_fold():
    """Sans alternance (forces plates), pas de correction ×1,5 : les écarts
    restent lus comme des noires et le repli d'octave ×2 s'applique tel quel
    — limite assumée, gelée par l'épingle dédiée de test_live_grid_policy.
    """
    from v2m.core.notes import Note

    notes = [
        Note(start=i * 1.5, end=i * 1.5 + 1.2, midi=60, confidence=0.9, strength=0.5)
        for i in range(8)
    ]
    bpm = estimate_bpm_live(notes)
    # 40 BPM -> repli x2 -> 80 ; jamais 60 (le tempo 1,5x)
    assert bpm is not None and abs(bpm - 80.0) <= 2.0
