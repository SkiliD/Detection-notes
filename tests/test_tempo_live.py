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
