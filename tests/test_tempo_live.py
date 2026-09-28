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
    # gaps of 1 s (60 BPM)... median -> 60; inside the range directly
    notes = _notes_at_beats(60.0, 6)
    bpm = estimate_bpm_live(notes)
    assert bpm is not None and 50.0 <= bpm <= 130.0


def test_estimate_live_too_few_notes():
    notes = _notes_at_beats(120.0, 3)
    assert estimate_bpm_live(notes) is None


def test_estimate_live_dotted_pulse_corrects_to_quarter():
    """Dotted beats with a strong/weak alternation: the median is a DOTTED
    beat, the true tempo is x1.5 — the x2 fold never repairs that ratio.

    Real 6/8 case: dotted beats at 100 BPM (1.5 quarters, 0.9 s gaps)
    published 66.7 BPM for the whole take.
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
    """Without an alternation (flat strengths), no x1.5 correction: the
    gaps stay read as quarters and the x2 octave fold applies as is —
    an accepted limit, frozen by test_live_grid_policy's dedicated pin.
    """
    from v2m.core.notes import Note

    notes = [
        Note(start=i * 1.5, end=i * 1.5 + 1.2, midi=60, confidence=0.9, strength=0.5)
        for i in range(8)
    ]
    bpm = estimate_bpm_live(notes)
    # 40 BPM -> x2 fold -> 80; never 60 (the 1.5x tempo)
    assert bpm is not None and abs(bpm - 80.0) <= 2.0
