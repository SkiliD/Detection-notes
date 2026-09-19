"""Tests for the local playback renderer (playback.render_notes)."""

import numpy as np

from v2m.core import audio_io
from v2m.core.notes import Note
from v2m.core.playback import render_notes


def test_render_produces_nonempty_signal():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9),
             Note(start=0.5, end=1.0, midi=64, confidence=0.9)]
    y = render_notes(notes, sr=22050)
    assert len(y) > 0
    assert np.max(np.abs(y)) > 0.1
    assert np.issubdtype(y.dtype, np.floating)


def test_render_duration_covers_last_note():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9),
             Note(start=1.5, end=2.0, midi=67, confidence=0.9)]
    y = render_notes(notes, sr=22050)
    assert len(y) >= int(2.0 * 22050)


def test_render_normalizes_peak():
    notes = [Note(start=0.0, end=1.0, midi=60, confidence=0.9)]
    y = render_notes(notes, sr=22050)
    peak = float(np.max(np.abs(y)))
    assert 0.5 < peak <= 1.0  # normalized, not clipped


def test_render_transpose_shifts_content():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    y_ref = render_notes(notes, sr=22050, transpose=0)
    y_up = render_notes(notes, sr=22050, transpose=12)  # one octave up
    assert len(y_ref) == len(y_up)
    assert not np.allclose(y_ref, y_up)


def test_render_empty_notes():
    y = render_notes([], sr=22050)
    assert len(y) == 0


def test_render_sample_exact_tone_length():
    """Regression: tone() re-derived its length through a seconds round-trip
    (int(dur * sr)) and lost one sample on some durations -> ValueError while
    mixing (30870 vs 30869 samples). `end` below makes i1 - i0 == 30870."""
    sr = audio_io.SR
    end = (30870 + 0.5) / sr  # int(end * sr) == 30870 exactly
    notes = [Note(start=0.0, end=end, midi=60, confidence=0.9)]
    y = render_notes(notes, sr=sr)  # used to raise ValueError
    assert len(y) > 0


def test_render_prime_and_edge_lengths():
    """Off-by-one-prone slice lengths must all render without a shape error."""
    for n in (1, 7, 30869, 30870, 30871):
        notes = [Note(start=0.0, end=(n + 0.5) / 22050, midi=60, confidence=0.9)]
        assert len(render_notes(notes, sr=22050)) >= n


def test_render_arbitrary_durations_never_crash():
    """A random sweep of note durations must never mis-shape the mix.

    ~9% of float durations lost a sample in the old round-trip, so a
    handful of notes was enough to hit the crash; 200 sweeps pin it for
    good at the project sample rate.
    """
    sr = audio_io.SR
    rng = np.random.default_rng(7)
    notes, t = [], 0.0
    for _ in range(200):
        dur = float(rng.uniform(0.02, 2.5))
        notes.append(Note(start=t, end=t + dur, midi=60, confidence=0.9))
        t += dur + 0.03
    y = render_notes(notes, sr=sr)
    assert len(y) >= int(t * sr)
    assert np.issubdtype(y.dtype, np.floating)
