"""Tests segmentation : 3 notes synthétiques -> 3 événements."""

import numpy as np

from conftest import SR, silence, sine_note
from v2m.core.notes import track_to_notes
from v2m.core.pitch import track_pitch_offline


def _three_notes_signal() -> np.ndarray:
    return np.concatenate(
        [
            sine_note(220.0, 0.5),   # A3
            silence(0.15),
            sine_note(261.63, 0.5),  # C4
            silence(0.15),
            sine_note(329.63, 0.5),  # E4
        ]
    )


def test_three_notes_detected():
    y = _three_notes_signal()
    track = track_pitch_offline(y, SR)
    notes = track_to_notes(track, SR)
    assert len(notes) == 3
    midis = [n.midi for n in notes]
    assert midis == [57, 60, 64]  # A3, C4, E4


def test_note_times_reasonable():
    y = _three_notes_signal()
    track = track_pitch_offline(y, SR)
    notes = track_to_notes(track, SR)
    # la première note commence près de 0
    assert notes[0].start < 0.15
    # durée de chaque note ~0.5 s (± tolérance)
    for n in notes:
        assert 0.3 < n.duration < 0.75
    # ordre chronologique
    assert notes[0].end <= notes[1].start + 0.1


def test_silence_gives_no_notes():
    track = track_pitch_offline(np.zeros(SR * 2, dtype=np.float32), SR)
    notes = track_to_notes(track, SR)
    assert notes == []


def test_vibrato_single_note():
    """Une note avec vibrato (±30 cents) reste une seule note."""
    dur = 1.0
    t = np.arange(int(dur * SR)) / SR
    hz = 440.0 * 2 ** (0.3 * np.sin(2 * np.pi * 5.5 * t) / 12)  # ±30 cents
    env = np.minimum(1, np.minimum(t / 0.01, (dur - t) / 0.01))
    y = (0.5 * np.sin(2 * np.pi * np.cumsum(hz) / SR) * env).astype(np.float32)
    track = track_pitch_offline(y, SR)
    notes = track_to_notes(track, SR)
    assert len(notes) == 1
    assert notes[0].midi == 69
