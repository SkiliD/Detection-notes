"""Tests pitch : sinus connu -> fréquence détectée juste."""

import numpy as np

from conftest import SR, sine_note
from v2m.core.music import hz_to_midi
from v2m.core.pitch import track_pitch_offline, yin_pitch


def test_yin_block_440hz():
    """YIN par bloc sur un la 440 Hz -> ~440."""
    frame = sine_note(440.0, 1024 / SR, sr=SR)
    hz = yin_pitch(frame, SR)
    assert hz == hz  # pas NaN
    assert abs(hz - 440.0) / 440.0 < 0.02  # < 2 %


def test_yin_block_silence_nan():
    """Silence -> NaN (non voisé)."""
    hz = yin_pitch(np.zeros(1024, dtype=np.float32), SR)
    assert np.isnan(hz)


def test_pyin_440hz_median():
    """pYIN sur 1 s de la 440 -> médiane ~440, note MIDI ~69."""
    y = sine_note(440.0, 1.0, sr=SR)
    track = track_pitch_offline(y, SR)
    voiced = track.hz[~np.isnan(track.hz)]
    assert len(voiced) > 20
    med = float(np.median(voiced))
    assert abs(hz_to_midi(med) - 69.0) < 0.5


def test_pyin_two_octaves():
    """Deux notes successives (220 puis 880 Hz) bien séparées."""
    from conftest import silence

    y = np.concatenate([sine_note(220.0, 0.6), silence(0.2), sine_note(880.0, 0.6)])
    track = track_pitch_offline(y, SR)
    voiced = ~np.isnan(track.hz)
    assert voiced.sum() > 30
    first_half = np.median(track.hz[voiced][: len(voiced) // 2])
    last_half = np.median(track.hz[voiced][-len(voiced) // 2 :])
    assert abs(hz_to_midi(first_half) - 57.0) < 0.7   # A3 = 220 Hz
    assert abs(hz_to_midi(last_half) - 81.0) < 0.7    # A5 = 880 Hz
