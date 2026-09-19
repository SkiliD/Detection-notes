"""Tests pitch : sinus connu -> fréquence détectée juste."""

import numpy as np
import pytest

from conftest import SR, sine_note
from v2m.core.music import hz_to_midi
from v2m.core.pitch import YinBlock, track_pitch_offline, yin_pitch


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


def _stream(y: np.ndarray, block: int, **kw) -> list[tuple[float, float]]:
    """Push a signal block by block, like the live capture does."""
    yin = YinBlock(SR, **kw)
    out: list[tuple[float, float]] = []
    for i in range(0, len(y), block):
        blk = y[i : i + block]
        if len(blk) < block:
            blk = np.pad(blk, (0, block - len(blk)))
        out.extend(yin.push(blk))
    return out


def test_yin_block_times_are_absolute_and_monotonic():
    """Le temps des trames doit avancer, même quand le buffer est purgé.

    Régression : après la purge, les trames repartaient du début du buffer
    (temps figé) — le segmenteur live ne confirmait alors plus aucune note.
    """
    frames = _stream(sine_note(440.0, 2.0, sr=SR), 1024, window=2048, hop=1024)
    times = [t for t, _ in frames]
    assert len(times) > 40
    diffs = np.diff(times)
    assert diffs.min() > 0  # strictement croissant
    assert diffs == pytest.approx([1024 / SR] * len(diffs), rel=1e-6)
    # couvre bien les 2 s, pas seulement les premiers blocs
    assert times[-1] > 1.8


def test_yin_block_stream_memory_is_bounded():
    """Le buffer glissant ne doit pas grossir avec la durée de la capture."""
    yin = YinBlock(SR, window=2048, hop=1024)
    y = sine_note(440.0, 3.0, sr=SR)
    for i in range(0, len(y), 1024):
        yin.push(y[i : i + 1024])
        assert len(yin.buffer) <= 2048 + 1024 + 1024


def test_yin_block_stream_tracks_pitch():
    """La hauteur détectée reste juste au fil des blocs (config live)."""
    frames = _stream(sine_note(440.0, 1.0, sr=SR), 1024, window=2048, hop=1024)
    hz = np.array([h for _, h in frames], dtype=float)
    voiced = hz[~np.isnan(hz)]
    assert len(voiced) > 0
    assert abs(float(np.median(voiced)) - 440.0) < 8.0


def test_yin_block_confirms_a_live_note():
    """Bout en bout : le segmenteur live doit voir une note sur un sinus."""
    from v2m.core.liveseg import LiveNoteSegmenter

    y = np.concatenate([sine_note(440.0, 0.5, sr=SR), np.zeros(SR // 2, dtype=np.float32)])
    seg = LiveNoteSegmenter()
    for t, hz in _stream(y, 1024, window=2048, hop=1024):
        seg.push(t, hz)
    seg.flush(1.0)
    assert [n.midi for n in seg.notes] == [69]


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
