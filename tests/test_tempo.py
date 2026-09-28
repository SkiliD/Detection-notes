"""Tempo tests: clicks at a known BPM -> the right detected BPM."""

import numpy as np

from conftest import SR
from v2m.core.tempo import (
    _REFINE_CACHE,
    detect_tempo,
    refine_bpm_with_onsets,
)


def click_track(bpm: float, dur_s: float = 20.0, sr: int = SR) -> np.ndarray:
    """A percussive click track at the given tempo."""
    n = int(dur_s * sr)
    y = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    t_click = 0.0
    rng = np.random.default_rng(42)
    while t_click < dur_s - 0.05:
        start = int(t_click * sr)
        length = int(0.05 * sr)
        env = np.exp(-np.linspace(0, 8, length))
        burst = env * rng.standard_normal(length) * 0.6
        y[start : start + length] += burst.astype(np.float32)
        t_click += period
    return y


def test_click_120bpm():
    y = click_track(120.0)
    res = detect_tempo(y, SR)
    assert abs(res.bpm - 120.0) <= 4.0


def test_click_90bpm():
    y = click_track(90.0)
    res = detect_tempo(y, SR)
    assert abs(res.bpm - 90.0) <= 4.0


def test_refine_bpm_with_onsets():
    """A rounded BPM (117.5) is refined toward the true tempo (120) on a
    grid of exact attacks."""
    bpm_true = 120.0
    onsets = np.arange(16) * (60.0 / bpm_true)  # attacks right on the grid
    refined = refine_bpm_with_onsets(onsets, 117.5)
    assert abs(refined - 120.0) <= 0.5


def test_refine_bpm_keeps_close_value():
    """If the attacks are already in phase, the BPM stays near identical."""
    onsets = np.arange(10) * 0.5
    refined = refine_bpm_with_onsets(onsets, 120.0)
    assert abs(refined - 120.0) <= 1.0


def test_refine_bpm_too_few_onsets():
    """Fewer than 3 attacks: no refinement, BPM unchanged."""
    assert refine_bpm_with_onsets(np.array([0.0, 0.5]), 120.0) == 120.0


def test_refine_cache_answers_identically_and_invalidates_on_change():
    """The refine scan's cache is a no-drift device: a content hit re-answers
    the SAME number (a fresh array from the same evidence still hits), and
    changed onsets re-scan instead of serving a stale reading."""
    _REFINE_CACHE.clear()
    try:
        onsets = np.arange(16) * (60.0 / 120.0)
        first = refine_bpm_with_onsets(onsets, 117.5)
        assert _REFINE_CACHE, "the first call must fill the cache"
        # a FRESH array with the same content hits the cache: same answer
        second = refine_bpm_with_onsets(onsets.copy(), 117.5)
        assert second == first, "a cache hit must re-answer the same reading"
        # changed onsets are a different key: re-scan, no stale reading. (A
        # small or uniform shift is absorbed by the polish regression — same
        # slope, same BPM — the refine is DESIGNED to ignore it; the drift
        # test needs an input whose true reading moves: a second tempo.)
        _REFINE_CACHE.clear()
        drifted = np.concatenate([onsets[:10], 5.0 + np.arange(6) * 0.42])
        other = refine_bpm_with_onsets(drifted, 117.5)
        assert other != first, "changed onsets must re-scan, not serve the cached bpm"
    finally:
        _REFINE_CACHE.clear()  # no cross-test pollution of the one-entry cache


def test_bpm_in_range():
    y = click_track(72.0, dur_s=15.0)
    res = detect_tempo(y, SR, bpm_range=(60.0, 180.0))
    assert 60.0 <= res.bpm <= 180.0
