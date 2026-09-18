"""Tests tempo : clics à BPM connu -> BPM détecté juste."""

import numpy as np

from conftest import SR
from v2m.core.tempo import detect_tempo, refine_bpm_with_onsets


def click_track(bpm: float, dur_s: float = 20.0, sr: int = SR) -> np.ndarray:
    """Piste de clics percussifs au tempo donné."""
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
    """Un BPM arrondi (117,5) est raffiné vers le vrai tempo (120) sur
    une grille d'attaques exactes."""
    bpm_true = 120.0
    onsets = np.arange(16) * (60.0 / bpm_true)  # attaques pile sur la grille
    refined = refine_bpm_with_onsets(onsets, 117.5)
    assert abs(refined - 120.0) <= 0.5


def test_refine_bpm_keeps_close_value():
    """Si les attaques sont déjà en phase, le BPM reste quasi identique."""
    onsets = np.arange(10) * 0.5
    refined = refine_bpm_with_onsets(onsets, 120.0)
    assert abs(refined - 120.0) <= 1.0


def test_refine_bpm_too_few_onsets():
    """Moins de 3 attaques : pas de raffinement, BPM inchangé."""
    assert refine_bpm_with_onsets(np.array([0.0, 0.5]), 120.0) == 120.0


def test_bpm_in_range():
    y = click_track(72.0, dur_s=15.0)
    res = detect_tempo(y, SR, bpm_range=(60.0, 180.0))
    assert 60.0 <= res.bpm <= 180.0
