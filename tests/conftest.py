"""Fixtures : signaux audio synthétiques."""

import numpy as np
import pytest

SR = 22_050


def sine_note(hz: float, dur_s: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Sinusoïde avec enveloppe ADSR simplifiée (évite les clics)."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    y = amp * np.sin(2 * np.pi * hz * t)
    # enveloppe : 10 ms fondu entrée/sortie
    fade = int(0.01 * sr)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (y * env).astype(np.float32)


def silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(dur_s * sr), dtype=np.float32)


@pytest.fixture
def sr() -> int:
    return SR
