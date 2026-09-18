"""Détection de hauteur : pYIN (hors-ligne) et YIN par bloc (temps réel)."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

FMIN = 65.0    # ~ C2
FMAX = 1050.0  # ~ C6


@dataclass
class PitchTrack:
    """Courbe de hauteur : Hz + confiance par frame."""

    times: np.ndarray   # (n,) temps en secondes (centre de frame)
    hz: np.ndarray      # (n,) fréquence (NaN si non-voisé)
    confidence: np.ndarray  # (n,) proba de voisement


def track_pitch_offline(y: np.ndarray, sr: int, frame_ms: float = 20.0) -> PitchTrack:
    """pYIN sur un signal complet. Précis, ~0.3x la durée réelle."""
    hop = max(64, int(sr * frame_ms / 1000))
    f0, voiced, _ = librosa.pyin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=2048,
        hop_length=hop,
        fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    confidence = np.where(np.isnan(f0), 0.0, voiced.astype(float))
    return PitchTrack(
        times=np.asarray(times, dtype=float),
        hz=np.asarray(f0, dtype=float),
        confidence=np.asarray(confidence, dtype=float),
    )


class YinBlock:
    """Estimateur YIN par bloc, incrémental (temps réel).

    On garde un buffer glissant : le pitch est estimé sur les `window`
    derniers échantillons à chaque `hop` échantillons.
    """

    def __init__(self, sr: int, window: int = 1024, hop: int = 256):
        self.sr = sr
        self.window = window
        self.hop = hop
        self.buffer = np.zeros(0, dtype=np.float32)
        self.pos = 0  # position du prochain hop

    def push(self, block: np.ndarray) -> list[tuple[float, float]]:
        """Ajoute un bloc, renvoie [(temps_s, hz | nan), ...] pour les nouveaux hops."""
        self.buffer = np.concatenate([self.buffer, np.asarray(block, dtype=np.float32)])
        out: list[tuple[float, float]] = []
        while self.pos + self.window <= len(self.buffer):
            frame = self.buffer[self.pos : self.pos + self.window]
            t = (self.pos + self.window / 2) / self.sr
            hz = yin_pitch(frame, self.sr)
            out.append((t, hz))
            self.pos += self.hop
        # on libère l'ancien buffer au-delà d'une fenêtre de grâce
        grace = self.window + self.hop
        if self.pos > len(self.buffer) - grace:
            cut = max(0, self.pos - self.window)
            self.buffer = self.buffer[cut:]
            self.pos -= cut
        return out


def yin_pitch(frame: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX) -> float:
    """YIN simplifié (différence moyenne + seuil) sur une trame."""
    x = np.asarray(frame, dtype=np.float64)
    if np.max(np.abs(x)) < 1e-4:
        return float("nan")
    w = len(x)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(w - 1, int(sr / fmin))
    if tau_max <= tau_min:
        return float("nan")

    # différence cumulative moyenne normalisée d'(tau)
    d = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        diff = x[: w - tau] - x[tau:]
        d[tau] = np.mean(diff * diff)
    cum = np.cumsum(d[1:]) / np.arange(1, tau_max + 1)
    d_prime = d[1:] / np.where(cum == 0, 1e-12, cum)

    # premier minimum sous le seuil, sinon le minimum global
    threshold = 0.15
    best = None
    for i in range(tau_min - 1, tau_max - 1):
        if d_prime[i] < threshold:
            # parabole autour du minimum local
            if i + 1 < len(d_prime) and d_prime[i] > d_prime[i + 1]:
                continue
            best = i
            break
    if best is None:
        seg = d_prime[tau_min - 1 : tau_max]
        if not len(seg):
            return float("nan")
        best = tau_min - 1 + int(np.argmin(seg))
        if d_prime[best] > 0.6:
            return float("nan")
    tau = best + 1
    # interpolation parabolique autour du minimum (précision sub-échantillon)
    # points : d_prime[best-1] (tau-1), d_prime[best] (tau), d_prime[best+1] (tau+1)
    if 0 < best < len(d_prime) - 1:
        a, b, c = d_prime[best - 1], d_prime[best], d_prime[best + 1]
        denom = 2.0 * (a + c - 2.0 * b)
        if abs(denom) > 1e-12:
            tau = tau + (a - c) / denom
    return float(sr / tau)
