"""Pitch detection: pYIN (offline) and block-wise YIN (real-time)."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

FMIN = 65.0    # ~ C2
# C7, not C6: the old 1050 Hz ceiling sat EXACTLY on the sung C6 boundary —
# pYIN had no room above to lock a soprano's top notes and folded them an
# octave down (measured: a sung 84 read 72, a 79-88 line read 67-76 — the
# classic octave error, 2 corpus cases). One octave of headroom reads them
# and moves NOTHING below (chant.wav/demo.wav byte-identical, corpus 15/15
# unchanged). pYIN's octave-halving is still cleaned by clean_pitch's
# median filter; the piano guard (A0..C8) rejects anything stray.
FMAX = 2100.0  # ~ C7


@dataclass
class PitchTrack:
    """Pitch curve: Hz + confidence per frame."""

    times: np.ndarray   # (n,) times in seconds (frame center)
    hz: np.ndarray      # (n,) frequency (NaN when unvoiced)
    confidence: np.ndarray  # (n,) voicing probability
    # True where |df0/dt| exceeds the transition threshold (vibrato peaks
    # aside, these are portamento ramps and note hand-overs). None until
    # clean_pitch computes it: pitch estimation must read STABLE plateaus
    # only, never the ramp that leads into them.
    transition: np.ndarray | None = None


def track_pitch_offline(y: np.ndarray, sr: int, frame_ms: float = 20.0) -> PitchTrack:
    """pYIN on a complete signal. Accurate, ~0.3x real duration.

    `confidence` carries pYIN's voicing PROBABILITY (the third return, once
    discarded), not the binary voiced flag: a breath's noise reads 0.01-0.1
    while a sung frame reads 0.95+ (measured on the voice harness), so
    `clean_pitch`'s `min_conf` mask actually has something to mask — with
    the flag (1.0 wherever f0 exists) nothing was ever filtered.
    """
    hop = max(64, int(sr * frame_ms / 1000))
    f0, _voiced_flag, vprob = librosa.pyin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=2048,
        hop_length=hop,
        fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    confidence = np.where(np.isnan(f0), 0.0, vprob)
    return PitchTrack(
        times=np.asarray(times, dtype=float),
        hz=np.asarray(f0, dtype=float),
        confidence=np.asarray(confidence, dtype=float),
    )


class YinBlock:
    """Incremental block-wise YIN estimator (real-time).

    Keeps a sliding buffer: pitch is estimated over the last `window`
    samples at every `hop` samples.
    """

    def __init__(self, sr: int, window: int = 1024, hop: int = 256):
        self.sr = sr
        self.window = window
        self.hop = hop
        self.buffer = np.zeros(0, dtype=np.float32)
        self.pos = 0     # index of the next frame inside `buffer`
        self.offset = 0  # absolute sample index of buffer[0]

    def push(self, block: np.ndarray) -> list[tuple[float, float]]:
        """Push a block, return [(time_s, hz | nan), ...] for the new hops.

        Times are absolute (measured from the first pushed sample): trimming
        the buffer must never restart the clock, or the segmenter would see
        a frozen time and never confirm a note.
        """
        self.buffer = np.concatenate([self.buffer, np.asarray(block, dtype=np.float32)])
        out: list[tuple[float, float]] = []
        while self.pos + self.window <= len(self.buffer):
            frame = self.buffer[self.pos : self.pos + self.window]
            t = (self.offset + self.pos + self.window / 2) / self.sr
            hz = yin_pitch(frame, self.sr)
            out.append((t, hz))
            self.pos += self.hop
        # release the samples we will never look at again (one window in hand)
        cut = max(0, self.pos - self.window)
        if cut:
            self.buffer = self.buffer[cut:]
            self.pos -= cut
            self.offset += cut
        return out


def yin_pitch(frame: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX) -> float:
    """Simplified YIN (mean difference + threshold) on one frame."""
    x = np.asarray(frame, dtype=np.float64)
    if np.max(np.abs(x)) < 1e-4:
        return float("nan")
    w = len(x)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(w - 1, int(sr / fmin))
    if tau_max <= tau_min:
        return float("nan")

    # cumulative mean normalized difference d'(tau)
    d = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        diff = x[: w - tau] - x[tau:]
        d[tau] = np.mean(diff * diff)
    cum = np.cumsum(d[1:]) / np.arange(1, tau_max + 1)
    d_prime = d[1:] / np.where(cum == 0, 1e-12, cum)

    # first minimum under the threshold, else the global minimum
    threshold = 0.15
    best = None
    for i in range(tau_min - 1, tau_max - 1):
        if d_prime[i] < threshold:
            # parabola around the local minimum
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
    # parabolic interpolation around the minimum (sub-sample precision)
    # points: d_prime[best-1] (tau-1), d_prime[best] (tau), d_prime[best+1] (tau+1)
    if 0 < best < len(d_prime) - 1:
        a, b, c = d_prime[best - 1], d_prime[best], d_prime[best + 1]
        denom = 2.0 * (a + c - 2.0 * b)
        if abs(denom) > 1e-12:
            tau = tau + (a - c) / denom
    return float(sr / tau)
