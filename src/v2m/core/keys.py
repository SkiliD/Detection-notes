"""Key detection: Krumhansl-Schmuckler on the pitch-class histogram."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class KeyCandidate:
    tonic: str       # "C", "F#", ...
    mode: str        # "major" | "minor"
    score: float     # Pearson correlation with the profile

    @property
    def label(self) -> str:
        return f"{self.tonic} {self.mode}"


@dataclass
class KeyResult:
    best: KeyCandidate
    alternatives: list[KeyCandidate]  # up to 2 other candidates


def pitch_class_histogram(notes, weights: str = "duration") -> np.ndarray:
    """Histogram of the 12 pitch classes weighted by duration (or confidence)."""
    hist = np.zeros(12)
    for n in notes:
        w = (n.end - n.start) if weights == "duration" else n.confidence
        hist[n.midi % 12] += float(w)
    return hist


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def detect_key(notes, top_k: int = 3) -> KeyResult:
    """Krumhansl-Schmuckler: histogram correlation against 24 maj/min profiles.

    Returns the best candidate plus the runner-ups (up to top_k).
    """
    hist = pitch_class_histogram(notes)
    if hist.sum() <= 0:
        return KeyResult(
            best=KeyCandidate(tonic="C", mode="major", score=0.0),
            alternatives=[],
        )

    cands: list[KeyCandidate] = []
    for shift in range(12):
        # tonic = PC_NAMES[shift]: profile degree j maps to pitch class
        # (shift + j) % 12 -> roll by -shift
        rolled = np.roll(hist, -shift)
        for mode, profile in (("major", KS_MAJ), ("minor", KS_MIN)):
            cands.append(
                KeyCandidate(
                    tonic=PC_NAMES[shift],
                    mode=mode,
                    score=_pearson(rolled, profile),
                )
            )
    cands.sort(key=lambda c: c.score, reverse=True)
    best = cands[0]
    alts = cands[1:top_k]
    return KeyResult(best=best, alternatives=alts)
