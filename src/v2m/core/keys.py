"""Key detection: Krumhansl-Schmuckler on the pitch-class histogram."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .music import NOTE_NAMES  # the one pitch-class name table (the music seam)
from .notes import AMBIGUOUS_DEV_ST  # the ambiguity threshold lives with its producer

KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# The closed set of key modes, each with its Krumhansl profile. One writer,
# one vocabulary: detect_key iterates this mapping, and the i18n guard tests
# pin tr_key_label's display table against the same keys.
KEY_PROFILES: dict[str, np.ndarray] = {
    "major": KS_MAJ,
    "minor": KS_MIN,
}


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
        # tonic = NOTE_NAMES[shift]: profile degree j maps to pitch class
        # (shift + j) % 12 -> roll by -shift
        rolled = np.roll(hist, -shift)
        for mode, profile in KEY_PROFILES.items():
            cands.append(
                KeyCandidate(
                    tonic=NOTE_NAMES[shift],
                    mode=mode,
                    score=_pearson(rolled, profile),
                )
            )
    cands.sort(key=lambda c: c.score, reverse=True)
    best = cands[0]
    alts = cands[1:top_k]
    return KeyResult(best=best, alternatives=alts)


def scale_pitch_classes(key: KeyCandidate) -> set[int]:
    """The key's 7 diatonic pitch classes (major or natural minor)."""
    steps = (0, 2, 4, 5, 7, 9, 11) if key.mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    tonic = NOTE_NAMES.index(key.tonic)
    return {(tonic + s) % 12 for s in steps}


def snap_to_key(notes, key: KeyCandidate, dev_attr: str = "deviation") -> int:
    """Pull the ambiguous notes toward the detected key's scale degrees.

    Musical context, not a blind chromatic round — two resolutions, both
    decided by the note's residual deviation (the signed mistune the global
    offset could not absorb):

    - a note whose pitch class IS in the scale is a scale degree: past
      ±`AMBIGUOUS_DEV_ST` of mistune it is re-anchored on its degree (the
      deviation is absorbed — the key says this degree, the lean was vocal
      noise). Below the threshold the shade stays (it is expression).
    - a note OUTSIDE the scale sung ambiguously (|deviation| >= threshold)
      resolves to the NEIGHBOUR THE PITCH LEANS TOWARD, which for a 7-note
      scale is always diatonic (every chromatic pitch class sits between
      two scale degrees — a property, not a luck). A flat F# in C major
      becomes F; a sharp one stays itself: the lean, not the label, names
      the intent.

    One pass, ascending time. `dev_attr` names the deviation attribute
    (`deviation` on `Note`, `dev` on `LiveNote`); notes without it are
    skipped. Returns the number of changed notes (tests assert the pull,
    not just the absence of crash).
    """
    if key is None or not notes:
        return 0
    scale = scale_pitch_classes(key)
    changed = 0
    for n in notes:
        dev = getattr(n, dev_attr, None)
        if dev is None or dev != dev:
            continue  # no deviation info, or NaN: leave it
        if -AMBIGUOUS_DEV_ST < dev < AMBIGUOUS_DEV_ST:
            continue  # not ambiguous: a chromatic shade keeps its semitone
        lean = 1 if dev > 0 else -1
        if n.midi % 12 in scale:
            setattr(n, dev_attr, 0.0)  # re-anchored on its degree
            changed += 1
            continue
        target = n.midi + lean
        if target % 12 in scale:
            n.midi = target
            setattr(n, dev_attr, 0.0)
            changed += 1
    return changed
