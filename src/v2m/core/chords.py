"""Chord recognition from detected notes.

Method: build a duration-weighted pitch-class histogram from the segmented
notes, then score every major/minor/dim/aug triad plus dominant/seventh/minor
seventh triads across all 12 roots, adding a small bonus when the root is the
lowest sounding pitch class (bass doubling supports root identification).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .music import NOTE_NAMES

# Interval structure (semitones above the root) of every supported chord type.
CHORD_TEMPLATES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),        # major
    "m": (0, 3, 7),       # minor
    "dim": (0, 3, 6),     # diminished
    "aug": (0, 4, 8),     # augmented
    "7": (0, 4, 7, 10),   # dominant seventh
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
}

BASS_BONUS = 0.05  # added to the normalized score when root is the bass note


@dataclass(frozen=True)
class ChordCandidate:
    """One scored chord guess (e.g. root 'C', quality '' -> label 'C')."""

    root: str
    quality: str
    score: float  # 0..1 template coverage

    @property
    def label(self) -> str:
        return f"{self.root}{self.quality}"


@dataclass
class ChordResult:
    """Best chord plus the runner-ups (max 3 total, mirroring KeyResult)."""

    best: ChordCandidate
    alternatives: list[ChordCandidate]


def detect_chords(notes, top_k: int = 3) -> ChordResult | None:
    """Global chord over the whole recording (None if fewer than 3 notes)."""
    if len(notes) < 3:
        return None
    return _classify(_histogram(notes), notes, top_k)


def _distinct_pitch_classes(notes) -> int:
    return len({n.midi % 12 for n in notes})


def detect_chord_segments(
    notes,
    segment_s: float = 2.0,
    min_notes: int = 3,
    top_k: int = 1,
) -> list[tuple[float, float, ChordCandidate]]:
    """Per-window chords: [(start_s, end_s, ChordCandidate), ...].

    The timeline is cut into **adjacent** windows of `segment_s` seconds —
    no overlap, so a given moment is never reported twice — and each window
    holding enough notes gets its own classification. A note crossing a
    window boundary belongs to both windows: it does sound in both.
    """
    if len(notes) < min_notes:
        return []
    t_end = max(n.end for n in notes)
    out: list[tuple[float, float, ChordCandidate]] = []
    t = 0.0
    while t < t_end:
        window = [n for n in notes if n.start < t + segment_s and n.end > t]
        res = _classify(_histogram(window), window, top_k) if len(window) >= min_notes else None
        if res is not None:
            out.append((t, min(t + segment_s, t_end), res.best))
        t += segment_s
    return out


def _histogram(notes) -> np.ndarray:
    """12-bin histogram of pitch classes weighted by note duration."""
    hist = np.zeros(12)
    for n in notes:
        hist[n.midi % 12] += float(n.duration)
    return hist


def _classify(histogram: np.ndarray, notes, top_k: int) -> ChordResult | None:
    """Score all templates and return the best chord plus runner-ups."""
    total = histogram.sum()
    if total <= 0 or _distinct_pitch_classes(notes) < 2:
        # a single sustained pitch class is not enough to call a chord
        return None
    weights = histogram / total
    bass_pc = min(notes, key=lambda n: n.midi).midi % 12

    scored: list[ChordCandidate] = []
    for root in range(12):
        for quality, intervals in CHORD_TEMPLATES.items():
            mask = np.zeros(12)
            mask[(root + np.asarray(intervals)) % 12] = 1.0
            # coverage = share of sounding weight inside the template
            coverage = float(weights[mask == 1].sum())
            if coverage < 0.5:  # a chord must explain at least half the music
                continue
            score = coverage + (BASS_BONUS if root == bass_pc else 0.0)
            scored.append(ChordCandidate(NOTE_NAMES[root], quality, round(score, 3)))

    if not scored:
        return None
    scored.sort(key=lambda c: c.score, reverse=True)
    best = scored[0]
    # one alternative per root pitch class (and never the best's own root):
    # "C (Am, Em)" informs the reader, "C (C7, Cmaj7)" would not
    seen_pcs = {_pc_of(best.root)}
    alts = []
    for c in scored[1:]:
        pc = _pc_of(c.root)
        if pc not in seen_pcs:
            seen_pcs.add(pc)
            alts.append(c)
        if len(alts) >= max(0, top_k - 1):
            break
    return ChordResult(best=best, alternatives=alts)


def _pc_of(name: str) -> int:
    return NOTE_NAMES.index(name)


def detect_chord_progression(
    notes,
    bpm: float,
    meter,
    num_measures: int,
    min_notes: int = 2,
    top_k: int = 1,
) -> list[tuple[int, int, ChordCandidate]]:
    """Chords aligned on measures: [(m0, m1, chord), ...] (1-based, m1 exclusive).

    Each measure is classified from the notes sounding inside it (weight =
    overlap with the measure span, so a tied note counts where it sounds).
    Consecutive measures sharing the same best chord are merged into one
    span — the output reads like a chord chart (C | C | F | G).
    """
    if not notes or bpm <= 0 or num_measures <= 0:
        return []
    if meter is None:
        measure_s = 4 * 60.0 / bpm
    else:
        measure_s = meter.measure_seconds(bpm)
    if measure_s <= 0:
        return []

    raw: list[tuple[int, int, ChordCandidate]] = []
    for m in range(1, num_measures + 1):
        m0 = (m - 1) * measure_s
        m1 = m * measure_s
        inside = []
        for n in notes:
            ov = min(n.end, m1) - max(n.start, m0)
            if ov > 0:
                # clipped clone so the histogram weights the overlap only
                inside.append(type(n)(start=m0, end=m0 + ov, midi=n.midi, confidence=n.confidence))
        if len(inside) < min_notes:
            continue
        res = _classify(_histogram(inside), inside, top_k)
        if res is not None:
            raw.append((m, m + 1, res.best))

    # merge consecutive equal chords into a readable progression
    merged: list[tuple[int, int, ChordCandidate]] = []
    for m0, m1, c in raw:
        if merged and merged[-1][2].label == c.label and merged[-1][1] == m0:
            merged[-1] = (merged[-1][0], m1, c)
        else:
            merged.append((m0, m1, c))
    return merged
