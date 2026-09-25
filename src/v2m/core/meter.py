"""Meter detection from note onsets.

A dotted pulse (see `reads_dotted_pulse`) skips straight to the ternary
grid. Otherwise, two stages, in this order:

1. **3/4 vs 4/4 from accents** (`_detect_simple_by_accents`):
   `Note.strength` (onset loudness relative to the take, see
   onsets.compute_onset_strengths) separates a waltz (every third beat
   accented) from common time. Returned only when the contrast between the
   two hypotheses is clear, otherwise nothing is claimed.
2. **simple vs compound from the grid** (`_detect_compound`): with no clear
   accent, the ternary grid (dotted-quarter beats and their offbeats) must
   both carry the onsets and beat the simple grid that would otherwise
   explain them; a simple meter renders those onsets off-beat. This is the
   case that truly breaks a fixed 4/4 rendering.

2/4 is decided INSIDE the accents stage: a strength pattern with period 2
(the march, strong/weak alternating) reaches the 4/4 verdict like any
other (its strong beats sit on phases 0 and 2 of the quarter grid), but
the SHAPE of its 4-phase profile separates it from a genuine 4/4 — the
march alternates (both even phases strong, both odd ones weak) while a
4/4 accents one phase only. This is the strength period breaking the
nested-grid tie that onsets alone cannot (2/4's bar is half a 4/4's). The
strong beats must lead (start at phase 0): a backbeat stays 4/4, whose
bars would otherwise begin on the weak beat — the grid has no bar-phase
parameter. Without a clear accent, 3/4 and 4/4 return None and the score
stays in 4/4 by default; a caller may still pass an explicit Meter to
`export_result` / `save_musicxml`.

This module DETECTS; `grid.py` owns the `Meter` value object and the
arithmetic that goes with it (seconds <-> beats, measures, positions). The
detector runs upstream of the grid — it takes the scalar tempo — so its own
beat period comes from a bpm-only `BeatGrid`.

Accent evidence wins over the ternary grid, because a 3/4 waltz also aligns
with that grid (beats 2-3 look like ternary offbeats) while the reverse is not true.
A dotted pulse skips the accent stage entirely: its onsets straddle the
quarter grid that stage assumes, and the dotted beats land on two of its
phases — a false simple meter instead of the ternary one.
"""

from __future__ import annotations

from .grid import BeatGrid, Meter

COMPOUND_ALIGN_THRESHOLD = 0.75  # share of onsets on the ternary grid
COMPOUND_MARGIN = 0.15           # ... and this much better than the simple grid
BPM_SCAN_REL = 0.08             # tolerated tempo deviation (+/-8% scan)
BPM_SCAN_STEP = 0.01            # scan step, relative (1%)
MIN_NOTES = 6                   # below: no conclusion
MIN_DOWNBEATS = 2               # accents needed per hypothesis
ACCENT_CONTRAST = 0.15          # required strength gap between 3/4 and 4/4
ALTERNATE_SHARE = 0.75          # period-2 strong/weak agreement: dotted pulse or march
MINORITY_SHARE = 0.30           # the weaker label must still be a real party


def _grid_offset(t: float, slot_s: float) -> float:
    """Distance (s) from t to the nearest slot of a grid anchored at 0."""
    r = t % slot_s
    return min(r, slot_s - r)


def reads_dotted_pulse(notes) -> bool:
    """True when the onset strengths alternate strong/weak with period 2.

    The signature of compound stress (strong dotted beat, weak offbeat):
    a quarter pulse accents on 3- or 4-beat periods instead, and a flat
    pulse has no alternation at all. Labels come from the take-relative
    mean, so the test works on the offline and the live (running-max)
    normalizations alike. Two decisions read it: estimate_bpm_live treats
    the median gap as a DOTTED beat (the true quarter is 2/3 of it — the
    octave fold can only repair ratios of 2, never 1.5), and detect_meter
    sends a dotted pulse straight to the ternary grid — the quarter-grid
    accent stage would read the dotted beats' phases as a waltz.
    """
    strengths = [getattr(n, "strength", 0.0) for n in notes]
    if len(strengths) < 4:
        return False
    mean = sum(strengths) / len(strengths)
    labels = [1.0 if s >= mean else 0.0 for s in strengths]
    if 0.0 not in labels or 1.0 not in labels:
        return False  # flat pulse: nothing alternates
    agree = sum(1 for a, b in zip(labels[2:], labels) if a == b)
    if agree / len(labels[2:]) < ALTERNATE_SHARE:
        return False
    # a constant run self-agrees at lag 2: a take with ONE strong note (a
    # lead-in artifact on a constant-amplitude signal, demo.wav measured)
    # and the rest flat scored 13/13 "agreement" and armed the dotted
    # reading. A real alternation is near 50/50 — the weaker label must
    # be a real party of the pulse.
    minority = min(labels.count(1.0), labels.count(0.0)) / len(labels)
    return minority >= MINORITY_SHARE


def dotted_pulse_tempo(notes, bpm: float) -> float:
    """Tempo corrected for a dotted pulse, `bpm` unchanged otherwise.

    When `reads_dotted_pulse` fires, the onset-interval median was a DOTTED
    beat and the true quarter sits at 2/3 of it (×1.5 — the octave fold only
    repairs ratios of 2). The live estimate (estimate_bpm_live) and the
    final analysis (pipeline, after refine_bpm_with_onsets) both consume
    this one correction so the same take publishes the same tempo.
    """
    return bpm * 1.5 if reads_dotted_pulse(notes) else bpm


def detect_meter(notes, bpm: float) -> Meter | None:
    """The module docstring holds the contract; this code follows its order.

    Returns None when the evidence is missing or unclear: the caller keeps
    the rendered 4/4 default (or passes an explicit Meter to the export).
    """
    if not notes or bpm <= 0:
        return None
    if len(notes) < MIN_NOTES:
        return None

    starts = [n.start for n in notes]
    if reads_dotted_pulse(notes):
        # compound stress: the accents stage assumes a quarter grid and
        # would misread the dotted beats' phases — the ternary grid decides
        compound = _detect_compound(starts, bpm)
        if compound is not None:
            return compound
    simple = _detect_simple_by_accents(notes, bpm)
    if simple is not None:
        return simple
    return _detect_compound(starts, bpm)


def _align_share(starts: list[float], slot_s: float, tol_rel: float = 0.20) -> float:
    """Share of onsets landing on a grid of period `slot_s` (+/-20% of it)."""
    tol = tol_rel * slot_s
    on_grid = sum(1 for t in starts if _grid_offset(t, slot_s) <= tol)
    return on_grid / len(starts)


def _best_alignment(starts: list[float], base_slot_s: float) -> float:
    """Best grid coherence over the tempo scan (+/-BPM_SCAN_REL).

    The detected tempo drifts by a few percent; a slightly stretched grid
    must be allowed to win, otherwise the comparison below would be unfair
    to one hypothesis and not the other.
    """
    best = 0.0
    k = 0
    while k * BPM_SCAN_STEP <= BPM_SCAN_REL + 1e-9:
        for rel in ((1.0 + k * BPM_SCAN_STEP, 1.0 - k * BPM_SCAN_STEP) if k else (1.0,)):
            best = max(best, _align_share(starts, base_slot_s / rel))
        k += 1
    return best


def _detect_compound(starts: list[float], bpm: float) -> Meter | None:
    """6/8 when the ternary grid explains the onsets and the simple one does not.

    Alignment alone is not enough: a plain 4/4 line sung on a metronome
    scores ~0.7 on the ternary grid too (its onsets are multiples of a
    dotted beat by accident). The decision is therefore COMPARATIVE — the
    ternary grid must beat the simple (quarter) grid by COMPOUND_MARGIN —
    which is what actually separates a 6/8 (1.00 vs 0.50 measured on a
    reference phrase) from a steady 4/4 (0.71 vs 0.86).
    """
    beat = BeatGrid(bpm).seconds(1.0)
    ternary = _best_alignment(starts, 1.5 * beat)
    if ternary < COMPOUND_ALIGN_THRESHOLD:
        return None
    simple = _best_alignment(starts, beat)
    if ternary - simple < COMPOUND_MARGIN:
        return None
    return Meter(6, 8, round(ternary, 2))


def _detect_simple_by_accents(notes, bpm: float) -> Meter | None:
    """3/4 vs 4/4 from relative onset strengths (Note.strength).

    For each hypothesis, the downbeat phase is the beat offset whose beats
    carry the highest mean strength; the contrast between the two
    hypotheses decides. Returns None when the evidence is not clear.
    """
    p = BeatGrid(bpm).seconds(1.0)
    beats_idx = [int(round(n.start / p)) for n in notes]
    strengths = [n.strength for n in notes]

    def _phase_means(n: int) -> list[float] | None:
        """Mean strength per phase; None when a phase has too few onsets."""
        counts = [0] * n
        sums = [0.0] * n
        for i, bi in enumerate(beats_idx):
            counts[bi % n] += 1
            sums[bi % n] += strengths[i]
        # a phase must be hit by at least MIN_DOWNBEATS onsets to mean anything
        if min(counts) < MIN_DOWNBEATS:
            return None
        return [sums[j] / counts[j] for j in range(n)]

    def best_phase(n: int) -> float | None:
        """Mean strength of downbeats for the best phase, None if too few.

        Per-phase validity: a short take may leave a phase nearly empty —
        that must not void the hypothesis whose strongest phase is well
        supported (the 2/4 parity read below is the one needing ALL
        phases, and it asks `_phase_means` directly).
        """
        counts = [0] * n
        sums = [0.0] * n
        for i, bi in enumerate(beats_idx):
            counts[bi % n] += 1
            sums[bi % n] += strengths[i]
        valid = [(sums[j] / counts[j], counts[j]) for j in range(n) if counts[j] >= MIN_DOWNBEATS]
        if not valid:
            return None
        return max(s for s, _ in valid)

    s3 = best_phase(3)
    s4 = best_phase(4)
    if s3 is None or s4 is None:
        return None
    contrast = s3 - s4
    if abs(contrast) < ACCENT_CONTRAST:
        return None  # accents do not separate the two grids
    if contrast > 0:
        return Meter(3, 4, round(1.0 + contrast, 2))
    # 4/4 would be claimed — but a strength pattern with PERIOD 2 produces
    # the same verdict (its strong beats sit on phases 0 and 2, so the
    # phase-0 mean is high too). The SHAPE of the 4-phase profile separates
    # them, on two conditions: (a) the march ALTERNATES — both even phases
    # strong, both odd ones weak — while a downbeat-only 4/4 leaves its
    # beat-3 phase as weak as the offbeats (the min/max gap below fails);
    # (b) its two strong phases are COMPARABLE — a genuine 4/4 carries a
    # hierarchy on them (primary beat 1, secondary beat 3), and that gap is
    # the 4-beat signature. Equal strong/weak pairs are the march/polka
    # convention and write 2/4.
    means4 = _phase_means(4)
    if means4 is not None:
        gap = min(means4[0], means4[2]) - max(means4[1], means4[3])
        hierarchy = abs(means4[0] - means4[2])
        if gap >= ACCENT_CONTRAST and hierarchy < ACCENT_CONTRAST:
            return Meter(2, 4, round(1.0 + gap, 2))
    return Meter(4, 4, round(1.0 - contrast, 2))


