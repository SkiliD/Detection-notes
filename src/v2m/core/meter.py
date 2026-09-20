"""Meter detection from note onsets.

Two stages, in this order:

1. **3/4 vs 4/4 from accents**: `Note.strength` (onset loudness relative to
   the take, see notes.compute_onset_strengths) separates a waltz (every
   third beat accented) from common time. Returned only when the contrast
   between the two hypotheses is clear, otherwise nothing is claimed.
2. **simple vs compound from the grid**: with no clear accent, the ternary
   grid (dotted-quarter beats and their offbeats) must both carry the onsets
   and beat the simple grid that would otherwise explain them; a simple
   meter renders those onsets off-beat. This is the case that truly breaks a
   fixed 4/4 rendering.

2/4, 3/4 and 4/4 share the same quarter grid: without convincing accents
(and, for 2/4, without bar-level context) detection returns None and the
score stays in 4/4 by default; a caller may still pass an explicit Meter to
`export_result` / `save_musicxml`.

This module DETECTS; `grid.py` owns the `Meter` value object and the
arithmetic that goes with it (seconds <-> beats, measures, positions). The
detector runs upstream of the grid — it takes the scalar tempo — so its own
beat period comes from a bpm-only `BeatGrid`.

Accent evidence wins over the ternary grid, because a 3/4 waltz also aligns
with that grid (beats 2-3 look like ternary offbeats) while the reverse is
not true.
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

def detect_meter(notes, bpm: float) -> Meter | None:
    """Detect the meter: accents first (3/4 vs 4/4), then ternary (6/8).

    Stage 1 - simple meters: a waltz accents every 3rd beat, common time
    every 4th; compare per-phase mean onset strengths and require a clear
    contrast.
    Stage 2 - compound: with no clear simple accent, enough onsets on a
    ternary grid -> 6/8 (a steady 6/8 melody sits on dotted-quarter beats
    and offbeats but carries no 3-beat accent pattern).
    Accent evidence wins because a 3/4 waltz ALSO aligns with the ternary
    grid (beats 2-3 look like ternary offbeats) while the reverse is false.
    """
    if not notes or bpm <= 0:
        return None
    if len(notes) < MIN_NOTES:
        return None

    simple = _detect_simple_by_accents(notes, bpm)
    if simple is not None:
        return simple
    return _detect_compound([n.start for n in notes], bpm)


def _align_share(starts: list[float], slot_s: float, tol_rel: float = 0.20) -> float:
    """Share of onsets landing on a grid of period `slot_s` (+/-20% of it)."""
    tol = tol_rel * slot_s
    on_grid = 0
    for t in starts:
        r = t % slot_s
        if min(r, slot_s - r) <= tol:
            on_grid += 1
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

    def best_phase(n: int) -> float | None:
        """Mean strength of downbeats for the best phase, None if too few."""
        counts = [0] * n
        sums = [0.0] * n
        for i, bi in enumerate(beats_idx):
            counts[bi % n] += 1
            sums[bi % n] += strengths[i]
        # a phase must be hit by at least MIN_DOWNBEATS onsets to mean anything
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
    return Meter(4, 4, round(1.0 - contrast, 2))


