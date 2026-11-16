"""Breaths, phrase gaps and lead-in silence as written rests.

The segmenter's job is notes; the gaps it leaves behind are usually MUSIC —
a breath between two phrases, the silence before the first note. This
module turns those gaps into `Rest` objects, classified with the SAME
figure machinery as the notes (`figures.classify_duration`, translated to
the canonical French rest names), so the score can show where the singer
actually breathed instead of chaining noteheads as if the line were
seamless.

Timeline conventions:
- the lead-in silence (t=0 -> first note) IS a rest by default: on the
  score it engraves the empty measures before the melody, which is what a
  real edition does — and it keeps note positions and rests on one shared
  grid (`include_lead_in=False` opts out for gap-only diagnostics);
- the tail AFTER the last note is the end of the TAKE, not the end of the
  music: never a rest (a recording rarely stops where the song does).

Guard rails (a caller must survive any input):
- a gap below `MIN_REST_S` is a segmentation seam, not a breath: no rest
  (the pitch interpolator already fills < 50 ms inside `track_to_notes`);
- rests between overlapping or unordered notes cannot go negative;
- everything degrades to an empty list — `AnalysisResult.rests == []` and
  every consumer must handle that (rests are an addition, never a
  prerequisite of the score).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import GATE_RMS
from .figures import classify_duration
from .grid import BeatGrid
from .observation import VoiceObservation

# Below this a gap is a seam of the segmentation, not a breath. Measured on
# a real take: sung phrases breathe ~0.2-0.5 s, while intra-note seams
# stay under ~70 ms.
MIN_REST_S = 0.070


@dataclass
class Rest:
    """A written silence: a gap between two sung notes."""

    start: float  # seconds
    end: float  # seconds
    figure: str | None = None  # canonical French rest name when classifiable
    breath: bool = False  # an air intake was HEARD inside the gap

    @property
    def duration(self) -> float:
        return self.end - self.start


def classify_rest(beats: float) -> str | None:
    """Rest figure for a duration in beats, or None when off-grid.

    Reuses the note-figure classifier (same tolerance, same complexity
    penalty, triplets excluded — a silence is never inside a triplet
    bracket) and translates the MusicXML type to the rest name — a gap that
    would have been labeled "noire" as a note becomes a "soupir".
    """
    if beats <= 0:
        return None
    c = classify_duration(beats, allow_triplets=False)
    if c is None:
        return None
    return c.rest


# A silence carries a HEARD air intake when its loudest frame clearly rises
# above the take's quiet floor (the 10th percentile of the accent envelope):
# a real breath is a localized bump, a bare musical gap stays near the floor.
# Measured on real takes: audible intakes peak 3-29x above the floor while
# silent gaps stay under ~1.5x; 2x keeps the headroom of the quietest
# confirmable one (1.5x would sit ON the measured silent gap). Floor-relative
# on purpose (ADR 0003's concern is a note-vs-noise SEGMENTER floor — this
# only labels a rest the segmentation already wrote).
BREATH_PEAK_RATIO = 2.0
# A bump must also be LOCAL: the peak clearly dominates the gap's own body
# (a breath is a burst into a silence, not a constant hiss across it).
BREATH_SHAPE_RATIO = 2.0


def detect_breaths(
    rests: list[Rest],
    accent_envelope: np.ndarray | None,
    accent_hop_s: float,
) -> None:
    """Mark IN PLACE the rests that carry a heard air intake.

    The evidence is the accent envelope the observation already carries
    (one smoothed RMS per hop): a rest is a breath when its loudest frame
    rises BREATH_PEAK_RATIO above the take's quiet floor AND its peak
    dominates the gap's body (BREATH_SHAPE_RATIO). Anything undecidable —
    no envelope, an empty rest — stays unmarked: the annotation is an
    addition, never a prerequisite of the rests (same contract as the
    figures).
    """
    if accent_envelope is None or len(accent_envelope) == 0 or accent_hop_s <= 0:
        return
    env = np.asarray(accent_envelope, dtype=float)
    floor = float(np.percentile(env, 10))
    # A zero background still permits an audible bump; numerical ripples do not.
    minimum_peak = BREATH_PEAK_RATIO * floor if floor > 0 else GATE_RMS
    for r in rests:
        i0 = max(0, int(r.start / accent_hop_s))
        i1 = min(len(env), max(i0 + 1, int(np.ceil(r.end / accent_hop_s))))
        body = env[i0:i1]
        if not len(body):
            continue
        peak = float(body.max())
        typical = float(np.median(body))
        r.breath = peak >= minimum_peak and peak >= BREATH_SHAPE_RATIO * max(typical, 1e-9)


def detect_rests(
    notes,
    grid: BeatGrid | None = None,
    include_lead_in: bool = True,
    min_rest_s: float = MIN_REST_S,
    observation: VoiceObservation | None = None,
) -> list[Rest]:
    """Gaps between consecutive notes, as chronological `Rest` objects.

    `grid` labels each rest with its figure (soupir, demi-pause...) when the
    duration fits the figure catalog; without it the rests come back
    unlabeled (the score still engraves them from their durations).
    `observation` adds the breath annotation: a rest with a HEARD air
    intake (its accent envelope rises over the take's floor) is marked
    `breath` — where the singer actually took air, not only where the line
    went silent.

    Guard rails: unordered/overlapping/degenerate notes are tolerated
    (prev_end never goes backwards), and gaps below `min_rest_s` are
    ignored as segmentation seams.
    """
    ordered = sorted(
        (n for n in notes if n is not None and n.end > n.start),
        key=lambda n: n.start,
    )
    rests: list[Rest] = []
    prev_end = 0.0 if include_lead_in else (ordered[0].start if ordered else 0.0)
    for n in ordered:
        if n.start - prev_end >= min_rest_s:
            rests.append(Rest(start=prev_end, end=n.start))
        prev_end = max(prev_end, n.end)
    if grid is not None:
        for r in rests:
            r.figure = classify_rest(grid.beats(r.duration))
    if observation is not None:
        try:
            detect_breaths(
                rests,
                observation.accent_envelope,
                observation.accent_hop / observation.sample_rate,
            )
        except (TypeError, ValueError, AttributeError):
            pass  # the annotation never costs the rests
    return rests


__all__ = [
    "BREATH_PEAK_RATIO",
    "BREATH_SHAPE_RATIO",
    "MIN_REST_S",
    "Rest",
    "classify_rest",
    "detect_breaths",
    "detect_rests",
]
