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

from .figures import classify_duration
from .grid import BeatGrid

# Below this a gap is a seam of the segmentation, not a breath. Measured on
# a real take: sung phrases breathe ~0.2-0.5 s, while intra-note seams
# stay under ~70 ms.
MIN_REST_S = 0.070

# MusicXML type -> canonical French rest name (user-visible labels are
# French by project convention; `i18n.tr_figure` translates for display).
_REST_BY_TYPE = {
    "whole": "pause",
    "half": "demi-pause",
    "quarter": "soupir",
    "eighth": "demi-soupir",
    "16th": "quart de soupir",
}


@dataclass
class Rest:
    """A written silence: a gap between two sung notes."""

    start: float                  # seconds
    end: float                    # seconds
    figure: str | None = None     # canonical French rest name when classifiable

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
    base = _REST_BY_TYPE[c.type_]
    return f"{base} pointée" if c.dots else base


def detect_rests(
    notes,
    grid: BeatGrid | None = None,
    include_lead_in: bool = True,
    min_rest_s: float = MIN_REST_S,
) -> list[Rest]:
    """Gaps between consecutive notes, as chronological `Rest` objects.

    `grid` labels each rest with its figure (soupir, demi-pause...) when the
    duration fits the figure catalog; without it the rests come back
    unlabeled (the score still engraves them from their durations).

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
    return rests


__all__ = ["MIN_REST_S", "Rest", "classify_rest", "detect_rests"]
