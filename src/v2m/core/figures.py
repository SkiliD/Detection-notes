"""Note figure labeling (whole -> sixteenth, dotted, triplets).

Method: each note's real duration (s) is converted to musical beats via the
detected tempo (beats = duration_s * BPM / 60), then matched against a
catalog of figures with a relative tolerance (human singing is not a
metronome).
"""

from __future__ import annotations

from .grid import BeatGrid, CATALOGUE, Figure

DEFAULT_TOLERANCE = 0.18  # +/-18% around the ideal duration


def classify_duration(
    beats: float,
    tolerance: float = DEFAULT_TOLERANCE,
    allow_triplets: bool = True,
) -> Figure | None:
    """Find the figure closest to a duration in beats (None if out of tolerance).

    The compared error is weighted by each catalogue entry's complexity;
    beyond the tolerance, no class is assigned.
    `allow_triplets=False` excludes triplet figures: rests use it, since a
    written silence between phrases never lives inside a triplet bracket —
    and at some tempos a plain beat classifies as a triplet before the
    plain figure (measured: 1.17 beats -> "triolet de blanche" at 13.5%
    vs "noire" at 16.9%), which would mislabel the rest.
    """
    if beats <= 0:
        return None
    best: Figure | None = None
    best_err = float("inf")
    for c in CATALOGUE:
        if c.export_only or (c.tuplet is not None and not allow_triplets):
            continue
        err = abs(beats - c.beats) / c.beats * c.weight
        if err < best_err:
            best, best_err = c, err
    if best is not None and best_err <= tolerance:
        return best
    return None


def label_notes(notes, grid: BeatGrid) -> None:
    """Label each note in place: note.figure = figure name or None.

    `grid` is the take's `grid.BeatGrid`: the duration in beats is its
    business, not this module's (the conversion is written once, there).
    """
    for n in notes:
        c = classify_duration(grid.beats(n.duration))
        n.figure = c.name if c is not None else None


def figures_summary(notes) -> dict[str, int]:
    """Count detected figures (unclassified notes are excluded)."""
    counts: dict[str, int] = {}
    for n in notes:
        if n.figure:
            counts[n.figure] = counts.get(n.figure, 0) + 1
    return counts


_SHORT = {"ronde": "R", "blanche": "B", "noire": "N", "croche": "C",
          "double-croche": "DC", "trente-deuxième": "32"}


def short_label(figure: str | None) -> str:
    """Short display label: 'N.', 'C', '3N' (triplet), 'N..' (double dotted)..."""
    if not figure:
        return ""
    if figure.startswith("triolet de "):
        base = figure.removeprefix("triolet de ")
        return f"3{_SHORT.get(base, '?')}"
    dots = 2 if "double-pointée" in figure else 1 if "pointée" in figure else 0
    base = figure.removesuffix(" pointée").removesuffix(" double-pointée")
    label = _SHORT.get(base, "?")
    return f"{label}." * dots if dots else label
