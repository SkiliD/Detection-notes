"""Note figure labeling (whole -> sixteenth, dotted, triplets).

Method: each note's real duration (s) is converted to musical beats via the
detected tempo (beats = duration_s * BPM / 60), then matched against a
catalog of figures with a relative tolerance (human singing is not a
metronome).
"""

from __future__ import annotations

from dataclasses import dataclass

# (name, duration in beats, MusicXML type)
BASES = [
    ("ronde", 4.0, "whole"),
    ("blanche", 2.0, "half"),
    ("noire", 1.0, "quarter"),
    ("croche", 0.5, "eighth"),
    ("double-croche", 0.25, "16th"),
]

DOT_MULT = 1.5           # dotted: x1.5
TUPLET_MULT = 2.0 / 3.0  # triplet: 3 against 2

DEFAULT_TOLERANCE = 0.18  # +/-18% around the ideal duration


@dataclass(frozen=True)
class FigureCandidate:
    """One entry of the figure catalog."""

    name: str                 # e.g. "noire pointée", "triolet de croche"
    beats: float              # ideal duration in beats
    type_: str                # base MusicXML type
    dots: int                 # 0 or 1
    tuplet: tuple[int, int] | None  # e.g. (3, 2)


def build_candidates() -> list[FigureCandidate]:
    """Catalog: plain, dotted, and triplet figures (no dot+triplet combos)."""
    cands: list[FigureCandidate] = []
    for base_name, base_beats, type_ in BASES:
        # plain and dotted
        for dots, mult in ((0, 1.0), (1, DOT_MULT)):
            name = f"{base_name} pointée" if dots else base_name
            cands.append(FigureCandidate(name, base_beats * mult, type_, dots, None))
        # triolet (simple uniquement)
        cands.append(
            FigureCandidate(
                f"triolet de {base_name}", base_beats * TUPLET_MULT, type_, 0, (3, 2)
            )
        )
    return cands


CANDIDATES = build_candidates()


# Complexity penalty: at near-equal error, prefer a plain figure, then a
# dotted one, then a triplet (a quarter at +15% is a late quarter, not a
# half triplet). A triplet only wins when it is clearly the right answer.
_COMPLEXITY_PENALTY = {0: 1.0, 1: 1.06, (3, 2): 1.10}


def classify_duration(
    beats: float,
    tolerance: float = DEFAULT_TOLERANCE,
    allow_triplets: bool = True,
) -> FigureCandidate | None:
    """Find the figure closest to a duration in beats (None if out of tolerance).

    The compared error is weighted by figure complexity (see
    _COMPLEXITY_PENALTY); beyond the tolerance, no class is assigned.
    `allow_triplets=False` excludes triplet figures: rests use it, since a
    written silence between phrases never lives inside a triplet bracket —
    and at some tempos a plain beat classifies as a triplet before the
    plain figure (measured: 1.17 beats -> "triolet de blanche" at 13.5%
    vs "noire" at 16.9%), which would mislabel the rest.
    """
    if beats <= 0:
        return None
    best: FigureCandidate | None = None
    best_err = float("inf")
    for c in CANDIDATES:
        if c.tuplet is not None and not allow_triplets:
            continue
        err = abs(beats - c.beats) / c.beats * _COMPLEXITY_PENALTY[c.tuplet if c.tuplet else c.dots]
        if err < best_err:
            best, best_err = c, err
    if best is not None and best_err <= tolerance:
        return best
    return None


def label_notes(notes, bpm: float) -> None:
    """Label each note in place: note.figure = figure name or None."""
    if not bpm or bpm <= 0:
        return
    for n in notes:
        beats = (n.end - n.start) * bpm / 60.0
        c = classify_duration(beats)
        n.figure = c.name if c is not None else None


def figures_summary(notes) -> dict[str, int]:
    """Count detected figures (unclassified notes are excluded)."""
    counts: dict[str, int] = {}
    for n in notes:
        if n.figure:
            counts[n.figure] = counts.get(n.figure, 0) + 1
    return counts


_SHORT = {"ronde": "R", "blanche": "B", "noire": "N", "croche": "C",
          "double-croche": "DC"}


def short_label(figure: str | None) -> str:
    """Short display label: 'N.', 'C', '3N' (triplet)..."""
    if not figure:
        return ""
    if figure.startswith("triolet de "):
        base = figure.removeprefix("triolet de ")
        return f"3{_SHORT.get(base, '?')}"
    base = figure.removesuffix(" pointée")
    label = _SHORT.get(base, "?")
    return f"{label}." if "pointée" in figure else label
