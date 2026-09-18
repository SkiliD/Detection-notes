"""Étiquetage des figures de notes (ronde → double-croche, pointées, triolets).

Principe : la durée réelle (s) de chaque note est convertie en temps musicaux
via le tempo détecté (durée_en_temps = durée_s × BPM / 60), puis comparée à
un catalogue de figures avec une tolérance relative (le chant humain n'est
pas métronomique).
"""

from __future__ import annotations

from dataclasses import dataclass

# (nom, durée en temps, type MusicXML)
BASES = [
    ("ronde", 4.0, "whole"),
    ("blanche", 2.0, "half"),
    ("noire", 1.0, "quarter"),
    ("croche", 0.5, "eighth"),
    ("double-croche", 0.25, "16th"),
]

DOT_MULT = 1.5        # point : ×1,5
TUPLET_MULT = 2.0 / 3.0  # triolet à 3 contre 2

DEFAULT_TOLERANCE = 0.18  # ±18 % autour de la durée idéale


@dataclass(frozen=True)
class FigureCandidate:
    """Une figure du catalogue."""

    name: str                 # ex. "noire pointée", "triolet de croche"
    beats: float              # durée idéale en temps
    type_: str                # type MusicXML de base
    dots: int                 # 0 ou 1
    tuplet: tuple[int, int] | None  # ex. (3, 2)


def build_candidates() -> list[FigureCandidate]:
    """Catalogue : bases simples, pointées, et triolets (sans cumul point+triolet)."""
    cands: list[FigureCandidate] = []
    for base_name, base_beats, type_ in BASES:
        # simple et pointée
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


# Pénalité de complexité : à erreur quasi égale, on préfère une figure simple,
# puis pointée, puis triolet (une noire à +15 % est une noire en retard, pas
# un triolet de blanche). Le triolet ne gagne que si c'est clairement lui.
_COMPLEXITY_PENALTY = {0: 1.0, 1: 1.06, (3, 2): 1.10}


def classify_duration(beats: float, tolerance: float = DEFAULT_TOLERANCE) -> FigureCandidate | None:
    """Trouve la figure la plus proche d'une durée en temps (None si hors tolérance).

    L'erreur comparée est pondérée par la complexité de la figure (cf.
    _COMPLEXITY_PENALTY) ; au-delà de la tolérance, aucune classe.
    """
    if beats <= 0:
        return None
    best: FigureCandidate | None = None
    best_err = float("inf")
    for c in CANDIDATES:
        err = abs(beats - c.beats) / c.beats * _COMPLEXITY_PENALTY[c.tuplet if c.tuplet else c.dots]
        if err < best_err:
            best, best_err = c, err
    if best is not None and best_err <= tolerance:
        return best
    return None


def label_notes(notes, bpm: float) -> None:
    """Étiquette chaque note en place : note.figure = nom de figure ou None."""
    if not bpm or bpm <= 0:
        return
    for n in notes:
        beats = (n.end - n.start) * bpm / 60.0
        c = classify_duration(beats)
        n.figure = c.name if c is not None else None


def figures_summary(notes) -> dict[str, int]:
    """Comptage des figures détectées (les non classées sont exclues)."""
    counts: dict[str, int] = {}
    for n in notes:
        if n.figure:
            counts[n.figure] = counts.get(n.figure, 0) + 1
    return counts


_SHORT = {"ronde": "R", "blanche": "B", "noire": "N", "croche": "C",
          "double-croche": "DC"}


def short_label(figure: str | None) -> str:
    """Étiquette courte pour l'affichage : 'N.', 'C', '3N' (triolet)…"""
    if not figure:
        return ""
    if figure.startswith("triolet de "):
        base = figure.removeprefix("triolet de ")
        return f"3{_SHORT.get(base, '?')}"
    base = figure.removesuffix(" pointée")
    label = _SHORT.get(base, "?")
    return f"{label}." if "pointée" in figure else label
