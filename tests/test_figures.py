"""Tests figures.py : classification ronde → double-croche, pointées, triolets."""

from v2m.core.figures import (
    classify_duration,
    figures_summary,
    label_notes,
    short_label,
)
from v2m.core.grid import CATALOGUE, BeatGrid
from v2m.core.notes import Note


def _fig(beats: float) -> str:
    """Le nom d'une durée classée — le test échoue fort si rien ne colle."""
    figure = classify_duration(beats)
    assert figure is not None, f"durée {beats} non classée"
    return figure.name

BPM = 120.0
GRID = BeatGrid(BPM)  # l'étiquetage passe par la Grille, pas par un bpm nu


def _beats(n: float) -> float:
    """Durée en secondes de n temps à 120 BPM (0.5 s par temps)."""
    return n * 60.0 / BPM


def test_simple_figures():
    # classify_duration prend des TEMPS (beats), pas des secondes
    assert _fig(4.0) == "ronde"
    assert _fig(2.0) == "blanche"
    assert _fig(1.0) == "noire"
    assert _fig(0.5) == "croche"
    assert _fig(0.25) == "double-croche"


def test_dotted_figures():
    assert _fig(3.0) == "blanche pointée"
    assert _fig(1.5) == "noire pointée"
    assert _fig(0.75) == "croche pointée"


def test_double_dotted_figures_win_only_on_a_deliberate_length():
    """Les double-pointées existent et capturent une bande ETROITE (~±1 %).

    Mesuré : à poids léger (1.12, puis 1.30), la croche double-pointée
    (0.875 temps) capturait le bruit humain autour de la noire — une note
    chantée à 0.86 temps se gravait "croche double-pointée" au lieu de
    noire, cassant l'épingle « ±15 % garde son étiquette simple ». Le poids
    10.0 réduit la double-pointée à une bande ±1 % : un chanteur n'atterrit
    PAS sur une double-pointée par accident, un graveur l'écrit à dessein.
    """
    assert _fig(0.875) == "croche double-pointée"
    assert _fig(0.88) == "croche double-pointée"
    assert _fig(0.86) == "noire"        # bruit humain : la simple gagne
    assert _fig(0.8) == "croche pointée"
    assert _fig(1.75) == "noire double-pointée"
    assert _fig(1.74) == "noire double-pointée"
    assert _fig(1.15) == "noire"


def test_every_catalogue_value_converts_exactly_at_24_divisions():
    """Toute valeur du catalogue s'écrit en divisions entières à 24/temps.

    AGENTS.md (invariant épinglé) : la table `<type>` de l'export MusicXML
    est DÉRIVÉE du catalogue — une valeur qui ne convertit pas exactement
    (double-croche double-pointée = 5.25 divisions, 32ᵉ pointée = 4.5)
    graverait un `<type>` contredisant sa `<duration>`. Refusé à l'entrée :
    deux valeurs candidates avaient été retirées pour ça.
    """
    from v2m.core.grid import MUSICXML_DIVISIONS

    for f in CATALOGUE:
        assert (f.beats * MUSICXML_DIVISIONS) % 1 == 0, (
            f"{f.name} = {f.beats * MUSICXML_DIVISIONS} divisions"
        )


def test_triplet_figures():
    assert _fig(2 / 3) == "triolet de noire"
    assert _fig(1 / 3) == "triolet de croche"
    assert _fig(4 / 3) == "triolet de blanche"


def test_tolerance_rejects_offgrid():
    """Au-delà de la tolérance (±18 % pondéré), aucune classe."""
    # 4.8 temps : dans le trou entre la ronde (4, +18 %) et la ronde pointée (6, -18 %)
    assert classify_duration(4.8) is None
    # 10 temps : plus long que toute figure du catalogue
    assert classify_duration(10.0) is None


def test_imperfect_human_timing_classified():
    """Une note humaine à ±15 % reste bien étiquetée (pas déclassée en triolet)."""
    assert _fig(0.86) == "noire"
    assert _fig(1.15) == "noire"   # pénalité triolet : noire gagne
    assert _fig(0.53) == "croche"  # idem croche vs triolet de noire


def test_triplet_wins_when_clearly_triplet():
    """Aux durées exactes de triolet, c'est bien le triolet qui gagne."""
    assert _fig(2 / 3) == "triolet de noire"
    assert _fig(4 / 3) == "triolet de blanche"
    assert _fig(1.3) == "triolet de blanche"  # à 2,5 % du triolet


def test_label_notes_on_analysis_like_notes():
    notes = [
        Note(start=0.0, end=_beats(1), midi=60, confidence=0.9),      # noire
        Note(start=_beats(1), end=_beats(1) + _beats(1.5), midi=62, confidence=0.9),  # noire pointée
        Note(start=_beats(2.5), end=_beats(2.5) + _beats(2 / 3), midi=64, confidence=0.9),  # triolet noire
        Note(start=_beats(3.2), end=_beats(3.2) + _beats(0.5), midi=65, confidence=0.9),  # croche
        Note(start=_beats(4), end=_beats(4) + 0.13, midi=67, confidence=0.9),  # double-croche
    ]
    label_notes(notes, GRID)
    assert [n.figure for n in notes] == [
        "noire", "noire pointée", "triolet de noire", "croche", "double-croche",
    ]


def test_label_notes_falls_back_on_a_sane_tempo():
    """Un tempo absent ne laisse plus les figures vides : la Grille retombe sur
    un tempo plausible (décidé en un endroit, `grid.BeatGrid`)."""
    notes = [Note(start=0.0, end=_beats(1), midi=60, confidence=0.9)]
    label_notes(notes, BeatGrid(0.0))
    assert notes[0].figure == "noire"


def test_summary_and_short_labels():
    notes = [
        Note(start=0, end=_beats(1), midi=60, confidence=0.9),
        Note(start=_beats(1), end=_beats(2), midi=60, confidence=0.9),
        Note(start=_beats(2), end=_beats(3), midi=60, confidence=0.9),
        Note(start=_beats(3), end=_beats(4), midi=60, confidence=0.9),
    ]
    label_notes(notes, GRID)
    assert figures_summary(notes) == {"noire": 4}
    assert [short_label(f) for f in ("noire", "croche pointée", "triolet de noire", None)] == [
        "N", "C.", "3N", "",
    ]
