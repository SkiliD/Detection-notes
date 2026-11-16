"""figures.py tests: classification whole -> sixteenth, dotted, triplets."""

from v2m.core.figures import (
    classify_duration,
    figures_summary,
    label_notes,
    short_label,
)
from v2m.core.grid import CATALOGUE, BeatGrid
from v2m.core.notes import Note


def _fig(beats: float) -> str:
    """The name of a classified duration — the test fails loud when nothing fits."""
    figure = classify_duration(beats)
    assert figure is not None, f"duration {beats} unclassified"
    return figure.name


BPM = 120.0
GRID = BeatGrid(BPM)  # labeling goes through the Grid, not a bare bpm


def _beats(n: float) -> float:
    """Duration in seconds of n beats at 120 BPM (0.5 s per beat)."""
    return n * 60.0 / BPM


def test_simple_figures():
    # classify_duration takes BEATS, not seconds
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
    """Double-dotted figures exist and capture a NARROW band (~±1%).

    Measured: at light weight (1.12, then 1.30), the double-dotted eighth
    (0.875 beats) captured the human noise around the quarter — a note
    sung at 0.86 beats engraved "croche double-pointée" instead of
    noire, breaking the "±15% keeps its simple figure" pin. The 10.0
    weight shrinks the double-dotted to a ±1% band: a singer does not
    land on a double-dotted by accident, an engraver writes it on
    purpose.
    """
    assert _fig(0.875) == "croche double-pointée"
    assert _fig(0.88) == "croche double-pointée"
    assert _fig(0.86) == "noire"  # human noise: the simple wins
    assert _fig(0.8) == "croche pointée"
    assert _fig(1.75) == "noire double-pointée"
    assert _fig(1.74) == "noire double-pointée"
    assert _fig(1.15) == "noire"


def test_every_catalogue_value_converts_exactly_at_24_divisions():
    """Every catalogue value writes as whole divisions at 24/beat.

    The MusicXML export's `<type>` table is
    DERIVED from the catalogue — a value that does not convert exactly
    (double-dotted sixteenth = 5.25 divisions, dotted 32nd = 4.5) would
    engrave a `<type>` contradicting its `<duration>`.
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
    """Beyond the tolerance (±18% weighted), no class."""
    # 4.8 beats: in the hole between the whole (4, +18%) and the dotted whole (6, -18%)
    assert classify_duration(4.8) is None
    # 10 beats: longer than any figure of the catalogue
    assert classify_duration(10.0) is None


def test_imperfect_human_timing_classified():
    """A human note at ±15% keeps its label (not demoted to a triplet)."""
    assert _fig(0.86) == "noire"
    assert _fig(1.15) == "noire"  # triplet penalty: the quarter wins
    assert _fig(0.53) == "croche"  # same for the eighth vs quarter triplet


def test_triplet_wins_when_clearly_triplet():
    """At exact triplet durations, the triplet genuinely wins."""
    assert _fig(2 / 3) == "triolet de noire"
    assert _fig(4 / 3) == "triolet de blanche"
    assert _fig(1.3) == "triolet de blanche"  # within 2.5% of the triplet


def test_label_notes_on_analysis_like_notes():
    notes = [
        Note(start=0.0, end=_beats(1), midi=60, confidence=0.9),  # quarter
        # Dotted quarter, then quarter triplet.
        Note(start=_beats(1), end=_beats(1) + _beats(1.5), midi=62, confidence=0.9),
        Note(start=_beats(2.5), end=_beats(2.5) + _beats(2 / 3), midi=64, confidence=0.9),
        Note(start=_beats(3.2), end=_beats(3.2) + _beats(0.5), midi=65, confidence=0.9),  # eighth
        Note(start=_beats(4), end=_beats(4) + 0.13, midi=67, confidence=0.9),  # sixteenth
    ]
    label_notes(notes, GRID)
    assert [n.figure for n in notes] == [
        "noire",
        "noire pointée",
        "triolet de noire",
        "croche",
        "double-croche",
    ]


def test_label_notes_falls_back_on_a_sane_tempo():
    """An absent tempo no longer leaves figures empty: the Grid falls back
    to a plausible tempo (decided in one place, `grid.BeatGrid`)."""
    notes = [Note(start=0.0, end=_beats(1), midi=60, confidence=0.9)]
    label_notes(notes, BeatGrid(0.0))
    assert notes[0].figure == "noire"


def test_double_dotted_short_labels_write_the_name_once():
    assert short_label("noire double-pointée") == "N.."
    assert short_label("croche double-pointée") == "C.."
    assert short_label("blanche double-pointée") == "B.."


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
        "N",
        "C.",
        "3N",
        "",
    ]
