"""The intonation summary (workstream 5).

`Note.deviation` is the RESIDUAL deviation (signed semitones) left after
absorbing the take's global offset. This module has a single design
rule: a tendency is only announced if the evidence carries it. An
alternation of opposite deviations is not a singer "accidentally in
tune" — it is noise, and the right answer is to claim nothing.
"""

from __future__ import annotations

from v2m.core.intonation import (
    IN_TUNE_CENTS,
    MIN_CENTS,
    MIN_NOTES,
    summarize,
)
from v2m.core.notes import Note


def _notes(deviations_st: list[float], start_midi: int = 60) -> list[Note]:
    """One note per deviation: `start` ascends so the time order is explicit."""
    return [
        Note(start=i * 0.5, end=i * 0.5 + 0.4, midi=start_midi + i, confidence=0.9, deviation=d)
        for i, d in enumerate(deviations_st)
    ]


def test_a_systematically_flat_take_is_named():
    """Five notes 25 cents flat: the tendency is "flat", and nothing else.

    The (register, position) groups lack the notes to conclude: the
    honest answer there is None, not a mean over two notes.
    """
    report = summarize(_notes([-0.25] * 5))
    assert report["n_notes"] == 5
    assert report["tendency"] == "flat"
    assert report["mean_cents"] == -25.0
    assert report["abs_mean_cents"] == 25.0
    assert report["flat_share"] == 1.0
    assert report["in_tune_share"] == 0.0
    assert report["sharp_share"] == 0.0
    assert all(v is None for v in report["by_register"].values())
    assert report["by_position"]["start"] is None


def test_alternating_deviations_claim_nothing():
    """A back-and-forth of ±30 cents: zero mean, hence NO tendency."""
    report = summarize(_notes([-0.30, 0.30] * 4))
    assert report["tendency"] == "even"
    assert report["mean_cents"] == 0.0
    assert report["sharp_share"] == 0.5
    assert report["flat_share"] == 0.5
    assert report["in_tune_share"] == 0.0
    # ...and no group lean survives either: a group of four alternating
    # notes has enough NOTES but its mean is still zero, so it is silent too
    assert report["by_position"]["start"] is None
    assert report["by_position"]["end"] is None
    assert report["by_register"]["low"] is None


def test_too_few_notes_and_no_notes_are_both_silent():
    """Three notes do not make a habit; zero notes say nothing at all."""
    few = summarize(_notes([-0.30, -0.30, -0.30]))
    assert few["tendency"] is None
    assert few["mean_cents"] == -30.0  # the number is still published...
    assert few["by_register"]["low"] is None  # ...but no habit is claimed
    assert MIN_NOTES > 3

    empty = summarize([])
    assert empty["n_notes"] == 0
    assert empty["mean_cents"] is None
    assert empty["tendency"] is None
    assert empty["by_register"] is None


def test_register_and_position_tendencies_are_read_separately():
    """A low register flat and a high one sharp, then a mid-take drift.

    The register is read by PITCH THIRDS (4/4/4 here), the position by
    time halves: the two habits coexist without merging.
    """
    by_register = summarize(_notes([-0.30] * 4 + [0.0] * 4 + [0.30] * 4))
    assert by_register["by_register"]["low"]["mean_cents"] == -30.0
    assert by_register["by_register"]["high"]["mean_cents"] == 30.0
    # the middle third averages out: no habit named there
    assert by_register["by_register"]["mid"] is None
    # the take-wide mean is zero, so nothing is claimed about the whole take
    assert by_register["tendency"] == "even"

    drifting = summarize(_notes([-0.30] * 6 + [0.30] * 6))
    assert drifting["by_position"]["start"]["mean_cents"] == -30.0
    assert drifting["by_position"]["end"]["mean_cents"] == 30.0


def test_one_pitch_never_invents_several_registers():
    notes = _notes([-0.2] * 4 + [0.2] * 8)
    for note in notes:
        note.midi = 60
    assert all(value is None for value in summarize(notes)["by_register"].values())


def test_register_bands_depend_on_pitch_not_repetition_count_or_order():
    notes = _notes([-0.2] * 12 + [0.2] * 4)
    for index, note in enumerate(notes):
        note.midi = 60 if index < 12 else 72
    bands = summarize(notes)["by_register"]
    assert bands["low"] == {"n": 12, "mean_cents": -20.0}
    assert bands["mid"] is None
    assert bands["high"] == {"n": 4, "mean_cents": 20.0}
    assert summarize(list(reversed(notes)))["by_register"] == bands


def test_the_in_tune_band_is_the_one_the_gui_meter_uses():
    """One scale for the report and the live visual feedback."""
    from v2m.gui.widgets import CENTER_CENTS

    assert CENTER_CENTS == IN_TUNE_CENTS
    report = summarize(_notes([0.05] * 5))  # 5 cents: on the note
    assert report["in_tune_share"] == 1.0
    assert report["tendency"] == "even"
    assert abs(report["mean_cents"]) < MIN_CENTS


def test_a_note_without_a_measurement_is_not_counted_as_in_tune():
    """No reading is not evidence of being on pitch, and not a note either.

    The page's port skips such a note (`centsOf` answers null); counting it as
    0.0 here would report an in-tune share from notes nobody measured.
    """
    from types import SimpleNamespace

    measured = _notes([-0.30] * 4)
    unread = [SimpleNamespace(midi=72), SimpleNamespace(midi=64, deviation=None)]
    report = summarize([*measured, *unread])
    assert report["n_notes"] == 4
    assert report["mean_cents"] == -30.0
    assert report["in_tune_share"] == 0.0
    assert summarize([SimpleNamespace(midi=60)])["n_notes"] == 0
    assert summarize([])["n_notes"] == 0
