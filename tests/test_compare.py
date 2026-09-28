"""The live/final comparison policy: ONE net, two readings.

`core.compare` carries the rule four consumers each read in their own
corner (the GUI marking, score_accuracy's live diagnosis, the AGENTS.md
measures, the JSON block): a `Move` carries its reason, the match window
is the grid sixteenth OR the live's dating lead, and the contour rate is
the health indicator published as is.

The test bites on the two historical defects: a purely "one sixteenth"
window that would outline notes read identically (the live's dating lead
exceeds the sixteenth at slow tempos), and an "off1" reason that would
swallow a real pitch move.
"""

from __future__ import annotations

from dataclasses import dataclass

from v2m.core.compare import (
    LIVE_DATING_LEAD_S,
    Move,
    breakdown,
    classify,
    match,
    match_window,
)


@dataclass
class FakeNote:
    start: float
    end: float
    midi: int


class FakeGrid:
    """A minimal BeatGrid at 120 BPM: one sixteenth = 0.125 s."""

    bpm = 120.0

    def step_seconds(self, per_beat: float = 4.0) -> float:
        return 60.0 / self.bpm / per_beat


def test_window_is_the_grid_step_or_the_dating_lead():
    # at 120 BPM a sixteenth is 0.125 s > the dating lead: the grid wins
    assert match_window(FakeGrid()) == 0.125
    # at a slow tempo the sixteenth exceeds the window... below, at 90 BPM it
    # is 0.1667: still > 0.10; a VERY slow grid (40 BPM) gives 0.375.
    # The case where the lead wins: no grid (fallback 0.12) or a grid
    # finer than 0.10.
    assert match_window(None) >= LIVE_DATING_LEAD_S
    assert match_window(FakeGrid(), window=0.05) == 0.05


def test_identical_readings_produce_no_move():
    live = [FakeNote(0.0, 0.5, 60), FakeNote(0.6, 1.1, 64)]
    final = [FakeNote(0.02, 0.52, 60), FakeNote(0.58, 1.12, 64)]
    assert match(live, final, FakeGrid()) == []
    b = breakdown(live, final, FakeGrid())
    assert b["contours"] == 0 and b["rate"] == 0.0
    assert b["live"] == 2 and b["final"] == 2


def test_a_semitone_neighbour_is_off1_not_moved():
    """The ±1 family: the same plateau read by two roundings — a reason of
    its own, because it is NOT a segmentation error."""
    live = [FakeNote(0.0, 0.5, 64)]
    final = [FakeNote(0.0, 0.5, 65)]
    moves = match(live, final, FakeGrid())
    assert len(moves) == 1 and moves[0].reason == "off1"
    b = breakdown(live, final, FakeGrid())
    assert b["off1"] == 1 and b["moved"] == 0 and b["contours"] == 1
    assert b["rate"] == 1.0


def test_a_beyond_window_onset_is_moved_even_on_the_same_pitch():
    live = [FakeNote(0.0, 0.5, 60)]
    final = [FakeNote(0.30, 0.8, 60)]  # 0.3 s apart: beyond the sixteenth
    moves = match(live, final, FakeGrid())
    assert len(moves) == 1 and moves[0].reason == "moved"
    assert moves[0].onset_shift_s > 0.2


def test_two_semitones_is_moved_not_off1():
    live = [FakeNote(0.0, 0.5, 60)]
    final = [FakeNote(0.0, 0.5, 62)]
    assert match(live, final, FakeGrid())[0].reason == "moved"


def test_a_merge_shows_as_its_moved_parts():
    """Two live notes merged on the final side: each is judged against the
    nearest final note — the table stays honest, part by part."""
    live = [FakeNote(0.0, 0.4, 60), FakeNote(0.5, 0.9, 62)]
    final = [FakeNote(0.0, 1.0, 60)]  # the offline pass merged them
    moves = match(live, final, FakeGrid())
    # the FIRST live note matches the merged final note (same pitch, onset
    # within the window) and is NOT a move; the second (62) has no 62
    # around and is — the merge shows as its moved parts, honestly
    assert len(moves) == 1
    assert moves[0].reason == "moved"
    assert moves[0].live.midi == 62


def test_absorbed_counts_final_notes_without_a_live_counterpart():
    """A glissando the offline reads as one extra note: the breakdown
    publishes it (the live honestly confirms its plateau)."""
    live = [FakeNote(0.0, 0.5, 60)]
    final = [FakeNote(0.0, 0.5, 60), FakeNote(0.7, 1.2, 67)]
    b = breakdown(live, final, FakeGrid())
    assert b["absorbed"] == 1
    assert b["contours"] == 0  # the live note itself matched


def test_empty_sides_never_raise():
    assert match([], [], None) == []
    assert match([FakeNote(0, 1, 60)], [], None) == []
    b = classify([], 0, 0)
    assert b["rate"] == 0.0 and b["contours"] == 0
