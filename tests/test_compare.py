"""The live/final comparison policy: ONE net, two readings.

`core.compare` carries the rule shared by the GUI marking,
score_accuracy's live diagnosis, replay measurements and the JSON block:
a `Move` carries its reason, the match window
is the grid sixteenth OR the live's dating lead, and the contour rate is
the health indicator published as is.

The tests pin two distinctions: the match window includes the live
segmenter's dating lead at slow tempos, and an "off1" reason applies only
to a nearby rounding, not a real pitch move.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from v2m.core import takes
from v2m.core.compare import (
    LIVE_DATING_LEAD_S,
    breakdown,
    classify,
    invalidate,
    match,
    match_window,
    read_block,
    write_sidecar,
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


def test_semitone_rounding_requires_an_onset_inside_the_window():
    for midi in [59, 61]:
        for shift in [-10.0, -0.126, 0.126, 10.0]:
            moves = match(
                [FakeNote(0.0, 0.5, 60)],
                [FakeNote(shift, shift + 0.5, midi)],
                FakeGrid(),
            )
            assert moves[0].reason == "moved"
        for shift in [-0.125, 0.125]:
            moves = match(
                [FakeNote(0.0, 0.5, 60)],
                [FakeNote(shift, shift + 0.5, midi)],
                FakeGrid(),
            )
            assert moves[0].reason == "off1"


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


# ---------------------------------------------------------------------
# The block as a whole: built, read, invalidated and archived through
# ONE module (the interface is the test surface).
# ---------------------------------------------------------------------


def test_the_wire_block_round_trips_through_read_block():
    """build -> read: the persisted shape and the parsed value agree."""
    live = [FakeNote(0.0, 0.5, 64)]
    final = [FakeNote(0.0, 0.5, 65)]
    plain = breakdown(live, final, FakeGrid())
    assert "dropped" not in plain and "drain_min" not in plain
    parsed = read_block(plain)
    assert parsed is not None
    assert (parsed.live, parsed.final, parsed.off1, parsed.moved) == (1, 1, 1, 0)
    assert parsed.rate == 1.0
    assert parsed.dropped is None and parsed.drain_min is None
    parsed = read_block(breakdown(live, final, FakeGrid(), dropped=3, drain_min=0.98))
    assert parsed is not None and parsed.dropped == 3 and parsed.drain_min == 0.98


def test_read_block_degrades_per_field_not_per_block():
    """Absent or garbage reads None; one corrupt field alone reads None —
    the tolerant parse has exactly ONE home."""
    assert read_block(None) is None
    assert read_block({}) is None
    assert read_block(["not", "a", "dict"]) is None
    parsed = read_block({"live": 4, "final": 3, "rate": "fast"})
    assert parsed is not None
    assert parsed.live == 4 and parsed.rate is None
    parsed = read_block({"live": 4, "final": 3, "dropped": "many"})
    assert parsed is not None and parsed.dropped is None
    parsed = read_block({"live": 4, "final": 3, "drain_min": "low"})
    assert parsed is not None and parsed.drain_min is None


def test_empty_readings_keep_wire_rate_but_have_no_comparable_rate():
    for live, final in [(3, 0), (0, 3), (0, 0)]:
        block = classify([], live, final)
        assert block["rate"] == 0.0
        parsed = read_block(block)
        assert parsed is not None
        assert parsed.comparable_rate is None
    parsed = read_block({"live": 3, "final": 3, "rate": 0.0})
    assert parsed is not None and parsed.comparable_rate == 0.0
    old = read_block({"rate": 0.4})
    assert old is not None and old.comparable_rate == 0.4


def test_invalidate_drops_only_the_block():
    """A stale decomposition dies; the edit path's own facts stay theirs."""
    result = SimpleNamespace(live_vs_final={"live": 1}, manual_pitch_corrections=False)
    invalidate(result)
    assert result.live_vs_final is None
    assert result.manual_pitch_corrections is False


def test_write_sidecar_archives_the_wire_shape(tmp_path):
    block = {
        "live": 5,
        "final": 4,
        "contours": 2,
        "off1": 1,
        "moved": 1,
        "absorbed": 0,
        "rate": 0.4,
    }
    stem = tmp_path / "live_20261004_120000"
    path = write_sidecar(stem, block)
    assert path is not None and path.endswith(takes.SIDECAR_SUFFIX)
    on_disk = json.loads(
        (tmp_path / "live_20261004_120000.live_vs_final.json").read_text(encoding="utf-8")
    )
    assert on_disk == block


def test_write_sidecar_is_best_effort():
    # an empty block: nothing to archive, no file, None — never an error
    assert write_sidecar("/nonexistent-dir-xyz/take", {}) is None
    # an unwritable folder: the artifacts are already on disk either way
    assert write_sidecar("/nonexistent-dir-xyz/take", {"live": 1}) is None
