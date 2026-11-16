"""Regression tests for edits to an analyzed take."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from v2m.core import edits as edits_module
from v2m.core.edits import (
    CURRENT_ACTION,
    EditHistory,
    correct_note_pitch,
    delete_notes,
    move_note,
    snapshot_state,
    splice_phrase,
)
from v2m.core.notes import Note
from v2m.core.pipeline import AnalysisResult
from v2m.core.tempo import TempoResult


def _result(*midis: int) -> AnalysisResult:
    """A small analyzed take, one note per semitone argument."""
    notes = [Note(0.5 * i + 0.1, 0.5 * i + 0.5, midi, 0.9) for i, midi in enumerate(midis)]
    return AnalysisResult(
        source="take.wav",
        duration_s=0.5 * len(notes) + 0.2,
        notes=notes,
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 2.0, 0.5)),
        key=None,
        pitch=None,
    )


def test_phrase_splice_trims_silence_and_moves_later_notes():
    notes = [
        Note(0.5, 1.0, 60, 0.9),
        Note(1.5, 2.0, 62, 0.9),
        Note(2.5, 3.0, 64, 0.9),
    ]
    source = np.arange(100, dtype=np.float32)
    sung = np.arange(70, dtype=np.float32)
    replacement = [Note(1.0, 6.0, 67, 0.9)]

    edited, audio, delta = splice_phrase(notes, source, replacement, sung, 1, 1, sr=10)

    assert len(audio) == 145
    assert audio[15:65].tolist() == sung[10:60].tolist()
    assert [note.midi for note in edited] == [60, 67, 64]
    assert edited[1].start == 1.5 and edited[1].end == 6.5
    assert edited[2].start == 7.0 and edited[2].end == 7.5
    assert delta == 4.5


def test_note_correction_preserves_timing_and_refreshes_derived_grid():
    notes = [
        Note(0.0, 0.4, 60, 0.9),
        Note(0.5, 0.9, 64, 0.9),
        Note(1.0, 1.4, 67, 0.9),
    ]
    result = AnalysisResult(
        source="take.wav",
        duration_s=1.5,
        notes=notes,
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 1.5, 0.5)),
        key=None,
    )
    before = [(note.start, note.end) for note in notes]

    correct_note_pitch(result, 1, 65)

    assert [(note.start, note.end) for note in result.notes] == before
    assert result.notes[1].midi == 65
    assert result.grid is not None
    assert result.note_positions is not None
    assert result.num_measures >= 1
    assert result.observation is None
    assert result.live_vs_final is None
    assert result.manual_pitch_corrections is True


def test_note_correction_invalidates_stale_pitch_evidence():
    stale: Any = object()  # a placeholder the correction must invalidate
    result = AnalysisResult(
        source="take.wav",
        duration_s=1.0,
        notes=[Note(0.1, 0.8, 60, 0.9)],
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 1.0, 0.5)),
        key=None,
        pitch=stale,
        observation=stale,
        live_vs_final={"live": 1, "final": 1},
    )

    correct_note_pitch(result, 0, 62)

    assert result.observation is None
    assert result.pitch is None
    assert result.live_vs_final is None
    assert result.manual_pitch_corrections


def test_move_note_drags_one_bar_and_rebuilds_the_facts():
    """A roll drag lands as one in-place edit: the onset/duration/pitch
    move together, every note-derived fact rebuilds, and the correction
    flag travels (the audio no longer describes the score)."""
    result = _result(60, 64)
    before_key = result.key

    move_note(result, 1, start_s=0.7, end_s=1.1, midi=62)

    moved = result.notes[1]
    assert (moved.start, moved.end, moved.midi) == (0.7, 1.1, 62)
    assert moved.deviation == 0.0
    assert result.notes[0].midi == 60  # the neighbours hold
    assert result.notes[0].start == 0.1 and result.notes[0].end == 0.5
    assert result.manual_pitch_corrections is True
    assert result.key is not before_key  # the derived facts rebuilt


def test_move_note_refuses_overlaps_and_absurd_edges():
    """The neighbours bound the drag: a note slides inside its own gap,
    never over the previous or the next one; a zero length, a negative
    onset and an off-piano pitch are refused."""
    result = _result(60, 64, 67)

    with pytest.raises(ValueError, match="preceding"):
        move_note(result, 1, start_s=0.3, end_s=0.9)
    with pytest.raises(ValueError, match="following"):
        move_note(result, 1, start_s=0.7, end_s=1.7)
    with pytest.raises(ValueError, match="duration"):
        move_note(result, 1, start_s=0.5, end_s=0.5)
    with pytest.raises(ValueError, match="before the take"):
        move_note(result, 0, start_s=-0.2, end_s=0.5)
    with pytest.raises(ValueError, match="piano"):
        move_note(result, 1, midi=12)
    with pytest.raises(IndexError):
        move_note(result, 9, start_s=0.5, end_s=0.9)
    # nothing above changed the take
    assert [(n.start, n.end, n.midi) for n in result.notes] == [
        (0.1, 0.5, 60),
        (0.6, 1.0, 64),
        (1.1, 1.5, 67),
    ]


def test_delete_notes_removes_only_the_selected_rows_and_marks_checks_unavailable():
    """A deletion touches the SCORE only: survivors keep their timing, and
    the audio-based checks go unavailable exactly like a pitch correction
    (the notes no longer describe the recording)."""
    stale: Any = object()
    result = _result(60, 64, 67)
    result.observation = stale
    result.live_vs_final = {"live": 1, "final": 1}

    delete_notes(result, [1])

    assert [n.midi for n in result.notes] == [60, 67]
    assert [n.start for n in result.notes] == pytest.approx([0.1, 1.1])
    assert result.observation is None
    assert result.live_vs_final is None
    assert result.manual_pitch_corrections is True


def test_delete_notes_refuses_to_empty_the_take_or_read_outside_it():
    """An empty score has no key to detect and no grid to engrave: the
    refusal happens BEFORE anything is mutated."""
    result = _result(60, 64)

    with pytest.raises(ValueError, match="at least one"):
        delete_notes(result, [0, 1])
    with pytest.raises(ValueError, match="no note"):
        delete_notes(result, [])
    with pytest.raises(IndexError):
        delete_notes(result, [2])

    assert [n.midi for n in result.notes] == [60, 64]  # nothing was touched


def test_a_deletion_snapshot_carries_its_own_history_label():
    """ "deletion" is a first-class EDIT_ACTIONS member: the timeline row
    of a state archived by a deletion says so."""
    history = EditHistory()
    history.record(snapshot_state(_result(60), None, action="deletion"))

    rows = history.entries(snapshot_state(_result(60, 64), None))

    assert rows[0][1] == "deletion"


def test_phrase_splice_rejects_an_incompatible_sample_rate():
    notes = [Note(0.0, 0.5, 60, 0.9)]
    with pytest.raises(ValueError, match="sample rates"):
        splice_phrase(
            notes,
            np.zeros(100, dtype=np.float32),
            [Note(0.0, 0.5, 62, 0.9)],
            np.zeros(200, dtype=np.float32),
            0,
            0,
            sr=200,
            replacement_sr=400,
        )


# -- the edit history: whole states, not deltas -------------------------------


def test_edit_snapshot_survives_a_later_in_place_correction():
    """A correction mutates the result object IN PLACE (refresh_result_facts
    reassigns its attributes): a snapshot holding a bare reference would
    alias the edited state and 'undo' would restore the correction."""
    result = _result(60, 64)
    state = snapshot_state(result, None)

    correct_note_pitch(result, 0, 62)

    assert result.notes[0].midi == 62
    assert state.result.notes[0].midi == 60, "the snapshot aliases the edited result"
    assert state.result.notes[1].midi == 64


def test_snapshot_accepts_a_non_dataclass_result_standalone():
    """A result that is not an AnalysisResult dataclass (a test double) is
    shared by reference instead of copied: the snapshot must still be
    built (the GUI's live analysis path snapshots whatever the take
    carries), and the copy that guards in-place corrections is only
    meaningful for dataclass results."""
    fake: Any = type("FakeResult", (), {"source": "live"})()
    state = snapshot_state(fake, None, live_notes=(1, 2), action="replacement")

    assert state.result is fake
    assert state.live_notes == (1, 2)
    assert state.action == "replacement"


def test_edit_history_round_trips_a_correction():
    result = _result(60, 64)
    history = EditHistory()
    assert not history.can_undo and not history.can_redo

    history.record(snapshot_state(result, None))
    correct_note_pitch(result, 1, 65)
    assert history.can_undo and not history.can_redo

    restored = history.undo(snapshot_state(result, None))
    assert restored is not None
    assert restored.result.notes[1].midi == 64
    assert restored.result.manual_pitch_corrections is False
    assert history.can_redo

    again = history.redo(snapshot_state(restored.result, None))
    assert again is not None
    assert again.result.notes[1].midi == 65
    assert not history.can_redo and history.can_undo


def test_a_new_edit_clears_the_redo_branch():
    history = EditHistory()
    history.record(snapshot_state(_result(60), None))
    history.undo(snapshot_state(_result(62), None))
    assert history.can_redo

    history.record(snapshot_state(_result(64), None))

    assert not history.can_redo, "a new edit invalidates the undone branch"
    assert history.can_undo


def test_the_history_keeps_at_most_its_depth():
    history = EditHistory(depth=2)
    for midi in (60, 62, 64):
        history.record(snapshot_state(_result(midi), None))

    first = history.undo(snapshot_state(_result(61), None))
    second = history.undo(snapshot_state(_result(63), None))
    third = history.undo(snapshot_state(_result(65), None))

    assert first is not None and first.result.notes[0].midi == 64
    assert second is not None and second.result.notes[0].midi == 62
    assert third is None, "the oldest state was kept beyond the bound"


def test_undo_and_redo_on_empty_branches_are_none():
    history = EditHistory()
    current = snapshot_state(_result(60), None)
    assert history.undo(current) is None
    assert history.redo(current) is None
    history.clear()
    assert not history.can_undo and not history.can_redo


def test_redo_row_keeps_the_action_that_separates_it_from_current():
    """Every timeline row is labelled by the edit on the edge between it
    and its neighbor toward the current state (the edge undo or redo
    re-crosses to land there). The undone current snapshot (whose own
    action means nothing — it is just the take as it is) must not leak its
    label onto the redo row: after an undo, the redo row is the state redo
    RESTORES (the edited one), labelled by the edit that produced it."""
    history = EditHistory()
    before = snapshot_state(_result(60), None, action="replacement")
    history.record(before)
    edited = _result(67)
    # the undo moves the caller ONTO the popped state (60)
    undone_to = history.undo(snapshot_state(edited, None))
    assert undone_to is not None and undone_to.result.notes[0].midi == 60

    rows = history.entries(snapshot_state(undone_to.result, None))

    # chronological: [current (the restored 60), redo row (the edited 67)]
    assert rows[0] == (None, CURRENT_ACTION)
    assert rows[1][0] is not None and rows[1][0].result.notes[0].midi == 67
    assert rows[1][1] == "replacement", "the redo row lost the edit's name"

    restored = history.redo(snapshot_state(undone_to.result, None))
    assert restored is not None and restored.result.notes[0].midi == 67
    # the state just left (60) is now an undo row labelled by the edit that
    # was re-applied to leave it
    rows = history.entries(snapshot_state(restored.result, None))
    assert rows[0][0] is not None and rows[0][0].result.notes[0].midi == 60
    assert rows[0][1] == "replacement"
    assert rows[-1] == (None, CURRENT_ACTION)


def test_redo_timeline_rows_are_oldest_to_newest():
    """Multiple redo steps appear chronologically after the current row."""
    history = EditHistory()
    history.record(snapshot_state(_result(60), None))
    history.record(snapshot_state(_result(62), None))
    history.record(snapshot_state(_result(64), None))
    current = snapshot_state(_result(66), None)

    history.undo(current)
    history.undo(snapshot_state(_result(64), None))
    rows = history.entries(snapshot_state(_result(62), None))

    assert [state.result.notes[0].midi for state, _ in rows if state is not None] == [
        60,
        64,
        66,
    ]
    assert rows[1] == (None, CURRENT_ACTION)
    # The UI's farthest-future row is valid and jumps through each redo step.
    move = history.jump(snapshot_state(_result(62), None), 3)
    assert move is not None and move.state.result.notes[0].midi == 66
    assert move.redone == 2


def test_snapshot_carries_the_edit_action_it_precedes():
    """`action` names the edit ABOUT to overwrite the state; it only feeds
    the history's timeline labels (undo/redo/jump treat states alike)."""
    correction_state = snapshot_state(_result(60), None)
    replacement_state = snapshot_state(_result(62), None, action="replacement")

    assert correction_state.action == "correction"  # the default edit
    assert replacement_state.action == "replacement"
    with pytest.raises(ValueError, match="unknown edit action"):
        snapshot_state(_result(64), None, action="ghost")


def test_history_lays_out_the_whole_timeline_for_the_state_browser():
    """`entries` lists archived states oldest first, then the live current
    state (None reference — it is not jumpable-to), then redo states in
    chronological order. A new edit clears the
    undone branch, so its states leave the timeline with it (the linear
    contract). Jumping stays undo/redo's bookkeeping: the browser reads
    rows, the window walks steps."""
    history = EditHistory()
    first = snapshot_state(_result(60), None)  # before correction 1
    history.record(first)  # ...applied, kept
    second = snapshot_state(_result(62), None)  # before correction 2
    history.record(second)  # ...applied, kept
    current = snapshot_state(_result(64), None)

    rows = history.entries(current)

    assert [(state, action) for state, action in rows if state is not None] == [
        (first, "correction"),
        (second, "correction"),
    ]
    assert rows[-1] == (None, CURRENT_ACTION), "the live state is labelled, never jumpable"

    # an undone state sits BETWEEN the undo rows and the redo rows: the
    # caller now sits on the second state (62), followed chronologically
    # by the state redo would restore (66, the second correction's result)
    history.undo(snapshot_state(_result(66), None))
    rows = history.entries(snapshot_state(_result(62), None))
    assert [
        (state.result.notes[0].midi, action) for state, action in rows if state is not None
    ] == [
        (60, "correction"),  # undo row, before the current row
        (66, "correction"),  # redo row, after the current row
    ]
    assert rows[1] == (None, CURRENT_ACTION)

    # a new edit on top of the undone state clears the whole redo branch
    # (the linear contract) and archives the state it overwrites — the 62
    # the caller sits on, now labelled by the NEW edit's kind
    history.record(snapshot_state(_result(62), None, action="replacement"))
    rows = history.entries(snapshot_state(_result(70), None))
    assert [
        (state.result.notes[0].midi, action) for state, action in rows if state is not None
    ] == [
        (60, "correction"),
        (62, "replacement"),
    ]


def test_history_walk_returns_landed_state_and_direction_counts():
    """The core owns multi-step branch traversal without a Qt adapter."""
    history = EditHistory()
    history.record(snapshot_state(_result(60), None))
    history.record(snapshot_state(_result(62), None))
    current = snapshot_state(_result(64), None)

    undone = history.walk(current, 2)

    assert undone is not None
    assert undone.state.result.notes[0].midi == 60
    assert (undone.undone, undone.redone) == (2, 0)
    assert undone.acknowledgement_key == "ui edit undone"

    redone = history.walk(undone.state, -2)

    assert redone is not None
    assert redone.state.result.notes[0].midi == 64
    assert (redone.undone, redone.redone) == (0, 2)
    assert redone.acknowledgement_key == "ui edit redone"


def test_history_jump_uses_timeline_rows_and_reports_partial_moves():
    """Jumps use chronological rows and preserve the undo/redo branch."""
    history = EditHistory()
    history.record(snapshot_state(_result(60), None))
    history.record(snapshot_state(_result(62), None))
    current = snapshot_state(_result(64), None)

    past = history.jump(current, 0)

    assert past is not None and past.state.result.notes[0].midi == 60
    assert (past.undone, past.redone) == (2, 0)
    assert past.acknowledgement_key == "ui edit jumped"
    assert history.jump(past.state, 0) is None
    middle = history.jump(past.state, 1)
    assert middle is not None and middle.state.result.notes[0].midi == 62
    assert middle.redone == 1

    future = history.jump(middle.state, 2)
    assert future is not None and future.state.result.notes[0].midi == 64
    assert (future.undone, future.redone) == (0, 1)


def test_history_walk_on_empty_or_zero_steps_returns_none():
    history = EditHistory()
    current = snapshot_state(_result(60), None)
    assert history.walk(current, 0) is None
    assert history.walk(current, 1) is None
    assert history.walk(current, -1) is None


def test_history_depth_bound_shows_in_the_timeline():
    """States dropped past the depth bound leave the timeline too."""
    history = EditHistory(depth=2)
    history.record(snapshot_state(_result(60), None))
    history.record(snapshot_state(_result(62), None))
    history.record(snapshot_state(_result(64), None))  # 60 fell off the bound

    rows = history.entries(snapshot_state(_result(66), None))

    assert [state.result.notes[0].midi for state, _ in rows if state is not None] == [62, 64]
    assert history.undo_depth == 2


def _edit_timeline_identity(history, result, signal, live_notes):
    """Describe history branch identities without comparing NumPy arrays."""
    rows = history.entries(snapshot_state(result, signal, live_notes))
    return [(id(state) if state is not None else None, action) for state, action in rows]


@pytest.mark.parametrize(
    ("intent_kind", "action", "midis"),
    [
        ("correct", "correction", [60, 65, 67]),
        ("move", "move", [60, 62, 67]),
        ("delete", "deletion", [60, 67]),
    ],
)
def test_note_edit_transaction_commits_one_whole_before_state(intent_kind, action, midis):
    """Each direct edit publishes a candidate and archives one take state."""
    result = _result(60, 64, 67)
    signal = np.arange(64, dtype=np.float32)
    live_notes = (object(), object())
    history = EditHistory()
    before_result = result

    intent = {
        "correct": edits_module.CorrectPitchIntent(index=1, midi=65),
        "move": edits_module.MoveNoteIntent(index=1, start_s=0.7, end_s=1.1, midi=62),
        "delete": edits_module.DeleteNotesIntent(indices=(1,)),
    }[intent_kind]
    outcome = edits_module.apply_note_edit(
        result,
        history,
        signal=signal,
        live_notes=live_notes,
        intent=intent,
    )

    assert outcome.result is not before_result
    assert [note.midi for note in outcome.result.notes] == midis
    assert outcome.before.result.notes[1].midi == 64
    assert outcome.before.action == action
    assert outcome.before.signal is signal
    assert outcome.before.live_notes == live_notes
    assert history.undo_depth == 1
    assert history.can_undo and not history.can_redo
    rows = history.entries(snapshot_state(outcome.result, signal, live_notes))
    assert len(rows) == 2
    assert rows[0][0] is outcome.before
    assert rows[0][1] == action
    assert rows[1] == (None, CURRENT_ACTION)


def test_note_edit_transaction_failure_preserves_original_and_both_history_branches(
    monkeypatch,
):
    """A late derived-fact failure never partially publishes the candidate."""
    result = _result(60, 64, 67)
    result.pitch = cast(Any, object())
    result.observation = cast(Any, object())
    result.live_vs_final = {"live": 3, "final": 3}
    signal = np.arange(64, dtype=np.float32)
    live_notes = (object(),)
    history = EditHistory()
    older = snapshot_state(_result(55, 57), signal, live_notes, action="move")
    history.record(older)
    history.record(snapshot_state(_result(58, 59), signal, live_notes))
    undone = history.undo(snapshot_state(result, signal, live_notes))
    assert undone is not None
    assert history.can_undo and history.can_redo
    timeline_before = _edit_timeline_identity(history, result, signal, live_notes)

    original_notes = result.notes
    original_note_facts = [
        (note.start, note.end, note.midi, note.deviation) for note in result.notes
    ]
    original_pitch = result.pitch
    original_observation = result.observation
    original_live_vs_final = result.live_vs_final

    def fail_after_candidate_mutation(candidate, notes, **_kwargs):
        candidate.notes = notes
        candidate.key = "partial candidate update"
        raise RuntimeError("derived facts failed")

    monkeypatch.setattr(edits_module, "refresh_result_facts", fail_after_candidate_mutation)
    with pytest.raises(RuntimeError, match="derived facts failed"):
        edits_module.apply_note_edit(
            result,
            history,
            signal=signal,
            live_notes=live_notes,
            intent=edits_module.CorrectPitchIntent(index=1, midi=65),
        )

    assert result.notes is original_notes
    assert [
        (note.start, note.end, note.midi, note.deviation) for note in result.notes
    ] == original_note_facts
    assert result.pitch is original_pitch
    assert result.observation is original_observation
    assert result.live_vs_final is original_live_vs_final
    assert history.can_undo and history.can_redo
    assert _edit_timeline_identity(history, result, signal, live_notes) == timeline_before


@pytest.mark.parametrize(
    "intent",
    [
        edits_module.CorrectPitchIntent(index=99, midi=62),
        edits_module.MoveNoteIntent(index=99, start_s=0.7, end_s=1.1, midi=62),
        edits_module.DeleteNotesIntent(indices=(99,)),
    ],
)
def test_note_edit_transaction_validation_failure_preserves_redo_branch(intent):
    """Rejected intents leave the current take and linear history untouched."""
    result = _result(60, 64, 67)
    signal = np.arange(32, dtype=np.float32)
    live_notes = (object(),)
    history = EditHistory()
    history.record(snapshot_state(_result(55, 57), signal, live_notes, action="move"))
    history.record(snapshot_state(_result(56, 58), signal, live_notes))
    history.undo(snapshot_state(_result(58, 59), signal, live_notes))
    timeline_before = _edit_timeline_identity(history, result, signal, live_notes)
    notes_before = result.notes
    facts_before = [(note.start, note.end, note.midi) for note in result.notes]

    with pytest.raises((IndexError, ValueError)):
        edits_module.apply_note_edit(
            result,
            history,
            signal=signal,
            live_notes=live_notes,
            intent=intent,
        )

    assert result.notes is notes_before
    assert [(note.start, note.end, note.midi) for note in result.notes] == facts_before
    assert history.can_undo and history.can_redo
    assert _edit_timeline_identity(history, result, signal, live_notes) == timeline_before
