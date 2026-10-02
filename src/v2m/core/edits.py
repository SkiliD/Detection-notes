"""Musical edits applied to an analyzed take before it is exported."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .chords import detect_chord_progression, detect_chord_segments, detect_chords
from .figures import label_notes
from .grid import BeatGrid
from .keys import detect_key
from .meter import detect_meter
from .notes import Note
from .rests import detect_rests

if TYPE_CHECKING:
    from .pipeline import AnalysisResult

# How many whole-take states the edit history keeps. Each state shares the
# audio signal by reference (nothing mutates an analyzed signal in place),
# so the cost is one small result object per edit — the bound exists so a
# long editing session cannot grow without limit.
DEFAULT_EDIT_DEPTH = 20

# What one archived state was BEFORE (the token that names the past state
# in the take's history). "correction" = a pitch fix; "replacement" = a
# kept re-sung phrase; "deletion" = notes removed from the score (the
# audio is untouched). CURRENT_ACTION is the live timeline row's label (a
# None state — never a state's own action: a snapshot always precedes an
# edit).
EDIT_ACTIONS = ("correction", "replacement", "deletion")
CURRENT_ACTION = "current"


@dataclass(frozen=True)
class EditState:
    """One restorable whole-take state: a note edit's "before" or "after".

    A note edit either mutates the result object in place (a pitch
    correction reassigns `notes`, `key`, `grid`, ...) or replaces it
    wholesale (a spliced re-sing installs a fresh analyzed result), so
    the only state that restores either exactly is the WHOLE state: the
    result, its audio, and the take's live reading. The result is a
    SHALLOW copy taken at snapshot time — the copy, not a reference, is
    what keeps the pre-edit values readable after an in-place correction.
    The signal and the Note objects are shared by reference: the pipeline
    never mutates an analyzed signal or an emitted note in place.

    `action` names the edit that was ABOUT to overwrite this state (a
    member of EDIT_ACTIONS): it is the word the take's history shows the
    singer, not a behavior switch — undo/redo/jump treat every state alike.
    """

    result: "AnalysisResult"
    signal: np.ndarray | None
    live_notes: tuple = ()
    action: str = "correction"


def snapshot_state(
    result: "AnalysisResult",
    signal: np.ndarray | None,
    live_notes: Any = (),
    action: str = "correction",
) -> EditState:
    """Capture the whole take state an edit is about to overwrite.

    `action` names the edit that is about to be applied (an EDIT_ACTIONS
    member); it travels ON the snapshot and only feeds the history's
    timeline labels.
    """
    if action not in EDIT_ACTIONS:  # CURRENT_ACTION is a row label, not a state
        raise ValueError(f"unknown edit action: {action}")
    # the shallow copy is what keeps the pre-edit values readable after an
    # in-place correction (refresh_result_facts reassigns attributes); a
    # non-dataclass result (a test double) cannot be copied this way and is
    # shared by reference — the edits that mutate an AnalysisResult in
    # place reject such an object long before the snapshot could alias it
    if dataclasses.is_dataclass(result):
        result = replace(result)
    return EditState(
        result=result,
        signal=signal,
        live_notes=tuple(live_notes),
        action=action,
    )


class EditHistory:
    """Undo/redo over one take's note edits: whole states, never deltas.

    Linear history: `record` clears the redo branch — a new edit on top of
    an undone state invalidates the undone branch. `undo`/`redo` take the
    CURRENT state (the caller snapshots its own facts) and return the state
    to restore, or None when the branch is empty; the caller swaps that
    state into its views.

    Every archived state carries an `action` naming the edit that separates
    it from the current state, so `entries()` can lay out the WHOLE
    timeline — archived states, then the live current one — for the
    results page's state dropdown (jump-to-any-previous-state, not only
    one step at a time).
    """

    def __init__(self, depth: int = DEFAULT_EDIT_DEPTH) -> None:
        self._depth = max(1, depth)
        self._undo: list[EditState] = []
        # the redo branch pairs each restorable state with the action redo
        # would REAPPLY there: `undo` pushes the caller's current snapshot
        # (whose own action means nothing — it is just "the take as it is"),
        # so the pair keeps the popped state's action as the row's label
        self._redo: list[tuple[EditState, str]] = []

    def record(self, state: EditState) -> None:
        """Archive the state an edit is about to overwrite (call BEFORE it)."""
        self._undo.append(state)
        if len(self._undo) > self._depth:
            self._undo.pop(0)
        self._redo.clear()

    def entries(self, current: EditState) -> list[tuple[EditState | None, str]]:
        """The take's whole timeline, chronological, for the state browser.

        Undo rows oldest first, then the CURRENT state (a None reference
        and the `CURRENT_ACTION` label — never jumpable-to, the caller
        holds it), then the redo rows nearest-future first: the timeline is
        a slider on the take's past AND future. Every row is labelled by
        the edit that separates it from the current state — applying that
        edit to CURRENT moves the timeline to the row (backward for an
        undo row, forward for a redo row). Row `i` sits `undo_depth - i`
        steps from the current row (negative = redo forward) — the
        arithmetic the window's jump applies. The caller walks to an
        archived row through `undo`/`redo`, which keeps ONE owner of the
        branch bookkeeping.
        """
        rows: list[tuple[EditState | None, str]] = [
            (state, state.action) for state in self._undo
        ]
        rows.append((None, CURRENT_ACTION))
        # the redo stack is stored nearest-first (LIFO) — already the
        # chronological order for the rows after the current one
        rows.extend(self._redo)
        return rows

    def undo(self, current: EditState) -> EditState | None:
        """Return the state to restore, or None when nothing can be undone."""
        if not self._undo:
            return None
        popped = self._undo.pop()
        self._redo.append((current, popped.action))
        return popped

    def redo(self, current: EditState) -> EditState | None:
        """Return the state to restore, or None when nothing can be redone."""
        if not self._redo:
            return None
        state, action = self._redo.pop()
        # the state just left becomes an undo row, labelled by the edit
        # that separates it from the restored one (the reapplied action)
        self._undo.append(EditState(
            result=current.result,
            signal=current.signal,
            live_notes=current.live_notes,
            action=action,
        ))
        return state

    def clear(self) -> None:
        """Forget every state: a new take (or a new READING of one) starts it."""
        self._undo.clear()
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_depth(self) -> int:
        """How many archived states undo can still reach (the undo stack's size)."""
        return len(self._undo)


def splice_phrase(
    notes: list[Note],
    signal: np.ndarray,
    replacement_notes: list[Note],
    replacement_signal: np.ndarray,
    first_index: int,
    last_index: int,
    sr: int,
    *,
    replacement_sr: int | None = None,
) -> tuple[list[Note], np.ndarray, float]:
    """Replace one contiguous note span and its audio, shifting later notes.

    Leading and trailing silence in the replacement is discarded. The splice
    boundaries are the selected notes' first onset and last offset; gaps
    inside the selection and inside the new singing remain part of the phrase.
    """
    if not 0 <= first_index <= last_index < len(notes):
        raise ValueError("the selected note range is invalid")
    if notes != sorted(notes, key=lambda note: note.start):
        raise ValueError("the source notes are not chronological")
    if replacement_notes != sorted(replacement_notes, key=lambda note: note.start):
        raise ValueError("the replacement notes are not chronological")
    if not replacement_notes:
        raise ValueError("the replacement contains no notes")
    if sr <= 0:
        raise ValueError("sample rate must be positive")
    if replacement_sr is not None and replacement_sr != sr:
        raise ValueError("the replacement and original sample rates differ")

    source = np.asarray(signal, dtype=np.float32)
    sung = np.asarray(replacement_signal, dtype=np.float32)
    chosen = notes[first_index : last_index + 1]
    start_sample = max(0, min(len(source), int(round(chosen[0].start * sr))))
    end_sample = max(0, min(len(source), int(round(chosen[-1].end * sr))))
    if end_sample <= start_sample:
        raise ValueError("the selected note range has no audio duration")

    sung_start = max(0, min(len(sung), int(round(min(n.start for n in replacement_notes) * sr))))
    sung_end = max(0, min(len(sung), int(round(max(n.end for n in replacement_notes) * sr))))
    if sung_end <= sung_start:
        raise ValueError("the replacement has no audio duration")

    inserted_audio = sung[sung_start:sung_end].copy()
    delta = (len(inserted_audio) - (end_sample - start_sample)) / sr
    splice_start_s = start_sample / sr
    trim_start_s = sung_start / sr

    merged_notes = [replace(note) for note in notes[:first_index]]
    inserted_notes = [
        replace(
            note,
            start=splice_start_s + note.start - trim_start_s,
            end=splice_start_s + note.end - trim_start_s,
        )
        for note in replacement_notes
    ]
    following_notes = [
        replace(note, start=note.start + delta, end=note.end + delta)
        for note in notes[last_index + 1 :]
    ]
    if merged_notes and inserted_notes and inserted_notes[0].start < merged_notes[-1].end:
        raise ValueError("the replacement overlaps the preceding note")
    if inserted_notes and following_notes and following_notes[0].start < inserted_notes[-1].end:
        raise ValueError("the replacement overlaps the following note")
    merged_signal = np.concatenate(
        (source[:start_sample], inserted_audio, source[end_sample:])
    ).astype(np.float32, copy=False)
    merged_notes.extend(inserted_notes)
    merged_notes.extend(following_notes)
    return merged_notes, merged_signal, delta


def refresh_result_facts(
    result,
    notes: list[Note],
    *,
    duration_s: float | None = None,
    pitch=None,
    observation=None,
) -> None:
    """Rebuild all musical fields that depend on the note list."""
    result.notes = notes
    if duration_s is not None:
        result.duration_s = duration_s
    result.pitch = pitch
    result.observation = observation

    result.key = detect_key(notes)
    result.chord = detect_chords(notes)
    result.chord_segments = detect_chord_segments(notes) if result.chord else []

    grid = None
    if result.tempo is not None and result.tempo.bpm > 0:
        grid = BeatGrid(result.tempo.bpm)
        label_notes(notes, grid)
        meter = detect_meter(notes, result.tempo.bpm)
        grid = grid.with_meter(meter)
        result.meter = meter
        result.note_positions = grid.positions(notes)
        result.num_measures = grid.measures(notes)
        result.chord_progression = detect_chord_progression(
            notes, grid, result.num_measures
        )
    else:
        result.meter = None
        result.note_positions = None
        result.num_measures = 0
        result.chord_progression = []
    result.grid = grid
    try:
        result.rests = detect_rests(notes, grid)
    except (TypeError, ValueError, AttributeError):
        result.rests = []


def correct_note_pitch(result, index: int, midi: int) -> None:
    """Change one note's pitch without changing its onset or duration."""
    if not 0 <= index < len(result.notes):
        raise IndexError("note index is outside the result")
    if not 21 <= midi <= 108:
        raise ValueError("MIDI pitch is outside the piano range")
    notes = [replace(note) for note in result.notes]
    notes[index].midi = midi
    notes[index].deviation = 0.0
    refresh_result_facts(
        result,
        notes,
        pitch=None,
        observation=None,
    )
    result.live_vs_final = None
    result.manual_pitch_corrections = True


def delete_notes(result, indices) -> None:
    """Remove the selected notes from the score — the audio is untouched.

    Every note-derived fact is rebuilt (key, chords, grid, figures, rests)
    and the audio-based checks go unavailable, exactly like a pitch
    correction: the notes no longer describe the recording. The take must
    keep at least one note — an empty score has no key to detect and no
    grid to engrave, so a deletion that would empty it is refused.
    """
    chosen = {int(i) for i in indices}
    if not chosen:
        raise ValueError("no note is selected")
    if min(chosen) < 0 or max(chosen) >= len(result.notes):
        raise IndexError("note index is outside the result")
    if len(chosen) == len(result.notes):
        raise ValueError("the take must keep at least one note")
    notes = [replace(note) for i, note in enumerate(result.notes) if i not in chosen]
    refresh_result_facts(
        result,
        notes,
        pitch=None,
        observation=None,
    )
    result.live_vs_final = None
    result.manual_pitch_corrections = True
