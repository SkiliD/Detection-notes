"""One explained diff between two readings of the SAME audio.

A re-interpretation or a full pYIN re-analysis replaces the reading the
user was looking at. The singer must see EXACTLY what changed, without
comparing two JSON reports and without losing the previous reading. This
module owns that comparison: it receives two `AnalysisResult` (the
reading shown before the action, the reading that just succeeded), pairs
their notes one-to-one and returns a typed, Qt-free presentation.

The labels ("stable", "near", "changed", "missing", "added") and the
reasons describe the DIFFERENCE BETWEEN TWO ENGINE OUTPUTS ONLY. They
never claim which reading is true; nothing here corrects a note, and no
threshold decides confidence — pYIN's confidence stays a measurement.

Reuse over re-derivation:
- the match tolerance is `compare.match_window` (top of the grid
  sixteenth or 0.10 s), the SAME bound the live/final comparison uses;
- the pairing is `compare.chronological_pairs` (the chronological
  dynamic program shared with the phrase comparison), so two readings of
  one take and two takes of one phrase never grow a second matching rule;
- the coherence suspects come from `continuity.coherence_report` for BOTH
  readings (this module never re-detects plateaus nor picks thresholds).

The result holds only presentation data (note spans, pairing reasons,
suspect rows) — no signal, no pitch track, no observation survive the
call. It is visual-only by design: the current result stays the single
source of every export.
"""

from __future__ import annotations

from dataclasses import dataclass

from .compare import chronological_pairs, match_window
from .continuity import coherence_report

# Pair statuses. "near" is the visible rounding divergence (an off-one
# neighbour at an unchanged position) — a fact to SHOW, never a fault to
# fix or a correction to apply.
STABLE = "stable"
NEAR = "near"
CHANGED = "changed"
MISSING = "missing"  # a before note with no after counterpart
ADDED = "added"  # an after note with no before counterpart

# Pair change reasons (they can coexist on one pair; deltas are signed).
REASON_PITCH = "pitch"
REASON_ATTACK = "attack"
REASON_DURATION = "duration"

# Suspect statuses, derived from the note pairing already computed.
SUSPECT_NEW = "new"
SUSPECT_PERSISTENT = "persistent"


@dataclass(frozen=True)
class NoteSnapshot:
    """One note of either reading, as the presentation needs it."""

    index: int
    start: float
    end: float
    midi: int


@dataclass(frozen=True)
class NotePair:
    """One before note and the after note it became, with the why.

    `status` is one of STABLE / NEAR / CHANGED; `reasons` carry the change
    causes of a CHANGED pair (empty otherwise). Deltas are signed:
    positive = later or higher in the AFTER reading.
    """

    before: NoteSnapshot
    after: NoteSnapshot
    status: str
    reasons: tuple[str, ...]
    pitch_delta: int  # after.midi - before.midi (st)
    onset_delta_s: float  # after.start - before.start
    end_delta_s: float  # after.end - before.end


@dataclass(frozen=True)
class SuspectRow:
    """One coherence suspect, ready for the panel, with its diff status.

    The fields are the ones `continuity.coherence_report` already
    publishes (where the hidden plateau sits, how long, at which pitch,
    the signed offset) — nothing re-measured here. `confidence` is the
    note's own pYIN mean, shown as context only: never compared to a
    threshold, never a severity. `status` is None when the pair is
    ambiguous or not computable, and the row then simply lists.
    """

    note_index: int
    start_s: float
    duration_s: float
    midi: int
    plateau_s: float
    plateau_midi: int
    plateau_offset_st: float
    confidence: float | None
    status: str | None


@dataclass(frozen=True)
class ReadingComparison:
    """The whole before/after presentation for one successful re-analysis.

    `before_notes` are every note of the OLD reading (the ghost bars);
    `pairs` pair the two readings one-to-one (a note never appears twice);
    `missing`/`added` are the unpaired ones. Suspects come in two lists:
    the current reading's (`after_suspects`, each labelled new/persistent
    or None) and the old reading's now-gone ones (`resolved_suspects`).
    `suspects_available` is False when either observation is absent — a
    missing datum is shown as unavailable, never as zero suspects.
    """

    before_notes: tuple[NoteSnapshot, ...]
    pairs: tuple[NotePair, ...]
    missing: tuple[NoteSnapshot, ...]
    added: tuple[NoteSnapshot, ...]
    after_suspects: tuple[SuspectRow, ...]
    resolved_suspects: tuple[SuspectRow, ...]
    suspects_available: bool

    def count(self, status: str) -> int:
        """How many pairs carry a status (one-off convenience for the UI)."""
        return sum(1 for p in self.pairs if p.status == status)


def _snapshots(notes) -> list[NoteSnapshot]:
    return [
        NoteSnapshot(index=i, start=float(n.start), end=float(n.end), midi=int(n.midi))
        for i, n in enumerate(notes or [])
    ]


def _suspect_note_index(notes, suspect: dict) -> int | None:
    """The note a suspect row was computed FROM (rounding-aware).

    `coherence_report` rounds the note's start and duration onto the row;
    matching on the same rounding keeps the mapping exact and honest —
    a note INDEX ALONE is never an identity across two segmentations.
    """
    start_s = suspect.get("start_s")
    duration_s = suspect.get("duration_s")
    midi = suspect.get("midi")
    for i, n in enumerate(notes or []):
        if (
            round(float(n.start), 3) == start_s
            and round(float(n.duration), 3) == duration_s
            and int(n.midi) == midi
        ):
            return i
    return None


def _pairs(
    before: list[NoteSnapshot], after: list[NoteSnapshot], tol: float, beat_s: float
) -> tuple:
    """One-to-one chronological pairing inside ONE BEAT; ties are decided.

    The pairing itself is `compare.chronological_pairs` (the shared
    chronological dynamic program): it minimises the whole alignment with
    an explicit cost per association, missing note and added note, so a
    nearest-first choice cannot leave a note unpaired while a complete
    pairing exists (the annotated case: before 0.4/0.7 s, after 0/0.5 s
    at a 0.5 s beat — greedy pairs 0.4 -> 0.5 and loses 0.7; the program
    pairs both).

    The ASSOCIATION threshold (one beat of the take's grid, per the
    design spec) is deliberately wider than the CHANGE threshold `tol`
    (the visible window): a note can pair with a counterpart whose attack
    was re-dated by much more than the visible-comparison bound, and the
    pair then reports that shift as its REASON instead of calling the
    note missing. Equal costs are settled toward the oldest note (see the
    aligner's candidate order), so the same two readings always produce
    the same pairing — a property a test pins.
    """
    matches, missing, _added = chronological_pairs(
        [b.start for b in before], [a.start for a in after], beat_s
    )
    pairs = tuple(_pair(before[i], after[j], tol) for i, j in matches)
    return pairs, tuple(before[i] for i in missing)


def _pair(b: NoteSnapshot, a: NoteSnapshot, tol: float) -> NotePair:
    """Classify one pair: stable, near (the off-one rounding) or changed."""
    onset_delta = a.start - b.start
    end_delta = a.end - b.end
    pitch_delta = a.midi - b.midi
    geometry_quiet = abs(onset_delta) <= tol and abs(end_delta) <= tol
    if pitch_delta == 0 and geometry_quiet:
        return NotePair(b, a, STABLE, (), pitch_delta, onset_delta, end_delta)
    if abs(pitch_delta) == 1 and geometry_quiet:
        # the ±1 rounding family of two roundings over the same plateau —
        # shown as itself, distinct from a correction's style
        return NotePair(b, a, NEAR, (), pitch_delta, onset_delta, end_delta)
    reasons: list[str] = []
    if pitch_delta != 0:
        reasons.append(REASON_PITCH)
    if abs(onset_delta) > tol:
        reasons.append(REASON_ATTACK)
    if abs(end_delta) > tol:
        reasons.append(REASON_DURATION)
    return NotePair(b, a, CHANGED, tuple(reasons), pitch_delta, onset_delta, end_delta)


def _suspect_row(notes, suspect: dict, index: int | None, status: str | None) -> SuspectRow:
    confidence: float | None = None
    if index is not None and index < len(notes or []):
        value = getattr(notes[index], "confidence", None)
        if isinstance(value, (int, float)):
            confidence = round(float(value), 3)
    return SuspectRow(
        note_index=index if index is not None else -1,
        start_s=float(suspect.get("start_s", 0.0)),
        duration_s=float(suspect.get("duration_s", 0.0)),
        midi=int(suspect.get("midi", 0)),
        plateau_s=float(suspect.get("secondary_plateau_s", 0.0)),
        plateau_midi=int(suspect.get("secondary_plateau_midi", 0)),
        plateau_offset_st=float(suspect.get("secondary_plateau_offset_st", 0.0)),
        confidence=confidence,
        status=status,
    )


def _suspects(
    before_result, after_result, note_mapping: dict[int, int]
) -> tuple[tuple[SuspectRow, ...], tuple[SuspectRow, ...]]:
    """Label both readings' suspects through the note pairing.

    The AFTER suspects are the display case: new (no older counterpart)
    versus persistent (the same doubt survived the re-read). The BEFORE
    suspects that no longer appear are the resolved ones. Suspects an
    absent observation or an unmatched note leave unlabelled simply list
    with no status — an unavailable check is never read as "clean".
    """
    after_notes = list(after_result.notes or [])
    before_notes = list(before_result.notes or [])
    before_obs = getattr(before_result, "observation", None)
    after_obs = getattr(after_result, "observation", None)
    has_before = before_obs is not None and bool(before_notes)
    has_after = after_obs is not None and bool(after_notes)
    if not (has_before and has_after):
        return (), ()

    before_block = coherence_report(before_obs, before_notes)
    after_block = coherence_report(after_obs, after_notes)
    before_rows: dict[int, dict] = {}
    for suspect in before_block["suspects"]:
        index = _suspect_note_index(before_notes, suspect)
        if index is not None:
            before_rows[index] = suspect

    after_rows: list[SuspectRow] = []
    resolved: list[SuspectRow] = []
    after_seen: set[int] = set()
    for suspect in after_block["suspects"]:
        index = _suspect_note_index(after_notes, suspect)
        status: str | None = SUSPECT_NEW
        if index is not None:
            before_index = note_mapping.get(index)
            if before_index is not None and before_index in before_rows:
                status = SUSPECT_PERSISTENT
        after_rows.append(_suspect_row(after_notes, suspect, index, status))
        if index is not None:
            after_seen.add(index)

    # a before suspect is resolved when its note's counterpart in the new
    # reading (or the note itself, gone) carries no more suspect
    for before_index, suspect in before_rows.items():
        after_index = None
        for k, v in note_mapping.items():
            if v == before_index:
                after_index = k
                break
        if after_index is not None and after_index in after_seen:
            continue
        resolved.append(_suspect_row(before_notes, suspect, before_index, "resolved"))
    return tuple(after_rows), tuple(resolved)


def compare_readings(before, after) -> ReadingComparison:
    """Compare two readings of the SAME audio; the only public operation.

    `before` is the reading the user was shown, `after` the one that just
    succeeded. Same audio is the CALLER'S contract (the window passes the
    result it displayed, then the fresh one) — a different take is not
    this function's to detect. The grid tolerance reads `after`'s grid
    (two readings of one take may refine the tempo differently, and the
    match_window fallback covers a gridless hand-built result).
    """
    before_notes = _snapshots(getattr(before, "notes", None))
    after_notes = _snapshots(getattr(after, "notes", None))
    tol = match_window(getattr(after, "grid", None))
    # the ASSOCIATION threshold: one beat of the take's grid (120 BPM +
    # 0.5 s). A gridless result falls back to the same bound it already
    # uses for a windowless match.
    grid_bpm = getattr(getattr(after, "grid", None), "bpm", None)
    beat_s = 60.0 / grid_bpm if grid_bpm else match_window(None) * 4.0
    pairs, missing = _pairs(before_notes, after_notes, tol, beat_s)
    added = tuple(a for i, a in enumerate(after_notes) if i not in used_indices(pairs))
    mapping = {p.after.index: p.before.index for p in pairs}
    after_suspects, resolved = _suspects(before, after, mapping)
    return ReadingComparison(
        before_notes=tuple(before_notes),
        pairs=pairs,
        missing=missing,
        added=added,
        after_suspects=after_suspects,
        resolved_suspects=resolved,
        suspects_available=bool(
            getattr(after, "observation", None) is not None
            and getattr(before, "observation", None) is not None
            and before_notes
            and after_notes
        ),
    )


def used_indices(pairs) -> set[int]:
    """The after-note indices already paired (pairing is EXCLUSIVE)."""
    return {p.after.index for p in pairs}
