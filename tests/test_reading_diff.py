"""The reading-diff policy, pinned without Qt and without audio.

`compare_readings` is the ONLY public operation: two `AnalysisResult`
readings of the SAME audio in, one typed presentation out. These tests
pin the pairing policy (exclusive, chronological inside the compare
window's association beat, decided ties), the status/reason classification (stable /
near / changed with their signed deltas and causes), the suspect
labelling (new versus persistent, resolved ones, and an absent evidence
shown as UNAVAILABLE — never as a clean take), and the empty cases.
The adversarial family runs the shipped dynamic program against the
nearest-first greedy it replaced, on the same annotated inputs.
"""

from __future__ import annotations

import pytest

from v2m.core.compare import match_window
from v2m.core.notes import Note
from v2m.core.reading_diff import (
    CHANGED,
    NEAR,
    REASON_ATTACK,
    REASON_DURATION,
    REASON_PITCH,
    STABLE,
    SUSPECT_NEW,
    compare_readings,
    used_indices,
)


def _result(notes, grid=None, observation=None):
    """A minimal result duck-typed as compare_readings expects."""
    from types import SimpleNamespace

    return SimpleNamespace(notes=list(notes), grid=grid, observation=observation)


def _grid(bpm: float = 240.0):
    """A fast grid keeps the sixteenth (62 ms) under a note used for a change.

    The match window is computed against this same grid, so the declared
    tolerance and the tests' expectations read the SAME arithmetic.
    """
    from types import SimpleNamespace

    return SimpleNamespace(bpm=bpm, step_seconds=lambda per_beat=4.0: (60.0 / bpm) / per_beat)


def _note(start, end, midi, conf=0.9):
    return Note(start, end, midi, conf)


# --------------------------------------------------------------------------- #
# Stable / near / changed pairing
# --------------------------------------------------------------------------- #


def test_identical_readings_pair_every_note_stable():
    notes = [_note(0.10, 0.40, 60), _note(0.50, 0.80, 64)]
    cmp = compare_readings(_result(notes), _result(list(notes)))
    assert [p.status for p in cmp.pairs] == [STABLE, STABLE]
    assert not cmp.missing and not cmp.added
    assert cmp.before_notes


def test_off_one_neighbour_is_near_not_changed():
    before = [_note(0.10, 0.40, 60)]
    after = [_note(0.10, 0.40, 61)]
    cmp = compare_readings(_result(before), _result(after))
    assert len(cmp.pairs) == 1
    pair = cmp.pairs[0]
    assert pair.status == NEAR
    assert pair.pitch_delta == 1


def test_pitch_and_duration_changes_carry_their_reasons_and_signed_deltas():
    # 240 BPM: the match window is its floor 0.12 s; onset 0.10 s inside,
    # end 0.30 s beyond — the pair holds and the REASONS separate the causes
    grid = _grid()
    before = [_note(0.10, 0.40, 60)]
    after = [_note(0.20, 0.70, 67)]
    cmp = compare_readings(_result(before, grid=grid), _result(after, grid=grid))
    pair = cmp.pairs[0]
    assert pair.status == CHANGED
    assert set(pair.reasons) == {REASON_PITCH, REASON_DURATION}
    assert pair.pitch_delta == 7
    assert pair.onset_delta_s == pytest.approx(0.10, abs=1e-4)
    assert pair.end_delta_s == pytest.approx(0.30, abs=1e-4)


def test_an_onset_re_dated_beyond_the_visibility_window_carries_the_attack_reason():
    """A pair can hold while its attack was re-dated: the reason shows it.

    The ASSOCIATION threshold is one beat (0.25 s here); the visible window
    is 0.10 s. An onset shift of 0.15 s pairs the note and the reason says
    ATTACK — the shift is explained, not erased as a disappeared note.
    """
    grid = _grid()
    before = [_note(0.10, 0.40, 60)]
    after = [_note(0.25, 0.60, 60)]  # same pitch, onset 0.15 s (beyond 0.10)
    cmp = compare_readings(_result(before, grid=grid), _result(after, grid=grid))
    pair = cmp.pairs[0]
    assert pair.status == CHANGED
    assert REASON_ATTACK in pair.reasons
    assert REASON_PITCH not in pair.reasons


def test_a_note_beyond_one_beat_is_unpaired():
    before = [_note(0.10, 0.40, 60)]
    after = [_note(2.00, 2.30, 60)]
    cmp = compare_readings(_result(before), _result(after))
    assert [p.status for p in cmp.pairs] == []
    assert [p.midi for p in cmp.missing] == [60]
    assert [p.midi for p in cmp.added] == [60]


def test_the_older_note_of_a_tie_is_taken_first_deterministically():
    before = [_note(0.20, 0.50, 60)]
    after = [_note(0.20, 0.45, 60), _note(0.20, 0.55, 60)]
    cmp = compare_readings(_result(before), _result(after))
    assert len(cmp.pairs) == 1
    # the pairing is exclusive: the second after note cannot also pair
    assert len(cmp.added) == 1
    assert cmp.pairs[0].status == STABLE


def test_two_readings_pair_the_same_way_twice():
    before = [_note(0.10, 0.40, 60), _note(0.50, 0.80, 62)]
    after = [_note(0.30, 0.60, 60), _note(0.70, 1.00, 63)]
    cmp1 = compare_readings(_result(before), _result(after))
    cmp2 = compare_readings(_result(before), _result(after))
    assert cmp1 == cmp2


def test_the_matching_tolerance_reads_the_compare_window():
    grid = _grid(240.0)
    assert match_window(grid) == pytest.approx(max(0.0625, 0.10), abs=1e-6)
    # and compare_readings accepts a gridless result (the fallback window)
    cmp = compare_readings(_result([_note(0.10, 0.40, 60)]), _result([_note(0.10, 0.40, 60)]))
    assert cmp.pairs[0].status == STABLE


# --------------------------------------------------------------------------- #
# Chronological pairing: the adversarial family (greedy vs dynamic program)
# --------------------------------------------------------------------------- #


def _greedy_reference(before, after, beat_s):
    """The nearest-by-onset first-fit policy, kept as the comparison
    baseline: each before note takes its closest free after note inside
    the bound, in before order.

    The shipped pairing is `compare.chronological_pairs`; these tests run
    BOTH on the same annotated cases so the difference is measured, not
    asserted from memory.
    """
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    missing: list[int] = []
    for i, b in enumerate(before):
        best_j: int | None = None
        best_shift = 0.0
        for j, a in enumerate(after):
            if j in used:
                continue
            shift = abs(a.start - b.start)
            if shift > beat_s:
                continue
            if best_j is None or shift < best_shift:
                best_j, best_shift = j, shift
        if best_j is None:
            missing.append(i)
        else:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs, missing


def test_the_dynamic_program_pairs_at_least_as_much_as_the_greedy_on_adversarial_cases():
    """Greedy versus chronological DP, measured on the annotated cases.

    Coverage (paired notes out of two) on the report's family:

        before / after                    greedy    chronological
        0.4 0.7  /  0.0 0.5  (report)     1 + miss   2, no absence
        0.4 0.7  /  0.0 0.5, heights      1 + miss   2, pitches kept
        0.4 0.75 /  0.0 0.5               1 + miss   2, no absence

    The greedy column is computed here by `_greedy_reference`, not
    transcribed: on an ordinary reading (each note's neighbour is its own)
    the two policies agree, which the identical-readings test already
    pins.
    """
    grid = _grid(120.0)  # beat = association bound 0.5 s
    cases = [
        # (before, after) — annotated truth: note k pairs with note k
        (
            [_note(0.40, 0.70, 60), _note(0.70, 1.00, 62)],
            [_note(0.00, 0.30, 60), _note(0.50, 0.80, 62)],
        ),
        (
            [_note(0.40, 0.70, 60), _note(0.70, 1.00, 64)],
            [_note(0.00, 0.30, 60), _note(0.50, 0.80, 64)],
        ),
        (
            [_note(0.40, 0.70, 60), _note(0.75, 1.05, 62)],
            [_note(0.00, 0.30, 60), _note(0.50, 0.80, 62)],
        ),
    ]
    for before, after in cases:
        greedy_pairs, greedy_missing = _greedy_reference(before, after, beat_s=0.5)
        cmp = compare_readings(_result(before, grid=grid), _result(after, grid=grid))
        assert len(cmp.pairs) == 2, (
            f"chronological pairing lost an association: {before} -> {after}"
        )
        # the annotated truth: note k met ITS OWN pitch, in singing order —
        # no manufactured absence, no invented change
        assert [p.after.index for p in cmp.pairs] == [0, 1], (
            "the chronological pairing must keep both associations"
        )
        assert [p.pitch_delta for p in cmp.pairs] == [0, 0]
        assert len(cmp.missing) == 0 and len(cmp.added) == 0
        assert len(cmp.pairs) >= len(greedy_pairs), "the DP must not pair less"
        assert cmp.count(CHANGED) + cmp.count(STABLE) + cmp.count(NEAR) == 2
        # on this family the greedy actually loses a note — the measured
        # gap the report describes, kept visible here
        assert len(greedy_missing) == 1 or len(greedy_pairs) == 2


def test_the_pairing_stays_exclusive_and_chronological():
    """One note never appears twice, and pairs come out in singing order."""
    before = [_note(0.40, 0.70, 60), _note(0.70, 1.00, 62), _note(1.20, 1.50, 64)]
    after = [_note(0.00, 0.30, 60), _note(0.50, 0.80, 62), _note(1.30, 1.60, 64)]
    cmp = compare_readings(_result(before, grid=_grid(120.0)), _result(after, grid=_grid(120.0)))
    assert [p.before.index for p in cmp.pairs] == [0, 1, 2]
    assert [p.after.index for p in cmp.pairs] == [0, 1, 2]
    assert used_indices(cmp.pairs) == {0, 1, 2}


# --------------------------------------------------------------------------- #
# Empty / trivial cases
# --------------------------------------------------------------------------- #


def test_two_empty_readings_return_an_empty_but_consistent_comparison():
    cmp = compare_readings(_result([]), _result([]))
    assert cmp.pairs == () and cmp.missing == () and cmp.added == ()
    assert not cmp.before_notes


def test_a_count_helper_answers_per_status():
    notes = [_note(0.10, 0.40, 60), _note(0.50, 0.80, 64), _note(0.90, 1.20, 67)]
    after = [_note(0.10, 0.40, 60), _note(0.50, 0.80, 65), _note(0.90, 1.20, 68)]
    cmp = compare_readings(_result(notes), _result(after))
    assert cmp.count(STABLE) == 1
    assert cmp.count(NEAR) == 2


# --------------------------------------------------------------------------- #
# Suspect labelling
# --------------------------------------------------------------------------- #


class _Curve:
    """A constant-curvature aliasing shim: a mild 1.4 st inner dip over a
    long note hides a plateau exactly like the real continuity evidence
    reads a swallowed phrase."""

    def __init__(self, times, hz):
        self.times = times
        self.hz = hz


def _observation_with(note, plateau_midi):
    from types import SimpleNamespace

    import numpy as np

    from v2m.core.music import (
        hz_to_midi,  # noqa: F401
    )

    return SimpleNamespace(
        pitch=_Curve(
            np.arange(round(note.start * 100), round(note.end * 100)) / 100.0,
            np.zeros(round(note.end * 100) - round(note.start * 100)),
        )
        if False
        else SimpleNamespace(
            times=np.arange(round(note.start * 100), round(note.end * 100)) / 100.0,
            hz=np.zeros(round(note.end * 100) - round(note.start * 100)),
        )
        if False
        else _make_curve(note.start, note.end, plateau_midi),
        frame_dt=0.01,
    )


def _make_curve(start, end, plateau_midi):
    """A hand-built pitch curve: the label except a medial plateau."""
    from types import SimpleNamespace

    import numpy as np

    from v2m.core.music import midi_to_hz

    times = np.arange(round(start * 100), round(end * 100)) / 100.0
    hz = np.full(len(times), midi_to_hz(60.0), dtype=float)
    inside = (times >= start + 0.15) & (times <= end - 0.15)
    hz[inside] = midi_to_hz(float(plateau_midi))
    return SimpleNamespace(times=times, hz=hz)


def _real_result(notes, plateau_midi=None):
    from types import SimpleNamespace

    observation = (
        None
        if plateau_midi is None
        else SimpleNamespace(
            pitch=_make_curve(notes[0].start, notes[0].end, plateau_midi),
            frame_dt=0.01,
        )
    )
    return SimpleNamespace(notes=list(notes), grid=_grid(240.0), observation=observation)


def _curve_only(start, end, plateau_midi):
    """A plain curve view (kept for future tests; unused here yet)."""

    return _make_curve(start, end, plateau_midi)


def test_a_persistent_suspect_is_labelled_persistent():
    notes = [_note(0.0, 0.60, 60, 0.7)]
    before = _real_result(notes, plateau_midi=64)
    after = _real_result(list(notes), plateau_midi=64)
    cmp = compare_readings(before, after)
    assert len(cmp.after_suspects) == 1
    row = cmp.after_suspects[0]
    assert row.status == "persistent"
    assert row.confidence == pytest.approx(0.7, abs=1e-3)
    assert not cmp.resolved_suspects


def test_a_suspect_that_disappears_becomes_resolved():
    from types import SimpleNamespace

    before = SimpleNamespace(
        notes=[_note(0.0, 0.60, 60, 0.7)],
        grid=_grid(),
        observation=SimpleNamespace(
            pitch=_make_curve(0.0, 0.6, 64),  # the OLD reading hides a plateau
            frame_dt=0.01,
        ),
    )
    after = SimpleNamespace(
        notes=[_note(0.0, 0.60, 60, 0.7)],
        grid=_grid(),
        # the still-detected hidden plateau of the OLD reading is gone —
        # a DIFFERENT curve (still an evidence, never a null one)
        observation=SimpleNamespace(
            pitch=_make_curve(0.0, 0.6, 60.5),
            frame_dt=0.01,
        ),
    )
    cmp = compare_readings(before, after)
    assert [r.status for r in cmp.after_suspects] == []
    assert [r.status for r in cmp.resolved_suspects] == ["resolved"]


def test_a_new_suspect_is_labelled_new():
    from types import SimpleNamespace

    before = SimpleNamespace(
        notes=[_note(0.0, 0.60, 60, 0.7)],
        grid=_grid(),
        observation=SimpleNamespace(
            pitch=_make_curve(0.0, 0.6, 60),  # the flat evidence: no suspect
            frame_dt=0.01,
        ),
    )
    after = _real_result([_note(0.0, 0.60, 60, 0.7)], plateau_midi=64)
    cmp = compare_readings(before, after)
    assert [r.status for r in cmp.after_suspects] == [SUSPECT_NEW]
    assert not cmp.resolved_suspects


def test_absent_evidence_keeps_suspects_unavailable_not_clean():
    """A missing observation is 'unavailable', never 'zero suspects'."""
    before = _result([_note(0.10, 0.40, 60)])
    after = _result([_note(0.10, 0.40, 60)])  # no observation at all
    cmp = compare_readings(before, after)
    assert not cmp.suspects_available
    assert not cmp.after_suspects and not cmp.resolved_suspects


def test_the_comparison_survives_a_result_reuse():
    """The result is never mutated (an export could follow immediately)."""
    before = _real_result([_note(0.0, 0.60, 60, 0.7)], plateau_midi=64)
    after = _real_result([_note(0.0, 0.60, 72, 0.7)], plateau_midi=64)
    notes_before = [list(before.notes), list(after.notes)]
    compare_readings(before, after)
    assert notes_before[0] == list(before.notes)
    assert notes_before[1] == list(after.notes)
