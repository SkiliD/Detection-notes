"""The phrase-repeat engine (`core.phrase`): one net over two takes.

The comparison reuses the live/final matching policy take-to-take; these
tests pin the classification contract (same / off1 / shifted / moved +
missing/added), the tempo-scaled timing and the guarded report reader —
no audio, no decode, just the exported reports.
"""

from __future__ import annotations

import json

import pytest

from v2m.core.phrase import (
    TEMPO_TOLERANCE,
    compare_reports,
    compare_take_files,
    read_report,
)


def _report(midis, starts, bpm=120.0):
    """A minimal `export_result` schema: just what the engine reads."""
    return {
        "tempo": {"bpm": bpm},
        "notes": [
            {"midi": m, "start_s": s, "end_s": s + 0.4} for m, s in zip(midis, starts, strict=True)
        ],
    }


def test_a_missing_or_corrupt_report_is_none_never_a_guess(tmp_path):
    assert read_report(tmp_path / "absent.json") is None
    bad = tmp_path / "cassé.json"
    bad.write_text("{not json", encoding="utf-8")
    assert read_report(bad) is None
    # a JSON that is not a report object is equally unreadable
    (tmp_path / "liste.json").write_text("[1, 2]", encoding="utf-8")
    assert read_report(tmp_path / "liste.json") is None


def test_identical_takes_pair_as_same_and_are_stable():
    a = _report([60, 62, 64], [0.2, 0.9, 1.6])
    b = _report([60, 62, 64], [0.21, 0.88, 1.63])
    cmp = compare_reports(a, b)
    assert [p.reason for p in cmp.paired] == ["same", "same", "same"]
    assert cmp.stable and not cmp.missing and not cmp.added


def test_off1_pairing_absorbs_the_onset_shift():
    """The ±1 family: a semitone neighbour reads as the rounding family,
    whatever the small onset shift — it never counts as a time shift."""
    a = _report([64], [0.5])
    b = _report([65], [0.62])  # beyond the sixteenth window, same family
    cmp = compare_reports(a, b)
    assert cmp.paired[0].reason == "off1"
    assert cmp.stable  # the ±1 family does not fail the verdict


def test_same_pitch_beyond_the_window_is_shifted_in_beats():
    a = _report([64], [0.5], bpm=120.0)
    b = _report([64], [0.8], bpm=120.0)  # 0.30 s: hesitation, not rounding
    cmp = compare_reports(a, b)
    p = cmp.paired[0]
    assert p.reason == "shifted"
    # 0.30 s at 120 BPM = 0.6 quarter beats of B
    assert p.shift_beats == pytest.approx(0.6, abs=0.01)
    assert not cmp.stable


def test_pitch_change_beyond_one_semitone_is_moved():
    cmp = compare_reports(_report([64], [0.5]), _report([67], [0.5]))
    assert cmp.paired[0].reason == "moved"
    assert not cmp.stable


def test_count_mismatch_lands_in_missing_and_added():
    a = _report([60, 62, 64, 67], [0.2, 0.9, 1.6, 2.3])
    b = _report([60, 64], [0.2, 1.6])
    cmp = compare_reports(a, b)
    # every B note pairs (60->60, 64->64); the two extra A notes (62, 67)
    # have no free counterpart left -> missing from B, nothing added
    assert cmp.missing == [1, 3]
    assert cmp.added == []
    assert not cmp.stable
    # the reverse imbalance: an extra B note lands in `added`
    cmp_rev = compare_reports(b, a)
    assert cmp_rev.added == [1, 3] and cmp_rev.missing == []


def test_pairs_form_within_one_beat_of_b_never_a_fixed_second_bound():
    """The pairing bound is ONE BEAT of take B (60/bpm_b — 0.5 s at 120),
    not a fixed second count: at 120 BPM these two onsets sit 0.7 s apart
    — inside the old fixed 1.0 s bound (which mis-paired them as a moved
    couple), outside one beat (the note is missing and its counterpart
    added, the honest reading of a phrase the singer did not re-sing)."""
    a = _report([62], [0.9])
    b = _report([64], [1.6])
    cmp = compare_reports(a, b)
    assert cmp.missing == [0] and cmp.added == [0]
    assert cmp.paired == []
    # a FASTER take B keeps proportionally more pairs: the bound is its beat
    cmp_fast = compare_reports(
        _report([62], [0.9]), _report([64], [1.3]), stem_a="a", stem_b="fast"
    )
    assert cmp_fast.paired, "0.4 s apart pairs inside one beat of a faster B"


def _greedy_reference(starts_a, starts_b, bound):
    """The previous nearest-by-onset first-fit policy, as a baseline.

    Kept IN THE TEST so the shipped chronological pairing is compared
    against it on the same annotated inputs instead of against a
    remembered result.
    """
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    missing: list[int] = []
    for ia, start_a in enumerate(starts_a):
        best_ib: int | None = None
        best_shift = 0.0
        for ib, start_b in enumerate(starts_b):
            if ib in used_b:
                continue
            shift = abs(start_b - start_a)
            if shift > bound:
                continue
            if best_ib is None or shift < best_shift:
                best_ib, best_shift = ib, shift
        if best_ib is None:
            missing.append(ia)
        else:
            used_b.add(best_ib)
            pairs.append((ia, best_ib))
    return pairs, missing


def test_the_adversarial_onset_case_pairs_take_b_without_manufacturing_an_absence():
    """The report's ALG18 case, take-to-take: both notes have a partner.

    Take A sings at 0.4 and 0.7 s, take B at 0 and 0.5 s (one beat =
    0.5 s). Greedy first-fit takes A0's closest neighbour (0.4 -> 0.5)
    and leaves A1 with nothing inside the bound, reporting one missing
    and one added note; the annotated truth is A0 -> B0 (0.4 s) and
    A1 -> B1 (0.2 s), a complete pairing where both notes moved in time
    and no note disappeared.
    """
    a = _report([60, 62], [0.4, 0.7])
    b = _report([60, 62], [0.0, 0.5])
    cmp = compare_reports(a, b)
    assert [(p.a_index, p.b_index) for p in cmp.paired] == [(0, 0), (1, 1)]
    assert cmp.missing == [] and cmp.added == []
    assert [p.reason for p in cmp.paired] == ["shifted", "shifted"]
    # the same inputs through the greedy baseline: one association is
    # SPENT on A0's nearest neighbour and A1 reads as not re-sung
    greedy_pairs, greedy_missing = _greedy_reference([0.4, 0.7], [0.0, 0.5], 0.5)
    assert len(greedy_pairs) == 1 and greedy_missing == [1]
    assert len(cmp.paired) > len(greedy_pairs)


def test_tempo_difference_does_not_read_as_shifts():
    """The same phrase sung 4% faster: every onset scales, yet no note is
    'shifted' — the shift is judged in B's beats, within the window."""
    starts = [0.2, 0.9, 1.6, 2.3]
    bpm_a, bpm_b = 120.0, 125.0  # 4.2% faster, inside TEMPO_TOLERANCE
    a = _report([60, 62, 64, 67], starts, bpm=bpm_a)
    b = _report([60, 62, 64, 67], [s * bpm_a / bpm_b for s in starts], bpm=bpm_b)
    cmp = compare_reports(a, b)
    assert abs(cmp.tempo_ratio - 1.0) <= TEMPO_TOLERANCE
    assert [p.reason for p in cmp.paired] == ["same"] * 4


@pytest.mark.parametrize("bpm_b", [115.2, 124.8])
def test_long_phrase_compensates_tempo_in_both_directions(bpm_b):
    starts = [0.0, 2.0, 4.0, 6.0, 8.0]
    a = _report([60, 62, 64, 65, 67], starts)
    b = _report([60, 62, 64, 65, 67], [s * 120 / bpm_b for s in starts], bpm_b)
    for first, second in [(a, b), (b, a)]:
        result = compare_reports(first, second)
        assert result.stable
        assert len(result.paired) == 5
        assert all(p.shift_s == 0.0 for p in result.paired)


def test_tempo_compensation_keeps_an_initial_delay():
    a = _report([60, 62, 64], [0.0, 4.0, 8.0])
    b = _report([60, 62, 64], [s * 120 / 124.8 + 0.3 for s in [0.0, 4.0, 8.0]], 124.8)
    result = compare_reports(a, b)
    assert [p.reason for p in result.paired] == ["shifted"] * 3
    assert all(p.shift_s == 0.3 for p in result.paired)


def test_take_files_helper_reads_and_names_stems(tmp_path):
    ra = _report([60, 62], [0.2, 0.9])
    rb = _report([60, 62], [0.2, 0.95])
    pa = tmp_path / "live_20260929_161442.json"
    pb = tmp_path / "live_20260929_161537.json"
    pa.write_text(json.dumps(ra), encoding="utf-8")
    pb.write_text(json.dumps(rb), encoding="utf-8")
    cmp = compare_take_files(pa, pb)
    assert cmp is not None and cmp.stable
    assert cmp.stem_a == "live_20260929_161442"
    # one unreadable side -> None (the adapter prints which file)
    assert compare_take_files(pa, tmp_path / "nulle.json") is None
