"""The take-so-far intonation feedback (the live twin of the report).

One design rule, shared with `intonation.py`: the feedback claims a
tendency only when the evidence carries it, and it reads the SAME
evidence table (`intonation.summarize`) the per-take report reads, so
the number watched while singing and the number reported afterwards can
never disagree.
"""

from __future__ import annotations

from v2m.core.intonation import IN_TUNE_CENTS, summarize
from v2m.core.intonation_live import LiveIntonation
from v2m.core.liveseg import LiveNote
from v2m.core.notes import Note


def _note(midi: int, dev_st: float, start: float = 0.0) -> LiveNote:
    return LiveNote(start=start, end=start + 0.3, midi=midi, dev=dev_st)


def test_the_share_counts_closed_notes_only():
    """The summary counts in-tune CLOSED notes on the report's scale.

    |dev| at the IN_TUNE_CENTS boundary counts as in tune (the same
    inclusive band the GUI meter paints green).
    """
    feed = LiveIntonation()
    assert feed.summary()["n_notes"] == 0  # no closed notes yet
    feed.add(_note(60, 0.0))
    feed.add(_note(62, IN_TUNE_CENTS / 100.0))
    feed.add(_note(64, 0.30))
    summary = feed.summary()
    assert summary["n_notes"] == 3
    assert summary["in_tune_share"] == round(2.0 / 3.0, 2)
    assert summary["tendency"] is None  # 3 notes: no habit to name yet


def test_a_cumulative_flat_lean_is_named_once_backed():
    """Four notes 25 cents flat: the tendency arrives and stays flat.

    Before MIN_NOTES the summary carries no tendency — the same right to
    say nothing the report has.
    """
    feed = LiveIntonation()
    for i in range(3):
        feed.add(_note(60, -0.25, start=i * 0.4))
        assert feed.summary()["tendency"] is None
    feed.add(_note(60, -0.25, start=1.2))
    summary = feed.summary()
    assert summary["tendency"] == "flat"
    assert summary["mean_cents"] == -25.0
    # a later well-sung note pulls the mean back: still flat, weaker
    feed.add(_note(64, 0.0, start=1.6))
    summary = feed.summary()
    assert summary["tendency"] == "flat"
    assert abs(summary["mean_cents"]) < 25.0


def test_alternating_deviations_read_even_not_sharp():
    """A back-and-forth of ±30 cents: zero mean, announced "even"."""
    feed = LiveIntonation()
    for i, d in enumerate((-0.30, 0.30) * 3):
        feed.add(_note(60 + i, d, start=i * 0.4))
    summary = feed.summary()
    assert summary["tendency"] == "even"
    assert summary["in_tune_share"] == 0


def test_reset_starts_a_fresh_take():
    """A new take must not inherit the previous one's history."""
    feed = LiveIntonation()
    feed.add(_note(60, 0.30))
    feed.reset()
    assert feed.summary()["n_notes"] == 0


def test_the_summary_is_the_report_its_own_shape():
    """summary() IS intonation.summarize's output (one evidence table)."""
    feed = LiveIntonation()
    for i, d in enumerate((-0.25,) * 5):
        feed.add(_note(60, d, start=i * 0.4))
    report = feed.summary()
    expected = summarize([Note(0.0, 0.3, 60, 0.9, deviation=-0.25) for _ in range(5)])
    assert report["tendency"] == expected["tendency"] == "flat"
    assert report["mean_cents"] == expected["mean_cents"] == -25.0
