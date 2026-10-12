"""`core/mic_cal` pins the evidence gates (pure, no Qt): enough rounds
AND a big enough mean, else nothing — and a missed target never counts.
"""

from __future__ import annotations

import pytest

from v2m.core import mic_cal
from v2m.core.intonation import MIN_CENTS, MIN_NOTES


def _rounds_at(offset_cents: float, n: int = mic_cal.ROUNDS) -> tuple[list[float], list[float]]:
    """Targets and heard pitches sitting `offset_cents` apart."""
    targets = [440.0 * (2.0 ** (r / 12.0)) for r in range(n)]
    heard = [t * (2.0 ** (offset_cents / 1200.0)) for t in targets]
    return targets, heard


def test_a_consistent_run_names_the_offset():
    targets, heard = _rounds_at(-30.0)
    verdict = mic_cal.collect(heard, targets)
    assert verdict.recordable
    assert verdict.offset_cents == pytest.approx(-30.0, abs=0.5)
    assert verdict.n_used == mic_cal.ROUNDS
    assert verdict.n_skipped == 0


def test_a_scattered_run_offers_nothing():
    """Noise never calibrates: the mean is inside MIN_CENTS."""
    targets, heard = _rounds_at(0.0)
    # nudge rounds in alternating directions a few cents apart
    heard = [t * (2.0 ** ((5.0 if r % 2 else -5.0) / 1200.0)) for r, t in enumerate(targets)]
    verdict = mic_cal.collect(heard, targets)
    assert not verdict.recordable
    assert verdict.offset_cents is None


def test_too_few_used_rounds_offer_nothing():
    targets, heard = _rounds_at(-30.0)
    used_sliced = heard[: MIN_NOTES - 1]  # one round wasted drops below the gate
    verdict = mic_cal.collect(used_sliced, targets[: len(used_sliced)])
    assert not verdict.recordable
    assert verdict.n_used == len(used_sliced)


def test_a_missed_target_is_not_evidence_about_the_device():
    """A half-semitone-off round measures the singer, never the mic."""
    targets, heard = _rounds_at(-30.0)
    heard[0] = targets[0] * 2.0  # an octave off: useless
    verdict = mic_cal.collect(heard, targets)
    assert verdict.n_used == mic_cal.ROUNDS - 1


def test_an_alternating_voice_never_averages_into_a_habit():
    """A ±0.18-st alternating voice passes TARGET_ST but fails MIN_CENTS:
    the mean's gate refuses the lean, nothing is recordable."""
    targets, heard = _rounds_at(18.0)
    heard = [t * (2.0 ** ((-18.0 if r % 2 else 18.0) / 1200.0)) for r, t in enumerate(targets)]
    verdict = mic_cal.collect(heard, targets)
    assert not verdict.recordable
    assert verdict.offset_cents is None
    assert verdict.n_used == mic_cal.ROUNDS


def test_the_thresholds_come_from_intonation():
    """The calibration subscribes to the report's own gates (no fork)."""
    assert mic_cal.ROUNDS >= MIN_NOTES
    targets, heard = _rounds_at(MIN_CENTS - 0.1)
    assert not mic_cal.collect(heard, targets).recordable
    targets, heard = _rounds_at(MIN_CENTS)
    assert mic_cal.collect(heard, targets).recordable


def test_the_persistence_round_trips_and_a_bad_file_is_nothing(tmp_path, monkeypatch):
    cal = tmp_path / "cal.json"
    monkeypatch.setattr(mic_cal, "CAL_FILE", cal)
    assert mic_cal.read() is None  # absent: no calibration

    targets, heard = _rounds_at(24.0)
    verdict = mic_cal.collect(heard, targets)
    assert mic_cal.write(verdict, targets)
    saved = mic_cal.read()
    assert saved is not None and saved["offset_cents"] == pytest.approx(24.0, abs=0.5)

    cal.write_text("not json", encoding="utf-8")
    assert mic_cal.read() is None  # corrupt: falls back silently

    assert mic_cal.clear()
    assert mic_cal.read() is None

    # a refused verdict writes nothing
    targets, heard = _rounds_at(0.0)
    scatter = [t * (2.0 ** ((3.0 if r % 2 else -3.0) / 1200.0)) for r, t in enumerate(targets)]
    assert not mic_cal.write(mic_cal.collect(scatter, targets), targets)
