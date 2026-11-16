"""The Voice observation -> Interpretation seam (workstream A).

What this file pins, and nothing pinned before:

- the interpretation can be exercised on a FABRICATED pitch curve, with
  no audio and no pYIN — that is the seam's reason for being: the
  segmentation rules (a hold, two plateaus, a re-attack) finally test
  without paying for the whole pipeline;
- the tuning offset is measured by the interpretation AND read back on
  the object (the old channel was an output dict);
- `from_signal` produces ALL the evidence the suite consumes (the frame
  clock, the accent envelope, the transition mask);
- the interface lost `sr` (never read) and the `tuning_out` side
  channel.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from conftest import SR, sine_note

from v2m.core.notes import interpret
from v2m.core.observation import VoiceObservation
from v2m.core.onsets import ACCENT_HOP, Attacks
from v2m.core.pitch import PitchTrack

DT = 0.02  # one frame = 20 ms (the offline hop)


def _curve(midis: list[float], dt: float = DT) -> PitchTrack:
    hz = 440.0 * 2.0 ** ((np.asarray(midis, dtype=float) - 69.0) / 12.0)
    times = np.arange(len(hz)) * dt
    return PitchTrack(times=times, hz=hz, confidence=np.ones(len(hz)))


def _observation(midis: list[float], attacks: Attacks | None = None) -> VoiceObservation:
    """An observation from a fabricated curve: no audio, no pYIN, no cleanup."""
    return VoiceObservation(
        pitch=_curve(midis),
        attacks=attacks,
        duration_s=len(midis) * DT,
        sample_rate=22_050,
        frame_dt=DT,
    )


def test_a_hand_built_curve_is_enough_to_interpret():
    """A hand-built observation is sufficient to run the interpretation rules."""
    held = interpret(_observation([60] * 100))
    assert [(n.midi, round(n.start, 2)) for n in held] == [(60, 0.0)]

    two = interpret(_observation([60] * 50 + [67] * 50))
    assert [n.midi for n in two] == [60, 67]

    # an attack INSIDE a held note is the re-articulation evidence: it cuts
    attack = Attacks(times=np.array([1.0]), strengths=np.array([1.0]))
    repeated = interpret(_observation([60] * 100, attacks=attack))
    assert [n.midi for n in repeated] == [60, 60]
    assert repeated[1].start > 0.9  # the cut sits at the attack


def test_the_tuning_offset_is_measured_and_read_back():
    """A take 30 cents flat reads ON the grid, and the number is on the object."""
    flat = [m - 0.30 for m in (60, 62, 64, 65, 67)]
    sequence: list[float] = []
    for midi in flat:
        sequence += [midi] * 30
    observation = _observation(sequence)
    notes = interpret(observation)
    assert [n.midi for n in notes] == [60, 62, 64, 65, 67]
    assert observation.tuning_offset_st == pytest.approx(-0.30, abs=0.01)
    # the field survives a second read: interpretation is idempotent here
    assert [n.midi for n in interpret(observation)] == [60, 62, 64, 65, 67]


def test_from_signal_builds_the_whole_evidence():
    """One door produces the curve, the attacks, the clock and the envelope."""
    y = np.concatenate([sine_note(440.0, 0.6), np.zeros(int(0.2 * SR), dtype=np.float32)])
    observation = VoiceObservation.from_signal(y, SR)
    assert observation.sample_rate == SR
    assert observation.frame_dt > 0
    assert observation.attacks is not None
    # the curve is CLEANED here: the transition mask exists (clean_pitch ran)
    assert observation.pitch.transition is not None
    # the accent envelope is one value per 256 samples, computed on the signal
    assert observation.accent_envelope is not None
    assert observation.accent_hop == ACCENT_HOP
    assert len(observation.accent_envelope) == len(y) // ACCENT_HOP
    # geometry only, when the caller asks for it
    assert VoiceObservation.from_signal(y, SR, use_attacks=False).attacks is None


def test_the_interface_carries_no_dead_rate_and_no_side_channel():
    """`sr` was never read; the tuning offset travelled in a dict."""
    params = list(inspect.signature(interpret).parameters)
    assert params == ["observation", "min_note_s", "gap_s", "semitones"]
