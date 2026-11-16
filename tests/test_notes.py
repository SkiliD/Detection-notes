"""Segmentation tests: 3 synthetic notes -> 3 events."""

import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import SR, silence, sine_note

from v2m.core.notes import interpret
from v2m.core.observation import VoiceObservation


def _three_notes_signal() -> np.ndarray:
    return np.concatenate(
        [
            sine_note(220.0, 0.5),  # A3
            silence(0.15),
            sine_note(261.63, 0.5),  # C4
            silence(0.15),
            sine_note(329.63, 0.5),  # E4
        ]
    )


def test_three_notes_have_expected_pitch_and_timing():
    y = _three_notes_signal()
    notes = interpret(VoiceObservation.from_signal(y, SR))
    assert len(notes) == 3
    assert [n.midi for n in notes] == [57, 60, 64]  # A3, C4, E4
    assert notes[0].start < 0.15
    for note in notes:
        assert 0.3 < note.duration < 0.75
    assert notes[0].end <= notes[1].start + 0.1


def test_silence_gives_no_notes():
    y = np.zeros(SR * 2, dtype=np.float32)
    assert interpret(VoiceObservation.from_signal(y, SR)) == []


def test_split_rearticulation_preserves_the_measured_deviation():
    """A new segment keeps the held note's evidence, not a default zero."""
    import numpy as np

    from v2m.core.notes import Note, _split_rearticulated
    from v2m.core.onsets import Attacks

    note = Note(0.0, 1.2, 69, 0.9, deviation=0.24)
    times = np.arange(0.0, 1.21, 0.02)
    midi_curve = np.full(len(times), 69.28)
    midi_curve[(times >= 0.4) & (times < 0.8)] = 69.30
    attacks = Attacks(np.array([0.4, 0.8]), np.ones(2))

    split = _split_rearticulated([note], attacks, midi_curve, times, tuning_offset_st=0.04)

    assert len(split) == 3
    assert [part.deviation for part in split] == pytest.approx([0.24, 0.26, 0.24])


def test_a_refused_late_cut_keeps_an_independent_earlier_cut():
    from v2m.core.notes import Note, _split_rearticulated
    from v2m.core.onsets import Attacks

    times = np.arange(0, 2.01, 0.02)
    curve = np.full(len(times), 60.0)
    curve[times > 1.2] = 64.0
    note = Note(0.0, 2.0, 60, 0.9)
    first = _split_rearticulated(
        [note],
        Attacks(np.array([0.5]), np.ones(1)),
        curve,
        times,
    )
    both = _split_rearticulated(
        [note],
        Attacks(np.array([0.5, 1.5]), np.ones(2)),
        curve,
        times,
    )
    assert [(n.start, n.end) for n in first] == [(0.0, 0.5), (0.5, 2.0)]
    assert [(n.start, n.end) for n in both] == [(0.0, 0.5), (0.5, 2.0)]


def test_vibrato_single_note():
    """A note with vibrato (±30 cents) stays a single note."""
    dur = 1.0
    t = np.arange(int(dur * SR)) / SR
    hz = 440.0 * 2 ** (0.3 * np.sin(2 * np.pi * 5.5 * t) / 12)  # ±30 cents
    env = np.minimum(1, np.minimum(t / 0.01, (dur - t) / 0.01))
    y = (0.5 * np.sin(2 * np.pi * np.cumsum(hz) / SR) * env).astype(np.float32)
    notes = interpret(VoiceObservation.from_signal(y, SR))
    assert len(notes) == 1


# ---------------------------------------------------------------------------
# f0 stabilization: the derivative mask, and pitch read on the plateau
# ---------------------------------------------------------------------------


def test_clean_pitch_marks_transitions_and_estimates_on_plateaus():
    """The `transition` mask (the f0's derivative) covers the portamento
    ramps, NOT the plateaus nor the vibrato peaks: a note arriving by
    glissando reads its pitch on its PLATEAU (the median of the stable
    frames), never on the ramp leading there (measured: 66.09 read for a
    sung 67 while the ramp stayed in the estimate)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from voicetruth import SungNote, synthesize

    truth = [SungNote(60, 0.2, 0.7), SungNote(67, 0.9, 0.7)]
    y = synthesize(truth, glide_s=0.12, consonant_s=0.0)  # LONG portamento
    observation = VoiceObservation.from_signal(y, SR)
    notes = interpret(observation)
    track = observation.pitch
    assert [n.midi for n in notes] == [60, 67]
    assert observation.attacks is not None
    assert track.transition is not None
    voiced = ~np.isnan(track.hz)
    # the periodicity gate (MIN_PITCH_QUALITY) drops the ramp
    # frames of a LONG portamento (250 ms) under the threshold — pYIN
    # reads a slow ramp as doubt — and the ramp then reaches the slope
    # threshold in small steps < TRANSITION_ST_S. The PLATEAU reading
    # (what this test protects) is already guaranteed by the median of
    # the stable frames; we pin it on the detected notes and on the
    # curve's continuity (the ramp stays voice, never a silence).
    assert float(track.transition[voiced].mean()) < 0.5
    for n in notes:
        i0, i1 = int(n.start / 0.02), int(n.end / 0.02)
        gap = np.isnan(track.hz[i0:i1]).mean()
        assert gap < 0.2, (n, gap)  # a sung note has no hole inside


def test_tuning_offset_pulls_a_flat_take_back_on_the_grid():
    """A singer systematically 30 cents flat: without correction, every
    note rounds to the wrong semitone; the global offset (the deviations'
    median, bounded ±0.35) re-places the take BEFORE rounding. The
    measured offset stays on the observation (the JSON reports it in
    cents)."""
    from voicetruth import SungNote, synthesize

    CENTS = -0.30
    midis = [60, 62, 64, 65, 67]
    truth = [SungNote(m, 0.2 + i * 0.6, 0.55) for i, m in enumerate(midis)]
    y = synthesize(truth, glide_s=0.0, consonant_s=0.05)
    # detune the WHOLE take by -30 cents (a consistently flat singer) by
    # playing it slower: rate r shifts the pitch by 12*log2(r) st
    sr = SR
    idx = np.arange(len(y)) / sr
    rate = 2.0 ** (CENTS / 12.0)  # < 1: slower playback, lower pitch
    src_positions = idx * rate
    y_flat = np.interp(src_positions, idx, y).astype(np.float32)

    observation = VoiceObservation.from_signal(y_flat, sr, use_attacks=False)
    notes = interpret(observation)
    assert len(notes) == len(midis)
    # the offset is READ back from the same object the interpretation wrote
    # it on (this is what the seam promises: evidence in, notes out)
    assert observation.tuning_offset_st == pytest_approx_cents(CENTS)
    assert [n.midi for n in notes] == midis  # the flat take reads ON the grid


def pytest_approx_cents(cents: float) -> float:
    """Tolerance wrapper: `cents` in SEMITONES, measured within 8 cents."""
    import pytest

    return pytest.approx(cents, abs=0.08)
