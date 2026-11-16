"""The glide corpus: pure slides, both directions, one note — never an arpeggio.

A glide is ONE note at the pitch it ARRIVES at (the live path reads it
that way; this corpus pins the same verdict in the final pass). The cases
cover both directions, both registers, with and without an arrival
plateau, the take-opening slide (no note before it), a different note
after a real break, and seed determinism. A pure slide engraved as more
than one note is the defect; a swallowed arrival or a swallowed following
note is the opposite defect the same tests watch for.
"""

from __future__ import annotations

import numpy as np
import pytest
from voicetruth import SungNote, detect_notes, glissando, synthesize


def _slide_with_arrival(frm: int, to: int, dur_s: float):
    slide = glissando(frm, to, dur_s=dur_s)
    arrival = SungNote(to, slide[-1].start_s + slide[-1].dur_s, 0.6)
    return slide + [arrival]


@pytest.mark.parametrize(
    ("frm", "to", "dur"),
    [(48, 59, 0.9), (59, 48, 0.9), (60, 71, 0.9), (71, 60, 0.9), (52, 64, 1.2)],
)
def test_a_slide_with_an_arrival_is_one_note(frm: int, to: int, dur: float):
    """Up AND down, low AND high: the slide merges into its arrival —
    one note at the arrival's pitch, no fragments on the way."""
    notes = _slide_with_arrival(frm, to, dur)
    detected = detect_notes(synthesize(notes, consonant_s=0.05, glide_s=0.03))
    assert len(detected) == 1, (
        f"slides must read as one note: {[(d.midi, round(d.end - d.start, 2)) for d in detected]}"
    )
    assert detected[0].midi == to


@pytest.mark.parametrize(
    ("frm", "to", "dur"),
    [(48, 59, 0.77), (48, 59, 0.9), (64, 50, 1.0), (48, 71, 1.5)],
)
def test_a_pure_slide_without_plateau_stays_one_note(frm: int, to: int, dur: float):
    """The take-opening slide (no note before it, no plateau after it):
    its pitch is where it ARRIVES — one note, never an arpeggio, never a
    zero-note silence (the slide was sung, it must be written)."""
    notes = glissando(frm, to, dur_s=dur)
    detected = detect_notes(synthesize(notes, consonant_s=0.05, glide_s=0.03))
    assert len(detected) == 1
    assert detected[0].midi == to


def test_a_slide_followed_by_a_different_note_gives_two_notes():
    """The slide rules must not swallow a REAL note: a slide that LANDS
    on 59, then a re-attacked 62 after the breath, is two notes."""
    slide_arrival = _slide_with_arrival(48, 59, 0.7)
    last = slide_arrival[-1]
    nxt = SungNote(62, last.start_s + last.dur_s + 0.18, 0.6)
    detected = detect_notes(
        synthesize(slide_arrival + [nxt], consonant_s=0.05, glide_s=0.03, seed=2)
    )
    midis = [d.midi for d in detected]
    assert 59 in midis and 62 in midis, f"the real note after the slide vanished: {midis}"


@pytest.mark.parametrize("seed", range(3))
def test_a_slide_reads_deterministically_across_seeds(seed: int):
    """The same slide at several seeds reads the same shape: the corpus
    must not depend on a burst's luck."""
    notes = _slide_with_arrival(55, 64, 0.8)
    detected = detect_notes(synthesize(notes, consonant_s=0.05, glide_s=0.03, seed=seed))
    assert len(detected) == 1 and detected[0].midi == 64


# ---------------------------------------------------------------------------
# The STAIRCASE slide: a slow glide the tracker reads as HELD mini-plateaus
# chained by fast ramps (one breath, no consonant between steps). Pinned on
# hand-built curves — no audio, no pYIN — with the real take's measured
# medians (47.9 -> 51.5 -> 56.1 -> 59.2, plateaus 0.14-0.20 s, steps 3.1-4.6
# st, all contiguous).
# ---------------------------------------------------------------------------

DT = 0.02  # one frame = 20 ms (the offline hop)


def _staircase(midis: list[float]) -> np.ndarray:
    """A curve of HELD plateaus joined by 3-frame climbing ramps."""
    curve: list[float] = []
    for k, m in enumerate(midis):
        curve += [m] * 10
        if k + 1 < len(midis):
            curve += list(np.linspace(m, midis[k + 1], 3))
    return np.asarray(curve, dtype=float)


def _observation(midis: np.ndarray):
    """An observation from a hand-built curve: no audio, no pYIN."""
    from v2m.core.observation import VoiceObservation
    from v2m.core.pitch import PitchTrack

    hz = 440.0 * 2.0 ** ((midis - 69.0) / 12.0)
    track = PitchTrack(times=np.arange(len(hz)) * DT, hz=hz, confidence=np.ones(len(hz)))
    return VoiceObservation(
        pitch=track,
        attacks=None,
        duration_s=len(hz) * DT,
        sample_rate=22_050,
        frame_dt=DT,
    )


@pytest.mark.parametrize(
    "midis, expected",
    [
        ([47.9, 51.5, 56.1, 59.2], [59]),  # the take's shape, ascending
        ([59.2, 56.1, 51.5, 47.9], [48]),  # descending
    ],
)
def test_a_staircase_slide_is_one_note_at_the_arrival(midis, expected):
    """Three plateaus and up, each a THIRD or more apart, contiguous and
    monotonic: the whole staircase is ONE note at the arrival."""
    from v2m.core.notes import interpret

    detected = interpret(_observation(_staircase(midis)))
    assert [n.midi for n in detected] == expected, (
        f"the staircase engraved as steps: {[(n.midi, round(n.duration, 2)) for n in detected]}"
    )


def test_an_articulated_third_arp_keeps_all_notes_across_seeds():
    """An acoustic onset at each third prevents a melody becoming one slide."""
    from v2m.core.notes import interpret
    from v2m.core.observation import VoiceObservation

    truth = [SungNote(60, 0.2, 0.2), SungNote(64, 0.4, 0.2), SungNote(67, 0.6, 0.2)]
    for seed in range(3):
        signal = synthesize(truth, consonant_s=0.05, glide_s=0.03, seed=seed)
        observation = VoiceObservation.from_signal(signal, 22_050)
        assert observation.attacks is not None
        assert len(observation.attacks.times) == 3
        assert [note.midi for note in interpret(observation)] == [60, 64, 67]


def test_a_two_plateau_leap_keeps_its_notes():
    """ONE link is not a staircase: two short plateaus a third apart (the
    hand-over shape of a fast melody) keep their notes — a chain needs
    three members before the slide verdict applies."""
    from v2m.core.notes import interpret

    detected = interpret(_observation(_staircase([60.0, 64.0])))
    assert [n.midi for n in detected] == [60, 64]


def test_a_contiguous_melody_of_real_notes_keeps_them():
    """The boundary chant.wav pins: contiguous short plateaus stepping
    2.8 st (under `SLIDE_STEP_ST`, its descending thirds measured at
    2.75-2.90) are real notes, even chained."""
    from v2m.core.notes import interpret

    detected = interpret(_observation(_staircase([63.2, 60.4, 57.6])))
    assert [n.midi for n in detected] == [63, 60, 57]
