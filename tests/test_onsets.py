"""onsets.py tests: offline and streaming attacks.

What these tests pin are the two measured errors: the detector counted 19
"attacks" on a single held note with vibrato (vibrato moves the
harmonics, hence the flux), and a level-only detector found nothing at
all on a legato line (the amplitude does not move from note to note).
"""

import numpy as np
import pytest
from conftest import SR
from voicetruth import SungNote, sequence, synthesize

from v2m.core.onsets import AttackTracker, detect_attacks

CONSONANT = 0.05
BLOCK = 1024


def _held(midi: int = 57, **kw) -> np.ndarray:
    """A 2 s held note, sung from 0.2 s."""
    return synthesize([SungNote(midi, 0.2, 1.8)], **kw)


def _stream_attacks(y: np.ndarray, sr: int = SR, block: int = BLOCK) -> list[float]:
    """Runs the signal through the real-time tracker (as the GUI does)."""
    tracker = AttackTracker()
    out: list[float] = []
    for i in range(0, len(y) - block, block):
        t = (i + block) / sr
        if tracker.push(t, y[i : i + block]):
            out.append(round(t, 3))
    return out


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #


def test_a_held_note_with_vibrato_gives_one_attack():
    """One attack per note: a vibrato cycle is not an attack (19 attacks
    on the held-with-vibrato signal would be one per cycle)."""
    attacks = detect_attacks(_held(vibrato_st=0.4, consonant_s=CONSONANT), SR)
    assert len(attacks) == 1
    assert attacks.times[0] == pytest.approx(0.2, abs=0.08)


def test_a_held_note_without_consonant_still_has_its_onset():
    attacks = detect_attacks(_held(), SR)
    assert len(attacks) == 1
    assert attacks.near(0.2, 0.08) is not None


def test_every_sung_onset_gets_an_attack():
    truth = sequence([60, 62, 64, 65, 67, 72], dur_s=0.6, start_s=0.2)
    y = synthesize(truth, glide_s=0.09, consonant_s=CONSONANT)
    attacks = detect_attacks(y, SR)
    for note in truth:
        assert attacks.near(note.start_s, 0.10) is not None


def test_no_attack_falls_inside_a_sung_note():
    """A false positive INSIDE a note is what manufactures micro-notes."""
    truth = sequence([60, 62, 64, 65, 67, 72], dur_s=0.6, start_s=0.2)
    y = synthesize(truth, glide_s=0.09, consonant_s=CONSONANT, vibrato_st=0.3)
    attacks = detect_attacks(y, SR)
    inside = [
        t
        for t in attacks.times
        if any(n.start_s + 0.10 < t < n.start_s + n.dur_s for n in truth)
    ]
    assert inside == []


def test_silence_has_no_attack():
    assert len(detect_attacks(np.zeros(SR, dtype=np.float32), SR)) == 0


def test_strengths_are_normalized_and_ordered():
    truth = sequence([60, 62, 64, 65], dur_s=0.5, start_s=0.2)
    attacks = detect_attacks(synthesize(truth, consonant_s=CONSONANT), SR)
    assert len(attacks) >= 2
    assert np.all(np.diff(attacks.times) > 0)  # chronological
    assert attacks.strengths.max() == pytest.approx(1.0)
    assert np.all((attacks.strengths > 0) & (attacks.strengths <= 1.0))


# --------------------------------------------------------------------------- #
# Streaming (real time)
# --------------------------------------------------------------------------- #


def test_streaming_tracker_fires_on_each_articulated_note():
    """The live tracker follows the LEVEL: every note re-taken after a
    silence (or re-struck louder) is an attack."""
    truth = sequence([60, 62, 64, 65], dur_s=0.5, gap_s=0.2, start_s=0.2)
    y = synthesize(truth, glide_s=0.0, consonant_s=CONSONANT)
    hits = _stream_attacks(y)
    for note in truth:
        assert any(abs(h - note.start_s) <= 0.14 for h in hits), note


def test_streaming_tracker_stays_quiet_inside_a_held_vibrato_note():
    """The test the spectral stream failed: a vibrato is not an attack.

    The note's own attack is legitimate; what follows must stay quiet
    (the detector used to find 11 of them in the hold).
    """
    hits = _stream_attacks(_held(vibrato_st=0.4, consonant_s=CONSONANT))
    assert all(abs(h - 0.2) <= 0.14 for h in hits), hits


def test_streaming_tracker_is_conservative_on_legato():
    """A constant-level legato line triggers nothing after the first note
    (accepted choice: a missed attack beats a false positive that would
    cut a held note in two)."""
    truth = sequence([60, 62, 64, 65, 67, 72], dur_s=0.6, start_s=0.2)
    y = synthesize(truth, glide_s=0.09, consonant_s=CONSONANT)
    hits = _stream_attacks(y)
    assert all(abs(h - 0.2) <= 0.14 for h in hits), hits


def test_streaming_tracker_does_not_fire_on_silence():
    assert _stream_attacks(np.zeros(SR, dtype=np.float32)) == []


def test_streaming_tracker_primes_on_the_first_block():
    """The first frame primes the baseline instead of firing."""
    tracker = AttackTracker()
    block = np.full(BLOCK, 0.25, dtype=np.float32)  # already loud
    assert tracker.push(0.05, block) is False
    assert tracker.baseline_rms > 0
    tracker.reset()
    assert tracker.baseline_rms == 0
    assert tracker.push(0.05, block) is False  # priming again


def test_streaming_tracker_keeps_a_minimum_gap():
    """Two loud blocks in a row count for one attack only."""
    tracker = AttackTracker()
    quiet = np.full(BLOCK, 0.002, dtype=np.float32)
    loud = np.full(BLOCK, 0.25, dtype=np.float32)
    for _ in range(6):
        tracker.push(0.0, quiet)
    assert tracker.push(0.1, loud) is True
    assert tracker.push(0.11, loud) is False  # < min_gap_s
