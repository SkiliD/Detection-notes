"""Tests for 3/4 vs 4/4 discrimination from onset accents (meter.py)."""

from v2m.core.meter import detect_meter
from v2m.core.notes import Note

BPM = 120.0
BEAT = 60.0 / BPM  # 0.5 s


def _make_meter_notes(beat_pcs: list[tuple[int, ...]], strengths: list[float],
                      base: int = 60) -> list[Note]:
    """One chord per beat: the whole attack shares the beat's strength.

    A strummed/hummed chord attacks together: every note of the beat gets
    the beat's accent (the beat, not the individual note, is accented).
    """
    notes: list[Note] = []
    t = 0.0
    for i, pcs in enumerate(beat_pcs):
        s = strengths[i]
        for pc in pcs:
            notes.append(Note(start=t, end=t + BEAT * 0.9,
                              midi=base + pc, confidence=0.9, strength=s))
        t += BEAT
    return notes


def test_waltz_detected_as_3_4():
    # 8 bars of 3/4: strong downbeat every 3 beats
    beat_pcs: list[tuple[int, ...]] = []
    strengths: list[float] = []
    for _ in range(8):
        beat_pcs += [(0, 4, 7), (7,), (7,)]      # I - V - V pattern
        strengths += [1.0, 0.3, 0.3]
    notes = _make_meter_notes(beat_pcs, strengths)
    meter = detect_meter(notes, BPM)
    assert meter is not None
    assert meter.label == "3/4"


def test_common_time_detected_as_4_4():
    # 6 bars of 4/4: strong downbeat every 4 beats
    beat_pcs: list[tuple[int, ...]] = []
    strengths: list[float] = []
    for _ in range(6):
        beat_pcs += [(0, 4, 7), (7,), (9,), (7,)]  # I - V - vi - V
        strengths += [1.0, 0.3, 0.3, 0.3]
    notes = _make_meter_notes(beat_pcs, strengths)
    meter = detect_meter(notes, BPM)
    assert meter is not None
    assert meter.label == "4/4"


def _pulse(strengths: list[float]) -> list[Note]:
    """A pulse of notes (one per beat) repeating the given profile.

    Repeated up to 12 notes: the 2/4 parity reading demands EVERY phase
    of the 4-beat grid be populated enough (MIN_DOWNBEATS) that its
    absence cannot mask the pattern.
    """
    n_reps = -(-12 // len(strengths))  # ceil(12 / len)
    return _make_meter_notes([(0,)] * (len(strengths) * n_reps), strengths * n_reps)


def test_march_pattern_is_2_4():
    """Strictly alternating strengths (period 2): the meter is 2/4.

    The nested grids cannot separate 2/4 from 4/4 (half bar); the
    strengths' period can — it is the proof that breaks the tie. Both
    strong beats must be comparable: a 4/4 with a hierarchy (beat 1
    primary, beat 3 secondary) keeps 4/4.
    """
    meter = detect_meter(_pulse([1.0, 0.15]), 120.0)
    assert meter is not None and meter.label == "2/4"


def test_unequal_strong_beats_stay_4_4():
    """Unequal strong beats (beat 1 primary, beat 3 a weaker secondary):
    the hierarchy between the even phases is the 4-beat signature — not
    2/4.
    """
    meter = detect_meter(_pulse([1.0, 0.3, 0.6, 0.3]), 120.0)
    assert meter is not None and meter.label == "4/4"


def test_backbeat_keeps_4_4():
    """Loud pulse on the WEAK beats: no 2/4 — the bars would start on a
    weak beat and the grid has no measure phase; the 4/4 default is non
    destructive.
    """
    meter = detect_meter(_pulse([0.15, 1.0]), 120.0)
    assert meter is not None and meter.label == "4/4"


def test_flat_accents_stay_undecided():
    # every beat equally strong: timing cannot decide, None keeps 4/4 default
    beat_pcs = [(0, 4, 7)] * 12
    strengths = [0.6] * 12
    notes = _make_meter_notes(beat_pcs, strengths)
    assert detect_meter(notes, BPM) is None


def test_strength_is_filled_by_pipeline():
    from v2m.core.onsets import compute_onset_strengths

    notes = [Note(start=0.0, end=0.4, midi=60, confidence=0.9),
             Note(start=0.5, end=0.9, midi=64, confidence=0.9)]
    import numpy as np

    from v2m.core.observation import VoiceObservation

    sr = 22050
    y = np.zeros(int(1.0 * sr), dtype=np.float32)
    # loud burst at t=0, quiet at t=0.5 -> relative strengths differ
    y[: int(0.1 * sr)] = 0.9
    y[int(0.5 * sr): int(0.55 * sr)] = 0.1
    # the accent envelope now travels on the observation (it owns the hop and
    # the rate), so the strength pass takes the evidence, not the audio
    observation = VoiceObservation.from_signal(y, sr, use_attacks=False)
    compute_onset_strengths(observation, notes)
    assert 0.0 <= notes[0].strength <= 1.0
    assert 0.0 <= notes[1].strength <= 1.0
    assert notes[0].strength > notes[1].strength
