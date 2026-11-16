"""The coherence check (continuity.py) and a take's verification.

Two contracts: (1) a note hiding a secondary plateau must appear in the
report (the doubt made visible, never silently rewritten); (2) a live
take's artifacts must describe the SAME signal — never another take's
WAV under this take's name.
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


from v2m.core.continuity import PLATEAU_ST, coherence_report  # noqa: E402
from v2m.core.observation import VoiceObservation  # noqa: E402

# --------------------------------------------------------------------------- #
# The secondary plateau is seen, the healthy hold passes
# --------------------------------------------------------------------------- #


def _hand_built_observation(notes_spec, sr=22_050, frame_s=0.02):
    """An observation whose curve follows `notes_spec` (midi, duration), no
    audio: the secondary plateau's contract pins directly on the curve
    (the test_observation harness already does it, it is the same road)."""
    times, hz = [], []
    t = 0.0
    for midi, dur_s in notes_spec:
        n = max(1, int(dur_s / frame_s))
        for _ in range(n):
            times.append(t)
            hz.append(440.0 * 2.0 ** ((midi - 69) / 12.0))
            t += frame_s
    from v2m.core.pitch import PitchTrack

    track = PitchTrack(
        times=np.asarray(times),
        hz=np.asarray(hz),
        confidence=np.ones(len(times)),
        transition=np.zeros(len(times), dtype=bool),
    )
    return VoiceObservation(
        pitch=track,
        attacks=None,
        duration_s=t,
        sample_rate=sr,
        frame_dt=frame_s,
        accent_envelope=None,
    )


def _interpret_simple(observation):
    from v2m.core.notes import interpret

    return interpret(observation)


def test_a_held_note_has_no_suspect():
    obs = _hand_built_observation([(60, 0.6), (62, 0.6)])
    notes = _interpret_simple(obs)
    rep = coherence_report(obs, notes)
    assert rep["checked"] >= 2
    assert rep["suspects"] == []


def test_a_swallowed_phrase_is_reported():
    """The user's case: two re-sung notes read as one hold.

    A 1.0 s note labeled 60 whose curve moves to 67 in the middle: the
    check must SHOW the 67 plateau hidden inside, with its pitch and its
    duration — whatever the interpreter decided.
    """
    obs = _hand_built_observation([(60, 0.5), (67, 0.5)])
    from v2m.core.notes import Note

    glued = Note(start=0.0, end=1.0, midi=60, confidence=0.9)
    rep = coherence_report(obs, [glued])
    assert rep["checked"] == 1
    assert rep["suspects"], "a 0.5 s secondary plateau must be reported"
    s = rep["suspects"][0]
    assert s["secondary_plateau_midi"] == 67
    assert s["secondary_plateau_s"] >= 0.3
    assert s["inner_range_st"] >= PLATEAU_ST


def test_secondary_plateau_keeps_its_time_after_missing_pitch():
    from v2m.core.notes import Note

    obs = _hand_built_observation([(60, 0.4), (64, 0.2), (60, 0.2)])
    obs.pitch.hz[:10] = np.nan
    note = Note(start=0.0, end=0.8, midi=60, confidence=0.9)
    suspects = coherence_report(obs, [note])["suspects"]
    assert len(suspects) == 1
    assert suspects[0]["secondary_plateau_start_s"] == 0.4
    assert suspects[0]["secondary_plateau_frames"] == 10
    assert suspects[0]["secondary_plateau_s"] == 0.2
    assert suspects[0]["secondary_plateau_midi"] == 64


def test_missing_pitch_separates_short_secondary_fragments():
    from v2m.core.notes import Note

    obs = _hand_built_observation([(60, 0.8)])
    obs.pitch.hz[10:14] = 440.0 * 2.0 ** ((64 - 69) / 12.0)
    obs.pitch.hz[14:17] = np.nan
    obs.pitch.hz[17:21] = 440.0 * 2.0 ** ((64 - 69) / 12.0)
    note = Note(start=0.0, end=0.8, midi=60, confidence=0.9)
    assert coherence_report(obs, [note])["suspects"] == []


def test_longest_plateau_after_several_holes_uses_original_clock():
    from v2m.core.notes import Note

    obs = _hand_built_observation([(60, 1.0)])
    high = 440.0 * 2.0 ** ((64 - 69) / 12.0)
    obs.pitch.hz[5:7] = np.nan
    obs.pitch.hz[10:14] = high
    obs.pitch.hz[14:17] = np.nan
    obs.pitch.hz[17:27] = high
    note = Note(start=0.0, end=1.0, midi=60, confidence=0.9)
    suspects = coherence_report(obs, [note])["suspects"]
    assert len(suspects) == 1
    assert suspects[0]["secondary_plateau_start_s"] == 0.34
    assert suspects[0]["secondary_plateau_frames"] == 10
    assert suspects[0]["secondary_plateau_s"] == 0.2
    assert suspects[0]["inner_range_st"] == 4.0


# --------------------------------------------------------------------------- #
# A take's verification: the WAV and the JSON tell the same story
# --------------------------------------------------------------------------- #


def test_take_verification_wav_matches_json(tmp_path):
    """`verify_take` validates a take's pair and refuses a mismatch."""
    import soundfile as sf

    from v2m.core.verify import verify_take

    sr = 22_050
    t = np.arange(int(1.5 * sr)) / sr
    y = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    wav = tmp_path / "prise.wav"
    sf.write(str(wav), y, sr)
    rep_path = tmp_path / "prise.json"
    rep_path.write_text('{"duration_s": 1.5, "n_notes": 3}', encoding="utf-8")
    ok, lines = verify_take(str(tmp_path / "prise"))
    assert ok and any("1.5" in line for line in lines)

    # a JSON describing another duration is REFUSED
    rep_path.write_text('{"duration_s": 9.94, "n_notes": 3}', encoding="utf-8")
    ok, lines = verify_take(str(tmp_path / "prise"))
    assert not ok and any("ECART" in line for line in lines)
