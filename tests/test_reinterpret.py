"""The persisted voice observation: lossless round-trip, cold reinterpretation.

The "evidence / interpretation" work is only worth it if the evidence can be
REPLAYED: `VoiceObservation.save` / `load` must return exactly the same
curve, the same transition mask, the same attacks and the same envelope —
and the reinterpretation that follows must be identical. The second contract
is the reverse: re-reading an observation must NEITHER decode audio NOR
re-run pYIN (the very cost we seek to avoid). We prove it by making
`track_pitch_offline` and `load_file` FAIL: if the code calls them, the test
falls.
"""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from voicetruth import sequence, synthesize  # noqa: E402

from v2m.core import audio_io  # noqa: E402
from v2m.core.notes import interpret  # noqa: E402
from v2m.core.observation import FORMAT_VERSION, VoiceObservation  # noqa: E402
from v2m.core.pipeline import (  # noqa: E402
    AnalyzeOptions,
    analyze_file,
    interpret_observation,
)
from v2m.core.pitch import PitchTrack  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "reinterpret", ROOT / "scripts" / "reinterpret.py"
)
assert SPEC is not None and SPEC.loader is not None  # the script exists
reinterpret = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("reinterpret", reinterpret)
SPEC.loader.exec_module(reinterpret)


def _observation():
    truth = sequence([60, 62, 64, 65, 67], dur_s=0.5, gap_s=0.15,
                     legato_s=0.0)
    y = synthesize(truth)
    return VoiceObservation.from_signal(y, 22_050)


def _fingerprint(obs: VoiceObservation) -> list[list]:
    return [
        [round(float(n.start), 5), round(float(n.duration), 5), int(n.midi)]
        for n in interpret(obs)
    ]


# --------------------------------------------------------------------------- #
# Round-trip: save -> load returns the evidence unchanged
# --------------------------------------------------------------------------- #


def test_observation_round_trip_is_lossless(tmp_path):
    original = _observation()
    path = tmp_path / "take.observation.npz"
    original.save(path)
    loaded = VoiceObservation.load(path)

    assert loaded.sample_rate == original.sample_rate
    assert loaded.duration_s == pytest.approx(original.duration_s)
    assert loaded.frame_dt == pytest.approx(original.frame_dt)
    assert loaded.accent_hop == original.accent_hop
    assert loaded.tuning_offset_st == pytest.approx(original.tuning_offset_st)

    np.testing.assert_array_equal(loaded.pitch.times, original.pitch.times)
    np.testing.assert_array_equal(loaded.pitch.hz, original.pitch.hz)
    np.testing.assert_array_equal(
        loaded.pitch.confidence, original.pitch.confidence
    )
    np.testing.assert_array_equal(
        loaded.pitch.transition, original.pitch.transition
    )
    assert loaded.attacks is not None
    assert original.attacks is not None  # the synthesis carries consonants
    np.testing.assert_array_equal(loaded.attacks.times, original.attacks.times)
    np.testing.assert_array_equal(
        loaded.attacks.strengths, original.attacks.strengths
    )
    np.testing.assert_array_equal(
        loaded.accent_envelope, original.accent_envelope
    )

    # the proof that counts: the SAME interpretation, not only the arrays
    assert _fingerprint(loaded) == _fingerprint(original)


def test_save_keeps_the_absence_of_attacks_meaningful(tmp_path):
    """"geometry only" (attacks None) is not "no attack found" (empty)."""
    truth = sequence([60, 62, 64], dur_s=0.5, gap_s=0.15)
    geometry = VoiceObservation.from_signal(synthesize(truth), 22_050,
                                            use_attacks=False)
    assert geometry.attacks is None
    path = tmp_path / "geo.observation.npz"
    geometry.save(path)
    assert VoiceObservation.load(path).attacks is None


# --------------------------------------------------------------------------- #
# Cold reinterpretation: no decoding, no pYIN
# --------------------------------------------------------------------------- #


def test_reinterpret_reads_a_saved_observation_without_pyin(tmp_path, monkeypatch):
    original = _observation()
    path = tmp_path / "take.observation.npz"
    original.save(path)

    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("reinterpretation must neither decode nor re-run pYIN")

    monkeypatch.setattr("v2m.core.pitch.track_pitch_offline", _forbidden)
    monkeypatch.setattr("v2m.core.observation.track_pitch_offline", _forbidden)
    monkeypatch.setattr("v2m.core.audio_io.load_file", _forbidden)

    loaded = VoiceObservation.load(path)
    result = interpret_observation(
        loaded,
        tempo=None,
        source=str(path),
        opts=AnalyzeOptions(),
        report=lambda f, m: None,
    )
    assert [n.midi for n in result.notes] == [60, 62, 64, 65, 67]
    # without a tempo provided, the interpretation only produces notes
    assert result.tempo is None


# --------------------------------------------------------------------------- #
# The diff: what the script shows to compare two readings
# --------------------------------------------------------------------------- #


def test_diff_notes_reports_count_and_per_note_moves():
    assert reinterpret.diff_notes([[0.0, 0.5, 60]], [[0.0, 0.5, 60]]) == []
    lines = reinterpret.diff_notes([[0.0, 0.5, 60]], [[0.0, 0.5, 62]])
    assert len(lines) == 1 and "note 1" in lines[0]
    count = reinterpret.diff_notes([[0.0, 0.5, 60]], [])
    assert "note count" in count[0]


def test_observation_path_sits_next_to_the_take():
    assert reinterpret.observation_path("chant.wav") == Path(
        "chant.observation.npz"
    )
    assert reinterpret.observation_path("prises/live.wav") == Path(
        "prises/live.observation.npz"
    )


# --------------------------------------------------------------------------- #
# --sweep: the readings table, evidence unchanged
# --------------------------------------------------------------------------- #


def test_sweep_scans_values_and_marks_the_current_calibration(capsys):
    observation = _observation()
    args = reinterpret.parse_args(
        ["x.wav", "--observation", "x.observation.npz",
         "--sweep", "0.3,0.5,0.7", "--sweep-note", "60,120"]
    )
    args.semitones = 0.5
    args.min_note = 60.0
    code = reinterpret.run_sweep(args, observation, None, "x.wav")
    out = capsys.readouterr().out
    assert code == 0
    # one line per combination (3 x 2), plus the header and the footer
    assert out.count("0.3") and out.count("0.7") and out.count("120")
    # the current calibration carries the star, not the other lines
    starred = [line for line in out.splitlines() if line.startswith("  *")]
    assert len(starred) == 1 and "*0.5" in starred[0] and "60" in starred[0]


def test_sweep_without_a_mode_uses_the_current_values_alone(capsys):
    observation = _observation()
    args = reinterpret.parse_args(["x.wav", "--observation", "x.observation.npz"])
    code = reinterpret.run_sweep(args, observation, None, "x.wav")
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("0.5") == 1  # a single line: the default value


def test_a_malformed_sweep_value_is_refused():
    with pytest.raises(SystemExit):
        reinterpret._floats("0.5,abc")


# --------------------------------------------------------------------------- #
# --against: the diff of EVIDENCE, not only of interpretation
# --------------------------------------------------------------------------- #


def test_evidence_diff_is_silent_for_the_same_observation():
    obs = _observation()
    assert reinterpret.evidence_diffs(obs, obs) == []


def test_evidence_diff_reports_engine_level_moves():
    """A different engine: the evidence moves BEFORE any interpretation."""
    a = _observation()

    def _fp(notes):
        return [
            [round(float(n.start), 4), round(float(n.duration), 4), int(n.midi)]
            for n in notes
        ]

    # shift the WHOLE curve by a quarter tone: that is what another
    # pitch detector would do its own way, thresholds untouched
    shifted_pitch = replace(
        a.pitch, hz=a.pitch.hz * 2.0 ** (0.25 / 12.0)
    )
    b = replace(a, pitch=shifted_pitch, tuning_offset_st=a.tuning_offset_st + 0.25)
    diffs = "\n".join(reinterpret.evidence_diffs(a, b))
    assert "courbe" in diffs
    assert "tuning" in diffs
    # the demonstration of the value: the uniform shift is ABSORBED by
    # the global tuning offset (TUNING_OFFSET_MAX_ST allows it), so the
    # segmentation reads EXACTLY the same notes — a segmentation diff
    # alone would have shown nothing, the evidence sees everything
    assert reinterpret.diff_notes(_fp(interpret(a)), _fp(interpret(b))) == []


def test_evidence_diff_reports_frame_count_and_attacks():
    a = _observation()
    times = np.linspace(0.0, a.duration_s, len(a.pitch.times) + 3)
    hz_a = a.pitch.hz
    valid = ~np.isnan(hz_a)
    hz2 = np.interp(
        times, a.pitch.times[valid], hz_a[valid]
    )  # filled everywhere: the diff is first about the FRAME count
    b = replace(
        a,
        pitch=PitchTrack(
            times=times,
            hz=hz2,
            confidence=a.pitch.confidence,
            transition=a.pitch.transition,
        ),
        attacks=None,
    )
    diffs = "\n".join(reinterpret.evidence_diffs(a, b))
    assert "frames" in diffs
    assert "attaques" in diffs


# --------------------------------------------------------------------------- #
# The cache: a valid observation replaces pYIN, a stale one is ignored
# --------------------------------------------------------------------------- #


def _take_file(tmp_path: Path) -> Path:
    truth = sequence([60, 62, 64], dur_s=0.5, gap_s=0.15)
    y = synthesize(truth)
    wav = tmp_path / "take.wav"
    sf.write(str(wav), y, 22_050)
    return wav


def test_pipeline_reuses_a_fresh_observation_cache(tmp_path, monkeypatch):
    wav = _take_file(tmp_path)
    opts = AnalyzeOptions()
    y = audio_io.load_file(str(wav))
    cache = wav.with_suffix(".observation.npz")
    obs = VoiceObservation.from_signal(y, 22_050)
    obs.source_size = wav.stat().st_size
    obs.source_mtime = wav.stat().st_mtime
    obs.save(cache)

    # the reference reading is computed BEFORE the monkeypatch (the test
    # proves analyze_file does NOT need from_signal; it itself, does)
    fresh = VoiceObservation.from_signal(y, 22_050)
    expected = reinterpret.notes_fingerprint(
        reinterpret.interpret_at(fresh, None, opts, str(wav))
    )

    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a fresh cache must avoid pYIN")

    monkeypatch.setattr("v2m.core.pipeline.VoiceObservation.from_signal", _forbidden)
    result = analyze_file(str(wav), opts=opts)
    assert [n.midi for n in result.notes] == [60, 62, 64]
    # the cache's result is the same reading as the full computation
    assert reinterpret.notes_fingerprint(result) == expected


def test_pipeline_ignores_a_stale_cache(tmp_path, monkeypatch):
    """Audio rewritten after the observation: pYIN must run again."""
    wav = _take_file(tmp_path)
    opts = AnalyzeOptions()
    y = audio_io.load_file(str(wav))
    cache = wav.with_suffix(".observation.npz")
    obs = VoiceObservation.from_signal(y, 22_050)
    obs.source_size = wav.stat().st_size
    obs.source_mtime = wav.stat().st_mtime - 10.0  # taken BEFORE the rewrite
    obs.save(cache)
    calls = []

    real = VoiceObservation.from_signal

    def _counting(signal, sr, **kwargs):
        calls.append(1)
        return real(signal, sr, **kwargs)

    monkeypatch.setattr(
        "v2m.core.pipeline.VoiceObservation.from_signal", _counting
    )
    analyze_file(str(wav), opts=opts)
    assert calls, "a stale cache must re-run the evidence build"


def test_pipeline_ignores_a_cache_without_attacks_when_attacks_are_asked(tmp_path):
    """--no-attacks produced this evidence: it cannot serve an analysis
    WITH attacks — the cache must never change a decision."""
    wav = _take_file(tmp_path)
    y = audio_io.load_file(str(wav))
    cache = wav.with_suffix(".observation.npz")
    obs = VoiceObservation.from_signal(y, 22_050, use_attacks=False)
    obs.source_size = wav.stat().st_size
    obs.source_mtime = wav.stat().st_mtime
    obs.save(cache)
    assert analyze_file(str(wav), opts=AnalyzeOptions()).notes


def test_pipeline_ignores_a_corrupt_cache(tmp_path):
    wav = _take_file(tmp_path)
    cache = wav.with_suffix(".observation.npz")
    cache.write_bytes(b"not an npz file")
    result = analyze_file(str(wav), opts=AnalyzeOptions())
    assert result.notes  # the unreadable cache did not cost the analysis


def test_analyze_file_autosaves_the_observation_cache(tmp_path, monkeypatch):
    """The first analysis WRITES the cache: the second does not re-run pYIN."""
    wav = _take_file(tmp_path)
    opts = AnalyzeOptions()
    assert not wav.with_suffix(".observation.npz").exists()
    first = analyze_file(str(wav), opts=opts)
    cache = wav.with_suffix(".observation.npz")
    assert cache.exists(), "the analysis did not leave its evidence beside the take"

    # the reference reading is computed BEFORE the monkeypatch
    y = audio_io.load_file(str(wav))
    fresh = VoiceObservation.from_signal(y, 22_050)
    expected = reinterpret.notes_fingerprint(
        reinterpret.interpret_at(fresh, None, opts, str(wav))
    )

    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the auto-saved cache must avoid pYIN on the 2nd analysis")

    monkeypatch.setattr("v2m.core.pipeline.VoiceObservation.from_signal", _forbidden)
    second = analyze_file(str(wav), opts=opts)
    assert reinterpret.notes_fingerprint(second) == expected
    assert reinterpret.notes_fingerprint(first) == expected


def test_autosave_survives_a_readonly_folder(tmp_path, monkeypatch):
    """A read-only folder costs the cache WRITE, not the analysis."""
    wav = _take_file(tmp_path)
    cache = wav.with_suffix(".observation.npz")
    real_save = VoiceObservation.save

    def _denied(self, path):
        raise OSError("read-only")

    monkeypatch.setattr(VoiceObservation, "save", _denied)
    result = analyze_file(str(wav), opts=AnalyzeOptions())
    assert result.notes  # the analysis went through
    assert not cache.exists()
    monkeypatch.setattr(VoiceObservation, "save", real_save)


def test_matches_take_rejects_a_wrong_sample_rate():
    obs = _observation()
    assert not obs.matches_take("whatever.wav", sr=44_100, duration_s=obs.duration_s)


def test_pipeline_ignores_a_cache_of_another_format_version(tmp_path):
    """A `.npz` of another layout must not be read by the cache
    (the arrays would no longer mean anything) — but
    `VoiceObservation.load` stays tolerant: the reinterpretation script
    must re-read old evidence."""
    wav = _take_file(tmp_path)
    y = audio_io.load_file(str(wav))
    cache = wav.with_suffix(".observation.npz")
    obs = VoiceObservation.from_signal(y, 22_050)
    obs.save(cache)
    # rewrites the version scalar with another value
    with np.load(cache, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    payload["format_version"] = np.asarray(FORMAT_VERSION + 1)
    np.savez(str(cache), **payload)

    # the cache refused, the analysis goes through the full build
    result = analyze_file(str(wav), opts=AnalyzeOptions())
    assert [n.midi for n in result.notes] == [60, 62, 64]
    # and the .npz itself stays READABLE (the loader's tolerance)
    assert VoiceObservation.load(cache).sample_rate == 22_050
