"""The diagnostic's --compare/--golden mode replays the SAME analysis.

The script replays the whole segmentation (decoding included) and
compares the numbers' fingerprint: this test checks the stable case on
synthetic audio (determinism = zero difference, exit 0), then that the
comparison BITES — a script that compared badly (always "stable")
would guarantee less than nothing. Same choice as test_profile_analysis:
the real script is executed, never copied.

The repo's REAL takes (demo.wav, chant.wav — gitignored) are replayed
against their golden fingerprint when present: a segmentation drift on
a real take is then seen by the full suite, not only when we think of
it. These replays cost ~16 s out of ~47 s (~34% of the suite —
2026-09-23 measurement): they are marked `slow` (deselect with
`-m "not slow"` for a fast loop), but stay IN the suite by default — a
net that no longer runs is not a net.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, silence, sine_note

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_accuracy.py"


def _wav(tmp_path) -> str:
    parts: list[np.ndarray] = []
    for hz in (220.0, 261.63, 329.63):
        parts.append(sine_note(hz, 0.4))
        parts.append(silence(0.1))
    path = tmp_path / "melody.wav"
    sf.write(str(path), np.concatenate(parts), SR)
    return str(path)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )


# --------------------------------------------------------------------------- #
# --diagnose: the audit of the thresholds actually exercised
# --------------------------------------------------------------------------- #


def test_diagnose_reports_the_thresholds_a_take_exercised(tmp_path):
    """The --diagnose mode prints the take's ranges: pitch curve vs reading
    bounds, absorbed tuning offset, attack strengths (and how many under
    the floor), ambiguous deviations, and the live path.
    Every announced section must exist in the output — an audit that says
    nothing diagnoses nothing."""
    wav = _wav(tmp_path)
    proc = _run(wav, "--diagnose")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    # ASCII-safe substrings only: the subprocess console encoding (cp1252
    # on Windows) mangles accents in the captured bytes, not the content
    assert "diagnosis" in out
    assert "65.0-2100.0" in out  # the reading bounds, FMIN-FMAX
    assert "confirm_s=0.09" in out  # the live path replayed
    assert "offset" in out or "tuning" in out
    assert "0.2" in out  # the strength floor appears in the report


@pytest.mark.slow
def test_golden_saves_then_detects_the_same_take(tmp_path):
    """First visit: the fingerprint is created (file beside the audio).
    Second visit: identical, exit 0 — the audit is green on a stable
    take.
    """
    wav = _wav(tmp_path)
    first = _run(wav, "--golden")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "created" in first.stdout
    golden = tmp_path / "melody.golden.json"
    assert golden.exists()

    second = _run(wav, "--golden")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "identical to the reference" in second.stdout


@pytest.mark.slow
def test_golden_catches_a_drift(tmp_path):
    """The audit BITES: the stored fingerprint cannot be fooled by a
    drifting engine (the reference segmentation changed) — exit 1 and
    the difference named.
    """
    wav = _wav(tmp_path)
    assert _run(wav, "--golden").returncode == 0
    golden = tmp_path / "melody.golden.json"
    data = json.loads(golden.read_text(encoding="utf-8"))
    data["tempo_bpm"] = 119.0
    data["notes"][0][0] = 0.37
    golden.write_text(json.dumps(data), encoding="utf-8")

    drifted = _run(wav, "--golden")
    assert drifted.returncode == 1, drifted.stdout
    assert "tempo_bpm" in drifted.stdout


def test_compare_reports_stable_segmentation(tmp_path):
    """Two passes on the same file: identical numbers, exit 0."""
    proc = _run(_wav(tmp_path), "--compare")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "identical" in proc.stdout


def test_compare_fingerprints_bites_on_a_drift():
    """The comparison reports ANY drift — tempo, note, strength, rest."""
    spec = importlib.util.spec_from_file_location("score_accuracy", SCRIPT)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    stable = {
        "tempo_bpm": 120.0,
        "meter": "4/4",
        "key": "C",
        "notes": [(0.0, 0.5, 60, "noire", 0.1)],
        "rests": [],
    }
    assert script.compare_fingerprints(stable, dict(stable)) == []

    drifted = dict(stable, tempo_bpm=119.0, notes=[(0.0, 0.5, 60, "noire", 0.9)])
    diffs = script.compare_fingerprints(stable, drifted)
    assert diffs and len(diffs) == 2  # the tempo AND note 1's strength


@pytest.mark.slow
@pytest.mark.parametrize("take", ["demo.wav", "chant.wav"])
def test_real_takes_match_their_golden_fingerprint(take: str):
    """The repo's real takes replay their golden fingerprint (when present).

    demo.wav is regenerable (scripts/make_demo.py, see README) and
    chant.wav is the local reference take: both are gitignored, hence
    the silent skip where they are absent. An existing golden
    fingerprint that drifts = hard failure: that is the --golden net
    wired into the full suite.
    """
    audio = Path(__file__).resolve().parents[1] / take
    if not audio.exists():
        pytest.skip(f"{take} absent (gitignored, regenerable) — fingerprint not replayed")
    proc = _run(str(audio), "--golden")
    if not (audio.parent / audio.with_suffix(".golden.json").name).exists():
        # first visit: the test just created the fingerprint, nothing to replay
        assert proc.returncode == 0, proc.stdout + proc.stderr
        pytest.skip(f"{take}'s golden fingerprint created on this pass; replay to verify")
    assert proc.returncode == 0, "segmentation drift on " + take + ":\n" + proc.stdout
