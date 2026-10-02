"""The diagnostic's --compare/--golden mode replays the SAME analysis.

The script replays the whole segmentation (decoding included) and
compares the numbers' fingerprint: this test checks the stable case on
synthetic audio (determinism = zero difference, exit 0), then that the
comparison BITES — a script that compared badly (always "stable")
would guarantee less than nothing. Same choice as test_profile_analysis:
the real script is executed, never copied — IN-PROCESS (`main()`
imported via importlib, `redirect_stdout` captures what it prints): six
subprocess boots paid librosa/numba imports six times (~20 s of the
suite measured); in-process the file analysis remains and the boots
disappear.

The repo's REAL takes (demo.wav, chant.wav — gitignored) are replayed
against their golden fingerprint when present: a segmentation drift on
a real take is then seen by the full suite, not only when we think of
it. These replays cost ~16 s out of ~47 s (~34% of the suite —
measured): they are marked `slow` (deselect with
`-m "not slow"` for a fast loop), but stay IN the suite by default — a
net that no longer runs is not a net.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, silence, sine_note

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_accuracy.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("score_accuracy", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the score_accuracy script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_script = _load_script()
main = _script.main

# The audit engine moved to the core (v2m.core.audit): the script only
# imports and prints it. The suite's drift nets read the engine from the
# core; the aliases below PIN the script to it — a local redefinition in
# the script would silently fork the frozen format.
from v2m.core.audit import compare_fingerprints as _core_compare  # noqa: E402

compare_fingerprints = _core_compare


def _wav(tmp_path) -> str:
    parts: list[np.ndarray] = []
    for hz in (220.0, 261.63, 329.63):
        parts.append(sine_note(hz, 0.4))
        parts.append(silence(0.1))
    path = tmp_path / "melody.wav"
    sf.write(str(path), np.concatenate(parts), SR)
    return str(path)


def _run(*args: str) -> tuple[int, str]:
    """Run the script's main() in-process; capture what it prints."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = main(list(args))
    return rc, out.getvalue()


def _expect(cond: bool, msg: str, out: str) -> None:
    """Flake capture: the first broken condition dumps the full stdout.

    The subprocess console encoding (cp1252 on Windows) no longer
    matters — the capture is an in-process StringIO, so assertions may
    target accented substrings directly (ASCII-safe only before).
    """
    if cond:
        return
    raise AssertionError(f"FLAKE CAPTURE — {msg}\n--- stdout:\n{out}")


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
    rc, out = _run(wav, "--diagnose")
    _expect(rc == 0, "--diagnose failed", out)
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
    first_rc, first_out = _run(wav, "--golden")
    _expect(first_rc == 0, "--golden first visit failed", first_out)
    _expect("created" in first_out, "the fingerprint was not created", first_out)
    golden = tmp_path / "melody.golden.json"
    assert golden.exists()

    second_rc, second_out = _run(wav, "--golden")
    _expect(second_rc == 0, "--golden second visit failed", second_out)
    _expect("identical to the reference" in second_out, "the second visit must be identical", second_out)


@pytest.mark.slow
def test_golden_catches_a_drift(tmp_path):
    """The audit BITES: the stored fingerprint cannot be fooled by a
    drifting engine (the reference segmentation changed) — exit 1 and
    the difference named.
    """
    wav = _wav(tmp_path)
    base_rc, base_out = _run(wav, "--golden")
    _expect(base_rc == 0, "--golden baseline failed", base_out)
    golden = tmp_path / "melody.golden.json"
    data = json.loads(golden.read_text(encoding="utf-8"))
    data["tempo_bpm"] = 119.0
    data["notes"][0][0] = 0.37
    golden.write_text(json.dumps(data), encoding="utf-8")

    drifted_rc, drifted_out = _run(wav, "--golden")
    _expect(drifted_rc == 1, "a forged fingerprint must exit 1", drifted_out)
    _expect("tempo_bpm" in drifted_out, "the forged tempo must be named", drifted_out)


def test_compare_reports_stable_segmentation(tmp_path):
    """Two passes on the same file: identical numbers, exit 0."""
    rc, out = _run(_wav(tmp_path), "--compare")
    _expect(rc == 0, "--compare failed", out)
    _expect("identical" in out, "the two passes must agree", out)


def test_compare_fingerprints_bites_on_a_drift():
    """The comparison reports ANY drift — tempo, note, strength, rest.

    Also pins the script as a thin adapter: its name binds the CORE
    engine (a local redefinition would fork the frozen format).
    """
    stable = {
        "tempo_bpm": 120.0,
        "meter": "4/4",
        "key": "C",
        "notes": [(0.0, 0.5, 60, "noire", 0.1)],
        "rests": [],
    }
    assert compare_fingerprints(stable, dict(stable)) == []
    # the script re-exports the core engine, never a copy
    assert _script.compare_fingerprints is _core_compare

    drifted = dict(stable, tempo_bpm=119.0, notes=[(0.0, 0.5, 60, "noire", 0.9)])
    diffs = compare_fingerprints(stable, drifted)
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
    rc, out = _run(str(audio), "--golden")
    if not (audio.parent / audio.with_suffix(".golden.json").name).exists():
        # first visit: the test just created the fingerprint, nothing to replay
        _expect(rc == 0, "--golden creation failed on " + take, out)
        pytest.skip(f"{take}'s golden fingerprint created on this pass; replay to verify")
    _expect(
        rc == 0,
        "segmentation drift on " + take,
        out,
    )
