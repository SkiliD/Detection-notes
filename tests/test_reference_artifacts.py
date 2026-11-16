"""Reference artifacts: the full output of two takes, frozen.

Before unifying the seconds <-> beats conversions (one Grid, see
CONTEXT.md), a net is needed that sees the slightest shift of the
Analysis result. This file replays the two takes present in the working
copy — `chant.wav`, a real sung take, and `demo.wav` — and compares the
score, the report and the MIDI, byte by byte, against
`tests/reference/artifacts.json`.

It is a net over the SEVEN consumers at once: the score proves the
figures, measures, rests, key signature and positions; the report
proves the counters and positions; the MIDI proves the quantization.

A second test queries the score for itself, without a reference: every
measure must total its signature's duration and every note's `<type>`
must equal its `<duration>` — two numbers coming from different paths
(the analysis, then the written-values catalogue).

Two accepted precautions:

- the reference is tied to the audio files, which are gitignored: the
  test is skipped when they are missing (another machine, a container)
  rather than failing;
- re-freezing the reference is a decision, not a reflex: run
  `V2M_FREEZE_REFERENCE=1 python -m pytest
  tests/test_reference_artifacts.py`, after SEEING what moved (the test
  writes the fresh artifacts into its temp folder and gives the path).

If `demo.wav` is regenerated with another recipe, or `chant.wav`
replaced, that is a new standard: re-freezing is then correct.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from scorexml import assert_measures_close, score_root

from v2m.core import audio_io
from v2m.core.pipeline import AnalyzeOptions, analyze_signal
from v2m.core.report import export_result

REFERENCE = Path(__file__).parent / "reference" / "artifacts.json"
ROOT = Path(__file__).parents[1]
TAKES = ("chant.wav", "demo.wav")
ARTIFACTS = ("score.musicxml", "score.json", "score.mid")
FREEZE = os.environ.get("V2M_FREEZE_REFERENCE") == "1"


def _analyse(take: Path, out_dir: Path) -> dict:
    """Analyze a take, write its artifacts, return counters + fingerprints.

    `source` is the file's NAME (not its path): the JSON report copies it,
    so the reference must not depend on where the test is run from.
    """
    res = analyze_signal(
        audio_io.load_file(str(take)), audio_io.SR, source=take.name, opts=AnalyzeOptions()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    export_result(res, str(out_dir / "score"))
    return {
        "n_notes": len(res.notes),
        "n_rests": len(res.rests or []),
        "tempo_bpm": res.tempo.bpm if res.tempo else 0.0,
        "sha256": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest() for name in ARTIFACTS
        },
    }


def _moved(want: dict, got: dict) -> str:
    """What moved in the counters, for a readable failure message."""
    return ", ".join(
        f"{key} {want[key]} → {got[key]}"
        for key in ("n_notes", "n_rests", "tempo_bpm")
        if want[key] != got[key]
    )


@pytest.fixture(scope="module")
def fresh_reference_artifacts(tmp_path_factory):
    present = [(name, ROOT / name) for name in TAKES if (ROOT / name).exists()]
    if not present:
        pytest.skip(f"no reference take in {ROOT}")

    output_root = tmp_path_factory.mktemp("reference-artifacts")
    fresh = {}
    for name, take in present:
        out_dir = output_root / name
        fresh[name] = {"summary": _analyse(take, out_dir), "out_dir": out_dir}
    return fresh


def test_reference_scores_fill_their_measures_and_agree_with_their_types(
    fresh_reference_artifacts,
):
    """Each reference score closes and its written types match durations."""
    for name, artifacts in fresh_reference_artifacts.items():
        xml = (artifacts["out_dir"] / "score.musicxml").read_text(encoding="utf-8")
        assert_measures_close(score_root(xml), name)


def test_reference_takes_still_export_the_same_bytes(fresh_reference_artifacts):
    fresh = {name: artifacts["summary"] for name, artifacts in fresh_reference_artifacts.items()}

    if FREEZE:
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(
            json.dumps(fresh, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pytest.skip(f"reference re-frozen : {REFERENCE}")

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for name, got in fresh.items():
        want = reference.get(name)
        assert want is not None, f"{name} missing from {REFERENCE}"
        assert got["sha256"] == want["sha256"], (
            f"{name} : the output moved ({_moved(want, got) or 'same counters, different bytes'})"
            f" — fresh artifacts to compare : {fresh_reference_artifacts[name]['out_dir']}"
        )
