"""The profiling diagnostic must stay wired to the real pipeline.

The script does not re-implement the stages: it reads them from the
`progress` callback of `analyze_signal`. This test therefore compares
the labels the pipeline ACTUALLY publishes to what the script shows —
if a stage is renamed or added, the table follows (and the test checks
it), with no list to maintain here.

A single script call covers the rich case (mean + cProfile + JSON):
every run pays the interpreter and the numba compilation, so the suite
keeps to a few seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, silence, sine_note
from v2m.core.audio_io import SR as ANALYSIS_SR
from v2m.core.pipeline import analyze_signal

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_analysis.py"


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
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(SCRIPT.parents[1]),
    )


def _pipeline_stages(wav: str) -> list[str]:
    """The stages the pipeline announces on this file, closing callback dropped.

    The last call (fraction 1.0, "Done") only closes the last stage: the
    timer drops it, hence `[:-1]`.
    """
    labels: list[str] = []
    y, _ = sf.read(wav, dtype="float32")
    from v2m.core.i18n import tr

    analyze_signal(y, ANALYSIS_SR, progress=lambda f, m: labels.append(m))
    assert labels[-1] == tr("progress done")  # the closing callback, any language
    return labels[:-1]


def test_table_json_and_profiler(tmp_path):
    wav = _wav(tmp_path)
    report = tmp_path / "mesures.json"
    proc = _run(wav, "--repeat", "2", "--profile", "--top", "3", "--json", str(report))
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout

    # stage table: every one of the pipeline's, one row each
    assert "Stage" in out and "TOTAL" in out and "Heaviest stage" in out
    assert "file decoding" in out
    assert "export (5 artifacts" in out
    assert "warm-up" in out, "the default warm-up must be announced"
    assert "mean over 2 runs" in out
    # each stage occupies ONE table row — the TOTAL row
    # "Heaviest stage : <label>" repeats a label on purpose, it
    # must not count as a second stage (timing flake otherwise: depending
    # on the machine, any stage can be the heaviest).
    body_lines = [
        line for line in out.splitlines() if not line.startswith("Étape la plus lourde")
    ]
    for label in _pipeline_stages(wav):
        assert sum(label in line for line in body_lines) == 1, (
            f"stage missing or duplicated : {label}"
        )
    assert "Done" not in out  # closing callback: never a row
    assert "20%" in out  # the fractions come from the callback, not a copy

    # cProfile: project code on one side, libraries on the other
    assert "Project code" in out and "Libraries" in out

    # per-stage memory peak (tracemalloc, Python allocations): the column
    # is in the table, the total in the JSON — the profile covers CPU
    # AND memory
    assert "Mem peak" in out and "Mem peak" in out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["runs"] == 2
    assert data["audio_s"] == pytest.approx(1.5, abs=0.05)
    assert data["n_notes"] >= 3
    assert data["total_s"] > 0
    assert data["peak_mib"] > 0
    stages = data["stages"]
    assert stages[0]["label"] == "file decoding"
    # >= 0: a nearly free stage (rest detection) may
    # measure under the JSON's write resolution — 0.0 is legitimate.
    assert all(row["seconds"] >= 0 for row in stages)
    # the decoding, outside the trace, carries a null peak; the analyzed
    # stages (pYIN at least) carry a positive one
    assert stages[0]["peak_mib"] == 0.0
    assert any(row["peak_mib"] > 0 for row in stages[1:-1])
    assert sum(row["share_pct"] for row in stages) == pytest.approx(100.0, abs=0.5)
    assert any(row["fraction"] is not None for row in stages)
    assert data["profile"] and isinstance(data["profile"][0]["project_code"], bool)


def test_bad_inputs_exit_cleanly(tmp_path):
    missing = _run(str(tmp_path / "absent.wav"))
    assert missing.returncode == 1
    assert "introuvable" in missing.stderr

    text_file = tmp_path / "notes.txt"
    text_file.write_text("pas de l'audio", encoding="utf-8")
    wrong = _run(str(text_file))
    assert wrong.returncode == 1
    assert "Format non pris en charge" in wrong.stderr
