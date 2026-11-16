"""The profiling diagnostic must stay wired to the real pipeline.

The script does not re-implement the stages: it reads them from the
`progress` callback of `analyze_signal`. This test therefore compares
the labels the pipeline ACTUALLY published — read from the script's own
JSON — to the table printed for the SAME run.

The script runs IN THIS PROCESS: `main()` is imported from
`scripts/profile_analysis.py` and called directly, no subprocess. Two
reasons, both measured:
- the conftest language sandbox is process-local, so a subprocess read
  the user's real `~/.v2m_lang` and could answer in English while this
  file's assertions assume one language — the structural hazard behind
  the single failure of `test_table_json_and_profiler` under load;
- the expected labels come from the script's own JSON (`data["stages"]`),
  so ONE analysis chain feeds both sides where the old test re-ran
  `analyze_signal` a second time just to collect labels.

A single `main()` call covers the rich case (mean + cProfile + JSON) in
a few seconds; the 2 s discarded warm-up stays (it is part of the run
the table describes).

The "Heaviest stage: …" trailer repeats one row's label on purpose; the
body filter drops it by its real prefix so each pipeline stage appears
once in the reported table.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf
from conftest import SR, silence, sine_note

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_analysis.py"


def _load_script() -> ModuleType:
    """Import the script as a module: `main()` runs in THIS process."""
    spec = importlib.util.spec_from_file_location("profile_analysis", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the profiling script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile_script = _load_script()


def _wav(tmp_path) -> str:
    parts: list[np.ndarray] = []
    for hz in (220.0, 261.63, 329.63):
        parts.append(sine_note(hz, 0.4))
        parts.append(silence(0.1))
    path = tmp_path / "melody.wav"
    sf.write(str(path), np.concatenate(parts), SR)
    return str(path)


def _run_main(*args: str) -> tuple[int, str, str]:
    """Run the script's main() in-process; capture what it prints."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = profile_script.main(list(args))
    return rc, out.getvalue(), err.getvalue()


def test_table_json_and_profiler(tmp_path):
    wav = _wav(tmp_path)
    report = tmp_path / "mesures.json"
    rc, out, err = _run_main(
        wav, "--repeat", "2", "--profile", "--top", "3", "--json", str(report)
    )
    data = json.loads(report.read_text(encoding="utf-8"))

    # The FLAKE CAPTURE: this test failed once under machine load (green
    # again on re-run and -p no:randomly — environment, not the code). To
    # capture its state at the NEXT failure, every assertion below goes
    # through `_expect(cond, msg)`: at the first broken condition the full
    # state (return code, stdout, stderr, JSON) is dumped in the message.
    def _expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(
                f"FLAKE CAPTURE — {msg}\n"
                f"--- returncode={rc}\n"
                f"--- stderr:\n{err[-2000:]}\n"
                f"--- stdout:\n{out}\n"
                f"--- json:\n{json.dumps(data, ensure_ascii=False, indent=1)}"
            )

    _expect(rc == 0, "main returned 0")
    # stage table: every one of the pipeline's, one row each. The labels
    # come from the script's own JSON — the SAME run that printed the
    # table (no second analysis, no subprocess re-reading the language).
    _expect("Stage" in out and "TOTAL" in out and "Heaviest stage" in out, "table header")
    _expect("file decoding" in out, "file decoding row")
    _expect("export (5 artifacts" in out, "export row")
    _expect("warm-up" in out, "the default warm-up must be announced")
    _expect("mean over 2 runs" in out, "mean over 2 runs")
    # each stage occupies ONE table row — the "Heaviest stage:" trailer
    # repeats a label on purpose, the body filter drops it (see module
    # docstring: timing flake otherwise, any stage can be the heaviest).
    body_lines = [
        line for line in out.splitlines() if not line.startswith("Heaviest stage")
    ]
    for label in (row["label"] for row in data["stages"]):
        _expect(
            sum(label in line for line in body_lines) == 1,
            f"stage missing or duplicated : {label}",
        )
    _expect("Done" not in out, "closing callback: never a row")
    _expect("20%" in out, "fractions come from the callback, not a copy")

    # cProfile: project code on one side, libraries on the other
    _expect("Project code" in out and "Libraries" in out, "cProfile sections")

    # per-stage memory peak (tracemalloc, Python allocations): the column
    # is in the table, the total in the JSON — the profile covers CPU
    # AND memory
    _expect("Mem peak" in out, "Mem peak column")
    _expect(data["runs"] == 2, "runs == 2")
    _expect(data["audio_s"] == pytest.approx(1.5, abs=0.05), f"audio_s {data['audio_s']}")
    _expect(data["n_notes"] >= 3, f"n_notes {data['n_notes']}")
    _expect(data["total_s"] > 0, "total_s > 0")
    _expect(data["peak_mib"] > 0, "peak_mib > 0")
    stages = data["stages"]
    _expect(stages[0]["label"] == "file decoding", "first stage is decoding")
    # >= 0: a nearly free stage (rest detection) may
    # measure under the JSON's write resolution — 0.0 is legitimate.
    _expect(all(row["seconds"] >= 0 for row in stages), "negative stage seconds")
    # the decoding, outside the trace, carries a null peak; the analyzed
    # stages (pYIN at least) carry a positive one
    _expect(stages[0]["peak_mib"] == 0.0, f"decoding peak {stages[0]['peak_mib']}")
    _expect(
        any(row["peak_mib"] > 0 for row in stages[1:-1]),
        "no analyzed stage carries a memory peak",
    )
    _expect(
        sum(row["share_pct"] for row in stages) == pytest.approx(100.0, abs=0.5),
        f"share sum {sum(row['share_pct'] for row in stages)}",
    )
    _expect(any(row["fraction"] is not None for row in stages), "no fraction published")
    _expect(
        bool(data["profile"]) and isinstance(data["profile"][0]["project_code"], bool),
        "profile rows carry project_code",
    )


def test_bad_inputs_exit_cleanly(tmp_path):
    rc, _out, err = _run_main(str(tmp_path / "absent.wav"))
    assert rc == 1
    assert "introuvable" in err

    text_file = tmp_path / "notes.txt"
    text_file.write_text("pas de l'audio", encoding="utf-8")
    rc, _out, err = _run_main(str(text_file))
    assert rc == 1
    assert "Format non pris en charge" in err


def test_rescue_cost_is_reported_per_run(tmp_path):
    """The rescue's own line and JSON block: the sweep is visible per take.

    The facts come from `pitch.pitch_stage_cost()` (one writer: the last
    reading), so the line never re-times the sweep — and on a synthetic
    melody whose bursts pYIN already reads, the cost is real while the
    grafted seconds are 0.0 (the sweep ran, the gates refused).
    """
    wav = _wav(tmp_path)
    report = tmp_path / "mesures.json"
    rc, out, err = _run_main(wav, "--json", str(report))
    assert rc == 0, err
    _expect_line = "Rescue" in out and "grafted" in out
    assert _expect_line, out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert "rescue" in data, data.get("rescue")
    assert set(data["rescue"]) == {"grafted_s", "cost_s"}
    assert data["rescue"]["cost_s"] >= 0.0
    assert data["rescue"]["grafted_s"] >= 0.0
