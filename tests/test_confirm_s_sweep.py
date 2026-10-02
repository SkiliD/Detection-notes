"""The low-amplitude axis of the confirm_s sweep (real amplitude range).

The real takes sat at RMS 0.019..0.062 — 3-7x under the reference takes —
and the question was whether the segmentation bounds hold on QUIET
signals. The sweep script (`scripts/confirm_s_sweep.py --amps`) answers
over the whole corpus; this test pins the load-bearing cell in-process:
`confirm_s` 0.09 (the shipped default) must stay EXACT at amp 0.10 on
every seed, on the two cases the full sweep shows as the fragile ones at
0.08 (legato, the window-pollution case). No subprocess: `main()` runs
in-process, and the harness's seed makes single-seed claims worthless —
the test repeats across seeds.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPEC = importlib.util.spec_from_file_location(
    "confirm_s_sweep", ROOT / "scripts" / "confirm_s_sweep.py"
)
assert SPEC is not None and SPEC.loader is not None  # narrows for the reader
sweep = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("confirm_s_sweep", sweep)
SPEC.loader.exec_module(sweep)


def test_confirm_s_009_is_exact_at_low_amplitude():
    """The shipped bound holds on quiet signals: amp 0.10, several seeds,
    the corpus's two legato cases (the 0.08-fragile ones) stay exact."""
    for name in ("legato", "gamme liée"):
        for seed in (1, 2, 3):
            ok, _lat = sweep.run_case(name, seed, confirm_s=0.09, amp=0.10)
            assert ok, f"{name} seed {seed}: confirm_s 0.09 failed at amp 0.10"


def test_quiet_signal_does_not_cost_a_note_at_009():
    """Exactness means the note COUNT too: at amp 0.10 the detached case
    (gaps a breath could swallow) still finds every note and no extra."""
    for seed in (1, 2, 3):
        ok, _lat = sweep.run_case("détaché", seed, confirm_s=0.09, amp=0.10)
        assert ok, f"détaché seed {seed}: the quiet take lost or invented a note"


def test_sweep_main_runs_in_process_and_reports_the_amp_axes(capsys):
    """`--amps` produces one block per amplitude, and main() is importable
    (the suite runs scripts in-process — a subprocess boot pays librosa)."""
    rc = sweep.main(["--values", "0.09", "--seeds", "1", "--amps", "0.10"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "amplitude 0.10" in out
    assert "confirm_s=0.09" in out
