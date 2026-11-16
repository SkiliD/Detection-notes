"""The code gate runs all three halves — and keeps the runner's clock honest.

`pages.yml` decides what may be PUBLISHED; this one decides whether the code is
sound, on every push and pull request: the Python suite, the headless GUI smoke
test (the only check that opens the window), and the page's Node tests. The
pin exists because a later edit that drops the smoke test or hands the workflow
`write-all` would look green in the diff and quietly reduce the gate.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"


def _jobs(workflow: str) -> tuple[str, str]:
    jobs = workflow[workflow.index("\njobs:") :]
    python_job = jobs[jobs.index("\n  python:") : jobs.index("\n  web:")]
    return python_job, jobs[jobs.index("\n  web:") :]


def test_the_code_gate_runs_every_half():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    python_job, web_job = _jobs(workflow)

    assert "pytest tests/ -q" in python_job, "the suite is not run"
    assert "scripts/gui_smoke_test.py" in python_job, "the window is never opened"
    assert "QT_QPA_PLATFORM: offscreen" in python_job, "a runner has no display"
    # The smoke test reads a gitignored take: without it the step would skip
    # rather than run, which is the quiet loss this pin refuses.
    assert "scripts/make_demo.py" in python_job, "the smoke test has no take to read"

    assert "node --test web/test/*.test.mjs" in web_job, "the page's tests are not run"


def test_the_runner_gets_time_without_moving_the_tripwire():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    python_job, _ = _jobs(workflow)
    # A shared runner is slower than the machine the 150 s budget was measured
    # on (conftest keeps the dated history): the workflow raises the number
    # through the environment, and the default stays where it was calibrated.
    assert 'V2M_SUITE_BUDGET_S: "600"' in python_job
    assert "timeout-minutes:" in python_job, "a stuck job would run for hours"


def test_the_gate_asks_for_nothing_but_the_checkout():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    block = workflow[workflow.index("\npermissions:") :]
    block = block[: block.index("\n\n")]
    keys = [line for line in block.splitlines() if line.startswith("  ") and ":" in line]
    granted = {line.split(":")[0].strip() for line in keys}
    assert granted == {"contents"}, granted
    assert "write-all" not in workflow
    # Two pushes in a row: the older run answers for a commit nobody merges.
    assert "concurrency:" in workflow
    assert "cancel-in-progress: true" in workflow
