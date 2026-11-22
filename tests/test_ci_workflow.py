"""The code gate runs all three halves — and keeps the runner's clock honest.

`pages.yml` decides what may be PUBLISHED; this one decides whether the code is
sound, on every push and pull request: the Python suite, the headless GUI smoke
test (the only check that opens the window), and the page's Node tests. The
pin exists because a later edit that drops the smoke test or hands the workflow
`write-all` would look green in the diff and quietly reduce the gate.
"""

from __future__ import annotations

import re
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

    # ONE PAGE FILE AT A TIME. Each file starts its own Chrome, and the
    # capture, CSP and keyboard checks need the page's REAL-TIME audio graph:
    # eighteen browsers at once starved it and two takes of one fake recording
    # drifted 0.314 s apart, while the same files pass serially (3/3 runs).
    assert "node --test --test-concurrency=1 web/test/*.test.mjs" in web_job, (
        "the page's tests are not run one file at a time"
    )


def test_the_page_is_also_replayed_in_the_other_two_engines():
    """One engine is not "the web", and this page already paid for it.

    `web/test/*.test.mjs` drives the runner's Chrome only. WebKit has no
    `Symbol.asyncIterator` on `ReadableStream`, so the archive's compressor threw,
    every entry was STORED, the ZIP stayed valid and 1.5x too big, and the suite
    was green throughout. The matrix lives OUTSIDE `web/`: the page ships as a
    dependency-free folder, so the driver is borrowed by this workflow, pinned —
    and the chosen-file path is one of the stops it replays.
    """
    assert (REPO / "web" / "test" / "file.test.mjs").exists(), "the chosen-file check is gone"
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "\n  web-matrix:" in workflow, "the other two engines are never exercised"
    job = workflow[workflow.index("\n  web-matrix:") :]
    assert "engine: [chromium, firefox, webkit]" in job, "the matrix lost an engine"
    assert "node tests/web_matrix/run.mjs ${{ matrix.engine }}" in job, (
        "the job runs something other than the harness"
    )
    assert "fail-fast: false" in job, "one engine's red would hide the others"
    pin = re.search(r"playwright@(\d+\.\d+\.\d+)", job)
    assert pin, "the driver is installed unpinned"
    assert "--no-save" in job and "--with-deps" in job

    harness = (REPO / "tests" / "web_matrix" / "run.mjs").read_text(encoding="utf-8")
    for check in ("segmenter", "yin", "bundle", "file", "parity"):
        assert f'["{check}",' in harness, f"the matrix no longer replays the {check} check"
    # The rule that catches the WebKit bug: an engine that ADVERTISES raw deflate
    # must be seen USING it — a silent STORED fallback keeps the archive valid.
    assert "deflate-raw" in harness and "method" in harness


def test_the_runner_gets_time_without_moving_the_tripwire():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    python_job, _ = _jobs(workflow)
    # A shared runner is slower than the machine the 150 s budget was measured
    # on (conftest keeps the dated history): the workflow raises the number
    # through the environment, and the default stays where it was calibrated.
    assert 'V2M_SUITE_BUDGET_S: "900"' in python_job
    assert "timeout-minutes:" in python_job, "a stuck job would run for hours"


def test_the_suite_runs_where_a_crash_can_be_read():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    python_job, _ = _jobs(workflow)
    # Serial: a native crash inside an xdist worker loses the step's output and
    # reports only "Segmentation fault", which names no test.
    assert "pytest tests/ -q -n 0" in python_job
    # faulthandler prints the stack of every thread when the process dies, and
    # the log survives the step view's truncation as an artifact.
    assert 'PYTHONFAULTHANDLER: "1"' in python_job
    assert "name: suite-log" in python_job
    assert "if: always()" in python_job


def test_every_dependency_is_pinned_where_the_references_were_frozen():
    """A range in pyproject is right for a user; a golden reference needs a VERSION.

    The audio references are frozen numbers from librosa/numba/numpy builds. An
    unconstrained resolve on the CI image installed librosa 1.0.1, numba 0.68.1,
    numpy 2.5.5 and PyQt6 6.12.1 where they were measured on 1.0.0/0.67.0/2.5.3/
    6.11.0, and 143 tests reported that drift as regressions.
    """
    import tomllib

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = list(project["dependencies"]) + list(project["optional-dependencies"]["dev"])
    required = {re.split(r"[<>=!~ ]", dep)[0] for dep in declared}

    # A pin may carry an environment marker (Qt is pinned per platform), so the
    # name is read before the `;` and the version after the `==`.
    pins: dict[str, set[str]] = {}
    for line in (REPO / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.split(";")[0].strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if separator and version.strip():
            pins.setdefault(name.strip(), set()).add(version.strip())

    missing = sorted(dep for dep in required if dep not in pins)
    assert not missing, f"not pinned in constraints.txt: {missing}"
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "-c constraints.txt" in workflow, "CI installs from the ranges again"


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
