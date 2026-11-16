"""Fixtures: synthetic audio signals, sandboxed preferences, suite budget."""

import json
import time

import numpy as np
import pytest

from v2m.core import i18n
from v2m.gui import main_window, widgets

SR = 22_050


@pytest.fixture(scope="session", autouse=True)
def _sandbox_language_file(tmp_path_factory):
    """The suite never reads or writes the user's language/theme preferences.

    Without the sandbox a manual `--lang en` (or an anglophone user's
    persisted choice) makes the suite NON-reproducible — `cli.main`
    without --lang and MainWindow both call `get_language()`, which
    re-reads that file
    mid-run and flipped the whole process to English: the tests that
    expect French (the `match="décoder"` asserts, the `tr("progress
    done")` stage labels) failed depending on ambient state. We also
    force French at startup: the starting state is the same everywhere,
    on every run.
    """
    original = i18n.LANG_FILE
    original_theme_file = main_window.THEME_FILE
    theme_directory = tmp_path_factory.mktemp("theme")
    i18n.LANG_FILE = tmp_path_factory.mktemp("lang") / "lang"
    main_window.THEME_FILE = theme_directory / "theme"
    widgets.set_theme("dark")
    i18n.set_language("fr")
    yield
    i18n.LANG_FILE = original
    main_window.THEME_FILE = original_theme_file
    widgets.set_theme("dark")


# Cost contract: the full suite must stay under this
# budget — a slow test added beyond it must be an EXPLICIT decision
# (measure, lower the budget with a reason, or mark the test slow).
#
# Calibration history (budget = measured × 1.85
# on two concordant runs):
#   2026-09-24  51.3 s   483 tests  →  95 s
#   2026-09-27  62.0 s   633 tests  → 115 s
#   2026-09-27  88.6 s   658 tests  → 165 s (3 real-time replays added)
#   2026-09-27 101.7 s   677 tests  → 190 s (5 real-time replays total)
#   2026-09-27 106.5 s   696 tests  → 197 s (weekly/retention features)
#   2026-09-28 109.3 s   721 tests  → 205 s (comment edits, no runtime
#                                     effect)
#   2026-09-28  81.0 s   727 tests  →  150 s (both diagnostic scripts
#                                     run in-process, the error_paths
#                                     take analyzed once — see diagnostics)
#   2026-10-02 131.9 s   853 tests — GREEN but a LOADED machine, not an
#                                     idle calibration: the 31 s replay
#                                     deadline wait was removed this day
#                                     (the fake feed delivers the take's
#                                     own audio, ~27 s saved) and the same
#                                     files re-ran ×1.45 slower than
#                                     earlier the same day (16.65 →
#                                     24.23 s). Budget untouched; rerun
#                                     on an idle machine before any
#                                     recalibration.
# Load margin measured the same day (spinning hog processes): 84.1 s
# at ~60 % of 8 cores, 90.4 s at ~87 %, but 197.2 s at ~half of ONE
# core (×2.43) — the factor covers load and ~1.8x uniform slowdown,
# not a 2x slower machine class (the budget is a tripwire, not a
# comfort margin; see the calibration procedure).
SUITE_BUDGET_S = 150.0

#   2026-10-12  34.6 s  1058 tests  →  150 s (KEPT: the run is now the
#                                     `-n 2` one pyproject's addopts
#                                     pins — pytest-xdist, a dev extra;
#                                     the same files cost ~62 s on one
#                                     process. -n 4 is NOT an option:
#                                     the two real-take goldens
#                                     (test_voice_accuracy drift,
#                                     test_score_accuracy fingerprint)
#                                     then race the shared
#                                     <take>.observation.npz and read
#                                     0.474/0.522 strengths where the
#                                     sequential run reads 0.473/0.523.)
#   2026-10-19  40.5 s  1111 tests  →  150 s (test consolidation kept
#                                     transferred assertions; no cost
#                                     recalibration)

# The start of the cost: this module loads during COLLECTION, before the
# first test (a pytest_sessionstart hook here would not be registered
# yet: conftests load after sessionstart).
_SUITE_STARTED_AT = time.perf_counter()


def suite_cost_verdict(elapsed_s: float, budget_s: float = SUITE_BUDGET_S) -> bool:
    """The decision alone, testable without waiting for a real suite."""
    return elapsed_s > budget_s


def pytest_sessionfinish(session, exitstatus):
    """Prints the total cost; makes a too-slow GREEN suite fail.

    `wrap_session` reads `session.exitstatus` AFTER this hook: mutating
    the attribute changes the exit code, without overwriting an already
    recorded failure/interrupt (1/2/3 stay what they are — only a green
    run drifts to 1).
    """
    total = time.perf_counter() - _SUITE_STARTED_AT
    line = f"suite cost: {total:.1f} s (budget {SUITE_BUDGET_S:.0f} s)"
    if suite_cost_verdict(total):
        line += " — OVER BUDGET: mark the slow tests or raise SUITE_BUDGET_S on purpose"
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(line)
    else:  # --no-terminalreporter : the cost stays visible anyway
        print(line)
    if suite_cost_verdict(total) and session.exitstatus == 0:
        session.exitstatus = 1


def expect_or_dump(cond: bool, msg: str, **state) -> None:
    """An `assert` that dumps the FULL captured state when it fails.

    The flake-capture pattern born in test_profile_analysis: a test that
    failed once under machine load must, at the NEXT failure, show
    everything that decides it (return code, stdout, stderr, parsed JSON)
    — a bare assert only names the broken condition, and the state that
    would explain it is gone by the time anyone looks. String values
    print as-is (slice a huge stderr at the call site); dict/list values
    print as JSON. The first broken condition raises.
    """
    if cond:
        return
    parts = [f"FLAKE CAPTURE — {msg}"]
    for name, value in state.items():
        rendered = (
            json.dumps(value, ensure_ascii=False, indent=1)
            if isinstance(value, (dict, list))
            else str(value)
        )
        parts.append(f"--- {name}:\n{rendered}")
    raise AssertionError("\n".join(parts))


@pytest.fixture(autouse=True)
def _sandbox_live_takes_dir(tmp_path, monkeypatch):
    """The suite neither reads nor writes the user's prises_live/.

    The coherence aggregator, the weekly recap and the regression banner
    read `coherence_journal.json` and the take folders under that root:
    left pointing at the project root, a GUI test's take audit lands in
    the user's journal AND the real take history decides whether the
    regression banner fires mid-test (a banner the test never expects).
    Both module attributes are patched because consumers read them at
    call time through different modules (`coh.LIVE_TAKES_DIR` in the GUI,
    a fresh `from .takes import` inside the export path).
    """
    from v2m.core import coherence as coh
    from v2m.core import takes as takes_mod

    root = tmp_path / "prises_live"
    monkeypatch.setattr(coh, "LIVE_TAKES_DIR", root)
    monkeypatch.setattr(takes_mod, "LIVE_TAKES_DIR", root)
    yield


@pytest.fixture(autouse=True)
def _sandbox_driver_endpoints(monkeypatch):
    """The suite never reads this machine's audio-endpoint registry.

    `driver_fx.enhancement_note` reads what Windows records for the selected
    microphone, and the note it returns sits on the home page: a developer's
    motherboard endpoint publishes an effects chain, so the sentence would
    exist in GUI tests on one machine and nowhere else. Sandboxed to "no
    endpoint"; the note's own tests pass their endpoints explicitly.
    """
    from v2m.core import driver_fx

    monkeypatch.setattr(driver_fx, "_read_endpoints", lambda: [])
    yield


def sine_note(hz: float, dur_s: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Sine wave with a simplified ADSR envelope (avoids clicks)."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    y = amp * np.sin(2 * np.pi * hz * t)
    # envelope: 10 ms fade in/out
    fade = int(0.01 * sr)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (y * env).astype(np.float32)


def silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(dur_s * sr), dtype=np.float32)


@pytest.fixture
def sr() -> int:
    return SR
