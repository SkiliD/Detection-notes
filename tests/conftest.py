"""Fixtures: synthetic audio signals, sandboxed language, suite budget."""

import time

import numpy as np
import pytest

from v2m.core import i18n

SR = 22_050


@pytest.fixture(scope="session", autouse=True)
def _sandbox_language_file(tmp_path_factory):
    """The suite neither reads nor writes the user's ~/.v2m_lang.

    Regression: a manual `--lang en` (or an anglophone user's persisted
    choice) made the suite NON-reproducible — `cli.main` without --lang
    and MainWindow both call `get_language()`, which re-read that file
    mid-run and flipped the whole process to English: the tests that
    expect French (the `match="décoder"` asserts, the `tr("progress
    done")` stage labels) failed depending on ambient state. We also
    force French at startup: the starting state is the same everywhere,
    on every run.
    """
    original = i18n.LANG_FILE
    i18n.LANG_FILE = tmp_path_factory.mktemp("lang") / "lang"
    i18n.set_language("fr")
    yield
    i18n.LANG_FILE = original


# Cost contract (AGENTS.md): the full suite must stay under this
# budget — a slow test added beyond it must be an EXPLICIT decision
# (measure, lower the budget with a reason, or mark the test slow).
# Recalibrated 2026-09-24 on the day's measured speed: full suite
# 51.3 s (483 tests), environment margin x1.85 => 95 s.
# Recalibrated 2026-09-27: the suite counts 633 tests (+ new live/final,
# coherence, replay measures) and the machine's speed of the day is very
# variable (the same suite measures 90 to 115 s under load) — nominal
# ~62 s x1.85 => 115 s. The calibration procedure lives in AGENTS.md
# ("Calibration du budget suite"): a runaway budget is first a machine
# SPEED to compare, not a test to delete.
# Recalibrated 2026-09-27 (EXPLICIT decision, same day): +26 tests (the
# 4 follow-up features: replay, explain, trends) including THREE
# real-time replays marked slow (~30 s: the 1x real clock IS the
# assertion); full suite measured 88.6 s on two concordant runs,
# 658 tests, x1.85 => 165 s.
# Recalibrated again on 2026-09-27 (health curve, regression, cache):
# 677 tests, five real-time replays in total (the accelerated cache adds
# two, needed to prove the cache does NOT change the reading);
# suite measured 101.7 s on two concordant runs x1.85 => 190 s.
# Recalibrated one last time on 2026-09-27 (user feedback: anti-starving
# guards, per-moment-of-day view, retention, weekly PDF, --changed):
# 696 tests, suite measured 106.5 s on two concordant runs
# x1.85 => 197 s.
# Recalibrated 2026-09-28 (comments/docstrings translated to English —
# no runtime effect): 721 tests, suite measured 109.3 s
# x1.85 => ~202 s, rounded to 205 s.
SUITE_BUDGET_S = 205.0

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
