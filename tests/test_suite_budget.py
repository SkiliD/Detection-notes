"""The suite's cost budget: a tripwire, and the ONE lever a shared runner needs.

`SUITE_BUDGET_S` is calibrated on a development machine — `tests/conftest.py`
keeps the dated history, and a green run over it exits 1. A CI runner is slower
and never idle, so the number is read from `V2M_SUITE_BUDGET_S`; the default is
the measured one, and an unreadable override must NOT disable the tripwire by
accident (a typo would otherwise buy unlimited time).
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import conftest  # noqa: E402


def test_the_default_is_the_measured_budget(monkeypatch):
    monkeypatch.delenv("V2M_SUITE_BUDGET_S", raising=False)
    assert conftest.suite_budget() == 150.0


def test_a_slower_runner_says_so_through_the_environment(monkeypatch):
    monkeypatch.setenv("V2M_SUITE_BUDGET_S", "600")
    assert conftest.suite_budget() == 600.0


def test_an_unreadable_override_keeps_the_measured_budget(monkeypatch):
    monkeypatch.setenv("V2M_SUITE_BUDGET_S", "ten minutes")
    assert conftest.suite_budget() == 150.0


def test_the_verdict_is_strictly_over_the_budget():
    # Exactly at the budget is not over it: the number is a ceiling, not a
    # target to cross.
    assert not conftest.suite_cost_verdict(150.0, 150.0)
    assert conftest.suite_cost_verdict(150.1, 150.0)
