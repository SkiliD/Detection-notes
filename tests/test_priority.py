"""Background priority: one call per thread, best-effort, caller's thread.

The background DSP (pYIN finalization, replay worker) runs below the
capture's priority so the drain thread wins the CPU. The contracts
pinned here: thread-local ONCE per thread (every worker pays its own
call), a second call is a no-op, and a platform without a usable
priority API degrades silently to the default priority. The call runs
FROM the thread being lowered: each test executes it inside a real
worker thread.
"""

from __future__ import annotations

import sys
import threading

import pytest

from v2m.gui import priority

# The lowering path is `ctypes.windll` itself, which does not exist off
# Windows: the two tests below fake that API, so they can only run where it
# exists. The cross-platform half (a new thread, a silent degradation) stays
# unmarked.


def _in_thread(fn) -> list:
    """Run fn in a fresh thread, return [result] or [exception]."""
    out: list = []

    def run() -> None:
        try:
            out.append(fn())
        except Exception as e:  # surfaced to the test, never swallowed
            out.append(e)

    t = threading.Thread(target=run)
    t.start()
    t.join(5.0)
    return out


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll: Windows API")
def test_a_lowering_platform_reports_true_once_then_noop(monkeypatch):
    # a platform that ACCEPTS the call: the first answer is True, the
    # second call in the same thread is a no-op (False) — the once-per-
    # thread contract, deterministically
    monkeypatch.setattr(priority.sys, "platform", "win32")

    class FakeWindll:
        class kernel32:
            @staticmethod
            def GetCurrentThread():
                return 1

            @staticmethod
            def SetThreadPriority(handle, prio):
                return True

    import ctypes

    monkeypatch.setattr(ctypes, "windll", FakeWindll())
    first, second = [], []

    def worker():
        first.append(priority.lower_background_priority())
        second.append(priority.lower_background_priority())

    t = threading.Thread(target=worker)
    t.start()
    t.join(5.0)
    assert first == [True], f"the first call must report the lowering: {first}"
    assert second == [False], "a second call in the same thread must be a no-op"


def test_a_new_thread_is_lowered_again():
    # the flag is thread-LOCAL: every worker pays its own call
    results = _in_thread(priority.lower_background_priority)
    results2 = _in_thread(priority.lower_background_priority)
    assert len(results) == 1 and not isinstance(results[0], Exception)
    assert len(results2) == 1 and not isinstance(results2[0], Exception)


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll: Windows API")
def test_a_raising_platform_degrades_to_the_status_quo(monkeypatch):
    # best-effort by contract: a priority API that raises leaves the thread
    # at the default priority, the function returns False, NO exception
    monkeypatch.setattr(priority._local, "done", False, raising=False)
    monkeypatch.setattr(priority.sys, "platform", "win32")

    def boom(*_a, **_k):
        raise OSError("refused")

    import ctypes

    monkeypatch.setattr(ctypes, "windll", None)  # attribute walk raises
    out = _in_thread(priority.lower_background_priority)
    assert out == [False], f"expected a silent False, got {out}"
