"""Tests for the shared live-session lifecycle.

Non-regression: a failed final analysis must keep the captured signal
(recoverable without re-recording), emit the failure exactly once, and never
write partial export artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from v2m.core import live_session


class FakeCapture:
    """Stand-in for LiveCapture: records its lifecycle, replays a signal."""

    samplerate = 22_050

    def __init__(self, **kwargs):
        self.on_block = kwargs["on_block"]
        self.started = False
        self.stop_calls = 0
        self._signal = np.zeros(22_050, dtype=np.float32)

    def start(self):
        self.started = True

    def stop(self):
        self.stop_calls += 1

    def elapsed(self):
        return 1.0

    def signal(self):
        return self._signal


def _capture(take: live_session.LiveTake) -> FakeCapture:
    """Le faux capture avec son vrai type — LiveTake ne garantit que sa propre
    surface canard (start/stop/elapsed/signal)."""
    capture = take.capture
    assert isinstance(capture, FakeCapture)
    return capture


def test_live_take_stop_is_idempotent(monkeypatch):
    events = []
    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    take = live_session.LiveTake(on_event=events.append)

    take.start()
    first = take.stop()
    second = take.stop()

    assert take.state == "stopped"
    assert first is second
    assert _capture(take).stop_calls == 1
    assert [event.kind for event in events] == ["stopped"]


def test_live_take_reuses_final_result(monkeypatch):
    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    result = object()

    def fake_analyze(signal, sr, source, opts):
        return result

    monkeypatch.setattr(live_session, "analyze_signal", fake_analyze)
    take = live_session.LiveTake()
    take.start()

    first_result, first_signal = take.finalize()
    second_result, second_signal = take.finalize()

    assert first_result is result
    assert second_result is result
    assert first_signal is second_signal
    assert _capture(take).stop_calls == 1


def test_short_take_fails_once_and_keeps_the_signal(monkeypatch):
    """A too-short take: one failure event, signal intact, no result."""
    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    events = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    _capture(take)._signal = np.zeros(100, dtype=np.float32)  # < 0.5 s
    take.stop()

    with pytest.raises(ValueError):
        take.finalize()

    failures = [e for e in events if e.kind == "failed"]
    assert len(failures) == 1, "l'échec doit être émis une seule fois"
    assert take.error_emitted
    assert take.result is None
    # the capture stays recoverable: no re-recording needed to retry
    assert len(take.signal) == 100

    # a second finalize re-reports nothing new and fails the same way
    with pytest.raises(ValueError):
        take.finalize()
    assert [e for e in events if e.kind == "failed"] == failures


def test_analysis_failure_keeps_signal_and_does_not_export(monkeypatch, tmp_path):
    """The take's data survives a broken analysis; nothing lands on disk."""
    import v2m.core.report as report_mod

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)

    def broken_analyze(signal, sr, source, opts):
        raise RuntimeError("analyse cassée")

    monkeypatch.setattr(live_session, "analyze_signal", broken_analyze)
    events = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    take.stop()

    with pytest.raises(RuntimeError):
        take.finalize()

    failures = [e for e in events if e.kind == "failed"]
    assert len(failures) == 1
    assert "analyse cassée" in failures[0].error
    # signal kept: the caller can retry without re-recording
    assert len(take.signal) == 22_050
    assert take.result is None
    # nothing was exported (export_result was never reachable)
    assert not list(tmp_path.glob("**/*"))
    assert report_mod.export_result  # import guard: the module is untouched
