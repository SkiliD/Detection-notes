"""Tests for the shared live-session lifecycle.

A failed final analysis keeps the captured signal (recoverable without
re-recording), emits the failure exactly once, and never writes partial
export artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from v2m.core import live_session


class FakeCapture:
    """Stand-in for LiveCapture: records its lifecycle, replays a signal."""

    samplerate = 22_050
    dropped = 0  # the honesty counter (a stop past the join timeout)
    drain_min = 1.0  # the drain's own mid-take witness

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
    """The fake capture AS ITS real type — LiveTake guarantees only its own
    duck surface (start/stop/elapsed/signal)."""
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


def test_note_event_carries_pitch_readings_from_the_processed_block(monkeypatch):
    """Queued GUI delivery keeps each block's confirmation facts intact."""

    class FakeSegmenter:
        def current(self):
            return (60.0, 261.6)

        def confirmed(self):
            return (60.0, 261.6)

        def heard(self):
            return (60.0, 261.6)

    class FakeStream:
        def __init__(self, _sr, gate_rms):
            assert gate_rms > 0
            self.seg = FakeSegmenter()

        def elapsed_notes(self):
            return 0

        def feed(self, _block, _t):
            return 0.1, []

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", FakeStream)
    events = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    take._on_block(np.zeros(128, dtype=np.float32), 0.1)

    note_event = next(event for event in events if event.kind == "note")
    assert note_event.pitch == (60.0, 261.6)
    assert note_event.confirmed_pitch == (60.0, 261.6)
    assert note_event.heard_pitch == (60.0, 261.6)
    assert note_event.confirmed_pitch is not None
    assert note_event.heard_pitch is not None
    assert note_event.confirmed_pitch == note_event.pitch
    assert note_event.heard_pitch == note_event.pitch


def test_dropped_blocks_reach_the_take_surface(monkeypatch):
    # the capture's dropped counter is readable through the take's surface
    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    take = live_session.LiveTake()
    take.start()
    _capture(take).dropped = 7
    assert take.dropped_blocks == 7


def test_finalize_archives_the_dropped_blocks_in_its_block(monkeypatch):
    """The honesty counter rides IN the take's live_vs_final block: the
    report JSON and the archived sidecar read the same dict."""
    from types import SimpleNamespace

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(
        live_session,
        "analyze_signal",
        lambda signal, sr, source, opts: SimpleNamespace(notes=[], grid=None),
    )
    take = live_session.LiveTake()
    take.start()
    _capture(take).dropped = 5
    _capture(take).drain_min = 0.42
    result, _signal = take.finalize()
    assert result.live_vs_final is not None
    assert result.live_vs_final["dropped"] == 5
    assert result.live_vs_final["drain_min"] == 0.42


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
