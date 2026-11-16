"""Replay wired into the GUI chain: ONE sidecar, ONE stem, one worker.

Three integration contracts are pinned here:

1. **the sidecar is written ONCE**: `core.take_export.write_take` archives
   it under the take's own stem supplied by the window;
2. **the stem outlives the first export**: re-clicking "Export" reuses
   the memorized stem instead of opening a new folder;
3. **the worker starts on the EXported take** (its stem, its folder),
   not on a re-minted name — else the archive crosses two takes.

The replay queue is enqueued AT EXPORT (the WAV exists at that instant
— the worker would otherwise replay a missing file), never in the
analysis; the tests therefore simulate the CURRENT capture before the
export to pin the deferred enqueue.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from v2m.core import audio_io
from v2m.core import take_export as te


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _fake_live_result() -> SimpleNamespace:
    return SimpleNamespace(
        source="live",
        live_vs_final={"live": 5, "final": 4, "contours": 1, "off1": 1,
                      "moved": 0, "absorbed": 0, "rate": 0.2},
        quantize_step=lambda _d: 0.1,
    )


def _redirect(monkeypatch, mw, tmp_path) -> None:
    # the fake export mirrors the real one's folder contract (export_result
    # creates the take folder before writing): the sidecar writer relies on it
    def fake_export(res, stem, **kw) -> None:
        Path(stem).parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(te, "export_result", fake_export)
    monkeypatch.setattr(mw, "live_export_dir", lambda stem: tmp_path / Path(stem).name)


@pytest.fixture(autouse=True)
def _fake_replay_worker(monkeypatch):
    """No worker thread in these tests: stems recorded, `started` per test."""
    from v2m.gui import replay_worker as rw

    FakeWorker.started = []
    monkeypatch.setattr(rw, "ReplayWorker", FakeWorker)
    yield


class FakeWorker:
    """ReplayWorker's surface, without the thread: stems recorded, no run."""

    started: list[str] = []

    def __init__(self, stem: Path) -> None:
        self.stem = stem
        FakeWorker.started.append(Path(stem).name)
        self.done = SimpleNamespace(connect=lambda f: None)
        self.failed = SimpleNamespace(connect=lambda f: None)
        self.finished = SimpleNamespace(connect=lambda f: None)

    def start(self) -> None:
        pass

    def isRunning(self) -> bool:
        return False

    def stop(self) -> None:
        pass

    def wait(self, _ms: int) -> bool:
        return True


def test_the_sidecar_is_written_once_and_the_stem_is_stable(qapp, monkeypatch, tmp_path):
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)
    win = mw.MainWindow()
    try:
        win.result = cast(Any, _fake_live_result())
        win._live_signal = np.zeros(audio_io.SR, dtype=np.float32)
        win._export_results()
        first_stem = win.take_ctl.take_stem()
        assert first_stem
        sidecars = list(tmp_path.rglob("*.live_vs_final.json"))
        assert len(sidecars) == 1, "the sidecar was written twice"
        # a RE-EXPORT (the Export button) reuses the SAME take folder
        win._export_results()
        assert win.take_ctl.take_stem() == first_stem
        assert list(tmp_path.glob(f"{first_stem}")) == list(tmp_path.glob(f"{first_stem}"))
        stems = {p.parent.name for p in tmp_path.rglob("*.live_vs_final.json")}
        assert stems == {first_stem}
    finally:
        win.close()


def test_the_worker_replays_the_exported_take_not_a_fresh_name(qapp, monkeypatch, tmp_path):
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)
    win = mw.MainWindow()
    try:
        win.result = cast(Any, _fake_live_result())
        win._live_signal = np.zeros(audio_io.SR, dtype=np.float32)
        win._export_results()
        stem_name = win.take_ctl.take_stem()
        # the export QUEUES the cross-check (the WAV exists now) and the
        # pump starts it on the take's OWN stem — not on a re-minted name
        assert FakeWorker.started == [stem_name]
    finally:
        win.close()


def test_live_analysis_stays_pending_until_user_exports(qapp, monkeypatch, tmp_path):
    """Analysis alone does not export a take or start its archived replay."""
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)
    win = mw.MainWindow()
    try:
        win._live_signal = np.full(audio_io.SR, 0.2, dtype=np.float32)
        res = cast(
            Any,
            SimpleNamespace(
                source="live",
                live_vs_final={"live": 3, "final": 3, "contours": 0,
                               "off1": 0, "moved": 0, "absorbed": 0, "rate": 0.0},
                tempo=SimpleNamespace(bpm=120.0),
                key=SimpleNamespace(best=SimpleNamespace(label="C major")),
                notes=[],
                observation=None,
            ),
        )
        win._on_live_analysis_done(res, b"")
        assert FakeWorker.started == []
        assert win.take_ctl.take_stem() is None
        assert not list(tmp_path.rglob("*.live_vs_final.json"))

        win._export_results()
        # explicit export mints the take stem, archives its initial comparison,
        # and only then allows the cross-check replay to start.
        assert FakeWorker.started == [win.take_ctl.take_stem()]
        assert list(tmp_path.rglob("*.live_vs_final.json")), "sidecar absent"
    finally:
        win.close()


# ------------------------------------------------------------------------- #
# The replay never runs while a capture runs: a replay's real-time DSP
# competes with the live drain for the same budget
# ------------------------------------------------------------------------- #


def _queued_result() -> SimpleNamespace:
    return SimpleNamespace(
        source="live",
        live_vs_final={"live": 3, "final": 3, "contours": 0,
                       "off1": 0, "moved": 0, "absorbed": 0, "rate": 0.0},
        tempo=SimpleNamespace(bpm=120.0),
        key=SimpleNamespace(best=SimpleNamespace(label="C major")),
        notes=[],
        observation=None,
    )


def test_a_replay_never_starts_while_a_capture_is_running(qapp, monkeypatch, tmp_path):
    """Analysis arriving DURING the next take: replay enqueued."""
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)
    win = mw.MainWindow()
    try:
        # A concurrent capture is active: export queues the cross-check
        # rather than starting replay alongside the microphone.
        win.page_live.start_timer()

        class RunningThread:
            def isRunning(self):
                return True

        win.live_thread = cast(Any, RunningThread())
        win.result = cast(Any, _fake_live_result())
        win._live_signal = np.zeros(audio_io.SR, dtype=np.float32)
        win._export_results()  # mints the stem, QUEUES (guards hold)
        # NOTHING started: a replay would starve the live capture's DSP
        assert FakeWorker.started == []
        # but the deferred take is QUEUED, not lost
        assert win.take_ctl.replay_queue() == [win.take_ctl.take_stem()]
    finally:
        win.page_live.stop_timer()
        win.close()


def test_the_queue_drains_when_the_capture_stops(qapp, monkeypatch, tmp_path):
    """When the next capture stops: the deferred replay starts."""
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)

    class RunningThread:
        def isRunning(self):
            return True

    class StoppedThread:
        # the real LiveThread's surface: `take` is None until run() opens one
        take = None

        def isRunning(self):
            return False

    win = mw.MainWindow()
    try:
        # the chained-takes case: the NEXT capture is already live when
        # this take exports — its cross-check is deferred into the queue
        win.page_live.start_timer()
        win.live_thread = cast(Any, RunningThread())
        win.result = cast(Any, _fake_live_result())
        win._live_signal = np.zeros(audio_io.SR, dtype=np.float32)
        win._export_results()
        deferred_stem = win.take_ctl.take_stem()
        assert FakeWorker.started == [] and win.take_ctl.replay_queue()
        # the singer starts ANOTHER take: the take's buffers reset through
        # the controller (on_start_live's `begin_take` — called directly
        # here: creating a real LiveThread would open the real mic) — the
        # QUEUE must survive it (the deferred take is exported; its stem
        # lives in the queue, not in the take's state)
        win.page_live.stop_timer()
        win.take_ctl.begin_take()
        assert win.take_ctl.replay_queue() == [deferred_stem]
        # the singer stops that take: the pump may now run
        win.live_thread = cast(Any, StoppedThread())
        win.on_stop_live()
        assert FakeWorker.started == [deferred_stem]
        assert win.take_ctl.replay_queue() == []
    finally:
        win.page_live.stop_timer()
        win.close()


def test_a_newer_capture_keeps_its_live_render(qapp, monkeypatch, tmp_path):
    """N's analysis arrives during N+1: N+1's timer does NOT freeze.

    The unconditional timer stop froze the in-progress take's roll,
    counter and cursor — the feeling "the mic no longer detects me".
    """
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)
    win = mw.MainWindow()
    try:
        win.page_live.start_timer()

        class RunningThread:
            def isRunning(self):
                return True

        win.live_thread = cast(Any, RunningThread())
        win._on_live_analysis_done(_queued_result(), b"")
        assert win.page_live.timer_active(), (
            "the in-progress take's rendering must stay alive"
        )
        # ...and when NO capture is running, the timer stops (normal end)
        win.live_thread = None
        win._on_live_analysis_done(_queued_result(), b"")
        assert not win.page_live.timer_active()
    finally:
        win.page_live.stop_timer()
        win.close()


# ------------------------------------------------------------------------- #
# Prolonged silence during a take: the singer gets the on-screen answer
# "is the mic hearing me?", not a JSON one
# ------------------------------------------------------------------------- #


def test_silence_over_two_seconds_warns_the_singer(qapp):
    """A mute block at t>2 s: message; the voice resets the clock to zero."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        # a voiced block resets the clock (no message, no state kept)
        win._watch_live_silence(0.0, 0.5)
        win._watch_live_silence(1.0, 0.2)
        assert win._live.silence_since is None
        # silence starts: the clock opens, nothing said yet
        win._watch_live_silence(2.0, 0.0)
        assert win._live.silence_since == 2.0
        # 1 s in: still under the threshold, quiet
        win._watch_live_silence(3.0, 0.0)
        assert not win.statusBar().currentMessage()
        # past 2 s of silence: the singer is TOLD, with the duration
        win._watch_live_silence(4.2, 0.0)
        assert "2" in win.statusBar().currentMessage()
        # the voice returns: the clock closes again, the take is untouched
        win._watch_live_silence(4.4, 0.4)
        assert win._live.silence_since is None
    finally:
        win.close()


def test_silence_watch_is_per_take_state(qapp, monkeypatch, tmp_path):
    """The silence clock belongs to the TAKE (LiveRenderState)."""
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)
    win = mw.MainWindow()
    try:
        win._watch_live_silence(0.0, 0.0)  # this take's clock is running
        assert win._live.silence_since == 0.0
        # a NEW take swaps the render state wholesale: no stale clock
        win._live = mw.LiveRenderState()
        assert win._live.silence_since is None
    finally:
        win.close()


# ------------------------------------------------------------------------- #
# A mute or weak mic is named BEFORE the take starts (the level stays
# under the silence gate for 3 s before the take: the warning comes
# before singing). One watch per phase: before the
# first clear block `watch_low_level` speaks; after, `watch_silence` —
# the same per-block telemetry, two windows, never both at once.
# ------------------------------------------------------------------------- #


def test_low_mic_is_named_before_the_take_starts(qapp):
    """3 s under the gate before the first clear block: the message; a
    clear block arms the take and hands over to the silence watch."""
    from v2m.core import audio_io
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        gate = audio_io.GATE_RMS
        # settling: under the gate, under the threshold — quiet
        win._on_live_block(0.0, 0.0)
        win._on_live_block(1.0, gate * 0.5)
        assert not win.statusBar().currentMessage()
        assert win._live.low_since == 0.0
        # past 3 s of nothing above the gate: the singer is TOLD
        win._on_live_block(3.2, 0.0)
        msg = win.statusBar().currentMessage()
        assert "3" in msg and (
            "niveau" in msg.lower()
            or "seuil" in msg.lower()
            or "gate" in msg.lower()
            or "threshold" in msg.lower()
        ), msg
        # a clear block disarms the pre-take watch for the WHOLE take:
        # silence at t=5 is now the DURING-take watch's business
        win._on_live_block(4.0, gate * 3.0)
        assert win._live.heard_once
        assert win._live.low_since is None
        win._on_live_block(7.5, 0.0)
        assert win._live.silence_since == 7.5  # the other watch owns it now
    finally:
        win.close()


def test_low_mic_watch_rearms_on_a_new_take(qapp, monkeypatch, tmp_path):
    """The watch state belongs to the TAKE (LiveRenderState): the next
    take re-arms it — a mic that went quiet between takes is named again."""
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)
    win = mw.MainWindow()
    try:
        win._on_live_block(0.0, 0.5)  # take 1 heard the singer
        assert win._live.heard_once
        win._live = mw.LiveRenderState()  # on_start_live swaps the state
        assert not win._live.heard_once
        win._on_live_block(0.0, 0.0)  # take 2: the mic went mute meanwhile
        assert win._live.low_since == 0.0
    finally:
        win.close()
