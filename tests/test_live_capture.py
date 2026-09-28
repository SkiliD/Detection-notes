"""LiveCapture: the DSP must run OUTSIDE the audio callback.

Regression: the PortAudio callback did the YIN and the segmentation
(several milliseconds per 23 ms block); when it exceeds the block's
duration, the driver drops audio (holes in the take). The callback now
only copies/stacks the block; `on_block` is called by a dedicated thread
draining the queue.
"""

from __future__ import annotations

import threading
import time
import types

import numpy as np

from v2m.core import audio_io

BLOCK = 1024
DRAIN_THREAD = "v2m-live-dsp"


class FakeStream:
    """A stand-in `sounddevice.InputStream`: reads no mic."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.kwargs = kwargs
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def _capture(monkeypatch, on_block, **kwargs) -> audio_io.LiveCapture:
    monkeypatch.setattr(audio_io, "sd", types.SimpleNamespace(InputStream=FakeStream))
    kwargs.setdefault("samplerate", 44_100)
    cap = audio_io.LiveCapture(on_block=on_block, **kwargs)
    cap.start()
    return cap


def _push(cap: audio_io.LiveCapture, value: float, n: int = 1) -> None:
    """Replays what PortAudio would deliver: mono float32 blocks."""
    block = np.full((BLOCK, 1), value, dtype=np.float32)
    for _ in range(n):
        cap._callback(block, BLOCK, None, None)


def test_on_block_runs_on_a_dedicated_thread(monkeypatch):
    # the test would call the callback from the main thread: on_block must
    # NOT run there (that is the whole point of the split)
    seen: list[tuple[float, float, str]] = []
    cap = _capture(monkeypatch, lambda blk, t: seen.append((blk[0], t, threading.current_thread().name)))
    try:
        _push(cap, 0.25, n=3)
        assert _wait(lambda: len(seen) == 3), "the blocks never reached on_block"
    finally:
        cap.stop()

    assert {s[2] for s in seen} == {DRAIN_THREAD}
    assert [round(s[0], 3) for s in seen] == [0.25, 0.25, 0.25]
    # timestamping: the capture time grows with the blocks, never recounting
    ts = [s[1] for s in seen]
    assert ts == sorted(ts) and ts[-1] > ts[0]


def test_callback_never_loses_blocks_and_stop_flushes(monkeypatch):
    seen: list[float] = []
    cap = _capture(monkeypatch, lambda blk, t: seen.append(t))
    _push(cap, 0.1, n=8)
    cap.stop()  # must drain the queue before returning
    assert len(seen) == 8
    assert len(cap.chunks) == 8
    assert cap.elapsed() == 8 * BLOCK / cap.samplerate


def test_signal_is_resampled_to_the_analysis_rate(monkeypatch):
    cap = _capture(monkeypatch, None, samplerate=48_000, sr=22_050)
    try:
        _push(cap, 0.5, n=20)
    finally:
        cap.stop()
    y = cap.signal()
    assert y.dtype == np.float32
    assert abs(len(y) - int(20 * BLOCK * 22_050 / 48_000)) <= 1


def test_a_broken_on_block_does_not_stop_the_capture(monkeypatch, capsys):
    seen: list[float] = []

    def on_block(block, t) -> None:
        seen.append(t)
        if len(seen) == 1:
            raise RuntimeError("bug de l'appelant")

    cap = _capture(monkeypatch, on_block)
    try:
        _push(cap, 0.2, n=4)
        assert _wait(lambda: len(seen) == 4), "la file a été abandonnée après l'erreur"
    finally:
        cap.stop()
    assert "bug de l'appelant" in capsys.readouterr().out


def test_elapsed_follows_the_sample_count(monkeypatch):
    """`elapsed` reste en O(1) pendant la capture (un compteur, pas une somme)."""
    cap = _capture(monkeypatch, None)
    try:
        _push(cap, 0.3, n=4)
        assert cap.elapsed() == 4 * BLOCK / 44_100
    finally:
        cap.stop()
