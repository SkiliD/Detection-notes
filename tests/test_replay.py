"""The replay core (`core.replay`) and the GUI ReplayWorker.

Replaying a take through the REAL chain has ONE owner: `replay_take`
(the CLI script and the GUI worker are adapters). This test pins the
measured rules:

1. **1x real time**: delivering faster lies (7 notes instead of 24 on
   take 140425) — the fake stream sleeps between two blocks;
2. **injection, not a global patch**: the stream is passed as a
   parameter — a GUI replay NEVER replaces `audio_io.sd` (the real mic
   is read through it);
3. **the full chain**: the final reading goes through
   `LiveTake.finalize`, the comparison through `core.compare.breakdown`
   (one policy only);
4. **the worker never blocks the UI**: one signal, `stop()` breaks the
   wait loop, and the `<stem>.live_vs_final.json` sidecar is written
   from the worker's thread (the archive is the product).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from v2m.core import audio_io
from v2m.core.replay import _fake_input_stream, replay_take

# the synthetic singer (and the pumping helper) the harnesses own: the
# replay tests ride the SAME voice instead of inventing one
from tests.test_live_gui import FakeMic, _pump_until
from tests.test_pipeline import _melody


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _save_sung_wav(tmp_path: Path) -> Path:
    """A short take on disk, exactly what the GUI export would write."""
    import soundfile as sf

    wav = tmp_path / "prise" / "prise.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), _melody().astype(np.float32), audio_io.SR, subtype="PCM_16")
    return wav


@pytest.mark.slow  # a full real-time replay of the take (~2 s audio + analysis)
def test_replay_feeds_the_real_chain_in_real_time(tmp_path):
    """A replay goes through LiveCapture (blocks on the real clock), not in bulk."""
    from v2m.core.compare import breakdown

    # the take's own chain, but the stream is INJECTED (no global swap of
    # the real InputStream): the fake feed delivers THE TAKE's audio at 1x,
    # so the wait loop ends on the audio's duration, never on its deadline
    opened: list[dict] = []

    wav = _save_sung_wav(tmp_path)
    y = audio_io.load_file(str(wav), sr=44_100)
    feed = _fake_input_stream(y, 44_100)

    class TimedFake:
        def __init__(self, samplerate, channels, blocksize, callback, **kwargs):
            opened.append({"samplerate": samplerate, "blocksize": blocksize})
            self._stream = feed(samplerate, channels, blocksize, callback, **kwargs)

        def start(self) -> None:
            self._stream.start()

        def stop(self) -> None:
            self._stream.stop()

        def close(self) -> None:
            self._stream.close()

    row = replay_take(wav, stream_factory=TimedFake)
    # the injected factory opened the stream at the MIC's rate and block
    assert opened == [{"samplerate": 44_100, "blocksize": 1024}]
    # the audit row carries the ONE policy's decomposition
    assert row["live_notes"] >= 3
    assert row["final_notes"] >= 3
    assert row["contours"] <= row["live_notes"]
    assert 0.0 <= row["rate"] <= 1.0
    # the row carries the SCRIPT TABLE vocabulary; the block vocabulary
    # (`live`/`final`) is the sidecar's, written by the worker test below
    assert {
        "stem", "duration_s", "live_notes", "final_notes", "contours",
        "off1", "moved", "absorbed", "rate", "tempo_bpm",
    } <= set(row)


@pytest.mark.slow  # real-time delivery IS the assertion: it costs the duration
def test_replay_is_never_faster_than_the_take(tmp_path, monkeypatch):
    """Delivering 1.9 s of audio costs ~1.9 s: the real clock IS the measure."""
    monkeypatch.setattr(audio_io, "sd", __import__("types").SimpleNamespace(InputStream=FakeMic))
    wav = _save_sung_wav(tmp_path)
    y = audio_io.load_file(str(wav), sr=44_100)
    t0 = time.monotonic()
    replay_take(wav, stream_factory=_fake_input_stream(y, 44_100))
    elapsed = time.monotonic() - t0
    duration = y.shape[0] / 44_100
    assert elapsed >= duration * 0.8, (
        f"the replay delivered {duration:.1f} s in {elapsed:.1f} s : too fast to be honest"
    )


@pytest.mark.slow  # a full real-time replay on the worker's thread
def test_the_worker_replays_and_archives_the_breakdown(qapp, tmp_path, monkeypatch):
    """Take replayed at rest: the `<stem>.live_vs_final.json` sidecar exists."""
    from types import SimpleNamespace

    from v2m.gui.replay_worker import ReplayWorker

    monkeypatch.setattr(audio_io, "sd", SimpleNamespace(InputStream=FakeMic))
    wav = _save_sung_wav(tmp_path)
    stem = wav.with_suffix("")
    worker = ReplayWorker(stem)
    rows: list[dict] = []
    errors: list[str] = []
    worker.done.connect(rows.append)
    worker.failed.connect(errors.append)
    worker.start()
    assert _pump_until(qapp, lambda: bool(rows) or bool(errors), timeout=120), (
        "the worker never delivered its audit"
    )
    worker.wait(2_000)
    assert not errors, errors
    # the sidecar sits BESIDE the take's artifacts, readable back: the
    # weekly trend reads `rate` and `live` out of exactly this shape
    sidecar = Path(str(stem) + ".live_vs_final.json")
    assert sidecar.exists()
    block = json.loads(sidecar.read_text(encoding="utf-8"))
    assert block["live"] == rows[0]["live_notes"]
    assert block["rate"] == rows[0]["rate"]


def test_a_cancelled_worker_archives_nothing(qapp, tmp_path, monkeypatch):
    """`stop()` before execution: no sidecar, no done signal."""
    from types import SimpleNamespace

    from v2m.gui.replay_worker import ReplayWorker

    monkeypatch.setattr(audio_io, "sd", SimpleNamespace(InputStream=FakeMic))
    wav = _save_sung_wav(tmp_path)
    stem = wav.with_suffix("")
    worker = ReplayWorker(stem)
    rows: list[dict] = []
    worker.done.connect(rows.append)
    worker.stop()
    worker.start()
    worker.wait(60_000)
    assert not rows
    assert not Path(str(stem) + ".live_vs_final.json").exists()


def test_a_worker_without_wav_stays_silent(qapp, tmp_path):
    """No WAV (audio box unticked): the worker does nothing, without an error."""
    from v2m.gui.replay_worker import ReplayWorker

    worker = ReplayWorker(tmp_path / "live_20260927_120000")
    rows: list[dict] = []
    errors: list[str] = []
    worker.done.connect(rows.append)
    worker.failed.connect(errors.append)
    worker.start()
    worker.wait(5_000)
    assert not rows and not errors
