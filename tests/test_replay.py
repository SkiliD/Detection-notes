"""The replay core (`core.replay`) and the GUI ReplayWorker.

Replaying a take through the REAL chain has ONE owner: `replay_take`
(the CLI script and the GUI worker are adapters). This test pins the
measured rules:

1. **1x real time**: the injected stream sleeps between blocks so the
   live recognizer receives the take at capture speed;
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

# the synthetic singer (and the pumping helper) the harnesses own: the
# replay tests ride the SAME voice instead of inventing one
from tests.test_live_gui import FakeMic, _pump_until
from tests.test_pipeline import _melody
from v2m.core import audio_io
from v2m.core.replay import _fake_input_stream, replay_take


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

    # the take's own chain, but the stream is INJECTED (no global swap of
    # the real InputStream): the fake feed delivers THE TAKE's audio at 1x,
    # so the wait loop ends on the audio's duration, never on its deadline
    opened: list[dict] = []

    wav = _save_sung_wav(tmp_path)
    y = audio_io.load_file(str(wav), sr=44_100)
    feed = _fake_input_stream(y, 44_100)
    started_at = time.monotonic()

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
    elapsed = time.monotonic() - started_at
    duration = y.shape[0] / 44_100
    assert elapsed >= duration * 0.8, (
        f"the replay delivered {duration:.1f} s in {elapsed:.1f} s : too fast to be honest"
    )
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
        "stem",
        "duration_s",
        "live_notes",
        "final_notes",
        "contours",
        "off1",
        "moved",
        "absorbed",
        "rate",
        "tempo_bpm",
        "dropped",
        "drain_min",
    } <= set(row)
    assert isinstance(row["dropped"], int) and row["dropped"] >= 0
    assert 0.0 <= row["drain_min"] <= 1.0


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


@pytest.mark.slow
def test_replay_reuses_valid_cache_without_pyin(tmp_path, monkeypatch):
    from v2m.core import pipeline
    from v2m.core.observation import VoiceObservation
    from v2m.core.pipeline import LIVE_CONF

    wav = _save_sung_wav(tmp_path)
    signal = audio_io.load_file(str(wav))
    obs = VoiceObservation.from_signal(signal, audio_io.SR, min_conf=LIVE_CONF)
    obs.source_size = wav.stat().st_size
    obs.source_mtime = wav.stat().st_mtime
    obs.save(wav.with_suffix(".observation.npz"))

    def forbidden(*args, **kwargs):
        raise AssertionError("a valid replay cache must avoid pYIN")

    measured_durations = []
    real_detect_tempo = pipeline.detect_tempo

    def detect_tempo(y, sr):
        assert sr == audio_io.SR
        measured_durations.append(len(y) / sr)
        return real_detect_tempo(y, sr)

    monkeypatch.setattr(VoiceObservation, "from_signal", forbidden)
    monkeypatch.setattr(pipeline, "detect_tempo", detect_tempo)
    row = replay_take(wav)
    assert row["final_notes"] >= 3
    assert measured_durations == pytest.approx([len(signal) / audio_io.SR], abs=0.001)


@pytest.mark.slow
def test_replay_counts_the_note_closed_at_stop(tmp_path, monkeypatch):
    import soundfile as sf

    from v2m.core import replay

    sr = replay.REPLAY_SR
    signal = (0.3 * np.sin(2 * np.pi * 330 * np.arange(sr) / sr)).astype(np.float32)
    wav = tmp_path / "held.wav"
    sf.write(wav, signal, sr)
    # The live count must include the flush independently of offline analysis.
    monkeypatch.setattr(replay, "_final_reading", lambda *args: None)
    row = replay_take(wav)
    assert row["live_notes"] == 1


def test_replay_delivers_every_sample_before_stopping(tmp_path, monkeypatch):
    from v2m.core import replay
    from v2m.core.live_session import LiveTake

    n_samples = replay.REPLAY_SR + 17
    signal = np.linspace(-0.1, 0.1, n_samples, dtype=np.float32)
    monkeypatch.setattr(audio_io, "load_file", lambda *args, **kwargs: signal)
    monkeypatch.setattr(replay, "_final_reading", lambda *args: None)
    delivered = []
    real_stop = LiveTake.stop

    def observed_stop(self):
        assert self.capture is not None
        delivered.append(sum(len(chunk) for chunk in self.capture.chunks))
        return real_stop(self)

    monkeypatch.setattr(LiveTake, "stop", observed_stop)
    # Remove wall-clock waits while exercising the actual stream callbacks.
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    replay_take(tmp_path / "tail.wav")
    assert delivered == [n_samples]


def test_cancelled_replay_stops_without_final_analysis(tmp_path, monkeypatch):
    from v2m.core import replay

    wav = _save_sung_wav(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("a cancelled replay must not finalize")

    monkeypatch.setattr(replay, "_final_reading", forbidden)
    assert replay_take(wav, stop=lambda: True) == {}


@pytest.mark.parametrize("ending", ["complete", "cancel", "timeout"])
def test_replay_waits_for_the_last_blocks_and_always_stops(tmp_path, monkeypatch, ending):
    from types import SimpleNamespace

    from v2m.core import live_session, replay

    clock = [0.0]
    capture = SimpleNamespace(
        state="recording",
        elapsed=0.0,
        segmenter=SimpleNamespace(notes=[]),
        dropped_blocks=0,
        drain_min=1.0,
    )
    stopped = []
    analysis_signals = []
    signal = np.zeros(replay.REPLAY_SR, dtype=np.float32)
    final_signal = np.zeros(audio_io.SR, dtype=np.float32)
    last_note = SimpleNamespace(start=0.98, end=1.0, midi=64)

    def advance(_):
        clock[0] += 31.0 if ending == "timeout" else 0.01
        capture.elapsed = min(1.0, capture.elapsed + 0.01)

    def stop_capture():
        stopped.append(capture.elapsed)
        if capture.elapsed >= 1.0:
            capture.segmenter.notes.append(last_note)
        return final_signal

    capture.start = lambda: None
    capture.stop = stop_capture
    monkeypatch.setattr(live_session, "LiveTake", lambda **kwargs: capture)
    monkeypatch.setattr(audio_io, "load_file", lambda *args, **kwargs: signal)
    monkeypatch.setattr(replay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(replay.time, "sleep", advance)
    monkeypatch.setattr(replay, "_final_reading", lambda take, path, y: analysis_signals.append(y))

    def cancel():
        return ending == "cancel" and capture.elapsed >= 0.5

    if ending == "timeout":
        with pytest.raises(TimeoutError, match="delivery"):
            replay_take(tmp_path / "tail.wav", stream_factory=lambda: None, stop=cancel)
        assert not analysis_signals
    else:
        row = replay_take(tmp_path / "tail.wav", stream_factory=lambda: None, stop=cancel)
        if ending == "cancel":
            assert row == {}
            assert not analysis_signals
        else:
            assert row["live_notes"] == 1
            assert stopped == [1.0]
            assert len(analysis_signals) == 1
            assert analysis_signals[0] is final_signal
    assert len(stopped) == 1
