"""LiveCapture: the DSP must run OUTSIDE the audio callback.

YIN and the segmentation cost several milliseconds per 23 ms block; run
inside the PortAudio callback they exceed the block's duration and the
driver drops audio (holes in the take). The callback only copies/stacks
the block; `on_block` is called by a dedicated thread draining the
queue.
"""

from __future__ import annotations

import threading
import time
import types

import numpy as np
import pytest

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


def test_input_devices_filter_outputs_and_make_duplicate_keys_unique(monkeypatch):
    fake = types.SimpleNamespace(
        query_devices=lambda: [
            {"name": "USB Mic", "max_input_channels": 1, "hostapi": 0},
            {"name": "Speakers", "max_input_channels": 0, "hostapi": 0},
            {"name": "USB Mic", "max_input_channels": 2, "hostapi": 0},
            {"name": "Built-in", "max_input_channels": 1, "hostapi": 1},
        ],
        query_hostapis=lambda: [{"name": "WASAPI"}, {"name": "MME"}],
    )
    monkeypatch.setattr(audio_io, "sd", fake)

    devices = audio_io.input_devices()

    assert [device.index for device in devices] == [0, 2, 3]
    assert len({device.key for device in devices}) == len(devices)
    assert devices[0].key == "WASAPI\x1fUSB Mic"
    assert devices[1].key == "WASAPI\x1fUSB Mic\x1f1"
    assert devices[0].label == "USB Mic — WASAPI"


def test_input_device_resolution_uses_saved_key_then_falls_back(monkeypatch):
    fake = types.SimpleNamespace(
        query_devices=lambda *args: (
            [
                {"name": "USB Mic", "max_input_channels": 1, "hostapi": 0},
                {"name": "Default Mic", "max_input_channels": 1, "hostapi": 0},
            ]
            if not args
            else {"name": "Default Mic"}
        ),
        query_hostapis=lambda: [{"name": "WASAPI"}],
        default=types.SimpleNamespace(device=(1, -1)),
    )
    monkeypatch.setattr(audio_io, "sd", fake)

    selected = audio_io.resolve_input_device("WASAPI\x1fUSB Mic")
    missing = audio_io.resolve_input_device("Removed\x1fOld Mic")
    default = audio_io.resolve_input_device(None)

    assert selected == audio_io.ResolvedInputDevice(0, "USB Mic — WASAPI")
    assert missing == audio_io.ResolvedInputDevice(1, "Default Mic — WASAPI", fell_back=True)
    assert default == audio_io.ResolvedInputDevice(1, "Default Mic — WASAPI")


def _default_fake(devices, default_device, host_apis, default_info):
    """A sounddevice stand-in whose default lives behind `kind="input"`."""

    def query_devices(*args, **kwargs):
        if kwargs.get("kind") == "input":
            if default_info is None:
                raise RuntimeError("no default input")
            return default_info
        return devices

    return types.SimpleNamespace(
        query_devices=query_devices,
        query_hostapis=lambda: [{"name": name} for name in host_apis],
        default=types.SimpleNamespace(device=default_device),
    )


def test_the_unnamed_default_is_resolved_to_its_real_endpoint(monkeypatch):
    """PortAudio's default pair is (-1, -1) until someone chooses.

    An anonymous "System default" cannot be identified by the driver-effects
    reader (nor captured raw), so the endpoint PortAudio would really open is
    named instead.
    """
    devices = [{"name": "Microphone (2- Arctis Nova 7X)", "max_input_channels": 1, "hostapi": 1}]
    fake = _default_fake(
        devices,
        (-1, -1),
        ["MME", "Windows WASAPI"],
        {"name": devices[0]["name"], "hostapi": 1},
    )
    monkeypatch.setattr(audio_io, "sd", fake)
    assert audio_io.resolve_input_device(None) == audio_io.ResolvedInputDevice(
        0, "Microphone (2- Arctis Nova 7X) — Windows WASAPI"
    )


def test_an_mme_default_is_taken_on_its_wasapi_twin(monkeypatch):
    """The system mapper is MME: 31-char names and no exclusive mode.

    WASAPI is where the name is complete and the raw capture possible, so the
    default is taken on the twin of the SAME microphone (one candidate only).
    """
    devices = [
        {"name": "SteelSeries Sonar - Microphone", "max_input_channels": 1, "hostapi": 0},
        {
            "name": "SteelSeries Sonar - Microphone (SteelSeries Sonar Virtual Audio Device)",
            "max_input_channels": 1,
            "hostapi": 1,
        },
    ]
    fake = _default_fake(
        devices,
        (0, -1),
        ["MME", "Windows WASAPI"],
        {"name": devices[0]["name"], "hostapi": 0},
    )
    monkeypatch.setattr(audio_io, "sd", fake)
    resolved = audio_io.resolve_input_device(None)
    assert resolved.index == 1
    assert resolved.name.endswith("— Windows WASAPI")


def test_an_ambiguous_twin_keeps_the_mapped_default(monkeypatch):
    """Two candidates extending the truncated name is a guess: refuse it."""
    devices = [
        {"name": "Mic", "max_input_channels": 1, "hostapi": 0},
        {"name": "Mic (Adapter A)", "max_input_channels": 1, "hostapi": 1},
        {"name": "Mic (Adapter B)", "max_input_channels": 1, "hostapi": 1},
    ]
    fake = _default_fake(devices, (0, -1), ["MME", "Windows WASAPI"], {"name": "Mic", "hostapi": 0})
    monkeypatch.setattr(audio_io, "sd", fake)
    assert audio_io.resolve_input_device(None) == audio_io.ResolvedInputDevice(0, "Mic — MME")


def test_an_unreadable_default_stays_unnamed(monkeypatch):
    """No answer is not an invitation to guess an endpoint."""
    fake = _default_fake([], (-1, -1), ["MME"], None)
    monkeypatch.setattr(audio_io, "sd", fake)
    assert audio_io.resolve_input_device(None) == audio_io.ResolvedInputDevice(
        None, "System default"
    )


def test_on_block_runs_on_a_dedicated_thread(monkeypatch):
    # the test would call the callback from the main thread: on_block must
    # NOT run there (that is the whole point of the split)
    seen: list[tuple[float, float, str]] = []
    cap = _capture(
        monkeypatch,
        lambda blk, t: seen.append((blk[0], t, threading.current_thread().name)),
    )
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


def test_queued_blocks_own_samples_when_callback_buffer_is_reused(monkeypatch):
    """The DSP queue must retain each callback's samples, not its buffer."""
    entered = threading.Event()
    release = threading.Event()
    seen: list[float] = []

    def delayed_on_block(block, _t):
        if not entered.is_set():
            entered.set()
            release.wait(2.0)
        seen.append(float(block[0]))

    cap = _capture(monkeypatch, delayed_on_block)
    shared = np.full((BLOCK, 1), 0.1, dtype=np.float32)
    try:
        cap._callback(shared, BLOCK, None, None)
        assert _wait(entered.is_set), "the drain never picked up the first block"
        for value in (0.2, 0.3):
            shared.fill(value)
            cap._callback(shared, BLOCK, None, None)
        release.set()
        cap.stop()
    finally:
        release.set()
        if cap._worker is not None:
            cap.stop()

    assert seen == pytest.approx([0.1, 0.2, 0.3])
    assert [float(chunk[0]) for chunk in cap.chunks] == pytest.approx([0.1, 0.2, 0.3])


def test_capture_opens_the_requested_input_endpoint(monkeypatch):
    cap = _capture(monkeypatch, None, device=7)
    assert cap._stream.kwargs["device"] == 7
    cap.stop()


def test_signal_is_resampled_to_the_analysis_rate(monkeypatch):
    cap = _capture(monkeypatch, None, samplerate=48_000, sr=22_050)
    try:
        _push(cap, 0.5, n=20)
    finally:
        cap.stop()
    y = cap.signal()
    assert y.dtype == np.float32
    assert abs(len(y) - int(20 * BLOCK * 22_050 / 48_000)) <= 1


def _tone(freq: float, sr: int, dur_s: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur_s)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _with_signal(x: np.ndarray, samplerate: int, sr: int) -> np.ndarray:
    """A capture holding `x` as its captured audio; no device involved."""
    cap = audio_io.LiveCapture(samplerate=samplerate, sr=sr)
    cap.chunks.append(x)
    return cap.signal()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_downsampling_does_not_fold_out_of_band_energy_into_the_audible_band():
    """A rate cut without an anti-alias filter invents audible content.

    A 20 kHz tone sampled at 44.1 kHz sits above the target Nyquist
    (11 025 Hz): the reading must throw it away. Measured against a plain
    linear interpolation, it comes out as a 2.05 kHz component whose RMS
    is 0.208 of an input at 0.354 — an audible tone nobody sang, ready for
    pYIN. The filtered conversion must keep it at least 40 dB down.
    """
    x = _tone(20_000, 44_100)
    out = _with_signal(x, samplerate=44_100, sr=22_050)
    assert _rms(out) < 0.01 * _rms(x), (
        f"out-of-band energy folded in: rms={_rms(out):.5f} vs input {_rms(x):.5f}"
    )


def test_the_filtered_conversion_keeps_clock_and_duration_at_44100_and_48000():
    """The conversion is a rate change, never a clock change.

    Whatever the device rate, the analysis-rate output keeps the take's
    duration (its length is the same ratio, ±1 sample) and a passband
    tone keeps its own frequency and level — the final reading must not
    drift because the device ran at 48 kHz.
    """
    for device_rate in (44_100, 48_000):
        out = _with_signal(_tone(440.0, device_rate), samplerate=device_rate, sr=22_050)
        assert out.dtype == np.float32
        assert abs(len(out) - 22_050) <= 1, f"{device_rate}: duration changed"
        spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
        freqs = np.fft.rfftfreq(len(out), d=1 / 22_050)
        peak_hz = float(freqs[int(np.argmax(spectrum))])
        assert peak_hz == pytest.approx(440.0, abs=1.0), f"{device_rate}: {peak_hz} Hz"
        assert _rms(out) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)


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


def test_a_stop_past_the_join_timeout_counts_the_dropped_blocks(monkeypatch):
    """A stop that cannot wait out the backlog counts the abandoned blocks
    (the blocks stay in the WAV: the final analysis reads the take whole)."""
    handled = threading.Event()
    release = threading.Event()

    def slow_on_block(block, t):
        handled.set()
        release.wait(2.0)  # the drain stays busy past any join timeout

    monkeypatch.setattr(audio_io, "DRAIN_JOIN_TIMEOUT_S", 0.05)
    cap = _capture(monkeypatch, slow_on_block)
    try:
        _push(cap, 0.1, 3)  # three blocks queued, one being handled
        assert _wait(handled.is_set), "the drain never picked up the first block"
        cap.stop()
    finally:
        release.set()
        if cap._worker is not None and cap._worker.is_alive():
            cap._worker.join(2.0)
    assert cap.dropped == 2, (
        "the two queued audio blocks must be counted exactly; the stop sentinel "
        f"is not audio: dropped={cap.dropped}"
    )
    # the WAV side is complete regardless: the callback archived every block
    assert len(cap.chunks) == 3


def test_drain_min_witnesses_a_mid_take_starvation_that_recovers(monkeypatch):
    """A take that starved and RECOVERED reads dropped=0 and progress 1.0 at
    stop: the worst mid-take sample the drain itself took is the only
    witness the hunger left."""
    first_done = threading.Event()
    release = threading.Event()

    def on_block(block, t):
        if not first_done.is_set():
            first_done.set()
            release.wait(2.0)  # the first block ties the drain up

    cap = _capture(monkeypatch, on_block)
    try:
        _push(cap, 0.1, 3)  # 1 in flight, 2 queued behind it
        assert _wait(first_done.is_set), "the drain never picked up the first block"
    finally:
        release.set()
        cap.stop()
    # everything was handled in the end: the stop-time numbers are clean
    assert cap.dropped == 0
    assert cap.drain_progress() > 0.99
    # ... but the worst sample survived the recovery (1 handled / 2 queued)
    assert cap.drain_min < 0.5, f"drain_min={cap.drain_min}"


def test_drain_progress_reports_one_when_the_dsp_keeps_up(monkeypatch):
    """A capture whose drain handled everything reads progress 1.0; the
    singer's "did it follow me?" question has a measured answer."""
    seen: list[float] = []
    gate = threading.Event()

    def on_block(block, t):
        seen.append(t)
        gate.set()

    cap = _capture(monkeypatch, on_block)
    try:
        _push(cap, 0.1, 2)
        assert _wait(lambda: len(seen) >= 2), "the drain never handled the blocks"
        # queue empty, everything handled: ~1.0 (no real-time loss)
        assert cap.drain_progress() > 0.99
        assert cap.dropped == 0
    finally:
        cap.stop()


def test_elapsed_follows_the_sample_count(monkeypatch):
    """`elapsed` stays O(1) during the capture (a counter, not a sum)."""
    cap = _capture(monkeypatch, None)
    try:
        _push(cap, 0.3, n=4)
        assert cap.elapsed() == 4 * BLOCK / 44_100
    finally:
        cap.stop()
