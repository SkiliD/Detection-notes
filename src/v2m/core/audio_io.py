"""Audio loading (file) and microphone capture (real-time)."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .i18n import tr

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - no audio device in CI
    sd = None

SR = 22_050  # analysis sample rate
GATE_RMS = 0.008  # below this level a live block counts as silence

# How long a stop waits for the drain to finish the backlog before it
# counts the rest as dropped (a constant, so the honesty test can shrink
# it instead of sleeping past a literal 5 s).
DRAIN_JOIN_TIMEOUT_S = 5.0

# A capture stream opener: the SAME signature and contract as
# `sd.InputStream` (opens, `start()`s, `stop()`/`close()`), so the real
# device and every test/replay fake are interchangeable. Injection is the
# whole point: a GUI replaying a take must not swap `sd.InputStream` on
# the module (a global) while the user's real microphone capture reads it.
StreamFactory = Callable[..., Any]

# The separator `InputDevice.label` appends before the host API. One spelling:
# `core.driver_fx` strips it to match a displayed label back to the OS's own
# endpoint record.
INPUT_LABEL_SEPARATOR = " — "


@dataclass(frozen=True)
class InputDevice:
    """One available input endpoint with a stable preference key."""

    index: int
    key: str
    name: str
    host_api: str

    @property
    def label(self) -> str:
        """A user-facing name that distinguishes backend duplicates."""
        return f"{self.name}{INPUT_LABEL_SEPARATOR}{self.host_api}"


@dataclass(frozen=True)
class ResolvedInputDevice:
    """The endpoint a capture should open and the name shown to the user."""

    index: int | None
    name: str
    fell_back: bool = False


def input_devices() -> list[InputDevice]:
    """List input-capable endpoints; device indexes are runtime-only."""
    if sd is None:
        return []
    try:
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
    except Exception:
        return []
    occurrences: dict[tuple[str, str], int] = {}
    result: list[InputDevice] = []
    for index, info in enumerate(devices):
        if int(info.get("max_input_channels", 0)) < 1:
            continue
        host_index = int(info.get("hostapi", -1))
        host_api = (
            str(host_apis[host_index]["name"]) if 0 <= host_index < len(host_apis) else "Audio"
        )
        name = str(info["name"])
        identity = (host_api, name)
        occurrence = occurrences.get(identity, 0)
        occurrences[identity] = occurrence + 1
        key = f"{host_api}\x1f{name}"
        if occurrence:
            key = f"{key}\x1f{occurrence}"
        result.append(InputDevice(index, key, name, host_api))
    return result


def _portaudio_default_input(available: list[InputDevice]) -> InputDevice | None:
    """The endpoint PortAudio would open by default, as a LISTED device.

    Matching is on the (name, host API) pair, never the name alone: the same
    microphone is listed once per backend, and a name-only match could name
    another backend's endpoint — a different stream than the one opened. None
    when the answer cannot be recovered, so the caller keeps its unnamed
    default instead of guessing.
    """
    if sd is None:
        return None
    try:
        info = sd.query_devices(kind="input")
        host_apis = sd.query_hostapis()
    except Exception:
        return None
    name = str(info.get("name", ""))
    host_index = int(info.get("hostapi", -1))
    if not (0 <= host_index < len(host_apis)):
        return None
    host_api = str(host_apis[host_index]["name"])
    return next(
        (device for device in available if device.name == name and device.host_api == host_api),
        None,
    )


WASAPI_HOST_API = "Windows WASAPI"


def _prefer_wasapi_twin(default: InputDevice, available: list[InputDevice]) -> InputDevice:
    """The same microphone on WASAPI, when exactly one candidate exists.

    PortAudio's default input is the system mapper: an MME endpoint, whose
    names Windows truncates to 31 characters and which has no exclusive mode.
    The registry reader matches a full friendly name and the raw capture needs
    WASAPI, so the default is taken on the WASAPI twin of the SAME
    microphone. Never a guess: one candidate whose name extends the mapper's
    truncated one is required, and anything else keeps the mapped default.
    """
    if default.host_api == WASAPI_HOST_API:
        return default
    twins = [
        device
        for device in available
        if device.host_api == WASAPI_HOST_API and device.name.startswith(default.name)
    ]
    return twins[0] if len(twins) == 1 else default


def resolve_input_device(device_key: str | None) -> ResolvedInputDevice:
    """Resolve a saved choice, falling back to the current system default."""
    available = input_devices()
    if device_key is not None:
        selected = next((device for device in available if device.key == device_key), None)
        if selected is not None:
            return ResolvedInputDevice(selected.index, selected.label)

    default_index: int | None = None
    default_name = "System default"
    if sd is not None:
        try:
            default_devices = sd.default.device
            if default_devices is not None:
                index = (
                    default_devices[0]
                    if isinstance(default_devices, (tuple, list))
                    else default_devices
                )
                if index is not None and int(index) >= 0:
                    default_index = int(index)
        except Exception:
            default_index = None
        default = next((device for device in available if device.index == default_index), None)
        if default is None:
            # `sd.default.device` is (-1, -1) when the user never chose one, so
            # the pair's index says nothing; PortAudio's default lives behind
            # `query_devices(kind="input")`. Naming THAT endpoint is what lets
            # the driver-effects reader identify it (and the raw mode engage)
            # instead of stopping at an anonymous "System default".
            default = _portaudio_default_input(available)
        if default is not None:
            default = _prefer_wasapi_twin(default, available)
        if default is not None:
            default_index = default.index
            default_name = default.label
        elif default_index is not None:
            try:
                info = sd.query_devices(default_index)
            except Exception:
                info = None
            if info is not None:
                default_name = str(info.get("name", default_name))

    return ResolvedInputDevice(
        index=default_index,
        name=default_name,
        fell_back=device_key is not None,
    )


# An audio OUTPUT stream opener for a replayed take (`core.replay`): same
# contract as `sd.OutputStream`. Optional — a replay without a speaker
# (headless machine, CI) still runs, it just does not monitor itself.

# Accepted input containers, shared by the CLI and the GUI (a tuple:
# `str.endswith` accepts it, unlike a set).
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")


class AudioLoadError(ValueError):
    """Raised when an audio file cannot be read or decoded."""


def load_file(path: str, sr: int = SR) -> np.ndarray:
    """Load an audio file as mono, resampled to `sr` (float32 in [-1, 1]).

    Raises AudioLoadError with a clear message on missing, empty or
    undecodable input instead of leaking a low-level exception.
    """
    p = Path(path)
    if not p.exists():
        raise AudioLoadError(tr("err missing file").format(path=p))
    try:
        import librosa

        y, _ = librosa.load(str(p), sr=sr, mono=True)
    except AudioLoadError:
        raise
    except Exception as e:
        raise AudioLoadError(tr("err undecodable").format(name=p.name, e=e)) from e
    if len(y) == 0:
        raise AudioLoadError(tr("err empty file").format(name=p.name))
    return np.asarray(y, dtype=np.float32)


def save_file(signal: np.ndarray, path: str, sr: int = SR) -> str:
    """Write a mono float32 signal to `path`; return the path written.

    Used for the live take's own audio (the live page's checkbox): the take is
    the user's data, so it is written as 16-bit PCM WAV, the format every
    editor reads, right next to the other artifacts. The folder is created if
    needed (a per-take live folder may not exist yet).
    """
    import soundfile as sf

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), np.asarray(signal, dtype=np.float32), sr, subtype="PCM_16")
    return str(p)


@dataclass
class LiveCapture:
    """Microphone capture in blocks, with a user callback.

    `on_block(block, t)` is called from a **worker thread**, never from the
    PortAudio callback: YIN and the note segmentation cost milliseconds per
    block and make the driver drop audio when they run inline (holes in the
    take). The callback only copies the block and queues it; the resampling
    to `sr` happens ONCE, in `signal()`, at read time.

    Real-time honesty: the drain thread can fall behind a hot DSP or a
    loaded machine, and the queue grows SILENTLY — the singer watches notes
    vanish with no explanation. Two counters make the loss visible:
    `dropped` (blocks abandoned when a stop cannot wait for the backlog
    any more) and `drain_progress()` (drained / captured, ~1.0 when the
    DSP keeps up). A capture that ends with `dropped > 0` or a progress
    far from 1.0 was a REAL-TIME loss, not a recognition failure: the
    offline analysis of the same WAV reads it fine.
    """

    samplerate: int = 44_100  # device sample rate
    blocksize: int = 1024  # ~23 ms per block
    sr: int = SR  # analysis sample rate
    gate_rms: float = GATE_RMS  # the silence gate (blocks below it are mute)
    on_block: Optional[Callable[[np.ndarray, float], None]] = None
    # The InputStream opener (same contract as `sd.InputStream`). None = the
    # real device; a replay or a test injects its own fake — never by
    # patching `audio_io.sd` while a GUI capture might read it.
    stream_factory: Optional[StreamFactory] = None
    device: int | None = None  # None delegates to PortAudio's current default
    # Open the endpoint in RAW (WASAPI exclusive) mode: shared mode is where
    # the driver's audio effects chain runs, so exclusive is the only way past
    # it. A refusal is not an error — the capture reopens shared — and
    # `exclusive_opened` reports what was actually opened, never what was
    # asked for.
    exclusive: bool = False
    exclusive_opened: bool = field(default=False, repr=False)
    chunks: list = field(default_factory=list, repr=False)
    # `Any` because sounddevice may be absent (the module-level guard), and a
    # closed stream is set back to None; both call sites re-guard.
    _stream: Any = None
    _queue: queue.Queue = field(default_factory=queue.Queue, repr=False)
    _worker: Optional[threading.Thread] = field(default=None, repr=False)
    _n_samples: int = field(default=0, repr=False)
    # Real-time honesty counters: a block QUEUED by the callback is not a
    # block HANDLED. `_n_dropped` counts the backlog abandoned at a stop;
    # `_n_drained` feeds `drain_progress()`.
    _n_drained: int = field(default=0, repr=False)
    _n_dropped: int = field(default=0, repr=False)
    # The drain samples its OWN progress after every handled block; the
    # WORST sample is the take's witness. No extra thread, no timer: the
    # sampling rides the loop that already exists, and a mid-take starvation
    # that the stop-time snapshot never sees is still in the number.
    _worst_drain: float = field(default=1.0, repr=False)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[mic] {status}")
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        owned = mono.copy()
        self.chunks.append(owned)
        self._n_samples += len(owned)
        if self.on_block is not None:
            # PortAudio may reuse `indata` as soon as this callback returns;
            # the drain owns its block until `on_block` finishes with it.
            self._queue.put((owned, self.elapsed()))

    def _drain(self) -> None:
        """Hand the queued blocks to `on_block` (its own thread)."""
        # start() only launches this thread when on_block exists; the alias
        # carries that fact where pyright cannot follow it.
        on_block = self.on_block
        if on_block is None:
            return
        while True:
            item = self._queue.get()
            if item is None:  # sentinel: everything captured has been handled
                return
            block, t = item
            try:
                on_block(block, t)
            except Exception as e:  # one bad block must not end the capture
                print(f"[live] {e}")
            finally:
                self._n_drained += 1
                # the drain's own witness: sample AFTER the bookkeeping, when
                # the backlog this block left behind is already counted
                progress = self.drain_progress()
                if progress < self._worst_drain:
                    self._worst_drain = progress

    def elapsed(self) -> float:
        return self._n_samples / self.samplerate

    @property
    def dropped(self) -> int:
        """Blocks the drain abandoned (a stop past the join timeout)."""
        return self._n_dropped

    @property
    def drain_min(self) -> float:
        """The WORST drain progress the drain itself observed mid-take.

        `drain_progress()` answers only when someone asks; the moment a
        take starved and RECOVERED, the stop-time snapshot reads 1.0 and
        the starvation is invisible. The drain samples after every block,
        so the worst sample survives the recovery — a hollow live note
        count whose counters all look clean has this number to blame.
        """
        return self._worst_drain

    def drain_progress(self) -> float:
        """Drained / captured blocks (~1.0 when the DSP keeps up).

        Below ~0.95 the drain is falling behind in real time: the live
        reading of the trailing seconds will be incomplete even though
        the WAV keeps everything.
        """
        n = self._n_drained + self._queue.qsize()
        return self._n_drained / n if n else 1.0

    def start(self) -> None:
        if sd is None:  # pragma: no cover - needs a real audio device
            raise RuntimeError(tr("err sounddevice missing"))
        stream_options = {
            "samplerate": self.samplerate,
            "channels": 1,
            "blocksize": self.blocksize,
            "callback": self._callback,
        }
        if self.device is not None:
            stream_options["device"] = self.device
        self._stream = self._open(stream_options)
        try:
            self._stream.start()
        except Exception:
            if not self.exclusive_opened:
                raise
            # the endpoint accepted the raw open and refused the clock: one
            # take is worth more than the mode, so reopen shared and report it
            self._stream.close()
            self.exclusive_opened = False
            self._stream = (self.stream_factory or sd.InputStream)(**stream_options)
            self._stream.start()
        opened_rate = getattr(self._stream, "samplerate", None)
        if opened_rate:
            # the STREAM's rate is the take's clock: an exclusive open takes
            # the device's own format, so `signal()` and `elapsed()` must read
            # the rate that was really opened, not the one requested
            self.samplerate = int(opened_rate)
        if self.on_block is not None:
            self._worker = threading.Thread(target=self._drain, daemon=True, name="v2m-live-dsp")
            self._worker.start()

    def _open(self, options: dict) -> Any:
        """Open the capture: raw first when asked, else as configured.

        WASAPI's exclusive mode takes only the device's OWN format, so the raw
        attempt asks for that rate instead of the shared-mode one. Every way a
        driver can refuse (a busy endpoint, no exclusive mode in the backend,
        a format it will not take) ends in the shared stream: the take still
        happens, and only `exclusive_opened` remembers the refusal.
        """
        factory = self.stream_factory
        if factory is None:
            if sd is None:  # pragma: no cover - start() already refused this
                raise RuntimeError(tr("err sounddevice missing"))
            factory = sd.InputStream
        if self.exclusive:
            settings = getattr(sd, "WasapiSettings", None)
            if settings is not None:
                raw = {**options, "samplerate": self._device_rate()}
                try:
                    stream = factory(**raw, extra_settings=settings(exclusive=True))
                except Exception:
                    self.exclusive_opened = False
                else:
                    self.exclusive_opened = True
                    return stream
        try:
            return factory(**options)
        except Exception:
            # A refused raw open can leave a SHARED endpoint that rejects the
            # requested rate too (measured: a virtual device listed at 48 kHz
            # refuses 44.1 kHz outright). Asking once more at the endpoint's
            # own rate keeps the take; only a device that refuses BOTH fails.
            rate = self._device_rate()
            if rate == options["samplerate"]:
                raise
            return factory(**{**options, "samplerate": rate})

    def _device_rate(self) -> int:
        """The endpoint's own sample rate — what an exclusive open accepts."""
        if self.device is None or sd is None:
            return self.samplerate
        try:
            info = sd.query_devices(self.device)
            return int(round(float(info["default_samplerate"]))) or self.samplerate
        except Exception:
            return self.samplerate

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._queue.put(None)  # finish the backlog before reporting the take
            self._worker.join(timeout=DRAIN_JOIN_TIMEOUT_S)
            if self._worker.is_alive():
                # the drain could not finish the backlog: COUNT the abandoned
                # blocks. Each is in the WAV (the callback queued it before
                # dying), so the final analysis still reads the take whole —
                # but this take's live reading has holes, and the counters
                # must say so instead of a silently hollow note count.
                # The sentinel marks the end of the backlog; if the worker
                # is still processing a block, it remains queued and is not
                # one of the captured audio blocks abandoned here.
                self._n_dropped = max(0, self._queue.qsize() - 1)
            self._worker = None

    def signal(self) -> np.ndarray:
        """Full recorded signal, resampled to `sr` (for the final analysis).

        The rate cut is FILTERED: linear interpolation between the device
        samples folds everything above the target Nyquist back into the
        audible band (measured: a 20 kHz tone at 44.1 kHz reads back as a
        2.05 kHz tone at 0.208 RMS, ready for pYIN), while the final
        analysis must hear only what was sung. `resample_poly` lowpasses
        before the decimation and keeps the clock: the output length is
        the same rate ratio (±1 sample) and a passband tone keeps its
        frequency and level at 44.1 and 48 kHz.
        """
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        y = np.concatenate(self.chunks).astype(np.float32)
        if self.samplerate == self.sr:
            return y
        # local import: the module stays import-light (librosa and
        # soundfile are read the same way, at their call site)
        from fractions import Fraction

        from scipy.signal import resample_poly

        ratio = Fraction(self.sr, self.samplerate).limit_denominator(1_000)
        out = resample_poly(y, ratio.numerator, ratio.denominator)
        return np.asarray(out, dtype=np.float32)

    @property
    def duration(self) -> float:
        return len(self.signal()) / self.sr if self.chunks else 0.0
