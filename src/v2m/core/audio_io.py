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

SR = 22_050        # analysis sample rate
GATE_RMS = 0.008   # below this level a live block counts as silence

# A capture stream opener: the SAME signature and contract as
# `sd.InputStream` (opens, `start()`s, `stop()`/`close()`), so the real
# device and every test/replay fake are interchangeable. Injection is the
# whole point: a GUI replaying a take must not swap `sd.InputStream` on
# the module (a global) while the user's real microphone capture reads it.
StreamFactory = Callable[..., Any]

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
    take). The callback only copies the block and queues it.

    Blocks are roughly resampled towards `sr` (linear) to stay consistent
    with the analysis engine.
    """

    samplerate: int = 44_100          # device sample rate
    blocksize: int = 1024             # ~23 ms per block
    sr: int = SR                      # analysis sample rate
    on_block: Optional[Callable[[np.ndarray, float], None]] = None
    # The InputStream opener (same contract as `sd.InputStream`). None = the
    # real device; a replay or a test injects its own fake — never by
    # patching `audio_io.sd` while a GUI capture might read it.
    stream_factory: Optional[StreamFactory] = None
    chunks: list = field(default_factory=list, repr=False)
    # `Any` because sounddevice may be absent (the module-level guard), and a
    # closed stream is set back to None; both call sites re-guard.
    _stream: Any = None
    _queue: queue.Queue = field(default_factory=queue.Queue, repr=False)
    _worker: Optional[threading.Thread] = field(default=None, repr=False)
    _n_samples: int = field(default=0, repr=False)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[mic] {status}")
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        self.chunks.append(mono.copy())
        self._n_samples += len(mono)
        if self.on_block is not None:
            self._queue.put((mono, self.elapsed()))

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

    def elapsed(self) -> float:
        return self._n_samples / self.samplerate

    def start(self) -> None:
        if sd is None:  # pragma: no cover - needs a real audio device
            raise RuntimeError(tr("err sounddevice missing"))
        self._stream = (
            self.stream_factory or sd.InputStream
        )(
            samplerate=self.samplerate,
            channels=1,
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()
        if self.on_block is not None:
            self._worker = threading.Thread(
                target=self._drain, daemon=True, name="v2m-live-dsp"
            )
            self._worker.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._queue.put(None)  # finish the backlog before reporting the take
            self._worker.join(timeout=5.0)
            self._worker = None

    def signal(self) -> np.ndarray:
        """Full recorded signal, resampled to `sr` (for the final analysis)."""
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        y = np.concatenate(self.chunks).astype(np.float32)
        if self.samplerate == self.sr:
            return y
        ratio = self.sr / self.samplerate
        n_out = int(len(y) * ratio)
        idx = np.linspace(0, len(y) - 1, n_out)
        return np.interp(idx, np.arange(len(y)), y).astype(np.float32)

    @property
    def duration(self) -> float:
        return len(self.signal()) / self.sr if self.chunks else 0.0
