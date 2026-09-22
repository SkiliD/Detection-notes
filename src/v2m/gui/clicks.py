"""Audio output for the metronome clicks (keeps playback out of the GUI)."""

from __future__ import annotations

import threading

import numpy as np

from ..core import audio_io
from ..core.metronome import click

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - no audio device in CI
    sd = None


class ClickPlayer:
    """Play one click per call through the default output device.

    `sd.play` would stop the previous sound before starting the next (and it
    steals the shared default stream the NotePlayer uses), so clicks stream
    through their own OutputStream: the writer pulls queued clicks, silence
    fills the gaps, and a click can overlap the tail of the previous one.
    """

    def __init__(self, sr: int = audio_io.SR):
        self.sr = sr
        self._queue: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None

    def _writer(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        with self._lock:
            pending, self._queue = self._queue, []
        buf = np.zeros(frames, dtype=np.float32)
        pos = 0
        for c in pending:
            end = min(len(buf), pos + len(c))
            if end > pos:
                buf[pos:end] += c[: end - pos]
            if end < len(c):
                with self._lock:
                    self._queue.insert(0, c[end - pos:])
                break
            pos = end
            if pos >= frames:
                break
        outdata[:, 0] = buf

    def start(self) -> None:
        if sd is None or self._stream is not None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sr, channels=1, dtype="float32",
                blocksize=512, callback=self._writer,
            )
            self._stream.start()
        except Exception:
            self._stream = None  # no output device: the metronome stays visual

    def play(self, accent: bool = False) -> None:
        with self._lock:
            self._queue.append(click(self.sr, accent=accent))

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            self._queue.clear()
