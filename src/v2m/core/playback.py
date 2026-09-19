"""Local audio playback of the analysis for ear-checking.

The exported MIDI is rendered to a float32 signal with a tiny additive
synth (one envelope per note) and played through sounddevice — no
external synth or soundfont needed. The player exposes the current
position so the GUI can move a playhead in sync.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import audio_io

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - no audio device in CI
    sd = None


def render_notes(notes, sr: int = audio_io.SR, transpose: int = 0) -> np.ndarray:
    """Render `notes` to a mono float32 signal with a small additive synth.

    Each note becomes a soft square-ish tone (fundamental + 2nd/3rd
    partials) with a 10 ms attack and a 40 ms release, so the melody is
    clearly audible without sounding like a beep test.
    """
    if not notes:
        return np.zeros(0, dtype=np.float32)
    length = int(max(n.end for n in notes) * sr) + sr // 10
    y = np.zeros(length, dtype=np.float64)
    t_axis_cache: dict[int, np.ndarray] = {}

    def tone(freq: float, n: int) -> np.ndarray:
        # `n` is the EXACT slice length in samples: re-deriving it from a
        # duration in seconds (int(dur * sr)) loses one sample on ~9% of
        # durations (float round-trip) and misshapes the += into a crash.
        idx = t_axis_cache.setdefault(n, np.arange(n) / sr)
        env = np.ones(n)
        a = min(n, int(0.010 * sr))
        r = min(n - a, int(0.040 * sr))
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
        env[n - r :] *= np.linspace(1.0, 0.0, r)
        wave = np.sin(2 * np.pi * freq * idx) + 0.35 * np.sin(4 * np.pi * freq * idx) + 0.15 * np.sin(6 * np.pi * freq * idx)
        return 0.28 * env * wave

    for n in notes:
        i0, i1 = int(n.start * sr), min(length, int(n.end * sr))
        if i1 <= i0:
            continue
        f = float(440.0 * 2.0 ** ((int(n.midi) + transpose - 69) / 12.0))
        y[i0:i1] += tone(f, i1 - i0)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y *= 0.85 / peak
    return y.astype(np.float32)


class NotePlayer:
    """Play the rendered analysis through the default output device.

    `position()` returns the playback position in seconds so the GUI can
    move a playhead; playback runs in a daemon thread and can be stopped.
    """

    def __init__(self, notes, sr: int = audio_io.SR, transpose: int = 0):
        self.sr = sr
        self.signal = render_notes(notes, sr=sr, transpose=transpose)
        self.duration = len(self.signal) / sr
        self._t0: float | None = None
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice indisponible (pas de périphérique audio ?)")
        if self.playing:
            return
        if len(self.signal) == 0:
            return
        self._stop_evt.clear()
        self._t0 = time.perf_counter()

        def _run() -> None:
            try:
                sd.play(self.signal, self.sr)
                sd.wait()
            except Exception:
                pass  # device errors end playback silently
            finally:
                self._t0 = None

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def play_from(self, t: float) -> None:
        """Start playback at time `t` seconds (double-click seek)."""
        self.stop()
        if len(self.signal) == 0 or t >= self.duration:
            return
        i0 = max(0, int(t * self.sr))
        self._stop_evt.clear()
        self._t0 = time.perf_counter() - t

        def _run(offset: int = i0) -> None:
            try:
                sd.play(self.signal[offset:], self.sr)
                sd.wait()
            except Exception:
                pass
            finally:
                self._t0 = None

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if sd is not None and self._thread is not None:
            self._stop_evt.set()
            try:
                sd.stop()
            except Exception:
                pass
        self._thread = None
        self._t0 = None

    def position(self) -> float:
        """Current playback position in seconds (duration when finished)."""
        if self._t0 is None:
            return self.duration
        t = time.perf_counter() - self._t0
        return min(t, self.duration)
