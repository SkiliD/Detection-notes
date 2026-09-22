"""Local audio playback of the analysis for ear-checking.

Two distinct jobs, two distinct types:

- :func:`render_notes` / :func:`pitch_shift_signal` / :class:`NotePlayer`
  are the RENDER ENGINE: they turn notes (or a voice signal) into the
  float32 buffers the results page listens to. No thread, no device.
- :class:`PlaybackSession` is the PLAYER: one clock, one OutputStream,
  three modes (piano / voice / both). It used to share that job with
  independent ``NotePlayer`` threads — the two sources drifted, the
  window had to arbitrate which one owned the playhead, and every
  rebuild had to know both ways of starting a sound. Playback now has
  one owner; the GUI never starts a sound itself.

``NotePlayer`` keeps a thin ``signal``/``duration`` shape because the GUI
stores it (the piano source is re-rendered on gain/transpose changes) and
tests assert on the rendered buffers.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import audio_io

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - no audio device in CI
    sd = None

_CHUNK = 1024  # ~23 ms at 44.1 kHz: stop latency stays imperceptible


@dataclass(frozen=True)
class Timeline:
    """Sample-accurate duration shared by every playback source."""

    samples: int
    sr: int

    @property
    def duration(self) -> float:
        return max(0, self.samples) / self.sr

    def sample_at(self, seconds: float) -> int:
        value = float(seconds)
        if not math.isfinite(value):
            value = 0.0 if math.isnan(value) or value < 0 else self.duration
        return min(max(0, int(value * self.sr)), max(0, self.samples))

    def clamp(self, seconds: float) -> float:
        value = float(seconds)
        if math.isnan(value):
            return 0.0
        return min(max(0.0, value), self.duration)


def render_notes(
    notes,
    sr: int = audio_io.SR,
    transpose: int = 0,
    gain: float = 1.0,
) -> np.ndarray:
    """Render `notes` to a mono float32 signal with a small additive piano.

    Each note becomes a struck string: hammer-fast attack, exponential
    decay while the key holds, inharmonic partials (a stiff string's
    partials sit slightly sharp, which is the piano's metallic edge) whose
    higher ranks die faster. Enough realism to ear-check a melody without a
    soundfont, cheap enough for hundreds of notes. `gain` scales the render
    for mixing against another source (e.g. the singer's own voice).
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
        # hammer attack (4 ms) and a key-lift fade; the decay itself lives
        # on each partial below (higher partials die faster, like a string)
        env = np.ones(n)
        a = min(n, int(0.004 * sr))
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
        r = min(n - a, int(0.060 * sr))
        env[n - r :] *= np.linspace(1.0, 0.0, r)
        wave = np.zeros(n)
        for k in range(1, 7):
            fk = freq * k * np.sqrt(1.0 + 0.0004 * k * k)
            if fk > sr / 2:
                break
            tau = 0.9 / k**0.8
            wave += k**-1.7 * np.exp(-idx / tau) * np.sin(2 * np.pi * fk * idx)
        return 0.5 * env * wave

    for n in notes:
        i0, i1 = int(n.start * sr), min(length, int(n.end * sr))
        if i1 <= i0:
            continue
        f = float(440.0 * 2.0 ** ((int(n.midi) + transpose - 69) / 12.0))
        y[i0:i1] += tone(f, i1 - i0)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y *= 0.85 / peak
    # the gain is applied AFTER normalization (it must be audible) and the
    # result clipped: the device would hard-clip anything beyond ±1 anyway
    return np.clip(y * gain, -1.0, 1.0).astype(np.float32)


def pitch_shift_signal(y: np.ndarray, semitones: float, sr: int = audio_io.SR) -> np.ndarray:
    """Shift `y` by `semitones` WITHOUT changing its duration.

    A naive resample raises the pitch but also the speed, which would
    desynchronise the voice from the piano over a take. Librosa is already
    a project dependency for the analysis pipeline; its phase-vocoder keeps
    transposition musical on voice and preserves the timeline, while a
    small hand-written OLA produced pitch drift, silent tails and poor
    downward shifts.
    """
    if semitones == 0 or len(y) < 128:
        # There is no meaningful spectral frame in a tiny signal; preserve it
        # rather than asking the phase-vocoder to invent one.
        return np.asarray(y, dtype=np.float32)
    import librosa

    shifted = librosa.effects.pitch_shift(
        y=np.asarray(y, dtype=np.float32), sr=sr, n_steps=semitones
    )
    shifted = np.asarray(shifted, dtype=np.float32)
    if len(shifted) != len(y):
        # Keep the timeline contract exact despite backend rounding.
        shifted = np.interp(
            np.linspace(0.0, 1.0, len(y), endpoint=False),
            np.linspace(0.0, 1.0, len(shifted), endpoint=False),
            shifted,
        ).astype(np.float32)
    source_rms = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2))) if len(y) else 0.0
    shifted_rms = float(np.sqrt(np.mean(shifted.astype(np.float64) ** 2))) if len(shifted) else 0.0
    if source_rms and shifted_rms:
        shifted *= source_rms / shifted_rms
    return np.clip(shifted, -1.0, 1.0).astype(np.float32)


class NotePlayer:
    """A rendered source, ready for a PlaybackSession: nothing plays here.

    Built from notes it renders the additive piano; from a ready signal
    (the voice) it keeps that. The old class also PLAYED its own signal in
    a thread — that second way of starting a sound is what the session
    replaced, so the play/stop/position half is gone: a NotePlayer is now
    just a buffer with its duration, and the GUI hands it to the session.
    """

    def __init__(
        self,
        notes=None,
        sr: int = audio_io.SR,
        transpose: int = 0,
        gain: float = 1.0,
        signal: np.ndarray | None = None,
        on_error=None,
    ):
        self.sr = sr
        if signal is not None:
            # a raw signal (the singer's voice) plays at its own level;
            # transpose shifts it in place, keeping the duration exact so
            # it stays in sync with the piano over the whole take
            self.signal = pitch_shift_signal(signal, transpose, sr) if transpose else np.asarray(signal, dtype=np.float32)
        else:
            self.signal = render_notes(notes or [], sr=sr, transpose=transpose, gain=gain)
        self.duration = len(self.signal) / sr


class PlaybackSession:
    """One clock and one output stream for piano, voice, or their mix.

    The GUI used to start two independent playing threads for the combined
    mode: the sources drifted and the window had to choose which player
    owned the playhead. This session owns the timeline; sources are
    prepared by the render engine and the selected one is mixed before
    output. `position()` is the only clock every consumer reads.
    """

    def __init__(self, sr: int = audio_io.SR, on_error=None) -> None:
        self.sr = sr
        self.piano = np.zeros(0, dtype=np.float32)
        self.voice = np.zeros(0, dtype=np.float32)
        self._signal = np.zeros(0, dtype=np.float32)
        self.timeline = Timeline(0, sr)
        self.mode: str | None = None
        self._t0: float | None = None
        self._gen = 0
        self._on_error = on_error

    @property
    def duration(self) -> float:
        return self.timeline.duration

    @property
    def playing(self) -> bool:
        return self._t0 is not None

    def set_sources(self, piano=None, voice=None) -> None:
        """Replace source signals; callers decide whether to resume playback."""
        self.stop()
        self.piano = np.asarray(piano if piano is not None else [], dtype=np.float32)
        self.voice = np.asarray(voice if voice is not None else [], dtype=np.float32)
        self.timeline = Timeline(max(len(self.piano), len(self.voice)), self.sr)

    def _selected(self, mode: str) -> np.ndarray:
        piano = self.piano
        voice = self.voice
        if mode == "voice":
            return voice
        if mode == "both" and len(voice):
            n = max(len(piano), len(voice))
            mixed = np.zeros(n, dtype=np.float32)
            mixed[: len(voice)] += voice
            mixed[: len(piano)] += piano
            return np.clip(mixed, -1.0, 1.0)
        return piano

    def _report_error(self, error: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(error)
            except Exception:
                pass

    @staticmethod
    def _shutdown(stream) -> None:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def play(self, mode: str = "piano", t: float = 0.0) -> bool:
        if sd is None:
            self._report_error("sounddevice indisponible (pas de périphérique audio ?)")
            return False
        self.stop()
        self._signal = self._selected(mode)
        self.timeline = Timeline(len(self._signal), self.sr)
        if not len(self._signal) or t >= self.duration:
            return False
        self.mode = mode
        self._gen += 1
        gen = self._gen
        offset = self.timeline.sample_at(t)
        t = self.timeline.clamp(t)
        self._t0 = time.perf_counter() - t
        # `sdkw` carries the not-None-narrowed module into the thread closure
        # (pyright does not follow the guard's narrowing across closures), and
        # `signal` is the snapshot the worker plays even if sources change.
        sdkw = sd
        signal = self._signal

        def run() -> None:
            stream = None
            try:
                stream = sdkw.OutputStream(samplerate=self.sr, channels=1, dtype="float32")
                stream.start()
                for i in range(offset, len(signal), _CHUNK):
                    if self._gen != gen:
                        break
                    stream.write(signal[i : i + _CHUNK])
            except Exception as error:
                self._report_error(str(error))
            finally:
                self._shutdown(stream)
                if self._gen == gen:
                    self._t0 = None
                    self.mode = None

        threading.Thread(target=run, daemon=True).start()
        return True

    def stop(self) -> None:
        self._gen += 1
        self._t0 = None
        self.mode = None

    def position(self) -> float:
        if self._t0 is None:
            return self.duration
        return self.timeline.clamp(time.perf_counter() - self._t0)
