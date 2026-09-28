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
from .i18n import tr

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - no audio device in CI
    sd = None

_CHUNK = 1024  # ~23 ms at 44.1 kHz: stop latency stays imperceptible

# --- Pedal & brightness (2026-09-25, user request) ------------------------
# A HELD note keeps ringing after the key is lifted: the damper is up
# (pedal down), the string decays through its own partials and only the
# tail fade ends it. Below HELD_MIN_S the note is a short one: the damper
# still lifts the key early (the old behavior, pinned by a test).
HELD_MIN_S = 0.35
PEDAL_TAIL_FACTOR = 0.30   # tail length = factor x written duration
PEDAL_TAIL_MAX_S = 0.90    # a pedal never rings forever
# Brightness: the partial AMPLITUDE exponent is velocity-dependent. A soft
# key excites mostly the fundamental (expo 2.1, dark); a hard hammer also
# excites the upper partials (expo 1.3, bright) — measured direction of a
# real piano's strike. The decay ranking (higher ranks die faster) is
# unchanged, so the brightness is strongest at the attack.
DARK_EXPO = 2.1
BRIGHT_EXPO = 1.3


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


def _note_velocity(n) -> float:
    """0..1 velocity of one note, from its measured onset accent.

    `Note.strength` is the take-relative attack strength the pipeline
    already computed (0..1, normalized on the loudest attack): it is the
    singer's own dynamics, so the piano follows it instead of hitting
    every key the same. A note without the field (or out of range) reads
    the neutral 0.5.
    """
    s = getattr(n, "strength", 0.5)
    s = 0.5 if s is None or s != s else float(s)
    return min(1.0, max(0.0, s))


def render_notes(
    notes,
    sr: int = audio_io.SR,
    transpose: int = 0,
    gain: float = 1.0,
) -> np.ndarray:
    """Render `notes` to a mono float32 signal with a small additive piano.

    Each note becomes a struck string: hammer attack, exponential decay
    while the key holds, inharmonic partials (a stiff string's partials sit
    slightly sharp, which is the piano's metallic edge) whose higher ranks
    die faster. Enough realism to ear-check a melody without a soundfont,
    cheap enough for hundreds of notes. `gain` scales the render for
    mixing against another source (e.g. the singer's own voice).

    The performance is deliberately NOT mechanical (the metronome-piano
    feedback): velocities follow the take's own accents (`_note_velocity`)
    and brighten the tone (a loud key rings with more upper partials), a
    soft key strikes slower than a loud one, notes are placed a few ms
    off the grid (later when weak), keyed a few cents apart, and HELD
    notes ring through a pedal tail after the key lift (a short note
    still has its damper). Everything sits in a light room (short noise
    IR). All deterministic: same notes -> same render.
    """
    if not notes:
        return np.zeros(0, dtype=np.float32)
    length = int(max(n.end for n in notes) * sr) + int(PEDAL_TAIL_MAX_S * sr) + sr // 10
    y = np.zeros(length, dtype=np.float64)
    t_axis_cache: dict[int, np.ndarray] = {}

    def tone(freq: float, n: int, vel: float, lift_s: float | None = None) -> np.ndarray:
        # `n` is the EXACT slice length in samples: re-deriving it from a
        # duration in seconds (int(dur * sr)) loses one sample on ~9% of
        # durations (float round-trip) and misshapes the += into a crash.
        idx = t_axis_cache.setdefault(n, np.arange(n) / sr)
        # hammer attack: a LOUD key strikes fast (3 ms), a soft one slow
        # (12 ms) — that attack-time spread is a big part of why a real
        # piano does not sound like a metronome
        env = np.ones(n)
        a = min(n, max(1, int((0.003 + 0.009 * (1.0 - vel)) * sr)))
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
        # velocity -> brightness: a hard hammer leaves more energy in the
        # upper partials (a softer exponent), a soft one sounds darker
        expo = DARK_EXPO - (DARK_EXPO - BRIGHT_EXPO) * vel
        # key-lift fade for a SHORT note: the damper sits on the string
        # and the decay is cut. A HELD note (lift_s given) instead steps
        # down softly to the string's own resonance level — the pedal lets
        # it ring — and fades only at the tail's end.
        r = min(n - a, int((0.06 + 0.04 * (1.0 - vel)) * sr))
        if lift_s is None:
            if r > 0:
                env[n - r :] *= np.linspace(1.0, 0.0, r)
        else:
            lift = min(n, int(lift_s * sr))
            if lift < n:
                # the hammer's direct sound stops; the string rings on at
                # ~60% of its current level, then the tail fades out over
                # its second half (a pedal lift, not a damper slam)
                step = min(n - lift, max(1, int(0.025 * sr)))
                env[lift : lift + step] *= np.linspace(1.0, 0.60, step)
                env[lift + step :] *= 0.60
                rest = n - lift - step
                if rest > 0:
                    half = rest // 2
                    env[lift + step + half :] *= np.linspace(1.0, 0.0, rest - half, endpoint=False)
        wave = np.zeros(n)
        for k in range(1, 7):
            fk = freq * k * np.sqrt(1.0 + 0.0004 * k * k)
            if fk > sr / 2:
                break
            tau = 0.9 / k**0.8
            wave += k**-expo * np.exp(-idx / tau) * np.sin(2 * np.pi * fk * idx)
        return (0.25 + 0.75 * vel) * env * wave

    for n in notes:
        vel = _note_velocity(n)
        # human placement: a weak note lands a few ms late, a strong one on
        # time; every key is released a touch early (the damper felt), and
        # tuned a few cents off its neighbour — a deterministic pseudo-
        # random walk of the tuning, so the render stays reproducible
        start = n.start + 0.018 * (1.0 - vel)
        # the damper sits the key a touch early; the duration keeps its
        # proportion from the actual (shifted) onset. A HELD note keeps
        # ringing past the lift (pedal): its tail is written too.
        dur = max(0.03, n.duration * 0.97)
        held = n.duration >= HELD_MIN_S
        tail = min(PEDAL_TAIL_MAX_S, PEDAL_TAIL_FACTOR * n.duration) if held else 0.0
        end = start + dur + tail
        cents = 4.0 * np.sin(n.start * 12.9898) + 2.5 * np.sin(n.start * 4.898)
        f = float(440.0 * 2.0 ** ((int(n.midi) + transpose - 69 + cents / 100.0) / 12.0))
        i0, i1 = max(0, int(start * sr)), min(length, int(end * sr))
        if i1 <= i0:
            continue
        y[i0:i1] += tone(f, i1 - i0, vel, lift_s=(dur if held else None))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y *= 0.85 / peak
    y = _room_tail(y, sr)
    # the gain is applied AFTER normalization (it must be audible) and the
    # result clipped: the device would hard-clip anything beyond ±1 anyway
    return np.clip(y * gain, -1.0, 1.0).astype(np.float32)


def _room_tail(y: np.ndarray, sr: int) -> np.ndarray:
    """A light room: one short decaying-noise IR, truncated to `len(y)`.

    A dry additive piano reads synthetic no matter how good the partials;
    ~120 ms of diffuse tail glues the notes together the way a room does.
    Cheap (one FFT convolution) and deterministic (fixed seed).
    """
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(20177)
    ir_n = int(0.12 * sr)
    ir = rng.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / (0.035 * sr))
    ir /= np.max(np.abs(ir))
    wet = fftconvolve(y, ir * 0.09)[: len(y)]
    return y + wet


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
            self._report_error(tr("err sounddevice missing"))
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
