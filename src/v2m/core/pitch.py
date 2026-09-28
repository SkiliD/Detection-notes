"""Pitch detection: pYIN (offline) and block-wise YIN (real-time)."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

FMIN = 65.0    # ~ C2
# C7, not C6: the old 1050 Hz ceiling sat EXACTLY on the sung C6 boundary —
# pYIN had no room above to lock a soprano's top notes and folded them an
# octave down (measured: a sung 84 read 72, a 79-88 line read 67-76 — the
# classic octave error, 2 corpus cases). One octave of headroom reads them
# and moves NOTHING below (chant.wav/demo.wav byte-identical, corpus 15/15
# unchanged). pYIN's octave-halving is still cleaned by clean_pitch's
# median filter; the piano guard (A0..C8) rejects anything stray.
FMAX = 2100.0  # ~ C7
# A "pitch" is only kept when the frame is actually PERIODIC: YIN's minimum
# of d'(tau) is the periodicity (0 = perfect period, 1 = noise). Measured on
# a low rumble (mic handling / desk thump, the "Sol#2 at the start of a live
# take" user report): d'min 0.73-1.12 — YIN still returns an Hz, the note
# segmenter reads a fake low note. Any sung frame sits far below (measured
# 0.000-0.05; the old fallback rejected above 0.6, so 0.35 keeps a wide
# margin under it while killing the rumble).
MIN_PITCH_QUALITY = 0.35
# A LOW RUMBLE (mic handling, desk thump — the fake low note at the start of
# a live take) carries no energy above ~400 Hz, while ANY voiced frame has
# harmonics and formants up there: the share of rms carried by the 500 Hz+
# band separates them by two orders of magnitude (measured: rumble 0.003,
# sung notes 0.41-0.61). The threshold sits at 0.03, not higher: a REAL
# take's quietest low notes dip to 0.034-0.049 of HF share (measured on
# chant.wav, 5 frames) — a floor of 0.05 erased one and shifted the tempo.
# The rumble, at 0.003, stays an order of magnitude under it.
HF_ENERGY_HZ = 500.0
HF_SHARE_MIN = 0.03
LOW_RUMBLE_MAX_HZ = 200.0   # the HF gate only judges LOW reads (G3 and below)


@dataclass
class PitchTrack:
    """Pitch curve: Hz + confidence per frame."""

    times: np.ndarray   # (n,) times in seconds (frame center)
    hz: np.ndarray      # (n,) frequency (NaN when unvoiced)
    confidence: np.ndarray  # (n,) voicing probability
    # True where |df0/dt| exceeds the transition threshold (vibrato peaks
    # aside, these are portamento ramps and note hand-overs). None until
    # clean_pitch computes it: pitch estimation must read STABLE plateaus
    # only, never the ramp that leads into them.
    transition: np.ndarray | None = None


def track_pitch_offline(y: np.ndarray, sr: int, frame_ms: float = 20.0) -> PitchTrack:
    """pYIN on a complete signal. Accurate, ~0.3x real duration.

    `confidence` carries pYIN's voicing PROBABILITY (the third return, once
    discarded), not the binary voiced flag: a breath's noise reads 0.01-0.1
    while a sung frame reads 0.95+ (measured on the voice harness), so
    `clean_pitch`'s `min_conf` mask actually has something to mask — with
    the flag (1.0 wherever f0 exists) nothing was ever filtered.
    """
    hop = max(64, int(sr * frame_ms / 1000))
    f0, _voiced_flag, vprob = librosa.pyin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=2048,
        hop_length=hop,
        fill_na=np.nan,
    )
    # The periodicity gate (MIN_PITCH_QUALITY): a rumble reads an Hz but is
    # not a sung frame. The gate kills a frame only when the WHOLE ±2-frame
    # window is low-confidence: pYIN's voicing confidence collapses on
    # SHORT notes (measured on 120-150 ms sixteenths: most frames mix the
    # note with the surrounding silence and read 0.01-0.15, one pure frame
    # reads 0.946), so a per-frame test would erase them; a rumble, though,
    # lasts for many frames and never recovers. One decent frame in five
    # keeps the whole neighbourhood voiced.
    gate = vprob >= MIN_PITCH_QUALITY
    from scipy.ndimage import maximum_filter1d

    # one good frame keeps its ±3-neighbourhood voiced: a 250 ms portamento
    # ramp carries mid-confidence frames (0.03-0.62, ADR 0005's territory)
    # and MUST stay continuous through the curve — the transition detector
    # reads the slope of the CLEANED curve, and a hole in the ramp breaks
    # the gradient into flat pieces below TRANSITION_ST_S (measured: the
    # legato arrival read its ramp when the ramp had holes). A rumble has
    # NO confident frame at all, so the gate still kills it entirely.
    any_good = maximum_filter1d(gate, size=7)
    f0 = np.where(any_good, f0, np.nan)
    # ...and the HF gate for LOW reads: a low rumble (mic handling) is
    # band-limited BELOW 500 Hz — no sung frame is (measured share of rms
    # above 500 Hz: rumble 0.003, sung notes 0.41-0.61, breath far above).
    # It applies ONLY where the read f0 is low (< LOW_RUMBLE_MAX_HZ): the
    # higher harmonics of a low sung note live above 500 Hz, but a pure
    # sine TEST tone above that limit (the level-test 220 Hz) must not be
    # gated — a pure sine IS indistinguishable from a rumble by HF share
    # alone, and the harness uses them as ground truth.
    n_fft = 2048
    if len(y) >= n_fft:
        from scipy.signal import butter, filtfilt

        bh, ah = butter(4, HF_ENERGY_HZ / (sr / 2), "high")
        hf = filtfilt(bh, ah, np.asarray(y, dtype=np.float64))
        n_frames = min(len(f0), (len(y) - n_fft) // hop + 1)
        for i in range(n_frames):
            if not gate[i]:
                continue
            fi = f0[i]
            if fi != fi or fi >= LOW_RUMBLE_MAX_HZ:
                continue
            i0 = i * hop
            tot = float(np.sqrt(np.mean(y[i0 : i0 + n_fft] ** 2)))
            hi = float(np.sqrt(np.mean(hf[i0 : i0 + n_fft] ** 2)))
            if hi < HF_SHARE_MIN * max(tot, 1e-9):
                f0[i] = np.nan
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    confidence = np.where(np.isnan(f0), 0.0, vprob)
    return PitchTrack(
        times=np.asarray(times, dtype=float),
        hz=np.asarray(f0, dtype=float),
        confidence=np.asarray(confidence, dtype=float),
    )


class YinBlock:
    """Incremental block-wise YIN estimator (real-time).

    Keeps a sliding buffer: pitch is estimated over the last `window`
    samples at every `hop` samples. A frame whose periodicity is worse
    than MIN_PITCH_QUALITY (a rumble, not a voice — same gate as offline,
    read from the same d' table YinPitch already computed) comes out NaN:
    the live segmenter then treats it as silence, so a mic-handling thump
    can no longer open a fake low note on the piano roll.
    """

    def __init__(self, sr: int, window: int = 1024, hop: int = 256):
        self.sr = sr
        self.window = window
        self.hop = hop
        self.buffer = np.zeros(0, dtype=np.float32)
        self.pos = 0     # index of the next frame inside `buffer`
        self.offset = 0  # absolute sample index of buffer[0]
        self._prev_periodic = False  # last frame passed the periodicity gate

    def push(self, block: np.ndarray) -> list[tuple[float, float]]:
        """Push a block, return [(time_s, hz | nan), ...] for the new hops.

        Times are absolute (measured from the first pushed sample): trimming
        the buffer must never restart the clock, or the segmenter would see
        a frozen time and never confirm a note.
        """
        self.buffer = np.concatenate([self.buffer, np.asarray(block, dtype=np.float32)])
        out: list[tuple[float, float]] = []
        while self.pos + self.window <= len(self.buffer):
            frame = self.buffer[self.pos : self.pos + self.window]
            t = (self.offset + self.pos + self.window / 2) / self.sr
            quality: dict = {}
            hz = yin_pitch(frame, self.sr, quality_out=quality)
            q = quality.get("quality", 1.0)
            periodic = hz == hz and q <= MIN_PITCH_QUALITY
            # a second live gate: at the short 512 window a low rumble
            # occasionally reads periodic by chance (measured 4 frames of 38
            # pass even a 0.5 threshold), but rarely twice in a row, while
            # a voiced note reads q=0 on essentially every frame (109 of
            # 110 consecutive pairs). A pitch frame is kept only when it or
            # the PREVIOUS hop (50 % overlapping) is periodic — an isolated
            # lucky frame cannot open a fake note on the roll.
            if hz == hz and not (periodic or self._prev_periodic):
                hz = float("nan")
            self._prev_periodic = periodic
            out.append((t, hz))
            self.pos += self.hop
        # release the samples we will never look at again (one window in hand)
        cut = max(0, self.pos - self.window)
        if cut:
            self.buffer = self.buffer[cut:]
            self.pos -= cut
            self.offset += cut
        return out


def yin_pitch(frame: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX,
              quality_out: dict | None = None) -> float:
    """Simplified YIN (mean difference + threshold) on one frame.

    `quality_out` (optional dict) receives the frame's periodicity quality
    (the minimum of d', 0 = perfectly periodic): the caller decides what
    to do with it — the offline tracker masks frames below
    `MIN_PITCH_QUALITY`, the live path keeps its own budget.
    """
    x = np.asarray(frame, dtype=np.float64)
    if np.max(np.abs(x)) < 1e-4:
        return float("nan")
    w = len(x)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(w - 1, int(sr / fmin))
    if tau_max <= tau_min:
        return float("nan")

    # cumulative mean normalized difference d'(tau)
    d = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        diff = x[: w - tau] - x[tau:]
        d[tau] = np.mean(diff * diff)
    cum = np.cumsum(d[1:]) / np.arange(1, tau_max + 1)
    d_prime = d[1:] / np.where(cum == 0, 1e-12, cum)

    # first minimum under the threshold, else the global minimum
    threshold = 0.15
    best = None
    for i in range(tau_min - 1, tau_max - 1):
        if d_prime[i] < threshold:
            # parabola around the local minimum
            if i + 1 < len(d_prime) and d_prime[i] > d_prime[i + 1]:
                continue
            best = i
            break
    if best is None:
        seg = d_prime[tau_min - 1 : tau_max]
        if not len(seg):
            return float("nan")
        best = tau_min - 1 + int(np.argmin(seg))
        if d_prime[best] > 0.6:
            return float("nan")
    if quality_out is not None:
        quality_out["quality"] = float(d_prime[best])
    tau = best + 1
    # parabolic interpolation around the minimum (sub-sample precision)
    # points: d_prime[best-1] (tau-1), d_prime[best] (tau), d_prime[best+1] (tau+1)
    if 0 < best < len(d_prime) - 1:
        a, b, c = d_prime[best - 1], d_prime[best], d_prime[best + 1]
        denom = 2.0 * (a + c - 2.0 * b)
        if abs(denom) > 1e-12:
            tau = tau + (a - c) / denom
    return float(sr / tau)
