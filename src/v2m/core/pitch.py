"""Pitch detection: pYIN (offline) and block-wise YIN (real-time)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import librosa
import numba
import numpy as np
from scipy import stats

from .audio_io import GATE_RMS

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
# a low rumble (mic handling / desk thump): d'min 0.73-1.12 — YIN still
# returns an Hz, the note segmenter reads a fake low note. Any sung frame
# sits far below (measured 0.000-0.05; the old fallback rejected above 0.6,
# so 0.35 keeps a wide margin under it while killing the rumble).
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

# The swallow-rescue sweep (the block-wise second reading). pYIN's 2048
# window (~93 ms) collapses through SHORT voiced bursts — a voiced
# consonant's note reads vprob 0.01 where the live engine's 23 ms window
# still sees the note (measured on the real takes: orphan attacks whose
# cleaned-curve gap holds NO voiced frame at all). Where the long-window
# curve is UNVOICED but the signal is above the absolute silence gate,
# the span is re-read at the live geometry (window 512, 23 ms); sustained
# PERIODICITY there grafts voiced frames into the curve. Voice presence
# is answered by a curve (the short window's own reading), never by an
# attack rule — a silent/unvoiced burst has no sustained period and stays
# unvoiced (the consonant corpus pins that). The level gate is ABSOLUTE
# (the same bar the live path draws), not a relative floor (ADR 0003):
# a quiet voice is re-read like a loud one.
RESCUE_WINDOW = 512        # samples: the live engine's short window
RESCUE_MIN_SPAN_S = 0.05   # a shorter unvoiced span is not a swallowed note
RESCUE_COVER = 0.6         # share of the span the short window must voice
RESCUE_MAX_SPAN_S = 4.0    # longer spans are silence/breath passages: skip
# A swallowed note is a PLATEAU at the short window; a legato ramp is a
# MOVING reading (measured: the ramp's frames trace neighbour -> neighbour
# and grafting their median sprouted an intruder note between the two).
# The plateau test is ROBUST: a minority of octave-folded frames sits
# inside a real burst's edges (measured 2 of 8 at 73 Hz against a 220 Hz
# burst), so the share of readings within the band decides — a ramp's
# readings spread along the slide and fail it, a bimodal pair of notes
# fails it, one plateau with folded edges passes.
RESCUE_SPREAD_ST = 1.0     # "one plateau" band around the median (st)
RESCUE_INLIER_SHARE = 0.7  # share of readings that must sit in the band
RESCUE_BRIDGE_ST = 1.0     # "on a neighbour's plateau" tolerance (st)

# Where the rescue's time went, per reading: the seconds grafted by the
# LAST `track_pitch_offline` call and its measured cost. One reader (the
# profile script's stage table via `pitch_stage_cost`), one writer (the
# read itself, reset each time) — never a global accumulation.
_last_rescue: dict = {"seconds": 0.0, "cost_s": 0.0}


def pitch_stage_cost() -> dict:
    """The last reading's rescue facts: seconds grafted + wall-clock cost.

    `profile_analysis`'s stage table reads this to show the rescue as its
    own row — the sweep runs INSIDE the pitch stage, and a sub-stage cost
    must not be re-implemented (or re-timed) by a display adapter.
    """
    return dict(_last_rescue)


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


def _rescue_swallowed_spans(y: np.ndarray, sr: int, hop: int, f0: np.ndarray,
                            vprob: np.ndarray) -> float:
    """Re-read unvoiced spans at the live geometry, IN PLACE on f0/vprob.

    pYIN's 2048 window reads vprob ~0.01 through a voiced burst whose note
    the live engine's 23 ms window sees — the measured deficit between the
    real takes' live path and the final reading. Each unvoiced span long
    enough to be a note (`RESCUE_MIN_SPAN_S`) and short enough not to be a
    rest passage (`RESCUE_MAX_SPAN_S`), where the signal stays above the
    ABSOLUTE silence gate, is re-read with the short window: where the
    short-window frames sustain their periodicity over `RESCUE_COVER` of
    the span, the curve inherits them — voiced, from the short reading.
    A breath (noise, no sustained period) stays unvoiced; so does a true
    silence. Nothing is re-read where the long window already voiced the
    curve, and `clean_pitch`'s median still runs on the merged curve.

    Returns the re-read span total in seconds (the rescue's measured cost
    driver — where the take's time went, the profile table shows it).
    """
    rescued_s = 0.0
    # 1) the ABSENT spans of the long-window curve. Only NaN frames are
    # candidates: a voiced-but-low-confidence frame still carries pYIN's
    # reading (clean_pitch bridges it) — overwriting it re-dates note
    # boundaries and moved a real take's BPM by 5 % (measured). The
    # swallow shape is a note the long window read NOWHERE: f0 is NaN
    # across the whole span.
    unvoiced = np.isnan(f0)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(unvoiced):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(unvoiced)))
    # 2) re-read each span at the live geometry (a sliding short window)
    for lo, hi in spans:
        # the ABSOLUTE silence gate (audio_io's, the live path's own bar):
        # below it nothing is re-read — a true silence stays a silence. The
        # graft covers only the span's AUDIBLE frames: a NaN span may run
        # into the surrounding true silence (a lone swallowed burst between
        # silences), and painting pitch over silence would be a fake note.
        frame_rms = np.zeros(hi - lo)
        for k, i in enumerate(range(lo, hi)):
            a = i * hop
            b = min(len(y), a + hop)
            if b > a:
                frame_rms[k] = float(np.sqrt(np.mean(y[a:b].astype(np.float64) ** 2)))
        audible = np.where(frame_rms >= GATE_RMS)[0]
        if len(audible) < 2:
            continue
        alo, ahi = lo + int(audible[0]), lo + int(audible[-1]) + 1
        span_s = (ahi - alo) * hop / sr
        if span_s < RESCUE_MIN_SPAN_S or span_s > RESCUE_MAX_SPAN_S:
            continue
        seg = y[alo * hop : min(len(y), ahi * hop)]
        rms = float(np.mean(frame_rms[audible[0] : audible[-1] + 1]))
        if rms < GATE_RMS:
            continue
        # coverage counts the sweep frames that CARRY signal: the share is
        # of the audible frames, never of the span's silence
        covered = 0
        eligible = 0
        w = RESCUE_WINDOW
        pos = 0
        read_hz: list[float] = []
        while pos + w <= len(seg):
            frame = seg[pos:pos + w]
            frms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            if frms >= GATE_RMS:
                eligible += 1
                quality: dict = {}
                hz = yin_pitch(frame, sr, quality_out=quality)
                q = quality.get("quality", 1.0)
                if hz == hz and q <= MIN_PITCH_QUALITY:
                    covered += 1
                    read_hz.append(hz)
            pos += w // 2
        if eligible < 2 or covered < RESCUE_COVER * eligible or not read_hz:
            continue
        # 3) the span must read as ONE PLATEAU, not a glide: most readings
        # sit in a one-semitone band around their median, and a ramp that
        # relays the two neighbouring plateaus is a legato transition, not
        # a swallowed note (grafting it sprouted an intruder between the
        # notes). Subharmonic folds (a reading 2x/3x/4x under another one)
        # are estimator artifacts at a burst's edges, dropped before the
        # vote — a ramp's readings spread along the slide and fail it.
        kept = [h for h in read_hz if not any(
            1.97 <= g / h <= 4.06 for g in read_hz if g > h
        )]
        if len(kept) < 3:
            continue
        read_midi = np.array([69.0 + 12.0 * np.log2(h / 440.0) for h in kept])
        median_m = float(np.median(read_midi))
        inlier = np.abs(read_midi - median_m) <= RESCUE_SPREAD_ST
        if float(inlier.mean()) < RESCUE_INLIER_SHARE:
            continue
        prev_hz = _neighbour_plateau(f0, vprob, lo, -1)
        next_hz = _neighbour_plateau(f0, vprob, hi, +1)
        if prev_hz is not None and next_hz is not None:
            inlier_m = read_midi[inlier]
            first_m, last_m = float(inlier_m[0]), float(inlier_m[-1])
            prev_m = 69.0 + 12.0 * np.log2(prev_hz / 440.0)
            next_m = 69.0 + 12.0 * np.log2(next_hz / 440.0)
            starts_on_prev = abs(first_m - prev_m) <= RESCUE_BRIDGE_ST
            ends_on_next = abs(last_m - next_m) <= RESCUE_BRIDGE_ST
            if starts_on_prev and ends_on_next and abs(prev_m - next_m) > RESCUE_SPREAD_ST:
                continue
        # 4) the span is a swallowed voiced note: graft the short reading in
        median_hz = 440.0 * 2.0 ** ((median_m - 69.0) / 12.0)
        for i in range(alo, ahi):
            f0[i] = median_hz
            vprob[i] = 1.0
        rescued_s += span_s
    return rescued_s


def _neighbour_plateau(f0: np.ndarray, vprob: np.ndarray, edge: int,
                       step: int) -> float | None:
    """The pitch plateau bordering a span on one side (median of ~100 ms).

    `edge` is the span's first (step=-1) or last (step=+1) frame index;
    None when that side has no voiced plateau (take start/end, or a
    neighbour that is itself unvoiced).
    """
    n = len(f0)
    window = 5  # ~100 ms of 20 ms frames
    if step < 0:
        lo, hi = max(0, edge - window), edge
    else:
        lo, hi = edge, min(n, edge + window)
    idx = [i for i in range(lo, hi)
           if f0[i] == f0[i] and vprob[i] >= MIN_PITCH_QUALITY]
    if not idx:
        return None
    return float(np.median([f0[i] for i in idx]))


_PYIN_FRAME_BATCH = 384


@numba.njit(cache=True)
def _pyin_observation_kernel(
    batch_yin,
    batch_shifts,
    sr,
    min_period,
    fmin,
    n_pitch_bins,
    bins_per_semitone,
    thresholds,
    beta_probs,
):
    """Construct librosa-compatible pYIN observations for a bounded frame batch."""
    n_periods, n_frames = batch_yin.shape
    observation = np.zeros((2 * n_pitch_bins, n_frames), dtype=np.float64)
    voiced = np.zeros(n_frames, dtype=np.float64)
    boltzmann_factor = 1.0 - np.exp(-2.0)

    for frame in range(n_frames):
        trough_rows = np.empty(n_periods, dtype=np.int64)
        trough_count = 0
        for row in range(n_periods):
            if row == 0:
                is_trough = batch_yin[row, frame] < batch_yin[row + 1, frame]
            elif row == n_periods - 1:
                is_trough = batch_yin[row, frame] < batch_yin[row - 1, frame]
            else:
                is_trough = (
                    batch_yin[row, frame] < batch_yin[row - 1, frame]
                    and batch_yin[row, frame] <= batch_yin[row + 1, frame]
                )
            if is_trough:
                trough_rows[trough_count] = row
                trough_count += 1

        if trough_count == 0:
            for pitch_bin in range(n_pitch_bins, 2 * n_pitch_bins):
                observation[pitch_bin, frame] = 1.0 / n_pitch_bins
            continue

        trough_heights = np.empty(trough_count, dtype=np.float64)
        global_min = 0
        for i in range(trough_count):
            height = batch_yin[trough_rows[i], frame]
            trough_heights[i] = height
            if height < trough_heights[global_min]:
                global_min = i

        candidate_probs = np.zeros(trough_count, dtype=np.float64)
        for threshold_index in range(len(beta_probs)):
            threshold = thresholds[threshold_index + 1]
            under_threshold = 0
            for candidate in range(trough_count):
                if trough_heights[candidate] < threshold:
                    under_threshold += 1
            if under_threshold == 0:
                continue

            normalizer = 1.0 - np.exp(-2.0 * under_threshold)
            rank = 0
            for candidate in range(trough_count):
                if trough_heights[candidate] < threshold:
                    prior = boltzmann_factor * np.exp(-2.0 * rank) / normalizer
                    candidate_probs[candidate] += prior * beta_probs[threshold_index]
                    rank += 1

        thresholds_without_trough = 0
        for threshold_index in range(len(beta_probs)):
            if trough_heights[global_min] >= thresholds[threshold_index + 1]:
                thresholds_without_trough += 1
        fallback_mass = 0.0
        for threshold_index in range(thresholds_without_trough):
            fallback_mass += beta_probs[threshold_index]
        candidate_probs[global_min] += 0.01 * fallback_mass

        for candidate in range(trough_count):
            probability = candidate_probs[candidate]
            if probability <= 0.0:
                continue
            period_row = trough_rows[candidate]
            period = min_period + period_row + batch_shifts[period_row, frame]
            frequency = sr / period
            pitch_bin = int(
                np.floor(
                    12.0 * bins_per_semitone * np.log2(frequency / fmin) + 0.5
                )
            )
            pitch_bin = min(n_pitch_bins, max(0, pitch_bin))
            observation[pitch_bin, frame] = probability

        total = 0.0
        for pitch_bin in range(n_pitch_bins):
            total += observation[pitch_bin, frame]
        total = min(1.0, total)
        voiced[frame] = total
        unvoiced_probability = (1.0 - total) / n_pitch_bins
        for pitch_bin in range(n_pitch_bins, 2 * n_pitch_bins):
            observation[pitch_bin, frame] = unvoiced_probability

    return observation, voiced



def _pyin_batched(
    y: np.ndarray,
    sr: int,
    hop: int,
    *,
    batch_size: int = _PYIN_FRAME_BATCH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pYIN's candidate and Viterbi stages, with candidate frames built in batches.

    librosa.pyin calls its threshold-probability helper once per frame via
    np.vectorize; that Python dispatch dominated finalization on takes of
    ordinary length. This is the same pYIN candidate distribution in bounded
    frame batches, followed by the same transition matrix and Viterbi decoder.
    `tests/test_pitch.py` compares all three returned tracks to librosa.pyin.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    frame_length = 2048
    min_period = int(np.floor(sr / FMAX))
    max_period = min(int(np.ceil(sr / FMIN)), frame_length - 1)
    padded = np.pad(y, (frame_length // 2, frame_length // 2), mode="constant")
    frames = librosa.util.frame(
        padded, frame_length=frame_length, hop_length=hop
    )
    librosa.util.valid_audio(y)
    from librosa import sequence
    from librosa.core.pitch import (
        _cumulative_mean_normalized_difference,
        _parabolic_interpolation,
    )

    yin = _cumulative_mean_normalized_difference(frames, min_period, max_period)
    shifts = _parabolic_interpolation(yin)
    n_thresholds = 100
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    beta_probs = np.diff(stats.beta.cdf(thresholds, 2, 18))
    bins_per_semitone = 10
    n_pitch_bins = int(
        np.floor(12 * bins_per_semitone * np.log2(FMAX / FMIN))
    ) + 1
    observation_probs = np.zeros(
        (2 * n_pitch_bins, yin.shape[1]), dtype=np.float64
    )
    voiced_prob = np.zeros(yin.shape[1], dtype=np.float64)

    for first in range(0, yin.shape[1], batch_size):
        last = min(yin.shape[1], first + batch_size)
        batch_yin = yin[:, first:last]
        batch_shifts = shifts[:, first:last]
        batch_observations, batch_voiced = _pyin_observation_kernel(
            batch_yin,
            batch_shifts,
            sr,
            min_period,
            FMIN,
            n_pitch_bins,
            bins_per_semitone,
            thresholds,
            beta_probs,
        )
        observation_probs[:, first:last] = batch_observations
        voiced_prob[first:last] = batch_voiced

    max_semitones_per_frame = round(35.92 * 12 * hop / sr)
    transition_width = max_semitones_per_frame * bins_per_semitone + 1
    transition = sequence.transition_local(
        n_pitch_bins, transition_width, window="triangle", wrap=False
    )
    transition = np.kron(sequence.transition_loop(2, 0.99), transition)
    initial = np.ones(2 * n_pitch_bins) / (2 * n_pitch_bins)
    states = sequence.viterbi(
        observation_probs[np.newaxis, :, :],
        transition,
        p_init=initial,
        transition_min_prob=1e-4,
    )[0]
    frequencies = FMIN * 2 ** (
        np.arange(n_pitch_bins) / (12 * bins_per_semitone)
    )
    f0 = frequencies[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins
    f0[~voiced_flag] = np.nan
    return f0, voiced_flag, voiced_prob


def track_pitch_offline(y: np.ndarray, sr: int, frame_ms: float = 20.0) -> PitchTrack:
    """Run final pYIN, restore swallowed voiced spans, and reject low rumble."""
    hop = max(64, int(sr * frame_ms / 1000))
    f0, _voiced_flag, vprob = _pyin_batched(y, sr, hop)
    started = time.perf_counter()
    grafted_s = _rescue_swallowed_spans(y, sr, hop, f0, vprob)
    _last_rescue["seconds"] = float(grafted_s)
    _last_rescue["cost_s"] = time.perf_counter() - started

    gate = vprob >= MIN_PITCH_QUALITY
    from scipy.ndimage import maximum_filter1d

    # A confident frame keeps its seven-frame neighbourhood: pYIN confidence
    # collapses on short notes and through legato ramps, while rumble has no
    # confident frame to propagate.
    any_good = maximum_filter1d(gate, size=7)
    f0 = np.where(any_good, f0, np.nan)

    # Low rumble is band-limited, unlike voiced notes. Apply the HF check only
    # to low reads; a pure-sine test tone above the boundary is valid evidence.
    n_fft = 2048
    if len(y) >= n_fft:
        from scipy.signal import butter, filtfilt

        bh, ah = butter(4, HF_ENERGY_HZ / (sr / 2), "high")
        hf = filtfilt(bh, ah, np.asarray(y, dtype=np.float64))
        n_frames = min(len(f0), (len(y) - n_fft) // hop + 1)
        for i in range(n_frames):
            if not gate[i]:
                continue
            frequency = f0[i]
            if not np.isfinite(frequency) or frequency >= LOW_RUMBLE_MAX_HZ:
                continue
            start = i * hop
            total_rms = float(np.sqrt(np.mean(y[start : start + n_fft] ** 2)))
            high_rms = float(np.sqrt(np.mean(hf[start : start + n_fft] ** 2)))
            if high_rms < HF_SHARE_MIN * max(total_rms, 1e-9):
                f0[i] = np.nan

    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    confidence = np.where(np.isnan(f0), 0.0, vprob)
    return PitchTrack(
        times=np.asarray(times, dtype=float),
        hz=np.asarray(f0, dtype=float),
        confidence=np.asarray(confidence, dtype=float),
    )


class YinBlock:
    """Incremental YIN estimator with a clock that survives buffer trims."""

    def __init__(self, sr: int, window: int = 1024, hop: int = 256):
        self.sr = sr
        self.window = window
        self.hop = hop
        self.buffer = np.zeros(0, dtype=np.float32)
        self.pos = 0
        self.offset = 0
        self._prev_periodic = False

    def push(self, block: np.ndarray) -> list[tuple[float, float]]:
        """Push a sample block and return newly available absolute-time pitches."""
        self.buffer = np.concatenate([self.buffer, np.asarray(block, dtype=np.float32)])
        out: list[tuple[float, float]] = []
        while self.pos + self.window <= len(self.buffer):
            frame = self.buffer[self.pos : self.pos + self.window]
            t = (self.offset + self.pos + self.window / 2) / self.sr
            quality: dict[str, float] = {}
            hz = yin_pitch(frame, self.sr, quality_out=quality)
            periodic = np.isfinite(hz) and quality.get("quality", 1.0) <= MIN_PITCH_QUALITY
            if np.isfinite(hz) and not (periodic or self._prev_periodic):
                hz = float("nan")
            self._prev_periodic = bool(periodic)
            out.append((t, hz))
            self.pos += self.hop

        # The overlap remains in the buffer, but its sample origin moves too.
        cut = max(0, self.pos - self.window)
        if cut:
            self.buffer = self.buffer[cut:]
            self.pos -= cut
            self.offset += cut
        return out


def yin_pitch(
    frame: np.ndarray,
    sr: int,
    fmin: float = FMIN,
    fmax: float = FMAX,
    quality_out: dict | None = None,
) -> float:
    """Estimate one frame's pitch; optionally return YIN's periodicity score."""
    x = np.asarray(frame, dtype=np.float64)
    if np.max(np.abs(x)) < 1e-4:
        return float("nan")
    width = len(x)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(width - 1, int(sr / fmin))
    if tau_max <= tau_min:
        return float("nan")

    # d(tau) = A + B - 2R. Prefix sums produce A/B for all lags, and one
    # FFT autocorrelation supplies R without a Python dispatch per period.
    power = x * x
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    total = cumulative[width]
    fft_size = 1
    while fft_size < 2 * width:
        fft_size <<= 1
    spectrum = np.fft.rfft(x, fft_size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)[:width]
    taus = np.arange(1, tau_max + 1)
    difference = np.empty(tau_max + 1)
    difference[0] = 0.0
    difference[1:] = (
        cumulative[width - taus]
        + (total - cumulative[taus])
        - 2.0 * correlation[taus]
    ) / (width - taus)
    running_mean = np.cumsum(difference[1:]) / taus
    d_prime = difference[1:] / np.where(running_mean == 0.0, 1e-12, running_mean)

    best = None
    for index in range(tau_min - 1, tau_max - 1):
        if d_prime[index] < 0.15:
            if index + 1 < len(d_prime) and d_prime[index] > d_prime[index + 1]:
                continue
            best = index
            break
    if best is None:
        region = d_prime[tau_min - 1 : tau_max]
        if not len(region):
            return float("nan")
        best = tau_min - 1 + int(np.argmin(region))
        if d_prime[best] > 0.6:
            return float("nan")
    if quality_out is not None:
        quality_out["quality"] = float(d_prime[best])

    tau = best + 1
    if 0 < best < len(d_prime) - 1:
        left, middle, right = d_prime[best - 1], d_prime[best], d_prime[best + 1]
        denominator = 2.0 * (left + right - 2.0 * middle)
        if abs(denominator) > 1e-12:
            tau += (left - right) / denominator
    return float(sr / tau)
