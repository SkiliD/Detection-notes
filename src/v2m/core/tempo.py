"""Tempo detection (BPM) and beat positions."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .meter import dotted_pulse_tempo

BPM_RANGE = (60.0, 180.0)  # beat-track fallback range (a lone voice halves/doubles)


@dataclass
class TempoResult:
    bpm: float
    beats: np.ndarray  # times (s) of the estimated beats


def detect_tempo(
    y: np.ndarray,
    sr: int,
    bpm_range: tuple[float, float] = BPM_RANGE,
) -> TempoResult:
    """Tempo via onset strength + beat tracking, with x2 / /2 fallback.

    On a lone voice (little percussive content) the estimator may be off by
    a factor of 2: constrain it to the requested range.
    """
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, start_bpm=120.0, tightness=100.0
    )
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # fallback: if the BPM is out of range, try x2 and /2
    lo, hi = bpm_range
    if not (lo <= bpm <= hi):
        candidates = [bpm, bpm * 2, bpm / 2, bpm * 2 / 3, bpm * 3 / 2]
        valid = [b for b in candidates if lo <= b <= hi]
        if valid:
            bpm = valid[0]
        else:
            bpm = float(np.clip(bpm, lo, hi))
    return TempoResult(bpm=round(bpm, 1), beats=np.asarray(beat_times, dtype=float))


def estimate_bpm_live(notes, min_notes: int = 4) -> float | None:
    """Approximate BPM during live recording.

    Median of onset intervals, folded into 60-180 by powers of 2. Good
    enough to label figures on screen; the final analysis refines it with
    refine_bpm_with_onsets.

    One reading is fixed BEFORE the fold: when the strengths alternate
    strong/weak with period 2 (reads_dotted_pulse), the median gap is a
    DOTTED beat and the true quarter sits at 2/3 of it — see
    meter.dotted_pulse_tempo, the one correction the live estimate and the
    final analysis share.
    """
    if len(notes) < min_notes:
        return None
    starts = [n.start for n in notes]
    gaps = [b - a for a, b in zip(starts, starts[1:]) if b - a > 0.1]
    if not gaps:
        return None
    bpm = 60.0 / float(np.median(gaps))
    bpm = dotted_pulse_tempo(notes, bpm)
    while bpm < 60.0:
        bpm *= 2.0
    while bpm > 180.0:
        bpm /= 2.0
    return round(bpm, 1)


def refine_bpm_with_onsets(
    onsets: np.ndarray,
    bpm: float,
    span_rel: float = 0.08,
    step: float = 0.0005,
    min_hits: int = 3,
) -> float:
    """Refine a BPM by maximizing onset phase coherence.

    `librosa.beat.beat_track` may return a rounded tempo (e.g. 117.5 for a
    true 120): note boundaries then fall off any grid (0.5 s = 1.378 beats
    at 117.5) and figure rendering fragments. Scan bpm+/-span_rel (relative);
    for each candidate, the k*period grid is least-squares aligned to the
    onsets (+/-15% of period window) and the best score wins:
    hits - 0.5*quadratic error. No proximity penalty: within a +/-8% span,
    the half/double tempo equivalents are out of reach.

    The scan is VECTORIZED by batches: candidates are stacked so the inner
    loop runs in NumPy over a (batch, n_onsets) block instead of one Python
    iteration per BPM candidate (~640 candidates on a refine — the heaviest
    project function on a long take, AGENTS.md's profile note). The tie
    rules match the sequential version exactly: first candidate wins an
    equal score (min on the first best), and candidates with too few hits
    are excluded before scoring, as before.
    """
    if bpm <= 0 or len(onsets) < min_hits:
        return float(bpm)
    onsets = np.asarray(onsets, dtype=float)
    lo, hi = bpm * (1.0 - span_rel), bpm * (1.0 + span_rel)
    cands = np.arange(lo, hi + step / 2, step)
    if len(cands) == 0:
        return float(bpm)

    BATCH = 64  # 64 x n_onsets floats per block: cache-friendly, no big alloc
    best_bpm, best_score = float(bpm), -np.inf
    for i in range(0, len(cands), BATCH):
        block = cands[i : i + BATCH]
        periods = 60.0 / block[:, None]                     # (B, 1)
        k = np.round(onsets[None, :] / periods)             # (B, N)
        resid = onsets[None, :] - k * periods               # (B, N)
        tol = np.maximum(0.045, 0.15 * periods)             # (B, 1)
        good = np.abs(resid) <= tol
        n_hits = good.sum(axis=1)                           # (B,)
        ok = n_hits >= min_hits
        if not ok.any():
            continue
        sq = np.where(good, resid * resid, 0.0)
        rms = np.sqrt(sq.sum(axis=1) / np.maximum(n_hits, 1))
        scores = n_hits - 0.5 * rms
        scores = np.where(ok, scores, -np.inf)
        j = int(np.argmax(scores))  # argmax keeps the FIRST best of the block
        if scores[j] > best_score:
            best_score, best_bpm = float(scores[j]), float(block[j])

    # polish: linear regression time = k*period + offset on the in-phase
    # onsets -> exact period (e.g. 120.75 -> 120.00)
    p = 60.0 / best_bpm
    k = np.round(onsets / p)
    tol = max(0.045, 0.15 * p)
    inl = np.abs(onsets - k * p) <= tol
    if inl.sum() >= min_hits:
        kk = k[inl]
        tt = onsets[inl]
        slope = float(np.polyfit(kk, tt, 1)[0])
        if slope > 0.05:
            refined = 60.0 / slope
            # guardrail: stay within the relative scan span
            if bpm * (1 - span_rel - 0.02) <= refined <= bpm * (1 + span_rel + 0.02):
                best_bpm = refined
    return best_bpm

