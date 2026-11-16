"""Intermediate vocal evidence, kept separate from musical interpretation.

One object carries what a signal said about the voice, before any musical
reading of it:

- the pitch curve, **cleaned** (`clean_pitch`: octave-jump median, unvoiced
  frames masked, the derivative transition mask computed) — the curve a
  segmenter can actually be run on, and the only curve the analysis keeps;
- the **attacks** (`onsets.detect_attacks`, or None when the caller asked for
  geometry only);
- the **frame clock** (`sample_rate`, `frame_dt`): how long one pitch frame
  is, read from the curve once instead of re-derived by every consumer;
- the **accent envelope** (`accent_envelope`, one smoothed RMS value per
  `accent_hop` samples) — the level evidence `compute_onset_strengths`
  quantizes per note. It is deliberately NOT merged with the offline
  detector's 20 ms envelope: accents and attack picking are two different
  measures, kept separate from attack picking;
- the take's measured **tuning offset**, filled by `notes.interpret` (the
  only side effect, documented and idempotent).

`pipeline.analyze_signal` builds one observation per signal with
`from_signal` and hands it to `notes.interpret`; the object travels on
`AnalysisResult.observation`, so the same evidence can be reinterpreted —
other thresholds, other figures — without decoding the audio again or
re-running pYIN (`scripts/reinterpret.py`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .music import PIANO_MAX_MIDI, PIANO_MIN_MIDI, hz_to_midi
from .onsets import ACCENT_HOP, Attacks, accent_envelope, detect_attacks
from .pitch import PitchTrack, track_pitch_offline

DEFAULT_MIN_CONF = 0.6
DEFAULT_GAP_S = 0.050  # gaps < 50 ms are interpolated
# f0 stabilization: a frame whose |df0/dt| exceeds this (semitones/second)
# is a TRANSITION (portamento ramp, note hand-over), not a plateau. The
# threshold sits ABOVE the vibrato peaks and BELOW the true ramps, both
# measured on the truth harness: vibrato ±0.4 st peaks at 10 st/s, ±0.6 st
# at 25 st/s, while a 90 ms legato glide reaches 17-85 st/s. A higher
# value would keep ramp frames in the plateaus (a real take's slow drift
# then reads as pitch); a lower one would shave the vibrato's turning
# points into phantom micro-notes.
TRANSITION_ST_S = 40.0
# The .npz layout version: a loader that meets a file written by a DIFFERENT
# layout refuses it (stale cache) instead of misreading its arrays.
# Version 4 requires independent audible islands in the swallow rescue:
# earlier cached curves may bridge silence, so they cannot serve a fresh
# reading (the cache saves time, never changes the engine's decisions).
FORMAT_VERSION = 5


@dataclass
class VoiceObservation:
    """Pitch and attack evidence from one signal, cleaned and ready.

    This is the input of the interpretation, not a record kept for show:
    `notes.interpret` reads the cleaned curve and the attacks from here, and
    the pipeline keeps the object on the `AnalysisResult`.
    """

    pitch: PitchTrack
    attacks: Attacks | None
    duration_s: float
    sample_rate: int = 0
    frame_dt: float = 0.01  # one pitch frame, in seconds
    accent_envelope: np.ndarray | None = None  # smoothed RMS per accent hop
    accent_hop: int = ACCENT_HOP  # samples per accent-envelope frame
    tuning_offset_st: float = 0.0  # measured by interpret (never by hand)
    # Provenance of the take this evidence came from, filled by `save` when
    # the sibling audio is available: what makes a CACHED observation valid
    # for an audio file (same size, same mtime) rather than a guess. Zero
    # means "unknown" — the consumer then falls back to weaker checks.
    source_size: int = 0
    source_mtime: float = 0.0

    @classmethod
    def from_signal(
        cls,
        y: np.ndarray,
        sr: int,
        min_conf: float = DEFAULT_MIN_CONF,
        use_attacks: bool = True,
    ) -> "VoiceObservation":
        """Listen once: pYIN, the evidence gates, the cleanup and the clock.

        `min_conf` is passed to `clean_pitch`, which accepts it for API
        stability and does NOT apply it as a hard mask (ADR 0005) — the
        parameter stays so the pipeline keeps one policy object.
        """
        track = track_pitch_offline(y, sr)
        clean_pitch(track, min_conf=min_conf)
        attacks = detect_attacks(y, sr) if use_attacks else None
        times = track.times
        frame_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
        return cls(
            pitch=track,
            attacks=attacks,
            duration_s=len(y) / sr,
            sample_rate=sr,
            frame_dt=frame_dt,
            accent_envelope=accent_envelope(y),
        )

    # ------------------------------------------------------------------ #
    # Persistence: the .npz file carries the EVIDENCE, not a copy
    # ------------------------------------------------------------------ #
    # The goal is to reinterpret a take (other thresholds, other figures)
    # without ever re-decoding the audio or re-running pYIN: `save` and
    # `load` are therefore the exact mirror of `from_signal`, field for
    # field. An ABSENT table (transition mask before `clean_pitch`, attacks
    # not requested, empty envelope) is stored as a zero-size array and
    # REVIVED as None on load, so `attacks is None` keeps its meaning:
    # "geometry alone" and "no attack detected" are not the same thing
    # (see `interpret`).

    def save(self, path: str | Path) -> None:
        """Write the evidence into a `.npz` archive (one observation per file)."""
        transition = self.pitch.transition
        arrays: dict[str, np.ndarray] = {
            "pitch_times": np.asarray(self.pitch.times),
            "pitch_hz": np.asarray(self.pitch.hz),
            "pitch_confidence": np.asarray(self.pitch.confidence),
            "pitch_transition": (
                np.asarray(transition) if transition is not None else np.zeros(0, dtype=bool)
            ),
            "attacks_times": (
                np.asarray(self.attacks.times) if self.attacks is not None else np.zeros(0)
            ),
            "attacks_strengths": (
                np.asarray(self.attacks.strengths) if self.attacks is not None else np.zeros(0)
            ),
            "accent_envelope": (
                np.zeros(0) if self.accent_envelope is None else np.asarray(self.accent_envelope)
            ),
        }
        scalars = {
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "frame_dt": self.frame_dt,
            "accent_hop": self.accent_hop,
            "tuning_offset_st": self.tuning_offset_st,
            "has_attacks": 1 if self.attacks is not None else 0,
            "format_version": FORMAT_VERSION,
            "source_size": self.source_size,
            "source_mtime": self.source_mtime,
            "written_at": time.time(),
        }
        payload = dict(arrays)
        payload.update({k: np.asarray(v) for k, v in scalars.items()})
        # numpy's stub types `**kwds` as ArrayLike only, so an untyped
        # dict of arrays makes pyright fear a key named `allow_pickle`:
        # cast the payload to what it is, one named array per key.
        np.savez(str(path), **cast(dict[str, Any], payload))

    @classmethod
    def load(cls, path: str | Path) -> "VoiceObservation":
        """Read back an observation written by `save` (lossless round-trip).

        The provenance scalars (`format_version`, the take's size/mtime) are
        read tolerantly: an older version's `.npz` still loads — the CACHE
        decides its freshness, not the loader (the reinterpret script must
        stay able to read evidence one version old).
        """
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)

            def scalar(key: str, default: int = 0) -> int:
                return int(data[key]) if key in files else default

            def fscalar(key: str, default: float = 0.0) -> float:
                return float(data[key]) if key in files else default

            transition = data["pitch_transition"]
            track = PitchTrack(
                times=data["pitch_times"],
                hz=data["pitch_hz"],
                confidence=data["pitch_confidence"],
                transition=transition if transition.size else None,
            )
            attacks: Attacks | None = None
            if int(data["has_attacks"]):
                attacks = Attacks(
                    times=data["attacks_times"],
                    strengths=data["attacks_strengths"],
                )
            env = data["accent_envelope"]
            return cls(
                pitch=track,
                attacks=attacks,
                duration_s=float(data["duration_s"]),
                sample_rate=int(data["sample_rate"]),
                frame_dt=float(data["frame_dt"]),
                accent_envelope=env if env.size else None,
                accent_hop=int(data["accent_hop"]),
                tuning_offset_st=float(data["tuning_offset_st"]),
                source_size=scalar("source_size"),
                source_mtime=fscalar("source_mtime"),
            )

    def matches_take(self, audio: str | Path, sr: int, duration_s: float) -> bool:
        """Can this evidence replace pYIN for THIS audio file?

        Three guards, weakest first: the rate must be the one asked for (a
        resampled observation would shift every frame time), the duration
        must match the decoded take (a lengthened take is no longer the
        same), and the take must not have been REWRITTEN since (size +
        mtime recorded by `save` — an mtime alone changes too easily, a
        size alone misses a same-length re-recording). Without recorded
        provenance (`save` without a neighbouring audio file), duration and
        rate suffice: the user wrote the `.npz` THEMSELVES, they know what
        they put in it.
        """
        if self.sample_rate != sr:
            return False
        if abs(self.duration_s - duration_s) > 0.05:
            return False
        if self.source_size or self.source_mtime:
            audio_path = Path(audio)
            if not audio_path.exists():
                return False
            stat = audio_path.stat()
            return stat.st_size == self.source_size and stat.st_mtime == self.source_mtime
        return True


def clean_pitch(track, min_conf: float = DEFAULT_MIN_CONF) -> None:
    """Median filter against octave jumps + masking of unvoiced frames.

    Modifies track.hz / track.confidence in place.

    `min_conf` is accepted for API stability but is NOT applied as a hard
    mask, and that refusal is measured, twice: pYIN scores every TRANSITION
    frame low (a glide in a 93 ms window is noise — measured on the truth
    harness, every frame of a 250 ms portamento carries vprob 0.016-0.62)
    while a breath ON a held note scores 0.49-0.59. Any threshold that
    excludes the breath also masks the glide, breaking `_attach_run`'s
    fusion (5 truth cases); a local-density variant (mask isolated doubt,
    keep collective doubt) shifted chant.wav's refined tempo 171.0 -> 153.5
    and broke a measure closure (102/96 divisions) — refused. The mask
    therefore only drops UNVOICED frames (`confidence == 0`, i.e. no f0):
    with the old binary flag it was already a no-op, and the voicing doubt
    now travels OUT instead — Note.confidence carries the real vprob, so
    report.json, the MIDI velocities and the MusicXML colors mean something.
    """
    hz = track.hz.copy()
    voiced = ~np.isnan(hz) & (track.confidence > 0.0)
    if voiced.sum() < 3:
        track.hz = np.where(voiced, hz, np.nan)
        track.confidence = np.where(voiced, track.confidence, 0.0)
        track.transition = np.zeros(len(track.hz), dtype=bool)
        return

    midi = hz_to_midi(hz)
    dt = float(np.median(np.diff(track.times))) if len(track.times) > 1 else 0.01
    # median filter (odd window ~ 5 frames) to remove octave jumps
    med = midi.copy()
    k = 2
    idx = np.arange(len(midi))
    for i in idx[voiced]:
        lo, hi = max(0, i - k), min(len(midi), i + k + 1)
        seg = midi[lo:hi][~np.isnan(midi[lo:hi])]
        if len(seg):
            med[i] = np.median(seg)
    # second pass: keep frames close to the median (< 1 semitone)
    ok = voiced & (np.abs(midi - med) < 1.0)
    ok &= voiced & (med >= PIANO_MIN_MIDI) & (med <= PIANO_MAX_MIDI)
    # vibrato smoothing: 9-frame median (~90 ms) — reduces vibrato while
    # PRESERVING sharp note transitions (a median does not blur hard edges,
    # unlike a moving average).
    smoothed = _median_filter_nan(med, k=4)
    smoothed = np.where(np.isnan(med), np.nan, smoothed)
    # a SHALLOW confidence dip straddling a note hand-over (a dip the
    # periodicity gate bridged: the f0 rises through it) must not cut the
    # curve: DEFAULT_GAP_S interpolation spans the hole and the transition
    # detector reads a continuous ramp again. A DEEP hole (a real breath)
    # is far longer than this and stays a gap.
    smoothed = _interpolate_short_gaps(smoothed, max_frames=max(1, int(DEFAULT_GAP_S / dt)))
    # f0 stabilization: where the pitch DERIVATIVE exceeds the
    # transition threshold the curve is MOVING (portamento ramp, note
    # hand-over). Those frames stay in the curve — the hysteresis and the
    # hand-over heuristics were calibrated with them, and masking them out
    # would open fake gaps under every legato slide (a sounding glide is
    # not a rest) — but they are MARKED so `_Segment.pitch` estimates a
    # note's height on its STABLE PLATEAU only. The mask is dilated one
    # frame on each side: a median-smoothed ramp edge still reads ~half
    # the true slope at the exact boundary.
    slope = np.abs(np.gradient(smoothed)) / dt
    moving = np.nan_to_num(slope, nan=0.0) > TRANSITION_ST_S
    dilated = moving.copy()
    dilated[1:] |= moving[:-1]
    dilated[:-1] |= moving[1:]
    track.transition = dilated & ok
    track.hz = np.where(ok, 440.0 * 2.0 ** ((smoothed - 69.0) / 12.0), np.nan)
    track.confidence = np.where(ok, track.confidence, 0.0)


def _interpolate_short_gaps(x: np.ndarray, max_frames: int) -> np.ndarray:
    """Linearly interpolate NaN runs up to `max_frames` long (in place logic)."""
    n = len(x)
    if n == 0 or max_frames < 1:
        return x
    out = x.copy()
    i = 0
    while i < n:
        if not np.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(out[j]):
            j += 1
        run = j - i
        if run <= max_frames and 0 < i and j < n:
            out[i:j] = np.linspace(out[i - 1], out[j], run + 2)[1:-1]
        i = j
    return out


def _median_filter_nan(x: np.ndarray, k: int = 4) -> np.ndarray:
    """Median filter (2k+1 window) ignoring NaNs, edge-preserving."""
    n = len(x)
    if k < 1 or n == 0:
        return x
    out = np.array(x, dtype=float, copy=True)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        seg = x[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = float(np.median(seg))
    return out
