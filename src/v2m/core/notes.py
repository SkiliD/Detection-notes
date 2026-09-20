"""Pitch-curve cleanup and note segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .music import hz_to_midi
from .onsets import Attacks

DEFAULT_MIN_CONF = 0.6
DEFAULT_MIN_NOTE_S = 0.060   # 60 ms
DEFAULT_GAP_S = 0.050        # gaps < 50 ms are interpolated
DEFAULT_SEMITONES = 0.5      # note break when |dmidi| > 0.5
NEAR_MERGE_ST = 0.4          # merge segments whose medians are close
ATTACK_TOL_S = 0.120         # an attack this close confirms a note start
MIN_UNCONFIRMED_S = 0.120    # a note nobody attacked must last this long
SWEEP_MIN_ST = 0.5           # travel (semitones) required to call a glide
SWEEP_THROUGH_ST = 0.4       # how far the middle may deviate from a straight path
SWEEP_TAIL_FRAMES = 3        # hand-over window (~60 ms)
SWEEP_TAIL_ST = 0.30         # a glide is still moving through that window
SAME_ONSET_S = 0.100         # two onsets this close are one articulation

# Why ATTACK_TOL_S equals MIN_UNCONFIRMED_S: measured on a real take
# (chant.wav), a consonant lands 90-130 ms before the plateau the pitch
# tracker reports. Narrower (80 ms) left 4 notes "orphaned" from their own
# attack; wider would reach past the shortest note the segmenter can emit,
# so a nearer attack can only be the same articulation (see AGENTS.md).


@dataclass
class Note:
    """A segmented note, ready for MIDI export."""

    start: float   # seconds
    end: float     # seconds
    midi: int      # MIDI number (rounded)
    confidence: float  # 0..1, mean confidence
    figure: str | None = None  # note figure (see figures.py), None if off-grid
    strength: float = 0.0  # onset accent 0..1 (local RMS vs its neighbours)

    @property
    def duration(self) -> float:
        return self.end - self.start


def compute_onset_strengths(notes, y: np.ndarray, sr: int) -> None:
    """Fill `note.strength` (0..1) from the audio envelope at each onset.

    The signal is cut into small RMS frames; a note's strength is the peak
    envelope around its start, normalized against the recording's own
    quiet/floor percentiles — so accents are RELATIVE within one take,
    which is what meter phase discrimination needs (no absolute loudness).
    """
    if not notes:
        return
    hop = 256
    n_frames = len(y) // hop
    if n_frames < 4:
        for n in notes:
            n.strength = 0.5
        return
    frames = np.asarray(y[: n_frames * hop], dtype=np.float64).reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    env = np.convolve(rms, np.ones(3) / 3, mode="same")  # ~3-frame smoothing
    floor = float(np.percentile(env, 20))
    span = max(float(np.percentile(env, 98)) - floor, 1e-9)
    for n in notes:
        i = min(n_frames - 1, max(0, int(n.start * sr / hop)))
        lo, hi = max(0, i - 2), min(n_frames, i + 3)
        local = float(np.max(env[lo:hi]))
        n.strength = round(min(1.0, max(0.0, (local - floor) / span)), 3)


def clean_pitch(track, min_conf: float = DEFAULT_MIN_CONF) -> None:
    """Median filter against octave jumps + masking of low-confidence frames.

    Modifies track.hz / track.confidence in place.
    """
    hz = track.hz.copy()
    voiced = ~np.isnan(hz) & (track.confidence >= min_conf)
    if voiced.sum() < 3:
        track.hz = np.where(voiced, hz, np.nan)
        track.confidence = np.where(voiced, track.confidence, 0.0)
        return

    midi = hz_to_midi(hz)
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
    ok &= voiced & (med >= 21.0) & (med <= 108.0)  # A0..C8
    # vibrato smoothing: 9-frame median (~90 ms) — reduces vibrato while
    # PRESERVING sharp note transitions (a median does not blur hard edges,
    # unlike a moving average).
    smoothed = _median_filter_nan(med, k=4)
    smoothed = np.where(np.isnan(med), np.nan, smoothed)
    track.hz = np.where(ok, 440.0 * 2.0 ** ((smoothed - 69.0) / 12.0), np.nan)
    track.confidence = np.where(ok, track.confidence, 0.0)


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


@dataclass
class _Segment:
    """A raw plateau of the pitch curve, before it becomes a `Note`."""

    start: float
    end: float
    frames: np.ndarray   # frame indices into the pitch track
    midis: np.ndarray    # pitch (float MIDI) of those frames
    confidence: float
    forced_pitch: float | None = None  # set when the run's arrival fixes it

    @property
    def duration(self) -> float:
        return self.end - self.start

    def holds_pitch(self) -> bool:
        """True when the pitch is settled over the last frames (an arrival).

        A glide hands over mid-motion; a note that has arrived holds its
        pitch. This is what separates a portamento (to be absorbed into the
        note it leads to) from a short note whose onset articulation was
        missed by the attack detector.
        """
        end = self.midis[-SWEEP_TAIL_FRAMES:]
        return len(end) >= 2 and float(np.max(end) - np.min(end)) < SWEEP_TAIL_ST

    @property
    def median(self) -> float:
        return float(np.median(self.midis))

    def pitch(self, semitones: float) -> float:
        """Plateau pitch: the median, trimmed around itself.

        A note keeps the frames of the legato glide that leads into it (they
        are closer to it than to the previous note); trimming beyond
        `semitones` around the median drops those stragglers, so the note
        reports the pitch actually held.
        """
        med = self.median
        near = self.midis[np.abs(self.midis - med) <= semitones]
        return float(np.median(near)) if len(near) else med


def _merge_segments(segments: list[_Segment]) -> _Segment:
    """One segment holding every frame of `segments` (chronological input)."""
    frames = np.concatenate([s.frames for s in segments])
    midis = np.concatenate([s.midis for s in segments])
    return _Segment(
        start=min(s.start for s in segments),
        end=max(s.end for s in segments),
        frames=frames,
        midis=midis,
        confidence=float(np.mean([s.confidence for s in segments])),
    )


def _hysteresis_segments(midi, times, conf, idx, semitones: float) -> list[_Segment]:
    """Split the voiced frames into plateaus (hysteresis + confirmation).

    The reference is the segment's own running median, so a slow slide is
    followed instead of exploding into one segment per frame; a break needs
    3 consecutive out-of-band frames, because a transition ramp triggers at
    its first real step.
    """
    segments: list[_Segment] = []
    seg_start: int | None = None
    seg_frames: list[int] = []

    def close(end_frame: int) -> None:
        nonlocal seg_start, seg_frames
        if seg_start is None or not seg_frames:
            return
        frames = np.asarray(seg_frames, dtype=int)
        segments.append(
            _Segment(
                start=float(times[seg_start]),
                end=float(times[end_frame]),
                frames=frames,
                midis=np.asarray(midi[frames], dtype=float),
                confidence=float(np.nanmean(conf[frames])),
            )
        )
        seg_start = None
        seg_frames = []

    prev = idx[0]
    seg_start = idx[0]
    seg_frames = [idx[0]]
    ref = float(midi[idx[0]])
    pending: int | None = None
    confirm = 3  # consecutive out-of-band frames to confirm a break
    for i in idx[1:]:
        if (i - prev) > 1:  # a gap the interpolation did not fill
            close(prev)
            seg_start = i
            seg_frames = [i]
            ref = float(midi[i])
            pending = None
            prev = i
            continue
        m = float(midi[i])
        if abs(m - ref) > semitones:
            # out of band vs the note REFERENCE (not the previous frame)
            if pending is None:
                pending = i
            if i - pending + 1 >= confirm:
                close(pending - 1)
                seg_start = pending
                seg_frames = list(range(pending, i + 1))
                ref = float(np.median(midi[seg_frames]))
                pending = None
        else:
            if pending is not None:
                # false alarm: re-absorb the pending frames
                seg_frames.extend(range(pending, i + 1))
                pending = None
            else:
                seg_frames.append(i)
            ref = float(np.median(midi[seg_frames]))
        prev = i
    close(prev)
    return segments


def _sweeps_between(seg: _Segment, prev, nxt) -> bool:
    """True when a plateau is the glide between two different notes.

    Signature of a transition: its pitch travels from the previous note's
    towards the next one's, it stays on the way in the middle, and it is
    STILL MOVING when it hands over to the next note. Three details matter:

    - Comparing the first and last THIRDS (not the endpoints): the pitch
      cleaning smears a transition by a frame or two on each side, so a held
      note looks like it moves if you only look at its first and last frame.
    - Passing through the middle: a held note (vibrato included) shows no
      travel between its thirds, a glide shows the travel on the way.
    - The hand-over window: a glide ends mid-motion, a note ends on its own
      pitch. Without this, a real short note sitting exactly between two
      others (the ambiguous case geometry alone cannot resolve) disappears
      into the sweep — measured on a synthetic voice, a 150 ms note between
      two notes 12 semitones apart was silently lost.
    """
    if prev is None or nxt is None:
        return False
    midis = seg.midis
    lo, hi = sorted((prev.median, nxt.median))
    if hi - lo < 1.0:
        return False  # not a sweep between two different pitches
    med = float(np.median(midis))
    if med < lo - 0.5 or med > hi + 0.5:
        return False
    third = len(midis) // 3
    if third < 1:
        return False
    head = float(np.median(midis[:third]))
    tail = float(np.median(midis[-third:]))
    if abs(tail - head) < SWEEP_MIN_ST:
        return False  # the plateau holds its pitch: it is a note
    if seg.holds_pitch():
        return False  # it arrived and held: this is a note, not a path
    mid = float(np.median(midis[third : 2 * third]))
    return min(head, tail) - SWEEP_THROUGH_ST <= mid <= max(head, tail) + SWEEP_THROUGH_ST


def _arrives_on_next(seg: _Segment, nxt) -> bool:
    """True when the segment slides and ENDS on the next note's pitch.

    The hysteresis can keep the portamento and the arrival hold in a single
    segment: its median is the middle of the slide while its last frames
    already hold the arrival pitch. That is the beginning of the next note,
    not a note of its own (measured on a détaché line sung with a 90 ms
    portamento: 2 micro-notes for 6 sung notes, at the neighbouring semitone).
    """
    if nxt is None:
        return False
    tail = float(np.median(seg.midis[-SWEEP_TAIL_FRAMES:]))
    if abs(tail - seg.median) < SWEEP_MIN_ST:
        return False  # it holds its own pitch: a note
    return abs(tail - nxt.median) <= NEAR_MERGE_ST


def _articulated_at_start(seg: _Segment, attacks: Attacks | None) -> bool:
    """An attack right at the segment start: the singer started it here."""
    return attacks is not None and attacks.near(seg.start, ATTACK_TOL_S) is not None


def _articulated_inside(seg: _Segment, attacks: Attacks | None) -> bool:
    """An attack STRICTLY inside the segment: a new note was articulated.

    Both ends are excluded, and that matters at the end: the consonant of the
    note that FOLLOWS a short segment falls inside it (measured on a
    détaché line sung with a 90 ms portamento — the arrival note's consonant
    landed in the middle of the glide fragment and saved it, giving 2 extra
    micro-notes for 6 notes). An attack that close to the segment's end does
    not belong to this segment, it belongs to the next one.
    """
    if attacks is None:
        return False
    return attacks.in_span(seg.start + ATTACK_TOL_S, seg.end - ATTACK_TOL_S) > 0


def _is_transition(seg: _Segment, prev, nxt, attacks: Attacks | None) -> bool:
    """A plateau that is not a note: a glide, or a fragment nobody attacked.

    Two signatures, in this order:

    1. **the sweep**: the pitch travels monotonically from the previous note
       to the next one. It wins even against an attack at the segment start,
       because that attack is precisely the arrival note's consonant ("ba"
       then slide) — only an attack INSIDE the segment means a new note.
    2. **the unarticulated fragment**: too short to be a note, with nobody
       starting it. Without attack evidence (callers that only have a pitch
       curve) this rule needs the sweep too, to stay strict.
    """
    if _sweeps_between(seg, prev, nxt) and not _articulated_inside(seg, attacks):
        return True
    if _arrives_on_next(seg, nxt) and not _articulated_inside(seg, attacks):
        return True
    if seg.duration >= MIN_UNCONFIRMED_S:
        return False
    if attacks is None:
        return _sweeps_between(seg, prev, nxt)
    return not _articulated_at_start(seg, attacks)


def _split_between(left: _Segment, run: list[_Segment], right: _Segment, times) -> None:
    """Hand a transition run's frames to the note they are closest to.

    The split is at the pitch midpoint, so the boundary lands where the
    sweep is ambiguous and both notes keep a duration close to what was
    sung (attributing the whole glide to the arrival note would stretch it
    by up to 15 %, enough to shift its figure).
    """
    if not run:
        return
    frames = np.concatenate([s.frames for s in run])
    midis = np.concatenate([s.midis for s in run])
    midpoint = 0.5 * (left.median + right.median)
    to_left = midis <= midpoint if right.median >= left.median else midis >= midpoint
    if to_left.any():
        left.frames = np.concatenate([left.frames, frames[to_left]])
        left.midis = np.concatenate([left.midis, midis[to_left]])
        left.end = float(times[frames[to_left][-1]])
    if not to_left.all():
        to_right = ~to_left
        right.frames = np.concatenate([frames[to_right], right.frames])
        right.midis = np.concatenate([midis[to_right], right.midis])
        right.start = float(times[frames[to_right][0]])


def _attach_run(
    prev: _Segment | None,
    run: list[_Segment],
    seg: _Segment,
    times,
    attacks: Attacks | None,
) -> _Segment:
    """Give a transition run to the note it leads to, or share it between both.

    When the run starts on an attack AND has arrived (its last frames hold
    their pitch), it is a note sung with a portamento: a ladder of
    intermediate plateaus plus its arrival, which taken separately would be
    a handful of phantom notes. It becomes ONE note, at the pitch it arrived
    at — not the median of the ladder it crossed — and the following segment
    is a different note and stays one.

    Otherwise the run is a plain glide: its frames are shared at the pitch
    midpoint between the two notes, so the arrival note keeps the onset it
    was sung with (merging the glide into it would start it early) and both
    keep their duration.
    """
    articulated = attacks is not None and attacks.near(run[0].start, ATTACK_TOL_S) is not None
    if articulated and run[-1].holds_pitch():
        merged = _merge_segments(run)
        merged.forced_pitch = float(np.median(run[-1].midis))
        return merged
    if prev is not None:
        _split_between(prev, run, seg, times)
        return seg
    return _merge_segments([*run, seg])  # leading transient


def _absorb_transitions(
    segments: list[_Segment], times, attacks: Attacks | None
) -> list[_Segment]:
    """Collapse every run of transitions onto the notes it links.

    Measured on a synthetic voice: a 90 ms legato glide turned 6 sung notes
    into 10 detected ones, the four extra notes sitting between the real
    pitches. Those fragments are not notes — they are the path from one note
    to the next, and they belong to the notes they connect.
    """
    if len(segments) < 2:
        return segments
    flags = []
    for i, seg in enumerate(segments):
        prev = segments[i - 1] if i else None
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        flags.append(_is_transition(seg, prev, nxt, attacks))

    notes: list[_Segment] = []
    pending: list[_Segment] = []
    for seg, is_transition in zip(segments, flags):
        if is_transition:
            pending.append(seg)
            continue
        if pending:
            seg = _attach_run(notes[-1] if notes else None, pending, seg, times, attacks)
            pending = []
        notes.append(seg)
    if pending and notes:
        notes[-1] = _merge_segments([notes[-1], *pending])  # trailing transient
    return notes


def _split_rearticulated(notes: list[Note], attacks: Attacks | None) -> list[Note]:
    """Cut a held note the singer re-attacked at the SAME pitch.

    The offline rules treat a plateau as one note: a same-pitch
    re-articulation (breath-less ta-ta-ta on one degree, glottal stop) has
    no pitch break and no gap, so nothing told the segmenter to cut it —
    while the live segmenter always could (it closes a note on any attack,
    same pitch or not; LiveNoteSegmenter.push). The attack evidence is
    already there, this pass just uses it: an attack STRICTLY inside a note
    cuts it in two.

    Guarded against false splits: the attack must sit more than
    ATTACK_TOL_S from both ends (an attack near the end belongs to the
    NEXT note, per the hand-over convention) and both halves must last at
    least MIN_UNCONFIRMED_S, so a stray mid-vibrato detection cannot spawn
    a fragment.
    """
    if not notes or attacks is None:
        return notes
    out: list[Note] = []
    for n in notes:
        cuts = [
            t
            for t in attacks.times
            if n.start + ATTACK_TOL_S < t < n.end - ATTACK_TOL_S
            and t - n.start >= MIN_UNCONFIRMED_S
            and n.end - t >= MIN_UNCONFIRMED_S
        ]
        if not cuts:
            out.append(n)
            continue
        prev = n.start
        for t in cuts:
            out.append(
                Note(start=prev, end=t, midi=n.midi, confidence=n.confidence)
            )
            prev = t
        out.append(Note(start=prev, end=n.end, midi=n.midi, confidence=n.confidence))
    return out


def _merge_same_onset(notes: list[Note]) -> list[Note]:
    """Collapse notes that share one articulation into the one that held.

    A vibrato excursion or a fast portamento can close a fragment a couple of
    frames before the real note's consonant, so both carry the same attack:
    kept, that turned one sung note into a note plus an 80 ms micro-note at
    the neighbouring semitone (measured on a synthetic voice, at 45 and at 64
    — glides up and down). Nobody articulates two notes within 100 ms, and an
    artefact is always short, so a short note starting `SAME_ONSET_S` before
    its neighbour is that neighbour: the survivor is the longer one, starting
    where the articulation happened.
    """
    kept: list[Note] = []
    for n in notes:
        if kept and n.start - kept[-1].start <= SAME_ONSET_S and min(n.duration, kept[-1].duration) < MIN_UNCONFIRMED_S:
            if n.duration > kept[-1].duration:
                n.start = min(n.start, kept[-1].start)
                n.confidence = max(n.confidence, kept[-1].confidence)
                kept[-1] = n
            continue
        kept.append(n)
    return kept


def _fuse_close(segments: list[_Segment], gap_s: float) -> list[_Segment]:
    """Merge adjacent segments with close medians (jitter, vowel changes)."""
    fused: list[_Segment] = []
    for s in segments:
        if (
            fused
            and s.start - fused[-1].end <= gap_s
            and abs(s.median - fused[-1].median) < NEAR_MERGE_ST
        ):
            fused[-1] = _merge_segments([fused[-1], s])
        else:
            fused.append(s)
    return fused


def track_to_notes(
    track,
    sr: int,
    min_conf: float = DEFAULT_MIN_CONF,
    min_note_s: float = DEFAULT_MIN_NOTE_S,
    gap_s: float = DEFAULT_GAP_S,
    semitones: float = DEFAULT_SEMITONES,
    attacks: Attacks | None = None,
) -> list[Note]:
    """Segment the cleaned pitch curve into notes (start, end, midi).

    `attacks` (see onsets.detect_attacks) tells a real note start from a
    legato glide; without it the segmentation falls back on geometry only.
    """
    clean_pitch(track, min_conf=min_conf)
    midi = hz_to_midi(track.hz)
    times = track.times

    voiced = ~np.isnan(midi)
    if not voiced.any():
        return []

    # interpolate short gaps
    idx = np.flatnonzero(voiced)
    gaps = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1:
            gap = (times[b] - times[a])
            if gap <= gap_s:
                gaps.append((a, b))
    for a, b in gaps:
        span = b - a
        for j in range(a + 1, b):
            w = (j - a) / span
            midi[j] = (1 - w) * midi[a] + w * midi[b]
            voiced[j] = True

    segments = _hysteresis_segments(midi, times, track.confidence, idx, semitones)
    segments = _absorb_transitions(segments, times, attacks)
    segments = _fuse_close(segments, gap_s)

    notes = [
        Note(
            start=s.start,
            end=s.end,
            midi=int(round(s.forced_pitch if s.forced_pitch is not None else s.pitch(semitones))),
            confidence=s.confidence,
        )
        for s in segments
        if s.duration >= min_note_s
    ]

    notes = _merge_same_onset(notes)

    # merge consecutive same-pitch notes separated by a short gap
    merged: list[Note] = []
    for n in notes:
        if (
            merged
            and n.midi == merged[-1].midi
            and n.start - merged[-1].end <= gap_s
        ):
            merged[-1].end = n.end
            merged[-1].confidence = max(merged[-1].confidence, n.confidence)
        else:
            merged.append(n)

    # a same-pitch re-articulation is two notes (the merge above must not
    # undo the cut, so this pass runs after it)
    merged = _split_rearticulated(merged, attacks)

    # temporal snap: center note starts/ends on the frame
    frame_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    for n in merged:
        n.start = max(0.0, n.start - frame_dt / 2)
        n.end = n.end + frame_dt / 2
    return merged
