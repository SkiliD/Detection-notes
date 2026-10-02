"""Pitch-curve cleanup and note segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .music import hz_to_midi, on_piano
from .observation import DEFAULT_GAP_S, VoiceObservation
from .onsets import Attacks

DEFAULT_MIN_NOTE_S = 0.060   # 60 ms
DEFAULT_SEMITONES = 0.5      # note break when |dmidi| > 0.5
NEAR_MERGE_ST = 0.4          # merge segments whose medians are close
ATTACK_TOL_S = 0.120         # an attack this close confirms a note start
MIN_UNCONFIRMED_S = 0.120    # a note nobody attacked must last this long
SWEEP_MIN_ST = 0.5           # travel (semitones) required to call a glide
SWEEP_THROUGH_ST = 0.4       # how far the middle may deviate from a straight path
SWEEP_TAIL_FRAMES = 3        # hand-over window (~60 ms)
SWEEP_TAIL_ST = 0.30         # a glide is still moving through that window
# A long CONTINUOUS slide (one segment, no plateau): its head and tail
# thirds disagree by at least this much and it never settles. Measured
# on the truth glide corpus (0.77-1.5 s slides): 0.8-1.6 st of travel;
# real notes (vibrato included) stay under 0.2.
SLIDE_TRAVEL_ST = 0.5
# A run is the SLIDE (rather than a note's approach) when its FIRST and
# LAST frame span at least this much: four times the slide's own travel,
# i.e. two real semitones — a rung ladder covers 7-11 st over 10-13 rungs.
SLIDE_RUN_SPAN_ST = SLIDE_TRAVEL_ST * 4
# A continuous SLIDE can also read as a STAIRCASE: short HELD plateaus
# (the tracker re-locks between the cleaning's ramp bridges) chained by
# non-held bridges, every step a third or more (the real take's steps
# read 3.1-4.6 st; the fastest REAL-note chain this must not swallow —
# chant.wav's descending thirds at 171 BPM — steps 2.75-2.90 st) and
# every step shorter than a melody note (0.3 s).
SLIDE_STEP_ST = 3.0
SLIDE_PLATEAU_MAX_S = 0.30
# A slide is its OWN note only when the next segment starts after a REAL
# break; sung straight INTO its arrival (hysteresis hand-over, gap of 1-3
# frames) the arrival is the slide's destination and both stay ONE note.
# Measured: a slide-to-arrival chain hands over with a 0.02 s gap, a slide
# followed by a different note after a breath sits 0.12 s away.
SLIDE_BREAK_S = 0.06
SAME_ONSET_S = 0.100         # two onsets this close are one articulation
# A singer's systematic offset absorbed by _measure_tuning_offset: beyond
# this the take is TRANSPOSED (or misread), not out of tune — absorbing it
# would re-key the melody.
TUNING_OFFSET_MAX_ST = 0.35
# A pitch the key snap may pull when it sits almost exactly between two
# semitones and neither belongs to the key (the neighbour note above or
# below IS in it). Beyond ±0.30 the note is a chromatic shade, never a
# mistuned scale degree.
AMBIGUOUS_DEV_ST = 0.30

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
    # signed residual pitch deviation (st, AFTER the global tuning offset:
    # +0.4 = 40 cents sharp of its semitone). Feeds the key-aware snap
    # (keys.snap_to_key); NOT exported as the report's confidence is.
    deviation: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class _Segment:
    """A raw plateau of the pitch curve, before it becomes a `Note`."""

    start: float
    end: float
    frames: np.ndarray   # frame indices into the pitch track
    midis: np.ndarray    # pitch (float MIDI) of those frames
    confidence: float
    transition: np.ndarray | None = None  # derivative mask (see clean_pitch)
    forced_pitch: float | None = None  # set when the run's arrival fixes it

    @property
    def stable(self) -> np.ndarray:
        """The non-transition frames (the plateau the pitch is read on).

        A property, not a field: `midis` is REASSIGNED by the fusion and
        split helpers below (`_merge_segments`, `_split_between*`, the
        forced-pitch trim), and a stored slice would silently go stale —
        the length check falls back to `midis` when the mask was not kept
        alongside, which keeps the pitch estimate defined (never wrong by
        construction, just less filtered).
        """
        if self.transition is not None and len(self.transition) == len(self.midis):
            return self.midis[~self.transition]
        return self.midis

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
        """Plateau pitch: the median of the STABLE frames, trimmed.

        Two protections, both measured:

        - the frames marked `transition` (clean_pitch's derivative mask:
          portamento ramps, note hand-overs) are excluded, so the pitch is
          read on the plateau the singer HELD, never on the ramp that leads
          into it — a take that drifts 0.6 st across a note used to have its
          height pulled mid-way by the ramp frames.
        - the remaining frames are trimmed around their median (`semitones`)
          to drop the legato glide's stragglers, which are closer to this
          note than to the previous one but are not its pitch.
        """
        if self.stable.size:
            base = self.stable
        else:
            base = self.midis
        med = float(np.median(base))
        near = base[np.abs(base - med) <= semitones]
        return float(np.median(near)) if len(near) else med


def _merge_segments(segments: list[_Segment]) -> _Segment:
    """One segment holding every frame of `segments` (chronological input)."""
    frames = np.concatenate([s.frames for s in segments])
    midis = np.concatenate([s.midis for s in segments])
    transitions = [
        s.transition for s in segments
        if s.transition is not None and len(s.transition) == len(s.midis)
    ]
    trans = (
        np.concatenate(transitions)
        if len(transitions) == len(segments)
        else None
    )
    return _Segment(
        start=min(s.start for s in segments),
        end=max(s.end for s in segments),
        frames=frames,
        midis=midis,
        transition=trans,
        confidence=float(np.mean([s.confidence for s in segments])),
    )


def _settle_on_tail(seg: _Segment, tail_frames: int) -> _Segment:
    """Pin `seg`'s pitch to its own TAIL and keep only the frames near it.

    The clean median of a merged glide trails the RAMP (a 400 ms portamento
    60->67 reads 65.4 while the arrival is 67 — the ladder carries two
    frames at each intermediate step and only a handful at the arrival):
    the honest pitch is the median of the arrival's own frames. The trim
    keeps the median and the reported pitch in agreement, and keeps
    `_fuse_close` (which compares MEDIANS) from fusing the note with the
    wrong neighbours (measured on chant.wav: 26 notes @ 154.7 BPM instead
    of 29 @ 171.0 without the trim).
    """
    seg.forced_pitch = float(np.median(seg.midis[-tail_frames:]))
    near = seg.midis[np.abs(seg.midis - seg.forced_pitch) <= NEAR_MERGE_ST]
    if len(near):
        seg.midis = near
    return seg


def _hysteresis_segments(midi, times, conf, idx, semitones: float, transition=None) -> list[_Segment]:
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
                transition=transition[frames] if transition is not None else None,
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
            elif i - pending + 1 >= confirm:
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


GLIDE_STAIR_ST = 1.4     # max median delta between two rungs of one glide


def _is_glide_stair(seg: _Segment, prev, nxt) -> bool:
    """True for a short rung CLIMBING STEPS toward a different arrival.

    A long glissando leaves a ladder of short hysteresis segments between
    the two notes; each sits one-ish semitone above the previous one and
    the arrival note (the NEXT note-length segment) is a step further in
    the same direction. Three conditions, all measured on the 0.25/0.5 s
    glides: the rung is short (below MIN_UNCONFIRMED_S), both neighbours
    exist, and prev < seg < nxt (or the descent's mirror) with every step
    at most GLIDE_STAIR_ST — a trill or a fast melody line has its steps
    BETWEEN notes, with note-length segments between the moves, while a
    glide's rungs CHAIN (every neighbour is itself a short rung or the
    arrival).
    """
    if seg.duration >= MIN_UNCONFIRMED_S or prev is None or nxt is None:
        return False
    # NOTE-LENGTH neighbours are exempt: a fast melody ("doubles", 120 ms
    # notes) IS a chain of short semitone-to-2st steps between note-length
    # segments — those are real notes, not a glide. The stairs evidence
    # applies only when the rung sits between two OTHER short rungs or an
    # arrival fragment, which is what a glide's hysteresis ladder looks
    # like (measured on the 0.25-0.5 s glides: every rung short, every
    # neighbour short). A GLIDE-RUNG neighbour also refuses on its own:
    # between two 120 ms notes of a fast line the runs are NOTE-length,
    # while a glide's ladder is entirely made of short rungs.
    if prev.duration >= MIN_UNCONFIRMED_S and nxt.duration >= MIN_UNCONFIRMED_S:
        return False
    if prev.duration >= MIN_UNCONFIRMED_S and abs(nxt.median - prev.median) <= GLIDE_STAIR_ST:
        # between two real notes a semitone apart, a short middle segment is
        # a hand-over frame, not a rung — the notes absorb it either way
        return False
    up = seg.median - prev.median
    over = nxt.median - seg.median
    if up <= 0.1 and over <= 0.1:  # not climbing nor descending
        return False
    if up > 0.1 and over < -0.1:  # up then down: a peak, not a stair
        return False
    if up < -0.1 and over > 0.1:  # down then up: a valley
        return False
    return abs(up) <= GLIDE_STAIR_ST and abs(over) <= GLIDE_STAIR_ST


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


def _is_long_slide(seg: _Segment, nxt) -> bool:
    """True for a segment whose pitch is one long CONTINUOUS slide.

    Two shapes, both measured on the truth glide corpus:
    - the take-opening slide: ONE long segment (the hysteresis follows
      the ramp), no note after it. Its signature is TRAVEL — the thirds
      disagree by more than SLIDE_TRAVEL_ST and it never settles.
    - a rung CHAIN of a slide whose ramp the cleaning broke into 40-60 ms
      steps: every step is short, climbs one-ish semitone (the stairs
      test also fires, but its NOTE-LENGTH-neighbour exemption lets a
      step pass when the arrival is already a note), and the ARRIVAL
      plateau follows. The arrival note itself is NOT a slide: it holds
      its pitch (`holds_pitch` True).
    A real note under 0.2 st of third travel is never a slide (vibrato's
    thirds oscillate around ONE centre); a fast melody's steps sit
    BETWEEN note-length segments, which the stairs rules already keep.
    """
    third = len(seg.midis) // 3
    if third < 1:
        return False
    head = float(np.median(seg.midis[:third]))
    tail = float(np.median(seg.midis[-third:]))
    if abs(tail - head) < SLIDE_TRAVEL_ST:
        return False
    if seg.holds_pitch():
        return False  # it arrived and held: a note (the arrival plateau)
    # the arrival note protects its own pitch: when a NOTE-length
    # neighbour exists and this segment ENDS at its pitch, the sweep and
    # arrival rules own the case (the slide-with-arrival)
    if nxt is not None and nxt.duration >= MIN_UNCONFIRMED_S and seg.duration >= MIN_UNCONFIRMED_S:
        return False
    return True


def _arrives_on_next(seg: _Segment, nxt) -> bool:
    """True when the segment slides and ENDS on the next note's pitch.

    The hysteresis can keep the portamento and the arrival hold in a single
    segment: its median is the middle of the slide while its last frames
    already hold the arrival pitch. That is the beginning of the next note,
    not a note of its own (measured on a détaché line sung with a 90 ms
    portamento: 2 micro-notes for 6 sung notes, at the neighbouring semitone).

    The "next" must be a NOTE length, not a fragment: a deep vibrato's
    hand-over leaves a 40 ms rung between the two plateaus, and comparing a
    0.8 s held note against that rung read the hold as "sliding toward it"
    (its tail was mid-oscillation, half a semitone from the rung) — the hold
    was absorbed into the rung and both plateaus engraved as ONE duplicated
    note (measured: vibrato 0.6 st around midi 64, two 67s for two sung
    notes). A fragment below MIN_UNCONFIRMED_S is a transition candidate
    itself, never a note to arrive at.
    """
    if nxt is None or nxt.duration < MIN_UNCONFIRMED_S:
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

    A rung of a LONG glide is a transition on the STAIRS evidence alone:
    each hysteresis rung is one semitone-ish above the previous one, and a
    singer does not articulate five 40 ms notes a semitone apart. The
    neighbour-based sweep test needs BOTH notes' medians (a long glide
    spans 7 st, so rung->rung travel never reaches SWEEP_MIN_ST against its
    1 st tail spread) and the arrival note is sometimes too short to be the
    sweep's anchor — the stairs never are (measured: a 500 ms glide read as
    note 60 + phantom 62 + note 67; the rungs were 0.8-1.3 st apart, every
    one legitimized by the arrival's consonant).
    """
    if _is_glide_stair(seg, prev, nxt) and not _articulated_inside(seg, attacks):
        return True
    if _sweeps_between(seg, prev, nxt) and not _articulated_inside(seg, attacks):
        return True
    if _arrives_on_next(seg, nxt) and not _articulated_inside(seg, attacks):
        return True
    # the LONG SLIDE (one continuous slide segment, no plateau): a
    # glissando engraved as an arpeggio is the defect this gate refuses.
    # An attack INSIDE still means a new note was articulated on the way.
    if _is_long_slide(seg, nxt) and not _articulated_inside(seg, attacks):
        return True
    if seg.duration >= MIN_UNCONFIRMED_S:
        return False
    if attacks is None:
        return _sweeps_between(seg, prev, nxt)
    # An attack at the start legitimizes the segment only when the segment
    # SETTLES: a real short note holds its own pitch over its last frames,
    # while a glide's rung is a RAMP still climbing. Calibrated on the tail
    # spread (max-min of the last 4 frames): real notes spread 0.0-0.5 (the
    # "doubles" melody 0.0/0.10/0.50 — the 0.50 is one boundary frame), a
    # glide's rungs 0.40-0.80 WITH the top frame last (still rising). So a
    # rung is refused when its tail rises: the LAST frame is the tail's max
    # and the spread exceeds 0.5. A settled note has its last frame inside
    # the tail's span (spread <= 0.5, or the last frame below the max — a
    # vibrato's turn).
    if not _articulated_at_start(seg, attacks):
        return True
    tail = seg.midis[-min(4, len(seg.midis)):]
    if float(tail[-1]) < float(np.max(tail)) - 0.05:
        return False  # the tail is coming back DOWN: it settled (vibrato)
    # A SETTLED tail (spread <= 0.5, last frame near the max) is a real short
    # note — the "doubles" melody segments spread exactly 0.0-0.50. A tail
    # still RISING (spread > 0.5, last frame = max) is a glide's rung. (The
    # comparison was once inverted: settled segments were absorbed as
    # transitions and 3 of 4 double-croches vanished.)
    return float(np.max(tail) - np.min(tail)) > 0.5


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
) -> list[_Segment]:
    """Give a transition run to the note it leads to, or share it between both.

    Returns the segments to emit in place of `seg` — one, or two when the
    run is a take-opening slide the next segment does not continue.

    When the run starts on an attack AND has arrived (its last frames hold
    their pitch), it is a note sung with a portamento: a ladder of
    intermediate plateaus plus its arrival, which taken separately would be
    a handful of phantom notes. It becomes ONE note, at the pitch it arrived
    at — not the median of the ladder it crossed.

    Otherwise the run is a plain glide: its frames are shared at the pitch
    midpoint between the two notes, so the arrival note keeps the onset it
    was sung with (merging the glide into it would start it early) and both
    keep their duration.
    """
    articulated = attacks is not None and attacks.near(run[0].start, ATTACK_TOL_S) is not None
    if articulated and run[-1].holds_pitch():
        merged = _merge_segments(run)
        # The attack at the run's start is the run's own onset ONLY if the
        # run is a real note length: a portamento is sung THROUGH a
        # consonant ("ta-aa" with a 500 ms glide), so a SHORT run (under
        # MIN_UNCONFIRMED_S) carrying the arrival's consonant is a glide
        # fragment, not a note of its own (measured: an 80 ms articulated
        # rung of a 500 ms glide engraved a phantom 62 between the two sung
        # notes). Real short notes keep the branch — their run is the note.
        if merged.duration >= MIN_UNCONFIRMED_S:
            return [_settle_on_tail(merged, len(run[-1].midis))]
    if prev is not None:
        _split_between_at_attack(prev, run, seg, times, attacks)
        # the arrival keeps the glide frames to its LEFT, but its PITCH is
        # the arrival plateau: the median of glide+arrival trails the ramp
        # (measured on the 500 ms glide: median 66.09 for a 67 arrival — a
        # wrong semitone). The honest pitch is the median of the arrival's
        # OWN tail — the frames that hold the plateau — which is exactly
        # what `holds_pitch` reads (measured: 66.99 = the sung 67). Applied
        # only when the arrival SETTLED (`holds_pitch`) — a tail median of a
        # segment still ramping pins the ramp — and only when the run does
        # not DOMINATE the arrival (>= 2x): on a dominant glide the tail is
        # short and the ramp's median is the safer read (rare; the
        # articulated branch handles most of those).
        if seg.holds_pitch() and _merge_segments(run).duration < 2.0 * seg.duration:
            tail = seg.midis[-SWEEP_TAIL_FRAMES:]
            seg.forced_pitch = float(np.median(tail))
        return [seg]
    # LEADING run (no note before it): a take-opening glissando. Past a REAL
    # break (`SLIDE_BREAK_S`) the long slide is its OWN note — the segment
    # after the break is a different note and stays one. Sung straight INTO
    # its arrival, the approach merges INTO the arrival note and the pitch
    # is the arrival's plateau, never the ramp's median.
    run_span = abs(run[-1].midis[-1] - run[0].midis[0])
    if (
        len(run) >= 3
        and run_span >= SLIDE_RUN_SPAN_ST
        and seg.start - run[-1].end >= SLIDE_BREAK_S
    ):
        slide = _merge_segments(run)
        return [_settle_on_tail(slide, SWEEP_TAIL_FRAMES), seg]
    merged = _merge_segments([*run, seg])
    if seg.holds_pitch():
        _settle_on_tail(merged, SWEEP_TAIL_FRAMES)
    return [merged]


def _split_between_at_attack(
    left: _Segment,
    run: list[_Segment],
    right: _Segment,
    times,
    attacks: Attacks | None,
) -> None:
    """Split a glide run at the attack that STARTS the arrival note.

    The midpoint split assumes the glide's frames sit symmetrically around
    the boundary; a glide the singer SANG THROUGH an attack ("ta" mid-
    slide) is owned by the arrival note from the consonant onward — its
    attack time is the boundary (measured on the 500 ms glide: the mid-
    point split put 62.59's frames on the left note and engraved a 62;
    the attack sits at 1.0 s, exactly where the sung note 67 begins).
    Falls back to the midpoint when no attack lies strictly between the
    two notes' spans.
    """
    if not run:
        return
    if attacks is not None:
        lo = left.end
        hi = right.start
        for t in attacks.times:
            # the boundary itself belongs to the ARRIVAL (the AGENTS.md
            # hand-over convention): the hysteresis cut the left note at the
            # consonant that caused the break, so an attack at exactly `lo`
            # is that consonant — measured on the 0.25 s descending 67->60
            # glide, where the strict `lo < t` fell back to the midpoint and
            # left the arrival starting 150 ms late. An attack at `hi` stays
            # excluded: it is the arrival's own onset, already consumed.
            if lo - 1e-9 <= t < hi:
                frames = np.concatenate([s.frames for s in run])
                midis = np.concatenate([s.midis for s in run])
                to_left = times[frames] < t
                if to_left.any():
                    left.frames = np.concatenate([left.frames, frames[to_left]])
                    left.midis = np.concatenate([left.midis, midis[to_left]])
                    left.end = float(times[frames[to_left][-1]])
                if not to_left.all():
                    to_right = ~to_left
                    right.frames = np.concatenate([frames[to_right], right.frames])
                    right.midis = np.concatenate([midis[to_right], right.midis])
                    right.start = float(times[frames[to_right][0]])
                return
    _split_between(left, run, right, times)


def _slide_note(run: list[_Segment]) -> _Segment:
    """A run of slide segments as ONE note, pitched where it ARRIVES."""
    return _settle_on_tail(_merge_segments(run), SWEEP_TAIL_FRAMES)


def _next_stair_step(segments: list[_Segment], i: int) -> int | None:
    """Index of the next HELD plateau a slide-step away from `segments[i]`.

    A slide-step: the next held segment (non-held bridges in between are
    the cleaning's ramp re-lockings) is itself SHORT (`SLIDE_PLATEAU_MAX_S`
    — a chain member that outlasts it is a melody note the chain must not
    swallow), sits `SLIDE_STEP_ST` or more away, and the whole link is
    CONTIGUOUS (a real break makes the chain a détaché line, each note
    with its own attack and its own silence).
    """
    j = i + 1
    while j < len(segments) and not segments[j].holds_pitch():
        if segments[j].start - segments[j - 1].end > SLIDE_BREAK_S:
            return None  # a real break: not one continuous chain
        j += 1
    if j >= len(segments):
        return None
    if segments[j].start - segments[j - 1].end > SLIDE_BREAK_S:
        return None
    if segments[j].duration >= SLIDE_PLATEAU_MAX_S:
        return None
    if abs(segments[j].median - segments[i].median) < SLIDE_STEP_ST:
        return None
    return j


def _flag_staircase(segments: list[_Segment], flags: list[bool]) -> None:
    """Flag every member of a contiguous staircase of >= 3 slide plateaus.

    The cleaning re-locks between a long slide's ramp bridges, so the
    slide can read as HELD mini-plateaus (the real take: 0.08-0.26 s,
    steps 3.1-4.6 st, all contiguous). ONE link is NOT evidence: a fast
    melody also chains two short plateaus a third apart through its
    hand-over rung (the "croches" scale reads 60 -> bridge -> 64, a 4 st
    step). A MELODY's chain ends at 2 — the next degree is under 2 st —
    while a slide's staircase keeps climbing (the real take: 4 plateaus,
    11 st end to end, MONOTONIC — a zigzag of up-then-down steps is a
    melody's contour, never a slide). So the whole chain is flagged only
    at 3 plateaus and up, stepping in ONE direction; the chain's plateaus
    become transitions and the slide reads as ONE note at the arrival
    tail.
    """
    i = 0
    while i < len(segments):
        seg = segments[i]
        if not seg.holds_pitch() or seg.duration >= SLIDE_PLATEAU_MAX_S:
            i += 1
            continue
        chain = [i]
        direction = 0.0
        while (j := _next_stair_step(segments, chain[-1])) is not None:
            step = segments[j].median - segments[chain[-1]].median
            if direction and (step > 0) != (direction > 0):
                break  # the staircase turned around: a melody contour
            direction = step
            chain.append(j)
        if len(chain) >= 3:
            for k in chain:
                flags[k] = True
            i = chain[-1] + 1
        else:
            i += 1


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
    # the STAIRCASE pass: a contiguous chain of >= 3 held mini-plateaus
    # stepping a third or more each is one slide, not a fast melody
    _flag_staircase(segments, flags)

    notes: list[_Segment] = []
    pending: list[_Segment] = []
    for seg, is_transition in zip(segments, flags):
        if is_transition:
            pending.append(seg)
            continue
        if pending:
            notes.extend(
                _attach_run(notes[-1] if notes else None, pending, seg, times, attacks)
            )
            pending = []
        else:
            notes.append(seg)
    if pending:
        if notes:
            # a TRAILING chain (a glissando nobody follows): when it
            # travels a slide's worth away from the last note, it is a
            # slide in its own right, pitched where it ARRIVES — never the
            # mid-slide median.
            tail_pitch = float(np.median(pending[-1].midis[-SWEEP_TAIL_FRAMES:]))
            if len(pending) >= 3 and abs(notes[-1].median - tail_pitch) >= SLIDE_TRAVEL_ST:
                # a note-length head keeps its identity (the note the slide
                # departs from); a short head is the slide's own first rung
                if notes[-1].duration >= MIN_UNCONFIRMED_S:
                    return notes + [_slide_note(pending)]
                return [*notes[:-1], _slide_note([notes[-1], *pending])]
            notes[-1] = _merge_segments([notes[-1], *pending])  # trailing transient
        else:
            # a take-opening chain (a pure slide): the pitch it ARRIVES at
            # is its identity
            notes.append(_slide_note(pending))
    return notes


def _split_rearticulated(
    notes: list[Note],
    attacks: Attacks | None,
    midi_curve: np.ndarray | None = None,
    times: np.ndarray | None = None,
) -> list[Note]:
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
    a fragment. A note whose pitch CHANGED across the attack is not a
    re-articulation either: the absorption hands glide frames to both
    sides, so the note that owns the glide's start ends late (measured: a
    500 ms glide's attack at 1.0 sat strictly inside the stretched
    0.18-1.22 note and split it into two 60s — a duplicate). The split
    needs the halves to DISAGREE about the pitch by less than half a
    semitone, i.e. the median of the frames around the attack must sit at
    the note's own pitch on both sides; a glide stretches the median
    upward and refuses the cut.
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
        # a REAL same-pitch re-articulation: the frames just around the cut
        # hold the note's own pitch on BOTH sides, AND the voice COMES BACK
        # to the note's pitch at least one frame after the cut — the MIN
        # |pitch - note| over the after-window, not its median: a re-attack
        # touches the pitch again before the next move (measured on
        # chant.wav: 53.29/53.99 right after the 0.62 cut, then the glide
        # to 57 — a median read the glide and refused a real split), while
        # a descent departs MONOTONICALLY and never returns (measured on
        # the 67->60 glide: after the hand-over cut the curve falls 65->61,
        # min gap 1.9 st). Without the after-test the hand-over cut the
        # stretched note at its own glide's start and left a duplicate.
        #
        # The RETURN test is a UNION of two references: the note's own
        # LABEL (the rounded plateau the whole segment was read at) OR the
        # LOCAL median around the cut. A real take drifts 0.5-0.7 st across
        # a long note, so the second half of a genuine ta-ta can land half
        # a semitone above the first — the label still says 53 for both
        # halves (measured on chant.wav: around-median 52.69, second half
        # at ~53.2, label 53; a local-only reference refused the cut and
        # glued the repetition into one note, losing one note of the
        # golden fingerprint). A glide STILL refuses: it never returns to
        # either reference (the after-frames travel on), and its
        # around-median sits mid-travel, away from the label.
        glide_cut = False
        if midi_curve is not None and times is not None:
            in_note = (times >= n.start) & (times <= n.end)
            note_midis = midi_curve[in_note]
            note_times = times[in_note]
            for t in cuts:
                # a cut at t splits the note in two; a REAL re-articulation
                # has BOTH halves at the note's pitch — the first half is
                # the held note. A note whose first half sits FAR from the
                # label is a slide merged with its arrival (the take-
                # opening glissando): its "attack" is the hand-over to the
                # arrival, not a re-articulation, and cutting it would
                # engrave the approach as a duplicate of the arrival.
                before = note_midis[(note_times >= n.start) & (note_times <= t)]
                before = before[~np.isnan(before)]
                if before.size and abs(float(np.median(before)) - n.midi) > SWEEP_MIN_ST:
                    glide_cut = True
                    break
                around = note_midis[(note_times >= t - 0.08) & (note_times <= t + 0.08)]
                around = around[~np.isnan(around)]
                if not around.size:
                    continue
                around_ref = float(np.median(around))
                after = note_midis[
                    (note_times > t + 0.08) & (note_times <= min(n.end, t + 0.30))
                ]
                after = after[~np.isnan(after)]
                if not after.size:
                    continue
                returned = (
                    float(np.min(np.abs(after - n.midi))) <= NEAR_MERGE_ST
                    or float(np.min(np.abs(after - around_ref))) <= NEAR_MERGE_ST
                )
                # the anti-glide anchor: the pitch AROUND the cut still sits
                # at the note's label (within SWEEP_MIN_ST) — a glide passing
                # through reads mid-travel here (measured: 62.5 around a
                # 60->67 ramp; the descending hand-over 65 vs 67).
                anchored = abs(around_ref - n.midi) <= SWEEP_MIN_ST
                if not returned or not anchored:
                    glide_cut = True
                    break
        if glide_cut:
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


def interpret(
    observation: VoiceObservation,
    min_note_s: float = DEFAULT_MIN_NOTE_S,
    gap_s: float = DEFAULT_GAP_S,
    semitones: float = DEFAULT_SEMITONES,
) -> list[Note]:
    """Read the observation's evidence into notes (start, end, midi).

    The evidence is already CLEANED and measured: `VoiceObservation.
    from_signal` owns `clean_pitch`, the attacks and the frame clock, so this
    function only interprets what it carries — no `sr`, no `min_conf` (both
    died with the old parameters: `sr` was never read, and the confidence
    parameter is inert by ADR 0005).

    `min_note_s` / `gap_s` / `semitones` keep their defaults HERE because
    notes.py is the ONE writer of the segmentation policy (a 0.6 written in
    the pipeline once detached the offline defaults from the tested ones).

    Writing `observation.tuning_offset_st` is this pass's one side effect:
    the take's systematic offset is measured on the candidate plateaus,
    before rounding, and the caller reads it back from the same object (it
    used to travel through a `tuning_out` dict side channel).
    """
    track = observation.pitch
    attacks = observation.attacks
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

    segments = _hysteresis_segments(
        midi, times, track.confidence, idx, semitones,
        transition=getattr(track, "transition", None),
    )
    segments = _absorb_transitions(segments, times, attacks)
    segments = _fuse_close(segments, gap_s)

    # float plateau pitch of every candidate note: the tuning offset is
    # measured on these (BEFORE the rounding), and the residual deviation
    # travels on each Note for the key-aware snap (keys.snap_to_key)
    candidates: list[tuple[_Segment, float]] = []
    for s in segments:
        if s.duration < min_note_s:
            continue
        fp = s.forced_pitch if s.forced_pitch is not None else s.pitch(semitones)
        if not on_piano(int(round(fp))):
            continue  # outside A0..C8: not a note a piano has
        candidates.append((s, fp))

    # global tuning correction: a singer sits consistently a little flat or
    # sharp — one offset for the whole take, measured on the deviation to
    # the nearest semitone, applied before rounding (otherwise every note
    # rounds on the wrong side)
    offset = _measure_tuning_offset([fp for _, fp in candidates])
    observation.tuning_offset_st = offset
    notes: list[Note] = []
    for s, fp in candidates:
        adjusted = fp - offset
        midi = int(round(adjusted))
        notes.append(
            Note(
                start=s.start, end=s.end, midi=midi, confidence=s.confidence,
                deviation=max(-0.5, min(0.5, adjusted - midi)),
            )
        )

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
    merged = _split_rearticulated(merged, attacks, hz_to_midi(track.hz), times)

    # temporal snap: center note starts/ends on the frame — the frame clock
    # is a field of the observation (one owner; a hand-built observation that
    # left it unset falls back on the analysis frame length)
    frame_dt = observation.frame_dt if observation.frame_dt > 0 else 0.01
    for n in merged:
        n.start = max(0.0, n.start - frame_dt / 2)
        n.end = n.end + frame_dt / 2
    return merged


def _measure_tuning_offset(pitches: list[float]) -> float:
    """Median deviation of the take's pitches to their nearest semitone.

    One number for the WHOLE take (a singer's offset is systematic), clipped
    to ±0.35 st: beyond that the take is transposed or misread, and
    absorbing it would re-key the melody. Empty input -> 0.
    """
    if not pitches:
        return 0.0
    devs = sorted(p - round(p) for p in pitches)
    n = len(devs)
    med = devs[n // 2] if n % 2 else 0.5 * (devs[n // 2 - 1] + devs[n // 2])
    return max(-TUNING_OFFSET_MAX_ST, min(TUNING_OFFSET_MAX_ST, med))
