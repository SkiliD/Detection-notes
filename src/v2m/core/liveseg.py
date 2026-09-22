"""Real-time segmentation of a pitch curve (on/off hysteresis).

Live has no look-ahead, so the offline rules (see notes.py: a legato glide is
not a note) are applied differently: the pitch only CONFIRMS a note once it
HOLDS, and the attack evidence (onsets.AttackTracker) pins the onset and marks
a re-articulation. Without attacks the segmenter still works, but a note starts
whenever its pitch settles — which is the end of the glide into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .music import hz_to_midi, midi_to_name, on_piano
from .onsets import AttackTracker

if TYPE_CHECKING:
    from .pitch import YinBlock

HOLD_ST = 0.6            # pitch spread (semitones) tolerated to call it held
MEMORY_S = 1.0           # seconds of frames kept in the pending window
MAX_ATTACKS = 64         # recent attack times kept for the display/tests
CLEAN_FRAMES = 5         # live pitch median (~115 ms): see LiveNoteSegmenter
MIN_GAP_HOLD_S = 0.045   # a mute run this long ENDS the note (re-take/gap)


@dataclass
class LiveNote:
    start: float
    end: float
    midi: int
    confidence: float = 0.9
    figure: str | None = None
    articulated: bool = False  # the singer re-attacked: a note of its own
    strength: float = 0.0      # attack strength 0..1, take-relative: accents

    @property
    def duration(self) -> float:
        return self.end - self.start


class LivePitchStream:
    """Captured blocks -> pitch frames -> notes, with the attack evidence.

    Both front-ends (the GUI thread and the CLI session) need exactly this
    chain: YIN on every block, the silence gate, the streaming attack tracker
    and the segmenter. It lives here so the two live paths cannot drift
    apart — they had already started to.

    The pitch window is short (512 samples, ~23 ms) on purpose: the offline
    path can afford 2048, the live display cannot — measured on a synthetic
    voice, eighth notes at 150 BPM came out as 3/8 with a 2048 window (the
    tracker averaged two notes into one) and 7/8 with 512. Short windows
    cost more frames per second but each is 16x cheaper, and the low end
    holds (a 55 Hz tone is read the same at every window).
    """

    def __init__(
        self,
        sr: int,
        gate_rms: float,
        window: int = 512,
        hop: int = 256,
    ):
        self.sr = sr
        self.gate_rms = gate_rms
        self.window = window
        self.hop = hop
        self.yin: "YinBlock | None" = None
        self.attacks = AttackTracker()
        self.seg = LiveNoteSegmenter()
        self._attack_pending = False
        self._pending_strength = 0.0

    def feed(self, block: np.ndarray, t: float) -> tuple[float, list[tuple[float, float]]]:
        """Feed one captured block; return (level, frames pushed).

        The attack flag is carried to the next frame when a block yields no
        pitch frame, so the evidence is never silently dropped.
        """
        from .pitch import YinBlock  # local: keeps the module import-light

        yin = self.yin
        if yin is None:
            yin = self.yin = YinBlock(self.sr, window=self.window, hop=self.hop)
        x = np.asarray(block, dtype=np.float64)
        rms = AttackTracker.level(x)
        if self.attacks.push(t, x):
            self._attack_pending = True
            self._pending_strength = self.attacks.strength
        frames: list[tuple[float, float]] = []
        for ft, hz in yin.push(block):
            if rms < self.gate_rms:
                hz = float("nan")
            self.seg.push(ft, hz, attack=self._attack_pending, strength=self._pending_strength)
            self._attack_pending = False
            self._pending_strength = 0.0
            frames.append((ft, hz))
        return rms, frames

    def elapsed_notes(self) -> int:
        return len(self.seg.notes)


class LiveNoteSegmenter:
    """State machine: IDLE -> PENDING -> ACTIVE.

    - A note is only confirmed ("on") when its pitch has HELD for
      `confirm_s`; the onset is then the attack that started it, or the
      moment it settled. This is what keeps a glide from becoming a note.
    - It ends ("off") on a new attack, on a pitch change of more than
      `semitones`, or after `MIN_GAP_HOLD_S` without voicing — so a blip or
      a spurious frame does not split or invent a note.
    """

    def __init__(
        self,
        confirm_s: float = 0.10,
        release_s: float = 0.12,
        min_note_s: float = 0.10,
        semitones: float = 0.6,
    ):
        self.confirm_s = confirm_s
        self.release_s = release_s
        self.min_note_s = min_note_s
        self.semitones = semitones

        self.state = "IDLE"
        self.pending_start: float | None = None
        self.pending_midis: list[float] = []
        self.pending_times: list[float] = []
        self.pending_attack: float | None = None
        self.note_start: float | None = None
        self.note_midis: list[float] = []
        self.last_voiced_t: float | None = None
        self.last_hz: float = float("nan")
        self.notes: list[LiveNote] = []
        self.attacks: list[float] = []
        self.note_articulated = False
        self.note_strength = 0.0
        self.pending_strength = 0.0
        self.recent_midis: list[float] = []
        self._staged_attack: float | None = None  # attack seen on a mute frame
        self._staged_strength = 0.0               # its strength, same sidecar

    # ------------------------------------------------------------------
    def push(self, t: float, hz: float, attack: bool = False, strength: float = 0.0) -> str:
        """Advance one frame. Return 'on', 'off' or ''.

        `attack` is the live attack evidence (onsets.AttackTracker); it is
        what makes a re-articulated note separable from a held one, and what
        dates the onset at the consonant instead of at the moment the pitch
        settles. `strength` is that attack's level rise (0..1): it travels
        with the attack flag and lands on the closed note as
        `LiveNote.strength` — the accent evidence `meter.detect_meter`
        reads, the same field `Note.strength` carries offline.
        """
        ev = ""
        if attack:
            self.attacks.append(t)
            del self.attacks[:-MAX_ATTACKS]
        voiced = hz == hz and hz > 0  # not NaN
        if not voiced:
            # An attack can land on a MUTE frame: the consonant of the
            # re-attacked note, or its release gap when the level tracker
            # fires there. A note must not swallow it — stage it until the
            # next voiced frame, which is the re-attacked note itself.
            if attack and self.state == "ACTIVE":
                self._staged_attack = t
                self._staged_strength = strength
            if self.state == "ACTIVE" and self.last_voiced_t is not None:
                if t - self.last_voiced_t >= MIN_GAP_HOLD_S:
                    # a mute run this long is a re-take (or a phrase gap),
                    # not a frame dropout: the note ends at its last voiced
                    # frame. Without this, a SOFT re-phonation of the SAME
                    # pitch — no pitch change, sub-threshold level dip — was
                    # absorbed into one endless note (aa-aa-aa case).
                    self._close_note(self.last_voiced_t)
                    ev = "off"
            elif self.state == "PENDING" and self.last_voiced_t is not None:
                if t - self.last_voiced_t >= self.release_s:
                    self._reset_pending()
            return ev

        self.last_hz = hz
        m = self._clean(float(hz_to_midi(hz)))

        if (
            attack
            and self.state == "ACTIVE"
            and self.note_start is not None
            and t - self.note_start >= self.min_note_s
        ):
            # the singer articulated again: a new note begins here
            self._close_note(t)
            ev = "off"
        elif self._staged_attack is not None and self.state == "ACTIVE":
            # the attack staged on the mute frame(s) before this one is
            # fulfilled: the previous note ends there and the re-attacked
            # note BEGINS there — dated like a direct attack, otherwise the
            # flush merge would glue the repetition back into one note
            staged = self._staged_attack
            staged_strength = self._staged_strength
            self._staged_attack = None
            self._staged_strength = 0.0
            self._close_note(staged)
            self._start_pending(staged, m, staged, staged_strength)
            ev = "off"
        if self._staged_attack is not None and self.state != "ACTIVE":
            self._staged_attack = None
            self._staged_strength = 0.0

        if self.state == "IDLE":
            self._start_pending(t, m, t if attack else None, strength)
        elif self.state == "PENDING":
            # a glide is FOLLOWED, never restarted: the pending window
            # keeps the ladder of plateaus and only closes when the pitch
            # holds, which is where the note really is.
            self._extend_pending(t, m, t if attack else None, strength)
            window = self._hold_window(t)
            if window is not None:
                self.state = "ACTIVE"
                # onset: the consonant when there was one, else the
                # moment the pitch settled
                self.note_start = self.pending_attack if self.pending_attack is not None else window[0][0]
                self.note_articulated = self.pending_attack is not None
                self.note_strength = self.pending_strength
                self.note_midis = [mm for _, mm in window]
                ev = "on"
        elif self.state == "ACTIVE":
            ref = float(np.median(self.note_midis))
            if abs(m - ref) <= self.semitones:
                self.note_midis.append(m)
            else:
                self._close_note(t)
                ev = "off"
                self._start_pending(t, m)
        self.last_voiced_t = t
        return ev

    # ------------------------------------------------------------------
    def _clean(self, m: float) -> float:
        """Median of the last few live frames (the live pitch is not cleaned).

        Offline, `notes.clean_pitch` runs a 9-frame median before segmenting;
        live there is no such pass, so a vibrato keeps its full oscillation
        and `HOLD_ST` has to be wide enough to accept it — which a slow slide
        can then slip through. A 5-frame median (~115 ms) removes most of the
        oscillation while preserving the note transitions (a median does not
        blur a hard edge, unlike a moving average).
        """
        self.recent_midis.append(m)
        if len(self.recent_midis) > CLEAN_FRAMES:
            del self.recent_midis[0]
        return float(np.median(self.recent_midis))

    def _start_pending(self, t: float, m: float, attack_t: float | None = None, strength: float = 0.0) -> None:
        self.state = "PENDING"
        self.pending_start = t
        self.pending_midis = [m]
        self.pending_times = [t]
        self.pending_attack = attack_t
        self.pending_strength = strength if attack_t is not None else 0.0

    def _extend_pending(self, t: float, m: float, attack_t: float | None = None, strength: float = 0.0) -> None:
        self.pending_midis.append(m)
        self.pending_times.append(t)
        if attack_t is not None and self.pending_attack is None:
            self.pending_attack = attack_t
            self.pending_strength = strength
        keep = [i for i, tt in enumerate(self.pending_times) if t - tt <= MEMORY_S]
        if len(keep) < len(self.pending_times):
            self.pending_midis = [self.pending_midis[i] for i in keep]
            self.pending_times = [self.pending_times[i] for i in keep]

    def _hold_window(self, t: float) -> list[tuple[float, float]] | None:
        """The last `confirm_s` of frames when the pitch held there, else None.

        This is the live form of "a note is a stable pitch": a glide keeps
        moving, so it never fills the window — until it arrives, where the
        note is confirmed at last (with its onset dated by the attack).

        Two traps, both measured. The hold must SPAN `confirm_s`, not merely
        fit inside it: a window holding the last frames of a glide satisfied
        the spread test 40 ms after the note began, and confirmed a note on
        every second frame of a 0.4 s portamento (thirteen notes for one
        sung note). And it needs three frames or more, because live pitches
        are not cleaned — two frames of a ladder can sit within `HOLD_ST` by
        accident.
        """
        window = [(tt, mm) for tt, mm in zip(self.pending_times, self.pending_midis) if t - tt <= self.confirm_s]
        if len(window) < 3:
            return None
        hop = (window[-1][0] - window[0][0]) / (len(window) - 1)
        if t - window[0][0] < self.confirm_s - hop:
            return None  # the window is not full yet: the pitch has not held
        pitches = [mm for _, mm in window]
        if max(pitches) - min(pitches) > HOLD_ST:
            return None
        return window

    def _reset_pending(self) -> None:
        self.state = "IDLE"
        self.pending_start = None
        self.pending_midis = []
        self.pending_times = []
        self.pending_attack = None

    def _close_note(self, end_t: float) -> None:
        if self.note_start is None or not self.note_midis:
            self._reset_pending()
            return
        start = self.note_start
        midi = int(round(float(np.median(self.note_midis))))
        # only piano keys make it into the take (A0..C8): a pitch outside is
        # an octave error or a breath, not a note the singer produced
        if end_t - start >= self.min_note_s and on_piano(midi):
            self.notes.append(
                LiveNote(
                    start=start, end=end_t, midi=midi,
                    articulated=self.note_articulated, strength=self.note_strength,
                )
            )
        self.note_start = None
        self.note_midis = []
        self.note_articulated = False
        self.note_strength = 0.0
        self._reset_pending()

    # ------------------------------------------------------------------
    def active_note(self, t: float) -> LiveNote | None:
        """The note being sung right now, as a preview (never stored).

        `notes` only grows when a note ends, so the live display needs this
        to draw a note while it is still being held. The returned object is
        a throwaway: mutating it does not touch the session.
        """
        if self.state != "ACTIVE" or self.note_start is None or not self.note_midis:
            return None
        midi = int(round(float(np.median(self.note_midis))))
        if not on_piano(midi):
            return None
        return LiveNote(
            start=self.note_start,
            end=max(t, self.note_start + 0.05),
            midi=midi,
        )

    # ------------------------------------------------------------------
    def current(self) -> tuple[float, float] | None:
        """(float midi, hz) of the current note, or None."""
        if self.state == "ACTIVE" and self.note_midis:
            return float(np.median(self.note_midis)), self.last_hz
        if self.state == "PENDING" and self.pending_midis:
            return float(np.median(self.pending_midis)), self.last_hz
        return None

    def recent_midi(self) -> float | None:
        """Median of the last smoothed frames: what is sung NOW.

        `current()` medians the WHOLE note, which is right for naming it but
        wrong for practicing intonation: correct the pitch mid-note and the
        needle would barely move (the correction drowns in the average).
        The last CLEAN_FRAMES frames answer "where am I at this instant"
        with the same smoothing the segmenter trusts.
        """
        if not self.recent_midis:
            return None
        return float(np.median(self.recent_midis[-CLEAN_FRAMES:]))

    def display_text(self) -> str:
        cur = self.current()
        if cur is None or self.last_hz != self.last_hz:
            return "—"
        name = midi_to_name(int(round(cur[0])))
        return f"{name}  {self.last_hz:.1f} Hz"

    def flush(self, t: float) -> None:
        """Force-close the current note and merge fragments.

        Merges consecutive same-pitch notes separated by a SHORT gap
        (<= release_s) when one side is a fragment, or when the gap is a
        sub-perceptual stutter (the 40 ms parasite case). Two full-length
        notes separated by a real gap are TWO notes — the singer came back
        to the same pitch on purpose, and without a seen attack the old
        any-gap merge glued every soft repetition into one long note (the
        "ta-ta-ta-ta" and aa-aa-aa cases, measured). A gap of a note's
        length is a phrase boundary, not a blip.
        """
        if self.state == "ACTIVE":
            self._close_note(self.last_voiced_t if self.last_voiced_t is not None else t)
        merged: list[LiveNote] = []
        for n in self.notes:
            # an articulated note is a note of its own, and so is one the
            # singer came back to after a real gap: the same-pitch merge
            # below must not undo either decision
            gap = n.start - merged[-1].end if merged else 1.0
            if (
                merged
                and not n.articulated
                and n.midi == merged[-1].midi
                and gap <= self.release_s
                and (
                    n.duration < self.min_note_s
                    or merged[-1].duration < self.min_note_s
                    or gap < MIN_GAP_HOLD_S
                )
            ):
                merged[-1].end = n.end
                merged[-1].confidence = max(merged[-1].confidence, n.confidence)
                merged[-1].strength = max(merged[-1].strength, n.strength)
            else:
                merged.append(n)
        self.notes = merged
