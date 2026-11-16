"""Real-time segmentation of a pitch curve (on/off hysteresis).

Live has no look-ahead, so the offline rules (see notes.py: a legato glide is
not a note) are applied differently: the pitch only CONFIRMS a note once it
HOLDS, and the attack evidence (onsets.AttackTracker) pins the onset and marks
a re-articulation. Without attacks the segmenter still works, but a note starts
whenever its pitch settles — which is the end of the glide into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .music import hz_to_midi, midi_to_name, on_piano
from .onsets import AttackTracker

if TYPE_CHECKING:
    from .pitch import YinBlock

HOLD_ST = 0.6  # pitch spread (semitones) tolerated to call it held
MEMORY_S = 1.0  # seconds of frames kept in the pending window
MAX_ATTACKS = 64  # recent attack times kept for the display/tests
CLEAN_FRAMES = 5  # live pitch median (~115 ms): see LiveNoteSegmenter
MIN_GAP_HOLD_S = 0.045  # a mute run this long ENDS the note (re-take/gap)
# A re-articulation on ONE pitch shows in the PITCH STREAM itself: the
# consonant burst is noise, so the frames it covers come out NaN or as
# harmonics of the burst (measured, ta-ta on one tone: 1-3 frames of NaN
# or midi 36/66/83 at each re-attack, where a held vowel's frame line is
# unbroken at +-0.1 st). A voiced break inside an ACTIVE note, shorter
# than the release and coming back to the same pitch, is a re-attack:
# cut the note there. Level cannot see these (the burst block's rise is
# 1.28x, under STREAM_LEVEL); the pitch stream is the one witness that
# never misses them. Measured frame economics: 1-3 frames lost + the
# hysteresis + the 5-frame median = >= 170 ms of silence, so a gap under
# 0.06 s of CLEAN frames (6 frames) is a burst, not a gap.
SPLIT_FRAMES = 6  # voiced-gap ceiling for a same-pitch cut (frames)
HOLD_RAW_FRAMES = 5  # raw frames the hold test also inspects (see _hold_window)
# The frame period a segmenter built by HAND falls back on: the analysis
# rate's (256 / 22 050), the frame economics every measurement below was
# taken at. A stream always overwrites it with its OWN hop / sr (ADR 0009),
# so no literal rate survives on the live path.
SPLIT_FRAME_S = 256 / 22_050
# Live pitch window in TIME, not in samples (ADR 0009): the same DURATION at
# every capture rate. Expressed in samples instead, the production window was
# half the intended one — the device runs at 44 100 while the calibration ran
# at 22 050 — and read phantom low notes (measured on chant.wav: 30 notes
# against 26, two Fa2 among them). 23.22 / 11.61 ms are the measured and
# tested settings and they land EXACTLY on the calibrated sample counts:
# 512 / 256 at 22 050 Hz (harness, sweep, audit) and 1024 / 512 at 44 100 Hz.
WINDOW_MS = 23.22
HOP_MS = 11.61
# How close the voiced pitch must be to the plateau to count as the SAME
# note re-attacked. Measured on the harness: a re-attacked vowel comes back
# within 0.1 st of the plateau; the first voiced frame after a hand-over
# burst is already 0.5 st up the glide to the next note. A burst that fails
# this test belongs to the NEXT note — the pitch change closes the note when
# the glide leaves the band, and closing on the consonant would cut the
# glide's tail into a duplicate fragment (measured: grave-voice line, one
# extra fragment per hand-over).
REARTIC_ST = 0.3
# A burst's garbage frames sit FAR from the plateau (measured 19-28 st),
# while a real melodic move inside a note stays under ~2-3 st. Marking a
# break there is only a MARK: the split still refuses unless the voice
# RETURNS to the confirmed pitch (a real octave jump refuses the split and
# closes the note by the normal pitch-change path).
BREAKOUT_ST = 3.0
# Second witness of a re-articulation on ONE pitch: the LEVEL of the frame's
# own hop slice. A real re-emission interrupts the airflow: the seam dips
# to NEAR-SILENCE (measured 20x below the plateau) before the same plateau
# returns. Depth is the discriminator, not timing: an amplitude tremolo
# never dips below ~0.4x the MEDIAN level even at ±60 % depth (measured),
# so 0.35 keeps its dips unmarked while every real re-emission crosses it.
# (The median — not the peak — is the reference, so a 3-hop dip cannot
# lower it.) The mark alone still proves nothing: the split fires only on
# the return to the SAME confirmed pitch within the SPLIT_FRAMES ceiling,
# which is what keeps a swelled vibrato (its AM dips last a full cycle,
# ~180 ms > the ceiling) from splitting every note.
WEAK_RATIO = 0.35
WEAK_PEEK_SLICES = 16  # recent level window for the median reference (~185 ms)
# A staged attack is consumed by the first voiced frame that follows it ONLY
# within this horizon: a consonant immediately precedes the vowel it belongs
# to (measured 12-46 ms: the pitch frame lags the burst by window + gate),
# while a breath's attack can sit 0.2 s before the phrase it precedes —
# dating the note at the breath would pull every phrase start early.
STAGE_MAX_S = 0.10
# A live note's plateau periodicity, measured by YIN's own score (0 = perfect
# period, 1 = noise): frames enter on `MIN_PITCH_QUALITY` 0.35, so a median
# above 0.6 x that bound is a note that passed by a thin margin. Publishing
# it costs nothing and never gates a frame — the offline `VOICE_DOUBT` band,
# expressed on the live scale.
LIVE_DOUBT_PERIODICITY = 0.21


@dataclass
class LiveNote:
    start: float
    end: float
    midi: int
    confidence: float = 0.9
    figure: str | None = None
    articulated: bool = False  # the singer re-attacked: a note of its own
    strength: float = 0.0  # attack strength 0..1, take-relative: accents
    # signed residual pitch deviation (st) of the confirmed plateau to its
    # semitone: the live twin of Note.deviation, for the same key-aware
    # pull when a whole take is displayed/explained through the live path.
    # The pull reads it and never rewrites it (the live "take so far"
    # intonation line reads the same number).
    dev: float = 0.0
    # semitones the key-aware pull moved this note's label (0.0 = none):
    # the live twin of Note.snap_st
    snap_st: float = 0.0
    # the reading's own doubt, the live twin of `Note.voice_doubt`: this
    # note's plateau rested on barely periodic frames. Published (a dashed
    # outline in the roll), never a reason to drop or move the note.
    voice_doubt: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


class LivePitchStream:
    """Captured blocks -> pitch frames -> notes, with the attack evidence.

    Both front-ends (the GUI thread and the CLI session) need exactly this
    chain: YIN on every block, the silence gate, the streaming attack tracker
    and the segmenter. It lives here so the two live paths cannot drift
    apart — they had already started to.

    The pitch window is SHORT on purpose (~23 ms): the offline path can afford
    2048 samples, the live display cannot — measured on a synthetic voice,
    eighth notes at 150 BPM came out as 3/8 with a 2048-sample window (the
    tracker averaged two notes into one) and 7/8 with the short one. Short
    windows cost more frames per second but each is 16x cheaper, and the low
    end holds (a 55 Hz tone is read the same at every window).

    It is expressed in TIME, never in samples (ADR 0009): the same DURATION
    at every capture rate, so the settings measured at 22 050 Hz still hold
    on a 44 100 Hz device. `window_ms` / `hop_ms` are converted here and the
    resulting frame period is published to the segmenter, which reads its
    frame-based ceilings from it — no literal rate is left on this path.
    """

    def __init__(
        self,
        sr: int,
        gate_rms: float,
        window_ms: float = WINDOW_MS,
        hop_ms: float = HOP_MS,
    ):
        self.sr = sr
        self.gate_rms = gate_rms
        # 512 / 256 samples at 22 050 Hz, 1024 / 512 at 44 100 Hz: the same
        # duration, converted by the ONE place that knows the rate
        self.window = max(64, int(round(sr * window_ms / 1000.0)))
        self.hop = max(64, int(round(sr * hop_ms / 1000.0)))
        self.frame_s = self.hop / sr
        self.yin: "YinBlock | None" = None
        self.attacks = AttackTracker()
        self.seg = LiveNoteSegmenter(frame_s=self.frame_s)
        self._attack_pending = False
        self._pending_strength = 0.0
        self._recent_levels: list[float] = []

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
        # The gate is ONE policy for the displayed level and the voice
        # evidence: a block under `gate_rms` is silence for BOTH (the level
        # meter draws it as silence, and the finder/live pages hold their
        # pitch readout on the same bound). The pitch must disappear with
        # it: a periodic E4 whose blocks peak at 0.001 against the 0.008
        # threshold reads as a whole confirmed note while its frames keep
        # their height — a sub-threshold rumble is acoustically silence.
        below_gate = rms < self.gate_rms
        if self.attacks.push(t, x):
            self._attack_pending = True
            self._pending_strength = self.attacks.strength
        frames: list[tuple[float, float]] = []
        for ft, hz, quality in yin.push_with_quality(block):
            # a sub-gate frame is pushed as UNVOICED, never dropped: the
            # frame keeps the segmenter's clock and its mute-run rules,
            # only the pitch evidence disappears
            if below_gate:
                hz = float("nan")
            # per-hop level of THIS frame's own slice: a re-emission's
            # audible seam dip (WEAK_RATIO below the recent plateau)
            # straddles capture blocks too coarse for the block-level
            # tracker to see — the hop slice is the finer grain the seam
            # lives at (~11.6 ms vs the 46 ms block).
            weak = False
            if hz == hz and hz > 0:
                # absolute sample of the frame center, minus the block's own
                # start (t is the block's END time) — slicing x, not the
                # trimmed internal buffer, keeps the mapping trivial
                center = int(round(ft * self.sr))
                block_start = int(round(t * self.sr)) - len(x)
                i0 = center - self.window // 2 - block_start
                lo, hi = max(0, i0), min(len(x), i0 + self.hop)
                if hi > lo:
                    lvl = float(np.sqrt(np.mean(x[lo:hi] ** 2)))
                    self._recent_levels.append(lvl)
                    del self._recent_levels[:-WEAK_PEEK_SLICES]
                    if len(self._recent_levels) >= 3:
                        median = float(np.median(self._recent_levels))
                        weak = lvl < WEAK_RATIO * median
            self.seg.push(
                ft,
                hz,
                attack=self._attack_pending,
                strength=self._pending_strength,
                weak=weak,
                quality=quality,
            )
            self._attack_pending = False
            self._pending_strength = 0.0
            frames.append((ft, hz))
        return rms, frames

    def elapsed_notes(self) -> int:
        return len(self.seg.notes)


class LiveNoteSegmenter:
    """State machine: IDLE -> PENDING -> ACTIVE.

    - A note is only confirmed ("on") when its pitch has HELD for
      `confirm_s` (0.09 s: swept 0.14 -> 0.10 -> 0.09 over the truth cases
      — zero behavior change, ~35 ms of display latency saved per step);
      the onset is then the attack that started it, or the moment it
      settled. This is what keeps a glide from becoming a note.
    - It ends ("off") on a new attack, on a pitch change of more than
      `semitones`, or after `MIN_GAP_HOLD_S` without voicing — so a blip or
      a spurious frame does not split or invent a note.
    """

    def __init__(
        self,
        confirm_s: float = 0.09,
        release_s: float = 0.12,
        min_note_s: float = 0.10,
        semitones: float = 0.6,
        frame_s: float = SPLIT_FRAME_S,
    ):
        self.confirm_s = confirm_s
        self.release_s = release_s
        self.min_note_s = min_note_s
        self.semitones = semitones
        # the length of one frame, in seconds: the ceiling below is written
        # in FRAMES, so it must read the period of the stream that feeds it
        # (LivePitchStream publishes hop / sr; ADR 0009). The default is the
        # analysis rate's, for a segmenter built by hand in a test.
        self.frame_s = frame_s

        self.state = "IDLE"
        self.last_frame_t: float | None = None  # ANY frame (voiced or not): the now
        self.pending_start: float | None = None
        self.pending_midis: list[float] = []
        self.pending_times: list[float] = []
        self.pending_attack: float | None = None
        self.note_start: float | None = None
        self.note_midis: list[float] = []
        self.note_periodicity: list[float] = []  # YIN's score for those frames
        self.note_pitch: float | None = None  # the CONFIRMED plateau: re-attack reference
        self.last_voiced_t: float | None = None
        self.last_hz: float = float("nan")
        self.notes: list[LiveNote] = []
        self.attacks: list[float] = []
        self.note_articulated = False
        self.note_strength = 0.0
        self.pending_strength = 0.0
        self.recent_midis: list[float] = []
        self.recent_raw: list[float] = []  # pre-cleaning frames, for the hold test
        self._voiced_break = False  # mute frames seen INSIDE the current note
        self._break_cut_t: float | None = None  # last REAL frame before the break
        self._staged_attack: float | None = None  # attack seen on a mute frame
        self._staged_strength = 0.0  # its strength, same sidecar

    # ------------------------------------------------------------------
    def push(
        self,
        t: float,
        hz: float,
        attack: bool = False,
        strength: float = 0.0,
        weak: bool = False,
        quality: float = float("nan"),
    ) -> str:
        """Advance one frame. Return 'on', 'off' or ''.

        `self.last_frame_t` is stamped HERE, for every frame (mute ones
        included): it is the segmenter's own "now", the freshness clock
        of `heard()`.

        `attack` is the live attack evidence (onsets.AttackTracker); it is
        what makes a re-articulated note separable from a held one, and what
        dates the onset at the consonant instead of at the moment the pitch
        settles. `strength` is that attack's level rise (0..1): it travels
        with the attack flag and lands on the closed note as
        `LiveNote.strength` — the accent evidence `meter.detect_meter`
        reads, the same field `Note.strength` carries offline.

        `weak` marks a frame whose own level dropped far below the recent
        plateau (the seam of a re-emitted note, see WEAK_RATIO): it is the
        second re-articulation witness, for the seams the level tracker
        misses because the dip straddled two capture blocks.
        """
        self.last_frame_t = t
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
            elif attack and self.state == "IDLE":
                # The flag also lands on mute frames BEFORE the note is even
                # pending (the tracker fires at block time, the first pitch
                # frame of the block is still in the gap): without staging,
                # the accent evidence died here and a whole melody read
                # strength 0.0 — meter blind. The next voiced frame consumes
                # it (within STAGE_MAX_S) and dates the note at the consonant.
                self._staged_attack = t
                self._staged_strength = strength
            # remember the break: when voicing returns on the SAME pitch
            # after a SHORT mute run, it is a re-articulation, not a hold
            # (see SPLIT_FRAMES)
            if self.state == "ACTIVE":
                if not self._voiced_break:
                    self._break_cut_t = self.last_voiced_t
                self._voiced_break = True
            # the mute run cuts a PENDING window: whatever was in it stopped
            # sounding. A hand-over's pending that keeps its pre-burst frames
            # confirms a fragment dated at the OLD pitch (the cleaned median
            # lags the burst), duplicating the previous note (measured: one
            # fragment per hand-over of the voice-accuracy grave/haute
            # lines). After the burst the window refills from the return. An
            # attack the window was opened WITH — a same-pitch split's cut,
            # or the consonant that fired while pending — must survive the
            # cut: it is staged here and dates the refilled window (dropping
            # it un-dated the re-attacked note and the flush merge glued it
            # back into the held note: ta-ta-ta-ta read as three).
            elif self.state == "PENDING":
                if self.pending_attack is not None:
                    self._staged_attack = self.pending_attack
                    self._staged_strength = self.pending_strength
                elif attack:
                    # a fresh attack on the mute frame that cuts the window:
                    # fresher than nothing, same articulation
                    self._staged_attack = t
                    self._staged_strength = strength
                self._reset_pending()
            if self.state == "ACTIVE" and self.last_voiced_t is not None:
                if t - self.last_voiced_t >= MIN_GAP_HOLD_S:
                    # a mute run this long is a re-take (or a phrase gap),
                    # not a frame dropout: the note ends at its last voiced
                    # frame. Without this, a SOFT re-phonation of the SAME
                    # pitch — no pitch change, sub-threshold level dip — was
                    # absorbed into one endless note (aa-aa-aa case).
                    self._close_note(self.last_voiced_t)
                    ev = "off"
            return ev

        self.last_hz = hz
        raw_midi = float(hz_to_midi(hz))
        m = self._clean(raw_midi)

        if (
            attack
            and self.state == "ACTIVE"
            and self.note_start is not None
            and t - self.note_start >= self.min_note_s
        ):
            # The singer articulated again: a new note begins here. A voiced
            # consonant carries NO mute frames, so `_voiced_break` is False
            # and the cut is direct (the classic ta-aa case, test
            # `test_re_attack_splits_a_held_note`); a burst consonant breaks
            # the pitch stream first, and the cut is the split path below
            # (with its guards: the return must be the note's own pitch —
            # see REARTIC_ST — so a hand-over's glide to the next note is
            # refused there and the pitch change closes the note instead).
            # The RAW pitch, not the cleaned median: the 5-frame median still
            # shows the plateau while the glide has already left it.
            if self._voiced_break:
                if self._split_same_pitch(t, raw_midi, strength):
                    ev = "off"
                else:
                    # The split refused: the attack landed ON the burst's own
                    # garbage frame (measured: the consonant's harmonic, midi
                    # 36 against a 64 plateau, far outside REARTIC_ST). The
                    # guard is right — the garbage frame proves nothing — but
                    # the attack is real evidence and was CONSUMED here: the
                    # next plausible frame saw no attack, disarmed the break
                    # and the note continued as one hold (the user's "same
                    # note repeated read as one held note"; 4 sung ta-ta -> 3
                    # notes, the middle two glued). Staging keeps the flag for
                    # the next voiced frame, which either splits the note
                    # (repetition) or closes it at the consonant (hand-over —
                    # the honest onset date either way).
                    self._staged_attack = t
                    self._staged_strength = strength
            else:
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
        if self._staged_attack is not None and self.state == "IDLE":
            # a same-pitch split's repetition, refilled after the burst cut
            # its pending window: the staged attack still dates the note the
            # singer is opening. PENDING keeps its own attack; a state other
            # than IDLE/ACTIVE never sees a staged attack survive. A STALE
            # stage (a breath's attack 0.2 s before the phrase — beyond
            # STAGE_MAX_S) is dropped instead of pulling the note start back:
            # the frame's own attack, if any, dates it normally.
            if t - self._staged_attack <= STAGE_MAX_S:
                staged = self._staged_attack
                staged_strength = self._staged_strength
                self._staged_attack = None
                self._staged_strength = 0.0
                self._start_pending(t, m, staged, staged_strength)
            else:
                self._staged_attack = None
                self._staged_strength = 0.0
        elif self._staged_attack is not None and self.state != "ACTIVE":
            self._staged_attack = None
            self._staged_strength = 0.0

        if self.state == "ACTIVE":
            ref = float(np.median(self.note_midis))
            confirmed = self.note_pitch if self.note_pitch is not None else ref
            # a burst's garbage frames (the burst's own harmonics, measured
            # midi 83/36 against a 64 plateau) must mark a break BEFORE the
            # 5-frame median absorbs them into the plateau — that absorption
            # is exactly what made the no-gap ta-ta read as one hold (the
            # glide_s=0 bursts never come out NaN, unlike the glide_s=0.03
            # ones). Marking only: the split's guards decide.
            if abs(raw_midi - confirmed) > BREAKOUT_ST and not self._voiced_break:
                self._break_cut_t = self.last_voiced_t
                self._voiced_break = True
            # the LEVEL witness (see WEAK_RATIO): an audible dip on the same
            # pitch marks a break exactly like a mute run, and must neither
            # append to the plateau nor disarm one — the split fires on the
            # strong same-pitch return, or on an attack flag landing there.
            if weak:
                if not self._voiced_break:
                    self._break_cut_t = self.last_voiced_t
                self._voiced_break = True
            elif abs(m - ref) <= self.semitones:
                # a short mute run mid-note that comes back to the SAME
                # pitch is the re-articulation the level cannot see (the
                # burst is noise: 1-3 NaN/garbage frames, then the same
                # plateau — see SPLIT_FRAMES)
                if self._voiced_break and self._split_same_pitch(t, raw_midi, 0.0):
                    ev = "off"
                else:
                    self.note_midis.append(m)
                    if quality == quality:  # not NaN: the evidence exists
                        self.note_periodicity.append(quality)
                    # a GARBAGE frame must not clear the break it marked:
                    # the else branch ran right after the marking above and
                    # disarmed it in the same frame (measured: the no-gap
                    # ta-ta burst 2 kept merging). Only a plausible frame —
                    # its pitch near the plateau — ends the break.
                    if abs(raw_midi - confirmed) <= BREAKOUT_ST:
                        self._voiced_break = False
                        self._break_cut_t = None
            else:
                self._close_note(t)
                ev = "off"
                self._start_pending(t, m)
                self._voiced_break = False
                self._break_cut_t = None
        elif self.state == "IDLE":
            # a weak frame is sub-plateau sound: it does not OPEN a note (a
            # soft onset only starts the pending once the level is real); the
            # confirmation test would reject its window anyway.
            if not weak:
                self._start_pending(t, m, t if attack else None, strength)
        elif self.state == "PENDING":
            # a glide is FOLLOWED, never restarted: the pending window
            # keeps the ladder of plateaus and only closes when the pitch
            # holds, which is where the note really is.
            self._extend_pending(t, m, t if attack else None, strength)
            window = self._hold_window(t)
            if window is not None:
                self.state = "ACTIVE"
                # onset: the consonant when there was one, else the moment
                # the pitch settled. The attack may PREDATE the window: a
                # glide's consonant fires at its start and the hold is only
                # confirmed at the arrival (measured: onset dated 0.42 s
                # late when clamped to window[0][0]), and a same-pitch split
                # dates the repetition at the break — the honest start of
                # the re-attacked note.
                self.note_start = (
                    self.pending_attack if self.pending_attack is not None else window[0][0]
                )
                self.note_articulated = self.pending_attack is not None
                self.note_strength = self.pending_strength
                self.note_midis = [mm for _, mm in window]
                # the pending window's frames are not kept: a doubt describes
                # the CONFIRMED plateau, the same one the pitch reads
                self.note_periodicity = []
                # the CONFIRMED pitch: the plateau the note was accepted at.
                # The running median below keeps absorbing glide frames
                # (hysteresis), so re-attack references read THIS value: a
                # re-articulation returns to the pitch the note started on,
                # a hand-over's return is the NEXT note's pitch.
                self.note_pitch = float(np.median([mm for _, mm in window]))
                ev = "on"
        self.last_voiced_t = t
        return ev

    # ------------------------------------------------------------------
    def _split_same_pitch(self, t: float, raw: float, strength: float = 0.0) -> bool:
        """Cut an ACTIVE same-pitch note across a short mute run.

        The consonant of a re-attacked note is noise: YIN comes back NaN or
        with the burst's own harmonics (midi 36/66/83 measured) for 1-3
        frames before the vowel settles back on the SAME pitch. Cutting
        there is the live twin of `_split_rearticulated` (offline, attack
        evidence): the pitch stream is the one witness that never misses a
        same-pitch re-attack, because the level rise sits under STREAM_LEVEL
        (measured 1.28x).

        The guards, all measured: the voiced pitch must RETURN to the pitch
        the note was CONFIRMED at (REARTIC_ST against `note_pitch` — the
        running median keeps absorbing glide frames, so a hand-over's return
        read 0.2 st off a polluted reference and duplicated a fragment; the
        confirmed pitch of the previous note is 1 st away and refuses); the
        break must be SHORT (under SPLIT_FRAMES of frames — a real re-take
        is caught by MIN_GAP_HOLD_S in the mute branch instead); and the held
        half must be a real note (>= min_note_s). Otherwise the frames simply
        continue the note.
        """
        if self.note_start is None or not self.note_midis:
            return False
        cut = self._break_cut_t if self._break_cut_t is not None else self.last_voiced_t
        ref = self.note_pitch if self.note_pitch is not None else float(np.median(self.note_midis))
        if cut is None or abs(raw - ref) > REARTIC_ST:
            return False
        if t - cut >= SPLIT_FRAMES * self.frame_s or cut - self.note_start < self.min_note_s:
            return False
        # close the held note at the break and open the repetition; the
        # pending window needs `confirm_s` again, which is the honest cost of
        # a display confirmed AFTER the split (the note was already sounding).
        # The repetition is dated as an ATTACK at the break (attack_t=cut):
        # `_voiced_break` is the articulation evidence here, and without the
        # mark the flush merge glued the halves back together (their 0.03 s
        # seam sits under MIN_GAP_HOLD_S, the "sub-perceptual stutter" rule —
        # a rule for noise gaps, not for a cut the segmenter itself decided).
        self._close_note(cut)
        self._start_pending(t, raw, attack_t=cut, strength=strength)
        self._voiced_break = False
        self._break_cut_t = None
        return True

    def _clean(self, m: float) -> float:
        """Median of the last few live frames (the live pitch is not cleaned).

        Offline, `notes.clean_pitch` runs a 9-frame median before segmenting;
        live there is no such pass, so a vibrato keeps its full oscillation
        and `HOLD_ST` has to be wide enough to accept it — which a slow slide
        can then slip through. A 5-frame median (~115 ms) removes most of the
        oscillation while preserving the note transitions (a median does not
        blur a hard edge, unlike a moving average). The RAW frames are kept
        beside the median: `_hold_window` needs them (a median rides a ramp).
        """
        self.recent_raw.append(m)
        if len(self.recent_raw) > HOLD_RAW_FRAMES:
            del self.recent_raw[0]
        self.recent_midis.append(m)
        if len(self.recent_midis) > CLEAN_FRAMES:
            del self.recent_midis[0]
        return float(np.median(self.recent_midis))

    def _start_pending(
        self, t: float, m: float, attack_t: float | None = None, strength: float = 0.0
    ) -> None:
        self.state = "PENDING"
        self.pending_start = t
        self.pending_midis = [m]
        self.pending_times = [t]
        self.pending_attack = attack_t
        self.pending_strength = strength if attack_t is not None else 0.0

    def _extend_pending(
        self, t: float, m: float, attack_t: float | None = None, strength: float = 0.0
    ) -> None:
        # Only frames inside the CURRENT confirm window are kept (the trim in
        # the body also drops frames older than confirm_s): a pending that
        # started on a late-dated onset is still full of the PREVIOUS note's
        # plateau frames (measured: the hand-over attack closed the note 20 ms
        # before its pitch left the band, so 3 frames of the old plateau
        # entered the window) and the spread test read the whole window as
        # held from t0 — confirming a start BEFORE the new pitch even began.
        # This trim makes `_hold_window`'s span test honest: the window must
        # fill with frames sung since the onset.
        keep = [
            i
            for i, tt in enumerate(self.pending_times)
            if t - tt <= MEMORY_S and tt >= t - self.confirm_s
        ]
        self.pending_midis = [self.pending_midis[i] for i in keep]
        self.pending_times = [self.pending_times[i] for i in keep]
        self.pending_midis.append(m)
        self.pending_times.append(t)
        if attack_t is not None and self.pending_attack is None:
            self.pending_attack = attack_t
            self.pending_strength = strength
        elif attack_t is not None and strength > self.pending_strength:
            # a fresh attack while the window already carries one: SAME
            # articulation seen twice (the seam cut dates the note, the
            # tracker's consonant burst fires ~70 ms later). The date stays
            # (earliest evidence) but the accent takes the stronger reading,
            # else the split path confirmed notes with the staged 0.0 while
            # the real consonant's strength arrived one frame too late
            # (measured: note 3 of a full-consonant ta-ta read 0.0).
            self.pending_strength = strength
        keep = [
            i
            for i, tt in enumerate(self.pending_times)
            if t - tt <= MEMORY_S and tt >= t - self.confirm_s
        ]
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
        window = [
            (tt, mm)
            for tt, mm in zip(self.pending_times, self.pending_midis, strict=False)
            if t - tt <= self.confirm_s
        ]
        if len(window) < 3:
            return None
        hop = (window[-1][0] - window[0][0]) / (len(window) - 1)
        if t - window[0][0] < self.confirm_s - hop:
            return None  # the window is not full yet: the pitch has not held
        pitches = [mm for _, mm in window]
        if max(pitches) - min(pitches) > HOLD_ST:
            return None
        # a STEADY MEDIAN is not a held pitch: the 5-frame median rides a
        # slow glide's ramp (measured on the hand-over 47->48: the median
        # window held 47.0..47.9 while every raw frame moved up one step),
        # so the spread test needs the RAW frames too. Without this the
        # ramp confirmed a note dated back at the previous pitch — a
        # duplicate fragment of the note that just ended.
        raws = self.recent_raw[-HOLD_RAW_FRAMES:]
        raw_spread = max(raws) - min(raws)
        if len(self.recent_raw) >= HOLD_RAW_FRAMES and raw_spread > HOLD_ST:
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
        reference = self.note_pitch
        if reference is None:
            reference = float(np.median(self.note_midis))
        midi = int(round(reference))
        # only piano keys make it into the take (A0..C8): a pitch outside is
        # an octave error or a breath, not a note the singer produced
        if end_t - start >= self.min_note_s and on_piano(midi):
            plateau = (
                self.note_pitch
                if self.note_pitch is not None
                else float(np.median(self.note_midis))
            )
            # the notes list stays ORDERED and non-overlapping: the previous
            # note's end is clamped to this start (measured on a real take:
            # a split dated its repetition 20 ms BEFORE the previous note's
            # end — the MIDI file still sorted, the JSON/CSV showed a note
            # ending after its successor started). Zero-length predecessors
            # are dropped.
            if self.notes and self.notes[-1].end > start:
                if start - self.notes[-1].start >= self.min_note_s:
                    self.notes[-1].end = start
                else:
                    self.notes.pop()
            periodicity = (
                float(np.median(self.note_periodicity)) if self.note_periodicity else float("nan")
            )
            self.notes.append(
                LiveNote(
                    start=start,
                    end=end_t,
                    midi=midi,
                    articulated=self.note_articulated,
                    strength=self.note_strength,
                    dev=max(-0.5, min(0.5, plateau - round(plateau))),
                    # PUBLISHED doubt (never a mask): the note keeps its bar
                    # and its place, the roll only says the evidence was thin
                    voice_doubt=bool(
                        periodicity == periodicity and periodicity > LIVE_DOUBT_PERIODICITY
                    ),
                )
            )
        self.note_start = None
        self.note_midis = []
        self.note_periodicity = []
        self.note_pitch = None
        self.note_articulated = False
        self.note_strength = 0.0
        self._voiced_break = False
        self._break_cut_t = None
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
        """(float midi, hz) of the current note, or None.

        Answers during PENDING too: that is the live feedback a singer watches
        while a note is still being confirmed. A display that claims "this IS
        a note" must read `confirmed()` instead.
        """
        if self.state == "ACTIVE" and self.note_midis:
            return float(np.median(self.note_midis)), self.last_hz
        if self.state == "PENDING" and self.pending_midis:
            return float(np.median(self.pending_midis)), self.last_hz
        return None

    def confirmed(self) -> tuple[float, float] | None:
        """(float midi, hz) of the CONFIRMED note only, or None.

        The twin of `current()` for the displays that name a note rather than
        accompany a singer: a PENDING window has not held `confirm_s` yet, and
        a microphone rumble slips a few frames past the periodicity gate
        without ever confirming. Measured on a synthetic handling noise (a
        low-frequency thump, five durations from 50 to 400 ms): ZERO notes and
        ZERO ACTIVE blocks, while `current()` returned a pitch for 1-5 blocks -
        which is exactly the key the note finder highlights.
        """
        if self.state != "ACTIVE" or self.note_pitch is None:
            return None
        midi = int(round(self.note_pitch))
        if not on_piano(midi):
            return None
        return float(midi), self.last_hz

    def heard(self, max_age_s: float = 0.5) -> tuple[float, float] | None:
        """(float midi, hz) of the RAW pitch last seen, or None when stale.

        The instant, PRE-cleaning reading — one YIN frame, no median, no
        state machine — for displays that must answer "what is the mic
        hearing RIGHT NOW" even when the segmenter holds nothing (a
        glide between notes, a wavering beginner tone, a quiet voice
        whose frames the hysteresis has not accepted yet). Median and
        confirmation are DELIBERATE here: this is the mic's ear, not the
        take's notes — it must not invent a note, and a stale reading
        (older than `max_age_s` against the segmenter's own frame clock,
        `last_frame_t` — mute frames included) returns None rather than
        repeat a pitch the voice left seconds ago.
        """
        if self.last_hz != self.last_hz or self.last_hz <= 0:
            return None
        if self.last_frame_t is None or self.last_voiced_t is None:
            return None
        if self.last_frame_t - self.last_voiced_t > max_age_s:
            return None
        midi = hz_to_midi(self.last_hz)
        return float(midi), self.last_hz

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
