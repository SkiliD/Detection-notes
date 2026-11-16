"""Real-time segmenter tests (hysteresis, attacks, transitions)."""

import numpy as np
import pytest

from v2m.core.liveseg import LiveNoteSegmenter
from v2m.core.music import midi_to_hz


def _hz(midi) -> float:
    """midi_to_hz returns a 0-d array; the segmenter expects a plain float."""
    return float(midi_to_hz(midi))


def _feed(seg: LiveNoteSegmenter, hz: float, start: float, dur: float, step: float = 0.01):
    t = start
    while t < start + dur:
        seg.push(t, hz)
        t += step


def test_note_confirmed_and_closed():
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.30)
    seg.flush(0.35)
    assert len(seg.notes) == 1
    assert seg.notes[0].midi == 69
    assert seg.notes[0].duration >= 0.25


def test_two_separate_notes():
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.30)
    _feed(seg, float("nan"), 0.30, 0.20)  # silence
    _feed(seg, _hz(72), 0.50, 0.30)
    seg.flush(0.85)
    assert len(seg.notes) == 2
    assert [n.midi for n in seg.notes] == [69, 72]


def test_attack_strength_lands_on_the_closed_note():
    """The attack strength travels with the flag and lands on the note.

    `LiveNote` must carry a `strength` field: `detect_meter` (the 3/4 vs
    4/4 accents stage) reads it on the first live take.
    The strength (0..1, normalized on the take's strongest attack) is
    the live counterpart of the offline `Note.strength`.
    """
    seg = LiveNoteSegmenter()
    seg.push(0.0, _hz(69), attack=True, strength=1.0)
    t = 0.01
    while t < 0.30:
        seg.push(t, _hz(69))
        t += 0.01
    seg.flush(0.35)
    assert len(seg.notes) == 1
    assert seg.notes[0].strength == pytest.approx(1.0)
    # a second note opened WITHOUT a seen attack carries no accent
    seg.push(0.40, _hz(69))
    t = 0.41
    while t < 0.70:
        seg.push(t, _hz(69))
        t += 0.01
    seg.flush(0.75)
    assert len(seg.notes) == 2
    assert seg.notes[1].strength == 0.0


def test_staged_attack_keeps_its_strength():
    """An attack on a mute frame keeps its strength until the note.

    The staged attack carries (time, strength) TOGETHER: reading the
    strength from the next voiced frame's parameter reads 0.0 there. On
    a real sung take (consonants on mute frames), every
    downbeat read 0.0 and meter detection lost its accents.
    """
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(64), 0.0, 0.40)
    seg.push(0.41, float("nan"), attack=True, strength=1.0)  # consonant, mute
    _feed(seg, _hz(64), 0.45, 0.40)  # the re-attacked note
    seg.flush(0.90)
    assert len(seg.notes) == 2
    assert seg.notes[1].articulated
    assert seg.notes[1].strength == pytest.approx(1.0)


def test_short_blip_ignored():
    """A 40 ms blip does not become a note."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.40)
    _feed(seg, _hz(81), 0.40, 0.04)  # blip
    _feed(seg, _hz(69), 0.44, 0.30)
    seg.flush(0.80)
    assert len(seg.notes) == 1
    assert seg.notes[0].midi == 69


def test_repetition_same_pitch_stays_two_notes():
    """Repeating the same note (aa-aa) with no attack seen: TWO notes.

    The flush merge must refuse two full-length notes separated by a
    real silence (60 ms) whenever the attack has not been detected — a
    soft repetition is a new note, not a longer hold.
    """
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.30)
    _feed(seg, float("nan"), 0.30, 0.06)  # same note taken again
    _feed(seg, _hz(69), 0.36, 0.30)
    seg.flush(0.70)
    assert len(seg.notes) == 2
    assert [n.duration for n in seg.notes] == pytest.approx([0.30, 0.30], abs=0.05)


def test_repetition_after_longer_gap_stays_two_notes():
    """A silence of a note's length between two repetitions is a phrase."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.30)
    _feed(seg, float("nan"), 0.30, 0.25)
    _feed(seg, _hz(69), 0.55, 0.30)
    seg.flush(0.90)
    assert len(seg.notes) == 2


def test_out_of_piano_notes_are_never_kept():
    """A piano only has 88 keys (A0..C8): an out-of-range pitch enters
    neither the take nor the preview — the octave error stays visible, the
    note does not exist."""
    from v2m.core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI

    seg = LiveNoteSegmenter()
    _feed(seg, _hz(PIANO_MIN_MIDI - 3), 0.0, 0.30)  # below A0
    _feed(seg, float("nan"), 0.30, 0.10)
    _feed(seg, _hz(PIANO_MAX_MIDI + 5), 0.40, 0.30)  # above C8
    _feed(seg, float("nan"), 0.70, 0.10)
    _feed(seg, _hz(PIANO_MIN_MIDI), 0.80, 0.30)  # A0, the extreme key
    seg.flush(1.15)
    assert [n.midi for n in seg.notes] == [PIANO_MIN_MIDI]

    seg2 = LiveNoteSegmenter()
    _feed(seg2, _hz(PIANO_MAX_MIDI + 5), 0.0, 0.30)
    assert seg2.active_note(0.25) is None


def test_active_note_shows_the_note_being_sung():
    """While a note is held, the preview describes it without storing it."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.20)
    assert seg.notes == []  # nothing is stored before the note ends
    prev = seg.active_note(0.25)
    assert prev is not None
    assert prev.midi == 69
    assert prev.start == pytest.approx(0.0, abs=0.05)
    assert prev.end == pytest.approx(0.25)
    # still not stored: the preview is disposable
    assert seg.notes == []


def test_active_note_none_when_idle():
    seg = LiveNoteSegmenter()
    assert seg.active_note(0.1) is None
    _feed(seg, float("nan"), 0.0, 0.20)
    assert seg.active_note(0.25) is None
    _feed(seg, _hz(60), 0.30, 0.30)
    seg.flush(0.65)
    assert seg.active_note(0.70) is None  # note closed -> nothing in progress


def test_active_note_end_never_before_start():
    """A late display tick must not produce an inverted bar."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.50, 0.20)
    prev = seg.active_note(0.10)
    assert prev is not None and prev.end > prev.start


def test_nan_ignored():
    seg = LiveNoteSegmenter()
    _feed(seg, float("nan"), 0.0, 0.30)
    seg.flush(0.30)
    assert seg.notes == []


# --------------------------------------------------------------------------- #
# Attacks and transitions (see notes.py for the same offline rules)
# --------------------------------------------------------------------------- #


def _feed_glide(
    seg,
    t0: float,
    glide_s: float,
    hold_s: float,
    start: int,
    target: int,
    step: float = 0.02,
    attack_first: bool = False,
):
    """Portamento from `start` to `target` over `glide_s`, then held."""
    n = int(glide_s / step)
    for i in range(n):
        w = (i + 1) / n
        attack = attack_first and i == 0
        seg.push(t0 + (i + 1) * step, _hz(start + w * (target - start)), attack=attack)
    t = t0 + glide_s
    while t < t0 + glide_s + hold_s:
        t += step
        seg.push(t, _hz(target))
    return t


def test_a_glide_does_not_become_a_note():
    """A portamento must be confirmed once, at its arrival only.

    With an 80 ms confirmation window (two frames), a scale step reads
    as a held note and a 60->62 glissando manufactures notes at 61 and
    63.

    Accepted limit: a portamento SLOWER than ~3 semitones per second
    (0.6 s for 2 semitones) transcribes as two notes, since no 140 ms
    window can then tell the glide from the hold.
    """
    seg = LiveNoteSegmenter()
    t = _feed_glide(seg, 0.0, 0.12, 0.6, 60, 62)
    seg.flush(t + 0.2)
    assert [n.midi for n in seg.notes] == [62]


def test_attack_dates_the_onset_of_a_note():
    """With the attack, the note starts at the portamento — not at its arrival."""
    without = LiveNoteSegmenter()
    t = _feed_glide(without, 0.0, 0.4, 0.4, 60, 67)
    without.flush(t + 0.2)

    with_attack = LiveNoteSegmenter()
    t = _feed_glide(with_attack, 0.0, 0.4, 0.4, 60, 67, attack_first=True)
    with_attack.flush(t + 0.2)

    assert [n.midi for n in with_attack.notes] == [67]
    assert with_attack.notes[0].start == pytest.approx(0.0, abs=0.03)
    # with no attack evidence the note can only be dated at the pitch's
    # arrival: that is the documented real-time fallback
    assert [n.midi for n in without.notes] == [67]
    assert without.notes[0].start > 0.35


def test_re_attack_splits_a_held_note():
    """Same pitch re-struck louder: two notes, not one hold."""
    seg = LiveNoteSegmenter()
    t = 0.0
    while t < 0.4:
        t += 0.02
        seg.push(t, _hz(64))
    seg.push(t + 0.02, _hz(64), attack=True)
    t += 0.02
    while t < 0.9:
        t += 0.02
        seg.push(t, _hz(64))
    seg.flush(1.0)
    assert [n.midi for n in seg.notes] == [64, 64]


def test_repeated_same_pitch_through_the_full_chain():
    """Four sung "ta" on one E must give FOUR notes (~0.25 s each).

    The level tracker's baseline must move with each attack (stuck at
    the previous notes' level it returns one attack in four), and the
    attacks landing on mute frames are carried — the flush merge
    then glued everything into one 1.1 s note. Full chain (1024-sample
    blocks, like the mic)."""
    from voicetruth import SungNote, synthesize

    from v2m.core.liveseg import LivePitchStream

    truth = [
        SungNote(64, 0.2, 0.25),
        SungNote(64, 0.5, 0.25),
        SungNote(64, 0.8, 0.25),
        SungNote(64, 1.1, 0.25),
    ]
    y = synthesize(truth, consonant_s=0.05, glide_s=0.03)
    stream = LivePitchStream(22_050, gate_rms=0.008)
    for i in range(0, len(y) - 1024, 1024):
        stream.feed(y[i : i + 1024], (i + 1024) / 22_050)
    stream.seg.flush(len(y) / 22_050)
    notes = stream.seg.notes
    assert [n.midi for n in notes] == [64, 64, 64, 64]
    assert all(n.duration < 0.45 for n in notes)  # neither one hold nor a merge


def test_split_attack_survives_a_pending_mute_run():
    """A split repetition's attack survives the mute PENDING reset.

    A same-pitch split (burst consonant) opens the PENDING repetition
    with `pending_attack` = the cut; when the burst extends over the
    following frames, they are MUTE and the mute PENDING reset erased
    the attack without staging it — the repetition then restarted dated
    at its confirmation (a measured 60 ms hole between the two halves)
    and with no articulation evidence, so the flush merge could glue it.
    """
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(64), 0.0, 0.40)  # held note (confirmed ~0.10)
    seg.push(0.41, float("nan"), attack=True)  # burst: the hold is broken
    seg.push(0.42, float("nan"))  # (silence < MIN_GAP_HOLD_S)
    seg.push(0.43, _hz(64), attack=True)  # voiced frame: the split cuts here
    seg.push(0.44, float("nan"))  # the burst's tail: PENDING mute
    seg.push(0.45, float("nan"))
    _feed(seg, _hz(64), 0.46, 0.40)  # the repetition's vowel
    seg.flush(0.90)
    assert [n.midi for n in seg.notes] == [64, 64]
    assert seg.notes[1].articulated
    assert seg.notes[1].start == pytest.approx(0.40, abs=0.02)
    assert seg.notes[1].start >= seg.notes[0].end - 0.02  # no hole


def test_staged_attack_dates_the_new_note():
    """An attack seen on a mute frame (an unvoiced consonant) must date
    the next note: otherwise `flush` merges the repetition as a blip."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(64), 0.0, 0.40)
    seg.push(0.41, float("nan"), attack=True)  # consonant on a mute frame
    _feed(seg, _hz(64), 0.45, 0.40)  # the re-attacked note
    seg.flush(0.90)
    assert len(seg.notes) == 2
    assert seg.notes[1].articulated
    assert seg.notes[1].start == pytest.approx(0.41, abs=0.05)


def test_a_fresh_note_is_not_split_by_a_late_attack():
    """A stray attack right after the onset does not cut the note"""
    seg = LiveNoteSegmenter()
    t = 0.0
    while t < 0.5:
        t += 0.02
        seg.push(t, _hz(60), attack=(t > 0.06 and t < 0.08))
    seg.flush(0.6)
    # exactly one frame carries the stray attack (~0.06 s, float stepping):
    # it lands inside the fresh note and must not cut it
    assert len(seg.notes) == 1, "the late attack cut the fresh note in two"
    assert seg.notes[0].midi == 60


# --------------------------------------------------------------------- #
# The constants' own teeth: each test below pins ONE constant's decision
# boundary at the frame level — the corpus nets (test_voice_accuracy)
# prove the whole chain, these prove the knob.
# --------------------------------------------------------------------- #


def test_weak_frames_do_not_open_a_note_and_their_level_witness_marks_breaks():
    """WEAK_RATIO (0.35): a frame far below the recent level median.

    Two decisions hang on it: a WEAK frame must not OPEN a note (a soft
    onset starts the pending only once the level is real), and a WEAK
    frame INSIDE a note marks a break (the re-emission seam the block
    tracker misses). Both boundaries are pinned here at frame level.
    """
    from v2m.core.liveseg import WEAK_RATIO

    seg = LiveNoteSegmenter()
    # establish the median with strong frames... only WEAK ones exist yet:
    # the note must not open on them
    seg.push(0.01, _hz(64), weak=True)
    seg.push(0.02, _hz(64), weak=True)
    assert seg.state == "IDLE", "a weak frame opened a note"
    # a WEAK frame inside an ACTIVE note marks a break: the next frame at
    # the SAME pitch re-opens the pending (the split's first half)
    _feed(seg, _hz(64), 0.03, 0.40)
    seg.push(0.44, _hz(64), weak=True)  # the seam dip, same pitch
    _feed(seg, _hz(64), 0.45, 0.40)  # the strong return: split
    seg.flush(0.90)
    assert [n.midi for n in seg.notes] == [64, 64], (
        "the weak-marked seam did not split the same-pitch re-attack"
    )
    assert WEAK_RATIO < 0.4  # above this an AM tremolo (0.4x dips) would split


def test_split_frames_ceiling_and_reartic_band_guard_the_same_pitch_cut():
    """SPLIT_FRAMES (6) and REARTIC_ST (0.3) are the split's two guards.

    A break that lasts past SPLIT_FRAMES is a re-take (MIN_GAP_HOLD_S
    path), not a burst; a return OUTSIDE REARTIC_ST is the NEXT note's
    pitch (the hand-over), not the same note re-attacked. Both refusals
    are pinned: the cut only fires inside the band AND under the ceiling.
    """
    from v2m.core.liveseg import REARTIC_ST, SPLIT_FRAMES

    # ceiling: a LONG same-pitch break is a re-take — two notes, the gap
    # kept (the split must not fire past SPLIT_FRAMES * frame_s)
    long_gap = SPLIT_FRAMES * 0.01161 + 0.05
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(64), 0.0, 0.40)
    seg.push(0.41, float("nan"))
    seg.push(0.41 + long_gap - 0.01, float("nan"))  # mute run past the ceiling
    _feed(seg, _hz(64), 0.41 + long_gap, 0.40)
    seg.flush(0.40 + long_gap + 0.45)
    # two notes with a REAL gap between them (no split-glue: the break
    # outlived SPLIT_FRAMES, this is a re-take)
    assert len(seg.notes) == 2
    gap = seg.notes[1].start - seg.notes[0].end
    assert gap >= SPLIT_FRAMES * 0.011

    # band: the return lands FAR from the confirmed pitch — the NEXT
    # note's onset, not a re-attack: no same-pitch split may fire
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(64), 0.0, 0.40)
    seg.push(0.41, float("nan"), attack=True)  # the hand-over burst
    _feed(seg, _hz(67), 0.43, 0.40)  # 3 st away >> REARTIC_ST
    seg.flush(0.90)
    assert [n.midi for n in seg.notes] == [64, 67]
    assert REARTIC_ST < 0.5  # a hand-over's glide tail reaches ~0.5 st


def test_hold_window_constants_reject_a_moving_median():
    """CLEAN_FRAMES (5) + HOLD_RAW_FRAMES (5) + HOLD_ST (0.6): the hold
    test's shape. The CLEANED median must not confirm a ramp (it rides
    it), so the RAW spread is tested too — a median-only hold confirms a
    glide frame by frame (measured: the hand-over 47->48 read a duplicate).
    """
    from v2m.core.liveseg import CLEAN_FRAMES, HOLD_RAW_FRAMES, HOLD_ST

    assert HOLD_RAW_FRAMES == CLEAN_FRAMES, "the raw and cleaned windows must agree"
    # a steady RAMP of raw frames (0.3 st per frame) never confirms: the
    # median stays flat-ish but the raw spread blows past HOLD_ST
    seg = LiveNoteSegmenter()
    t = 0.0
    midi = 60.0
    while t < 0.60:
        t += 0.02
        midi += 0.3
        seg.push(t, float(midi_to_hz(midi)))
    seg.flush(0.65)
    assert seg.notes == [], "a ramp confirmed as a note (the raw guard is off?)"
    assert HOLD_ST < 1.0  # a wider hold glues whole steps (measured 64->65)


def test_memory_and_attack_log_stay_bounded():
    """MEMORY_S / MAX_ATTACKS / CLEAN_FRAMES: the buffers that MUST stay
    bounded in a process that runs for hours (the YIN stream's own bound
    is pinned in test_pitch; these are the segmenter's)."""
    seg = LiveNoteSegmenter()
    t = 0.0
    for _ in range(2000):  # ~20 s of frames, more than any window keeps
        seg.push(t, _hz(64))
        t += 0.01
    assert len(seg.recent_midis) <= 6  # CLEAN_FRAMES
    assert len(seg.recent_raw) <= 6  # HOLD_RAW_FRAMES
    assert len(seg.attacks) <= 64  # MAX_ATTACKS
    # one long note is running (flush will close it): nothing leaked
    assert len(seg.notes) == 0


# --------------------------------------------------------------------- #
# The silence gate is ONE policy for the displayed level and the voice
# evidence (audio_io.GATE_RMS): below it, no pitch enters the segmenter.
# --------------------------------------------------------------------- #


def _block_chain(x, sr: int = 22_050, gate_rms: float = 0.008):
    """Run a signal through the full live chain in 1024-sample blocks."""
    from v2m.core.liveseg import LivePitchStream

    stream = LivePitchStream(sr, gate_rms=gate_rms)
    y = np.asarray(x, dtype=np.float32)
    for i in range(0, len(y) - 1024, 1024):
        stream.feed(y[i : i + 1024], (i + 1024) / sr)
    stream.seg.flush(len(y) / sr)
    return stream.seg.notes


def _max_block_rms(x, sr: int = 22_050) -> float:
    y = np.asarray(x, dtype=np.float64)
    return max(float(np.sqrt(np.mean(y[i : i + 1024] ** 2))) for i in range(0, len(y) - 1024, 1024))


def test_a_periodic_tone_below_the_gate_reads_no_note():
    """A sub-gate rumble is silence for the pitch evidence too.

    YIN is amplitude-blind (its difference function is normalized), so a
    periodic E4 at 0.0015 amplitude — ~0.0011 per block, seven times under
    `audio_io.GATE_RMS` — is pitched perfectly while the level meter draws
    the block as silence. The gate must therefore mask the frame, not only
    arm the weak-frame witness, or the segmenter confirms a whole Mi4 on
    what the user is shown as a silent input.
    """
    from v2m.core.audio_io import GATE_RMS

    sr = 22_050
    t = np.arange(int(sr * 1.2)) / sr
    rumble = (0.0015 * np.sin(2 * np.pi * 329.63 * t)).astype(np.float32)
    assert _max_block_rms(rumble) < GATE_RMS, "the case must sit under the gate"
    assert _block_chain(rumble, gate_rms=GATE_RMS) == []


def test_a_quiet_voice_straddling_the_gate_is_still_transcribed_live():
    """The gate must not eat the soft singer (the `voix douce` contract).

    At amp 0.05 the voice peaks near 0.010 per block — just ABOVE the
    0.008 gate — while its softest voiced blocks dip to ~0.0049, under it.
    Those frames turn into mute runs, so the chain has to keep reading the
    six notes WITHOUT splitting a hold on a sub-gate dip: a mask that
    loses notes or fragments them would trade one defect for another.
    """
    from voicetruth import sequence, synthesize

    from v2m.core.audio_io import GATE_RMS

    truth = sequence([60, 62, 64, 65, 67, 72], dur_s=0.6, start_s=0.2)
    y = synthesize(truth, glide_s=0.09, consonant_s=0.05, amp=0.05)
    assert 0.0049 < _max_block_rms(y) < 0.02, "the case must straddle the gate"
    notes = _block_chain(y, gate_rms=GATE_RMS)
    assert [n.midi for n in notes] == [60, 62, 64, 65, 67, 72]
