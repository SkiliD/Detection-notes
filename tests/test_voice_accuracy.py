"""Transcription accuracy: every sung note must be found back.

The `voicetruth` harness synthesizes a voice whose every note is known
(portamentos, consonants, vibrato, noise) and compares what the chain
detects against that truth. These tests pin the non-regression thresholds:

- OFFLINE segmentation is exact on all 19 cases — before the attack
  detection and the transition rules, a 90 ms glissando turned 6 sung
  notes into 10, and a held note with vibrato counted 19 "attacks";
- the attack evidence never degrades geometry alone;
- no note is invented INSIDE a sung note, the defect that made the score
  illogical;
- the real-time path, with no look-ahead, finds the articulated phrases
  exactly.

Known real-time limits (documented in AGENTS.md, not tested here):
semiquavers (120 ms) are merged, and a very long portamento (250 ms)
shifts the arrival note's attack.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from conftest import SR
from voicetruth import (
    Accuracy,
    SungNote,
    add_breath,
    detect_notes,
    measure,
    sequence,
    synthesize,
)
from v2m.core.liveseg import LiveNoteSegmenter, LivePitchStream
from v2m.core.music import midi_to_hz
from v2m.core.notes import Note
from v2m.core.pipeline import analyze_signal

MIDIS = [60, 62, 64, 65, 67, 72]
GAMME = [60, 62, 64, 65, 67, 69, 71, 72]
BLOCK = 1024
GATE = 0.008

# name -> (sung notes, synthesizer settings)
CASES: dict[str, tuple[list[SungNote], dict]] = {
    "consonnes": (sequence(MIDIS, dur_s=0.6, start_s=0.2), dict(glide_s=0.09, consonant_s=0.05)),
    "legato": (sequence(MIDIS, dur_s=0.6, start_s=0.2), dict(glide_s=0.09, consonant_s=0.0)),
    "détaché": (sequence(MIDIS, dur_s=0.6, gap_s=0.15, start_s=0.2), dict(glide_s=0.09, consonant_s=0.05)),
    "silences longs": (
        sequence([60, 64, 67], dur_s=0.5, gap_s=0.5, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05),
    ),
    "vibrato": (
        sequence(MIDIS, dur_s=0.6, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05, vibrato_st=0.3),
    ),
    "bruit": (
        sequence(MIDIS, dur_s=0.6, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05, noise=0.03),
    ),
    "bruit et vibrato": (
        sequence(MIDIS, dur_s=0.6, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05, vibrato_st=0.3, noise=0.03),
    ),
    "croches": (sequence(GAMME, dur_s=0.2, start_s=0.2), dict(glide_s=0.03, consonant_s=0.04)),
    "doubles": (sequence([60, 62, 64, 65], dur_s=0.12, start_s=0.2), dict(glide_s=0.02, consonant_s=0.03)),
    "note courte": (
        [SungNote(60, 0.2, 0.6), SungNote(67, 0.8, 0.15), SungNote(72, 0.95, 0.6)],
        dict(glide_s=0.03, consonant_s=0.05),
    ),
    "glissando long": (
        [SungNote(60, 0.2, 0.8), SungNote(67, 1.0, 0.8)],
        dict(glide_s=0.25, consonant_s=0.05),
    ),
    "tenue vibrato": ([SungNote(57, 0.2, 1.5)], dict(vibrato_st=0.4, consonant_s=0.05)),
    "voix grave": (
        sequence([45, 47, 48, 50, 52], dur_s=0.8, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05),
    ),
    "voix haute": (
        sequence([69, 71, 72, 74, 76], dur_s=0.8, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05),
    ),
    "gamme liée": (sequence(GAMME, dur_s=0.4, start_s=0.2), dict(glide_s=0.06, consonant_s=0.04)),
    "descendante liée": (
        sequence(list(reversed(GAMME)), dur_s=0.4, start_s=0.2),
        dict(glide_s=0.06, consonant_s=0.04),
    ),
    "saut d'octave": (
        [SungNote(60, 0.2, 0.6), SungNote(72, 0.8, 0.6)],
        dict(glide_s=0.05, consonant_s=0.05),
    ),
    "saut octave haut": (
        [SungNote(72, 0.2, 0.6), SungNote(84, 0.8, 0.6)],
        dict(glide_s=0.05, consonant_s=0.05),
    ),
    "soprano gamme": (
        sequence([79, 81, 83, 84, 86, 88], dur_s=0.6, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05),
    ),
    "glissando descendant": (
        [SungNote(67, 0.2, 0.8), SungNote(60, 1.0, 0.8)],
        dict(glide_s=0.25, consonant_s=0.05),
    ),
    "vibrato profond": (
        [SungNote(64, 0.2, 0.82), SungNote(67, 1.02, 0.8)],
        dict(glide_s=0.09, consonant_s=0.05, vibrato_st=0.6),
    ),
    "trille": (
        [SungNote(64, 0.2, 0.35), SungNote(65, 0.55, 0.35),
         SungNote(64, 0.9, 0.35), SungNote(65, 1.25, 0.35)],
        dict(glide_s=0.02, consonant_s=0.03),
    ),
    "répétition": ([SungNote(64, 0.2, 0.7), SungNote(64, 1.2, 0.7)], dict(consonant_s=0.05)),
    "répétition aa-aa": (
        sequence([69, 69, 69, 69], dur_s=0.30, gap_s=0.15, start_s=0.2),
        dict(consonant_s=0, glide_s=0, noise=0.02),
    ),
    # 2026-09-25, "repeated same note read as one hold": the harness's
    # consonant at FULL strength (1.4) produced 3 notes out of 4 live —
    # the attack landed on the burst's garbage frame, the split refused it
    # and consumed it. The case freezes the repair (re-staging the attack
    # the split refused).
    "répétition ta-ta": (
        sequence([64, 64, 64, 64], dur_s=0.50, gap_s=0.05, start_s=0.2),
        dict(consonant_s=0.05, consonant_amp=1.4, glide_s=0),
    ),
    # ...and the seam WITHOUT consonant or silence is one hold only if the
    # singer does not close the glottis: with the modeled closure (~30 ms,
    # level notch), the 4 re-emissions read.
    "répétition collée + fermeture": (
        sequence([64, 64, 64, 64], dur_s=0.50, start_s=0.2),
        dict(consonant_s=0, glide_s=0.09, closure_s=0.03),
    ),
}

# the real-time path finds exactly these phrases (2026-09-25: the three
# new truth cases — descending glissando, deep vibrato, soprano — join;
# they already passed at their arrival, the live chain holds them)
LIVE_CASES = [
    "consonnes",
    "legato",
    "détaché",
    "silences longs",
    "croches",
    "gamme liée",
    "descendante liée",
    "voix grave",
    "voix haute",
    "bruit et vibrato",
    "saut d'octave",
    "répétition",
    "répétition aa-aa",
    "répétition ta-ta",
    "répétition collée + fermeture",
    "glissando descendant",
    "vibrato profond",
    "soprano gamme",
]
LIVE_TOL_S = 0.20

# real time has no look-ahead: these extreme cases (120 ms semiquavers,
# a 250 ms portamento, a 150 ms note preceded by a portamento) are merged
# or shifted — never wrong in pitch.
LIVE_KNOWN_LIMITS = {"doubles", "glissando long", "note courte"}


@lru_cache(maxsize=None)
def _signal(name: str) -> np.ndarray:
    truth, kwargs = CASES[name]
    return synthesize(truth, **kwargs)


@lru_cache(maxsize=None)
def _offline(name: str, with_attacks: bool) -> Accuracy:
    truth, _ = CASES[name]
    return measure(detect_notes(_signal(name), SR, with_attacks=with_attacks), truth)


@lru_cache(maxsize=None)
def _live(name: str) -> Accuracy:
    truth, _ = CASES[name]
    stream = LivePitchStream(SR, gate_rms=GATE)
    y = _signal(name)
    for i in range(0, len(y) - BLOCK, BLOCK):
        stream.feed(y[i : i + BLOCK], (i + BLOCK) / SR)
    stream.seg.flush(len(y) / SR)
    return measure(stream.seg.notes, truth, tol_s=LIVE_TOL_S)


# --------------------------------------------------------------------------- #
# The harness itself
# --------------------------------------------------------------------------- #


def test_the_harness_catches_a_lost_note():
    """Sans quoi les tests ci-dessous ne prouveraient rien."""
    truth, _ = CASES["consonnes"]
    acc = measure([], truth)
    assert acc.matched == 0 and acc.missing == len(truth) and not acc.exact


def test_the_harness_catches_a_note_in_the_middle_of_another():
    """The original defect: a micro-note wedged inside a sung note."""
    truth, _ = CASES["tenue vibrato"]
    detected = [
        Note(start=truth[0].start_s, end=truth[0].start_s + truth[0].dur_s, midi=truth[0].midi, confidence=1.0),
        Note(start=truth[0].start_s + 0.5, end=truth[0].start_s + 0.58, midi=truth[0].midi + 1, confidence=1.0),
    ]
    acc = measure(detected, truth)
    assert acc.matched == 1 and acc.extra == 1 and acc.intruders == 1
    assert not acc.exact


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(CASES))
def test_offline_transcription_is_exact(name: str):
    acc = _offline(name, True)
    assert acc.exact, acc.line()


@pytest.mark.parametrize("name", sorted(CASES))
def test_attack_evidence_never_makes_it_worse(name: str):
    with_attacks = _offline(name, True)
    geometry = _offline(name, False)
    assert with_attacks.matched >= geometry.matched
    assert with_attacks.extra <= geometry.extra
    assert with_attacks.intruders == 0


@pytest.mark.parametrize(
    "name",
    ["legato", "consonnes", "glissando long", "saut d'octave", "répétition", "répétition aa-aa"],
)
def test_geometry_alone_is_still_strict(name: str):
    """Without attack evidence, the glissando's geometric rule must suffice
    everywhere it is unambiguous (the pipeline always supplies
    attacks, but a caller may not have them)."""
    acc = _offline(name, False)
    assert acc.exact, acc.line()


def test_onsets_and_pitches_are_close():
    for name in ("consonnes", "gamme liée", "croches", "voix grave"):
        acc = _offline(name, True)
        # the detected note starts on the consonant, not the stabilization
        assert acc.worst_start_ms <= 80.0, (name, acc.line())
        assert acc.worst_pitch_st == 0.0, (name, acc.line())


def test_high_octave_reaches_the_pitch_ceiling():
    """Notes ABOVE the old reading ceiling (C6) stay accurate.

    The old FMAX (1050 Hz) sat EXACTLY on the sung C6: pYIN had no room
    above and folded the soprano an octave down (measured: a sung 84 read
    72, a 79-88 line read 67-76). A C7 of headroom reads those notes
    without changing anything below (chant.wav/demo.wav unchanged).
    """
    from v2m.core.pitch import FMAX

    assert FMAX > 440.0 * 2 ** ((84 - 69) / 12)  # above the sung C6
    for name in ("saut octave haut", "soprano gamme"):
        acc = _offline(name, True)
        assert acc.exact, (name, acc.line())


def test_descending_glide_leaves_no_duplicate():
    """A DESCENDING glissando reads like the ascending one: the starting
    note ends at the hand-over (the arrival's consonant belongs to it) and
    no duplicated fragment survives. Two traps, both measured: the
    arrival's re-attack cut the stretched note at its own ramp (the
    duplicate), and _split_between_at_attack's strict bound rejected the
    attack falling ON the boundary (the arrival started 150 ms late).
    """
    acc = _offline("glissando descendant", True)
    assert acc.exact, acc.line()
    detected = detect_notes(_signal("glissando descendant"), SR)
    arrival = [n for n in detected if n.midi == 60]
    assert len(arrival) == 1 and arrival[0].start <= 1.03, [
        (n.midi, round(n.start, 2)) for n in detected
    ]


def test_arrival_pitch_reads_the_plateau_not_the_ramp():
    """A note arriving by glissando reads its PLATEAU, not the median of the
    ramp leading there (measured: 62 read 61, 67 read 66, 72 read 71 when
    the forced_pitch condition stood inverted — 25 red tests). The rule:
    tail median when the arrival settled.
    """
    # the corpus already freezes the correct reading; this case pokes the
    # mechanism: a legato line whose notes 2/3/5 arrive by a 90 ms portamento
    acc = _offline("legato", True)
    assert acc.exact and acc.worst_pitch_st == 0.0, acc.line()


def test_deep_vibrato_and_trill_stay_exact():
    """Deep vibrato (±0.6 st) and a fast trill stay exact: vibrato does not
    fragment (the 9-frame median follows the oscillation) and a 350 ms
    trill does not merge."""
    for name in ("vibrato profond", "trille"):
        acc = _offline(name, True)
        assert acc.exact, (name, acc.line())


# --------------------------------------------------------------------------- #
# Real time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", LIVE_CASES)
def test_live_chain_reads_the_sung_phrase(name: str):
    acc = _live(name)
    assert acc.exact, acc.line()
    assert acc.worst_start_ms <= 120.0, acc.line()


def test_live_chain_never_invents_notes_inside_a_sung_note():
    for name in CASES:
        if name in LIVE_KNOWN_LIMITS:
            continue  # see test_live_limits_are_bounded
        assert _live(name).intruders == 0, name


def test_live_limits_are_bounded():
    """Characterizes the real-time path's accepted limits so they stay
    bounded: a lost or merged note, never an invented pitch."""
    for name in LIVE_KNOWN_LIMITS:
        truth, _ = CASES[name]
        stream = LivePitchStream(SR, gate_rms=GATE)
        y = _signal(name)
        for i in range(0, len(y) - BLOCK, BLOCK):
            stream.feed(y[i : i + BLOCK], (i + BLOCK) / SR)
        stream.seg.flush(len(y) / SR)
        sung = {n.midi for n in truth}
        assert len(stream.seg.notes) <= len(truth), name
        assert all(n.midi in sung for n in stream.seg.notes), name


# --------------------------------------------------------------------------- #
# Breaths: articulated phrases -> rests in the model
# --------------------------------------------------------------------------- #


def test_dotted_pulse_take_publishes_6_8_offline():
    """A real dotted-pulse signal gets its 6/8 OFFLINE, like live — the
    tempo is corrected from dotted beat to quarter (x1.5) by the same
    alternation evidence (meter.dotted_pulse_tempo), applied after the
    refinement. Dotted beats at 90 BPM: the quarter at 135, octave fold in
    range; the strengths alternate (strong dotted beat, weak offbeat) to
    arm the alternation evidence.
    """
    quarter = 60.0 / 135.0            # the expected quarter at 135 BPM
    dotted = 1.5 * quarter            # the sung dotted beat
    notes = []
    for i in range(9):                # 9 dotted beats = 4.5 bars of 6/8
        notes.append(SungNote(60 + (i % 3), start_s=i * dotted,
                              dur_s=dotted * 0.85,
                              amp=(1.0 if i % 2 == 0 else 0.45)))
    # level-attack articulation: the consonant bursts are not scaled per
    # note, their constant loudness would blur the strength
    y = synthesize(notes, consonant_s=0)
    res = analyze_signal(y, SR)
    assert res.meter is not None and res.meter.label == "6/8"
    assert res.tempo is not None and res.tempo.bpm == pytest.approx(135.0, abs=4.0)


def test_breaths_between_phrases_become_rests():
    """The CASES["détaché"] truth: 6 notes in 6 phrases; every breath must
    become a rest (the
    intro included) and none may overlap a note. The figures are either None
    (gap too short for a figure at this tempo — the staff engraves it from
    its duration anyway) or from the catalogue."""
    y = _signal("détaché")
    pipeline = analyze_signal(y, SR)
    rests = pipeline.rests or []
    assert len(rests) >= 5, [(r.start, r.end, r.figure) for r in rests]
    catalog = {"pause", "demi-pause", "soupir", "demi-soupir", "quart de soupir"}
    catalog |= {f"{f} pointée" for f in catalog}
    for r in rests:
        assert r.figure is None or r.figure in catalog, r.figure
    # no breath overlaps a note
    notes = pipeline.notes
    for r in rests:
        for n in notes:
            assert r.end <= n.start + 0.02 or r.start >= n.end - 0.02, (r, n)


def test_long_breath_is_labeled_a_quarter_rest():
    """A real breath (0.6 s at 80 BPM = 0.8 beats) must be labeled a quarter
    rest — the figure an editor engraves between two phrases."""
    y = synthesize(sequence([60, 62, 64], dur_s=0.6, gap_s=0.6, start_s=0.2),
                   glide_s=0.09, consonant_s=0.05)
    rests = analyze_signal(y, SR).rests or []
    figures = [r.figure for r in rests if r.figure]
    assert "soupir" in figures, [(r.start, r.end, r.figure) for r in rests]


def test_rearticulated_note_on_same_pitch_is_split():
    """Two "la"s separated by 150 ms of silence: one note would betray a
    swallowed attack — the chain must separate them (live behavior)."""
    from voicetruth import synthesize as _syn

    truth = [
        SungNote(start_s=0.2, dur_s=0.6, midi=60),
        SungNote(start_s=0.95, dur_s=0.6, midi=60),
    ]
    y = _syn(truth, glide_s=0.09, consonant_s=0.05)
    detected = detect_notes(y, SR)
    acc = measure(detected, truth)
    assert acc.exact, acc.line()


# --------------------------------------------------------------------------- #
# Human errors: breath, soft voice, careless timing
# --------------------------------------------------------------------------- #


def test_a_breathy_voice_does_not_sprout_extra_notes():
    """A BREATHY voice (much air) must not cover itself in micro-notes:
    measured, the breath's undulations produced up to 12 fragments inside
    the sung notes. The attacks' relative strength floor (MIN_STRENGTH)
    rejects the noise (<= 0.06) and keeps the consonants (>= 0.86)."""
    truth = sequence(MIDIS, dur_s=0.6, start_s=0.2)
    for noise in (0.05, 0.08, 0.10):
        y = synthesize(truth, glide_s=0.09, consonant_s=0.05, noise=noise)
        acc = measure(detect_notes(y, SR), truth)
        assert acc.exact, f"noise={noise}: {acc.line()}"


def test_breaths_never_become_notes():
    """Breaths NEVER become notes, in any form: neither an air intake in a
    legato gap (the 90 ms breath precedes each re-attack), nor an isolated
    breath in a long silence (which must stay a REST, not a noise note),
    nor the continuous breath of a long hold. Measured: a breath 3x stronger
    than the contract above (0.10 absolute) really fragments a hold — that
    is off-contract, the harness documents it rather than suffering it."""
    # 1) legato line: the singer takes air between two legato notes
    truth = sequence([60, 64, 67, 72], dur_s=0.5, gap_s=0.15, start_s=0.2)
    y = synthesize(truth, SR, glide_s=0.09, consonant_s=0.0)
    for n in truth[1:]:  # the breath precedes each re-attack
        add_breath(y, n.start_s - 0.06)
    acc = measure(detect_notes(y, SR), truth)
    assert acc.exact, f"legato + air intakes: {acc.line()}"

    # 2) isolated breath in a long silence: a rest, not a note
    truth2 = [SungNote(60, 0.2, 0.5), SungNote(67, 1.4, 0.5)]
    y2 = synthesize(truth2, SR, glide_s=0.0, consonant_s=0.0)
    add_breath(y2, 0.8, dur_s=0.15, amp=0.3, seed=9)
    acc2 = measure(detect_notes(y2, SR), truth2)
    assert acc2.exact, f"isolated breath: {acc2.line()}"

    # 3) the harness's continuous breath (0.10 = the frozen contract) on
    #    LONG holds: never a micro-note inside
    for dur in (1.0, 2.4):
        truth3 = [SungNote(60, 0.2, dur), SungNote(67, 0.2 + dur + 0.3, 1.0)]
        y3 = synthesize(truth3, SR, glide_s=0.0, consonant_s=0.05, noise=0.10)
        acc3 = measure(detect_notes(y3, SR), truth3)
        assert acc3.exact, f"tenue {dur}s soufflée: {acc3.line()}"


def test_a_quiet_voice_is_still_transcribed():
    """Singing SOFTLY (amp 0.05) must neither lose the notes nor fall under
    the silence gates."""
    truth = sequence(MIDIS, dur_s=0.6, start_s=0.2)
    y = synthesize(truth, glide_s=0.09, consonant_s=0.05, amp=0.05)
    assert measure(detect_notes(y, SR), truth).exact


def test_careless_timing_is_still_transcribed():
    """IRREGULAR durations and breaths (human song): 5/5 notes. Neither
    path assumes a regular tempo."""
    truth = [
        SungNote(60, 0.20, 0.35), SungNote(62, 0.70, 0.70),
        SungNote(64, 1.55, 0.28), SungNote(65, 1.95, 0.55),
        SungNote(67, 2.65, 0.90),
    ]
    y = synthesize(truth, glide_s=0.09, consonant_s=0.05)
    assert measure(detect_notes(y, SR), truth).exact


# ---------------------------------------------------------------------------
# 2026-09-25 repairs: "repeated same note read as one hold"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [0.30, 0.40, 0.50, 0.60])
def test_a_tremolo_never_becomes_several_notes(depth: float):
    """An amplitude tremolo is ONE held note, not a series of notes.

    Every ±30..60 % AM cycle read as an "attack" (a rise > 1.5x the floor)
    and the hold covered itself in 7-10 fragments. Two guards: a level
    attack requires a DEEP DIP before it (a real re-attack is preceded by a
    near-silence, a tremolo never falls under ~0.4x) — and every attack
    consumes its dip (a mark left by the note start must not license the
    next cycle).
    """
    truth = [SungNote(64, 0.2, 1.7)]
    y = synthesize(truth, glide_s=0, consonant_s=0.05, tremolo_depth=depth)
    stream = LivePitchStream(SR, gate_rms=GATE)
    for i in range(0, len(y) - BLOCK, BLOCK):
        stream.feed(y[i : i + BLOCK], (i + BLOCK) / SR)
    stream.seg.flush(len(y) / SR)
    acc = measure(stream.seg.notes, truth, tol_s=LIVE_TOL_S)
    assert acc.exact, f"tremolo ±{int(depth*100)}%: {acc.line()}"


def test_live_attacks_carry_their_accent_evidence():
    """The attack strength lives on articulated notes — even at the take's
    start. Three measured leaks, all repaired: (1) the attack landing on
    mute frames in the IDLE state was dropped (no staging there); (2) the
    normalization read the re-armed floor, which every gap resets to zero —
    "risen from silence" at every note, strengths all 0 or all 1; the
    measure is now the level against the recent PEAK (bounded 0..1, keeps
    the take's contrast); (3) the strength did not cross the same-note
    split path.
    """
    truth = [SungNote(m, 0.2 + i * 0.7, 0.55, amp=1.0 if i % 2 == 0 else 0.4)
             for i, m in enumerate([60, 62, 60, 62, 60, 62])]
    y = synthesize(truth, glide_s=0, consonant_s=0)
    stream = LivePitchStream(SR, gate_rms=GATE)
    for i in range(0, len(y) - BLOCK, BLOCK):
        stream.feed(y[i : i + BLOCK], (i + BLOCK) / SR)
    stream.seg.flush(len(y) / SR)
    acc = measure(stream.seg.notes, truth, tol_s=LIVE_TOL_S)
    assert acc.exact, acc.line()
    strengths = [n.strength for n in stream.seg.notes]
    # the take's contrast reads: strong >= 0.3, weak <= 0.6, and at least
    # three attacks actually seen (the legato ones have none)
    assert any(s >= 0.3 for s in strengths), strengths
    strong = [s for n, s in zip(stream.seg.notes, strengths) if n.midi == 60]
    weak = [s for n, s in zip(stream.seg.notes, strengths) if n.midi == 62]
    assert sum(strong) / len(strong) > sum(weak) / len(weak), (strong, weak)


def test_live_stale_attack_does_not_date_a_note_early():
    """A pre-phrase breath's attack does not pull the next note's start: a
    staged attack beyond STAGE_MAX_S is abandoned (the consonant precedes
    its vowel by <= 100 ms, measured 12-46 ms; the breath may precede the
    phrase by 0.2 s).
    """
    from v2m.core.liveseg import STAGE_MAX_S

    assert STAGE_MAX_S <= 0.10
    seg = LiveNoteSegmenter()
    seg.push(0.0, float("nan"), attack=True, strength=0.9)  # an orphan attack
    # the voice arrives AFTER the staging horizon: the stale attack is
    # abandoned — the note starts at the voice (0.15), not the breath (0.0).
    t = 0.15
    while t < 0.50:
        seg.push(t, float(midi_to_hz(69)))
        t += 0.01
    seg.flush(0.55)
    assert len(seg.notes) == 1
    assert seg.notes[0].start == pytest.approx(0.15, abs=0.05), seg.notes[0].start
    assert seg.notes[0].strength == 0.0  # the stale attack dated nothing


def test_live_confirmation_latency_stays_under_120ms():
    """Live confirmation was lowered (140 -> 100 -> 90 ms) for display: the
    robustness must hold — vibrato, legato, eighths, repetitions. The
    measured floor is 0.09: below it (0.06-0.08), the "noise and vibrato"
    case fails sporadically by seed.
    """
    stream = LivePitchStream(SR, gate_rms=GATE)
    assert stream.seg.confirm_s == pytest.approx(0.09)
    for name in ("vibrato", "legato", "croches", "répétition"):
        acc = _live(name)
        assert acc.exact, f"{name}: {acc.line()}"


# ---------------------------------------------------------------------------
# Drift on the REAL take: the golden fingerprint replayed at every run
# ---------------------------------------------------------------------------


def test_real_take_segmentation_does_not_drift():
    """An automatic net against segmentation drift on chant.wav: the
    fingerprint (--compare of the diagnostic) is replayed HERE, at every
    suite run, against the fingerprint recorded beside the take
    (<audio>.golden.json). Any difference (tempo, meter, key, every note,
    every rest) fails THIS test — no need to remember to run the script.
    The take is gitignored: skipped when absent.
    """
    from v2m.core import audio_io
    from v2m.core.pipeline import AnalyzeOptions, analyze_signal
    from scripts.score_accuracy import (  # type: ignore[import-not-found]
        compare_fingerprints,
        load_golden,
        segmentation_fingerprint,
    )

    audio = Path(__file__).resolve().parents[1] / "chant.wav"
    if not audio.exists():
        pytest.skip("chant.wav absent (gitignored, regenerable) — drift not replayed")
    golden = load_golden(str(audio))
    if golden is None:
        pytest.skip("no golden fingerprint for chant.wav — run the script --golden once")
    res = analyze_signal(
        audio_io.load_file(str(audio)), audio_io.SR,
        source=str(audio), opts=AnalyzeOptions(),
    )
    diffs = compare_fingerprints(golden, segmentation_fingerprint(res))
    assert not diffs, "segmentation drift on chant.wav:\n" + "\n".join(diffs)
