"""Export the segmentation cases the web port must reproduce.

The browser runs a JavaScript copy of `core/liveseg.py`. A copy drifts unless
something compares it: this script drives the REAL Python segmenter over hand
written frame sequences and writes `web/test/cases.json`, which
`web/test/segmenter.test.mjs` replays through the port.

The frame sequences are the trap families the Python docstrings name
(a glide that only becomes a note once it holds, a same-pitch re-attack, a
mute run, a borderline-periodicity plateau, a flush merge) — never a happy
path only, or the fixture would pass on a segmenter that dropped its guards.

It also writes `web/constants.js`: the page's ported VALUES, read from the very
Python attributes they mirror — every threshold, and the note-name table the
page prints. A value restated in a JavaScript module is a value that can move
on one side and never be noticed, so the page imports this file instead of
declaring its own.

It also writes `web/test/yin.json`: the frames the DESKTOP's YIN reads on two
excerpts of the demo take (the ascent, every scale degree, and the tail with its
low pedal), so the browser's own estimator can be compared frame by frame on
real samples instead of only on hand-built pitch sequences — and
`web/test/yin-rates.json`, the same comparison at the two CAPTURE rates
(44.1 and 48 kHz), where the frame geometry is not the analysis rate's.

It also writes `tests/data/parity.wav` and its reading
(`tests/reference/parity.json`): the ONE fixture where the page is compared
to the desktop on AUDIO instead of on frames. `cases.json` hands pitch frames
to the segmenter and `yin.json` hands samples to the estimator; neither says
what the whole chain — YIN, gate, attacks, segmenter, flush — decides about a
sung phrase, and that reading is what a reader of the page's `notes.json`
trusts.

Run `./.venv/Scripts/python.exe -m scripts.export_web_fixtures` after ANY
change to liveseg.py, pitch.py or onsets.py; `tests/test_web_build.py` fails
until the checked-in JSON matches this script again.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "web" / "test" / "cases.json"
YIN_DEST = REPO / "web" / "test" / "yin.json"
YIN_RATES_DEST = REPO / "web" / "test" / "yin-rates.json"
INTONATION_DEST = REPO / "web" / "test" / "intonation.json"
CONSTANTS_DEST = REPO / "web" / "constants.js"
# The parity pair. The WAV is COMMITTED (`tests/data/*.wav` is the one audio
# exception in `.gitignore`) because a fixture a machine has to build is a
# fixture CI cannot compare with — the web job builds no take at all.
PARITY_WAV = REPO / "tests" / "data" / "parity.wav"
# The reading lives with the other FROZEN reference (`.gitignore` keeps
# `tests/reference/*.json` and `test_docs_guard.py` tracks no other JSON), not
# beside the audio it describes: it is a fingerprint of the Python engine, not
# an artifact of a run.
PARITY_GOLDEN = REPO / "tests" / "reference" / "parity.json"
PARITY_SR = 44_100
# The page reads a file in fixed 2048-sample blocks (`FILE_BLOCK` in app.js):
# the desktop twin must be fed the very same geometry, or the comparison would
# measure two different FEEDS instead of two implementations.
PARITY_BLOCK = 2048
# The rates the page can read that file AT. `decodeAudioData` resamples to the
# AudioContext's rate, which is the DEVICE's, so the browser feeds the chain
# the same phrase at 44.1 kHz, 48 kHz, or 96 kHz on a desk whose file is
# 44.1 — measured: 96 kHz here. That is not a drift of the port, and a golden
# written at one rate would read a red test as one; each rate therefore gets
# its own frozen reading, from the SAME linear interpolation `yin-rates.json`
# uses (the browser's resampler is not reproducible on either side).
# Measured, the desktop at the page's rate reproduces the page's notes exactly.
PARITY_RATES = (22_050, 44_100, 48_000, 96_000)
# Python module -> the attributes the page reads. Every name keeps its own
# spelling on both sides: a renamed threshold must fail here, not silently
# become a different number.
CONSTANT_SOURCES: dict[str, tuple[str, ...]] = {
    "v2m.core.pitch": ("FMIN", "FMAX", "MIN_PITCH_QUALITY"),
    "v2m.core.liveseg": (
        "WINDOW_MS",
        "HOP_MS",
        "HOLD_ST",
        "MEMORY_S",
        "CLEAN_FRAMES",
        "MIN_GAP_HOLD_S",
        "SPLIT_FRAMES",
        "HOLD_RAW_FRAMES",
        "REARTIC_ST",
        "BREAKOUT_ST",
        "WEAK_RATIO",
        "STAGE_MAX_S",
        "LIVE_DOUBT_PERIODICITY",
    ),
    # the note-name table is a ported VALUE like any threshold: the page prints
    # `C#4` from this list, so a renamed spelling has to move both sides
    "v2m.core.music": ("A4_HZ", "PIANO_MIN_MIDI", "PIANO_MAX_MIDI", "NOTE_NAMES"),
    "v2m.core.onsets": (
        "STREAM_LEVEL",
        "STREAM_FLOOR",
        "MIN_GAP_S",
        "FLATNESS_TRIGGER",
        "DIP_RATIO",
        "SUBWINDOW_DIP",
        "DEEP_DIP_RATIO",
        "DEEP_DIP_WINDOW_S",
        "STREAM_ALPHA",
        "STREAM_ALPHA_DOWN",
    ),
    "v2m.core.playback": (
        "HELD_MIN_S",
        "PEDAL_TAIL_FACTOR",
        "PEDAL_TAIL_MAX_S",
        "DARK_EXPO",
        "BRIGHT_EXPO",
    ),
    "v2m.core.intonation": ("IN_TUNE_CENTS", "MIN_NOTES", "MIN_CENTS"),
    "v2m.core.audio_io": ("GATE_RMS",),
    "v2m.gui.widgets": ("NEAR_CENTS",),
}
# The two rates a capture device opens at. The analysis rate is 22 050 Hz, so
# the live geometry there is 512/256 samples; a device gets 1024/512 and
# 1115/557 instead, and the 48 kHz window is NOT a power of two — the
# difference function's autocorrelation is zero-padded to 4096, a path the
# analysis-rate fixture never takes.
CAPTURE_RATES = (44_100, 48_000)
# The demo's phrase ends with a 1.0 s C2 pedal — the only note whose period
# reaches `tau_max` (a C4..C5 melody never does) — behind a 1.0 s tonic. The
# TAIL is therefore the part worth keeping, but not all of it: the window takes
# 0.35 s of that tonic (a stable short-lag reading PLUS the lag jump, where the
# period goes from ~85 to ~678 samples) and 0.6 s of the pedal's clamped dips,
# which is every pitch the tail ever reads. Reading the last 1.6 s spent 53
# frames per rate re-reading the same C4.
PEDAL_S = 1.0
EXCERPT_LEAD_S = 0.35
EXCERPT_PEDAL_S = 0.6
# Generated by `scripts/make_demo.py`, gitignored: the take is reproducible,
# the WAV itself is not committed. The fixture records its digest, so a
# different file under that name is refused instead of silently compared.
DEMO_TAKE = REPO / "demo.wav"
FRAME_S = 256 / 22_050  # the analysis rate's frame period (a hand-built seg)


def _frames(pitches, *, attack_at=(), quality=0.05, start=0.0):
    """(t, hz, attack, strength, weak, quality) rows, one per frame."""
    rows = []
    t = start
    for i, hz in enumerate(pitches):
        rows.append(
            [
                round(t, 6),
                None if hz is None else float(hz),
                1.0 if i in attack_at else 0.0,
                0.8 if i in attack_at else 0.0,
                0.0,
                None if hz is None else float(quality),
            ]
        )
        t += FRAME_S
    return rows


def _hold(hz, count, **kwargs):
    return [hz] * count, kwargs


def build_cases() -> dict:
    """The cases, keyed by name, each with its params, frames and outcome."""
    cases = {}

    # 1. A glide only becomes a note once the pitch HOLDS: the pending window
    #    is fed a ladder, then a plateau. One note, dated at the plateau.
    ladder = [55.0, 58.0, 61.5, 65.0, 68.0, 70.0, 72.0]
    frames = _frames(ladder + [73.42] * 14)
    cases["a_glide_confirms_when_it_holds"] = {"frames": frames}

    # 2. A voiced re-attack on ONE pitch splits the note at the attack.
    frames = _frames([220.0] * 16, attack_at=(12,))
    cases["a_re_attack_splits_a_held_note"] = {"frames": frames}

    # 3. A mute run longer than MIN_GAP_HOLD_S ends the note at its last
    #    voiced frame (a re-take, not a dropout).
    frames = _frames([220.0] * 16 + [None] * 8)
    cases["a_mute_run_ends_the_note"] = {"frames": frames}

    # 4. A short voiced break that RETURNS to the confirmed pitch is a
    #    re-articulation the level witness misses: the split path fires.
    frames = _frames([220.0] * 14 + [None, None] + [220.0] * 12)
    cases["a_short_break_on_one_pitch_splits"] = {"frames": frames}

    # 5. A pitch change closes the note and opens the next one.
    frames = _frames([220.0] * 14 + [261.63] * 14)
    cases["a_pitch_change_closes_and_reopens"] = {"frames": frames}

    # 6. Borderline periodicity is PUBLISHED, not masked: the note exists and
    #    carries the doubt.
    frames = _frames([220.0] * 16, quality=0.30)
    cases["a_borderline_plateau_is_published_as_doubt"] = {"frames": frames}

    # 7. A periodic plateau carries no doubt.
    frames = _frames([220.0] * 16, quality=0.05)
    cases["a_periodic_plateau_carries_no_doubt"] = {"frames": frames}

    # 8. Flush merges two short same-pitch fragments across a stutter gap.
    frames = _frames([220.0] * 12 + [None] * 4 + [220.0] * 12)
    cases["flush_merges_a_stutter"] = {"frames": frames, "flush": True}

    # 9. A pitch OUTSIDE the piano never becomes a note (octave error).
    frames = _frames([30.0] * 16)
    cases["a_non_piano_pitch_is_dropped"] = {"frames": frames}

    for case in cases.values():
        case.setdefault("params", {})
        case["params"]["frame_s"] = FRAME_S
    return cases


def render(cases: dict) -> dict:
    """Drive the real segmenter over each case and record what it decided."""
    from v2m.core.liveseg import LiveNoteSegmenter

    out = []
    for name in sorted(cases):
        case = cases[name]
        params = case.get("params", {})
        seg = LiveNoteSegmenter(**params)
        events = []
        last_t = 0.0
        for t, hz, attack, strength, weak, quality in case["frames"]:
            last_t = t
            events.append(
                seg.push(
                    t,
                    math.nan if hz is None else hz,
                    attack=bool(attack),
                    strength=strength,
                    weak=bool(weak),
                    quality=math.nan if quality is None else quality,
                )
            )
        if case.get("flush"):
            seg.flush(last_t + FRAME_S)
        out.append(
            {
                "name": name,
                "params": params,
                "frames": case["frames"],
                "flush": bool(case.get("flush")),
                "events": events,
                "notes": [
                    {
                        "start": n.start,
                        "end": n.end,
                        "midi": n.midi,
                        "articulated": n.articulated,
                        "strength": n.strength,
                        "dev": n.dev,
                        "voice_doubt": n.voice_doubt,
                    }
                    for n in seg.notes
                ],
            }
        )
    return {
        "generated_by": "scripts/export_web_fixtures.py",
        "frame_s": FRAME_S,
        "cases": out,
    }


# The trap cases the page's visible summary line must decide like the desktop,
# as (midi, cents) per note. The four traps the line can get wrong are all
# here — fewer than MIN_NOTES notes (nothing is systematic), a mean under
# MIN_CENTS (intonation, not a habit), identically-pitched notes (no extent to
# divide into registers) and an ODD count (the extra note belongs to the end
# half) — plus the two inclusive bars and an empty take.
#
# Every deviation is a WHOLE number of cents on purpose. The page summarises
# the cents its table PRINTS (`centsOf`, a rounded integer) while
# `core.intonation.summarize` reads the raw float, and that difference is
# deliberate: a note painted "in tune" must not be missing from the in-tune
# count one line below it. A fixture built on fractional cents would compare
# those two POLICIES; whole cents make both sides read the same numbers, so
# every disagreement left is the port's own logic — thresholds, bands, halves
# and shares.
INTONATION_CASES: dict[str, tuple[tuple[int, int | None], ...]] = {
    "fewer_than_four_notes_never_leans": ((60, 30), (62, 30), (64, 30)),
    "a_mean_under_twelve_cents_is_not_a_habit": (
        (60, 7),
        (60, 7),
        (60, 7),
        (60, 7),
        (61, 7),
        (62, 7),
    ),
    "identical_pitches_divide_no_register": (
        (67, 20),
        (67, 20),
        (67, 20),
        (67, 20),
        (67, -20),
        (67, -20),
        (67, -20),
        (67, -20),
    ),
    "an_odd_count_gives_its_extra_note_to_the_end": (
        (60, 20),
        (61, 20),
        (62, 20),
        (63, 20),
        (64, -20),
        (65, -20),
        (66, -20),
        (67, -20),
        (68, -20),
    ),
    "the_three_registers_split_the_extent": (
        (60, -20),
        (60, -20),
        (60, -20),
        (60, -20),
        (66, 0),
        (66, 0),
        (66, 0),
        (66, 0),
        (72, 20),
        (72, 20),
        (72, 20),
        (72, 20),
    ),
    "exactly_twelve_cents_is_a_habit": ((60, 12), (62, 12), (64, 12), (65, 12)),
    "exactly_ten_cents_is_in_tune": ((60, 10), (62, 10), (64, 10), (65, 10)),
    "no_notes_says_nothing": (),
    # A note with NO measurement is not evidence of being in tune: the page
    # skips it (`centsOf` answers null), so the desktop has to count the same
    # notes or the visible line describes a different take.
    "a_note_without_a_measurement_is_not_in_tune": ((60, 20), (62, None), (64, 20), (65, 20)),
}


def intonation_reference() -> dict:
    """What `core.intonation.summarize` decides on each trap, frozen.

    `web/intonation.js` is the one port with no conformity proof of its own: it
    is pure logic over notes the segmenter already produced, so nothing moved
    when the segmenter or the YIN was compared. The line it writes is VISIBLE
    ("N of M notes in tune"), and its rules are thresholds — exactly the kind
    of port that drifts to a different sentence while every other check stays
    green.
    """
    from types import SimpleNamespace

    from v2m.core.intonation import summarize

    cases = []
    for name, notes in INTONATION_CASES.items():
        summary = summarize(
            [
                SimpleNamespace(midi=midi)
                if cents is None
                else SimpleNamespace(midi=midi, deviation=cents / 100)
                for midi, cents in notes
            ]
        )
        cases.append(
            {
                "name": name,
                "notes": [
                    {"midi": midi, "dev": None if cents is None else cents / 100}
                    for midi, cents in notes
                ],
                "summary": summary,
            }
        )
    return {"generated_by": "scripts/export_web_fixtures.py", "cases": cases}


def piano_reference() -> dict:
    """A dry render of a few notes, as a sampled digest.

    The room is patched out: the desktop convolves a noise impulse response
    from numpy's PCG64 and the browser seeds its own, so only the DRY signal
    can be compared sample for sample. Stride keeps the fixture small while
    still pinning the envelopes, the partials, the placement and the detune —
    a changed attack, exponent or tail moves these samples.
    """
    from v2m.core import playback

    class _Note:
        def __init__(self, start, end, midi, strength):
            self.start = start
            self.end = end
            self.midi = midi
            self.strength = strength

        @property
        def duration(self):
            return self.end - self.start

    notes = [
        _Note(0.0, 0.6, 60, 0.9),  # a loud short note: fast attack, damper
        _Note(0.7, 1.7, 67, 0.6),  # a held note: the pedal tail rings
        _Note(1.8, 2.0, 72, 0.2),  # a soft one: late, slow attack, dark
        _Note(2.2, 2.5, 64, 0.5),  # neutral velocity
    ]
    sr = 8000
    original = playback._room_tail
    playback._room_tail = lambda y, sr: y
    try:
        rendered = playback.render_notes(notes, sr=sr)
    finally:
        playback._room_tail = original
    stride = 32
    return {
        "sr": sr,
        "stride": stride,
        "notes": [
            {
                "start": n.start,
                "end": n.end,
                "midi": n.midi,
                "strength": n.strength,
            }
            for n in notes
        ],
        "length": int(len(rendered)),
        "peak": float(abs(rendered).max()),
        "rms": float((rendered.astype("float64") ** 2).mean() ** 0.5),
        "samples": [round(float(v), 9) for v in rendered[::stride]],
        "room": "dry: the browser seeds its own noise IR",
    }


# `web/test/yin.json` reads two EXCERPTS, not the whole take: (name, start
# sample or None for "from the end", sample count). The ASCENT carries every
# scale degree the take sings (C4 259 Hz up to C5 523 Hz); the TAIL carries the
# old tonic and the low C2 pedal, whose period sits at the lag search's own
# limit. The descent repeats pitches the ascent already pins, and the middle of
# the take adds bytes, not bands — `tests/test_web_build.py` refuses a cut that
# loses either end (`_yin_spans_the_take`).
YIN_EXCERPTS: tuple[tuple[str, int | None, int], ...] = (
    ("ascent", 0, 92_610),   # 4.2 s from the start: C4 -> C5
    ("tail", None, 28_665),  # 1.3 s at the end: the tonic and the C2 pedal
)


def yin_reference(take: Path = DEMO_TAKE) -> dict | None:
    """The Python live YIN's frames over the fixture's excerpts, or None.

    The page carries its own YIN (`web/yin.js`) and the two are separate
    arithmetic paths (numpy's rfft against a radix-2 FFT), so the agreement is
    measured rather than assumed: `web/test/yin.test.mjs` recomputes every
    frame from the same samples and holds the difference to a pinned bound.
    """
    import hashlib

    import numpy as np

    from v2m.core.audio_io import SR, load_file
    from v2m.core.liveseg import HOP_MS, WINDOW_MS
    from v2m.core.pitch import YinBlock

    if not take.exists():
        return None
    signal = load_file(str(take), SR)
    window = max(64, round(SR * WINDOW_MS / 1000))
    hop = max(64, round(SR * HOP_MS / 1000))
    excerpts = []
    for name, start, count in YIN_EXCERPTS:
        begin = max(0, len(signal) - count) if start is None else max(0, start)
        chunk = np.asarray(signal[begin:begin + count], dtype=np.float32)
        frames = YinBlock(SR, window=window, hop=hop).push_with_quality(chunk)
        excerpts.append(
            {
                "name": name,
                "start": int(begin),
                "samples": int(len(chunk)),
                "frames": [
                    [
                        round(t, 6),
                        None if math.isnan(hz) else round(hz, 6),
                        None if math.isnan(quality) else round(quality, 6),
                    ]
                    for t, hz, quality in frames
                ],
            }
        )
    return {
        "generated_by": "scripts/export_web_fixtures.py",
        "source": {
            "name": take.name,
            "sha256": hashlib.sha256(take.read_bytes()).hexdigest(),
            "sr": SR,
            "samples": int(len(signal)),
        },
        "geometry": {"window": window, "hop": hop, "window_ms": WINDOW_MS, "hop_ms": HOP_MS},
        "excerpts": excerpts,
    }


def _js_literal(value) -> str:
    """The JavaScript spelling of one ported value.

    Only the two shapes the page ports are accepted: a number, or a flat list of
    strings (the note-name table). Anything else fails HERE, where the failing
    name is known, instead of writing Python syntax into a file the browser
    would refuse to parse.
    """
    if isinstance(value, bool):
        raise TypeError("a ported boolean has no JavaScript literal yet")
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value):
        return "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in value) + "]"
    raise TypeError(f"no JavaScript literal for {type(value).__name__}: {value!r}")


def constants_source() -> str:
    """`web/constants.js`: one declaration per ported value.

    Each value is read from the Python attribute itself, in the order its own
    module declares it, so a reader can find the source line by searching the
    new value in `src/v2m/core/*.py`. A module that needs one imports it; none
    of them may restate it (`tests/test_web_build.py` checks both halves).
    """
    from importlib import import_module

    lines = [
        "// The page's ported values, generated from the Python sources: the",
        "// thresholds it decides with and the note-name table it prints.",
        "//",
        "// The browser and the desktop must decide notes from the SAME numbers: a",
        "// threshold widened on one side segments differently, a renamed note",
        "// spelling prints a different name, and nothing the page shows would say",
        "// so. These values therefore have ONE declaration \u2014 this file \u2014 and",
        "// every JavaScript module imports them instead of restating them.",
        "//",
        "// Generated by `scripts/export_web_fixtures.py`; DO NOT EDIT BY HAND. Run",
        "// `./.venv/Scripts/python.exe -m scripts.export_web_fixtures` after a source",
        "// value moves: `tests/test_web_build.py` fails while this file is stale, and",
        "// when a module declares one of these names again.",
    ]
    for module_name, names in CONSTANT_SOURCES.items():
        module = import_module(module_name)
        lines.append("")
        lines.append(f"// {module_name}")
        # the module's own declaration order: the frozen file then reads like
        # the Python source it mirrors
        declared = [name for name in vars(module) if name in names]
        missing = sorted(set(names) - set(declared))
        if missing:
            raise AttributeError(f"{module_name} no longer declares {', '.join(missing)}")
        for name in declared:
            lines.append(f"export const {name} = {_js_literal(getattr(module, name))};")
    return "\n".join(lines) + "\n"


def _live_geometry(sr: int) -> dict:
    """The frame geometry the live path opens at a capture rate, and its FFT.

    Lifted from the ONE owner (`LivePitchStream` converts the millisecond
    settings into samples) rather than re-deriving the formula here: a drift
    in that conversion must move the fixture, and the Python guard asserts
    the frozen numbers against the stream itself.
    """
    from v2m.core.liveseg import LivePitchStream
    from v2m.core.pitch import FMAX, FMIN

    stream = LivePitchStream(sr, gate_rms=0.0)
    window = stream.window
    fft_size = 1
    while fft_size < 2 * window:
        fft_size <<= 1
    return {
        "window": window,
        "hop": stream.hop,
        # yin_pitch's own padding rule: the radix-2 transform must hold the
        # whole frame twice, so a non-power-of-two window is zero-padded.
        "fft_size": fft_size,
        # the lag span yin_pitch searches, in samples at THIS rate
        "tau_min": max(2, int(sr / FMAX)),
        "tau_max": min(window - 1, int(sr / FMIN)),
    }


def _upsample_linear(samples, sr_in: int, sr_out: int):
    """Linear interpolation, so both engines read the very same samples.

    The page cannot reproduce a polyphase resampler, and a comparison of two
    arbitrary resamplings would measure the resamplers instead of the YIN. The
    fixture therefore never stores audio: both sides interpolate the same
    float32 excerpt with this rule, and the FROZEN frames are what prove the
    two interpolations agree.
    """
    import numpy as np

    count = int(len(samples) * sr_out / sr_in)
    out = np.empty(count, dtype=np.float32)
    last = len(samples) - 1
    for i in range(count):
        position = i * sr_in / sr_out
        index = int(position)
        fraction = position - index
        left = float(samples[index])
        right = float(samples[min(index + 1, last)])
        out[i] = left + (right - left) * fraction
    return out


def yin_rates_reference(take: Path = DEMO_TAKE) -> dict | None:
    """The Python live YIN's frames at the two CAPTURE rates, or None.

    The analysis-rate fixture proves the arithmetic on the demo take; this one
    proves the GEOMETRY, which is a different thing: a port that hardcoded the
    512-sample window, or that assumed a power-of-two frame (true at 44.1 kHz,
    false at 48 kHz) would pass the 22 050 Hz comparison and still misread
    every real capture. Both rates get their own frozen window, hop, FFT size,
    lag span and frames. The excerpt is the take's TAIL on purpose — but only
    the part that differs: a short head of the tonic (a short lag and the jump
    to the pedal's) and the start of the low C2 pedal, whose period is the only
    one that reaches `tau_max`, so the lag search's low limit is exercised
    instead of merely declared.
    """
    import hashlib

    import numpy as np

    from v2m.core.audio_io import SR, load_file
    from v2m.core.pitch import YinBlock

    if not take.exists():
        return None
    signal = load_file(str(take), SR)
    # anchored on the PEDAL, not on the take's end: the window ends inside the
    # pedal, so a longer tail (another take) cannot silently shorten the part
    # that reaches `tau_max`.
    pedal = max(0, len(signal) - round(PEDAL_S * SR))
    start = max(0, pedal - round(EXCERPT_LEAD_S * SR))
    count = round((EXCERPT_LEAD_S + EXCERPT_PEDAL_S) * SR)
    excerpt = np.asarray(signal[start:start + count], dtype=np.float32)
    rates = []
    for sr in CAPTURE_RATES:
        geometry = _live_geometry(sr)
        samples = _upsample_linear(excerpt, SR, sr)
        yin = YinBlock(sr, window=geometry["window"], hop=geometry["hop"])
        frames = yin.push_with_quality(samples)
        rates.append(
            {
                "sr": sr,
                **geometry,
                "samples": int(len(samples)),
                "frames": [
                    [
                        round(t, 6),
                        None if math.isnan(hz) else round(hz, 6),
                        None if math.isnan(quality) else round(quality, 6),
                    ]
                    for t, hz, quality in frames
                ],
            }
        )
    return {
        "generated_by": "scripts/export_web_fixtures.py",
        "source": {
            "name": take.name,
            "sha256": hashlib.sha256(take.read_bytes()).hexdigest(),
            "sr": SR,
            "samples": int(len(signal)),
            "excerpt_start": int(start),
            "excerpt_samples": int(len(excerpt)),
        },
        "rates": rates,
    }


def parity_audio():
    """A 2.9 s voice-like phrase the whole live chain must read the same way.

    Synthetic on purpose. A short excerpt of a real take would be the stronger
    fixture, but the takes this repository holds are PERSONAL (`*.wav` stays
    ignored, `prises_verite/` too): committing one would publish a voice the
    singer never offered. So the phrase is built here, deterministically, and
    it carries exactly the facts the two engines have to agree on:

    - a HELD note with vibrato (a note, not a wobble),
    - a LEGATO GLIDE into the next one (followed, never a note of its own),
    - a consonant BURST on the SAME pitch (a re-articulation: two notes),
    - a below-gate breath (silence for the level meter AND the pitch),
    - two more pitch changes after it.
    """
    import numpy as np

    def voice(f0_from, f0_to, seconds, *, amp=0.4, attack=0.02, release=0.05):
        n = int(round(seconds * PARITY_SR))
        t = np.arange(n) / PARITY_SR
        glide = f0_from * (f0_to / f0_from) ** (t / seconds)
        vibrato = 0.2 * np.sin(2 * np.pi * 5.0 * t)  # ±20 cents, as make_demo
        phase = 2 * np.pi * np.cumsum(glide * 2.0 ** (vibrato / 12.0)) / PARITY_SR
        y = np.zeros(n)
        for k, a in enumerate((1.0, 0.5, 0.25, 0.12, 0.06), start=1):
            y += a * np.sin(k * phase)
        env = np.ones(n)
        head, tail = int(attack * PARITY_SR), int(release * PARITY_SR)
        env[:head] = np.linspace(0.0, 1.0, head)
        env[-tail:] = np.linspace(1.0, 0.0, tail)
        return (amp * y * env).astype(np.float32)

    def consonant(seconds=0.04, amp=0.5):
        """A bright burst on ONE pitch: the consonant of a re-attacked note.

        Differenced noise is the bright part (the vowel above is the harmonic
        stack); the rising/falling window keeps it inside its own four hundred
        samples, so the pitch frames it covers come back unvoiced — the break
        a same-pitch re-articulation is split on.
        """
        n = int(round(seconds * PARITY_SR))
        noise = np.random.default_rng(7).standard_normal(n + 1).astype(np.float32)
        burst = np.diff(noise) * amp
        return (burst * np.hanning(len(burst))).astype(np.float32)

    def breath(seconds=0.15, rms=0.004):
        """Air UNDER the gate: silence for the level meter and the pitch.

        A gap has to be silent for both engines the same way; noise loud
        enough to pass the gate would let one engine's YIN read a plateau in
        it and the other none, and the disagreement would be the fixture's,
        not either side's.
        """
        n = int(round(seconds * PARITY_SR))
        noise = np.random.default_rng(11).standard_normal(n).astype(np.float32)
        level = float(np.sqrt((noise.astype("float64") ** 2).mean())) or 1.0
        return (noise / level * rms).astype(np.float32)

    c4, d4, e4, g4 = 261.63, 293.66, 329.63, 392.0
    return np.concatenate(
        [
            voice(c4, c4, 0.60),  # a held C4
            voice(c4, e4, 0.22),  # a glide up: no note of its own
            voice(e4, e4, 0.46),  # the E4 it lands on
            consonant(),  # the consonant of the re-attack
            voice(e4, e4, 0.48, attack=0.005, release=0.04),  # the SAME E4, again
            breath(),  # a breath: silence, not a note
            voice(d4, d4, 0.50),  # D4
            voice(g4, g4, 0.50),  # G4
        ]
    )


def parity_wav_bytes() -> bytes:
    """The committed fixture, as 16-bit mono PCM at the analysis rate."""
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, parity_audio(), PARITY_SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def parity_reference() -> dict:
    """The desktop's LIVE reading of the fixture, block for block, per rate.

    The page has no offline pipeline: a chosen file goes through
    `LivePitchStream` in 2048-sample blocks, exactly like a capture. The twin
    is therefore the desktop's SAME class — not `analyze_file`, whose pYIN and
    look-ahead rules are a second, desktop-only reader the page never claims
    to be.

    One reading per `PARITY_RATES` entry, because the chain is fed the phrase
    at the rate the BROWSER decoded it to, and the note boundaries of a glide
    are a function of that rate (measured on this fixture: the C4 hands over
    to the glide at 0.580 s at 44.1 kHz and 0.650 s at 96 kHz, both engines
    agreeing at each). A single-rate golden would fail on a desk and pass on
    CI while saying nothing about either implementation.
    """
    import hashlib

    from v2m.core.audio_io import GATE_RMS
    from v2m.core.liveseg import LivePitchStream

    signal = parity_audio()
    rates = []
    for sr in PARITY_RATES:
        samples = (
            signal
            if sr == PARITY_SR
            else _upsample_linear(signal, PARITY_SR, sr).astype("float32")
        )
        stream = LivePitchStream(sr, gate_rms=GATE_RMS)
        last_t = 0.0
        for offset in range(0, len(samples), PARITY_BLOCK):
            block = samples[offset : offset + PARITY_BLOCK]
            last_t = (offset + len(block)) / sr
            stream.feed(block, last_t)
        stream.seg.flush(last_t)
        rates.append(
            {
                "sr": sr,
                "samples": int(len(samples)),
                "notes": [
                    {
                        "start": round(note.start, 4),
                        "end": round(note.end, 4),
                        "midi": note.midi,
                        "articulated": bool(note.articulated),
                        "strength": round(note.strength, 4),
                        "voice_doubt": bool(note.voice_doubt),
                    }
                    for note in stream.seg.notes
                ],
            }
        )
    return {
        "generated_by": "scripts/export_web_fixtures.py",
        "source": {
            "name": PARITY_WAV.name,
            "sha256": hashlib.sha256(parity_wav_bytes()).hexdigest(),
            "sr": PARITY_SR,
            "samples": int(len(signal)),
        },
        "chain": {
            "reader": "v2m.core.liveseg.LivePitchStream",
            "block": PARITY_BLOCK,
            "gate_rms": GATE_RMS,
            "flush": True,
        },
        "rates": rates,
    }


def _write_text(path: Path, text: str) -> None:
    """Write a fixture with LF on every platform.

    The fixtures are compared as bytes by both halves of the gate: the browser
    parses these files and `tests/test_web_export.py` re-derives the golden
    from the Python engine. Windows' text mode (CRLF) would leave one
    checkout matching and the other reading as drift.
    """
    path.write_text(text, encoding="utf-8", newline="\n")


def main(dest: Path = DEST, yin_dest: Path = YIN_DEST) -> None:
    _write_text(CONSTANTS_DEST, constants_source())
    print(f"ported constants -> {CONSTANTS_DEST}")

    payload = render(build_cases())
    payload["piano"] = piano_reference()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _write_text(dest, json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"{len(payload['cases'])} cases -> {dest}")

    _write_text(
        INTONATION_DEST, json.dumps(intonation_reference(), indent=1, sort_keys=True) + "\n"
    )
    print(f"{len(INTONATION_CASES)} intonation cases -> {INTONATION_DEST}")

    reference = yin_reference()
    if reference is None:
        print(f"{DEMO_TAKE.name} is absent: {yin_dest.name} left untouched")
        return
    _write_text(yin_dest, json.dumps(reference, indent=1, sort_keys=True) + "\n")
    excerpt_frames = sum(len(excerpt["frames"]) for excerpt in reference["excerpts"])
    print(f"{excerpt_frames} YIN frames in {len(reference['excerpts'])} excerpts -> {yin_dest}")

    rates = yin_rates_reference()
    if rates is None:
        print(f"{DEMO_TAKE.name} is absent: {YIN_RATES_DEST.name} left untouched")
        return
    _write_text(YIN_RATES_DEST, json.dumps(rates, indent=1, sort_keys=True) + "\n")
    print(
        f"{sum(len(rate['frames']) for rate in rates['rates'])} YIN frames at "
        f"{', '.join(str(rate['sr']) for rate in rates['rates'])} Hz -> {YIN_RATES_DEST}"
    )

    PARITY_WAV.parent.mkdir(parents=True, exist_ok=True)
    PARITY_WAV.write_bytes(parity_wav_bytes())
    reading = parity_reference()
    _write_text(PARITY_GOLDEN, json.dumps(reading, indent=1, sort_keys=True) + "\n")
    print(
        f"{sum(len(rate['notes']) for rate in reading['rates'])} notes read by "
        f"{reading['chain']['reader']} at "
        f"{', '.join(str(rate['sr']) for rate in reading['rates'])} Hz -> "
        f"{PARITY_WAV.name} + {PARITY_GOLDEN.name}"
    )


if __name__ == "__main__":
    main()
