"""The singer's intonation profile: per-note deviation, and what is systematic.

`Note.deviation` is the residual mistune, in semitones, LEFT AFTER the take's
global tuning offset was absorbed (`notes._measure_tuning_offset`). So it says
how far one note sat from its own semitone once the singer's systematic shift
was removed — the number that separates "she sits flat" from "she sang that
one note flat".

A summary is only useful if it tells a HABIT from noise, so every tendency
below is announced only when the evidence supports it: enough notes AND a big
enough mean. Otherwise the field is None — the same right to say nothing the
chords have (`chords.detect_chords`), and the same refusal to describe noise.
"""

from __future__ import annotations

import math
from typing import Protocol

# Fewer notes than this in a group: nothing is systematic, say nothing.
MIN_NOTES = 4
# A mean smaller than this (in cents) is intonation, not a habit.
MIN_CENTS = 12.0
# |cents| at or under this counts as ON the note. The GUI's tuning meter uses
# the same scale (it reads THIS constant), so the report and the live feedback
# can never disagree about what "in tune" means.
IN_TUNE_CENTS = 10.0
# Relative registers divide the take's MIDI extent, never its note count.
BANDS = ("low", "mid", "high")


class _Measured(Protocol):
    """What a summary reads from a note: the pitch sung, and its own reading."""

    midi: int
    deviation: float | None


def _cents(note) -> float | None:
    """One note's signed deviation in cents, or None when it has no reading.

    A note nobody measured is NOT evidence of being in tune: counting it as
    0.0 would report a perfect take from notes with no measurement — the rule
    the page's port already applies (`cents.js` answers null).
    """
    deviation = getattr(note, "deviation", None)
    if deviation is None:
        return None
    cents = 100.0 * float(deviation)
    return None if math.isnan(cents) else cents


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _tendency(values: list[float]) -> dict | None:
    """A group's systematic lean, or None when it is not one.

    Returned shape: `{"n": k, "mean_cents": -21.4}`. Both conditions are
    required — a group of two notes leans however those two notes leaned.
    """
    if len(values) < MIN_NOTES:
        return None
    mean = _mean(values)
    if abs(mean) < MIN_CENTS:
        return None
    return {"n": len(values), "mean_cents": round(mean, 1)}


def summarize(notes) -> dict:
    """The take's intonation: one line per habit, nothing invented.

    `by_register` uses equal-width MIDI bands over the take's extent;
    `by_position` uses note-count halves (first half / second half).
    A singer drifting flat may show up there, not in the take-wide mean.
    A note without a measurement is not counted at all: no reading is not
    evidence of being in tune, so it cannot fill a share or a band.
    """
    measured: list[tuple[_Measured, float]] = []
    for note in notes:
        cents = _cents(note)
        if cents is not None:
            measured.append((note, cents))
    devs = [cents for _, cents in measured]
    n = len(devs)
    empty = {
        "n_notes": n,
        "mean_cents": None,
        "median_cents": None,
        "abs_mean_cents": None,
        "in_tune_share": None,
        "sharp_share": None,
        "flat_share": None,
        "tendency": None,
        "by_register": None,
        "by_position": None,
    }
    if not devs:
        return empty
    mean = _mean(devs)
    on_note = sum(1 for d in devs if abs(d) <= IN_TUNE_CENTS)
    sharp = sum(1 for d in devs if d > IN_TUNE_CENTS)
    flat = n - on_note - sharp
    # the take-wide lean needs the same evidence as a group's: with two or
    # three notes there is no habit to name, only those notes
    tendency: str | None = None
    if n >= MIN_NOTES:
        tendency = "even"
        if abs(mean) >= MIN_CENTS:
            tendency = "sharp" if mean > 0 else "flat"

    # Identical pitches belong together regardless of repetitions or order.
    lowest = min(note.midi for note, _ in measured)
    highest = max(note.midi for note, _ in measured)
    values: list[list[float]] = [[], [], []]
    if highest > lowest:
        for note, deviation in measured:
            index = min(2, int(3 * (note.midi - lowest) / (highest - lowest)))
            values[index].append(deviation)
    bands = {name: _tendency(group) for name, group in zip(BANDS, values, strict=True)}

    # position: the first half of the take against the second (a drift)
    half = n // 2
    by_position = {
        "start": _tendency(devs[:half]),
        "end": _tendency(devs[half:]),
    }

    return {
        "n_notes": n,
        "mean_cents": round(mean, 1),
        "median_cents": round(_median(devs), 1),
        "abs_mean_cents": round(_mean([abs(d) for d in devs]), 1),
        "in_tune_share": round(on_note / n, 2),
        "sharp_share": round(sharp / n, 2),
        "flat_share": round(flat / n, 2),
        "tendency": tendency,
        "by_register": bands,
        "by_position": by_position,
    }
