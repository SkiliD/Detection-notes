"""The metric frame of a take: seconds <-> beats, position, snapping.

Why a module of its own: `60.0 / bpm` was rewritten in seven modules, each
with its own rounding, and "4/4 by default" was decided in three places. Every
consumer DOWNSTREAM of the tempo reads the grid; the detectors
(`detect_tempo`, `detect_meter`) stay upstream and take the scalar tempo,
because the grid is built from their result.

- the unit that crosses the seam is the **float quarter beat** (`beats`,
  `seconds`);
- integer divisions (24 per beat for the MusicXML export, a sixteenth for MIDI
  quantization) go through `step_seconds` / `snap` / `snap_notes` / `steps`,
  whose convention is stated once, on `step_seconds`;
- the last rounding of a duration — a note the detector left without a figure
  is still written as some value — is `written`, whose catalogue (`CATALOGUE`)
  lives here too: same metric business, and it used to sit in the engraving
  grammar, which read like drawing but decided time;
- the grid is **immutable**: the pipeline builds it from the tempo, labels the
  figures, then derives the measure-aware one with `with_meter` once the meter
  is detected. Figures only need the beat, which is why that intermediate grid
  is deliberate and not an oversight;
- `Meter` lives here rather than in `meter.py`: it is the same metric
  arithmetic. `meter.py` keeps the detection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .music import sane_bpm

# denominator -> unit duration in quarter beats
UNIT_QUARTERS = {2: 2.0, 4: 1.0, 8: 0.5}
MUSICXML_DIVISIONS = 24

@dataclass(frozen=True)
class Figure:
    """One written value, shared by notes, rests and MusicXML."""

    name: str
    type_: str
    beats: float
    rest: str
    dots: int = 0
    tuplet: tuple[int, int] | None = None
    weight: float = 1.0
    export_only: bool = False


# Plain values come before dotted values: this order resolves equal distances
# in favour of the simpler engraving. Tuplets are in the same catalogue, but
# remain a classifier choice rather than a fallback written value. The
# double-dotted values sit last among the writable ones, and they are VERY
# heavy (weight 10.0): the ±18% relative tolerance is a HUMAN-timing budget
# and a plain or dotted neighbour always sits inside a double dot's reach,
# so a light double dot captures the singer's noise around the plain values
# (measured: a 0.86-beat note read "double-dotted eighth" at weight 1.12
# AND at 1.30 — its 1.7% error only beats the noire's 14% above weight ~8).
# Weight 10 narrows the double dot to a measured ±1% band: a singer does not
# land a double dot by accident, an engraver writes one deliberately. Only
# values that convert EXACTLY at 24 divisions may enter
# the catalogue (the MusicXML types are DERIVED from it): the double-dotted
# 16th (5.25) and the dotted 32nd (4.5) cannot be expressed at 24 — a forge
# of them would write a <type> contradicting its <duration>. The plain 32nd
# stays export_only: a snap path bottoms at a 16th (6 divisions), so a 32nd
# engraved on screen comes only from the classifier, never from the fallback
# `written`.
CATALOGUE = (
    Figure("ronde", "whole", 4.0, "pause"),
    Figure("blanche", "half", 2.0, "demi-pause"),
    Figure("noire", "quarter", 1.0, "soupir"),
    Figure("croche", "eighth", 0.5, "demi-soupir"),
    Figure("double-croche", "16th", 0.25, "quart de soupir"),
    Figure("ronde pointée", "whole", 6.0, "pause pointée", dots=1, weight=1.06),
    Figure("blanche pointée", "half", 3.0, "demi-pause pointée", dots=1, weight=1.06),
    Figure("noire pointée", "quarter", 1.5, "soupir pointée", dots=1, weight=1.06),
    Figure("croche pointée", "eighth", 0.75, "demi-soupir pointée", dots=1, weight=1.06),
    Figure("double-croche pointée", "16th", 0.375, "quart de soupir pointée", dots=1, weight=1.06),
    Figure("noire double-pointée", "quarter", 1.75, "soupir double-pointée", dots=2, weight=10.0),
    Figure("croche double-pointée", "eighth", 0.875, "demi-soupir double-pointée", dots=2, weight=10.0),
    Figure("triolet de ronde", "whole", 4.0 * 2 / 3, "pause", tuplet=(3, 2), weight=1.10),
    Figure("triolet de blanche", "half", 2.0 * 2 / 3, "demi-pause", tuplet=(3, 2), weight=1.10),
    Figure("triolet de noire", "quarter", 1.0 * 2 / 3, "soupir", tuplet=(3, 2), weight=1.10),
    Figure("triolet de croche", "eighth", 0.5 * 2 / 3, "demi-soupir", tuplet=(3, 2), weight=1.10),
    Figure("triolet de double-croche", "16th", 0.25 * 2 / 3, "quart de soupir", tuplet=(3, 2), weight=1.10),
    Figure("trente-deuxième", "32nd", 0.125, "seizième de soupir", export_only=True),
)


@dataclass(frozen=True)
class Meter:
    """A meter (e.g. 4/4 default, 6/8 detected)."""

    beats_per_measure: int   # numerator
    beat_unit: int           # denominator (4 = quarter, 8 = eighth)
    score: float = 1.0       # share of onsets supporting it (compound)

    @property
    def label(self) -> str:
        return f"{self.beats_per_measure}/{self.beat_unit}"

    @property
    def is_compound(self) -> bool:
        return self.beat_unit == 8

    @property
    def n_slots(self) -> int:
        """Displayed beats (6/8 -> 2 dotted beats, otherwise the numerator)."""
        return self.beats_per_measure // 3 if self.is_compound else self.beats_per_measure

    @property
    def slot_quarters(self) -> float:
        """Duration of one displayed beat in quarter beats (6/8 -> 1.5)."""
        if self.is_compound:
            return 3.0 * UNIT_QUARTERS[8]  # temps pointé
        return UNIT_QUARTERS[self.beat_unit]

    @property
    def measure_quarters(self) -> float:
        """Measure length in quarter beats."""
        return self.n_slots * self.slot_quarters


DEFAULT_METER = Meter(4, 4)


@dataclass(frozen=True)
class BeatGrid:
    """Seconds <-> beats and measures, at one tempo and one meter.

    `meter=None` (what the detector returns when it is not sure) means the
    default 4/4, decided here and nowhere else; a missing or failed tempo
    falls back on a plausible one for the same reason — an analysis already
    paid for must not raise while it is being engraved. Both are normalized in
    `__post_init__`, so a grid is always usable and consumers never have to
    ask.
    """

    # None (no usable tempo) is normalized to 120 in __post_init__: the
    # declared type is what every consumer may rely on
    bpm: float | None
    meter: Meter = DEFAULT_METER

    def __post_init__(self) -> None:
        if self.meter is None:
            object.__setattr__(self, "meter", DEFAULT_METER)
        object.__setattr__(self, "bpm", sane_bpm(self.bpm))

    # ------------------------------------------------------------ construction
    def with_meter(self, meter: Meter | None) -> BeatGrid:
        """The same tempo on another meter (the detected one, or 4/4)."""
        return replace(self, meter=meter or DEFAULT_METER)

    # ------------------------------------------------------- seconds <-> beats
    def beats(self, seconds: float) -> float:
        """Duration in quarter beats."""
        return seconds * sane_bpm(self.bpm) / 60.0

    def seconds(self, beats: float) -> float:
        """Duration in seconds."""
        return beats * 60.0 / sane_bpm(self.bpm)

    # ------------------------------------------------------------- measures
    @property
    def slot_beats(self) -> float:
        """Displayed beat in quarter beats (1 in 4/4, 1.5 in 6/8)."""
        return self.meter.slot_quarters

    @property
    def measure_beats(self) -> float:
        """Measure length in quarter beats."""
        return self.meter.measure_quarters

    @property
    def measure_seconds(self) -> float:
        """Measure length in seconds."""
        return self.seconds(self.measure_beats)

    @property
    def slot_seconds(self) -> float:
        """Displayed beat in seconds."""
        return self.seconds(self.slot_beats)

    def position(self, seconds: float) -> tuple[int, float]:
        """(measure, displayed beat) of an instant — both 1-based."""
        measure = int(seconds // self.measure_seconds) + 1
        beat = (seconds % self.measure_seconds) / self.slot_seconds + 1.0
        return measure, round(beat, 2)

    def positions(self, notes) -> list[tuple[int, float]]:
        """(measure, beat) of each note, in the order given."""
        return [self.position(n.start) for n in notes]

    def measures(self, notes) -> int:
        """How many measures a take covers (at least one).

        Ceiling by floor division, not `math.ceil(last / measure_s)`: the
        rounded quotient would tip to the next integer where the exact one is
        a whole measure, and the count is part of the exported report.
        """
        if not notes:
            return 0
        last = max(n.end for n in notes)
        return max(1, int(-(-last // self.measure_seconds)))

    # ------------------------------------------------- integer divisions
    def step_seconds(self, per_beat: float = 4.0) -> float:
        """Seconds per grid step, `per_beat` steps to the quarter beat.

        The convention, stated once: 4 steps per beat is a sixteenth (what the
        MIDI export calls division 16), 24 is the MusicXML export's division.
        """
        return self.seconds(1.0) / per_beat

    def snap(self, seconds: float, per_beat: float = 4.0) -> float:
        """Nearest grid instant: idempotent, and bounded by half a step."""
        step = self.step_seconds(per_beat)
        return round(seconds / step) * step

    def snap_notes(self, notes, per_beat: float = 4.0) -> list:
        """Snap each note's boundaries onto the grid, in the order given.

        The detected tempo drifts from the played one; off-grid boundaries
        would engrave as 16th/32nd fragments. A span that would round below
        half a step collapses to nothing, so it keeps one step instead — the
        shortest figure a score can write. Notes come back copied.
        """
        step = self.step_seconds(per_beat)
        out = []
        for n in notes:
            s = self.snap(n.start, per_beat)
            e = self.snap(n.end, per_beat)
            out.append(replace(n, start=s, end=e if e - s >= step / 2 else s + step))
        return out

    def steps(self, seconds: float, per_beat: float = MUSICXML_DIVISIONS) -> int:
        """Position of an instant in whole steps (same `per_beat` convention)."""
        return int(round(seconds / self.step_seconds(per_beat)))

    # ------------------------------------------------- written durations
    def written(self, seconds: float) -> Figure:
        """Choose the nearest non-triplet fallback value, floored at a 16th."""
        beats = max(0.25, self.beats(seconds))
        values = _WRITTEN_VALUES
        return min(values, key=lambda f: abs(beats - f.beats))


_WRITTEN_VALUES = tuple(f for f in CATALOGUE if f.tuplet is None and not f.export_only)


__all__ = [
    "CATALOGUE", "DEFAULT_METER", "BeatGrid", "Figure", "MUSICXML_DIVISIONS",
    "Meter", "UNIT_QUARTERS",
]
