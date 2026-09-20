"""Notation: pitch spelling and staff layout, in musical units (no Qt).

The GUI staff view (`gui/staff.py`) only paints what this module computes:
the staff step + accidental a MIDI note gets in the detected key, the written
duration of each note, and where it lands on the measure grid. Kept free of
Qt so the layout grammar is unit-testable.

Staff geometry convention used everywhere below: a note's `step` counts
diatonic steps above C0 (C0 = 0, D0 = 1, ... C4 = 28). The treble staff lines
are the even steps 30 (E4), 32 (G4), 34 (B4), 36 (D5), 38 (F5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .figures import CANDIDATES
from .grid import BeatGrid

LETTERS = ("C", "D", "E", "F", "G", "A", "B")
LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Circle of fifths position of each tonic as the key detector spells it
# (sharps only), so enharmonic keys land on their engraved signature.
FIFTHS_BY_TONIC = {
    "C": 0, "C#": 7, "D": 2, "D#": -3, "E": 4, "F": -1,
    "F#": 6, "G": 1, "G#": -4, "A": 3, "A#": -2, "B": 5,
}

SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")
# Staff steps of the accidentals as engraved in treble clef (see module doc).
SHARP_STEPS = (38, 35, 39, 36, 33, 37, 34)
FLAT_STEPS = (34, 37, 33, 36, 32, 35, 31)

# Treble staff geometry (step values).
STAFF_LINES = (30, 32, 34, 36, 38)
MIDDLE_LINE = 34  # B4


@dataclass(frozen=True)
class Written:
    """How a note is written: notehead type + duration in quarter beats."""

    type_: str                       # whole | half | quarter | eighth | 16th
    beats: float                     # duration in quarter beats
    dots: int = 0
    tuplet: tuple[int, int] | None = None
    figure: str | None = None        # canonical French figure name, if known


@dataclass(frozen=True)
class Pitch:
    """A spelled pitch: letter + octave + accidental + staff step."""

    letter: str
    octave: int
    alter: int                       # -1 flat, 0 natural, +1 sharp
    step: int                        # diatonic steps above C0
    midi: int


@dataclass(frozen=True)
class StaffNote:
    """One notehead on the staff, in musical units (beats, not pixels)."""

    step: int
    alter: int
    onset: float                     # quarter beats from the start
    written: Written
    measure: int                     # 1-based
    beat: float                      # displayed beat inside the measure, 1-based
    accidental: bool = False         # engrave an explicit #/b
    preview: bool = False            # being sung right now (live)


# Rest shapes the staff view can draw (canonical French rest names). The
# 16th ("quart de soupir") falls back to an eighth shape: a 70+ ms gap is
# rarely a clean 16th anyway, and the engraving stays readable.
REST_SHAPES = ("pause", "demi-pause", "soupir", "demi-soupir")


@dataclass(frozen=True)
class StaffRest:
    """A written silence on the staff, in musical units.

    `shape` is the canonical French rest name ("soupir", "demi-pause"...)
    or None for an off-grid gap: it is then drawn as a quarter rest scaled
    by its written duration rather than lied about.
    """

    onset: float                     # quarter beats from the start
    written: Written
    measure: int                     # 1-based
    beat: float                      # displayed beat inside the measure, 1-based
    shape: str | None = None


@dataclass
class StaffLayout:
    """Everything the staff view needs to draw one system stack."""

    notes: list[StaffNote] = field(default_factory=list)
    rests: list[StaffRest] = field(default_factory=list)
    measures: int = 0
    beats_per_measure: float = 4.0   # quarter beats per measure
    slot_quarters: float = 1.0       # displayed beat unit (6/8 -> 1.5)
    fifths: int = 0                  # signed key signature
    meter_label: str = "4/4"
    key_tonic: str | None = None
    key_mode: str | None = None

    @property
    def signature(self) -> list[tuple[int, int]]:
        """Engraved key signature: [(staff step, alter), ...]."""
        return key_signature(self.fifths)

    @property
    def total_beats(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.onset + n.written.beats for n in self.notes)


# ----------------------------------------------------------------------
# Spelling
# ----------------------------------------------------------------------
def key_fifths(key) -> int:
    """Signed circle-of-fifths position of a detected key (None -> 0).

    Accepts a `keys.KeyCandidate` (tonic/mode) or a plain label like
    "C major"; a minor key shares the relative major's signature.
    """
    if key is None:
        return 0
    tonic = getattr(key, "tonic", None)
    mode = getattr(key, "mode", None)
    if tonic is None:  # a "C major" label
        tonic, _, mode = str(key).partition(" ")
    fifths = FIFTHS_BY_TONIC.get(str(tonic), 0)
    return fifths - 3 if mode == "minor" else fifths


def key_pitch_classes(fifths: int) -> dict[int, tuple[str, int]]:
    """Map each diatonic pitch class of the key to (letter, alter)."""
    sharps = SHARP_ORDER[: max(0, fifths)]
    flats = FLAT_ORDER[: max(0, -fifths)]
    out: dict[int, tuple[str, int]] = {}
    for letter, pc in LETTER_PC.items():
        alter = 1 if letter in sharps else -1 if letter in flats else 0
        out[(pc + alter) % 12] = (letter, alter)
    return out


def key_signature(fifths: int) -> list[tuple[int, int]]:
    """Accidentals of the key signature: [(staff step, alter), ...]."""
    fifths = max(-7, min(7, int(fifths)))
    if fifths > 0:
        return [(step, 1) for step in SHARP_STEPS[:fifths]]
    return [(step, -1) for step in FLAT_STEPS[:-fifths]]


def spell(midi: int, fifths: int = 0) -> Pitch:
    """Spell a MIDI note in the given key.

    Diatonic notes keep their letter; chromatic notes take the neighbour
    letter one accidental away ("C#" in C major, "Db" in F major).
    """
    pc = int(midi) % 12
    scale = key_pitch_classes(fifths)
    if pc in scale:
        letter, alter = scale[pc]
    else:
        # Chromatic note: take the neighbouring letter one accidental away.
        # delta=-1 raises the letter below (a sharp spelling: F# in C),
        # delta=+1 lowers the letter above (a flat spelling: Gb in F).
        # Sharpen in sharp keys and in C, flatten in flat keys.
        directions = ((1, -1), (-1, 1)) if fifths < 0 else ((-1, 1), (1, -1))
        letter, alter = "C", 0
        for delta, shift in directions:
            neighbor = scale.get((pc + delta) % 12)
            if neighbor is not None:
                letter, alter = neighbor[0], neighbor[1] + shift
                break
    octave = (int(midi) - LETTER_PC[letter] - alter) // 12 - 1
    step = octave * 7 + LETTERS.index(letter)
    return Pitch(letter=letter, octave=octave, alter=alter, step=step, midi=int(midi))


def accidental_needed(alter: int, letter: str, fifths: int) -> bool:
    """True when a notehead needs an explicit accidental in this key."""
    sharps = SHARP_ORDER[: max(0, fifths)]
    flats = FLAT_ORDER[: max(0, -fifths)]
    key_alter = 1 if letter in sharps else -1 if letter in flats else 0
    return alter != key_alter


# ----------------------------------------------------------------------
# Durations
# ----------------------------------------------------------------------
FIGURE_BEATS = {c.name: c.beats for c in CANDIDATES}
_FIGURE_WRITTEN = {
    c.name: Written(c.type_, c.beats, c.dots, c.tuplet, c.name) for c in CANDIDATES
}


def _written(figure, seconds: float, grid: BeatGrid, table: dict) -> Written:
    """A figure's own written value, else the grid's nearest drawable one."""
    known = table.get(figure)
    if known is not None:
        return known
    type_, beats, dots = grid.written(seconds)
    return Written(type_, beats, dots)


def notated_duration(note, grid: BeatGrid) -> Written:
    """Written duration of a note: its figure, else what the grid decides."""
    return _written(getattr(note, "figure", None), note.duration, grid, _FIGURE_WRITTEN)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
# Written values of the canonical French rest figures (staff + XML share them).
_REST_WRITTEN = {
    "pause": Written("whole", 4.0, 0),
    "demi-pause": Written("half", 2.0, 0),
    "soupir": Written("quarter", 1.0, 0),
    "demi-soupir": Written("eighth", 0.5, 0),
    "quart de soupir": Written("16th", 0.25, 0),
    "pause pointée": Written("whole", 6.0, 1),
    "demi-pause pointée": Written("half", 3.0, 1),
    "soupir pointée": Written("quarter", 1.5, 1),
    "demi-soupir pointée": Written("eighth", 0.75, 1),
}


def notated_rest_duration(rest, grid: BeatGrid) -> Written:
    """Written duration of a silence, same grammar as the notes.

    The rest's own figure ("soupir"...) is authoritative when present;
    otherwise the grid's nearest drawable value for the real duration.
    """
    return _written(getattr(rest, "figure", None), rest.duration, grid, _REST_WRITTEN)


def layout_staff(
    notes,
    *,
    grid: BeatGrid,
    key=None,
    positions=None,
    rests=None,
    preview=None,
) -> StaffLayout:
    """Place notes and silences on the measure grid (quarter beats, 1-based).

    `positions` are the pipeline's (measure, beat) pairs; without them (the
    live take, whose notes have no grid yet) notes are chained by their
    written durations.

    `rests` are the core's Rest objects (seconds + figure): each one that
    closes before a note begins is engraved in the written gap before that
    note. On the grid path the rest must fit the gap left by the figures
    (a tiny remaining seam is dropped); on the chained path the rest
    CONSUMES written time, so the following notes start after the silence
    instead of glued to the previous note.
    """
    fifths = key_fifths(key)
    beats_per_measure = grid.measure_beats
    slot_quarters = grid.slot_beats
    rest_list = sorted(rests, key=lambda r: r.start) if rests else []

    out: list[StaffNote] = []
    silences: list[StaffRest] = []
    cursor = 0.0
    ri = 0
    for i, note in enumerate(notes):
        written = notated_duration(note, grid)
        on_grid = positions is not None and i < len(positions)
        if on_grid:
            measure_no, beat_no = positions[i]
            onset = (measure_no - 1) * beats_per_measure + (beat_no - 1) * slot_quarters
        else:
            onset = cursor
        while ri < len(rest_list) and rest_list[ri].end <= note.start + 0.02:
            r = rest_list[ri]
            ri += 1
            w_rest = notated_rest_duration(r, grid)
            if on_grid:
                if onset - cursor >= 0.25:  # a drawable seam exists
                    silences.append(
                        _place_rest(r, w_rest, cursor, beats_per_measure, slot_quarters)
                    )
            else:
                silences.append(
                    _place_rest(r, w_rest, cursor, beats_per_measure, slot_quarters)
                )
                cursor += w_rest.beats
                onset = cursor
        cursor = onset + written.beats
        out.append(
            _place(note, written, onset, beats_per_measure, slot_quarters, fifths)
        )

    if preview is not None:
        written = notated_duration(preview, grid)
        out.append(
            _place(
                preview,
                written,
                cursor,
                beats_per_measure,
                slot_quarters,
                fifths,
                preview=True,
            )
        )

    measures = max(1, int(out[-1].onset // beats_per_measure) + 1) if out else 0
    return StaffLayout(
        notes=out,
        rests=silences,
        measures=measures,
        beats_per_measure=beats_per_measure,
        slot_quarters=slot_quarters,
        fifths=fifths,
        meter_label=grid.meter.label,
        key_tonic=getattr(key, "tonic", None) or (str(key).partition(" ")[0] or None),
        key_mode=getattr(key, "mode", None) or (str(key).partition(" ")[2] or None),
    )


def _place_rest(rest, written, onset, beats_per_measure, slot_quarters) -> StaffRest:
    measure = int(onset // beats_per_measure) + 1
    beat = (onset - (measure - 1) * beats_per_measure) / slot_quarters + 1.0
    shape = getattr(rest, "figure", None)
    return StaffRest(
        onset=onset,
        written=written,
        measure=measure,
        beat=round(beat, 2),
        shape=shape,
    )


def _place(note, written, onset, beats_per_measure, slot_quarters, fifths, preview=False):
    pitch = spell(int(note.midi), fifths)
    measure = int(onset // beats_per_measure) + 1
    beat = (onset - (measure - 1) * beats_per_measure) / slot_quarters + 1.0
    return StaffNote(
        step=pitch.step,
        alter=pitch.alter,
        onset=onset,
        written=written,
        measure=measure,
        beat=round(beat, 2),
        # a note outside the key needs its own accidental engraved in front
        accidental=bool(pitch.alter) and accidental_needed(pitch.alter, pitch.letter, fifths),
        preview=preview,
    )


__all__ = [
    "FIGURE_BEATS",
    "LETTERS",
    "MIDDLE_LINE",
    "REST_SHAPES",
    "STAFF_LINES",
    "Pitch",
    "StaffLayout",
    "StaffNote",
    "StaffRest",
    "Written",
    "accidental_needed",
    "key_fifths",
    "key_pitch_classes",
    "key_signature",
    "layout_staff",
    "notated_duration",
    "notated_rest_duration",
    "spell",
]
