"""MIDI file writing: notes + tempo + key signatures + pedal (CC64)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mido

from .grid import BeatGrid
from .keys import KeyResult
from .music import NOTE_NAMES
from .notes import Note
from .playback import HELD_MIN_S, PEDAL_TAIL_FACTOR, PEDAL_TAIL_MAX_S

MidiEventKind = Literal["note_on", "note_off", "pedal_down", "pedal_up"]


@dataclass(frozen=True)
class _MidiEvent:
    """One timestamped MIDI event, with explicit ordering at shared times."""

    time_s: float
    order: int
    kind: MidiEventKind
    pitch: int = 0
    value: int = 0


# Key names accepted by mido for the Key Signature meta event (FF 59).
# All minor keys exist as sharps; three major keys are written with flats.
_MAJOR_MIDO = {
    "C": "C",
    "C#": "C#",
    "D": "D",
    "D#": "Eb",
    "E": "E",
    "F": "F",
    "F#": "F#",
    "G": "G",
    "G#": "Ab",
    "A": "A",
    "A#": "Bb",
    "B": "B",
}


def _mido_key(tonic: str, mode: str, transpose: int = 0) -> str:
    """Encode (tonic, mode) as a mido key string: e.g. ('A','minor') -> 'Am'."""
    tonic = NOTE_NAMES[(NOTE_NAMES.index(tonic) + transpose) % 12]
    if mode == "minor":
        return f"{tonic}m"
    return _MAJOR_MIDO.get(tonic, "C")


def build_midi(
    notes: list[Note],
    grid: BeatGrid,
    key: KeyResult | None,
    tpq: int = 480,
    quantize_s: float | None = None,
    transpose: int = 0,
) -> mido.MidiFile:
    """Build the MidiFile.

    - Track 0: meta (tempo, best key signature, end)
    - Track 1: voice notes + sustain pedal (CC64) on held notes
    - `grid` carries the tempo (its `bpm` goes into the tempo meta) and turns
      seconds into ticks; `quantize_s` (e.g. a sixteenth) snaps starts/ends.

    The pedal mirrors the RENDER's behavior for held notes (>= HELD_MIN_S):
    damper down at the note's start, up at written end + tail (the same
    PEDAL_TAIL_FACTOR / PEDAL_TAIL_MAX_S), so a file played on any synth
    keeps the resonance the built-in piano plays.
    """
    mid = mido.MidiFile(ticks_per_beat=tpq)

    # --- meta track: tempo + key ---
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(grid.bpm), time=0))
    if key is not None and key.best.score > 0:
        # one signature only: alternative candidates live in the JSON report
        meta.append(
            mido.MetaMessage(
                "key_signature", key=_mido_key(key.best.tonic, key.best.mode, transpose), time=0
            )
        )
    meta.append(mido.MetaMessage("end_of_track", time=0))

    # --- voice track: notes ---
    voice = mido.MidiTrack()
    mid.tracks.append(voice)
    voice.append(mido.MetaMessage("track_name", name="Voice", time=0))

    written_notes: list[tuple[float, float, int, int]] = []
    for n in notes:
        start, end = n.start, n.end
        if quantize_s and quantize_s > 0:
            start = round(start / quantize_s) * quantize_s
            end = max(start + quantize_s, round(end / quantize_s) * quantize_s)
        pitch = max(0, min(127, int(n.midi) + transpose))
        velocity = max(1, min(127, int(round(40 + 80 * n.confidence))))
        written_notes.append((start, end, pitch, velocity))

    # One key sounds one note at a time. A same-pitch repeat can start BEFORE
    # the note before it ends — measured on chant.wav, where the analysis leaves
    # a 20 ms window shared — and then the earlier note's OFF closes the repeat
    # at its own attack: the repeat lasts the OVERLAP instead of its length. The
    # earlier note is released AT the repeat's attack, so both events land on
    # one tick, where the OFF sorts first (see the ordering below).
    sounding: dict[int, int] = {}
    for index in sorted(range(len(written_notes)), key=lambda i: written_notes[i][:2]):
        start, end, pitch, velocity = written_notes[index]
        held = sounding.get(pitch)
        if held is not None:
            held_start, held_end, _held_pitch, held_velocity = written_notes[held]
            if held_end > start:
                written_notes[held] = (held_start, start, pitch, held_velocity)
        sounding[pitch] = index

    events: list[_MidiEvent] = []
    for start, end, pitch, velocity in written_notes:
        events.append(_MidiEvent(start, order=2, kind="note_on", pitch=pitch, value=velocity))
        events.append(_MidiEvent(end, order=1, kind="note_off", pitch=pitch))

        # Quantization (and the release above) changes the written note
        # duration, so its sustain window must be derived from these same
        # written event times.
        if end - start >= HELD_MIN_S:
            tail = min(PEDAL_TAIL_MAX_S, PEDAL_TAIL_FACTOR * (end - start))
            events.append(_MidiEvent(start, order=0, kind="pedal_down", value=127))
            events.append(_MidiEvent(end + tail, order=3, kind="pedal_up", value=0))

    # CC64 is channel-wide. Emit only the edges where the active set changes
    # between empty and non-empty; suppressed edges still advance the tick
    # clock so rounded note timings remain unchanged.
    sec_per_tick = grid.seconds(1.0) / tpq
    events.sort(key=lambda event: (event.time_s, event.order))

    # Events are given their ABSOLUTE tick first, then ordered at that tick by
    # kind (pedal down, note off, note on, pedal up). A repeat the product pins
    # to the same instant — the live stream clamps a split to the previous end,
    # a drag can land a bar on its neighbour's edge, and the release above moves
    # an overlapped note's end onto the repeat's attack — gets ONE tick, so the
    # order inside it decides: an ON written before the OFF of the same pitch
    # would close the new note at the instant it opens. A suppressed pedal edge
    # still advances the clock, which is why the tick is carried per message
    # instead of folded into a delta.
    written: list[tuple[int, int, mido.Message]] = []
    last_t = 0.0
    active_pedals = 0
    clock_ticks = 0
    for event in events:
        dt_s = max(0.0, event.time_s - last_t)
        ticks = int(round(dt_s / sec_per_tick))
        last_t = event.time_s
        clock_ticks += ticks

        if event.kind == "pedal_down":
            active_pedals += 1
            if active_pedals > 1:
                continue
            written.append(
                (
                    clock_ticks,
                    event.order,
                    mido.Message("control_change", control=64, value=127, time=0),
                )
            )
            continue

        if event.kind == "pedal_up":
            active_pedals -= 1
            if active_pedals > 0:
                continue
            written.append(
                (
                    clock_ticks,
                    event.order,
                    mido.Message("control_change", control=64, value=0, time=0),
                )
            )
            continue

        if event.kind == "note_on":
            message = mido.Message("note_on", note=event.pitch, velocity=event.value, time=0)
        else:
            message = mido.Message("note_off", note=event.pitch, velocity=0, time=0)
        written.append((clock_ticks, event.order, message))

    written.sort(key=lambda item: item[0])  # stable: time order inside a tick
    cursor = 0
    for tick, _order, message in written:
        message.time = tick - cursor
        cursor = tick
        voice.append(message)
    voice.append(mido.MetaMessage("end_of_track", time=0))
    return mid
