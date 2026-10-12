"""MIDI file writing: notes + tempo + key signatures + pedal (CC64)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mido

from .grid import BeatGrid
from .keys import KeyResult
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
    "C": "C", "C#": "C#", "D": "D", "D#": "Eb", "E": "E", "F": "F",
    "F#": "F#", "G": "G", "G#": "Ab", "A": "A", "A#": "Bb", "B": "B",
}


def _mido_key(tonic: str, mode: str) -> str:
    """Encode (tonic, mode) as a mido key string: e.g. ('A','minor') -> 'Am'."""
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
                "key_signature", key=_mido_key(key.best.tonic, key.best.mode), time=0
            )
        )
    meta.append(mido.MetaMessage("end_of_track", time=0))

    # --- voice track: notes ---
    voice = mido.MidiTrack()
    mid.tracks.append(voice)
    voice.append(mido.MetaMessage("track_name", name="Voice", time=0))

    events: list[_MidiEvent] = []
    for n in notes:
        start, end = n.start, n.end
        if quantize_s and quantize_s > 0:
            start = round(start / quantize_s) * quantize_s
            end = max(start + quantize_s, round(end / quantize_s) * quantize_s)
        pitch = max(0, min(127, int(n.midi) + transpose))
        velocity = max(1, min(127, int(round(40 + 80 * n.confidence))))
        events.append(_MidiEvent(start, order=1, kind="note_on", pitch=pitch, value=velocity))
        events.append(_MidiEvent(end, order=2, kind="note_off", pitch=pitch))

        # Quantization changes the written note duration, so its sustain
        # window must be derived from these same written event times.
        if end - start >= HELD_MIN_S:
            tail = min(PEDAL_TAIL_MAX_S, PEDAL_TAIL_FACTOR * (end - start))
            events.append(_MidiEvent(start, order=0, kind="pedal_down", value=127))
            events.append(_MidiEvent(end + tail, order=3, kind="pedal_up", value=0))

    # CC64 is channel-wide. Emit only the edges where the active set changes
    # between empty and non-empty; suppressed edges still advance the tick
    # clock so rounded note timings remain unchanged.
    sec_per_tick = grid.seconds(1.0) / tpq
    events.sort(key=lambda event: (event.time_s, event.order))

    last_t = 0.0
    pending_ticks = 0
    active_pedals = 0
    for event in events:
        dt_s = max(0.0, event.time_s - last_t)
        ticks = int(round(dt_s / sec_per_tick))
        last_t = event.time_s

        if event.kind == "pedal_down":
            active_pedals += 1
            if active_pedals > 1:
                pending_ticks += ticks
                continue
            voice.append(
                mido.Message(
                    "control_change", control=64, value=127,
                    time=ticks + pending_ticks,
                )
            )
            pending_ticks = 0
            continue

        if event.kind == "pedal_up":
            active_pedals -= 1
            if active_pedals > 0:
                pending_ticks += ticks
                continue
            voice.append(
                mido.Message(
                    "control_change", control=64, value=0,
                    time=ticks + pending_ticks,
                )
            )
            pending_ticks = 0
            continue

        ticks += pending_ticks
        pending_ticks = 0
        if event.kind == "note_on":
            voice.append(
                mido.Message("note_on", note=event.pitch, velocity=event.value, time=ticks)
            )
        else:
            voice.append(mido.Message("note_off", note=event.pitch, velocity=0, time=ticks))
    voice.append(mido.MetaMessage("end_of_track", time=0))
    return mid
