"""MIDI file writing: notes + tempo + key signatures + pedal (CC64)."""

from __future__ import annotations

import mido

from .grid import BeatGrid
from .playback import HELD_MIN_S, PEDAL_TAIL_FACTOR, PEDAL_TAIL_MAX_S


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
    notes,
    grid: BeatGrid,
    key,
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

    events: list[tuple[float, str, int, int]] = []  # (time_s, kind, pitch, vel)
    pedal: list[tuple[float, int]] = []             # (time_s, value 0/127)
    for n in notes:
        start, end = n.start, n.end
        if quantize_s and quantize_s > 0:
            start = round(start / quantize_s) * quantize_s
            end = max(start + quantize_s, round(end / quantize_s) * quantize_s)
        pitch = int(n.midi) + transpose
        pitch = max(0, min(127, pitch))
        vel = max(1, min(127, int(round(40 + 80 * n.confidence))))
        events.append((start, "on", pitch, vel))
        events.append((end, "off", pitch, 0))
        # the sustain pedal rides WITH the note: down at the attack, up
        # after the tail (same tail arithmetic as playback.render_notes)
        if n.end - n.start >= HELD_MIN_S:
            tail = min(PEDAL_TAIL_MAX_S, PEDAL_TAIL_FACTOR * (n.end - n.start))
            pedal.append((start, 127))
            pedal.append((end + tail, 0))

    # Pedal CCs are merged into the event timeline: CC64 down BEFORE the
    # note_on at the same instant (order key 0 < 1), so the string rings
    # from the attack; up after the note_off + tail.
    last_t = 0.0
    sec_per_tick = grid.seconds(1.0) / tpq
    merged: list[tuple[float, int, object]] = []
    for t, kind, pitch, vel in events:
        merged.append((t, 1 if kind == "on" else 2, ("note", pitch, vel)))
    for t, value in pedal:
        merged.append((t, 0, ("cc64", value)))
    merged.sort(key=lambda e: (e[0], e[1]))

    last_t = 0.0
    for t, _order, payload in merged:
        dt_s = max(0.0, t - last_t)
        ticks = int(round(dt_s / sec_per_tick))
        last_t = t
        if payload[0] == "note":
            _, pitch, vel = payload
            if vel > 0:
                voice.append(mido.Message("note_on", note=pitch, velocity=vel, time=ticks))
            else:
                voice.append(mido.Message("note_off", note=pitch, velocity=0, time=ticks))
        else:
            voice.append(mido.Message("control_change", control=64, value=payload[1], time=ticks))
    voice.append(mido.MetaMessage("end_of_track", time=0))
    return mid
