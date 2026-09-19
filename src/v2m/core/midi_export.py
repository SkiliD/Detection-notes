"""MIDI file writing: notes + tempo + key signatures."""

from __future__ import annotations

import mido


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
    bpm: float,
    key,
    tpq: int = 480,
    quantize_s: float | None = None,
    transpose: int = 0,
) -> mido.MidiFile:
    """Build the MidiFile.

    - Track 0: meta (tempo, best key signature, end)
    - Track 1: voice notes
    - `quantize_s`: when given (e.g. a sixteenth duration), snaps
      starts/ends to the grid.
    """
    mid = mido.MidiFile(ticks_per_beat=tpq)

    # --- meta track: tempo + key ---
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
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

    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

    last_t = 0.0
    sec_per_tick = 60.0 / (bpm * tpq)
    for t, kind, pitch, vel in events:
        dt_s = max(0.0, t - last_t)
        ticks = int(round(dt_s / sec_per_tick))
        last_t = t
        if kind == "on":
            voice.append(mido.Message("note_on", note=pitch, velocity=vel, time=ticks))
        else:
            voice.append(mido.Message("note_off", note=pitch, velocity=0, time=ticks))
    voice.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def quantize_step_s(bpm: float, division: int = 16) -> float:
    """Duration of one grid division (16 = sixteenth) in seconds."""
    return 60.0 / bpm / (division / 4.0)
