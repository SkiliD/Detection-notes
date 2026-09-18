"""Écriture d'un fichier MIDI : notes + tempo + armures de tonalité."""

from __future__ import annotations

import mido


# Noms de tonalités acceptés par mido pour le meta-event Key Signature (FF 59).
# Toutes les mineures existent en dièses ; trois majeures se notent en bémols.
_MAJOR_MIDO = {
    "C": "C", "C#": "C#", "D": "D", "D#": "Eb", "E": "E", "F": "F",
    "F#": "F#", "G": "G", "G#": "Ab", "A": "A", "A#": "Bb", "B": "B",
}


def _mido_key(tonic: str, mode: str) -> str:
    """Encode (tonique, mode) en chaîne mido : ex. ('A','minor') -> 'Am'."""
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
    """Construit le MidiFile.

    - Piste 0 : méta (tempo, armures des candidates de tonalité, fin)
    - Piste 1 : notes de la voix
    - `quantize_s` : si fourni (ex. durée d'une double-croche), aligne
      débuts/fins sur la grille.
    """
    mid = mido.MidiFile(ticks_per_beat=tpq)

    # --- Piste méta : tempo + tonalité ---
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    if key is not None and key.best.score > 0:
        # une seule armure : les candidates alternatives figurent dans le rapport JSON
        meta.append(
            mido.MetaMessage(
                "key_signature", key=_mido_key(key.best.tonic, key.best.mode), time=0
            )
        )
    meta.append(mido.MetaMessage("end_of_track", time=0))

    # --- Piste voix : notes ---
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
    """Durée d'une division de mesure (16 = double-croche) en secondes."""
    return 60.0 / bpm / (division / 4.0)


def save_midi(mid: mido.MidiFile, path: str) -> None:
    mid.save(path)
