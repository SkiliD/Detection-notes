"""Tests export MIDI : meta-events tempo/armure + notes relues."""

import mido
import pytest

from v2m.core.keys import detect_key
from v2m.core.midi_export import build_midi, quantize_step_s
from v2m.core.notes import Note


def _sample_notes() -> list[Note]:
    return [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),
        Note(start=0.5, end=1.0, midi=64, confidence=0.8),
        Note(start=1.2, end=2.0, midi=67, confidence=0.7),
    ]


def test_midi_has_tempo_meta(tmp_path):
    mid = build_midi(_sample_notes(), bpm=120.0, key=None)
    path = str(tmp_path / "out.mid")
    mid.save(path)
    reloaded = mido.MidiFile(path)
    tempos = [m for t in reloaded.tracks for m in t if m.type == "set_tempo"]
    assert len(tempos) == 1
    assert abs(mido.tempo2bpm(tempos[0].tempo) - 120.0) < 0.1


def test_midi_has_key_signature(tmp_path):
    notes = _sample_notes()
    key = detect_key(notes)
    mid = build_midi(notes, bpm=120.0, key=key)
    path = str(tmp_path / "out.mid")
    mid.save(path)
    reloaded = mido.MidiFile(path)
    keys = [m for t in reloaded.tracks for m in t if m.type == "key_signature"]
    assert len(keys) == 1
    # mido décode l'armure en chaîne ('C', 'Am', 'F#', ...)
    k = keys[0].key
    assert isinstance(k, str) and k
    valid = {"Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"}
    base = k[:-1] if k.endswith("m") else k
    assert base in valid


def test_midi_note_count_and_pitch(tmp_path):
    mid = build_midi(_sample_notes(), bpm=120.0, key=None)
    path = str(tmp_path / "out.mid")
    mid.save(path)
    reloaded = mido.MidiFile(path)
    ons = [m for t in reloaded.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    assert sorted(m.note for m in ons) == [60, 64, 67]


def test_midi_transpose_and_velocity():
    mid = build_midi(_sample_notes(), bpm=120.0, key=None, transpose=12)
    ons = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    assert sorted(m.note for m in ons) == [72, 76, 79]
    # vélocité liée à la confiance (0.9 -> élevée)
    assert all(40 <= m.velocity <= 127 for m in ons)


def test_quantize_aligns_to_grid():
    step = quantize_step_s(120.0, division=16)  # double-croche à 120 BPM
    assert abs(step - 0.125) < 1e-9
    mid = build_midi(
        [Note(start=0.13, end=0.49, midi=60, confidence=0.9)],
        bpm=120.0,
        key=None,
        quantize_s=step,
    )
    ons = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    # 0.13 -> arrondi à 0.125 s = 1 double-croche = 120 ticks (480 tpq, 120 BPM)
    assert ons[0].time == 120
