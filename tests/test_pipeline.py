"""Tests pipeline complet + rapports (bout en bout)."""

import json

import mido
import numpy as np
import soundfile as sf

from conftest import SR, silence, sine_note
from v2m.core.pipeline import analyze_signal
from v2m.core.report import export_result


def _melody() -> np.ndarray:
    """Mélodie simple : A3, C4, E4, G4 (arpège de La mineur)."""
    parts = []
    for hz in (220.0, 261.63, 329.63, 392.0):
        parts.append(sine_note(hz, 0.5))
        parts.append(silence(0.1))
    return np.concatenate(parts)


def test_analyze_signal_full():
    y = _melody()
    res = analyze_signal(y, SR, source="synthetic")
    assert len(res.notes) == 4
    assert [n.midi for n in res.notes] == [57, 60, 64, 67]
    assert res.tempo is not None and res.tempo.bpm > 0
    assert res.key is not None
    assert res.duration_s > 2.0


def test_export_result_one_interface(tmp_path):
    """Une seule interface écrit les quatre artefacts + le dict de rapport."""
    y = _melody()
    res = analyze_signal(y, SR, source="synthetic")

    export_result(res, tmp_path / "out")

    reloaded = mido.MidiFile(str(tmp_path / "out.mid"))
    ons = [m for t in reloaded.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    assert sorted(m.note for m in ons) == [57, 60, 64, 67]

    loaded = json.loads(open(tmp_path / "out.json", encoding="utf-8").read())
    assert loaded["n_notes"] == 4
    assert len(loaded["notes"]) == 4
    assert all("figure" in n for n in loaded["notes"])
    assert loaded["quantized"] is False
    assert loaded["transpose_semitones"] == 0

    lines = open(tmp_path / "out.csv", encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 5  # header + 4 notes
    assert lines[0].startswith("index,measure,beat,start_s")

    xml = open(tmp_path / "out.musicxml", encoding="utf-8").read()
    assert xml.count("<measure ") >= 1


def test_wav_file_roundtrip(tmp_path):
    """E2E : écrit un WAV, l'analyse via le chemin fichier."""
    from v2m.core.audio_io import load_file

    wav_path = str(tmp_path / "melody.wav")
    sf.write(wav_path, _melody(), SR)
    y = load_file(wav_path)
    res = analyze_signal(y, SR, source=wav_path)
    assert [n.midi for n in res.notes] == [57, 60, 64, 67]
