"""Tests pipeline complet + rapports (bout en bout)."""

import json
import types
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from conftest import SR, silence, sine_note
from v2m.core.pipeline import analyze_signal
from v2m.core.report import export_result, live_export_dir, live_take_name


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


def test_export_uses_the_analysis_grid_without_changing_the_artifacts(tmp_path):
    """The exporter must honor the metric frame produced by the pipeline."""
    res = analyze_signal(_melody(), SR, source="synthetic")
    assert res.grid is not None
    carried = tmp_path / "carried"
    rebuilt = tmp_path / "rebuilt"
    export_result(res, carried)
    res.grid = type(res.grid)(res.grid.bpm, res.grid.meter)
    export_result(res, rebuilt)
    for suffix in (".json", ".mid", ".musicxml"):
        assert (carried.with_suffix(suffix)).read_bytes() == (rebuilt.with_suffix(suffix)).read_bytes()


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


def test_live_take_name_is_unique_and_recognisable():
    n1, n2 = live_take_name(), live_take_name()
    assert n1.startswith("live_")
    assert n1 == n2  # same second = same take, distinct only across seconds


def test_cli_live_session_exports_into_a_per_take_folder(tmp_path, monkeypatch):
    """Le chemin CLI `v2m live` finit dans le sous-dossier dédié, comme la GUI."""
    import v2m.live as live_mod
    from v2m.live import LiveSession

    monkeypatch.chdir(tmp_path)  # the take lands under tmp, not the repo root
    session = LiveSession()
    # Pretend a capture happened: enough signal to pass the length gate.
    # The fake capture lives ON the take (its single owner): the adapter
    # no longer mirrors it — export() goes through LiveTake.finalize.
    monkeypatch.setattr(
        session.take, "capture",
        types.SimpleNamespace(
            signal=lambda: sine_note(220.0, 1.5),
            elapsed=lambda: 1.5,
            stop=lambda: None,
        ),
    )
    returned = session.export()
    assert returned is not None
    take_dir = tmp_path / "prises_live" / session.out_stem
    assert Path(returned).name == f"{session.out_stem}.mid"
    assert Path(returned).exists()
    for ext in (".mid", ".musicxml", ".json", ".csv"):
        assert (take_dir / f"{session.out_stem}{ext}").exists(), f"{ext} manquant"


def test_live_exports_are_isolated_per_take(tmp_path):
    """Un live écrit dans `prises_live/<stem>/`, à l'abri des autres prises."""
    res = analyze_signal(_melody(), SR, source="live")
    stem = live_take_name()
    export_result(res, live_export_dir(stem) / stem)
    take_dir = live_export_dir(stem)
    assert take_dir.is_dir()
    assert take_dir.parent.name == "prises_live"
    assert sorted(p.name for p in take_dir.glob(f"{stem}.*")) == [
        f"{stem}.csv", f"{stem}.json", f"{stem}.mid", f"{stem}.musicxml",
    ]


def test_wav_file_roundtrip(tmp_path):
    """E2E : écrit un WAV, l'analyse via le chemin fichier."""
    from v2m.core.audio_io import load_file

    wav_path = str(tmp_path / "melody.wav")
    sf.write(wav_path, _melody(), SR)
    y = load_file(wav_path)
    res = analyze_signal(y, SR, source=wav_path)
    assert [n.midi for n in res.notes] == [57, 60, 64, 67]
