"""Full pipeline + reports tests (end to end)."""

import csv
import json
import types
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
from conftest import SR, silence, sine_note

from v2m.core.pipeline import analyze_signal
from v2m.core.report import ARTIFACT_EXTENSIONS, export_result
from v2m.core.takes import (
    live_export_dir,
    live_take_name,
    parse_take_name,
)


def _melody() -> np.ndarray:
    """Simple melody: A3, C4, E4, G4 (an A-minor arpeggio)."""
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
    """One interface writes the four artifacts + the report dict."""
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


def test_the_take_name_grammar_round_trips_through_its_owner():
    """Whoever writes take names can re-read them — one grammar.

    `parse_take_name` is the reader of what `live_take_name` writes:
    grouping them by date (coherence follow-up) does not re-encode the
    format. A name that is not a take returns None instead of being
    guessed.
    """
    import time as _time

    stamp = parse_take_name(live_take_name())
    assert stamp is not None
    date, clock = stamp
    assert date == _time.strftime("%Y%m%d") and len(clock) == 6
    # the passed path is reduced to its name (a folder may precede)
    assert parse_take_name(Path("prises_live") / live_take_name()) == stamp
    for not_a_take in ("chant", "live_", "live_2026_000000",
                       "live_20260926_00063", "prise2"):
        assert parse_take_name(not_a_take) is None, not_a_take


def test_cli_live_session_exports_into_a_per_take_folder(tmp_path, monkeypatch):
    """The CLI path `v2m live` ends in the dedicated sub-folder, like the GUI."""
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
    for ext in ARTIFACT_EXTENSIONS:
        assert (take_dir / f"{session.out_stem}{ext}").exists(), f"{ext} missing"


def test_live_exports_are_isolated_per_take(tmp_path):
    """A live writes into `prises_live/<stem>/`, sheltered from other takes."""
    res = analyze_signal(_melody(), SR, source="live")
    stem = live_take_name()
    export_result(res, live_export_dir(stem) / stem)
    take_dir = live_export_dir(stem)
    assert take_dir.is_dir()
    assert take_dir.parent.name == "prises_live"
    # the constant IS the folder contract: the disk must obey it
    assert sorted(p.name for p in take_dir.glob(f"{stem}.*")) == sorted(
        f"{stem}{ext}" for ext in ARTIFACT_EXTENSIONS
    )


def test_coherence_csv_next_to_the_take(tmp_path):
    """The coherence block also ships as CSV: swallowed phrases followed
    from one session to the next in a spreadsheet, without re-reading the
    JSON."""
    res = analyze_signal(_melody(), SR, source="synthetic")
    stem = tmp_path / "prise"
    export_result(res, stem)
    with open(tmp_path / "prise.coherence.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "start_s", "duration_s", "midi", "inner_range_st",
        "secondary_plateau_s", "secondary_plateau_start_s",
        "secondary_plateau_midi", "secondary_plateau_offset_st",
    ]
    # clean melody: one row per suspect, header alone when none
    rep = json.loads(open(tmp_path / "prise.json", encoding="utf-8").read())
    assert len(rows) == 1 + len(rep["coherence"]["suspects"])
    for row, s in zip(rows[1:], rep["coherence"]["suspects"], strict=True):
        assert float(row[0]) == s["start_s"]
        assert int(row[2]) == s["midi"]


def test_wav_file_roundtrip(tmp_path):
    """E2E: writes a WAV, analyzes it through the file path."""
    from v2m.core.audio_io import load_file

    wav_path = str(tmp_path / "melody.wav")
    sf.write(wav_path, _melody(), SR)
    y = load_file(wav_path)
    res = analyze_signal(y, SR, source=wav_path)
    assert [n.midi for n in res.notes] == [57, 60, 64, 67]


def test_threshold_policy_has_one_owner():
    """The analysis thresholds have ONE policy: pipeline FILE_CONF / LIVE_CONF.

    The regression to refuse: a default rewritten locally (the GUI
    hardcoded its own 0.6, the CLI and LiveTake theirs) — changing the
    policy would become a copy hunt. min_note_s belongs to notes.py (the
    domain), bpm_range to tempo.py (the fallback is its trade): the CLI
    and the dataclass read them, nobody rewrites them.
    """
    from v2m import cli
    from v2m.core.live_session import LiveTake
    from v2m.core.notes import DEFAULT_MIN_NOTE_S
    from v2m.core.pipeline import FILE_CONF, LIVE_CONF, AnalyzeOptions
    from v2m.core.tempo import BPM_RANGE

    assert AnalyzeOptions().min_conf == FILE_CONF
    assert AnalyzeOptions().min_note_s == DEFAULT_MIN_NOTE_S
    assert AnalyzeOptions().bpm_range == BPM_RANGE
    assert LiveTake().min_conf == LIVE_CONF
    parser = cli.build_parser()
    analyzed = parser.parse_args(["analyze", "x.wav"])
    live = parser.parse_args(["live"])
    assert analyzed.min_conf == FILE_CONF
    assert analyzed.min_note == int(DEFAULT_MIN_NOTE_S * 1000)
    assert live.min_conf == LIVE_CONF
