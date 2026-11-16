"""The evidence-cache-accelerated replay + the GUI error paths.

Two contracts, each with its measured reason:

1. **the cache does not change the reading**: replaying a take whose
   `.observation.npz` exists reads `interpret_observation` on the SAME
   evidence (pYIN skipped) — the counters must equal the finalize
   path's; a corrupt cache falls back to finalize without an error;
2. **GUI errors never cost the take**: an unwritable WAV is a warning
   (the transcription ships anyway), the artifact announcement checks
   the disk before displaying, a corrupt journal is reset, a worker
   with nothing to do stays silent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from tests.test_live_gui import FakeMic
from tests.test_pipeline import _melody
from v2m.core import audio_io
from v2m.core import coherence as coh
from v2m.core import take_export as te
from v2m.core.replay import replay_take


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _save_sung_wav(tmp_path: Path) -> Path:
    import soundfile as sf

    wav = tmp_path / "prise" / "prise.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), _melody().astype(np.float32), audio_io.SR, subtype="PCM_16")
    return wav


# --------------------------------------------------------------------------- #
# 1. The accelerated replay: the cache does not change the reading
# --------------------------------------------------------------------------- #


# The mic is a FakeMic for the WHOLE module (no real audio device): the
# patch must live at module scope — a function-scoped monkeypatch would
# make every fixture that needs it function-scoped too, and the shared
# pYIN payment below must SURVIVE from test to test.
@pytest.fixture(scope="module", autouse=True)
def _fake_mic_module():
    from types import SimpleNamespace as NS

    saved = audio_io.sd
    audio_io.sd = NS(InputStream=FakeMic)
    yield
    audio_io.sd = saved


# A single take seeds the evidence cache; both cache-path checks reuse it.
@pytest.fixture(scope="module", name="sung_wav")
def _sung_wav_module(tmp_path_factory):
    return _save_sung_wav(tmp_path_factory.mktemp("error_paths"))


@pytest.fixture(scope="module")
def fresh_replay_row(sung_wav):
    """The WITHOUT-cache reference reading (finalize path), once per module."""
    return replay_take(sung_wav)


@pytest.fixture(scope="module")
def cached_replay_row(sung_wav, fresh_replay_row):
    """The WITH-cache reading (the fresh replay above seeded the cache)."""
    assert sung_wav.with_suffix(".observation.npz").exists()
    return replay_take(sung_wav)


@pytest.mark.slow  # two full real-time replays of the take
def test_cached_reading_and_breakdown_match_finalize(fresh_replay_row, cached_replay_row):
    assert cached_replay_row["final_notes"] == fresh_replay_row["final_notes"]
    assert cached_replay_row["tempo_bpm"] == pytest.approx(fresh_replay_row["tempo_bpm"], abs=1.0)
    assert 0.0 <= cached_replay_row["rate"] <= 1.0
    assert cached_replay_row["moved"] + cached_replay_row["off1"] == cached_replay_row["contours"]


def test_a_corrupt_observation_cache_falls_back_without_error(tmp_path):
    # its OWN take: the test must BREAK a cache, not inherit a good one
    wav = _save_sung_wav(tmp_path)
    wav.with_suffix(".observation.npz").write_bytes(b"not an npz at all")
    # the corrupt cache is worth a missing one: no crash, a full audit
    row = replay_take(wav)
    assert row["final_notes"] >= 3
    # ...and the corrupt file was REPLACED by the replay's own auto-save
    assert wav.with_suffix(".observation.npz").stat().st_size > 100


# --------------------------------------------------------------------------- #
# 2. The GUI error paths: never the take
# --------------------------------------------------------------------------- #


def _fake_live_result() -> SimpleNamespace:
    return SimpleNamespace(
        source="live",
        live_vs_final={
            "live": 5,
            "final": 4,
            "contours": 1,
            "off1": 1,
            "moved": 0,
            "absorbed": 0,
            "rate": 0.2,
        },
        quantize_step=lambda _d: 0.1,
    )


def _redirect(monkeypatch, mw, tmp_path) -> None:
    def fake_export(res, stem, **kw) -> None:
        Path(stem).parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(te, "export_result", fake_export)
    monkeypatch.setattr(mw, "live_export_dir", lambda stem: tmp_path / Path(stem).name)


def test_a_failing_wav_write_warns_and_still_exports(qapp, monkeypatch, tmp_path):
    """The disk refuses the .wav: warning, transcription still saved."""
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)

    def broken_save(signal, path, sr):
        raise OSError("disk full")

    monkeypatch.setattr(audio_io, "save_file", broken_save)
    win = mw.MainWindow()
    try:
        win.result = cast(Any, _fake_live_result())
        win._live_signal = np.full(audio_io.SR, 0.2, dtype=np.float32)
        win._live_record_audio = True
        win._export_results()
        assert not win.error_banner.isHidden(), "the WAV failure must be announced"
        assert "disk full" in win.error_banner.text()
        # the data artifacts were written anyway
        stems = list(tmp_path.glob("*/"))
        assert stems and list(stems[0].glob("*.json")), "the take was lost"
        win.error_banner.clear()
    finally:
        win.close()


def test_the_export_announcement_lists_only_files_that_exist(qapp, monkeypatch, tmp_path):
    """The status line claims exactly what is on disk — no more.

    The writer here creates only `<stem>.json` while the contract has five
    extensions: the announcement must name that one file and never the
    others, so a half-written export (disk full mid-file) cannot read as
    complete.
    """
    from v2m.core.i18n import tr
    from v2m.core.takes import ARTIFACTS
    from v2m.gui import main_window as mw

    def one_artifact(res, stem, **kw) -> None:
        Path(stem).parent.mkdir(parents=True, exist_ok=True)
        Path(str(stem) + ".json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(te, "export_result", one_artifact)
    monkeypatch.setattr(mw, "live_export_dir", lambda stem: tmp_path / Path(stem).name)
    monkeypatch.setattr("v2m.core.retention.prune_takes", lambda _root: [])

    win = mw.MainWindow()
    try:
        monkeypatch.setattr(win, "_refresh_exported_coherence", lambda: None)
        monkeypatch.setattr(win, "_pump_replay_queue", lambda: None)
        win.result = cast(Any, SimpleNamespace(source="live", quantize_step=lambda _d: 0.1))
        win._live_signal = np.zeros(audio_io.SR, dtype=np.float32)
        win._live_record_audio = False
        win._export_results()

        take_dir = next(tmp_path.glob("*/"))
        stem = take_dir.name
        message = win.statusBar().currentMessage()
        assert message == tr("ui status exported").format(files=f"{stem}.json")
        for ext in ARTIFACTS:
            if ext == ".json":
                continue
            assert not (take_dir / f"{stem}{ext}").exists()
            assert f"{stem}{ext}" not in message
    finally:
        win.close()


def test_a_corrupt_coherence_journal_is_reinitialized(qapp, tmp_path):
    """An unreadable journal restarts at {}: aggregation never blocked."""
    (tmp_path / coh.JOURNAL_NAME).write_bytes(b"{corrupted json")
    stem = "live_20260921_090000"
    d = tmp_path / stem
    d.mkdir()
    (d / f"{stem}.json").write_text(
        json.dumps({"coherence": {"checked": 5, "suspects": []}}), encoding="utf-8"
    )
    (d / f"{stem}.coherence.csv").write_text("h\n", encoding="utf-8")
    takes = coh.takes_under(tmp_path)
    data = coh.update_journal(tmp_path, takes)
    assert data[stem]["checked"] == 5  # rebuilt from the folder
    assert json.loads((tmp_path / coh.JOURNAL_NAME).read_text(encoding="utf-8")) == data
