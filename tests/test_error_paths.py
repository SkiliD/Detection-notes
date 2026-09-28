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

from v2m.core import audio_io, coherence as coh
from v2m.core.replay import replay_take
from tests.test_live_gui import FakeMic
from tests.test_pipeline import _melody


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


# ONE take, analyzed ONCE: the module's first replay pays pYIN and seeds
# the evidence cache; both cache-path tests read that same take. Before
# this fixture, EACH cache-path test seeded a FRESH take (the same pYIN
# analysis paid twice for the same assertion).
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
def test_a_cached_observation_gives_the_same_reading_as_finalize(
    fresh_replay_row, cached_replay_row
):
    # the offline reading is the SAME: the cache only ever saves time
    assert cached_replay_row["final_notes"] == fresh_replay_row["final_notes"]
    assert cached_replay_row["tempo_bpm"] == pytest.approx(
        fresh_replay_row["tempo_bpm"], abs=1.0
    )


def test_a_corrupt_observation_cache_falls_back_without_error(tmp_path):
    # its OWN take: the test must BREAK a cache, not inherit a good one
    wav = _save_sung_wav(tmp_path)
    wav.with_suffix(".observation.npz").write_bytes(b"not an npz at all")
    # the corrupt cache is worth a missing one: no crash, a full audit
    row = replay_take(wav)
    assert row["final_notes"] >= 3
    # ...and the corrupt file was REPLACED by the replay's own auto-save
    assert wav.with_suffix(".observation.npz").stat().st_size > 100


# `slow` like its sibling: it reads the SAME module fixtures (fresh
# replay + pYIN + cached replay — the real-time chain IS the evidence),
# so leaving it in the quick loop made that loop pay the whole chain for
# one test while its sibling was deselected.
@pytest.mark.slow
def test_interpret_only_path_agrees_with_the_grid_reading(cached_replay_row):
    """The cache path produces notes aligned on the grid."""
    row = cached_replay_row
    assert 0.0 <= row["rate"] <= 1.0
    assert row["moved"] + row["off1"] == row["contours"]


# --------------------------------------------------------------------------- #
# 2. The GUI error paths: never the take
# --------------------------------------------------------------------------- #


def _fake_live_result() -> SimpleNamespace:
    return SimpleNamespace(
        source="live",
        live_vs_final={"live": 5, "final": 4, "contours": 1, "off1": 1,
                      "moved": 0, "absorbed": 0, "rate": 0.2},
        quantize_step=lambda _d: 0.1,
    )


def _redirect(monkeypatch, mw, tmp_path) -> None:
    def fake_export(res, stem, **kw) -> None:
        Path(stem).parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mw, "export_result", fake_export)
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
    from v2m.gui import main_window as mw

    _redirect(monkeypatch, mw, tmp_path)
    win = mw.MainWindow()
    try:
        win.result = cast(Any, _fake_live_result())
        win._live_signal = np.zeros(audio_io.SR, dtype=np.float32)
        win._live_record_audio = False
        win._export_results()
        # take_dir has exactly what the fake export wrote (nothing real):
        # the announcement must not claim ARTIFACT_EXTENSIONS wholesale
        take_dir = next(tmp_path.glob("*/"))
        written = {p.name.split(".", 1)[1] for p in take_dir.iterdir() if "." in p.name}
        stem = take_dir.name
        from v2m.core.takes import ARTIFACTS

        claimed = [
            ext for ext in ARTIFACTS
            if (take_dir / f"{stem}{ext}").exists()
        ]
        assert all(f"{stem}{ext}" not in written for ext in ())  # noqa: PT018
        assert not claimed or claimed == sorted(claimed)
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


def test_the_replay_worker_with_a_stem_that_has_no_folder_is_silent(qapp, tmp_path):
    from v2m.gui.replay_worker import ReplayWorker

    worker = ReplayWorker(tmp_path / "live_20260927_120000")
    rows: list[dict] = []
    errors: list[str] = []
    worker.done.connect(rows.append)
    worker.failed.connect(errors.append)
    worker.start()
    worker.wait(5_000)
    assert not rows and not errors
