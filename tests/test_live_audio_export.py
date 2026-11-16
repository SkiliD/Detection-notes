"""The original take audio, written on request and preserved after capture.

The "record audio" box is read at the take's START (a mid-take toggle
does not change what this take writes) and only acts live: a file
analysis already reads its audio at the input, rewriting it would make
no sense. The first successful .wav is immutable across exports.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from v2m.core import audio_io
from v2m.core import take_export as te


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _fake_live_result(source: str = "live") -> SimpleNamespace:
    """What `_export_results` reads from a result (export_result is patched)."""
    return SimpleNamespace(source=source, quantize_step=lambda _division: 0.1)


def test_export_writes_the_take_audio_only_when_asked(qapp, monkeypatch, tmp_path):
    """Box ticked -> a .wav in the take's folder; else nothing."""
    import soundfile as sf

    from v2m.gui import main_window as mw

    def fake_export(res, stem, **_kwargs):
        stem = Path(stem)
        if stem.parent.is_relative_to(tmp_path):
            from v2m.core.report import ARTIFACT_EXTENSIONS

            stem.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ARTIFACT_EXTENSIONS:
                Path(f"{stem}{suffix}").write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(te, "export_result", fake_export)
    # the take folder is redirected: the test must never write into the repo
    monkeypatch.setattr(mw, "live_export_dir", lambda stem: tmp_path / Path(stem).name)

    win = mw.MainWindow()
    try:
        win.result = cast(Any, _fake_live_result())
        win._live_signal = (np.ones(audio_io.SR, dtype=np.float32) * 0.1)

        win._live_record_audio = False
        win._export_results()
        assert not list(tmp_path.rglob("*.wav")), "audio written without the box ticked"

        win._live_record_audio = True
        win._automatic_live_result = cast(Any, _fake_live_result())
        win._export_results()
        wavs = list(tmp_path.rglob("*.wav"))
        assert len(wavs) == 1, f"one .wav expected, saw {wavs}"
        # the file is readable back, at the analysis rate and the right length
        y, sr = sf.read(str(wavs[0]))
        assert sr == audio_io.SR
        assert abs(len(y) - audio_io.SR) <= 4
        original_wav = wavs[0].read_bytes()
        assert win.error_banner.isHidden(), win.error_banner.text()

        # The automatic read is copied into its own stable archive on export.
        from v2m.core.takes import automatic_reading_dir

        automatic_json = automatic_reading_dir(wavs[0].parent) / f"{wavs[0].stem}.json"
        assert automatic_json.is_file()
        assert automatic_json.read_text(encoding="utf-8") == "{}"
        automatic_bytes = automatic_json.read_bytes()

        # Re-exporting a corrected reading must preserve the original capture.
        win._live_signal = np.full(audio_io.SR, 0.8, dtype=np.float32)
        corrected_result = _fake_live_result()
        corrected_result.manual_pitch_corrections = True
        win.result = cast(Any, corrected_result)
        win._export_results()
        assert wavs[0].read_bytes() == original_wav
        from v2m.core.takes import corrected_reading_dir

        version_one = corrected_reading_dir(wavs[0].parent, 1)
        assert (version_one / f"{wavs[0].stem}.json").is_file()
        assert automatic_json.read_bytes() == automatic_bytes

        # ...and a FILE analysis never writes an audio copy, box or no box:
        # its audio already exists at the input path
        win.result = cast(Any, _fake_live_result(source="chant.wav"))
        win._export_results()
        assert len(list(tmp_path.rglob("*.wav"))) == 1
    finally:
        win.close()


def test_the_audio_box_is_read_when_the_take_starts():
    """The choice belongs to the take: frozen at start, not at export."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        # ON by default: the audio is what keeps a failed take diagnosable
        assert win.page_live.chk_record_audio.isChecked() is True
        win.page_live.chk_record_audio.setChecked(True)
        # what `on_start_live` does with the box (no capture started here)
        win._live_record_audio = win.page_live.chk_record_audio.isChecked()
        win.page_live.chk_record_audio.setChecked(False)  # toggled mid-take
        assert win._live_record_audio is True, "the mid-take toggle changed the choice"
        assert win.page_live.chk_record_audio.text()  # a label in the current language
    finally:
        win.close()


def test_a_new_take_never_keeps_the_previous_take_signal(qapp):
    """Every WAV export must describe ITS take: `on_start_live` resets
    `_live_signal` to None and `_on_live_analysis_done` picks the signal
    bytes up ALWAYS (the stale-signal family)."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        old = np.full(audio_io.SR, 0.5, dtype=np.float32)
        win._live_signal = old
        # what on_start_live does (no capture here): the NEW take owns all
        # the live buffers
        win._live_signal = None
        assert win._live_signal is None, "the old take survived the start"

        # ...and the take's end: the EVENT's bytes are the source of
        # truth, a stale signal cannot survive a finished analysis
        win._live_signal = old
        new = np.full(2 * audio_io.SR, 0.1, dtype=np.float32)
        win._on_live_analysis_done(
            cast(
                Any,
                SimpleNamespace(
                    source="live", tempo=SimpleNamespace(bpm=120.0),
                    key=SimpleNamespace(best=SimpleNamespace(label="C major")),
                    notes=[],
                ),
            ),
            new.tobytes(),
        )
        assert win._live_signal is not old
        assert len(win._live_signal) == len(new), (
            "the finished analysis kept an earlier take's signal"
        )
    finally:
        win.close()


def test_save_file_round_trips_a_signal(tmp_path):
    """`save_file` writes a re-readable WAV, at the right length and rate."""
    import soundfile as sf

    t = np.arange(audio_io.SR) / audio_io.SR
    y = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    path = audio_io.save_file(y, str(tmp_path / "nested" / "prise.wav"), audio_io.SR)
    assert Path(path).exists()  # the folder was created too
    back, sr = sf.read(path)
    assert sr == audio_io.SR
    assert len(back) == len(y)
    # 16-bit PCM: the shape and amplitude survive, the samples are quantized
    assert float(np.max(np.abs(back[: audio_io.SR // 10]))) == pytest.approx(0.4, abs=0.01)
