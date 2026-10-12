"""The export contract through the two ADAPTERS (thin: rendering only).

The module's sequence and receipt are pinned Qt-free in
`test_take_export.py`; this file pins that the GUI window and the CLI RENDER
that receipt without re-deriving it. Both drive the REAL writers and are
checked against the SAME disk facts (`export_contract_checks`): every
announced name is on disk and the announcement names exactly the stem's
files, a missing/failed PDF is never announced, and the write order is the
take's WAV (live), the artifacts, then the PDF.

The window's post-export side effects (the coherence refresh, the replay
pump, the retention trim) read the REAL user folders and are covered
elsewhere — they are silenced here so the export under test stops at the
artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from tests import export_contract_checks as checks
from v2m.core import audio_io
from v2m.core import take_export as te
from v2m.core.grid import BeatGrid
from v2m.core.i18n import tr
from v2m.core.keys import KeyCandidate, KeyResult
from v2m.core.notes import Note
from v2m.core.pdf_export import PdfExportError, PdfToolMissing
from v2m.core.pipeline import AnalysisResult
from v2m.core.tempo import TempoResult


@pytest.fixture(autouse=True)
def _sandbox_export_directory(monkeypatch, tmp_path):
    """Keep GUI tests from reading or writing the user's export preference."""
    from v2m.gui import main_window as mw

    monkeypatch.setattr(mw, "EXPORT_DIRECTORY_FILE", tmp_path / "export-directory")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _result(source: str = "live", *, with_block: bool = False) -> AnalysisResult:
    """A minimal but REAL result `export_result` can engrave (tempo + key + grid)."""
    result = AnalysisResult(
        source=source,
        duration_s=2.0,
        notes=[Note(0.2, 0.7, 60, 0.9), Note(0.8, 1.2, 64, 0.9)],
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 2.0, 0.5)),
        key=KeyResult(KeyCandidate("C", "major", 0.9), []),
        grid=BeatGrid(120.0),
    )
    if with_block:
        result.live_vs_final = {
            "live": 2, "final": 2, "contours": 0,
            "off1": 0, "moved": 0, "absorbed": 0, "rate": 0.0,
        }
    return result


def _isolate_window(win, monkeypatch, tmp_path) -> None:
    """Redirect the take folder and silence the post-export side effects."""
    from v2m.gui import main_window as mw

    monkeypatch.setattr(mw, "live_export_dir", lambda stem: tmp_path / Path(stem).name)
    monkeypatch.setattr("v2m.core.retention.prune_takes", lambda _root: [])
    monkeypatch.setattr(win, "_refresh_exported_coherence", lambda: None)
    monkeypatch.setattr(win, "_pump_replay_queue", lambda: None)


def _announced(win) -> list[str]:
    """The file names the window's status line lists after an export."""
    message = win.statusBar().currentMessage()
    prefix = tr("ui status exported").split("{files}")[0]
    assert message.startswith(prefix), message
    return [name.strip() for name in message[len(prefix):].split(",") if name.strip()]


def _take_stem(tmp_path) -> Path:
    take_dir = next(tmp_path.glob("*/"))
    return take_dir / take_dir.name


# --------------------------------------------------------------------------- #
# The GUI window: renders the receipt, never re-derives it
# --------------------------------------------------------------------------- #


def test_the_window_announces_exactly_the_files_on_disk(qapp, monkeypatch, tmp_path):
    """Every contract artifact reaches disk — plus the asked-for WAV and the
    take's live/final sidecar — and the status line names exactly those."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        _isolate_window(win, monkeypatch, tmp_path)
        win.result = _result("live", with_block=True)
        win._live_signal = np.full(audio_io.SR, 0.2, dtype=np.float32)
        win._live_record_audio = True

        win._export_results()

        checks.assert_announcement_matches_disk(_announced(win), _take_stem(tmp_path))
        assert win.error_banner.isHidden(), win.error_banner.text()
    finally:
        win.close()


def test_the_save_as_button_writes_only_the_chosen_file(qapp, monkeypatch, tmp_path):
    """Exercise the user's dialog action through its real export writer."""
    from PyQt6.QtWidgets import QFileDialog

    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    destination = tmp_path / "picked" / "song.json"
    try:
        win.result = _result("source/chant.wav")
        monkeypatch.setattr(win, "_refresh_exported_coherence", lambda: None)
        monkeypatch.setattr(win, "_pump_replay_queue", lambda: None)
        monkeypatch.setattr(
            mw, "live_export_dir", lambda _stem: tmp_path / "unused-live-archive"
        )
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *_args, **_kwargs: (str(destination), "JSON report (*.json)")),
        )

        win.on_export_again()

        assert destination.is_file(), (
            win.error_banner.text(), win.statusBar().currentMessage()
        )
        assert list(destination.parent.iterdir()) == [destination]
        assert win.statusBar().currentMessage().endswith(str(destination))
        assert win.error_banner.isHidden(), win.error_banner.text()
    finally:
        win.close()


def test_live_save_as_keeps_its_archive_and_copies_the_selected_file(
    qapp, monkeypatch, tmp_path
):
    """The desktop's live Save As action preserves the archive and writes one copy."""
    from PyQt6.QtWidgets import QFileDialog

    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    destination = tmp_path / "picked" / "take.musicxml"
    archive_dir = tmp_path / "archive"
    try:
        win.result = _result("live", with_block=True)
        win._live_signal = np.full(audio_io.SR, 0.2, dtype=np.float32)
        win._live_record_audio = False
        monkeypatch.setattr(
            mw, "live_export_dir", lambda stem: archive_dir / Path(stem).name
        )
        monkeypatch.setattr(win, "_refresh_exported_coherence", lambda: None)
        monkeypatch.setattr(win, "_pump_replay_queue", lambda: None)
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            staticmethod(
                lambda *_args, **_kwargs: (
                    str(destination), "MusicXML (*.musicxml)"
                )
            ),
        )

        win.on_export_again()

        assert destination.is_file(), win.error_banner.text()
        take_dir = next(archive_dir.iterdir())
        stem = take_dir / take_dir.name
        for suffix in (".mid", ".musicxml", ".json", ".csv", ".coherence.csv"):
            assert Path(f"{stem}{suffix}").is_file(), suffix
        assert destination.read_bytes() == Path(f"{stem}.musicxml").read_bytes()
        assert list(destination.parent.iterdir()) == [destination]
        assert win.error_banner.isHidden(), win.error_banner.text()
    finally:
        win.close()


def test_save_as_remembers_the_last_folder_across_windows(qapp, monkeypatch, tmp_path):
    """The dialog reopens at the selected folder and cancellation preserves it."""
    from PyQt6.QtWidgets import QFileDialog

    from v2m.gui import main_window as mw

    settings_path = tmp_path / "export-directory"
    destination_dir = tmp_path / "picked"
    destination_dir.mkdir()
    destination = destination_dir / "song.json"
    seen_directories: list[str] = []

    def choose_once(parent, caption, directory, file_filter, initialFilter, options=None):
        seen_directories.append(directory)
        if len(seen_directories) == 1:
            return str(destination), "JSON report (*.json)"
        return "", ""

    monkeypatch.setattr(mw, "EXPORT_DIRECTORY_FILE", settings_path)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(choose_once)
    )

    first_window = mw.MainWindow()
    try:
        picked = first_window._choose_export_destination(_result("source/chant.wav"))
        assert picked == (destination, ".json")
        assert settings_path.read_text(encoding="utf-8") == str(destination_dir)
    finally:
        first_window.close()

    second_window = mw.MainWindow()
    try:
        assert second_window._choose_export_destination(
            _result("another/song.wav")
        ) is None
        assert Path(seen_directories[1]).parent == destination_dir
        assert settings_path.read_text(encoding="utf-8") == str(destination_dir)
    finally:
        second_window.close()


def test_the_window_does_not_export_when_save_as_is_cancelled(qapp, monkeypatch):
    """Cancel closes the dialog without writing or changing the result."""
    from PyQt6.QtWidgets import QFileDialog

    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = _result("source/chant.wav")
    try:
        win.result = result
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", staticmethod(lambda *_args, **_kwargs: ("", ""))
        )
        monkeypatch.setattr(win, "_export_results", lambda *_args: pytest.fail("exported"))

        win.on_export_again()

        assert win.result is result
    finally:
        win.close()


def test_the_window_writes_the_wav_before_the_artifacts_and_the_pdf(
    qapp, monkeypatch, tmp_path
):
    """The take's own audio is the rawest data: written FIRST, then the
    artifacts, and the PDF (an external tool) comes last."""
    from v2m.gui import main_window as mw

    order = checks.instrument_write_order(monkeypatch, te)

    win = mw.MainWindow()
    try:
        _isolate_window(win, monkeypatch, tmp_path)
        win.result = _result("live")
        win._live_signal = np.full(audio_io.SR, 0.2, dtype=np.float32)
        win._live_record_audio = True
        win.page_results.chk_pdf.setChecked(True)

        win._export_results()

        assert order == ["wav", "artifacts", "pdf"]
        checks.assert_announcement_matches_disk(_announced(win), _take_stem(tmp_path))
    finally:
        win.close()


def test_the_window_never_announces_a_failed_pdf(qapp, monkeypatch, tmp_path):
    """A failed engraving warns, costs no artifact, and never appears in the
    announcement — the disk is the only truth."""
    from v2m.gui import main_window as mw

    def broken_pdf(stem) -> None:
        raise PdfExportError("engraving tool fell over")

    monkeypatch.setattr(te, "engrave_pdf", broken_pdf)

    win = mw.MainWindow()
    try:
        _isolate_window(win, monkeypatch, tmp_path)
        win.result = _result("live")
        win._live_signal = np.ones(audio_io.SR, dtype=np.float32)
        win._live_record_audio = False
        win.page_results.chk_pdf.setChecked(True)

        win._export_results()

        stem = _take_stem(tmp_path)
        announced = _announced(win)
        checks.assert_no_pdf(announced, stem)
        checks.assert_announcement_matches_disk(announced, stem)
        assert Path(f"{stem}.json").exists()  # the failure cost nothing
        assert tr("err ui pdf").split("{e}")[0] in win.error_banner.text()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# The CLI adapter: the same receipt facts, one shared assertion set
# --------------------------------------------------------------------------- #


def _wav(tmp_path) -> str:
    import soundfile as sf

    from tests.test_pipeline import _melody

    wav = tmp_path / "chant.wav"
    sf.write(str(wav), _melody().astype(np.float32), audio_io.SR)
    return str(wav)


def _cli_announced(out: str) -> list[str]:
    """The file paths the CLI's export summary line lists."""
    label = tr("key export label")
    line = next(line for line in reversed(out.splitlines()) if line.startswith(label))
    body = line.split(":", 1)[1]
    return [name.strip() for name in body.split(",") if name.strip()]


def test_the_cli_announces_exactly_the_files_on_disk(tmp_path, capsys):
    """A file export writes its artifacts beside the output prefix and the
    summary names exactly those files."""
    from v2m import cli

    stem = tmp_path / "out"
    code = cli.main(["analyze", _wav(tmp_path), "-o", str(stem)])

    out, _ = capsys.readouterr()
    assert code == 0
    checks.assert_announcement_matches_disk(_cli_announced(out), stem)


def test_the_cli_writes_no_wav_and_engraves_the_pdf_last(tmp_path, capsys, monkeypatch):
    """A file analysis owns its audio at the input path: no WAV is written,
    and the external-tool PDF is engraved after the artifacts."""
    from v2m import cli

    order = checks.instrument_write_order(monkeypatch, te)
    stem = tmp_path / "out"
    code = cli.main(["analyze", _wav(tmp_path), "-o", str(stem), "--pdf"])

    out, _ = capsys.readouterr()
    assert code == 0
    assert order == ["artifacts", "pdf"]
    checks.assert_announcement_matches_disk(_cli_announced(out), stem)


def test_the_cli_never_announces_a_missing_pdf(tmp_path, capsys, monkeypatch):
    """No engraving tool: a clear warning, exit 0, and the .pdf is not part of
    the written artifacts."""
    from v2m import cli

    def no_tool(stem) -> None:
        raise PdfToolMissing("PDF ignoré : installez MuseScore ou LilyPond")

    monkeypatch.setattr(te, "engrave_pdf", no_tool)

    stem = tmp_path / "out"
    code = cli.main(["analyze", _wav(tmp_path), "-o", str(stem), "--pdf"])

    out, err = capsys.readouterr()
    assert code == 0, "a missing tool is skipped, not failed"
    assert "MuseScore" in err
    checks.assert_no_pdf(_cli_announced(out), stem)
    checks.assert_announcement_matches_disk(_cli_announced(out), stem)
