"""CLI: the PDF engraving is a separate step, never fatal to the rest.

A failed engraving must not unbind the export's `stem` and `rep` (the
.mid/.json/.csv/.musicxml artifacts are already written at that point)
and must not reach the summary as an unbound name.

Tool DISCOVERY answers beyond the PATH on every supported platform:
Windows installer directories, macOS .app bundles and POSIX
package-manager bin directories are probed when `shutil.which` is blind.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import soundfile as sf

from conftest import SR, silence, sine_note
from v2m import cli
from v2m.core.pdf_export import PdfExportError, PdfToolMissing


def _wav(tmp_path) -> str:
    parts: list[np.ndarray] = []
    for hz in (220.0, 261.63, 329.63):
        parts.append(sine_note(hz, 0.45))
        parts.append(silence(0.1))
    path = tmp_path / "melody.wav"
    sf.write(str(path), np.concatenate(parts), SR)
    return str(path)


def test_pdf_failure_still_writes_artifacts_and_prints_summary(tmp_path, monkeypatch, capsys):
    def _boom(stem):
        raise PdfExportError("MuseScore failed")

    monkeypatch.setattr(cli, "engrave_pdf", _boom)
    stem = tmp_path / "out"

    code = cli.main(["analyze", _wav(tmp_path), "-o", str(stem), "--pdf"])

    out, err = capsys.readouterr()
    assert code == 2, "a PDF failure must exit 2, not crash"
    assert "BPM" in out, "the analysis summary must be printed despite the failure"
    assert "PDF" in err and "MuseScore failed" in err
    for ext in (".mid", ".musicxml", ".json", ".csv"):
        assert (tmp_path / f"out{ext}").exists(), f"out{ext} missing"
    assert not (tmp_path / "out.pdf").exists()
    assert ".pdf" not in out.split("Export")[-1]


def test_missing_tool_skips_the_pdf_without_failing_the_run(tmp_path, monkeypatch, capsys):
    """Engraving impossible for lack of a tool: clear warning, exit 0, and
    the .pdf is not announced in the list of written artifacts."""

    def _no_tool(stem):
        raise PdfToolMissing("PDF ignored: installez MuseScore ou LilyPond")

    monkeypatch.setattr(cli, "engrave_pdf", _no_tool)

    code = cli.main(["analyze", _wav(tmp_path), "-o", str(tmp_path / "out"), "--pdf"])

    out, err = capsys.readouterr()
    assert code == 0
    assert "BPM" in out and "MuseScore" in err
    assert ".pdf" not in out.split("Export")[-1]
    assert (tmp_path / "out.mid").exists()


def test_pdf_success_is_listed_with_the_other_artifacts(tmp_path, monkeypatch, capsys):
    written: list[str] = []

    def _fake(stem):
        written.append(str(stem))
        (tmp_path / "out.pdf").write_bytes(b"%PDF-1.4")
        return str(tmp_path / "out.pdf")

    monkeypatch.setattr(cli, "engrave_pdf", _fake)

    code = cli.main(["analyze", _wav(tmp_path), "-o", str(tmp_path / "out"), "--pdf"])

    out, _ = capsys.readouterr()
    assert code == 0
    assert written, "the engraving was not requested"
    assert ".pdf" in out


# --------------------------------------------------------------------------- #
# Discovery beyond the PATH: one probe per platform
# --------------------------------------------------------------------------- #


def _no_path_tools(monkeypatch) -> None:
    """Deterministic probe: nothing on the PATH, no real install answering."""
    from v2m.core import pdf_export

    monkeypatch.setattr(pdf_export, "shutil", SimpleNamespace(which=lambda cmd: None))
    monkeypatch.setattr(pdf_export, "_installed_bases", lambda: [])
    monkeypatch.setattr(pdf_export, "_mac_app_roots", lambda: [])
    monkeypatch.setattr(pdf_export, "_POSIX_BIN_DIRS", ())


def test_notation_tool_finds_a_windows_install_dir(tmp_path, monkeypatch):
    """A Windows installer directory (absent from the PATH) is probed."""
    from v2m.core import pdf_export

    _no_path_tools(monkeypatch)
    base = tmp_path / "Program Files"
    exe = base / "MuseScore 4" / "bin" / "MuseScore4.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    monkeypatch.setattr(pdf_export, "_installed_bases", lambda: [base])
    assert pdf_export.find_notation_tool() == ("MuseScore", str(exe))


def test_notation_tool_finds_a_macos_app_bundle(tmp_path, monkeypatch):
    """A .app bundle is never on the PATH: the executable inside it answers."""
    from v2m.core import pdf_export

    _no_path_tools(monkeypatch)
    exe = tmp_path / "MuseScore 4.app" / "Contents" / "MacOS" / "mscore"
    exe.parent.mkdir(parents=True)
    exe.touch()
    monkeypatch.setattr(pdf_export, "_mac_app_roots", lambda: [tmp_path])
    assert pdf_export.find_notation_tool() == ("MuseScore", str(exe))


def test_notation_tool_finds_a_posix_bin_dir(tmp_path, monkeypatch):
    """A package-manager bin dir a minimal GUI PATH does not carry."""
    from v2m.core import pdf_export

    _no_path_tools(monkeypatch)
    exe = tmp_path / "musescore4"
    exe.touch()
    monkeypatch.setattr(pdf_export, "_POSIX_BIN_DIRS", (str(tmp_path),))
    assert pdf_export.find_notation_tool() == ("MuseScore", str(exe))
