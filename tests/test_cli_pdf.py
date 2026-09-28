"""CLI: the PDF engraving is a separate step, never fatal to the rest.

Regression: `except PdfExportError` left the `try` without `stem` and
`rep` bound (export_result raised before returning); the summary that
followed hit an UnboundLocalError, even though .mid/.json/.csv/
.musicxml had just been written.
"""

from __future__ import annotations

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
