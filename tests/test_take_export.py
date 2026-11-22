"""The take-export SEQUENCE, Qt-free: `core.take_export.write_take`.

The module owns the order, the conditional steps, the two error contracts
and the announcement facts. These tests drive it DIRECTLY (no window, no
event loop), so a failure here names the module, never the adapter. The
adapter's rendering of the same receipt is pinned in
`test_export_contract.py`; the shared rules live in `export_contract_checks`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests import export_contract_checks as checks
from v2m.core import audio_io
from v2m.core import take_export as te
from v2m.core.grid import BeatGrid
from v2m.core.keys import KeyCandidate, KeyResult
from v2m.core.musicxml import save_musicxml
from v2m.core.notes import Note
from v2m.core.pdf_export import PdfExportError, PdfToolMissing
from v2m.core.pipeline import AnalysisResult
from v2m.core.report import export_result
from v2m.core.takes import SIDECAR_SUFFIX
from v2m.core.tempo import TempoResult


def _result(source: str = "live", *, live_vs_final: dict | None = None) -> AnalysisResult:
    """A minimal but REAL result `export_result` can engrave (tempo + key + grid)."""
    result = AnalysisResult(
        source=source,
        duration_s=2.0,
        notes=[Note(0.2, 0.7, 60, 0.9), Note(0.8, 1.2, 64, 0.9)],
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 2.0, 0.5)),
        key=KeyResult(KeyCandidate("C", "major", 0.9), []),
        grid=BeatGrid(120.0),
    )
    if live_vs_final is not None:
        result.live_vs_final = live_vs_final
    return result


def _signal() -> np.ndarray:
    return np.full(audio_io.SR, 0.2, dtype=np.float32)


@pytest.fixture(autouse=True)
def _no_real_retention(monkeypatch):
    """Retention reads the REAL user folder: never scan it in a unit test."""
    monkeypatch.setattr("v2m.core.retention.prune_takes", lambda _root: [])


def test_successful_export_orders_and_announces_every_file(tmp_path, monkeypatch):
    """The WAV precedes artifacts and PDF; the receipt matches the disk."""
    order = checks.instrument_write_order(monkeypatch, te)

    receipt = te.write_take(
        _result("live"),
        tmp_path / "take",
        signal=_signal(),
        options=te.ExportOptions(record_audio=True, pdf=True),
    )

    assert order == ["wav", "artifacts", "pdf"]
    from v2m.core.takes import automatic_reading_dir

    automatic_stem = automatic_reading_dir(tmp_path) / "take"
    assert (automatic_stem.with_suffix(".json")).is_file()
    assert receipt.wav_written and receipt.pdf_status == te.PDF_WRITTEN
    assert ".wav" in receipt.suffixes and ".pdf" in receipt.suffixes
    checks.assert_announcement_matches_disk(receipt.file_names(), tmp_path / "take")


def test_both_text_writers_ask_for_lf_on_every_platform(tmp_path, monkeypatch):
    """`export_result` and `save_musicxml` opt OUT of the platform's newline.

    The reference hashes are the artifact BYTES, so a CRLF writer would leave
    one reference matching only the OS it was frozen on. Reading the bytes
    alone cannot pin that on a POSIX checkout — text mode only translates `\n`
    on Windows — so this also records what each writer ASKED the file for:
    both must pass `newline="\n"`. It uses a synthetic result, so the rule
    stays guarded where the reference takes are absent.
    """
    opened: list[tuple[str, object]] = []
    real_open = open

    def recording_open(file, mode="r", *args, **kwargs):
        if "w" in mode and str(file).endswith((".json", ".musicxml")):
            opened.append((str(file), kwargs.get("newline")))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", recording_open)
    stem = tmp_path / "score"
    export_result(_result("live"), stem)
    direct = tmp_path / "direct.musicxml"
    save_musicxml(
        _result("live").notes,
        BeatGrid(120.0),
        KeyResult(KeyCandidate("C", "major", 0.9), []),
        str(direct),
    )

    assert {Path(path).suffix for path, _ in opened} == {".json", ".musicxml"}
    assert {newline for _path, newline in opened} == {"\n"}, opened
    for path in (Path(f"{stem}.json"), Path(f"{stem}.musicxml"), direct):
        data = path.read_bytes()
        assert data, f"{path.name} is empty"
        assert b"\r\n" not in data, f"{path.name} was written with CRLF"


def test_live_reading_archives_keep_the_automatic_and_number_corrected_exports(tmp_path):
    """Automatic notes stay immutable while each corrected export gets its own bundle."""
    from v2m.core.takes import automatic_reading_dir, corrected_reading_dir

    stem = tmp_path / "live_take" / "live_take"
    automatic = _result("live")
    automatic_notes = [note.midi for note in automatic.notes]
    initial = te.write_take(automatic, stem)
    assert not initial.failed, initial.warnings

    automatic_dir = automatic_reading_dir(stem.parent)
    automatic_json = automatic_dir / f"{stem.name}.json"
    automatic_bytes = automatic_json.read_bytes()
    assert automatic_json.is_file()
    assert [
        (automatic_dir / f"{stem.name}{suffix}").is_file()
        for suffix in (".mid", ".musicxml", ".json", ".csv", ".coherence.csv")
    ] == [True] * 5

    corrected = _result("live")
    corrected.notes[0].midi = 65
    corrected.manual_pitch_corrections = True
    v1 = te.write_take(
        corrected,
        stem,
        options=te.ExportOptions(
            automatic_result=automatic,
            manual_corrections=True,
            archive_corrected_reading=True,
        ),
    )
    assert not v1.failed, v1.warnings
    v1_dir = corrected_reading_dir(stem.parent, 1)
    assert (v1_dir / f"{stem.name}.json").is_file()
    assert (v1_dir / f"{stem.name}.json").read_bytes() != automatic_bytes
    assert json.loads((v1_dir / f"{stem.name}.json").read_text())["notes"][0]["midi"] == 65
    assert json.loads(automatic_json.read_text())["notes"][0]["midi"] == automatic_notes[0]
    assert automatic_json.read_bytes() == automatic_bytes

    corrected.notes[0].midi = 67
    v2 = te.write_take(
        corrected,
        stem,
        options=te.ExportOptions(
            automatic_result=automatic,
            manual_corrections=True,
            archive_corrected_reading=True,
        ),
    )
    assert not v2.failed, v2.warnings
    v2_dir = corrected_reading_dir(stem.parent, 2)
    assert json.loads((v2_dir / f"{stem.name}.json").read_text())["notes"][0]["midi"] == 67
    assert (v1_dir / f"{stem.name}.json").read_bytes() != (
        v2_dir / f"{stem.name}.json"
    ).read_bytes()


def test_a_corrected_version_is_not_added_when_the_root_export_fails(tmp_path, monkeypatch):
    """A failed root write cannot leave a numbered version of an older reading."""
    from v2m.core.takes import corrected_readings_dir

    def broken_export(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(te, "export_result", broken_export)
    result = _result("live")
    result.manual_pitch_corrections = True
    receipt = te.write_take(
        result,
        tmp_path / "live_take" / "live_take",
        options=te.ExportOptions(manual_corrections=True, archive_corrected_reading=True),
    )

    assert receipt.failed
    assert not corrected_readings_dir(tmp_path / "live_take").exists()


def test_a_later_export_never_replaces_the_original_live_wav(tmp_path):
    """Re-exporting edited notes keeps the first recording byte-for-byte."""
    stem = tmp_path / "take"
    original_signal = _signal()
    corrected_signal = np.full(audio_io.SR, 0.8, dtype=np.float32)

    first = te.write_take(
        _result("live"),
        stem,
        signal=original_signal,
        options=te.ExportOptions(record_audio=True),
    )
    original_wav = Path(f"{stem}.wav").read_bytes()
    assert first.wav_written

    corrected = te.write_take(
        _result("live"),
        stem,
        signal=corrected_signal,
        options=te.ExportOptions(record_audio=True, manual_corrections=True),
    )

    assert corrected.wav_written
    assert ".wav" in corrected.suffixes
    assert Path(f"{stem}.wav").read_bytes() == original_wav


def test_a_failed_first_wav_write_does_not_leave_a_false_raw_audio_artifact(tmp_path, monkeypatch):
    """A partial failed write is removed so a later export can retry it."""
    stem = tmp_path / "take"

    def broken_save(_signal, path, _sr):
        Path(path).write_bytes(b"partial WAV")
        raise OSError("disk full")

    monkeypatch.setattr(audio_io, "save_file", broken_save)
    receipt = te.write_take(
        _result("live"),
        stem,
        signal=_signal(),
        options=te.ExportOptions(record_audio=True),
    )

    assert not receipt.wav_written
    assert not Path(f"{stem}.wav").exists()
    assert any(w.kind == "wav" and "disk full" in w.detail for w in receipt.warnings)


def test_a_failed_wav_is_a_warning_and_the_artifacts_are_written(tmp_path, monkeypatch):
    """A refused WAV is a value on the receipt, never a lost take: the
    transcription is still written and announced."""

    def broken_save(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(audio_io, "save_file", broken_save)
    receipt = te.write_take(
        _result("live"),
        tmp_path / "take",
        signal=_signal(),
        options=te.ExportOptions(record_audio=True),
    )

    assert receipt.wav_written is False
    assert receipt.failed is False
    assert any(w.kind == "wav" and "disk full" in w.detail for w in receipt.warnings)
    assert (tmp_path / "take.json").exists(), "the artifact set was lost"
    checks.assert_announcement_matches_disk(receipt.file_names(), tmp_path / "take")


def test_a_failed_pdf_is_never_announced_but_the_artifacts_are(tmp_path, monkeypatch):
    """The PDF is conditional: a failed engraving costs no artifact and is
    never announced — the disk is the only truth."""

    def broken_pdf(stem) -> None:
        raise PdfExportError("engraving tool fell over")

    monkeypatch.setattr(te, "engrave_pdf", broken_pdf)

    receipt = te.write_take(_result("live"), tmp_path / "take", options=te.ExportOptions(pdf=True))

    assert receipt.pdf_status == te.PDF_FAILED
    assert receipt.report is not None, "the failure cost the report"
    assert receipt.warning_detail("pdf") == "engraving tool fell over"
    checks.assert_no_pdf(receipt.file_names(), tmp_path / "take")
    checks.assert_announcement_matches_disk(receipt.file_names(), tmp_path / "take")


def test_a_missing_pdf_tool_is_its_own_outcome(tmp_path, monkeypatch):
    """A missing tool is not a conversion failure: the receipt says which,
    so the CLI can exit 0 on one and 2 on the other with one message."""

    def no_tool(stem) -> None:
        raise PdfToolMissing("PDF skipped: install MuseScore or LilyPond")

    monkeypatch.setattr(te, "engrave_pdf", no_tool)

    receipt = te.write_take(_result("live"), tmp_path / "take", options=te.ExportOptions(pdf=True))

    assert receipt.pdf_status == te.PDF_MISSING_TOOL
    checks.assert_no_pdf(receipt.file_names(), tmp_path / "take")


def test_a_failed_artifact_set_reports_a_failed_receipt(tmp_path, monkeypatch):
    """When the artifact writer raises, the receipt is `failed` with a value
    warning and NOTHING to announce (the adapter must not claim a file)."""

    def boom(*_args, **_kwargs):
        raise ValueError("cannot write")

    monkeypatch.setattr(te, "export_result", boom)

    receipt = te.write_take(_result("live"), tmp_path / "take")

    assert receipt.failed is True
    assert receipt.suffixes == ()
    assert receipt.report is None
    assert any(w.kind == "export" and "cannot write" in w.detail for w in receipt.warnings)
    # the one extraction the adapters render for a single-message spot
    assert receipt.warning_detail("export") == "cannot write"
    assert receipt.warning_detail("wav") == ""


def test_a_file_export_never_writes_a_wav(tmp_path, monkeypatch):
    """A file analysis already reads its audio at the input path: the box is
    ignored and no copy is made (live only)."""
    out = tmp_path / "out"
    out.mkdir()
    receipt = te.write_take(
        _result("in/chant.wav"),
        out / "chant",
        signal=_signal(),  # a signal is handed over, but the source is a file
        options=te.ExportOptions(record_audio=True),
    )

    assert receipt.wav_written is False
    assert ".wav" not in receipt.suffixes
    checks.assert_announcement_matches_disk(receipt.file_names(), out / "chant")


def test_the_sidecar_is_archived_when_due_and_removed_after_a_correction(tmp_path):
    """The live/final sidecar is written when the controller says it is due,
    and a manual correction removes a stale one (the audio no longer matches)."""
    block = {"live": 2, "final": 2, "rate": 0.0}
    stem = tmp_path / "take"

    first = te.write_take(
        _result("live", live_vs_final=block),
        stem,
        options=te.ExportOptions(archive_sidecar=True),
    )
    assert first.sidecar_written is True and SIDECAR_SUFFIX in first.suffixes
    assert Path(f"{stem}{SIDECAR_SUFFIX}").exists()

    te.write_take(
        _result("live", live_vs_final=block),
        stem,
        options=te.ExportOptions(manual_corrections=True),
    )
    assert not Path(f"{stem}{SIDECAR_SUFFIX}").exists(), "a stale audit survived"


@pytest.mark.parametrize("suffix", (".mid", ".musicxml", ".json", ".csv"))
def test_file_save_as_writes_only_the_selected_artifact(tmp_path, monkeypatch, suffix):
    """A file result has one selected output, not a hidden full bundle."""
    destination = tmp_path / "nested" / "song.other"
    receipt = te.write_take(
        _result("source/chant.wav"),
        tmp_path / "unused",
        options=te.ExportOptions(save_path=destination, format_suffix=suffix),
    )

    selected = destination.with_suffix(suffix)
    assert receipt.failed is False, receipt.warnings
    assert receipt.saved_path == selected
    assert receipt.suffixes == (suffix,)
    assert selected.is_file()
    assert list(selected.parent.iterdir()) == [selected]
    assert receipt.file_names() == [selected.name]


def test_live_save_as_keeps_the_full_archive_and_copies_the_selected_file(tmp_path):
    """The user's one-file choice must not destroy the take's audit archive."""
    archive_stem = tmp_path / "archive" / "live_take" / "live_take"
    destination = tmp_path / "chosen" / "song.mid"
    receipt = te.write_take(
        _result("live"),
        archive_stem,
        options=te.ExportOptions(save_path=destination, format_suffix=".mid"),
    )

    assert receipt.failed is False, receipt.warnings
    assert receipt.saved_path == destination
    assert destination.read_bytes() == Path(f"{archive_stem}.mid").read_bytes()
    for suffix in (".mid", ".musicxml", ".json", ".csv", ".coherence.csv"):
        assert Path(f"{archive_stem}{suffix}").is_file(), suffix
    assert list(destination.parent.iterdir()) == [destination]


def test_selected_pdf_reports_a_missing_tool_without_claiming_a_saved_file(tmp_path, monkeypatch):
    """A PDF choice that cannot be engraved is never announced as written."""
    from v2m.core import report

    def no_tool(_stem) -> None:
        raise PdfToolMissing("engraving tool is missing")

    monkeypatch.setattr(report, "find_notation_tool", lambda: None)
    monkeypatch.setattr(te, "engrave_pdf", no_tool)
    destination = tmp_path / "song.pdf"
    receipt = te.write_take(
        _result("source/chant.wav"),
        tmp_path / "unused",
        options=te.ExportOptions(save_path=destination, format_suffix=".pdf"),
    )

    assert receipt.failed is True
    assert receipt.pdf_status == te.PDF_MISSING_TOOL
    assert receipt.warning_detail("pdf") == "engraving tool is missing"
    assert receipt.saved_path is None
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_selected_export_failure_preserves_an_existing_destination(tmp_path, monkeypatch):
    """A failed scratch write cannot truncate a pre-existing user file."""
    destination = tmp_path / "song.mid"
    destination.write_bytes(b"existing file")

    def broken(*_args, **_kwargs):
        raise OSError("cannot finish export")

    monkeypatch.setattr(te, "export_result", broken)
    receipt = te.write_take(
        _result("source/chant.wav"),
        tmp_path / "unused",
        options=te.ExportOptions(save_path=destination, format_suffix=".mid"),
    )

    assert receipt.failed is True
    assert receipt.warning_detail("export") == "cannot finish export"
    assert destination.read_bytes() == b"existing file"
    assert list(tmp_path.iterdir()) == [destination]


def test_retention_prunes_only_a_live_take(tmp_path, monkeypatch):
    """The retention trim runs for a live take (its folder was just written)
    and never for a file export."""
    monkeypatch.setattr("v2m.core.retention.prune_takes", lambda _root: ["live_old"])

    live = te.write_take(_result("live"), tmp_path / "a")
    assert live.pruned == ("live_old",)

    out = tmp_path / "out"
    out.mkdir()
    file_take = te.write_take(_result("in/chant.wav"), out / "chant")
    assert file_take.pruned == ()
