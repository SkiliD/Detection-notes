"""The take-export SEQUENCE: one entry, one receipt, no Qt.

`report.export_result` writes the artifact set and stays the single artifact
writer; this module owns the POLICY around it — the order (the take's own WAV
first, the artifacts, then the external-tool PDF), the extras that are only
written on request, the two error contracts, the sidecar archive and the
retention trim. Both adapters (the GUI window and the CLI) call `write_take`
and RENDER its receipt; neither re-derives what was written.

The interface is one function and one result value:

    write_take(res, stem, signal=..., options=...) -> ExportReceipt

Nothing here raises to announce a failure: a failed export, a refused WAV and
a failed PDF are all VALUES on the receipt (`ExportWarning`s, `pdf_status`,
`failed`). Rendering lives in the adapters, which is what keeps this module
testable without an event loop, a banner or a status bar.

The ordering rules, each with its measured reason:

- **The take's WAV is written FIRST**, before `export_result`, and OUTSIDE the
  artifact try: it is the rawest form of the user's data, and a later failure
  (a full disk mid-artifact) must not cost it. Its own failure is a WARNING —
  the singer keeps the transcription even when the disk refuses the WAV.
- **The PDF is LAST and separate** (`report.engrave_pdf`): it needs an
  external tool, so a missing or broken tool must never cost the data
  artifacts. Missing tool and conversion failure are DIFFERENT outcomes on
  the receipt, so the CLI can exit 0 on one and 2 on the other while the GUI
  shows one message for both.
- **The announcement reads the DISK**: `takes.has_artifacts` decides the
  suffixes, so a half-written export can never be announced as complete.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import AnalysisResult

from . import audio_io
from .pdf_export import PdfExportError, PdfToolMissing
from .report import engrave_pdf, export_result
from .takes import SIDECAR_SUFFIX, has_artifacts, sidecar_path

# The PDF outcome, a value on the receipt (never a printed string here).
PDF_SKIPPED = "skipped"          # no PDF was requested
PDF_MISSING_TOOL = "missing_tool"  # requested, but no engraving tool is installed
PDF_FAILED = "failed"            # requested, the tool/conversion failed
PDF_WRITTEN = "written"          # requested and engraved

# Formats offered by the desktop Save As action. The coherence CSV is an
# internal audit artifact, not a standalone export choice.
SELECTABLE_FORMATS = (".mid", ".musicxml", ".pdf", ".json", ".csv")


@dataclass(frozen=True)
class ExportOptions:
    """The facts an export needs beyond the result and its stem.

    Every field is a caller choice or a caller fact — the module holds no
    window on the widgets:

    - `quantize_s` / `transpose` / `pitch_curve`: `export_result`'s knobs.
    - `record_audio`: the live "record the take's audio" box, read at the
      take's start (a file analysis owns its audio at the input path and
      writes no copy, box or no box — the module enforces that, not the
      caller).
    - `pdf`: whether the optional engraving was requested.
    - `live_notes`: the take's confirmed live notes, for the report's
      live/final breakdown (a live export only; a file analysis has none).
    - `manual_corrections`: a manual pitch edit means the recorded audio no
      longer describes the notes — no live/final block is archived and any
      stale sidecar is removed.
    - `archive_sidecar`: whether the take's live/final decomposition is due,
      decided by the single owner of the take's stem (the live session
      controller, ADR 0010) and handed over as a fact.
    - `save_path` / `format_suffix`: optional desktop Save As choice. Live
      takes still keep their complete internal archive; the selected artifact
      is additionally copied to the requested destination.
    """

    quantize_s: float | None = None
    transpose: int = 0
    pitch_curve: bool = False
    record_audio: bool = False
    pdf: bool = False
    live_notes: list | None = None
    manual_corrections: bool = False
    archive_sidecar: bool = False
    save_path: Path | None = None
    format_suffix: str | None = None


@dataclass(frozen=True)
class ExportWarning:
    """One non-fatal export problem, as a VALUE (the adapter renders it).

    `kind` is `"wav"` (the take's own audio was refused), `"pdf"` (the
    optional engraving failed — `detail` tells a missing tool from a failed
    conversion through `pdf_status`) or `"export"` (the artifact set itself
    failed, `ExportReceipt.failed` is then True).
    """

    kind: str
    detail: str


@dataclass(frozen=True)
class ExportReceipt:
    """What one export ACTUALLY wrote — the only source both adapters render.

    - `stem`: the path prefix the artifacts share.
    - `suffixes`: the artifact suffixes confirmed ON DISK, in announcement
      order (the data artifacts, then the requested WAV, the PDF and the
      sidecar when they exist).
    - `wav_written` / `pdf_status` / `sidecar_written`: what the conditional
      steps did.
    - `pruned`: the old take stems the retention trim removed (a live export).
    - `warnings`: the non-fatal problems, as values.
    - `failed`: the artifact set could not be written — the adapter must not
      announce anything.
    - `report`: the JSON dict `export_result` returned, or None on failure.
    - `saved_path`: the optional user-selected destination, if one was written.
    """

    stem: Path
    suffixes: tuple[str, ...] = ()
    wav_written: bool = False
    pdf_status: str = PDF_SKIPPED
    sidecar_written: bool = False
    pruned: tuple[str, ...] = ()
    warnings: tuple[ExportWarning, ...] = ()
    failed: bool = False
    report: dict | None = None
    saved_path: Path | None = None

    def file_names(self) -> list[str]:
        """The announced names, relative to the stem's folder (`<name><ext>`)."""
        return [f"{self.stem.name}{ext}" for ext in self.suffixes]

    def warning_detail(self, kind: str) -> str:
        """The first warning of `kind`'s detail ("" when there is none).

        The ONE extraction the adapters render for a single-message spot
        (a failed export, a PDF problem): they ask the receipt instead of
        re-filtering `warnings` themselves. Adapters that render EVERY
        warning iterate `warnings` directly.
        """
        return next((w.detail for w in self.warnings if w.kind == kind), "")


def _archive_sidecar(stem: Path, block: dict) -> bool:
    """Write the take's live/final breakdown beside its artifacts (best-effort)."""
    from .compare import write_sidecar

    return bool(write_sidecar(stem, block))


def _delete_sidecar(stem: Path) -> None:
    """Remove a stale live/final sidecar after a manual pitch correction."""
    try:
        sidecar_path(stem.parent, stem.name).unlink(missing_ok=True)
    except OSError:
        pass  # best-effort: a stale audit's removal must never cost the take


def _prune_takes() -> tuple[str, ...]:
    """Trim the oldest take folders under the retention budget (best-effort)."""
    from .retention import prune_takes
    from .takes import LIVE_TAKES_DIR

    try:
        return tuple(prune_takes(LIVE_TAKES_DIR))
    except Exception:  # a trim failure must never cost the take
        return ()


def _selected_destination(path: str | Path, suffix: str) -> Path:
    """Make the selected format authoritative for the filename extension."""
    destination = Path(path).expanduser()
    if destination.suffix.lower() != suffix:
        destination = destination.with_suffix(suffix)
    return destination


def _write_selected_file(
    res: AnalysisResult, destination: Path, suffix: str, options: ExportOptions
) -> ExportReceipt:
    """Write one chosen format without leaving the other report artifacts behind."""
    if suffix not in SELECTABLE_FORMATS:
        raise ValueError(f"Unsupported export format: {suffix}")

    destination = _selected_destination(destination, suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v2m-export-", dir=destination.parent) as temp:
        scratch = Path(temp) / "export"
        report = export_result(
            res,
            scratch,
            quantize_s=options.quantize_s,
            transpose=options.transpose,
            with_pitch_curve=options.pitch_curve,
            live_notes=options.live_notes if not options.manual_corrections else None,
        )
        if suffix == ".pdf":
            engrave_pdf(scratch)
            source = Path(f"{scratch}.pdf")
        else:
            source = Path(f"{scratch}{suffix}")
        if not source.is_file():
            raise OSError(f"Export writer did not create {source.name}")
        os.replace(source, destination)

    return ExportReceipt(
        stem=destination.with_suffix(""),
        suffixes=(suffix,),
        pdf_status=PDF_WRITTEN if suffix == ".pdf" else PDF_SKIPPED,
        report=report,
        saved_path=destination,
    )


def write_take(
    res: AnalysisResult,
    stem: str | Path,
    *,
    signal: Any = None,
    options: ExportOptions | None = None,
) -> ExportReceipt:
    """Write one take's artifact set; return what was written (never raises).

    `res` is the `AnalysisResult`, `stem` the artifact prefix (already named by
    the caller — the live session controller mints a take's stem ONCE, ADR
    0010), `signal` the live take's audio (None for a file analysis) and
    `options` the caller's choices. The sequence and its reasons live in the
    module docstring; the outcome is always a receipt.
    """
    options = options or ExportOptions()
    stem = Path(stem)
    live = res.source == "live"
    warnings: list[ExportWarning] = []

    # File analyses have no take archive to preserve: save exactly the chosen
    # artifact, keeping scratch files beside the destination for atomic replace.
    if not live and options.save_path is not None:
        try:
            if options.format_suffix is None:
                raise ValueError("A selected export needs a format suffix")
            if options.format_suffix not in SELECTABLE_FORMATS:
                raise ValueError(f"Unsupported export format: {options.format_suffix}")
            return _write_selected_file(
                res,
                _selected_destination(options.save_path, options.format_suffix),
                options.format_suffix,
                options,
            )
        except PdfToolMissing as error:
            selected = _selected_destination(
                options.save_path, options.format_suffix or ".pdf"
            )
            return ExportReceipt(
                stem=selected.with_suffix(""),
                pdf_status=PDF_MISSING_TOOL,
                warnings=(ExportWarning("pdf", str(error)),),
                failed=True,
            )
        except PdfExportError as error:
            selected = _selected_destination(
                options.save_path, options.format_suffix or ".pdf"
            )
            return ExportReceipt(
                stem=selected.with_suffix(""),
                pdf_status=PDF_FAILED,
                warnings=(ExportWarning("pdf", str(error)),),
                failed=True,
            )
        except Exception as error:
            return ExportReceipt(
                stem=Path(options.save_path).with_suffix(""),
                warnings=(ExportWarning("export", str(error)),),
                failed=True,
            )

    # 1. The take's own audio FIRST, when asked for and only for a live take.
    wav_written = False
    if live and options.record_audio and signal is not None and len(signal):
        try:
            audio_io.save_file(signal, str(stem) + ".wav", audio_io.SR)
            wav_written = True
        except Exception as e:
            warnings.append(ExportWarning("wav", str(e)))

    # 2. The artifacts, then the optional PDF (an external tool, always last).
    pdf_status = PDF_SKIPPED
    try:
        report = export_result(
            res,
            stem,
            quantize_s=options.quantize_s,
            transpose=options.transpose,
            with_pitch_curve=options.pitch_curve,
            # the take's live reading, for the report's live/final breakdown:
            # a file analysis hands over none, and a manual correction makes
            # the recorded audio describe other notes than the result carries
            live_notes=(
                options.live_notes if live and not options.manual_corrections else None
            ),
        )
        if options.pdf:
            try:
                engrave_pdf(stem)
                pdf_status = PDF_WRITTEN
            except PdfToolMissing as e:
                pdf_status = PDF_MISSING_TOOL
                warnings.append(ExportWarning("pdf", str(e)))
            except PdfExportError as e:
                pdf_status = PDF_FAILED
                warnings.append(ExportWarning("pdf", str(e)))
    except Exception as e:  # never crash a caller on a failed export
        warnings.append(ExportWarning("export", str(e)))
        return ExportReceipt(
            stem=stem,
            wav_written=wav_written,
            pdf_status=pdf_status,
            warnings=tuple(warnings),
            failed=True,
        )

    # 3. The live/final sidecar: archived when due, removed when a manual
    #    correction made the recorded audio stale.
    sidecar_written = False
    if live:
        block = getattr(res, "live_vs_final", None) or {}
        if options.archive_sidecar and block:
            sidecar_written = _archive_sidecar(stem, block)
        elif options.manual_corrections:
            _delete_sidecar(stem)

    # 4. The announcement reads the DISK: only what exists is listed.
    extras: list[str] = []
    if wav_written:
        extras.append(".wav")
    if pdf_status == PDF_WRITTEN:
        extras.append(".pdf")
    if sidecar_written:
        extras.append(SIDECAR_SUFFIX)
    suffixes = tuple(has_artifacts(stem, tuple(extras)))

    # 5. The retention trim runs at export time (a live take was just
    #    written): the oldest folders go whole under the documented budget.
    pruned = _prune_takes() if live else ()

    saved_path = None
    if live and options.save_path is not None and options.format_suffix is not None:
        try:
            if options.format_suffix not in SELECTABLE_FORMATS:
                raise ValueError(f"Unsupported export format: {options.format_suffix}")
            saved_path = _selected_destination(options.save_path, options.format_suffix)
        except (TypeError, ValueError) as error:
            warnings.append(ExportWarning("export", str(error)))
            return ExportReceipt(
                stem=stem,
                suffixes=suffixes,
                wav_written=wav_written,
                pdf_status=pdf_status,
                sidecar_written=sidecar_written,
                pruned=pruned,
                warnings=tuple(warnings),
                report=report,
                failed=True,
            )
        archive_path = Path(f"{stem}{options.format_suffix}")
        try:
            if not archive_path.is_file():
                raise OSError(f"The take archive has no {options.format_suffix} artifact")
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            if archive_path.resolve() != saved_path.resolve():
                with tempfile.NamedTemporaryFile(
                    prefix=f".{saved_path.name}.",
                    suffix=".tmp",
                    dir=saved_path.parent,
                    delete=False,
                ) as staged:
                    staged_path = Path(staged.name)
                try:
                    shutil.copy2(archive_path, staged_path)
                    os.replace(staged_path, saved_path)
                finally:
                    staged_path.unlink(missing_ok=True)
        except OSError as error:
            warnings.append(ExportWarning("export", str(error)))
            return ExportReceipt(
                stem=stem,
                suffixes=suffixes,
                wav_written=wav_written,
                pdf_status=pdf_status,
                sidecar_written=sidecar_written,
                pruned=pruned,
                warnings=tuple(warnings),
                report=report,
                failed=True,
            )

    return ExportReceipt(
        stem=stem,
        suffixes=suffixes,
        wav_written=wav_written,
        pdf_status=pdf_status,
        sidecar_written=sidecar_written,
        pruned=pruned,
        warnings=tuple(warnings),
        report=report,
        saved_path=saved_path,
    )
