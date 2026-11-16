"""CLI: v2m analyze | v2m live | v2m gui | v2m serve."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .core.audio_io import AUDIO_EXTS
from .core.companion import DEFAULT_PORT as COMPANION_PORT
from .core.i18n import (
    get_language,
    note_display_pair,
    set_language,
    tr,
    tr_chord_label,
    tr_figure,
    tr_key_label,
)
from .core.live_session import LiveEvent, LiveTake
from .core.notes import DEFAULT_MIN_NOTE_S
from .core.pdf_export import PdfExportError
from .core.pipeline import FILE_CONF, LIVE_CONF, AnalyzeOptions, analyze_file
from .core.take_export import PDF_FAILED, PDF_MISSING_TOOL, ExportOptions, ExportReceipt, write_take
from .core.takes import live_export_dir, live_take_name


def _cmd_analyze(args: argparse.Namespace) -> int:
    # --- validate inputs early, with clear messages -----------------------
    path = Path(args.input)
    if not path.exists():
        print(tr("err file not found").format(path=path), file=sys.stderr)
        return 1
    if path.suffix.lower() not in AUDIO_EXTS:
        fmts = ", ".join(sorted(e.lstrip(".") for e in AUDIO_EXTS))
        print(tr("err bad format").format(ext=path.suffix or "?", fmts=fmts), file=sys.stderr)
        return 1
    if args.min_conf < 0.0 or args.min_conf > 1.0:
        print(tr("err minconf").format(v=args.min_conf), file=sys.stderr)
        return 1

    pdf_failed = False
    receipt: ExportReceipt | None = None
    try:
        print(tr("cli analyzing").format(path=path))
        file_analysis = analyze_file(
            str(path),
            opts=AnalyzeOptions(min_conf=args.min_conf, min_note_s=args.min_note / 1000.0),
            progress=lambda f, m: print(f"  [{int(f * 100):3d}%] {m}", end="\r"),
        )
        res = file_analysis.analysis
        print()

        stem = Path(args.output).with_suffix("") if args.output else path.with_suffix("")
        quantize_s = res.quantize_step(16) if args.quantize else None
        # ONE export entry for both adapters (the GUI window and this one):
        # the module owns the sequence and the PDF's error contract, this
        # adapter only RENDERS the receipt — a missing tool and a failed
        # conversion are the SAME message here, they differ by exit code.
        receipt = write_take(
            res,
            stem,
            options=ExportOptions(
                quantize_s=quantize_s,
                transpose=args.transpose,
                pitch_curve=True,
                pdf=args.pdf,
            ),
        )
        if receipt.failed:
            detail = receipt.warning_detail("export")
            print("\n" + tr("err analyze failed").format(e=detail), file=sys.stderr)
            return 1
        if receipt.pdf_status in (PDF_MISSING_TOOL, PDF_FAILED):
            detail = receipt.warning_detail("pdf")
            print("\n" + tr("err ui pdf").format(e=detail), file=sys.stderr)
            # a missing tool is skipped (exit 0), a conversion failure is not
            pdf_failed = receipt.pdf_status == PDF_FAILED
    except KeyboardInterrupt:
        print("\n" + tr("cli interrupted"))
        return 130
    except Exception as e:
        print("\n" + tr("err analyze failed").format(e=e), file=sys.stderr)
        return 1
    assert receipt is not None and receipt.report is not None
    rep = receipt.report

    # --- human-readable summary ------------------------------------------
    # analyze_signal always produces a tempo and a key (the detectors return
    # an object even on silence), so the summary reads them unguarded.
    assert res.tempo is not None and res.key is not None
    # the summary is display: key labels ('C major') and figure names
    # ('noire') are stored canonical and translated HERE, never in the data
    print(f"{tr('key tempo')}     : {res.tempo.bpm} BPM")
    print(f"{tr('key key label')} : {tr_key_label(res.key.best.label)} (conf. {res.key.best.score:.2f})")
    for a in res.key.alternatives:
        print(f"  {tr('key candidate')}: {tr_key_label(a.label)} ({a.score:.2f})")
    print(f"{tr('key meter'):<10}: {res.meter.label if res.meter else tr('key detect meter')}")
    print(f"{tr('key chord label')} : {tr_chord_label(rep['chord']['label']) if rep['chord'] else '—'}")
    print(f"{tr('key notes label')} : {len(res.notes)}")
    figs = rep["figures_summary"]
    if figs:
        print(f"{tr('key figures')}   : {', '.join(f'{tr_figure(k)} ×{v}' for k, v in figs.items())}")
    # one owner for the artifact set (the receipt's on-disk suffixes): the
    # summary announces what the export ACTUALLY wrote, nothing it did not
    exports = ", ".join(f"{stem}{ext}" for ext in receipt.suffixes)
    print(f"{tr('key export label')} : {exports}")
    return 0 if not pdf_failed else 2


class LiveSession:
    """The `v2m live` adapter: print a shared LiveTake, export its result.

    One module answers to "the live session" — `core/live_session.py`,
    the actual owner; this CLI adapter is thin by construction (~60
    lines of printing and the export call) and lives here so the name
    stays unambiguous when navigating. Consumption-only, per ADR 0001.
    """

    def __init__(self, min_conf: float = LIVE_CONF, transpose: int = 0, out_stem: str | None = None):
        self.min_conf = min_conf
        self.transpose = transpose
        self.out_stem = out_stem or live_take_name()
        self.take = LiveTake(min_conf=min_conf, on_event=self._on_event)

    def _on_event(self, event: LiveEvent) -> None:
        if event.kind == "note_closed" and event.note is not None:
            note = event.note
            # The figures are already labeled by LiveTake (the policy's
            # single owner) BEFORE this event is emitted: the adapter only
            # prints them.
            fig = f"  [{tr_figure(note.figure)}]" if note.figure else ""
            print(f"  ♪ {note_display_pair(note.midi)}  ({note.duration:.2f} s){fig}")

    def run(self) -> None:
        print(tr("cli live banner"))
        self.take.start()
        try:
            while True:
                time.sleep(0.1)
                seg = self.take.segmenter
                if seg is None:
                    continue
                cur = seg.display_text()
                total = len(seg.notes)
                print(
                    f"\r  {self.take.elapsed:7.1f} s | {cur:>24} | {total} notes ",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.take.stop()
        print(tr("cli stop capture"))
        self.export()

    def export(self) -> str | None:
        if self.take.state != "stopped":
            self.take.stop()
        if self.take.result is None and self.take.signal.size < self.take.sr // 2:
            print(tr("cli too short"))
            return None
        print(tr("key final analysis"))
        # The analysis has ONE path: LiveTake.finalize owns the drain, the
        # length gate and the idempotency; the adapter only exports what it
        # returns (a short take re-raises the take's own 'failed' error).
        res, signal = self.take.finalize()
        if res.tempo is None or res.key is None:
            raise ValueError(tr("err live incomplete"))
        take_dir = live_export_dir(self.out_stem)
        receipt = write_take(
            res,
            take_dir / self.out_stem,
            signal=signal,
            options=ExportOptions(
                transpose=self.transpose,
                record_audio=True,
                live_notes=(self.take.segmenter.notes if self.take.segmenter else []),
                # no controller in this adapter: `write_take` gates the
                # sidecar on the block's presence itself
                archive_sidecar=True,
            ),
        )
        if receipt.failed:
            raise RuntimeError(
                tr("err analyze failed").format(e=receipt.warning_detail("export"))
            )
        for warning in receipt.warnings:
            if warning.kind == "wav":
                print(tr("err ui wav failed").format(e=warning.detail), file=sys.stderr)
        print(f"{tr('key tempo')}    : {res.tempo.bpm} BPM")
        print(f"{tr('key key label')} : {tr_key_label(res.key.best.label)}")
        if res.chord:
            print(f"{tr('key chord label')} : {tr_chord_label(res.chord.best.label)}")
        print(f"{tr('key notes label')} : {len(res.notes)}")
        listed = ", ".join(str(take_dir / name) for name in receipt.file_names())
        print(f"{tr('key export label')} : {listed}")
        return str(take_dir / f"{self.out_stem}.mid")


def _cmd_live(args: argparse.Namespace) -> int:
    try:
        session = LiveSession(min_conf=args.min_conf, transpose=args.transpose)
        session.run()
    except KeyboardInterrupt:
        print("\n" + tr("cli interrupted"))
        return 130
    except Exception as e:
        print(tr("err live failed").format(e=e), file=sys.stderr)
        return 1
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from .gui.main_window import run_gui

    run_gui()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface. Threshold defaults come from pipeline's one policy."""
    parser = argparse.ArgumentParser(
        prog="v2m",
        description="Voice -> MIDI: notes, tempo, key and chords (fr / en).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--lang", choices=("fr", "en"), default=None,
        help="Message and note-name language (default: fr, persisted in ~/.v2m_lang)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Analyze an audio file -> MIDI + MusicXML + JSON + CSV")
    p_an.add_argument("input", help="Audio file (WAV/MP3/FLAC/OGG)")
    p_an.add_argument("-o", "--output", help="Output stem (default: same name as input)")
    p_an.add_argument("--min-conf", type=float, default=FILE_CONF, help="Pitch confidence threshold (0..1)")
    p_an.add_argument("--min-note", type=int, default=int(DEFAULT_MIN_NOTE_S * 1000), help="Minimum note duration in ms")
    p_an.add_argument("--quantize", action="store_true", help="Quantize to the 1/16 grid at the detected tempo")
    p_an.add_argument("--transpose", type=int, default=0, help="Transpose in semitones")
    p_an.add_argument("--pdf", action="store_true", help="Also engrave the score to PDF (needs MuseScore or LilyPond)")

    p_live = sub.add_parser("live", help="Real-time mode: sing, Ctrl+C to export")
    p_live.add_argument("--min-conf", type=float, default=LIVE_CONF, help="Pitch confidence threshold (0..1)")
    p_live.add_argument("--transpose", type=int, default=0, help="Transpose in semitones")

    p_gui = sub.add_parser("gui", help="Launch the graphical interface")

    p_serve = sub.add_parser(
        "serve",
        help="Serve /health, /analyze, /render and /compare to the web page on 127.0.0.1 (opt-in)",
    )
    p_serve.add_argument(
        "--port", type=int, default=COMPANION_PORT,
        help=f"Loopback port (default: {COMPANION_PORT})",
    )
    p_serve.add_argument(
        "--allow-origin", action="append", default=[], metavar="ORIGIN",
        help=(
            "Also let this page origin call it (repeatable; a loopback page and "
            "non-browser clients are always allowed)"
        ),
    )

    p_wk = sub.add_parser(
        "weekly", help="Weekly recap: real-time health, best practice time of day, days"
    )
    p_wk.add_argument(
        "root", nargs="?", default=None,
        help="Take folder (default: prises_live)",
    )
    p_wk.add_argument(
        "--pdf", action="store_true",
        help="Also engrave the recap to PDF (needs MuseScore or LilyPond)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()

    args = parser.parse_args(argv)
    if args.lang:
        set_language(args.lang)
    else:
        get_language()  # restore the persisted choice
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "live":
        return _cmd_live(args)
    if args.command == "gui":
        return _cmd_gui(args)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "weekly":
        return _cmd_weekly(args)
    parser.print_help()
    return 1


def _cmd_serve(args: argparse.Namespace) -> int:
    """The companion: loopback only, until Ctrl+C.

    It prints the URL it actually bound (port 0 is a legal request), so the
    user pastes a real address into the page's settings instead of a guess.
    """
    from .core.companion import make_server

    server = make_server(args.port, tuple(args.allow_origin))
    host, port = server.server_address[:2]
    # flushed: the URL is the one instruction the user needs, and a redirect to
    # a log file must not swallow it until the process ends
    print(tr("cli companion listening").format(url=f"http://{host}:{port}"), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print(tr("cli companion stopped"), flush=True)
    return 0


def _cmd_weekly(args: argparse.Namespace) -> int:
    """The weekly recap: text always, PDF on request (same split as analyze)."""
    from .core import weekly as wk
    from .core.retention import prune_observations, prune_takes
    from .core.weekly_pdf import (
        WeeklyPdfExportError,
        WeeklyPdfToolMissing,
        engrave_weekly_pdf,
    )

    root = Path(args.root) if args.root else None
    data = wk.build_weekly(root)
    # the retention runs where the user asked for a recap: the folder is
    # looked at, its size is the moment to trim BOTH sweeps — orphan
    # caches, then whole take folders under the budget (the journal keeps
    # history)
    base = root or Path("prises_live")
    pruned = prune_observations(base)
    if pruned:
        print(tr("cli weekly pruned caches").format(n=pruned))
    pruned_takes = prune_takes(base)
    if pruned_takes:
        print(tr("ui retention pruned").format(
            n=len(pruned_takes), stems=", ".join(pruned_takes)
        ))
    text = wk.weekly_text(data)
    out = sys.stdout.buffer if hasattr(sys.stdout, "buffer") else sys.stdout
    out.write(text.encode("utf-8", errors="replace") if hasattr(out, "write") else text)  # type: ignore[union-attr]
    out.flush()
    if args.pdf:
        stem = base / "recap_hebdo"
        try:
            path = engrave_weekly_pdf(data, stem)
            print(tr("cli weekly pdf ok").format(path=path))
        except WeeklyPdfToolMissing:
            print(tr("err pdf no tool").format(hint=""))
            return 0  # a missing tool is a skip, not a failure (the analyze rule)
        except (WeeklyPdfExportError, PdfExportError) as e:
            print(tr("err ui pdf").format(e=e), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
