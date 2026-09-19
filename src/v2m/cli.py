"""CLI: v2m analyze | v2m live | v2m gui."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core.audio_io import AUDIO_EXTS
from .core.i18n import get_language, note_display, set_language, tr
from .core.pdf_export import PdfExportError, PdfToolMissing
from .core.pipeline import AnalyzeOptions, analyze_file
from .core.report import engrave_pdf, export_result


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
    pdf_written = False
    try:
        print(tr("cli analyzing").format(path=path))
        res = analyze_file(
            str(path),
            opts=AnalyzeOptions(min_conf=args.min_conf, min_note_s=args.min_note / 1000.0),
            progress=lambda f, m: print(f"  [{int(f * 100):3d}%] {m}", end="\r"),
        )
        print()

        stem = Path(args.output).with_suffix("") if args.output else path.with_suffix("")
        quantize_s = res.quantize_step(16) if args.quantize else None
        rep = export_result(
            res, stem, quantize_s=quantize_s, transpose=args.transpose,
            with_pitch_curve=True,
        )
        if args.pdf:
            # after the report: a failed engraving must not cost the summary
            engrave_pdf(stem)
            pdf_written = True
    except KeyboardInterrupt:
        print("\n" + tr("cli interrupted"))
        return 130
    except PdfToolMissing as e:
        # no engraving tool: the PDF is skipped, not failed (exit 0)
        print("\n" + tr("err ui pdf").format(e=e), file=sys.stderr)
    except PdfExportError as e:
        # the analysis and the other artifacts succeeded: only the PDF failed
        print("\n" + tr("err ui pdf").format(e=e), file=sys.stderr)
        pdf_failed = True
    except Exception as e:
        print("\n" + tr("err analyze failed").format(e=e), file=sys.stderr)
        return 1

    # --- human-readable summary ------------------------------------------
    print(f"{tr('key tempo')}     : {res.tempo.bpm} BPM")
    print(f"{tr('key key label')} : {res.key.best.label} (conf. {res.key.best.score:.2f})")
    for a in res.key.alternatives:
        print(f"  {tr('key candidate')}: {a.label} ({a.score:.2f})")
    print(f"{tr('key meter')}    : {res.meter.label if res.meter else tr('key detect meter')}")
    print(f"{tr('key chord label')} : {rep['chord']['label'] if rep['chord'] else '—'}")
    print(f"{tr('key notes label')} : {len(res.notes)}")
    figs = rep["figures_summary"]
    if figs:
        print(f"{tr('key figures')}   : {', '.join(f'{k} ×{v}' for k, v in figs.items())}")
    exports = f"{stem}.mid, {stem}.musicxml, {stem}.json, {stem}.csv"
    if pdf_written:
        exports += f", {stem}.pdf"
    print(f"{tr('key export label')} : {exports}")
    return 0 if not pdf_failed else 2


def _cmd_live(args: argparse.Namespace) -> int:
    from .live import LiveSession

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v2m",
        description="Voix -> MIDI : notes, tempo, tonalité et accords (fr / en).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--lang", choices=("fr", "en"), default=None,
        help="Langue des messages et du format des notes (défaut : fr, persisté dans ~/.v2m_lang)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Analyze an audio file -> MIDI + MusicXML + JSON + CSV")
    p_an.add_argument("input", help="Audio file (WAV/MP3/FLAC/OGG)")
    p_an.add_argument("-o", "--output", help="Output stem (default: same name as input)")
    p_an.add_argument("--min-conf", type=float, default=0.6, help="Pitch confidence threshold (0..1)")
    p_an.add_argument("--min-note", type=int, default=60, help="Minimum note duration in ms")
    p_an.add_argument("--quantize", action="store_true", help="Quantize to the 1/16 grid at the detected tempo")
    p_an.add_argument("--transpose", type=int, default=0, help="Transpose in semitones")
    p_an.add_argument("--pdf", action="store_true", help="Also engrave the score to PDF (needs MuseScore or LilyPond)")

    p_live = sub.add_parser("live", help="Real-time mode: sing, Ctrl+C to export")
    p_live.add_argument("--min-conf", type=float, default=0.5, help="Pitch confidence threshold (0..1)")
    p_live.add_argument("--transpose", type=int, default=0, help="Transpose in semitones")

    p_gui = sub.add_parser("gui", help="Launch the graphical interface")

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
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
