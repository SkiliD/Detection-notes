"""CLI : v2m analyze | v2m live | v2m gui."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core import audio_io
from .core.pipeline import AnalyzeOptions, analyze_file, analyze_signal, export_result


def _cmd_analyze(args: argparse.Namespace) -> int:
    path = args.input
    if not Path(path).exists():
        print(f"Fichier introuvable : {path}", file=sys.stderr)
        return 1

    opts = AnalyzeOptions(
        min_conf=args.min_conf,
        min_note_s=args.min_note / 1000.0,
    )
    print(f"Analyse de {path} …")
    res = analyze_file(path, opts=opts, progress=lambda f, m: print(f"  [{int(f*100):3d}%] {m}", end="\r"))
    print()

    stem = Path(args.output).with_suffix("") if args.output else Path(path).with_suffix("")
    quantize_s = res.quantize_step(16) if args.quantize else None
    rep = export_result(res, stem, quantize_s=quantize_s, transpose=args.transpose,
                        with_pitch_curve=True)

    print(f"Tempo     : {res.tempo.bpm} BPM")
    print(f"Tonalité  : {res.key.best.label} (conf. {res.key.best.score:.2f})")
    for a in res.key.alternatives:
        print(f"  candidat: {a.label} ({a.score:.2f})")
    print(f"Notes     : {len(res.notes)}")
    figs = rep["figures_summary"]
    if figs:
        print(f"Figures   : {', '.join(f'{k} ×{v}' for k, v in figs.items())}")
    print(f"Export    : {stem}.mid, {stem}.musicxml, {stem}.json, {stem}.csv")
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    from .live import LiveSession

    session = LiveSession(min_conf=args.min_conf, transpose=args.transpose)
    try:
        session.run()
    except KeyboardInterrupt:
        print("\nInterrompu — export des résultats…")
        session.export()
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from .gui.main_window import run_gui

    run_gui()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v2m",
        description="Voix -> MIDI : détection de notes, tempo et tonalité.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Analyse un fichier audio -> MIDI + MusicXML + JSON + CSV")
    p_an.add_argument("input", help="Fichier audio (WAV/MP3/FLAC/OGG)")
    p_an.add_argument("-o", "--output", help="Base de sortie (défaut : même nom que l'entrée)")
    p_an.add_argument("--min-conf", type=float, default=0.6, help="Seuil de confiance pitch (0..1)")
    p_an.add_argument("--min-note", type=int, default=60, help="Durée minimale d'une note en ms")
    p_an.add_argument("--quantize", action="store_true", help="Quantifie sur la grille 1/16 selon le tempo détecté")
    p_an.add_argument("--transpose", type=int, default=0, help="Transposition en demi-tons")

    p_live = sub.add_parser("live", help="Mode temps réel : chanter au micro, Ctrl+C pour exporter")
    p_live.add_argument("--min-conf", type=float, default=0.5, help="Seuil de confiance pitch (0..1)")
    p_live.add_argument("--transpose", type=int, default=0, help="Transposition en demi-tons")

    p_gui = sub.add_parser("gui", help="Lance l'interface graphique")

    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "live":
        return _cmd_live(args)
    if args.command == "gui":
        return _cmd_gui(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
