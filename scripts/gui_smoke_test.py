"""Smoke test GUI offscreen : fenêtre, panneau résultats, mode notation.

Usage : QT_QPA_PLATFORM=offscreen python scripts/gui_smoke_test.py [audio.wav]

Important : utiliser QApplication (pas QCoreApplication) — les widgets Qt
plantent nativement sous une simple QCoreApplication.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from v2m.core.pipeline import analyze_file
from v2m.gui.main_window import MainWindow


def main() -> int:
    print("[smoke] démarrage", flush=True)
    app = QApplication(sys.argv)
    print("[smoke] qapp ok", flush=True)

    audio = sys.argv[1] if len(sys.argv) > 1 else "demo.wav"
    res = analyze_file(audio)
    print(f"analyse ok : {len(res.notes)} notes, {res.tempo.bpm} bpm", flush=True)

    win = MainWindow()
    print("fenetre ok", flush=True)

    win.results.set_results(res)
    print("set_results ok", flush=True)

    win.piano.set_notation_mode(True)
    print("notation on ok", flush=True)

    win.piano.set_notation_mode(True)  # idempotent
    print("notation re-on ok", flush=True)

    win.piano.set_notation_mode(False)
    print("notation off ok", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
