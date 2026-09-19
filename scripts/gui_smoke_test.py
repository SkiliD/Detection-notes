"""Offscreen GUI smoke test: window, results panel, notation mode, error banner,
playhead, piano keyboard and the polished first-run states.

Usage: QT_QPA_PLATFORM=offscreen python scripts/gui_smoke_test.py [audio.wav]

Important: use QApplication (not QCoreApplication) - Qt widgets crash
natively under a bare QCoreApplication.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from v2m.core.pipeline import analyze_file
from v2m.gui.main_window import MainWindow


def main() -> int:
    print("[smoke] startup", flush=True)
    app = QApplication(sys.argv)
    print("[smoke] qapp ok", flush=True)

    audio = sys.argv[1] if len(sys.argv) > 1 else "demo.wav"
    res = analyze_file(audio)
    print(f"analyse ok : {len(res.notes)} notes, {res.tempo.bpm} bpm", flush=True)

    win = MainWindow()
    print("fenetre ok", flush=True)

    win.results.set_results(res)
    print("set_results ok (chord + tables)", flush=True)

    win.piano.set_notation_mode(True)
    print("notation on ok", flush=True)

    win.piano.set_notation_mode(True)  # idempotent
    print("notation re-on ok", flush=True)

    win.piano.set_notation_mode(False)
    print("notation off ok", flush=True)

    win.show_error("erreur de test")
    assert not win.error_banner.isHidden(), "banner should be visible"
    win.error_banner.clear()
    assert win.error_banner.isHidden(), "banner should hide again"
    print("error banner ok", flush=True)

    # polish: guidance card, PDF badge and empty-state text are translated
    from v2m.core.i18n import tr as _tr

    assert win._steps.text() == _tr("ui steps")
    assert win.lbl_pdf_note.text() == _tr("ui pdf badge")
    assert win.results._waiting.text() == _tr("ui panel waiting")

    # playback primitives: playhead shows/hides, empty-result guard holds
    win.result = None
    win.on_toggle_play()  # no result yet: must be a no-op, not a crash
    win.piano.set_playhead(1.0)
    assert win.piano._play_line.isVisible()
    win.piano.seek_requested.emit(0.5)  # no player yet: no-op, no crash
    win.piano.hide_playhead()
    assert not win.piano._play_line.isVisible()
    print("polish states ok", flush=True)

    # piano keyboard: indexes notes, lights keys with the playhead, paints
    win.piano_keys.set_notes(res.notes)
    assert win.piano_keys._spans
    win.piano_keys.set_playhead(res.notes[0].start + 0.1)
    assert res.notes[0].midi in win.piano_keys._active
    win.piano_keys.set_playhead(-1.0)
    assert not win.piano_keys._active
    assert not win.piano_keys.grab().isNull()
    print("piano keyboard ok", flush=True)

    # language toggle: fr -> en -> fr, labels must follow
    from v2m.core.i18n import get_language, set_language

    set_language("fr")
    fr_text = win.btn_file.text()
    win.on_toggle_language()
    assert get_language() == "en"
    assert win.btn_file.text() != fr_text, "button label must change language"
    assert win.btn_file.text() == "📁  Analyze an audio file"
    win.on_toggle_language()
    assert get_language() == "fr"
    assert win.btn_file.text() == fr_text
    print("language toggle ok", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
