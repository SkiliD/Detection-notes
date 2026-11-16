"""Offscreen GUI smoke test: window, results panel, notation mode, error banner,
playhead, piano keyboard and the polished first-run states.

Usage: QT_QPA_PLATFORM=offscreen python scripts/gui_smoke_test.py [audio.wav]

Important: use QApplication (not QCoreApplication) - Qt widgets crash
natively under a bare QCoreApplication.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

# Deterministic language: sandbox LANG_FILE (as tests/conftest.py does for the
# suite) and force fr BEFORE the window is built — the assertions are written
# against fr. Without this, a persisted "en" (GUI toggle, or a CLI --lang en
# run) breaks every fr-expecting assertion before the toggle section runs.
import v2m.core.i18n as i18n
from v2m.core.pipeline import analyze_file
from v2m.gui.main_window import MainWindow

i18n.LANG_FILE = Path(tempfile.gettempdir()) / ".v2m_lang_smoke_test"
i18n.set_language("fr")


def main() -> int:
    print("[smoke] startup", flush=True)
    app = QApplication(sys.argv)
    print("[smoke] qapp ok", flush=True)

    audio = sys.argv[1] if len(sys.argv) > 1 else "demo.wav"
    file_analysis = analyze_file(audio)
    res = file_analysis.analysis
    assert res.tempo is not None, "file analysis must produce a tempo"
    assert len(file_analysis.audio) > 0, "file analysis must return its decoded audio"
    print(f"analyse ok : {len(res.notes)} notes, {res.tempo.bpm} bpm", flush=True)

    win = MainWindow()
    print("fenetre ok", flush=True)

    win.page_results.results.set_results(res)
    print("set_results ok (chord + tables)", flush=True)

    win.page_results.piano.set_notation_mode(True)
    print("notation on ok", flush=True)

    win.page_results.piano.set_notation_mode(True)  # idempotent
    print("notation re-on ok", flush=True)

    win.page_results.piano.set_notation_mode(False)
    print("notation off ok", flush=True)

    win.show_error("erreur de test")
    assert not win.error_banner.isHidden(), "banner should be visible"
    win.error_banner.clear()
    assert win.error_banner.isHidden(), "banner should hide again"
    print("error banner ok", flush=True)

    # polish: guidance card, PDF badge and empty-state text are translated
    from v2m.core.i18n import tr as _tr

    assert win.page_home.steps.text() == _tr("ui steps")
    assert win.page_results.lbl_pdf_note.text() == _tr("ui pdf badge")
    assert win.page_results.results._waiting.text() == _tr("ui panel waiting")

    # playback primitives: playhead shows/hides, empty-result guard holds
    win.result = None
    win.on_toggle_play()  # no result yet: must be a no-op, not a crash
    win.page_results.piano.set_playhead(1.0)
    assert win.page_results.piano._play_line.isVisible()
    win.page_results.piano.seek_requested.emit(0.5)  # no player yet: no-op, no crash
    win.page_results.piano.hide_playhead()
    assert not win.page_results.piano._play_line.isVisible()
    print("polish states ok", flush=True)

    # piano keyboard: indexes notes, lights keys with the playhead, paints
    win.page_results.piano_keys.set_notes(res.notes)
    assert win.page_results.piano_keys._spans
    win.page_results.piano_keys.set_playhead(res.notes[0].start + 0.1)
    assert res.notes[0].midi in win.page_results.piano_keys._active
    win.page_results.piano_keys.set_playhead(-1.0)
    assert not win.page_results.piano_keys._active
    assert not win.page_results.piano_keys.grab().isNull()
    print("piano keyboard ok", flush=True)

    # results page by TASK: named groups, named listen actions, and ONE
    # reading surface at a time (the page must read as tasks, never as a
    # grid of symbols where every view is stacked at once)
    assert win.page_results.group_listen.title() == _tr("ui group listen")
    assert win.page_results.group_adjust.title() == _tr("ui group adjust")
    assert win.page_results.group_export.title() == _tr("ui group export")
    assert win.page_results.btn_voice.text() == f"▶ {_tr('ui play voice')}"
    assert win.page_results.btn_export.text() == _tr("ui export")
    assert win.page_results.group_export.isAncestorOf(win.page_results.btn_export)
    assert win.page_results.options_host.isHidden()  # advanced settings collapsed
    assert win.page_results.view_stack.currentWidget() is win.page_results.piano
    win.page_results.btn_view_staff.click()
    assert win.page_results.view_stack.currentWidget() is win.page_results.staff
    assert win.page_results.btn_view_staff.isChecked()
    win.page_results.btn_view_wave.click()
    assert win.page_results.view_stack.currentWidget() is win.page_results.waveform
    win.page_results.btn_view_notes.click()
    assert win.page_results.view_stack.currentWidget() is win.page_results.piano
    assert not win.page_results.group_listen.isHidden()  # listening is offered at once
    assert win.page_results.group_adjust.isHidden()
    assert win.page_results.group_export.isHidden()
    assert win.page_results.btn_guide.text() in (
        _tr("ui guide listen passage"),
        _tr("ui guide listen take"),
        _tr("ui export"),
    )
    win.page_results.tools_toggle.click()
    assert not win.page_results.group_listen.isHidden()
    assert not win.page_results.group_adjust.isHidden()
    win.page_results.tools_toggle.click()
    win.page_results.btn_guide_skip.click()
    assert win.page_results.btn_guide.text() == _tr("ui export")
    print("results page tasks and guidance ok", flush=True)

    # note finder: a named voice span by default, the whole piano on request,
    # labels translated, no capture started here
    assert not win.page_finder.thread
    assert len(win.page_finder.keys._whites) == 22  # C3..C6
    win.page_finder.combo_range.setCurrentIndex(2)
    assert len(win.page_finder.keys._whites) == 52  # A0..C8
    win.page_finder.combo_range.setCurrentIndex(0)
    assert (win.page_finder.keys.min_midi, win.page_finder.keys.max_midi) == (48, 84)
    win.page_finder.keys.set_pressed({57})
    assert win.page_finder.keys._active == {57}
    assert win.page_home.btn_finder.text() == _tr("ui finder button")
    assert win.page_finder.label_hint.text() == _tr("ui finder hint")
    assert win.page_finder.label_note.text() == _tr("ui finder idle")
    # first run must read as states, not as a blank page: the mic says whether
    # it is open, and the invite is not styled like an answer
    assert win.page_finder.btn_toggle.text() == _tr("ui finder start")
    assert win.page_finder.label_state.text() == _tr("ui finder off")
    idle_style = win.page_finder.label_note.styleSheet()
    win._set_finder_note(60)
    assert win.page_finder.label_note.text() != _tr("ui finder idle")
    assert win.page_finder.label_note.styleSheet() != idle_style
    win._set_finder_note(None)
    assert win.page_finder.label_note.styleSheet() == idle_style
    print("note finder ok", flush=True)

    # language toggle: fr -> en -> fr, labels must follow
    from v2m.core.i18n import get_language, set_language

    set_language("fr")
    fr_text = win.page_home.btn_file.text()
    win.on_toggle_language()
    assert get_language() == "en"
    assert win.page_home.btn_file.text() != fr_text, "button label must change language"
    assert win.page_home.btn_file.text() == _tr("ui open file")
    assert win.page_finder.label_heard.text() == ""
    win.page_finder._thread = cast(
        Any, type("RunningThread", (), {"isRunning": lambda self: True})()
    )
    win.page_finder._last_rms = 0.0
    win.page_finder._note_heard = False
    win.page_finder._refresh_state()
    win.page_finder._refresh_input_state()
    assert win.page_finder.label_heard.text() == _tr("ui finder waiting")
    win.page_finder._thread = None
    win.page_finder._refresh_state()
    win.page_finder._refresh_input_state()
    assert win.page_finder.label_heard.text() == ""
    # The roll's axis titles follow the selected language too.
    assert win.page_results.piano.getPlotItem().getAxis("bottom").labelText == _tr("ui axis time")
    assert win.page_results.piano.getPlotItem().getAxis("left").labelText == _tr("ui axis note")
    win.on_toggle_language()
    assert get_language() == "fr"
    assert win.page_home.btn_file.text() == fr_text
    print("language toggle ok", flush=True)

    # tuning meter: the cents needle follows the sung pitch, clamped at ±50
    win.page_live.live_indicator.update_note(60.2, 265.0)
    assert win.page_live.live_indicator.tuning._cents == pytest.approx(20.0, abs=0.5)
    win.page_live.live_indicator.update_note(None, float("nan"))
    assert win.page_live.live_indicator.tuning._cents is None
    assert not win.page_live.live_indicator.tuning.grab().isNull()
    print("tuning meter ok", flush=True)

    # metronome: opt-in, tempo editable, beat counter runs on the Qt loop
    assert win.metronome is None  # off by default
    win.page_live.chk_metronome.setChecked(True)
    assert win.metronome is not None
    assert win.page_live.spin_bpm.isEnabled()
    win.on_metronome_bpm_changed(120)
    assert win.metronome.bpm == 120
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.35:
        app.processEvents()
    beat_text = win.page_live.label_beat.text()
    assert _tr("ui beat count").split("{")[0] in beat_text
    win.page_live.chk_metronome.setChecked(False)
    assert win.metronome is None
    assert win.page_live.label_beat.text() == ""
    print("metronome ok", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
