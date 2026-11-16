"""Space on the FROZEN live roll auditions the CAPTURED note.

The live page's roll holds captured notes, distinct from the final
analysis. Space auditions the selected captured note from the take's own
audio and shows its readout on the live page; the shared stop path clears it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from v2m.core.audio_io import SR


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _live_window():
    """A window holding a live take: a final result, the take's signal and
    the CAPTURED live notes the frozen roll renders."""
    from v2m.core.liveseg import LiveNote
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    win.result = AnalysisResult(
        source="live",
        duration_s=1.5,
        notes=[Note(0.1, 0.4, 60, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win._live_signal = np.ones(int(1.5 * SR), dtype=np.float32)
    win._last_take_live_notes = [
        LiveNote(0.2, 0.5, 60),
        LiveNote(0.6, 0.9, 64),
    ]
    return win


def _press(widget, key, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    """A real QKeyEvent delivered to the widget's own key handler."""
    widget.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, modifiers))


def _audition_probe(monkeypatch, win) -> dict:
    """Capture what the audition route hands the playback session."""
    played: dict = {}
    monkeypatch.setattr(win.playback, "set_sources", lambda **kw: played.update(kw))
    monkeypatch.setattr(win.playback, "play", lambda mode: played.update(mode=mode) or True)
    return played


def test_space_on_the_live_roll_auditions_the_captured_note(qapp, monkeypatch):
    """Space slices the take's audio at the CAPTURED note, not the final one."""
    win = _live_window()
    try:
        roll = win.page_live.piano_live
        roll.set_selectable(True)  # the take has stopped: the roll is explorable
        roll.set_notes(win._last_take_live_notes)
        roll.set_selection([1])  # the captured E4: 0.6 s -> 0.9 s
        win.stack.setCurrentWidget(win.page_live)
        played = _audition_probe(monkeypatch, win)

        _press(roll, Qt.Key.Key_Space)

        assert played["mode"] == "voice"
        assert len(played["voice"]) == pytest.approx(int(0.3 * SR), abs=4)
        assert win._auditions.active_identity == ("captured", 1)
        assert not win.page_live.label_live_audition.isHidden()
        assert "E4" in win.page_live.label_live_audition.text()
    finally:
        win.close()


def test_a_second_space_and_the_stop_button_clear_the_readout(qapp, monkeypatch):
    """One stop path: Space toggles, the readout's stop action clears."""
    from v2m.core.i18n import tr

    win = _live_window()
    try:
        roll = win.page_live.piano_live
        roll.set_selectable(True)
        roll.set_notes(win._last_take_live_notes)
        roll.set_selection([0])
        _audition_probe(monkeypatch, win)

        _press(roll, Qt.Key.Key_Space)
        label = win.page_live.label_live_audition
        stop = win.page_live.btn_live_audition_stop
        assert not label.isHidden()
        assert not stop.isHidden()
        assert stop.text() == tr("ui piano stop audition")

        _press(roll, Qt.Key.Key_Space)  # the same note: stop

        assert win._auditions.active_identity is None
        assert label.isHidden() and stop.isHidden()

        _press(roll, Qt.Key.Key_Space)
        assert not label.isHidden()
        stop.click()  # the readout's own stop action shares the path
        assert win._auditions.active_identity is None
        assert label.isHidden() and stop.isHidden()
    finally:
        win.close()


def test_a_running_take_auditions_nothing(qapp, monkeypatch):
    """The roll is frozen out during a capture: Space picks and plays nothing."""
    win = _live_window()
    try:
        roll = win.page_live.piano_live
        roll.set_selectable(False)  # a running take is never picked
        roll.set_notes(win._last_take_live_notes)
        roll.set_selection([0])  # gated out: the selection stays empty
        played = _audition_probe(monkeypatch, win)

        _press(roll, Qt.Key.Key_Space)

        assert "mode" not in played
        assert win._auditions.active_identity is None
        assert roll._selection == []
    finally:
        win.close()


def test_the_live_readout_clears_when_playback_ends(qapp, monkeypatch):
    """The 33 ms tick is the clock: a stale readout is cleared on completion."""
    win = _live_window()
    try:
        roll = win.page_live.piano_live
        roll.set_selectable(True)
        roll.set_notes(win._last_take_live_notes)
        roll.set_selection([0])
        _audition_probe(monkeypatch, win)
        _press(roll, Qt.Key.Key_Space)
        assert not win.page_live.label_live_audition.isHidden()

        win._on_play_tick()  # the probe's play left the session stopped

        assert win._auditions.active_identity is None
        assert win.page_live.label_live_audition.isHidden()
    finally:
        win.close()
