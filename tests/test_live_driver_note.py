"""The live page names a driver's effects chain for the endpoint it OPENED.

A raw (exclusive) open is the way past the chain, so a raw take says nothing.
A shared take on an endpoint that publishes a chain is exactly the case the
singer cannot see from the audio. The page stores the FLAG and names it by
key, so a language switch re-renders the same fact.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core import audio_io, driver_fx
from v2m.core.i18n import set_language, tr
from v2m.gui import main_window as mw

MIC = "Microphone (Realtek(R) Audio)"


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "THEME_FILE", tmp_path / "theme")
    win = mw.MainWindow()
    yield win
    win.close()
    set_language("fr")


def test_a_shared_capture_names_the_driver_effects(window, monkeypatch):
    seen: list[str] = []

    def active(name: str) -> bool:
        seen.append(name)
        return True

    monkeypatch.setattr(driver_fx, "driver_effects_active", active)
    window.live_thread = mw.LiveThread(device=3, device_name=MIC)
    window._on_live_capture_opened(False)
    label = window.page_live.label_driver_effects
    # the endpoint the capture OPENED, not the one selected for the next take
    assert seen == [MIC]
    assert label.text() == tr("ui input device effects")
    assert label.isHidden() is False


def test_a_raw_capture_says_nothing(window, monkeypatch):
    monkeypatch.setattr(driver_fx, "driver_effects_active", lambda name: True)
    window.live_thread = mw.LiveThread(device=3, device_name=MIC)
    window._on_live_capture_opened(False)
    window._on_live_capture_opened(True)  # the chain was bypassed
    label = window.page_live.label_driver_effects
    assert label.isHidden() is True
    assert label.text() == ""


def test_the_sentence_follows_the_language(window):
    set_language("fr")
    window.page_live.set_driver_effects(True)
    french = window.page_live.label_driver_effects.text()
    set_language("en")
    window.page_live.retranslate()
    english = window.page_live.label_driver_effects.text()
    assert french != english
    assert "améliorations audio" in french
    assert "audio enhancements" in english


def test_the_calibration_card_names_the_chain_of_its_own_capture(window, monkeypatch):
    # two parameters: the real one is called by the page's own note builder
    monkeypatch.setattr(driver_fx, "driver_effects_active", lambda name, endpoints=None: True)
    thread = mw.LiveThread(device=3, device_name=MIC)
    monkeypatch.setattr(thread, "start", lambda: None)  # no audio in a GUI test
    monkeypatch.setattr(window, "_new_input_thread", lambda: thread)
    window._start_calibration_capture()
    card = window.page_home.calibration_card
    assert card.label_driver_effects.isHidden() is True  # nothing opened yet
    thread.captureOpened.emit(False)  # the shared stream really opened
    assert card.label_driver_effects.text() == tr("ui input device effects")
    assert card.label_driver_effects.isHidden() is False
    thread.captureOpened.emit(True)  # a raw open bypassed the chain
    assert card.label_driver_effects.isHidden() is True
    window._stop_calibration_capture()
    assert card.label_driver_effects.isHidden() is True


def test_the_finder_names_the_chain_of_the_capture_it_opened(window, monkeypatch):
    monkeypatch.setattr(driver_fx, "driver_effects_active", lambda name, endpoints=None: True)
    thread = window._new_finder_input_thread()
    window.page_finder._thread = thread  # what toggle() does before start()
    thread.captureOpened.emit(False)
    label = window.page_finder.label_driver_effects
    assert label.text() == tr("ui input device effects")
    assert label.isHidden() is False
    # leaving the finder drops the capture AND its sentence
    window.page_finder._thread = None
    window.page_finder.set_driver_effects(False)
    assert label.isHidden() is True


def test_both_extra_notes_follow_the_language(window):
    set_language("fr")
    window.page_home.calibration_card.set_driver_effects(True)
    window.page_finder.set_driver_effects(True)
    french = (
        window.page_home.calibration_card.label_driver_effects.text(),
        window.page_finder.label_driver_effects.text(),
    )
    set_language("en")
    window.page_home.calibration_card.retranslate()
    window.page_finder.retranslate()
    english = (
        window.page_home.calibration_card.label_driver_effects.text(),
        window.page_finder.label_driver_effects.text(),
    )
    assert french[0] != english[0] and french[1] != english[1]
    assert "audio enhancements" in english[0] and "audio enhancements" in english[1]


def test_the_next_capture_asks_for_raw_only_when_effects_are_active(window, monkeypatch):
    monkeypatch.setattr(
        window,
        "_resolve_input_device",
        lambda: audio_io.ResolvedInputDevice(3, MIC),
    )
    monkeypatch.setattr(driver_fx, "driver_effects_active", lambda name: True)
    assert window._new_input_thread().exclusive is True

    monkeypatch.setattr(driver_fx, "driver_effects_active", lambda name: False)
    assert window._new_input_thread().exclusive is False
