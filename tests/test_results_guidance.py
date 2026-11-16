"""One optional next action, backed by the take's existing evidence."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from v2m.core.audio_io import SR
from v2m.core.continuity import coherence_report
from v2m.core.i18n import set_language, tr
from v2m.core.notes import Note
from v2m.core.observation import VoiceObservation
from v2m.core.pipeline import AnalysisResult
from v2m.core.pitch import PitchTrack


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def result(suspect=False):
    times = np.arange(100) * 0.02
    midis = np.full(100, 60.0)
    if suspect:
        midis[30:50] = 64.0
    pitch = PitchTrack(times, 440 * 2 ** ((midis - 69) / 12), np.ones(100))
    return AnalysisResult(
        source="guide.wav",
        duration_s=2.0,
        notes=[Note(0.0, 0.5, 60, 0.9), Note(0.5, 1.5, 60, 0.9)],
        tempo=None,
        key=None,
        pitch=pitch,
        observation=VoiceObservation(
            pitch=pitch,
            attacks=None,
            duration_s=2,
            sample_rate=SR,
            frame_dt=0.02,
        ),
    )


@pytest.fixture
def win(qapp):
    from v2m.gui.main_window import MainWindow

    window = MainWindow()
    window.result = result(True)
    window._signal_cache = {"guide.wav": np.ones(2 * SR, dtype=np.float32)}
    window._show_result(window.result)
    window.show()
    qapp.processEvents()
    yield window
    window.close()


def test_initial_suggestion_uses_the_first_coherence_doubt(win):
    page = win.page_results
    assert page.btn_guide.text() == tr("ui guide listen passage")
    assert page._guide_target == [1]
    # the piano reading is not a tool to unfold: it is offered as soon as a
    # result is shown, and the guide only points at WHERE to listen
    assert not page.group_listen.isHidden()
    assert page.group_adjust.isHidden()
    assert page.group_export.isHidden()
    assert not page.piano.isHidden()
    assert not page.details_toggle.isHidden()
    page.btn_guide.setFocus()
    QTest.keyClick(page.btn_guide, Qt.Key.Key_Tab)
    assert page.btn_guide_skip.hasFocus()
    QTest.keyClick(page.btn_guide_skip, Qt.Key.Key_Tab)
    assert page.tools_toggle.hasFocus()


def test_audition_plays_only_the_target_and_advances_on_stop(win, monkeypatch):
    page = win.page_results
    played = []
    monkeypatch.setattr(win.playback, "play", lambda mode="piano", t=0: played.append(mode) or True)
    page.btn_guide.click()
    assert played == ["voice"]
    assert page._selected_note_indices() == [1]
    assert page._guide_listening
    assert len(win.playback.voice) == SR
    win.stop_playback()
    assert page.btn_guide.text() == tr("ui guide review")
    page.btn_guide.click()
    assert not page.tools_toggle.isChecked()
    assert not page.group_adjust.isHidden()
    assert page.group_export.isHidden()
    assert page._selected_note_indices() == [1]
    assert page.btn_guide.text() == tr("ui export")


def test_note_audition_temporarily_reveals_controls_and_survives_retranslation(win):
    page = win.page_results
    page.show_note_audition(1, 60, 2)
    page.retranslate()
    assert not page.group_listen.isHidden()
    assert not page.btn_stop_audition.isHidden()
    page.stop_note_audition()
    assert not page.group_listen.isHidden()  # the listening group stays put
    assert page.btn_stop_audition.isHidden()
    assert page.btn_guide.text() == tr("ui guide listen passage")


def test_completed_phrase_advances_without_an_explicit_stop(win, monkeypatch):
    page = win.page_results
    monkeypatch.setattr(win.playback, "play", lambda mode="piano", t=0: True)
    page.btn_guide.click()
    assert page._guide_listening
    win._on_play_tick()
    assert not page._guide_listening
    assert page.btn_guide.text() == tr("ui guide review")
    assert not page.group_listen.isHidden()


def test_failed_audio_start_does_not_advance(win, monkeypatch):
    page = win.page_results
    monkeypatch.setattr(win.playback, "play", lambda mode="piano", t=0: False)
    page.btn_guide.click()
    assert page.btn_guide.text() == tr("ui guide listen passage")
    assert not page._guide_listening


def test_skip_exports_without_forcing_listening_or_an_edit(win, monkeypatch):
    page = win.page_results
    exported = []
    page.btn_export.clicked.disconnect()
    page.btn_export.clicked.connect(lambda: exported.append(True))
    QTest.mouseClick(page.btn_guide_skip, Qt.MouseButton.LeftButton)
    page.btn_guide.click()
    assert exported == [True]
    assert page._selected_note_indices() == []


def test_clean_missing_audio_and_new_take_reset(win):
    page = win.page_results
    page.btn_guide_skip.click()
    page.tools_toggle.click()
    page.set_result(result(), np.ones(2 * SR, dtype=np.float32))
    assert page.btn_guide.text() == tr("ui guide listen take")
    assert not page.tools_toggle.isChecked()
    page.set_result(result(), None)
    assert page.btn_guide.text() == tr("ui export")
    assert page.btn_guide.isEnabled()


def test_language_switch_and_busy_decision_preserve_facts(win):
    page = win.page_results
    try:
        set_language("en")
        page.retranslate()
        assert page.btn_guide.text() == "Listen to this passage"
        assert page._guide_target == [1]
        page.set_replacement_pending(True)
        assert page.guide_card.isHidden()
        assert not page.group_adjust.isHidden()
        assert not page.btn_accept_replacement.isHidden()
        assert not page.btn_export.isEnabled()
        page.set_replacement_pending(False)
        assert not page.guide_card.isHidden()
    finally:
        set_language("fr")
        page.retranslate()


def multiple_passages():
    reading = result(True)
    reading.notes = [Note(0, 0.95, 60, 0.9), Note(1, 1.95, 60, 0.9)]
    reading.pitch.hz[10:25] = 440 * 2 ** ((64 - 69) / 12)
    reading.pitch.hz[65:80] = 440 * 2 ** ((65 - 69) / 12)
    return reading


def test_passages_are_reviewed_in_order_with_explicit_advancement(win, monkeypatch):
    page = win.page_results
    win.result = multiple_passages()
    win._show_result(win.result)
    monkeypatch.setattr(win.playback, "play", lambda mode="piano", t=0: True)
    assert page._guide_targets == [[0], [1]]
    assert tr("ui guide progress").format(current=1, total=2) in page.label_guide.text()
    for index in (0, 1):
        assert page.btn_guide.text() == tr("ui guide listen passage")
        page.btn_guide.click()
        assert page._selected_note_indices() == [index]
        win.stop_playback()
        page.btn_guide.click()
        assert page._guide_index == index
        assert not page.group_adjust.isHidden()
        if index == 0:
            assert page.btn_guide.text() == tr("ui guide next")
            page.retranslate()
            assert page._guide_index == 0
            page.btn_guide.click()
            assert page._guide_index == 1
            assert page._selected_note_indices() == [1]
            assert page.group_adjust.isHidden()
    assert page.btn_guide.text() == tr("ui export")
    assert page.btn_guide_skip.isHidden()


def test_new_reading_rebuilds_passages_without_old_indices(win):
    page = win.page_results
    page.set_result(multiple_passages(), np.ones(2 * SR, dtype=np.float32))
    assert len(page._guide_targets) == 2
    updated = result()
    updated.notes = [Note(0, 1.95, 60, 0.9)]
    page.set_result(updated, np.ones(2 * SR, dtype=np.float32))
    assert page._guide_targets == []
    assert page._guide_target == []
    assert page._guide_index == 0
    assert page.btn_guide.text() == tr("ui guide listen take")


def test_correcting_a_passage_recomputes_the_remaining_doubts(win, monkeypatch):
    page = win.page_results
    win.result = multiple_passages()
    win._show_result(win.result)
    monkeypatch.setattr(win.playback, "play", lambda mode="piano", t=0: True)
    page.btn_guide.click()
    win.stop_playback()
    page.btn_guide.click()
    win.on_correct_note(0, 64)
    expected = [
        [i for i, note in enumerate(win.result.notes) if note.start == suspect["start_s"]]
        for suspect in coherence_report(win.result.observation, win.result.notes)["suspects"]
    ]
    assert page._guide_targets == expected
    assert page._guide_index == 0
    if expected:
        assert page.btn_guide.text() == tr("ui guide listen passage")
    else:
        assert page._guide_target == []
        assert page.btn_guide.text() == tr("ui guide listen take")


def test_skip_from_a_multiple_passage_review_offers_export(win, monkeypatch):
    page = win.page_results
    win.result = multiple_passages()
    win._show_result(win.result)
    monkeypatch.setattr(win.playback, "play", lambda mode="piano", t=0: True)
    page.btn_guide.click()
    page.btn_guide_skip.click()
    assert not page._guide_listening
    assert page.btn_guide.text() == tr("ui export")


def test_guidance_does_not_clear_the_reading_diff(win):
    from v2m.core.reading_diff import compare_readings

    page = win.page_results
    comparison = compare_readings(result(), result(True))
    page.set_reading_diff(comparison)
    page.btn_guide_skip.click()
    page.tools_toggle.click()
    assert page._reading_diff is comparison
    assert not page.label_diff_legend.isHidden()
