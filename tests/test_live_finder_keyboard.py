"""Tab traversal on the live and finder pages — the results-page audit extended.

Every page offers its tasks to the keyboard: an explicit focus chain in
task order (`_focus_order` + `setTabOrder`, the shape
`tests/test_results_keyboard.py` pins for the results page), the widgets
reached by Tab round the page, and one deliberate non-stop: the live
mirror keyboard only draws what is sounding and has no action of its own.

The finder's piano is a real stop (arrows stage a key, Enter auditions it):
its route lives on `PianoView` itself, so the page only has to give the
widget focus.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QVBoxLayout, QWidget


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _traverse(qapp, page, limit: int = 100, extra=()) -> list:
    """Tab round the page from its first stop: the chain's reached widgets."""
    stops = (*page._focus_order, *extra)
    first = page._focus_order[0]
    first.setFocus()
    qapp.processEvents()
    assert qapp.focusWidget() is first, "the chain's first stop must take focus"
    traversed = []
    for _ in range(limit):
        current = qapp.focusWidget()
        if current is first and traversed:
            break
        if current in stops and current not in traversed:
            traversed.append(current)
        page.focusNextPrevChild(True)
        qapp.processEvents()
    return traversed


def _missing(traversed, required: dict) -> list[str]:
    return [name for name, widget in required.items() if widget not in traversed]


# --------------------------------------------------------------------------- #
# The live page: control the take, set what it records, read the notes
# --------------------------------------------------------------------------- #


def test_the_live_pages_tab_chain_reaches_every_take_action(qapp):
    """Tab walks the take's actions in task order, hidden ones skipped."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.show()
        win.stack.setCurrentWidget(win.page_live)
        win.activateWindow()
        page = win.page_live
        assert page.btn_review_back.isHidden()  # review-only, Qt skips it

        assert page.details_panel.isHidden()
        traversed = _traverse(qapp, page)
        assert page.details_toggle in traversed
        assert page.chk_metronome not in traversed
        # the roll that draws the take belongs to the SINGING view: it is
        # reachable by Tab without unfolding anything
        assert page.piano_live in traversed
        page.details_toggle.setFocus()
        QTest.keyClick(page.details_toggle, Qt.Key.Key_Space)
        qapp.processEvents()
        assert not page.details_panel.isHidden()
        traversed = _traverse(qapp, page)
        required = {
            "stop take": page.btn_stop_live,
            "metronome": page.chk_metronome,
            "bpm": page.spin_bpm,
            "record audio": page.chk_record_audio,
            "captured roll": page.piano_live,
        }
        missing = _missing(traversed, required)
        visited = [type(w).__name__ for w in traversed]
        assert not missing, f"Tab skipped {missing}; visited {visited}"

        # reviewing the frozen roll makes the way back a Tab stop
        page.enter_review()
        assert not page.btn_review_back.isHidden()
        traversed = _traverse(qapp, page)
        assert page.btn_review_back in traversed, "the way back must be reachable"
        assert page.btn_stop_live in traversed
    finally:
        win.close()


@pytest.mark.parametrize("language", ["fr", "en"])
def test_live_details_keep_values_and_translation_when_collapsed(qapp, language):
    from v2m.core.i18n import set_language, tr
    from v2m.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.show()
        page = win.page_live
        win.stack.setCurrentWidget(page)
        page.details_toggle.click()
        page.spin_bpm.setValue(123)
        page.chk_record_audio.setChecked(False)
        page.details_toggle.click()
        set_language(language)
        page.retranslate()
        assert page.details_panel.isHidden()
        assert not page.details_toggle.isChecked()
        assert page.details_toggle.text() == tr("ui show live details")
        assert page.details_toggle.accessibleName() == tr("ui live details")
        page.details_toggle.click()
        assert page.spin_bpm.value() == 123
        assert not page.chk_record_audio.isChecked()
        assert page.box_monitor.y() < page.piano_live.y() < page.details_toggle.y()
        assert page.details_panel.y() > page.details_toggle.y()
        # the roll is part of the singing view, not a child of the disclosure
        assert page.piano_live.parentWidget() is page.scroll.widget()
        assert not page.piano_live.isAncestorOf(page.details_panel)
        page._set_input_state("ui mic waiting")
        assert page.live_indicator.label_input_state.text() == tr("ui mic waiting")
        assert page.details_toggle.text() == tr("ui hide live details")

        page.begin_take()
        page.on_block(0.0, 0.02)
        assert page.live_indicator.label_input_state.text() == tr("ui mic received")
        page.on_block(0.01, 0.0)
        page.on_block(2.1, 0.0)
        assert page.live_indicator.label_input_state.text() == tr("ui mic silence").format(s=2)
        page.on_block(2.2, 0.02)
        assert page.live_indicator.label_input_state.text() == tr("ui mic received")
    finally:
        win.close()
        set_language("fr")


def test_the_live_mirror_keyboard_is_not_a_tab_stop(qapp):
    """The mirror keyboard only draws what is sounding: a Tab stop with no
    action would be a place the keyboard gets stuck."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.show()
        win.stack.setCurrentWidget(win.page_live)
        page = win.page_live
        assert page.piano_live_keys.focusPolicy() == Qt.FocusPolicy.NoFocus
        assert page.piano_live_keys not in page._focus_order
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# The finder: leave, listen, read the history, audition a key
# --------------------------------------------------------------------------- #


def test_the_finders_tab_chain_reaches_every_action(qapp):
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.show()
        win.stack.setCurrentWidget(win.page_finder)
        win.activateWindow()
        page = win.page_finder
        # one confirmed note: the history is a row of chips, and Tab must
        # reach the chip as well as the page's fixed actions
        page._on_note((60.0, 261.6), confirmed=(60.0, 261.6), heard=(60.0, 261.6))
        chips = page.history_chips
        assert len(chips) == 1

        traversed = _traverse(qapp, page, extra=chips)
        required = {
            "back home": page.btn_back,
            "listen": page.btn_toggle,
            "history chip": chips[0],
            "piano": page.keys,
        }
        missing = _missing(traversed, required)
        visited = [type(w).__name__ for w in traversed]
        assert not missing, f"Tab skipped {missing}; visited {visited}"
    finally:
        win.close()


def test_the_finders_piano_auditions_from_the_keyboard(qapp):
    """Tab reaches the piano, an arrow stages a key, Enter plays it, and
    the staged key is dropped when focus leaves the keyboard."""
    from v2m.gui.finder_page import FinderPage

    played: list[int] = []

    def _no_capture():
        """This test never listens: reaching for the mic is a failure."""
        raise AssertionError("the keyboard route must not open the mic")

    page = FinderPage(
        thread_factory=_no_capture,
        go_home=lambda: None,
        show_error=lambda _msg: None,
        clear_error=lambda: None,
        audition_note=played.append,
    )
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(page)
    host.show()
    try:
        page.keys.setFocus()
        qapp.processEvents()
        assert qapp.focusWidget() is page.keys

        QTest.keyClick(page.keys, Qt.Key.Key_Right)  # nothing staged: from middle C
        assert page.keys.keyboard_cursor == 61
        QTest.keyClick(page.keys, Qt.Key.Key_Return)
        QTest.keyClick(page.keys, Qt.Key.Key_Right)
        QTest.keyClick(page.keys, Qt.Key.Key_Enter)
        assert played == [61, 62]

        for _ in range(80):  # the arrows stay inside the rendered keyboard
            QTest.keyClick(page.keys, Qt.Key.Key_Left)
        assert page.keys.keyboard_cursor == page.keys.min_midi

        page.btn_back.setFocus()
        qapp.processEvents()
        assert page.keys.keyboard_cursor is None
    finally:
        host.close()
