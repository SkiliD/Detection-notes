"""The results page's keyboard: ONE selection, driven from table and roll.

The notes TABLE owns the selection and the roll renders it. These tests pin
that the table's own keys (Up/Down/Home/End, Shift to extend) move the SAME
selection the roll shows, that Space auditions the cursor note's own segment,
and that every edit move (undo, redo, jump) acknowledges itself on the status
bar through the ONE shared route.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest

from v2m.core.audio_io import SR


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_real_audio(monkeypatch):
    """Never open a real output stream: an audition (Space, or choosing a
    pitch on the piano before Enter) must not touch the sound device here.
    Tests that need the playing state install their own `play` probe, which
    overrides this instance attribute."""
    from v2m.core.playback import PlaybackSession

    monkeypatch.setattr(
        PlaybackSession, "play", lambda self, mode="piano", t=0.0: True
    )


def _result(source: str = "song.wav"):
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult

    return AnalysisResult(
        source=source,
        duration_s=1.5,
        notes=[
            Note(0.1, 0.4, 60, 0.9),
            Note(0.5, 0.8, 64, 0.9),
            Note(0.9, 1.2, 67, 0.9),
        ],
        tempo=None,
        key=None,
        pitch=None,
    )


def _window():
    """A real window showing a three-note file take (audio cached for the roll)."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = _result()
    win.result = result
    win._signal_cache = {result.source: np.ones(2 * SR, dtype=np.float32)}
    win._show_result(result)
    return win


def _press(widget, key, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    """A real QKeyEvent delivered to the widget's own key handler."""
    widget.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, modifiers))


# --------------------------------------------------------------------------- #
# The table's keys move the selection the roll renders
# --------------------------------------------------------------------------- #


def test_the_table_keys_move_the_one_selection_the_roll_shows(qapp):
    """Up/Down and Home/End step the table's cursor; the roll always renders
    that same selection (one owner, two views)."""
    win = _window()
    try:
        table = win.page_results.results._notes_table
        roll = win.page_results.piano
        table.setCurrentCell(0, 0)

        _press(table, Qt.Key.Key_Down)
        assert table.currentRow() == 1
        assert roll._selection == [1], "the roll must render the table's selection"

        _press(table, Qt.Key.Key_End)
        assert roll._selection == [2]
        _press(table, Qt.Key.Key_Down)  # the last row holds
        assert table.currentRow() == 2

        _press(table, Qt.Key.Key_Home)
        assert roll._selection == [0]
        _press(table, Qt.Key.Key_Up)  # the first row holds
        assert table.currentRow() == 0
    finally:
        win.close()


def test_shift_extends_the_table_selection_as_one_range(qapp):
    """Shift+arrow builds the same contiguous range a drag on the roll does."""
    win = _window()
    try:
        table = win.page_results.results._notes_table
        roll = win.page_results.piano
        table.setCurrentCell(0, 0)

        _press(table, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
        _press(table, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)

        assert sorted({i.row() for i in table.selectedIndexes()}) == [0, 1, 2]
        assert roll._selection == [0, 1, 2]
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# Space auditions the cursor note (table and roll, one entry)
# --------------------------------------------------------------------------- #


def _audition_probe(monkeypatch, win) -> dict:
    """Capture what the audition route hands the playback session."""
    played: dict = {}
    monkeypatch.setattr(win.playback, "set_sources", lambda **kw: played.update(kw))
    monkeypatch.setattr(win.playback, "play", lambda mode: played.update(mode=mode) or True)
    return played


def test_space_on_the_table_auditions_that_note(qapp, monkeypatch):
    win = _window()
    try:
        table = win.page_results.results._notes_table
        played = _audition_probe(monkeypatch, win)

        table.setCurrentCell(1, 0)  # the second note: 0.5 s -> 0.8 s
        _press(table, Qt.Key.Key_Space)

        assert played["mode"] == "voice"
        assert len(played["voice"]) == pytest.approx(int(0.3 * SR), abs=4)
    finally:
        win.close()


def test_space_audition_shows_the_note_and_stop_button(qapp, monkeypatch):
    """Keyboard audition is visibly identified and a second Space stops it."""
    from v2m.core.i18n import tr

    win = _window()
    try:
        table = win.page_results.results._notes_table
        played = _audition_probe(monkeypatch, win)
        table.setCurrentCell(1, 0)

        _press(table, Qt.Key.Key_Space)
        label = win.page_results.label_note_audition
        stop = win.page_results.btn_stop_audition
        assert not label.isHidden()
        assert "E4" in label.text()
        assert not stop.isHidden() and stop.text() == tr("ui piano stop audition")

        _press(table, Qt.Key.Key_Space)
        assert played["mode"] == "voice"
        assert label.isHidden() and stop.isHidden()
    finally:
        win.close()


def test_space_on_the_roll_auditions_its_cursor_note(qapp, monkeypatch):
    win = _window()
    try:
        roll = win.page_results.piano
        played = _audition_probe(monkeypatch, win)

        roll.set_selection([2])  # the last note: 0.9 s -> 1.2 s
        _press(roll, Qt.Key.Key_Space)

        assert played["mode"] == "voice"
        assert len(played["voice"]) == pytest.approx(int(0.3 * SR), abs=4)
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# Every edit move acknowledges itself through the one route
# --------------------------------------------------------------------------- #


def test_focus_order_reaches_keyboard_actions_and_export(qapp):
    """The live results-page focus chain exposes all note tasks by Tab."""
    from v2m.core.i18n import tr

    win = _window()
    try:
        page = win.page_results
        table = page.results._notes_table
        page.set_result(_result("live"), np.ones(2 * SR, dtype=np.float32))
        page.chk_live_comparison.setChecked(False)
        page.chk_live_comparison.setChecked(True)
        assert not page.chk_live_comparison.isHidden()
        page.show_note_audition(0, 60, 3)
        assert table.focusPolicy() != Qt.FocusPolicy.NoFocus
        assert page.piano.focusPolicy() != Qt.FocusPolicy.NoFocus
        assert page.piano_keys.focusPolicy() != Qt.FocusPolicy.NoFocus
        assert page.btn_stop_audition.focusPolicy() != Qt.FocusPolicy.NoFocus

        win.show()
        win.stack.setCurrentWidget(page)
        win.activateWindow()
        first = page._focus_order[0]
        first.setFocus()
        traversed = []
        for _ in range(100):
            current = qapp.focusWidget()
            if current is first and traversed:
                break
            if current in page._focus_order and current not in traversed:
                traversed.append(current)
            if current is page.details_toggle:
                page.details_toggle.click()  # reveal the controls before tabbing into them
            page.focusNextPrevChild(True)
            qapp.processEvents()

        required_widgets = {
            "comparison toggle": page.chk_live_comparison,
            "show details": page.details_toggle,
            "notes table": table,
            "piano roll": page.piano,
            "piano keyboard": page.piano_keys,
            "export": page.btn_export,
        }
        missing = [name for name, widget in required_widgets.items() if widget not in traversed]
        visited = [type(w).__name__ for w in traversed]
        assert not missing, f"Tab skipped {missing}; visited {visited}"
        assert page.chk_live_comparison.text() == tr("ui live comparison")
    finally:
        win.close()


def test_audition_stop_button_works_when_keyboard_focused(qapp, monkeypatch):
    """The stop action is reachable by Tab while the readout is shown."""
    win = _window()
    try:
        played = _audition_probe(monkeypatch, win)
        page = win.page_results
        table = page.results._notes_table
        table.setCurrentCell(0, 0)
        win.show()
        win.stack.setCurrentWidget(page)
        qapp.processEvents()
        _press(table, Qt.Key.Key_Space)
        assert not page.btn_stop_audition.isHidden()

        page.btn_both.setFocus()
        qapp.processEvents()
        QTest.keyClick(page.btn_both, Qt.Key.Key_Tab)
        assert qapp.focusWidget() is page.btn_stop_audition
        QTest.keyClick(page.btn_stop_audition, Qt.Key.Key_Space)

        assert played["mode"] == "voice"
        assert page.btn_stop_audition.isHidden()
    finally:
        win.close()


def test_file_results_hide_live_comparison_and_disable_overlay(qapp):
    win = _window()
    try:
        page = win.page_results
        page.chk_live_comparison.setChecked(False)
        win.page_live.piano_live.set_notes(_result().notes)
        final = [type(_result().notes[0])(0.2, 0.4, 62, 0.9)]
        win.page_live.set_comparison_notes(final)
        assert win.page_live.piano_live._compare_items == []

        assert page.chk_live_comparison.isHidden()
        assert win.page_live._comparison_visible is False
        page.set_result(_result("live"))
        assert not page.chk_live_comparison.isHidden()
        assert win.page_live._comparison_visible is True
    finally:
        win.close()


def test_live_comparison_checkbox_toggles_the_captured_roll_overlay(qapp):
    """The visible checkbox controls only the final overlay, preserving the capture."""
    from v2m.core.notes import Note

    win = _window()
    try:
        page = win.page_results
        page.set_result(_result("live"))
        roll = win.page_live.piano_live
        captured = _result().notes
        final = [Note(0.2, 0.4, 62, 0.9), Note(0.5, 0.8, 64, 0.9)]
        roll.set_notes(captured)
        win.page_live.set_comparison_notes(final, moved={0})

        page.chk_live_comparison.setChecked(False)
        assert len(roll._bar_items) == len(captured)
        assert roll._compare_items == []

        page.chk_live_comparison.setChecked(True)
        assert len(roll._bar_items) == len(captured)
        assert len(roll._compare_items) == len(final)
    finally:
        win.close()


def test_keyboard_focus_then_arrows_and_enter_correct_the_selected_note(qapp):
    """Tab focus reaches the piano, arrows move pitch, Enter applies it."""
    win = _window()
    try:
        page = win.page_results
        table = page.results._notes_table
        table.setCurrentCell(0, 0)
        win.show()
        win.stack.setCurrentWidget(page)
        page.piano_keys.setFocus()
        qapp.processEvents()
        assert qapp.focusWidget() is page.piano_keys

        QTest.keyClick(page.piano_keys, Qt.Key.Key_Right)
        QTest.keyClick(page.piano_keys, Qt.Key.Key_Right)
        assert page.piano_keys.keyboard_cursor == 62
        QTest.keyClick(page.piano_keys, Qt.Key.Key_Return)

        assert win.result is not None
        assert win.result.notes[0].midi == 62
        assert win.result.notes[1].midi == 64
    finally:
        win.close()


def test_piano_keyboard_escape_cancels_without_editing(qapp):
    win = _window()
    try:
        page = win.page_results
        page.results._notes_table.setCurrentCell(0, 0)
        win.show()
        win.stack.setCurrentWidget(page)
        page.piano_keys.setFocus()
        qapp.processEvents()
        QTest.keyClick(page.piano_keys, Qt.Key.Key_Right)
        QTest.keyClick(page.piano_keys, Qt.Key.Key_Escape)
        assert win.result is not None
        assert win.result.notes[0].midi == 60
        assert page.piano_keys.keyboard_cursor is None
    finally:
        win.close()


def test_choosing_a_pitch_plays_it_before_enter(qapp, monkeypatch):
    """Arrows stage a key AND sound it: the candidate is heard BEFORE Enter,
    and validating does not replay it a second time."""
    win = _window()
    try:
        page = win.page_results
        page.results._notes_table.setCurrentCell(0, 0)  # the first note is C4
        played: list[int] = []
        monkeypatch.setattr(page, "_audition_pitch", played.append)
        win.show()
        win.stack.setCurrentWidget(page)
        page.piano_keys.setFocus()
        qapp.processEvents()

        QTest.keyClick(page.piano_keys, Qt.Key.Key_Right)  # C4 -> C#4, heard
        assert played == [61]
        QTest.keyClick(page.piano_keys, Qt.Key.Key_Right)  # C#4 -> D4
        assert played == [61, 62]

        QTest.keyClick(page.piano_keys, Qt.Key.Key_Return)  # validate: no replay
        assert played == [61, 62]
        assert win.result is not None and win.result.notes[0].midi == 62
    finally:
        win.close()


def test_staging_a_pitch_really_plays_a_tone(qapp, monkeypatch):
    """The preview goes through the window's ONE one-tone path (a piano
    render handed to the shared playback session)."""
    win = _window()
    try:
        page = win.page_results
        page.results._notes_table.setCurrentCell(0, 0)
        played: dict = {}
        monkeypatch.setattr(win.playback, "set_sources", lambda **kw: played.update(kw))
        monkeypatch.setattr(win.playback, "play", lambda mode: played.update(mode=mode) or True)
        win.show()
        win.stack.setCurrentWidget(page)
        page.piano_keys.setFocus()
        qapp.processEvents()

        QTest.keyClick(page.piano_keys, Qt.Key.Key_Right)

        assert played.get("mode") == "piano"
        assert len(played["piano"]) > 0
    finally:
        win.close()


def test_the_pitch_preview_is_the_windows_one_tone_route(qapp):
    """The page is wired to the window's shared key-advice callback."""
    win = _window()
    try:
        page = win.page_results
        assert page._audition_pitch.__func__ is win.on_audition_key.__func__
    finally:
        win.close()


def test_arrows_stay_on_the_rendered_keyboard(qapp):
    """Arrows step inside the keyboard's own range: the staged pitch is
    always a key the user can see (the mouse cannot point outside it)."""
    win = _window()
    try:
        page = win.page_results
        win.show()
        win.stack.setCurrentWidget(page)
        page.results._notes_table.setCurrentCell(0, 0)
        page.piano_keys.setFocus()
        qapp.processEvents()

        for _ in range(40):
            QTest.keyClick(page.piano_keys, Qt.Key.Key_Right)
        assert page.piano_keys.keyboard_cursor == page.piano_keys.max_midi

        for _ in range(60):
            QTest.keyClick(page.piano_keys, Qt.Key.Key_Left)
        assert page.piano_keys.keyboard_cursor == page.piano_keys.min_midi
    finally:
        win.close()


def test_leaving_the_keyboard_clears_the_staged_key(qapp):
    """The staged pitch belongs to the focused keyboard: once focus
    leaves, Enter elsewhere can never apply a stale correction."""
    win = _window()
    try:
        page = win.page_results
        win.show()
        win.stack.setCurrentWidget(page)
        page.results._notes_table.setCurrentCell(0, 0)
        page.piano_keys.setFocus()
        qapp.processEvents()
        assert page.piano_keys.keyboard_cursor == 60  # staged on focus

        page.results._notes_table.setFocus()
        qapp.processEvents()

        assert page.piano_keys.keyboard_cursor is None
    finally:
        win.close()


def test_a_note_outside_the_keyboard_is_not_staged(qapp):
    """A pitch this keyboard does not render cannot be staged: Enter
    stays inert; arrows recover from middle C instead."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult

    win = _window()
    try:
        page = win.page_results
        win.show()
        win.stack.setCurrentWidget(page)
        page.set_result(
            AnalysisResult(
                source="song.wav",
                duration_s=0.5,
                notes=[Note(0.1, 0.4, 96, 0.9)],  # above the keyboard's top
                tempo=None,
                key=None,
                pitch=None,
            )
        )
        page.results._notes_table.setCurrentCell(0, 0)
        page.piano_keys.setFocus()
        qapp.processEvents()

        assert page.piano_keys.keyboard_cursor is None
        QTest.keyClick(page.piano_keys, Qt.Key.Key_Right)
        assert page.piano_keys.keyboard_cursor == 61
    finally:
        win.close()


def test_space_stop_button_clears_the_audition_readout(qapp, monkeypatch):
    """The visible stop action and Space use one stop path."""
    win = _window()
    try:
        played = _audition_probe(monkeypatch, win)
        win.page_results.results._notes_table.setCurrentCell(0, 0)
        _press(win.page_results.results._notes_table, Qt.Key.Key_Space)
        assert not win.page_results.label_note_audition.isHidden()

        win.page_results.btn_stop_audition.click()

        assert played["mode"] == "voice"
        assert win.page_results.label_note_audition.isHidden()
        assert win.page_results.btn_stop_audition.isHidden()
    finally:
        win.close()


def test_audition_playback_completion_hides_readout(qapp, monkeypatch):
    """The 33 ms playback tick clears a stale label if the device ends early."""
    win = _window()
    try:
        _audition_probe(monkeypatch, win)
        win.page_results.results._notes_table.setCurrentCell(0, 0)
        _press(win.page_results.results._notes_table, Qt.Key.Key_Space)
        assert not win.page_results.label_note_audition.isHidden()

        monkeypatch.setattr(win.playback, "_t0", None)
        win._on_play_tick()

        assert win.page_results.label_note_audition.isHidden()
        assert win.page_results.btn_stop_audition.isHidden()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# The audition readout's lifetime IS the playback's (one clock, one clear)
# --------------------------------------------------------------------------- #

def _sounding_probe(monkeypatch, win) -> dict:
    """`play` behaves like the real session (mode + playing) without audio."""
    played: dict = {}

    def play(mode, t=0.0):
        played.update(mode=mode, t=t)
        win.playback.mode = mode
        win.playback._t0 = time.perf_counter() - t
        return True

    monkeypatch.setattr(win.playback, "play", play)
    return played


def test_the_audition_readout_outlives_the_clip_length(qapp, monkeypatch):
    """One clock: no length estimate — the readout survives past the
    slice's nominal end while the session still sounds."""
    win = _window()
    try:
        _sounding_probe(monkeypatch, win)
        table = win.page_results.results._notes_table
        table.setCurrentCell(1, 0)  # a 0.3 s slice, loaded for real
        _press(table, Qt.Key.Key_Space)
        label = win.page_results.label_note_audition
        assert not label.isHidden()

        deadline = time.monotonic() + 0.45  # past the slice's nominal end
        while time.monotonic() < deadline:
            qapp.processEvents()
            assert not label.isHidden(), "the readout must outlive the slice"
    finally:
        win.close()


def test_a_seek_after_an_audition_reads_the_take_not_the_slice(qapp, monkeypatch):
    """An audition leaves its short slice in the session; a double-click
    still plays from the clicked position, the take's source preserved."""
    from v2m.core.playback import NotePlayer

    win = _window()
    try:
        played = _sounding_probe(monkeypatch, win)
        assert win.result is not None
        win.player = NotePlayer(win.result.notes, sr=SR)
        table = win.page_results.results._notes_table
        table.setCurrentCell(1, 0)
        _press(table, Qt.Key.Key_Space)  # real set_sources: the 0.3 s slice
        assert played["mode"] == "voice"

        win.on_seek(1.0)  # the take is 1.5 s

        assert win.page_results.label_note_audition.isHidden()
        assert win._auditions.active_identity is None
        assert played["mode"] == "voice"  # the source the user was hearing
        assert played["t"] == pytest.approx(1.0)
    finally:
        win.close()


def test_a_seek_at_the_take_end_resolves_sounding_audio(qapp, monkeypatch):
    """The end starts nothing — and leaves nothing sounding or shown."""
    from v2m.core.playback import NotePlayer

    win = _window()
    try:
        played = _sounding_probe(monkeypatch, win)
        assert win.result is not None
        win.player = NotePlayer(win.result.notes, sr=SR)
        table = win.page_results.results._notes_table
        table.setCurrentCell(0, 0)
        _press(table, Qt.Key.Key_Space)
        assert not win.page_results.label_note_audition.isHidden()

        win.on_seek(1.48)  # the take is 1.5 s: nothing to play from here

        assert win.page_results.label_note_audition.isHidden()
        assert win.page_results.btn_stop_audition.isHidden()
        assert win._auditions.active_identity is None
        assert not win.playback.playing
        assert played["t"] == 0.0  # the audition's own start, no restart
    finally:
        win.close()


def test_a_failed_audition_shows_no_readout(qapp, monkeypatch):
    """Nothing sounds → nothing shows: the one clear path runs either way."""
    win = _window()
    try:
        monkeypatch.setattr(win.playback, "play", lambda mode, t=0.0: False)
        table = win.page_results.results._notes_table
        table.setCurrentCell(0, 0)

        _press(table, Qt.Key.Key_Space)

        assert win.page_results.label_note_audition.isHidden()
        assert win.page_results.btn_stop_audition.isHidden()
        assert win._auditions.active_identity is None
    finally:
        win.close()


def test_tab_skips_the_stop_action_while_the_readout_is_hidden(qapp):
    """The stop action lives in the ONE static focus order; Qt skips it
    while hidden, so the chain needs no relinking when the readout ends."""
    win = _window()
    try:
        page = win.page_results
        win.show()
        win.stack.setCurrentWidget(page)
        qapp.processEvents()
        assert page.btn_stop_audition.isHidden()

        page.btn_both.setFocus()
        qapp.processEvents()
        QTest.keyClick(page.btn_both, Qt.Key.Key_Tab)

        assert qapp.focusWidget() is page.slider_volume
    finally:
        win.close()


def test_every_edit_move_acknowledges_itself(qapp):
    """A correction, an undo, a redo and a state jump all report through
    `_ack_edit`; a jump names itself rather than reading as an undo."""
    from v2m.core.i18n import tr

    win = _window()
    try:
        table = win.page_results.results._notes_table
        table.setCurrentCell(0, 0)
        win.page_results.piano_keys.key_clicked.emit(62)
        table.setCurrentCell(1, 0)
        win.page_results.piano_keys.key_clicked.emit(65)

        win.on_undo_edit()
        assert win.statusBar().currentMessage() == tr("ui edit undone")

        win.on_redo_edit()
        assert win.statusBar().currentMessage() == tr("ui edit redone")

        # a jump to an earlier row is its own acknowledgement
        assert win.page_results._edit_states
        win.on_jump_edit_state(0)
        assert win.statusBar().currentMessage() == tr("ui edit jumped")
    finally:
        win.close()
