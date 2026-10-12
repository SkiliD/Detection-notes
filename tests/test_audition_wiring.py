"""`MainWindow` only prepares clips and applies transition facts (the approved
spec): the one controller owns active identity, toggle/replace, and the drag
sequence. These tests pin the WIRING — every entry point submits a request
with the same buffer/mode as today, readouts follow the returned transitions,
the whole-take buttons cancel an active audition by starting their own mode,
and volume/transpose leave an in-flight audition untouched.

The playback session keeps its identity (the controller holds it): the probe
replaces the instance's `play`/`stop`/`set_sources` so the wiring is observed
end to end without a sound device, and completion is driven through the
window's own `_on_play_tick()` — the observable path, never an internal one.
"""

from __future__ import annotations

import os
import time
from types import MethodType

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


def _result(source: str = "song.wav"):
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult

    return AnalysisResult(
        source=source,
        duration_s=1.5,
        notes=[
            Note(0.1, 0.4, 60, 0.9),
            Note(0.5, 0.8, 64, 0.9),
        ],
        tempo=None,
        key=None,
        pitch=None,
    )


def _window():
    """A real window showing a two-note file take (audio cached)."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = _result()
    win.result = result
    win._signal_cache = {result.source: np.ones(2 * SR, dtype=np.float32)}
    win._show_result(result)
    win.stack.setCurrentWidget(win.page_results)
    return win


def _press_space(widget) -> None:
    widget.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    )


def _session_probe(monkeypatch, win, *, fail: bool = False) -> dict:
    """Replace the session instance's transport; record what lands there.

    `play` sets the session's `_t0` so the REAL `playing` property (and the
    window's own toggles) behave like a sounding clip; only `_on_play_tick`
    completes it — completion is observed, never assumed.
    """
    calls: dict = {"plays": [], "stops": 0, "pianos": [], "fail": fail}

    def fake_set_sources(piano=None, voice=None) -> None:
        calls["pianos"].append(piano)

    def fake_play(self, mode: str = "piano", t: float = 0.0) -> bool:
        if calls["fail"]:
            return False
        calls["plays"].append(mode)
        self.mode = mode
        self._t0 = time.perf_counter()
        return True

    def fake_stop(self) -> None:
        self._t0 = None
        self.mode = None
        calls["stops"] += 1

    # set_sources keeps its normal self signature as an instance attribute;
    # play/stop need the session as self (they touch _t0/mode)
    monkeypatch.setattr(win.playback, "set_sources", fake_set_sources)
    monkeypatch.setattr(win.playback, "play", MethodType(fake_play, win.playback))
    monkeypatch.setattr(win.playback, "stop", MethodType(fake_stop, win.playback))
    return calls


def test_note_space_toggles_off_through_the_controller(qapp, monkeypatch):
    """The same note's second Space never reaches the session's play again."""
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win)
        table = win.page_results.results._notes_table
        table.setCurrentCell(1, 0)

        _press_space(table)
        assert calls["plays"] == ["voice"]
        assert not win.page_results.label_note_audition.isHidden()

        _press_space(table)
        assert calls["plays"] == ["voice"]  # the toggle-off replayed nothing
        # exactly ONE stop for this toggle: the controller's cancel; the
        # FIRST Space never pre-stopped (the clip replaced whatever sounded)
        assert calls["stops"] == 1
        assert win.page_results.label_note_audition.isHidden()
    finally:
        win.close()


def test_a_note_replacement_clears_only_the_orphaned_readout(qapp, monkeypatch):
    """Move the audition to another note: the old readout clears, the new shows."""
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win)
        table = win.page_results.results._notes_table

        table.setCurrentCell(0, 0)
        _press_space(table)
        assert not win.page_results.label_note_audition.isHidden()

        table.setCurrentCell(1, 0)
        _press_space(table)
        assert calls["plays"] == ["voice", "voice"]
        assert not win.page_results.label_note_audition.isHidden()  # the new one
    finally:
        win.close()


def test_a_whole_take_button_cancels_an_active_audition_and_starts_its_mode(qapp, monkeypatch):
    """The piano button with a voice audition active: cancel, then play piano."""
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win)
        table = win.page_results.results._notes_table
        table.setCurrentCell(0, 0)
        _press_space(table)
        assert not win.page_results.label_note_audition.isHidden()

        win.on_toggle_play()
        assert calls["plays"] == ["voice", "piano"]
        # exactly ONE stop for this hand-over: the button's cancel of the
        # note audition; the clip never pre-stopped (its stage replaces)
        assert calls["stops"] == 1
        assert win.page_results.label_note_audition.isHidden()
    finally:
        win.close()


def test_without_an_audition_the_same_mode_button_still_toggles(qapp, monkeypatch):
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win)
        win.on_toggle_play()
        assert calls["plays"] == ["piano"]

        win.on_toggle_play()
        assert calls["plays"] == ["piano"]  # the same-mode stop replays nothing
        # ONE stop per stop request (the session stop has one owner)
        assert calls["stops"] == 2
    finally:
        win.close()


def test_the_mode_matching_voice_button_replaces_a_note_audition(qapp, monkeypatch):
    """Both buttons cancel an audition even when the mode matches (the spec)."""
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win)
        table = win.page_results.results._notes_table
        table.setCurrentCell(0, 0)
        _press_space(table)
        assert not win.page_results.label_note_audition.isHidden()

        win.on_toggle_voice()
        assert calls["plays"] == ["voice", "voice"]
        assert win.page_results.label_note_audition.isHidden()
    finally:
        win.close()


def test_seek_cancels_the_audition_and_follows_its_own_guard(qapp, monkeypatch):
    win = _window()
    try:
        from v2m.core.playback import NotePlayer

        calls = _session_probe(monkeypatch, win)
        result = win.result
        assert result is not None
        win.player = NotePlayer(result.notes, sr=SR)
        table = win.page_results.results._notes_table
        table.setCurrentCell(0, 0)
        _press_space(table)

        win.on_seek(0.3)
        assert calls["plays"] == ["voice", "voice"]
        assert win.page_results.label_note_audition.isHidden()
    finally:
        win.close()


def test_volume_change_leaves_an_in_flight_audition_untouched(qapp, monkeypatch):
    """Nothing stops, restarts or clears; the new gain applies next start."""
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win)
        table = win.page_results.results._notes_table
        table.setCurrentCell(0, 0)
        _press_space(table)
        plays_before = len(calls["plays"])
        stops_before = calls["stops"]

        win.on_piano_volume_changed(70)
        assert len(calls["plays"]) == plays_before
        assert calls["stops"] == stops_before
        assert not win.page_results.label_note_audition.isHidden()
    finally:
        win.close()


def test_transpose_change_leaves_an_in_flight_audition_untouched(qapp, monkeypatch):
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win)
        table = win.page_results.results._notes_table
        table.setCurrentCell(0, 0)
        _press_space(table)
        plays_before = len(calls["plays"])

        win.on_transpose_changed(-2)
        assert len(calls["plays"]) == plays_before
        assert not win.page_results.label_note_audition.isHidden()
    finally:
        win.close()


def test_the_drag_sequence_runs_entirely_through_the_controller(qapp, monkeypatch):
    """Old note, then moved note: the window prepares, the controller sequences."""
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win)
        from v2m.core.notes import Note

        calls["pianos"].clear()  # only the drag's own stagings from here
        win.page_results.piano.note_dragged.emit(1, 0.7, 1.1, 62)
        result = win.result
        assert result is not None
        moved = result.notes[1]
        assert (moved.start, moved.end, moved.midi) == (0.7, 1.1, 62)
        assert calls["plays"] == ["piano"]
        # the FIRST clip is the OLD note's render — the note as it was
        # BEFORE the move (0.5-0.8, midi 64), not the moved one
        old = Note(0.5, 0.8, 64, 0.9)
        # the drag restages the take's own sources first (the move handler),
        # then the controller stages the clip: the clip is the LAST staging
        assert len(calls["pianos"]) == 2
        assert np.array_equal(calls["pianos"][-1], win._rendered_note(old))

        win.playback._t0 = None  # the first clip completes
        win._on_play_tick()
        assert calls["plays"] == ["piano", "piano"]
        assert len(calls["pianos"]) == 3
        assert moved is not None
        assert np.array_equal(calls["pianos"][-1], win._rendered_note(moved))

        win.playback._t0 = None  # the second clip completes
        win._on_play_tick()
        assert calls["plays"] == ["piano", "piano"]  # nothing restarts
        assert len(calls["pianos"]) == 3
    finally:
        win.close()


def test_an_explicit_stop_cancels_a_running_drag_sequence(qapp, monkeypatch):
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win)
        win.page_results.piano.note_dragged.emit(1, 0.7, 1.1, 62)
        win.stop_playback()
        win.playback._t0 = None
        win._on_play_tick()  # a tick after the cancel starts nothing
        assert calls["plays"] == ["piano"]
    finally:
        win.close()


def test_finder_and_phrase_auditions_start_the_observer(qapp, monkeypatch):
    """Every successful short audition runs the 33 ms observer (the spec)."""
    from v2m.core.pipeline import AnalysisResult

    win = _window()
    try:
        _session_probe(monkeypatch, win)
        win.on_audition_key(64)
        assert win._play_timer.isActive()

        win.on_audition_original()
        assert win._play_timer.isActive()

        win._replacement_decision = (None, 0.1, 0.4)
        win.result = AnalysisResult(
            source="live",
            duration_s=1.5,
            notes=[],
            tempo=None,
            key=None,
            pitch=None,
        )
        win._live_signal = np.ones(int(1.5 * SR), dtype=np.float32)
        win.on_audition_replacement()
        assert win._play_timer.isActive()
    finally:
        win.close()


def test_a_live_note_audition_moves_its_readout_through_the_controller(qapp, monkeypatch):
    """Space on the frozen roll: same semantics, one controller path."""
    from v2m.core.liveseg import LiveNote

    win = _window()
    try:
        win._live_signal = np.ones(int(1.5 * SR), dtype=np.float32)
        win._last_take_live_notes = [LiveNote(0.2, 0.5, 60), LiveNote(0.6, 0.9, 64)]
        calls = _session_probe(monkeypatch, win)

        win.on_audition_live_note(1)
        assert calls["plays"] == ["voice"]
        assert not win.page_live.label_live_audition.isHidden()

        # switching the audition: the live one is the orphan readout cleared
        win.on_audition_note(0)
        assert win.page_live.label_live_audition.isHidden()
        assert not win.page_results.label_note_audition.isHidden()
        assert calls["plays"] == ["voice", "voice"]
    finally:
        win.close()


def test_a_failed_start_leaves_no_active_observation(qapp, monkeypatch):
    """A failed `play` shows no readout, keeps NO identity and starts NO
    timer: the observer only watches a sounding clip.
    """
    win = _window()
    try:
        calls = _session_probe(monkeypatch, win, fail=True)
        table = win.page_results.results._notes_table
        table.setCurrentCell(0, 0)

        _press_space(table)
        assert calls["plays"] == []
        assert win.page_results.label_note_audition.isHidden()
        assert not win._play_timer.isActive()

        # the failed start also left nothing for a later completion to spend:
        # nothing sounds, a tick changes nothing
        win.playback._t0 = None
        win._on_play_tick()
        assert not win._play_timer.isActive()
    finally:
        win.close()
