"""End to end of the live GUI mode: fake mic -> capture -> analysis -> export.

After stopping a live recording, the final analysis and the export run
BEFORE any rendering: the export is the take's data, a cosmetic hiccup
below must never cost it.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from v2m.core import audio_io

MELODY_HZ = [220.0, 261.63, 293.66, 329.63]  # A3 C4 D4 E4
TONE_S = 0.35
GAP_S = 0.12
CAPTURE_SR = 44_100


class FakeMic:
    """A stand-in `sounddevice.InputStream`: replays a known melody."""

    def __init__(self, samplerate, channels, blocksize, callback, **kwargs):  # noqa: ANN001
        self.rate = samplerate
        self.block = blocksize
        self.callback = callback
        self.on_block = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._play, daemon=True)
        self._thread.start()

    def _play(self) -> None:
        for hz in MELODY_HZ:
            n = int(self.rate * TONE_S)
            t = np.arange(n) / self.rate
            tone = (0.35 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
            if not self._emit(tone) or not self._emit(np.zeros(int(self.rate * GAP_S), np.float32)):
                return

    def _emit(self, y: np.ndarray) -> bool:
        for i in range(0, len(y), self.block):
            if self._stop.is_set():
                return False
            blk = y[i : i + self.block]
            if len(blk) < self.block:
                blk = np.pad(blk, (0, self.block - len(blk)))
            self.callback(blk.reshape(-1, 1), self.block, None, None)
            time.sleep(0.005)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _pump_until(app, predicate, timeout: float) -> bool:
    """Run the Qt event loop until `predicate()` (queued signals need it)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_live_take_reaches_results_and_export(qapp, monkeypatch):
    import sounddevice as sd

    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    exported: list[tuple] = []
    monkeypatch.setattr(
        mw, "export_result", lambda res, stem, **kw: exported.append((res, stem, kw))
    )

    win = mw.MainWindow()
    try:
        win.on_start_live()
        captured = _pump_until(
            qapp,
            lambda: win.live_thread is not None and win.live_thread.elapsed() >= 1.8,
            timeout=60,
        )
        assert captured, "the fake mic delivered nothing to the live capture"
        live_thread = win.live_thread
        assert live_thread is not None

        # during capture: closed notes are already on the piano roll and
        # on the scrolling staff
        seg = live_thread.segmenter
        assert seg is not None and seg.notes, "no note detected during live"
        assert win.page_live.piano_live._bar_items, "the live piano roll stayed empty"
        assert win.page_live.staff_live.layout().notes, "the live staff stayed empty"

        win.on_stop_live()
        assert _pump_until(qapp, lambda: win.result is not None, timeout=60), (
            "no final analysis after stopping the live"
        )

        result = win.result
        assert result is not None
        assert result.notes, "the live final analysis is empty"
        assert win._live_signal is not None and len(win._live_signal) > 0
        assert win._voice_signal() is win._live_signal
        assert not exported, "a live transcription must remain pending until export"
        win._export_results()
        assert exported, "the explicit export action did not write the live take"
        res, stem, _ = exported[-1]
        assert res is win.result
        assert Path(stem).name.startswith("live_"), f"unexpected file name : {stem}"
        assert Path(stem).parent.parent.name == "prises_live", (
            f"the take must live in prises_live/<stem>/ : {stem}"
        )

        # the results page shows the take, and nothing was flagged red
        assert win.stack.currentWidget() is win.page_results
        assert len(win.page_results.piano._bar_items) == len(result.notes)
        assert win.page_results.piano_keys._spans
        staff_notes = win.page_results.staff.layout().notes
        assert len(staff_notes) == len(result.notes), "the staff does not follow the analysis"
        assert all(1 <= n.measure for n in staff_notes)
        assert win.error_banner.isHidden(), win.error_banner.text()
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)


def test_phrase_replacement_stays_pending_until_explicit_export(qapp, monkeypatch):
    """A re-sung range replaces the selected phrase without auto-exporting."""
    import sounddevice as sd

    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    exported: list[tuple] = []
    monkeypatch.setattr(
        mw, "export_result", lambda res, stem, **kw: exported.append((res, stem, kw))
    )

    win = mw.MainWindow()
    win.page_live.chk_record_audio.setChecked(False)
    win._pump_replay_queue = lambda: None
    try:
        win.on_start_live()
        assert _pump_until(
            qapp,
            lambda: win.live_thread is not None and win.live_thread.elapsed() >= 1.8,
            timeout=60,
        )
        win.on_stop_live()
        assert _pump_until(qapp, lambda: win.result is not None, timeout=60)
        original = win.result
        assert original is not None
        original_audio = win._live_signal
        assert original_audio is not None
        original_count = len(original.notes)
        original_samples = len(original_audio)
        assert not exported

        win.page_results.results._notes_table.selectRow(1)
        assert win.page_results._selected_note_indices() == [1]
        win.on_start_phrase_replacement()
        assert win._live_record_audio is False
        assert win.page_live._replacement_mode
        assert _pump_until(
            qapp,
            lambda: win.live_thread is not None and win.live_thread.elapsed() >= 1.8,
            timeout=60,
        )
        win.on_finish_phrase_replacement()
        assert _pump_until(
            qapp,
            lambda: not win._replacement_pending_analysis
            and win._signal_analysis_thread is None
            and win.result is not original,
            timeout=90,
        ), "the edited take was not re-analyzed"

        assert win.result is not None
        assert len(win.result.notes) > original_count
        assert win._live_signal is not None
        assert len(win._live_signal) > original_samples
        assert win.result.observation is not None

        # the spliced take awaits the singer's decision: the audition/keep/
        # cancel row is up, export waits with it (isHidden: the window is
        # never shown here — isVisible() is False for every widget)
        assert win._replacement_decision is not None
        assert not win.page_results.btn_accept_replacement.isHidden()
        assert not exported, "phrase replacement must not export implicitly"
        win._export_results()
        assert not exported, "export waits for the keep/cancel decision"

        win._on_accept_replacement()
        assert win._replacement_decision is None
        assert win._edit_history.can_undo, "the kept splice is an undoable edit"
        win._export_results()
        assert len(exported) == 1
        assert exported[0][0] is win.result
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)
        win.close()


def test_audition_clip_of_a_real_resing_contains_the_new_singing(qapp, monkeypatch):
    """End to end with the fake mic: re-sing the selected phrase as a
    DIFFERENT melody, then the decision phase's audition clip must SOUND
    like the new singing — the replaced span of the merged audio, not the
    original take's span. The check is spectral: the clip's dominant
    component sits at the replacement's frequency, the original melody's
    frequency is gone."""
    import sounddevice as sd

    from v2m.core.pipeline import AnalysisResult
    from v2m.core.notes import Note
    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    monkeypatch.setattr(
        mw, "export_result", lambda res, stem, **kw: None  # the decision waits
    )

    win = mw.MainWindow()
    win.page_live.chk_record_audio.setChecked(False)
    win._pump_replay_queue = lambda: None
    try:
        win.on_start_live()
        assert _pump_until(
            qapp,
            lambda: win.live_thread is not None and win.live_thread.elapsed() >= 1.8,
            timeout=60,
        )
        win.on_stop_live()
        assert _pump_until(qapp, lambda: win.result is not None, timeout=60)
        original = win.result
        assert original is not None and win._live_signal is not None
        original_audio = win._live_signal
        notes = original.notes
        assert len(notes) >= 1, "the fake-mic take found no notes"

        # the phrase to re-sing, and the melody the singer performs instead
        first = last = 0  # the take's FIRST note: the fake mic's 220 Hz tone
        span_lo = int(round(notes[first].start * original.sr))
        span_hi = int(round(notes[last].end * original.sr))
        span = original_audio[span_lo:span_hi]
        old_hz, new_hz = MELODY_HZ[0], 660.0

        # splice the new singing by hand (the capture->splice flow is pinned
        # by the two tests above; THIS test pins the AUDITION CONTENT)
        from v2m.core.edits import splice_phrase

        sr = original.sr
        sung_n = span_hi - span_lo + int(0.05 * sr)  # a slightly longer phrase
        t = np.arange(sung_n) / sr
        sung = (0.4 * np.sin(2 * np.pi * new_hz * t)).astype(np.float32)
        # the replacement's note times live in the SUNG clip's own timeline
        # (splice_phrase trims the sung signal at its first onset)
        replacement = [Note(0.0, sung_n / sr - 0.02, 76, 0.9)]  # E5: the new pitch
        notes_merged, merged_signal, delta = splice_phrase(
            notes, original_audio, replacement, sung, first, last, sr=sr
        )
        spliced = AnalysisResult(
            source="live",
            duration_s=len(merged_signal) / sr,
            notes=notes_merged,
            tempo=original.tempo,
            key=None,
            pitch=None,
            sr=sr,
        )
        window = (
            notes[first].start,
            notes[last].end + delta,
        )
        win._replacement_decision = (win._edit_snapshot(), window[0], window[1])
        win._on_replacement_analysis_done(spliced, merged_signal)
        assert win._replacement_decision is not None

        # the audition clip: spectral fingerprint of the NEW singing
        win.on_audition_replacement()
        clip = win.playback.voice
        assert len(clip) > 0, "the audition produced no clip"
        spectrum = np.abs(np.fft.rfft(clip * np.hanning(len(clip))))
        freqs = np.fft.rfftfreq(len(clip), 1.0 / sr)
        band = (freqs > 200) & (freqs < 1200)
        dominant = freqs[band][int(np.argmax(spectrum[band]))]
        assert abs(dominant - new_hz) < 25.0, (
            f"the audition clip does not sound like the new singing: "
            f"dominant {dominant:.0f} Hz, expected ~{new_hz:.0f} Hz"
        )
        old_share = float(np.mean(spectrum[band & (freqs > old_hz - 25) & (freqs < old_hz + 25)]))
        overall = float(np.mean(spectrum[band]))
        assert old_share < overall, "the original melody still sounds in the audition clip"

        # the original span's own fingerprint, for contrast: the OLD melody
        # dominates THERE, proving the two clips are different audio (its
        # OWN frequency grid — a different clip length, a different rfft)
        old_spectrum = np.abs(np.fft.rfft(span * np.hanning(len(span))))
        old_freqs = np.fft.rfftfreq(len(span), 1.0 / sr)
        old_band = (old_freqs > 200) & (old_freqs < 1200)
        old_dominant = old_freqs[old_band][int(np.argmax(old_spectrum[old_band]))]
        assert abs(old_dominant - old_hz) < 25.0
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)
        win.close()


def test_correction_and_replacement_recover_with_undo_and_redo(qapp, monkeypatch):
    """The two edit paths recover exactly: a corrected note undoes and
    redoes, and a re-sung phrase is auditioned, cancelled or kept — the
    history restoring the take (notes, audio, exportable bytes) whole.
    The capture -> splice -> analysis flow is pinned by
    `test_phrase_replacement_stays_pending_until_explicit_export`; this
    test drives the DECISION and the history at GUI level, on a synthetic
    take, so the recovery wiring is the only thing under test.
    """
    from v2m.core.edits import splice_phrase
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.core.tempo import TempoResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        sr = 8000
        result = AnalysisResult(
            source="live",
            duration_s=3.0,
            notes=[
                Note(0.2, 0.7, 60, 0.9),
                Note(0.9, 1.4, 64, 0.9),
                Note(1.7, 2.2, 67, 0.9),
            ],
            tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 3.0, 0.5)),
            key=None,
            pitch=None,
            sr=sr,
        )
        win.result = result
        win._live_signal = np.linspace(-0.2, 0.2, 3 * sr, dtype=np.float32)
        win._signal_cache = {}
        win._show_result(result)

        # -- a correction is undoable and redoable ---------------------------
        win.page_results.results._notes_table.selectRow(1)
        win.page_results.piano_keys.key_clicked.emit(65)
        assert win.result.notes[1].midi == 65
        assert win.result.manual_pitch_corrections
        assert win.page_results.btn_undo.isEnabled()
        assert not win.page_results.btn_redo.isEnabled()

        win.on_undo_edit()
        assert win.result is not result  # the whole state was swapped in
        assert win.result.notes[1].midi == 64
        assert not win.result.manual_pitch_corrections
        assert win.page_results.btn_redo.isEnabled()
        assert win.take_ctl.signal() == win._live_signal.tobytes(), (
            "the exportable audio follows the restored state"
        )

        win.on_redo_edit()
        assert win.result.notes[1].midi == 65

        # -- the replacement's decision phase ---------------------------------
        replacement = [Note(0.9, 1.3, 72, 0.9), Note(1.3, 1.7, 74, 0.9)]
        sung = np.linspace(0.1, -0.1, 2 * sr, dtype=np.float32)

        def begin_decision():
            """Splice the phrase by hand and open the decision phase."""
            before = win._edit_snapshot()
            current = win.result
            signal = win._live_signal
            assert before is not None and current is not None and signal is not None
            notes, merged, delta = splice_phrase(
                current.notes, signal, replacement, sung, 1, 1, sr=sr
            )
            spliced = AnalysisResult(
                source="live",
                duration_s=len(merged) / sr,
                notes=notes,
                tempo=current.tempo,
                key=None,
                pitch=None,
                sr=sr,
            )
            win._replacement_decision = (before, 0.9, 1.4 + delta)
            win._on_replacement_analysis_done(spliced, merged)
            return merged

        merged = begin_decision()
        assert win._replacement_decision is not None
        assert not win.page_results.btn_accept_replacement.isHidden()
        assert not win.page_results.btn_export.isEnabled()
        assert not win.page_results.btn_undo.isEnabled()
        assert not win.page_results.btn_redo.isEnabled()

        played: list[str] = []
        monkeypatch.setattr(
            win.playback, "play", lambda mode, t=0.0: played.append(mode) or True
        )
        win.on_audition_replacement()
        assert played == ["voice"]
        # the audition clip IS the replaced span of the merged audio
        assert len(win.playback.voice) == int(round(0.8 * sr))

        win._on_cancel_replacement()
        assert win._replacement_decision is None
        assert len(win.result.notes) == 3  # the splice is gone
        assert win.result.notes[1].midi == 65  # the correction survived it
        assert len(win._live_signal) == 3 * sr  # the original audio is back
        assert win.page_results.btn_accept_replacement.isHidden()
        assert win.page_results.btn_undo.isEnabled()  # the correction is still undoable

        # -- keeping the splice turns it into an undoable edit -----------------
        begin_decision()
        win._on_accept_replacement()
        assert win._replacement_decision is None
        assert len(win.result.notes) == 4
        assert len(win._live_signal) == len(merged)
        assert win.page_results.btn_accept_replacement.isHidden()
        assert win._edit_history.can_undo

        win.on_undo_edit()
        assert len(win.result.notes) == 3
        assert win.result.notes[1].midi == 65
        assert len(win._live_signal) == 3 * sr
        assert win.take_ctl.signal() == win._live_signal.tobytes()
        assert win._edit_history.can_redo

        win.on_redo_edit()
        assert len(win.result.notes) == 4
        assert len(win._live_signal) == len(merged)
        assert win.take_ctl.signal() == win._live_signal.tobytes()
    finally:
        win.close()


def test_state_dropdown_jumps_to_any_previous_state(qapp, monkeypatch):
    """The results page's dropdown lists the take's whole edit timeline
    (every state + the current one) and a choice jumps straight there —
    equivalent to walking undo/redo steps, and the timeline keeps the
    states the jump passed (a slider on the take's past AND future)."""
    from v2m.core.i18n import get_language, set_language, tr
    from v2m.core.pipeline import AnalysisResult
    from v2m.core.notes import Note
    from v2m.core.tempo import TempoResult
    from v2m.gui import main_window as mw

    set_language("fr")
    win = mw.MainWindow()
    try:
        sr = 8000
        result = AnalysisResult(
            source="live",
            duration_s=3.0,
            notes=[Note(0.2, 0.7, 60, 0.9), Note(0.9, 1.4, 64, 0.9)],
            tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 3.0, 0.5)),
            key=None,
            pitch=None,
            sr=sr,
        )
        win.result = result
        win._live_signal = np.zeros(3 * sr, dtype=np.float32)
        win._show_result(result)

        # two corrections build a two-row timeline (the rows always end with
        # the current state, never jumpable-to)
        win.page_results.results._notes_table.selectRow(0)
        win.page_results.piano_keys.key_clicked.emit(62)
        win.page_results.results._notes_table.selectRow(1)
        win.page_results.piano_keys.key_clicked.emit(65)

        combo = win.page_results.combo_edit_states
        assert combo.count() == 3
        assert [combo.itemText(i) for i in range(3)] == [
            tr("correction"), tr("correction"), tr("current"),
        ]
        assert combo.currentIndex() == 2, "the dropdown sits on the current take"

        # jump straight to the FIRST state: both corrections are undone —
        # and because every move re-archives the state it leaves, the two
        # corrections are now REDO rows: the dropdown is a slider, jumping
        # back keeps the future reachable
        monkeypatch.setattr(win, "_edit_jumps_allowed", lambda: True)
        combo.setCurrentIndex(0)
        assert win.result.notes[0].midi == 60
        assert win.result.notes[1].midi == 64
        assert not win.page_results.btn_undo.isEnabled()
        assert win.page_results.btn_redo.isEnabled()
        assert combo.currentIndex() == 0, "the dropdown follows the restored state"
        assert combo.count() == 3, "the timeline keeps every state the jump passed"

        # redo walks forward through both corrections again
        win.on_redo_edit()
        assert win.result.notes[0].midi == 62 and win.result.notes[1].midi == 64
        win.on_redo_edit()
        assert win.result.notes[0].midi == 62 and win.result.notes[1].midi == 65
        assert combo.currentIndex() == 2, "the dropdown sits on the current row again"

        # picking the current row does nothing
        before = win.result
        combo.setCurrentIndex(2)
        assert win.result is before
    finally:
        set_language(get_language())
        win.close()


def test_audition_original_plays_the_selected_span_of_the_current_audio(qapp, monkeypatch):
    """▷ plays the selected notes' span of the CURRENT take's audio — the
    like-for-like companion of the re-sung phrase's audition clip."""
    from v2m.core.pipeline import AnalysisResult
    from v2m.core.notes import Note
    from v2m.core.tempo import TempoResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        sr = 8000
        result = AnalysisResult(
            source="live",
            duration_s=3.0,
            notes=[
                Note(0.2, 0.7, 60, 0.9),
                Note(0.9, 1.4, 64, 0.9),
                Note(1.7, 2.2, 67, 0.9),
            ],
            tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 3.0, 0.5)),
            key=None,
            pitch=None,
            sr=sr,
        )
        win.result = result
        signal = np.linspace(-0.3, 0.3, 3 * sr, dtype=np.float32)
        win._live_signal = signal
        win._show_result(result)

        win.page_results.results._notes_table.selectRow(1)
        played: list[str] = []
        monkeypatch.setattr(
            win.playback, "play", lambda mode, t=0.0: played.append(mode) or True
        )
        win.on_audition_original()

        assert played == ["voice"]
        # the clip is notes[1]'s span (0.9 s -> 1.4 s) of the CURRENT audio
        expected = signal[int(round(0.9 * sr)):int(round(1.4 * sr))]
        assert len(win.playback.voice) == len(expected)
        assert np.array_equal(win.playback.voice, expected)

        # no selection, no audition (and nothing replaces the playing clip)
        win.page_results.results._notes_table.clearSelection()
        win.on_audition_original()
        assert played == ["voice"]
    finally:
        win.close()


def test_clicking_results_piano_corrects_only_the_selected_note(qapp):
    """A clicked piano key updates a selected note before explicit export."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    original = Note(0.2, 0.7, 60, 0.9)
    result = AnalysisResult(
        source="song.wav",
        duration_s=1.0,
        notes=[original, Note(0.8, 1.0, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    win.page_results.results._notes_table.selectRow(0)

    try:
        win.page_results.piano_keys.key_clicked.emit(62)

        assert result.notes[0].midi == 62
        assert result.notes[1].midi == 64
        assert result.notes[0] is not original and original.midi == 60
        assert (result.notes[0].start, result.notes[0].end) == (0.2, 0.7)
        assert result.pitch is None
        assert result.manual_pitch_corrections
        assert not win.error_banner.isVisible()
        assert "live/final" in win.page_results.results._lbl_explain.text()
    finally:
        win.close()


def test_a_graph_gesture_selects_the_row_and_drives_the_edit(qapp):
    """Pressing a bar on the piano roll selects that note's row: the edit
    controls and the correction path then see EXACTLY that selection."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw
    from v2m.gui.widgets import ACCENT

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
        duration_s=1.0,
        notes=[Note(0.2, 0.7, 60, 0.9), Note(0.8, 1.0, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    try:
        page = win.page_results
        table = page.results._notes_table

        page.piano.note_selection.emit([0, 1])  # the gesture on the graph

        rows = sorted({i.row() for i in table.selectedIndexes()})
        assert rows == [0, 1]
        assert page.piano._selection == [0, 1]
        assert page.piano._bar_items[1].opts["brush"] == ACCENT
        assert page.btn_audition_original.isEnabled()  # contiguous rows

        # a single-bar press narrows the selection — the correction path
        # corrects exactly one row
        page.piano.note_selection.emit([1])
        assert sorted({i.row() for i in table.selectedIndexes()}) == [1]
        assert page.piano._bar_items[0].opts["brush"] == page.piano._color_for_conf(0.9)

        page.piano_keys.key_clicked.emit(62)  # correct THROUGH the selection
        assert [n.midi for n in result.notes] == [60, 62]  # only row 1 moved
    finally:
        win.close()


def test_a_table_row_lights_the_graph_and_a_new_result_clears_it(qapp):
    """The selection stays ONE fact: a row click renders on the bars, and a
    fresh analysis starts with no highlight anywhere."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw
    from v2m.gui.widgets import ACCENT

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
        duration_s=1.0,
        notes=[Note(0.2, 0.7, 60, 0.9), Note(0.8, 1.0, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    try:
        page = win.page_results
        page.results._notes_table.selectRow(1)

        assert page.piano._selection == [1]
        assert page.piano._bar_items[1].opts["brush"] == ACCENT
        assert page.piano._bar_items[0].opts["brush"] == page.piano._color_for_conf(0.9)

        fresh = AnalysisResult(
            source="take2.wav",
            duration_s=0.5,
            notes=[Note(0.1, 0.4, 62, 0.9)],
            tempo=None,
            key=None,
            pitch=None,
        )
        win.result = fresh
        win._show_result(fresh)

        assert page.piano._selection == []
        assert not page.results._notes_table.selectedIndexes()
        assert page.piano._bar_items[0].opts["brush"] == page.piano._color_for_conf(0.9)
    finally:
        win.close()


def test_the_toolbar_readout_reports_the_selected_notes(qapp):
    """The selection's concrete shape — how many notes over which seconds —
    sits next to the edit buttons, hidden while nothing is selected, and a
    language switch re-renders it in the new language."""
    from v2m.core.i18n import tr
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
        duration_s=1.5,
        notes=[Note(0.2, 0.7, 60, 0.9), Note(0.8, 1.0, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    try:
        label = win.page_results.label_selection
        assert label.isHidden()  # no selection, no readout

        win.page_results.results._notes_table.selectRow(1)
        assert not label.isHidden()
        assert label.text() == tr("ui selection label").format(n=1, t0=0.8, t1=1.0)

        win.page_results.results._notes_table.selectAll()
        assert label.text() == tr("ui selection label").format(n=2, t0=0.2, t1=1.0)

        win.page_results.results._notes_table.clearSelection()
        assert label.isHidden()

        win.page_results.results._notes_table.selectRow(0)
        before = label.text()
        win.on_toggle_language()
        assert label.text() == tr("ui selection label").format(n=1, t0=0.2, t1=0.7)
        assert label.text() != before  # the other language's wording
        win.on_toggle_language()  # leave the shared language as it was
        assert label.text() == before
    finally:
        win.close()


def test_del_removes_the_selected_range_as_one_undoable_edit(qapp):
    """Del deletes the selected rows through the edit history: one undo
    brings every note back, one redo removes them again, and Del with no
    selection is a silent no-op."""
    from v2m.core.i18n import tr
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
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
    win.result = result
    win._show_result(result)
    try:
        page = win.page_results
        page.results._notes_table.selectRow(1)
        win.on_delete_selection()

        assert [n.midi for n in result.notes] == [60, 67]
        assert page.label_selection.isHidden()  # the new result starts unselected
        # the state browser names the edit that archived the past state
        assert page.combo_edit_states.itemText(0) == tr("deletion")

        win.on_undo_edit()  # ONE undo brings every deleted note back
        # (undo swaps in the archived whole-state: read it on the window)
        assert [n.midi for n in win.result.notes] == [60, 64, 67]
        win.on_redo_edit()  # and one redo removes them again
        assert [n.midi for n in win.result.notes] == [60, 67]

        page.results._notes_table.clearSelection()
        depth = win._edit_history.undo_depth
        win.on_delete_selection()  # no selection: silent, nothing recorded
        assert [n.midi for n in win.result.notes] == [60, 67]
        assert win._edit_history.undo_depth == depth
        assert win.error_banner.isHidden()
    finally:
        win.close()


def test_del_that_would_empty_the_take_is_refused_and_announced(qapp):
    """The core refuses to empty the take: the notes stay, the history
    gains nothing, and the refusal reaches the error banner."""
    from v2m.core.i18n import tr
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
        duration_s=1.0,
        notes=[Note(0.2, 0.7, 60, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    try:
        win.page_results.results._notes_table.selectRow(0)
        win.on_delete_selection()

        assert [n.midi for n in result.notes] == [60]  # untouched
        assert not win._edit_history.can_undo
        prefix = tr("err note deletion failed").split("{e}")[0]
        assert prefix in win.error_banner.text()  # the banner adds its ⚠ mark
    finally:
        win.close()


def test_the_live_roll_is_pickable_only_after_the_take_stops(qapp, monkeypatch):
    """A running take's roll is never picked; once the take stops its closed
    notes sit frozen on the live page and open up for read-only exploring —
    and the final analysis never re-arms the gate."""
    import sounddevice as sd

    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    win = mw.MainWindow()
    try:
        win.on_start_live()
        assert win.live_thread is not None
        assert not win.page_live.piano_live._selectable  # a running take

        captured = _pump_until(
            qapp,
            lambda: win.live_thread is not None and win.live_thread.elapsed() >= 1.8,
            timeout=60,
        )
        assert captured, "the fake mic delivered nothing to the live capture"
        assert win.page_live.piano_live._bar_items, "no live notes to explore"

        win.on_stop_live()
        # the final analysis is still queued: the closed notes are frozen
        # on the live page, now pickable read-only
        assert win.page_live.piano_live._selectable
        assert win.page_live.piano_live._bar_items

        assert _pump_until(qapp, lambda: win.result is not None, timeout=60)
        assert win.page_live.piano_live._selectable  # the analysis changes nothing
    finally:
        win.close()


def test_finder_keyboard_audition_replaces_the_previous_note(qapp, monkeypatch):
    """Clicking a new finder key replaces audio; leaving stops the sound."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    played: list[str] = []
    try:
        monkeypatch.setattr(win.playback, "play", lambda mode: played.append(mode) or True)
        win.on_audition_finder_note(60)
        first = win._finder_audition.copy()
        win.on_audition_finder_note(64)
        second = win._finder_audition.copy()
        assert played == ["piano", "piano"]
        assert not np.array_equal(first, second)
        win.page_finder.leave()
        assert win.playback.mode is None
    finally:
        win.close()


def test_live_grid_is_the_take_published_one(qapp, monkeypatch):
    """The GUI consumes the live Grid published by LiveTake: it no longer
    re-derives the approximated BPM itself (the old code rebuilt a
    BeatGrid(estimate_bpm_live(...)) on every note-count change in the
    tick)."""
    import sounddevice as sd

    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    monkeypatch.setattr(mw, "export_result", lambda res, stem, **kw: None)

    win = mw.MainWindow()
    try:
        win.on_start_live()

        def published_grid() -> bool:
            thr = win.live_thread
            seg = thr.segmenter if thr is not None else None
            return seg is not None and bool(seg.notes) and win._live_grid.bpm != 120.0

        assert _pump_until(
            qapp, published_grid, timeout=60
        ), "the live grid published by LiveTake never reached the GUI"
        # the live staff engraves the consumed grid (the staff reads
        # grid.bpm — any drift of the publication would show here)
        assert _pump_until(
            qapp, lambda: win.page_live.staff_live._grid is win._live_grid, timeout=5
        ), "the live staff did not receive the published grid"
        # the published meter reaches the status label, next to the beat count
        assert win._live_grid.meter.label in win.page_live.label_status.text(), (
            f"meter absent from the live status : {win.page_live.label_status.text()}"
        )
        # clean stop: the take finishes its analysis during THIS test,
        # else the queued event delivers during the next test
        win.on_stop_live()
        assert _pump_until(qapp, lambda: win.result is not None, timeout=60)
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)


def test_a_failed_take_leaves_a_trace_in_its_folder(qapp, monkeypatch, tmp_path):
    """ADR 0031 end to end: a take whose finalization fails (mic failure,
    capture too short) leaves `<stem>.trace.json` in its own folder. The
    failure arrives through the REAL event path (the queued `failed`
    signal), not by calling `fail_take` directly: the pin is the
    window's wiring, not the controller's internals."""
    import sounddevice as sd

    from v2m.core import takes
    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    monkeypatch.setattr(mw, "export_result", lambda res, stem, **kw: None)
    monkeypatch.setattr(takes, "LIVE_TAKES_DIR", tmp_path)  # the trace stays in the test

    win = mw.MainWindow()
    try:
        win.on_start_live()
        captured = _pump_until(
            qapp,
            lambda: win.live_thread is not None and win.live_thread.elapsed() >= 0.3,
            timeout=60,
        )
        assert captured, "the fake mic delivered nothing to the live capture"
        # make finalize() refuse the take (a capture too short): the failure
        # event is the real queued signal the window must answer
        thread = win.live_thread
        assert thread is not None
        thread._stop = True
        take = thread.take
        assert take is not None
        original_finalize = take.finalize

        def finalize_too_short():
            take._signal = np.zeros(int(take.sr * 0.2), dtype=np.float32)
            return original_finalize()

        monkeypatch.setattr(take, "finalize", finalize_too_short)
        win.on_stop_live()
        # the trace IS the outcome: pump on the file (the failure signal is
        # queued — it delivers while the loop runs)
        assert _pump_until(
            qapp, lambda: list(tmp_path.rglob("*.trace.json")), timeout=60
        ), "the failed take left no trace in its folder"
        import json

        payload = json.loads(
            next(tmp_path.rglob("*.trace.json")).read_text(encoding="utf-8")
        )
        assert "too short" in payload["reason"]
        assert payload["exported"] is False
        # the failure reached the singer too: the status label says why
        assert win.page_live.label_status.text().strip()
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)


def test_note_finder_lights_the_sung_key_and_keeps_nothing(qapp, monkeypatch):
    """The "find the note" section has one role: light the sung key. No
    final analysis, no export, nothing kept."""
    import sounddevice as sd

    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    exported: list[tuple] = []
    monkeypatch.setattr(
        mw, "export_result", lambda res, stem, **kw: exported.append((res, stem, kw))
    )

    win = mw.MainWindow()
    try:
        win.on_open_finder()
        assert win.stack.currentWidget() is win.page_finder
        # pump on the GROUND TRUTH of confirmation: the python list only
        # ever holds CONFIRMED notes (the widget shows a placeholder while
        # empty, so a widget count says nothing) — the big name flips
        # between answer and invite as the fake singer breathes, it is not
        # pumpable
        assert _pump_until(qapp, lambda: len(win.page_finder.session.history) > 0, timeout=60), (
            "no note confirmed by the note finder"
        )
        # the lit key is the fake melody's (A3 C4 D4 E4), never a number:
        # the piano keyboard is the only displayed answer
        assert win.page_finder.keys._active <= {57, 60, 62, 64}
        assert win.page_finder.label_note.text() != ""
        assert "🎹" in win.page_finder.history.item(0).text()

        win.on_close_finder()
        assert win.stack.currentWidget() is win.page_home
        assert win.page_finder.thread is None
        assert win.page_finder.keys._active == set()
        assert win.page_finder.label_note.text() == tr("ui finder idle")
        assert not exported, "the note finder must not export anything"
        assert win.result is None, "the note finder must not analyze anything"
    finally:
        if win.page_finder.thread is not None and win.page_finder.thread.isRunning():
            win.page_finder.thread.stop()
            win.page_finder.thread.wait(3000)


def test_metronome_follows_detected_bpm(qapp):
    """The optional metronome tracks the take's detected tempo, not a fixed one."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        # 4 onsets 0.5 s apart -> ~120 BPM detected
        notes = []
        for i in range(5):
            notes.append(type("N", (), {"start": i * 0.5})())
        win.page_live.chk_metronome.setChecked(True)  # builds the Metronome
        try:
            assert win.metronome is not None
            win._follow_detected_bpm(120.0)
            assert win.metronome.bpm == 120.0
            assert win.page_live.spin_bpm.value() == 120
            # the spin box is a silent follower: dragging it still works
            win.page_live.spin_bpm.setValue(100)
            assert win.metronome.bpm == 100.0
        finally:
            win.page_live.chk_metronome.setChecked(False)
            assert win.metronome is None
    finally:
        win.close()


def _fake_sd_for_gui(monkeypatch) -> list:
    """Patch the playback module's sd with recording fake streams."""
    import types as _types

    from v2m.core import playback as pb

    class FakeOut:
        instances: list = []

        def __init__(self, **kwargs):
            self.written = 0
            self.closed = False
            self.sr = kwargs.get("samplerate", 22_050)
            FakeOut.instances.append(self)

        def start(self):
            pass

        def write(self, data):
            self.written += len(data)
            # ~0.3x real time: slow enough for the tick to observe playback
            import time as _time

            _time.sleep(0.3 * len(data) / self.sr)

        def stop(self):
            self.closed = True

        def close(self):
            self.closed = True

    FakeOut.instances = []
    monkeypatch.setattr(pb, "sd", _types.SimpleNamespace(OutputStream=FakeOut))
    return FakeOut.instances


def _synthetic_result(win):
    """A result + cached voice, without any microphone."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult

    result = AnalysisResult(source="live", duration_s=1.0, notes=[], tempo=None, key=None)
    result.notes = [
        Note(start=0.1, end=0.9, midi=60, confidence=0.9)
    ]
    win.result = result
    sr = audio_io.SR
    win._live_signal = (
        0.4 * np.sin(2 * np.pi * 220 * np.arange(int(0.3 * sr)) / sr)
    ).astype(np.float32)


def test_voice_and_both_follow_the_playhead(qapp, monkeypatch):
    """🎙 and 🎹+🎙 drive the same playhead as the piano button."""
    from v2m.gui import main_window as mw

    _fake_sd_for_gui(monkeypatch)
    win = mw.MainWindow()
    try:
        _synthetic_result(win)
        assert win.result is not None
        win.player = mw.NotePlayer(win.result.notes, sr=audio_io.SR)

        win.on_toggle_voice()
        assert _pump_until(
            qapp, lambda: win.page_results.waveform._line.value() > 0.0, 5
        ), "the playhead does not follow the voice"
        assert win.page_results.btn_voice.text() == "■"
        win.stop_playback()
        assert win.playback.mode is None

        win.on_toggle_both()
        assert _pump_until(
            qapp,
            lambda: win.playback.playing and win.playback.mode == "both",
            5,
        ), "the combined mode must sound both sources"
        win.stop_playback()
        assert win.page_results.btn_both.text() == "🎹+🎙"
    finally:
        win.close()


def test_voice_replay_uses_cached_file_audio(qapp, monkeypatch):
    """The voice button replays files too; live is not a special source type."""
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    _fake_sd_for_gui(monkeypatch)
    win = mw.MainWindow()
    try:
        sr = audio_io.SR
        result = AnalysisResult(source="take.wav", duration_s=1.0, notes=[], tempo=None, key=None)
        win.result = result
        win._signal_cache["take.wav"] = np.zeros(sr, dtype=np.float32)
        win.player = mw.NotePlayer(result.notes, sr=sr)
        win.on_toggle_voice()
        assert win.playback.mode == "voice"
        win.stop_playback()
    finally:
        win.close()


def test_voice_transpose_shifts_without_reanalysis(qapp, monkeypatch):
    """The spin box re-creates the voice player at the new pitch, in place."""
    from v2m.gui import main_window as mw

    _fake_sd_for_gui(monkeypatch)
    win = mw.MainWindow()
    try:
        _synthetic_result(win)
        assert win.result is not None
        win.player = mw.NotePlayer(win.result.notes, sr=audio_io.SR)
        win.on_toggle_voice()
        assert _pump_until(qapp, lambda: win._voice_player is not None and win.playback.playing, 5)
        voice_player = win._voice_player
        assert voice_player is not None
        old_signal = voice_player.signal
        pos = win.playback.position()

        win.page_results.spin_transpose.setValue(2)  # audible live transpose
        voice_player = win._voice_player
        assert voice_player is not None
        assert voice_player.signal is not old_signal
        assert len(voice_player.signal) == len(old_signal), (
            "the duration must stay exact so the piano does not desync"
        )
        assert not np.allclose(voice_player.signal, old_signal)
        assert win.playback.playing, "transposing must not cut the playback"
        assert win.playback.mode == "voice"
        assert win.playback.position() >= pos - 0.2
        win.stop_playback()
    finally:
        win.close()


def test_both_is_noop_without_a_take(qapp, monkeypatch):
    """🎹+🎙 on a file take (no live signal) plays the piano alone."""
    from v2m.gui import main_window as mw

    _fake_sd_for_gui(monkeypatch)
    win = mw.MainWindow()
    try:
        from v2m.core.pipeline import AnalysisResult

        result = AnalysisResult(source="file.wav", duration_s=1.0, notes=[], tempo=None, key=None)
        win.result = result
        from v2m.core.notes import Note

        result.notes = [Note(start=0.1, end=0.9, midi=60, confidence=0.9)]
        win.player = mw.NotePlayer(result.notes, sr=audio_io.SR)
        win.on_toggle_both()  # no voice available: the piano still sounds
        assert win.playback.playing
        assert win.playback.mode == "both"
        win.stop_playback()
    finally:
        win.close()


def test_live_accuracy_history_is_displayed(qapp):
    """The per-note mean |cents| line fills as the singer sings."""
    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        assert win.page_live.label_accuracy.text() != ""
        win._on_live_note((60.1, 262.0))
        win._on_live_note((60.2, 263.0))
        txt = win.page_live.label_accuracy.text()
        # mean of 10 and 20 cents on C4: any language, no history-dependence
        assert ("C4 / Do4" in txt or "Do4 / C4" in txt) and "15" in txt
        win.page_live.accuracy.reset()
        win._refresh_live_accuracy()
        assert win.page_live.label_accuracy.text() == tr("ui accuracy empty")
    finally:
        win.close()


def test_live_level_bar_shows_the_silence_gate(qapp):
    """The mic level bar on the live page (the finder's LevelMeter):
    fed by every block, the gate drawn on it — "am I being heard?" is
    answerable before and during a take."""
    from v2m.core import audio_io
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        meter = win.page_live.level_meter
        assert meter._gate_share > 0.02  # the gate IS drawn on the bar
        win._on_live_block(1.0, audio_io.GATE_RMS * 6.0)  # a sung-level block
        assert not meter._fill.isHidden()
        win._on_live_block(1.1, 0.0)  # silence: the fill drops
        assert meter._fill.isHidden()
    finally:
        win.close()


def test_piano_volume_rebuilds_player(qapp):
    """Moving the volume slider re-renders the piano at the new gain."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.result = AnalysisResult(
            source="live", duration_s=1.0,
            notes=[Note(0.0, 1.0, 60, 0.9)], tempo=None, key=None,
        )
        win._live_signal = np.zeros(audio_io.SR, np.float32)
        win.page_results.slider_volume.setValue(70)
        win.on_piano_volume_changed(70)
        assert win.player is not None and len(win.player.signal) > 0
        peak70 = float(np.max(np.abs(win.player.signal)))
        win.page_results.slider_volume.setValue(35)  # half gain
        win.on_piano_volume_changed(35)
        peak35 = float(np.max(np.abs(win.player.signal)))
        assert peak35 == pytest.approx(peak70 / 2, rel=0.1)
        # the voice button toggles against the stored live signal
        assert win._live_signal is not None
    finally:
        win.close()


def test_back_home_stops_the_audio(qapp, monkeypatch):
    """Going back home stops the playback: the page changes, the sound
    too. One owner (`on_back_home`) stops BEFORE navigating — the test
    forces a running playback then goes back home.
    """
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    _fake_sd_for_gui(monkeypatch)
    win = mw.MainWindow()
    try:
        win.result = AnalysisResult(
            source="live", duration_s=1.0,
            notes=[Note(0.0, 1.0, 60, 0.9)], tempo=None, key=None,
        )
        win._live_signal = np.zeros(audio_io.SR, np.float32)
        win.player = mw.NotePlayer(win.result.notes, sr=audio_io.SR)
        win.on_toggle_both()
        assert win.playback.playing, "the playback did not start"

        win.on_back_home()
        assert not win.playback.playing, "the audio keeps playing after going back home"
    finally:
        win.close()


def test_stopped_take_announces_the_dropped_blocks(qapp, monkeypatch):
    """A stop with dropped blocks shows the count on the status label."""
    from types import SimpleNamespace
    from typing import Any, cast

    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.live_thread = cast(
            Any,
            SimpleNamespace(
                segmenter=None,
                isRunning=lambda: False,
                take=SimpleNamespace(dropped_blocks=3),
            ),
        )
        win.on_stop_live()
        assert tr("ui dropped blocks").format(n=3) in win.page_live.label_status.text()
        assert win.live_thread is None  # the reference still goes away
    finally:
        win.close()


def test_reinterpret_button_rereads_evidence_without_pyin(qapp, monkeypatch):
    """↻ re-run the analysis: the interpretation alone, never pYIN or
    decoding (via the cached observation). The proof is the same as the
    script's: the heavy inputs are replaced by raising guards — if the
    handler goes through them, the test falls.
    """
    from v2m.core.notes import Note
    from v2m.core.observation import VoiceObservation
    from v2m.core.pipeline import AnalysisResult
    from v2m.core.pitch import PitchTrack
    from v2m.gui import main_window as mw

    # a small real observation: 3 held semitones, far enough for interpret
    hz = [440.0 * 2.0 ** ((m - 69) / 12.0) for m in (57, 60, 62)]
    times, curve = [], []
    t = 0.0
    for f in hz:
        for _ in range(15):
            times.append(t)
            curve.append(f)
            t += 0.02
    observation = VoiceObservation(
        pitch=PitchTrack(
            times=np.asarray(times),
            hz=np.asarray(curve),
            confidence=np.ones(len(times)),
            transition=np.zeros(len(times), dtype=bool),
        ),
        attacks=None,
        duration_s=t,
        sample_rate=audio_io.SR,
        frame_dt=0.02,
        accent_envelope=None,
    )

    win = mw.MainWindow()
    try:
        win.result = AnalysisResult(
            source="test.wav", duration_s=t,
            notes=[Note(0.0, 0.3, 57, 0.9)], tempo=None, key=None,
            observation=observation,
        )
        before = len(win.result.notes)
        before_result = win.result

        def _forbidden(name):
            raise AssertionError(f"the re-run button must not call {name}")

        import soundfile as sf

        from v2m.core import pitch as pitch_mod

        monkeypatch.setattr(pitch_mod, "track_pitch_offline",
                            lambda *a, **k: _forbidden("pYIN"))
        monkeypatch.setattr(sf, "read", lambda *a, **k: _forbidden("décodage"))

        win.page_results.spin_min_note.setValue(80)
        win.on_reinterpret()

        assert win.result is not None and win.result is not before_result
        assert len(win.result.notes) >= before  # a reading happened, views refilled
        assert win.result.observation is observation  # same evidence object
        assert win.result.source == "test.wav"  # identity preserved
    finally:
        win.close()


def test_reinterpret_button_errors_without_evidence(qapp):
    """Without a cached observation, the button warns instead of crashing."""
    from v2m.core.i18n import tr
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.result = AnalysisResult(
            source="x", duration_s=1.0, notes=[], tempo=None, key=None,
        )
        win.on_reinterpret()  # must show the banner, not raise
        # the warning is announced, not silently swallowed
        assert not win.error_banner.isHidden(), "the missing evidence must be announced"
        assert tr("err reinterpret none") in win.error_banner.text()
    finally:
        win.close()


def test_live_keyboard_marks_expected_pitch(qapp):
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw
    from v2m.gui.widgets import AMBER_COLOR, MATCH_COLOR, MISMATCH_COLOR

    win = mw.MainWindow()
    try:
        win.result = AnalysisResult(
            source="live", duration_s=1.0,
            notes=[Note(0.0, 1.0, 60, 0.9)], tempo=None, key=None,
        )
        win._live_t = 0.5
        win._on_live_note((60.0, 261.63))
        # dead-on pitch -> green (tuning feedback, not only the expected match)
        assert win.page_live.piano_live_keys._active_colors[60] == MATCH_COLOR

        win._on_live_note((62.0, 293.66))
        # wrong note: red however pure
        assert win.page_live.piano_live_keys._active_colors[62] == MISMATCH_COLOR
        # +20 cents on the right note: amber (NEAR), not a binary match
        win._on_live_note((60.2, 264.6))
        assert win.page_live.piano_live_keys._active_colors[60] == AMBER_COLOR
    finally:
        win.close()
