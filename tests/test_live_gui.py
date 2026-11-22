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
from v2m.core import take_export as te
from v2m.core.i18n import format_number

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
        te, "export_result", lambda res, stem, **kw: exported.append((res, stem, kw))
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

        # during capture: the page groups capture settings, monitoring and
        # feedback, while the roll and staff already show closed notes.
        from v2m.core.i18n import tr

        assert win.page_live.box_controls.title() == tr("ui live controls")
        assert win.page_live.box_monitor.title() == tr("ui live monitor")
        assert win.page_live.box_feedback.title() == tr("ui live feedback")
        page = win.page_live
        assert page.details_panel.isHidden(), "secondary singing details should start collapsed"
        assert not page.btn_stop_live.isHidden(), (
            "the finish action must remain in the singing view"
        )
        assert not page.piano_live.isHidden(), (
            "the roll that draws the take belongs to the singing view"
        )
        assert not page.box_monitor.isHidden() and not page.level_meter.isHidden()
        assert page.live_indicator.label_note.text() != "—", "the sung pitch was not shown"
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
        from v2m.core.i18n import tr, tr_key_label

        summary = win.page_results.label_summary.text()
        assert summary == tr("ui summary facts").format(
            notes=len(result.notes),
            duration=format_number(result.duration_s, 1),
            tempo=f"{format_number(result.tempo.bpm, 0)} BPM",
            key=tr_key_label(result.key.best.label) if result.key is not None else "—",
        )
        assert win.page_results.summary_card.title() == tr("ui take summary")
        assert win.page_results.progress.value() == 100
        assert win.page_results.details_scroll.isHidden()
        staff_notes = win.page_results.staff.layout().notes
        assert len(staff_notes) == len(result.notes), "the staff does not follow the analysis"
        assert all(1 <= n.measure for n in staff_notes)
        assert win.error_banner.isHidden(), win.error_banner.text()
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)
        win.close()


def test_phrase_replacement_stays_pending_until_explicit_export(qapp, monkeypatch, tmp_path):
    """A re-sung range replaces the selected phrase without auto-exporting."""
    import sounddevice as sd

    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    exported: list[tuple] = []

    def fake_export(res, stem, **kwargs):
        stem = Path(stem)
        exported.append((res, stem, kwargs))
        if stem.parent.is_relative_to(tmp_path):
            import json

            from v2m.core.report import ARTIFACT_EXTENSIONS

            stem.parent.mkdir(parents=True, exist_ok=True)
            notes = [{"midi": note.midi} for note in res.notes]
            for suffix in ARTIFACT_EXTENSIONS:
                content = json.dumps({"notes": notes}) if suffix == ".json" else "{}"
                Path(f"{stem}{suffix}").write_text(content, encoding="utf-8")
        return {}

    monkeypatch.setattr(te, "export_result", fake_export)
    monkeypatch.setattr(mw, "live_export_dir", lambda stem: tmp_path / Path(stem).name)

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
        automatic_result = win._automatic_live_result
        automatic_signal = win._automatic_live_signal
        assert original_audio is not None
        assert automatic_result is not None
        assert automatic_signal is not None
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
            lambda: (
                win._workflow.phase is mw.WorkflowPhase.REPLACEMENT_DECISION
                and win._signal_analysis_thread is None
                and win.result is not original
            ),
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
        assert win._workflow.phase is mw.WorkflowPhase.REPLACEMENT_DECISION
        assert not win.page_results.btn_accept_replacement.isHidden()
        assert not exported, "phrase replacement must not export implicitly"
        win._export_results()
        assert not exported, "export waits for the keep/cancel decision"

        win._on_accept_replacement()
        assert win._workflow.phase is mw.WorkflowPhase.IDLE
        assert win._edit_history.can_undo, "the kept splice is an undoable edit"
        assert win.result.manual_pitch_corrections
        assert win._automatic_live_result is automatic_result
        assert win._automatic_live_signal is automatic_signal
        win._export_results()
        assert len(exported) == 2
        assert exported[0][0] is automatic_result
        assert exported[1][0] is win.result
        assert exported[0][1] != exported[1][1]
        from v2m.core.takes import automatic_reading_dir, corrected_reading_dir

        take_dir = exported[1][1].parent
        automatic_json = automatic_reading_dir(take_dir) / f"{exported[1][1].name}.json"
        corrected_json = corrected_reading_dir(take_dir, 1) / f"{exported[1][1].name}.json"
        import json

        assert json.loads(automatic_json.read_text(encoding="utf-8"))["notes"] == [
            {"midi": note.midi} for note in automatic_result.notes
        ]
        assert json.loads(corrected_json.read_text(encoding="utf-8"))["notes"] == [
            {"midi": note.midi} for note in win.result.notes
        ]
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

    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    monkeypatch.setattr(
        te,
        "export_result",
        lambda res, stem, **kw: None,  # the decision waits
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
        before = win._edit_snapshot(action="replacement")
        assert before is not None
        context = win._workflow.begin_replacement(before, first, last, win._live_record_audio)
        assert context is not None
        assert win._workflow.begin_splice_analysis(context.token) is not None
        assert win._workflow.set_splice_window(context.token, window)
        win._on_replacement_analysis_done(spliced, merged_signal, context.token)
        assert win._workflow.phase is mw.WorkflowPhase.REPLACEMENT_DECISION

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
            context = win._workflow.begin_replacement(before, 1, 1, win._live_record_audio)
            assert context is not None
            assert win._workflow.begin_splice_analysis(context.token) is not None
            assert win._workflow.set_splice_window(context.token, (0.9, 1.4 + delta))
            win._on_replacement_analysis_done(spliced, merged, context.token)
            return merged

        before_state = win._edit_snapshot()
        assert before_state is not None
        merged = begin_decision()
        assert win._workflow.phase is mw.WorkflowPhase.REPLACEMENT_DECISION
        assert not win.page_results.btn_accept_replacement.isHidden()
        assert not win.page_results.btn_export.isEnabled()
        assert not win.page_results.btn_undo.isEnabled()
        assert not win.page_results.btn_redo.isEnabled()

        played: list[str] = []
        monkeypatch.setattr(win.playback, "play", lambda mode, t=0.0: played.append(mode) or True)
        win.on_audition_replacement()
        assert played == ["voice"]
        # the audition clip IS the replaced span of the merged audio
        assert len(win.playback.voice) == int(round(0.8 * sr))

        win._on_cancel_replacement()
        assert win._workflow.phase is mw.WorkflowPhase.IDLE
        assert len(win.result.notes) == 3  # the splice is gone
        assert win.result.notes[1].midi == 65  # the correction survived it
        assert len(win._live_signal) == 3 * sr  # the original audio is back
        assert win.page_results.btn_accept_replacement.isHidden()
        assert win.page_results.btn_undo.isEnabled()  # the correction is still undoable

        # -- keeping the splice turns it into an undoable edit -----------------
        begin_decision()
        win._on_accept_replacement()
        assert win._workflow.phase is mw.WorkflowPhase.IDLE
        assert win.result.manual_pitch_corrections
        assert win._automatic_live_result is not before_state.result
        assert [note.midi for note in win._automatic_live_result.notes] == [
            note.midi for note in before_state.result.notes
        ]
        assert win._automatic_live_signal is not before_state.signal
        assert np.array_equal(win._automatic_live_signal, before_state.signal)
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
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
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
            tr("correction"),
            tr("correction"),
            tr("current"),
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
        signal = np.linspace(-0.3, 0.3, 3 * sr, dtype=np.float32)
        win._live_signal = signal
        win._show_result(result)

        win.page_results.results._notes_table.selectRow(1)
        played: list[str] = []
        monkeypatch.setattr(win.playback, "play", lambda mode, t=0.0: played.append(mode) or True)
        win.on_audition_original()

        assert played == ["voice"]
        # the clip is notes[1]'s span (0.9 s -> 1.4 s) of the CURRENT audio
        expected = signal[int(round(0.9 * sr)) : int(round(1.4 * sr))]
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

        assert win.result is not result
        assert win.result.notes[0].midi == 62
        assert win.result.notes[1].midi == 64
        assert result.notes[0] is original and original.midi == 60
        assert (win.result.notes[0].start, win.result.notes[0].end) == (0.2, 0.7)
        assert win.result.pitch is None
        assert win.result.manual_pitch_corrections

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
        assert [n.midi for n in win.result.notes] == [60, 62]  # only row 1 moved
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
        assert label.text() == tr("ui selection label").format(
            n=1, t0=format_number(0.8, 1), t1=format_number(1.0, 1)
        )

        win.page_results.results._notes_table.selectAll()
        assert label.text() == tr("ui selection label").format(
            n=2, t0=format_number(0.2, 1), t1=format_number(1.0, 1)
        )

        win.page_results.results._notes_table.clearSelection()
        assert label.isHidden()

        win.page_results.results._notes_table.selectRow(0)
        before = label.text()
        win.on_toggle_language()
        assert label.text() == tr("ui selection label").format(
            n=1, t0=format_number(0.2, 1), t1=format_number(0.7, 1)
        )
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

        assert [n.midi for n in win.result.notes] == [60, 67]
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

        assert [n.midi for n in win.result.notes] == [60]  # untouched
        assert not win._edit_history.can_undo
        prefix = tr("err note deletion failed").split("{e}")[0]
        assert prefix in win.error_banner.text()  # the banner adds its ⚠ mark
    finally:
        win.close()


def test_the_status_bar_reports_a_deletion_and_a_correction(qapp):
    """Both note edits acknowledge themselves where the eye already is:
    the status bar names the repaired note, then the removed count."""
    from v2m.core.i18n import note_display_pair, tr
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
        duration_s=1.5,
        notes=[Note(0.1, 0.4, 60, 0.9), Note(0.5, 0.8, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    try:
        table = win.page_results.results._notes_table
        table.selectRow(0)
        win.page_results.piano_keys.key_clicked.emit(62)  # correct one note
        assert win.statusBar().currentMessage() == tr("ui status note corrected").format(
            note=note_display_pair(62)
        )

        table.selectRow(1)
        win.on_delete_selection()
        assert win.statusBar().currentMessage() == tr("ui status notes deleted").format(n=1)
    finally:
        win.close()


def test_results_page_final_render_sets_name_progress_and_signal(qapp, monkeypatch):
    """The final render owns its canonical label, completion and views."""
    from v2m.core.i18n import set_language, tr
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    page = win.page_results
    received = []
    monkeypatch.setattr(
        page.waveform,
        "set_signal",
        lambda signal, sr: received.append((signal, sr)),
    )
    file_result = AnalysisResult(
        source="/music/song.wav",
        duration_s=1.0,
        notes=[],
        tempo=None,
        key=None,
        pitch=None,
    )
    audio = np.linspace(-0.1, 0.1, 32, dtype=np.float32)
    page.set_progress(37)
    try:
        page.set_result(file_result, audio)
        assert page.label_file.text() == "song.wav"
        assert page.summary_card.title() == tr("ui take summary")
        assert page.label_summary.text() == tr("ui summary facts").format(
            notes=0, duration=format_number(1.0, 1), tempo="—", key="—"
        )
        assert page.details_toggle.text() == tr("ui show details")
        assert page.details_scroll.isHidden()
        assert page.progress.value() == 100

        set_language("en")
        page.retranslate()
        assert page.summary_card.title() == "Your transcription"
        assert page.label_summary.text() == "0 notes · 1.0 s · — · —"
        assert page.details_toggle.text() == "＋ Show details"
        page.details_toggle.setChecked(True)
        assert page.details_toggle.text() == "− Hide details"
        set_language("fr")
        page.retranslate()
        assert page.summary_card.title() == tr("ui take summary")
        assert page.details_toggle.text() == tr("ui hide details")
        assert received == [(audio, audio_io.SR)]

        page.details_toggle.setChecked(True)
        assert not page.details_scroll.isHidden()
        assert page.details_toggle.text() == tr("ui hide details")
        page.details_toggle.setChecked(False)
        assert page.details_scroll.isHidden()

        page.set_result(file_result, None)
        page.set_result(file_result, np.zeros(0, dtype=np.float32))
        assert received == [(audio, audio_io.SR)]  # absent audio preserves the waveform

        live_result = AnalysisResult(
            source="live",
            duration_s=1.0,
            notes=[],
            tempo=None,
            key=None,
            pitch=None,
        )
        live_audio = np.ones(16, dtype=np.float32)
        page.set_result(live_result, live_audio)
        assert page.label_file.text() == tr("ui live file")
        assert page.details_scroll.isHidden()
        assert page.details_toggle.text() == tr("ui show details")
        assert not page.btn_live_roll.isHidden()
        assert received[-1] == (live_audio, audio_io.SR)
    finally:
        win.close()


def test_show_result_resolves_audio_in_main_window_before_rendering(qapp, monkeypatch):
    """The window resolves explicit, cached-file and current-live audio."""
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    received = []
    monkeypatch.setattr(
        win.page_results.waveform,
        "set_signal",
        lambda signal, sr: received.append(signal),
    )
    file_result = AnalysisResult(
        source="/music/song.wav",
        duration_s=1.0,
        notes=[],
        tempo=None,
        key=None,
        pitch=None,
    )
    cached = np.full(8, 0.1, dtype=np.float32)
    explicit = np.full(8, 0.2, dtype=np.float32)
    live = np.full(8, 0.3, dtype=np.float32)
    try:
        win._signal_cache = {file_result.source: cached}
        win._show_result(file_result)
        win._show_result(file_result, explicit)
        assert received == [cached, explicit]

        # No cache or live signal means the old waveform is left alone.
        win._signal_cache = {}
        win._show_result(file_result)
        assert received == [cached, explicit]

        live_result = AnalysisResult(
            source="live",
            duration_s=1.0,
            notes=[],
            tempo=None,
            key=None,
            pitch=None,
        )
        win._live_signal = live
        win._show_result(live_result)
        win._show_result(live_result, explicit)
        assert received == [cached, explicit, live, explicit]
    finally:
        win.close()


def test_the_results_page_reaches_the_frozen_live_roll(qapp):
    """A live take offers a way back to its captured roll from the final
    analysis; a file take never does. Review is the page's own mode, and
    the way back to the results is one click."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    file_result = AnalysisResult(
        source="song.wav",
        duration_s=1.0,
        notes=[Note(0.2, 0.7, 60, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = file_result
    win._show_result(file_result)
    try:
        page = win.page_results
        assert page.btn_live_roll.isHidden()  # a file take has no captured roll

        live_result = AnalysisResult(
            source="live",
            duration_s=1.0,
            notes=[Note(0.2, 0.7, 60, 0.9)],
            tempo=None,
            key=None,
            pitch=None,
        )
        win.result = live_result
        win._show_result(live_result)
        assert not page.btn_live_roll.isHidden()

        win.on_show_live_roll()
        assert win.stack.currentWidget() is win.page_live
        assert win.page_live.review_active()
        assert not win.page_live.btn_review_back.isHidden()

        win.on_exit_live_roll()
        assert win.stack.currentWidget() is win.page_results
        assert not win.page_live.review_active()
        assert win.page_live.btn_review_back.isHidden()

        # stopping (no capture running here) leaves the review behind
        win.on_show_live_roll()
        assert win.page_live.review_active()
        win.on_stop_live()
        assert not win.page_live.review_active()
        assert win.page_live.btn_review_back.isHidden()
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
        staged: list = []
        monkeypatch.setattr(
            win.playback,
            "set_sources",
            lambda **kw: staged.append(kw.get("piano")),
        )
        monkeypatch.setattr(win.playback, "play", lambda mode: played.append(mode) or True)
        win.on_audition_key(60)
        first = staged[-1].copy()
        win.on_audition_key(64)
        second = staged[-1].copy()
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
    monkeypatch.setattr(te, "export_result", lambda res, stem, **kw: None)

    win = mw.MainWindow()
    try:
        win.on_start_live()

        def published_grid() -> bool:
            thr = win.live_thread
            seg = thr.segmenter if thr is not None else None
            return seg is not None and bool(seg.notes) and win._live_grid.bpm != 120.0

        assert _pump_until(qapp, published_grid, timeout=60), (
            "the live grid published by LiveTake never reached the GUI"
        )
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
        win.close()


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
    monkeypatch.setattr(te, "export_result", lambda res, stem, **kw: None)
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
        assert _pump_until(qapp, lambda: list(tmp_path.rglob("*.trace.json")), timeout=60), (
            "the failed take left no trace in its folder"
        )
        import json

        payload = json.loads(next(tmp_path.rglob("*.trace.json")).read_text(encoding="utf-8"))
        assert "too short" in payload["reason"]
        assert payload["exported"] is False
        # the failure reached the singer too: the status label says why
        assert win.page_live.label_status.text().strip()
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)
        win.close()


def test_note_finder_lights_the_sung_key_and_keeps_nothing(qapp, monkeypatch):
    """The "find the note" section has one role: light the sung key. No
    final analysis, no export, nothing kept."""
    import sounddevice as sd

    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    exported: list[tuple] = []
    monkeypatch.setattr(
        te, "export_result", lambda res, stem, **kw: exported.append((res, stem, kw))
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
        chips = win.page_finder.history_chips
        assert chips, "the confirmed note produced no history chip"
        assert chips[-1].text(), "the history chip carries no note name"

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
        win.close()


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
    result.notes = [Note(start=0.1, end=0.9, midi=60, confidence=0.9)]
    win.result = result
    sr = audio_io.SR
    win._live_signal = (0.4 * np.sin(2 * np.pi * 220 * np.arange(int(0.3 * sr)) / sr)).astype(
        np.float32
    )


def test_voice_and_both_follow_the_playhead(qapp, monkeypatch):
    """The Voice and Together actions drive the same playhead as Piano."""
    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    _fake_sd_for_gui(monkeypatch)
    win = mw.MainWindow()
    try:
        _synthetic_result(win)
        assert win.result is not None
        win.player = mw.NotePlayer(win.result.notes, sr=audio_io.SR)

        win.on_toggle_voice()
        assert _pump_until(qapp, lambda: win.page_results.waveform._line.value() > 0.0, 5), (
            "the playhead does not follow the voice"
        )
        assert win.page_results.btn_voice.text() == f"■ {tr('ui play voice')}"
        win.stop_playback()
        assert win.playback.mode is None

        win.on_toggle_both()
        assert _pump_until(
            qapp,
            lambda: win.playback.playing and win.playback.mode == "both",
            5,
        ), "the combined mode must sound both sources"
        win.stop_playback()
        assert win.page_results.btn_both.text() == f"▶ {tr('ui play both')}"
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
            source="live",
            duration_s=1.0,
            notes=[Note(0.0, 1.0, 60, 0.9)],
            tempo=None,
            key=None,
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
            source="live",
            duration_s=1.0,
            notes=[Note(0.0, 1.0, 60, 0.9)],
            tempo=None,
            key=None,
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


def test_dragging_a_bar_moves_the_note_as_one_recorded_edit(qapp):
    """The roll's drag lands through `on_note_moved` like a correction: ONE
    whole-state snapshot, the note changed, the announcement out, and one
    undo restores the take."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
        duration_s=2.0,
        notes=[Note(0.1, 0.5, 60, 0.9), Note(0.6, 1.0, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    try:
        win.page_results.piano.note_dragged.emit(1, 0.7, 1.1, 62)
        moved = win.result.notes[1]
        assert (moved.start, moved.end, moved.midi) == (0.7, 1.1, 62)
        assert "déplacée" in win.statusBar().currentMessage(), win.statusBar().currentMessage()
        assert win._edit_history.can_undo
        win.on_undo_edit()
        restored = win.result.notes[1]  # the undo lands a restored COPY
        assert (restored.start, restored.end, restored.midi) == (0.6, 1.0, 64)
    finally:
        win.close()


def test_a_drag_is_refused_where_no_edit_may_happen(qapp):
    """The shared edit guard answers NO off the results page: the emitted
    drag changes nothing and the bar snaps back (the roll already restored
    its own preview)."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
        duration_s=2.0,
        notes=[Note(0.1, 0.5, 60, 0.9), Note(0.6, 1.0, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    try:
        win.on_back_home()  # the edits live on the results page only
        win.page_results.piano.note_dragged.emit(1, 0.7, 1.1, 62)
        assert [(n.start, n.end, n.midi) for n in win.result.notes] == [
            (0.1, 0.5, 60),
            (0.6, 1.0, 64),
        ]
        assert not win._edit_history.can_undo
    finally:
        win.close()


def test_the_roll_drag_state_machine_previews_and_publishes_one_move(qapp):
    """The drag's mechanics: MOVE slides the whole bar, RESIZE (last
    quarter) drags its right edge, the pitch rides in whole semitones,
    the preview clamps to the neighbours, a no-op release emits nothing
    and the bar always lands back on the real note."""
    from v2m.core.notes import Note
    from v2m.gui.widgets import PianoRoll

    piano = PianoRoll(selectable=True)
    piano.set_editable(True)
    notes = [
        Note(0.1, 0.5, 60, 0.9),
        Note(0.6, 1.0, 64, 0.9),
        Note(1.1, 1.5, 67, 0.9),
    ]
    piano.set_notes(notes)
    captured: list[tuple] = []
    piano.note_dragged.connect(lambda i, s, e, m: captured.append((i, s, e, m)))
    try:
        # MOVE: the bar slides right by 0.1 s and up two semitones
        piano._begin_note_drag(1, 0.7, 64.0)
        piano._preview_note_drag(0.8, 66.4)
        assert piano._bar_items[1].opts["x0"][0] == pytest.approx(0.7)
        piano._end_note_drag(0.8, 66.4)
        assert captured == [(1, pytest.approx(0.7), pytest.approx(1.1), 66)]
        assert piano._bar_items[1].opts["x1"] == [1.0]  # snapped back on the truth

        # RESIZE: the grab sits in the bar's last quarter, the right edge
        # follows the DELTA (release at 1.0 = +0.05 over the grab)
        piano._begin_note_drag(1, 0.95, 64.0)  # 0.95 >= 0.6 + 0.75*0.4
        piano._end_note_drag(1.00, 64.0)
        assert captured[-1] == (1, pytest.approx(0.6), pytest.approx(1.05), 64)

        # the clamp: a move may not cross the next note's onset
        piano._begin_note_drag(1, 0.7, 64.0)
        piano._end_note_drag(1.4, 64.0)  # would need end 1.5 > next start 1.1
        assert captured[-1] == (1, pytest.approx(0.7), pytest.approx(1.1), 64)

        # a release with no movement publishes nothing
        piano._begin_note_drag(1, 0.7, 64.0)
        piano._end_note_drag(0.7, 64.0)
        assert len(captured) == 3
    finally:
        piano.deleteLater()


def test_ctrl_c_at_the_console_closes_the_window(qapp):
    """Ctrl+C never raises KeyboardInterrupt into a Qt slot (PyQt aborts the
    process on an unhandled slot exception): the installed handler routes
    SIGINT through the window's own close path."""
    import signal as signal_module

    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    previous = signal_module.getsignal(signal_module.SIGINT)
    closed = []
    win.close = lambda: closed.append(1) or True  # type: ignore[method-assign]
    try:
        mw._install_console_quit(qapp, win)
        handler = signal_module.getsignal(signal_module.SIGINT)
        assert handler is not None and handler is not signal_module.default_int_handler
        handler(signal_module.SIGINT, None)
        assert closed, "SIGINT did not reach the window's close path"
    finally:
        signal_module.signal(signal_module.SIGINT, previous)
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
            source="test.wav",
            duration_s=t,
            notes=[Note(0.0, 0.3, 57, 0.9)],
            tempo=None,
            key=None,
            observation=observation,
        )
        before = len(win.result.notes)
        before_result = win.result

        def _forbidden(name):
            raise AssertionError(f"the re-run button must not call {name}")

        import soundfile as sf

        from v2m.core import pitch as pitch_mod

        monkeypatch.setattr(pitch_mod, "track_pitch_offline", lambda *a, **k: _forbidden("pYIN"))
        monkeypatch.setattr(sf, "read", lambda *a, **k: _forbidden("décodage"))

        win.page_results.spin_min_note.setValue(80)
        win.on_reinterpret()

        assert win.result is not None and win.result is not before_result
        assert len(win.result.notes) >= before  # a reading happened, views refilled
        assert win.result.observation is observation  # same evidence object
        assert win.result.source == "test.wav"  # identity preserved
    finally:
        win.close()


def test_full_reanalysis_uses_retained_audio_and_publishes_only_on_success(qapp, monkeypatch):
    """The full action sends audio to pYIN and only then replaces the reading."""
    from v2m.core.i18n import tr
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    class SignalProbe:
        def __init__(self):
            self.slots = []

        def connect(self, slot):
            self.slots.append(slot)

        def emit(self, *args):
            for slot in tuple(self.slots):
                slot(*args)

    class FakeWorker:
        def __init__(self, signal, sr, source, opts):
            self.audio = signal.copy()
            self.sr = sr
            self.source = source
            self.opts = opts
            self.progress = SignalProbe()
            self.finishedOk = SignalProbe()
            self.failed = SignalProbe()
            self.finished = SignalProbe()
            self.running = False

        def start(self):
            self.running = True

        def isRunning(self):
            return self.running

        def succeed(self, result):
            self.finishedOk.emit(result)
            self.running = False
            self.finished.emit()

        def fail(self, message):
            self.failed.emit(message)
            self.running = False
            self.finished.emit()

    monkeypatch.setattr(mw, "FullReanalysisThread", FakeWorker)
    win = mw.MainWindow()
    audio = np.linspace(-0.2, 0.2, 512, dtype=np.float32)
    original = AnalysisResult(
        source="voice.wav",
        duration_s=1.0,
        notes=[Note(0.1, 0.5, 60, 0.9)],
        tempo=None,
        key=None,
    )
    win.result = original
    win._signal_cache = {original.source: audio}
    win._show_result(original, audio)
    try:
        win.page_results.spin_min_note.setValue(80)
        win.page_results.btn_full_reanalyze.click()
        worker = win._full_reanalysis_thread
        assert isinstance(worker, FakeWorker)
        assert worker.audio.dtype == np.float32
        assert np.array_equal(worker.audio, audio)
        assert worker.source == original.source
        assert worker.opts.min_note_s == pytest.approx(0.08)
        assert win.result is original  # the previous reading remains until success
        assert win.page_results._full_reanalysis_active
        assert not win.page_results.btn_export.isEnabled()

        worker.progress.emit(0.42, "Tracking pitch (pYIN)...")
        assert win.page_results.progress.value() == 42
        assert win.page_results.label_play_pos.text() == "Tracking pitch (pYIN)..."

        updated = AnalysisResult(
            source="worker-source",
            duration_s=1.0,
            notes=[Note(0.1, 0.3, 57, 0.9), Note(0.4, 0.7, 60, 0.9)],
            tempo=None,
            key=None,
        )
        worker.succeed(updated)
        assert win.result is updated
        assert win.result.source == original.source
        assert np.array_equal(win._signal_cache[original.source], audio)
        assert win.page_results.results._values["key notes count"].text() == "2"
        assert not win.page_results._full_reanalysis_active
        assert win.page_results.btn_export.isEnabled()
        assert win.statusBar().currentMessage() == tr("ui status full reanalyzed").format(n=2)

        # A duplicate completion carries the consumed worker token and must
        # not publish a second result or clear a later workflow's context.
        later = AnalysisResult(
            source="later.wav",
            duration_s=0.5,
            notes=[Note(0.1, 0.3, 62, 0.9)],
            tempo=None,
            key=None,
        )
        later_audio = np.full(64, 0.25, dtype=np.float32)
        win.result = later
        win._signal_cache = {later.source: later_audio}
        win._show_result(later, later_audio)
        worker.finishedOk.emit(updated)
        assert win.result is later
        assert win._signal_cache[later.source] is later_audio
    finally:
        win.close()


def test_full_reanalysis_failure_preserves_result_and_audio(qapp, monkeypatch):
    """A failed pYIN pass unlocks the view but retains the last valid take."""
    from v2m.core.i18n import tr
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    class SignalProbe:
        def __init__(self):
            self.slots = []

        def connect(self, slot):
            self.slots.append(slot)

        def emit(self, *args):
            for slot in tuple(self.slots):
                slot(*args)

    class FakeWorker:
        def __init__(self, signal, sr, source, opts):
            self.progress = SignalProbe()
            self.finishedOk = SignalProbe()
            self.failed = SignalProbe()
            self.finished = SignalProbe()
            self.running = False

        def start(self):
            self.running = True

        def isRunning(self):
            return self.running

        def fail(self, message):
            self.failed.emit(message)
            self.running = False
            self.finished.emit()

    monkeypatch.setattr(mw, "FullReanalysisThread", FakeWorker)
    win = mw.MainWindow()
    audio = np.ones(128, dtype=np.float32)
    original = AnalysisResult(
        source="voice.wav",
        duration_s=1.0,
        notes=[Note(0.1, 0.5, 60, 0.9)],
        tempo=None,
        key=None,
    )
    win.result = original
    win._signal_cache = {original.source: audio}
    win._show_result(original, audio)
    try:
        win.on_full_reanalyze()
        worker = win._full_reanalysis_thread
        assert isinstance(worker, FakeWorker)
        worker.fail("pYIN failed")

        assert win.result is original
        assert np.array_equal(win._signal_cache[original.source], audio)
        assert not win.page_results._full_reanalysis_active
        assert win.page_results.btn_export.isEnabled()
        assert tr("err full reanalysis failed").format(e="pYIN failed") in win.error_banner.text()
    finally:
        win.close()


def test_full_reanalysis_requires_available_audio(qapp):
    from v2m.core.i18n import tr
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.result = AnalysisResult(
            source="missing.wav",
            duration_s=1.0,
            notes=[],
            tempo=None,
            key=None,
        )
        win.on_full_reanalyze()
        assert tr("err full reanalysis audio") in win.error_banner.text()
        assert win._full_reanalysis_thread is None
    finally:
        win.close()


def test_abandoned_replacement_completion_cannot_replace_a_newer_reading(qapp):
    """A queued splice completion is inert after its decision was abandoned."""
    from v2m.core.edits import snapshot_state
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.core.tempo import TempoResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    sr = 8000
    original = AnalysisResult(
        source="live",
        duration_s=1.5,
        notes=[Note(0.1, 0.4, 60, 0.9), Note(0.7, 1.0, 64, 0.9)],
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 1.5, 0.5)),
        key=None,
        pitch=None,
        sr=sr,
    )
    original_audio = np.linspace(-0.2, 0.2, int(1.5 * sr), dtype=np.float32)
    win.result = original
    win._live_signal = original_audio
    win.take_ctl.set_signal(original_audio.tobytes())
    win._show_result(original, original_audio)
    active_token: int | None = None
    try:
        before = snapshot_state(original, original_audio, action="replacement")
        replacement = win._workflow.begin_replacement(before, 0, 0, False)
        assert replacement is not None
        assert win._workflow.begin_splice_analysis(replacement.token) is not None
        assert win._workflow.set_splice_window(replacement.token, (0.1, 0.4))
        stale_token = replacement.token

        win._abandon_pending_replacement()
        newer = AnalysisResult(
            source="new.wav",
            duration_s=0.5,
            notes=[Note(0.1, 0.3, 67, 0.9)],
            tempo=None,
            key=None,
            pitch=None,
            sr=sr,
        )
        newer_audio = np.full(int(0.5 * sr), 0.125, dtype=np.float32)
        win.result = newer
        win._live_signal = None
        win._signal_cache = {newer.source: newer_audio}
        win._show_result(newer, newer_audio)
        active = win._workflow.begin_full_reanalysis(newer, newer_audio)
        assert active is not None
        active_token = active.token
        history = win._edit_history.entries(snapshot_state(newer, newer_audio))

        stale_candidate = AnalysisResult(
            source="live",
            duration_s=1.0,
            notes=[Note(0.2, 0.8, 72, 0.9)],
            tempo=None,
            key=None,
            pitch=None,
            sr=sr,
        )
        win._on_replacement_analysis_done(
            stale_candidate, np.ones(sr, dtype=np.float32), stale_token
        )

        assert win.result is newer
        assert win._signal_cache == {newer.source: newer_audio}
        assert win._live_signal is None
        assert win._edit_history.entries(snapshot_state(newer, newer_audio)) == history
        assert win._workflow.is_current(active.token, mw.WorkflowPhase.FULL_REANALYSIS)
        assert win.page_results.results._values["key notes count"].text() == "1"
    finally:
        win._workflow.abandon_replacement()
        # The synthetic full-reanalysis operation has no thread to finish.
        if active_token is not None:
            win._workflow.fail_full_reanalysis(active_token)
        win.close()


def test_results_control_column_scrolls_at_compact_window_height(qapp):
    """The diagnostics and lower actions stay reachable without enlarging."""
    from PyQt6.QtCore import QRect

    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = AnalysisResult(
        source="voice.wav",
        duration_s=4.0,
        notes=[Note(i * 0.4, i * 0.4 + 0.25, 60 + i % 5, 0.9) for i in range(8)],
        tempo=None,
        key=None,
    )
    win.result = result
    win._show_result(result, np.ones(4 * audio_io.SR, dtype=np.float32))
    try:
        win.resize(1100, 420)
        win.show()
        win.page_results.details_toggle.setChecked(True)
        qapp.processEvents()
        page = win.page_results
        assert not page.details_scroll.isHidden()
        assert page.right_scroll.verticalScrollBar().maximum() > 0

        page.right_scroll.verticalScrollBar().setValue(
            page.right_scroll.verticalScrollBar().maximum()
        )
        qapp.processEvents()
        button_rect = QRect(
            page.btn_open.mapTo(page.right_scroll.viewport(), page.btn_open.rect().topLeft()),
            page.btn_open.size(),
        )
        assert page.right_scroll.viewport().rect().intersects(button_rect)
        assert button_rect.bottom() <= page.right_scroll.viewport().rect().bottom()
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
            source="x",
            duration_s=1.0,
            notes=[],
            tempo=None,
            key=None,
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
            source="live",
            duration_s=1.0,
            notes=[Note(0.0, 1.0, 60, 0.9)],
            tempo=None,
            key=None,
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


def test_live_intonation_feedback_fills_as_notes_close(qapp):
    """The take-so-far intonation line: the report's words on the live page.

    Fed by CLOSED live notes only (the core's `note_closed` event), the
    share and the tendency arrive when the evidence carries them — and
    the wording is the per-take report's own (one rendering, two views).
    """
    from v2m.core.i18n import tr
    from v2m.core.liveseg import LiveNote
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        assert win.page_live.label_intonation.text() == ""
        for i, d in enumerate((-0.25,) * 3):
            win.page_live.on_live_note_closed(
                LiveNote(start=i * 0.4, end=i * 0.4 + 0.3, midi=60, dev=d)
            )
        # 3 notes: the share is there, the tendency is not yet
        txt = win.page_live.label_intonation.text()
        assert tr("ui intonation share").format(share=0) in txt, txt
        assert "tendance" not in txt and "tendency" not in txt, txt
        win.page_live.on_live_note_closed(LiveNote(1.2, 1.5, 60, dev=-0.25))
        # 4 flat notes: the report's tendency line appears
        txt = win.page_live.label_intonation.text()
        expected = tr("ui intonation tendency").format(
            dir=tr("ui intonation flat").format(cents=-25.0)
        )
        assert expected in txt, txt

        # the worker's event reaches the page through its Qt signal
        from v2m.core.live_session import LiveEvent

        live_thread = mw.LiveThread()
        live_thread.liveNoteClosed.connect(win.page_live.on_live_note_closed)
        live_thread._on_take_event(
            LiveEvent(kind="note_closed", note=LiveNote(1.6, 1.9, 60, dev=-0.25))
        )
        summary = win.page_live.intonation.summary()
        assert summary["n_notes"] == 5
    finally:
        win.close()


def test_live_intonation_feedback_follows_the_language_switch(qapp):
    """A hot-switch mid-take re-labels the feedback line (mirror rule)."""
    from v2m.core.i18n import set_language, tr
    from v2m.core.liveseg import LiveNote
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        set_language("fr")
        for i in range(4):
            win.page_live.on_live_note_closed(
                LiveNote(start=i * 0.4, end=i * 0.4 + 0.3, midi=60, dev=-0.25)
            )
        assert "en dessous" in win.page_live.label_intonation.text()
        set_language("en")
        win.page_live.retranslate()
        assert "below" in win.page_live.label_intonation.text()
        assert tr("ui intonation even") == "steady"  # sanity: en table active
    finally:
        set_language("fr")
        win.close()


def test_releasing_a_drag_auditions_the_note_before_then_after(qapp, monkeypatch):
    """The drag's companion: the OLD note sounds, then the moved one.

    Both clips come from the piano engine at the take's own gain; the
    first clip's completion (the 33 ms tick's clock) plays the second,
    and the second's completion leaves the fact silent.
    """
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
        duration_s=2.0,
        notes=[Note(0.1, 0.5, 60, 0.9), Note(0.6, 1.0, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    try:
        assert win._auditions.active_identity is None
        played: list[str] = []
        monkeypatch.setattr(win.playback, "play", lambda mode, t=0.0: played.append(mode) or True)
        win.page_results.piano.note_dragged.emit(1, 0.7, 1.1, 62)

        moved = win.result.notes[1]
        assert (moved.start, moved.end, moved.midi) == (0.7, 1.1, 62)
        # the controller owns the sequence: the drag identity is active
        assert win._auditions.active_kind == "drag"
        # the FIRST clip is the OLD note's render, and it is playing
        assert played == ["piano"]
        assert win.playback.playing is False  # play() is stubbed here
        assert len(win.playback.piano) > 0
        old = Note(0.6, 1.0, 64, 0.9)
        assert np.array_equal(
            win.playback.piano,
            win._rendered_note(old),
        )

        # the first clip finishes -> the tick plays the moved note
        win.playback.stop()
        win._on_play_tick()
        assert win._auditions.active_kind == "drag"  # the second clip runs
        assert played == ["piano", "piano"]
        assert np.array_equal(
            win.playback.piano,
            win._rendered_note(moved),
        )

        # the second clip finishes -> the state is spent, nothing restarts
        win.playback.stop()
        win._on_play_tick()
        assert win._auditions.active_identity is None
        assert played == ["piano", "piano"]
    finally:
        win.close()


def test_a_stop_or_another_playback_replaces_the_move_audition(qapp, monkeypatch):
    """One owner of what is sounding: a stop or another start resets it."""
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    result = AnalysisResult(
        source="song.wav",
        duration_s=2.0,
        notes=[Note(0.1, 0.5, 60, 0.9), Note(0.6, 1.0, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    try:
        monkeypatch.setattr(win.playback, "play", lambda mode, t=0.0: True)
        win.page_results.piano.note_dragged.emit(1, 0.7, 1.1, 62)
        assert win._auditions.active_kind == "drag"

        # an explicit stop kills the armed sequence before the second clip
        win.stop_playback()
        assert win._auditions.active_identity is None

        # ...and a second audition of the same clip is also spent: the tick
        # after an explicit stop must NOT resurrect the sequence
        win.page_results.piano.note_dragged.emit(1, 0.7, 1.1, 62)
        assert win._auditions.active_kind == "drag"
        win.stop_playback()
        win._on_play_tick()  # a late tick arriving after the stop
        assert win._auditions.active_identity is None

        # another playback start (the whole take) replaces the sequence
        win.page_results.piano.note_dragged.emit(1, 0.7, 1.1, 62)
        win.on_toggle_play()
        assert win._auditions.active_identity is None  # _start() stopped it first
    finally:
        win.close()


def test_live_intonation_feedback_carries_the_register_leans(qapp):
    """The live line ends with the register verdicts, the report's rows.

    The same 12-note pattern the results-panel test uses: each band holds
    MIN_NOTES notes, so mid and high leans are announced while the low
    band (a 2.5 c mean) stays silent — through the report's own painter,
    so the two views cannot drift.
    """
    from v2m.core.liveseg import LiveNote
    from v2m.gui import main_window as mw

    devs = (0.02, 0.03, 0.02, 0.03, 0.05, 0.10, 0.15, 0.25, 0.28, 0.30, 0.30, 0.28)
    win = mw.MainWindow()
    try:
        for i, d in enumerate(devs):
            win.page_live.on_live_note_closed(
                LiveNote(start=i * 0.4, end=i * 0.4 + 0.3, midi=55 + i, dev=d)
            )
        txt = win.page_live.label_intonation.text()
        assert "registres" in txt, txt
        assert "médiums" in txt and "+14 c" in txt, txt
        assert "aigus" in txt and "+29 c" in txt, txt
        assert "graves" not in txt, txt  # the low band's mean is 2.5 c: silent
    finally:
        win.close()
