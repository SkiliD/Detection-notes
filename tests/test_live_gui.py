"""End to end of the live GUI mode: fake mic -> capture -> analysis -> export.

Regression: after stopping a live recording, the final analysis WAS
computed but the view raised an exception (a NumPy array's truth value)
BEFORE the export — the piano roll stayed empty and no file was written,
with no error message.
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
        assert exported, "no export after the live recording"
        res, stem, _ = exported[-1]
        assert res is win.result
        assert Path(stem).name.startswith("live_"), f"unexpected file name : {stem}"
        assert Path(stem).parent.parent.name == "prises_live", (
            f"the take must live in prises_live/<stem>/ : {stem}"
        )

        # the results page shows the take, and nothing was flagged red
        assert win.stack.currentWidget() is win.page_analyze
        assert len(win.piano._bar_items) == len(result.notes)
        assert win.piano_keys._spans
        staff_notes = win.staff.layout().notes
        assert len(staff_notes) == len(result.notes), "the staff does not follow the analysis"
        assert all(1 <= n.measure for n in staff_notes)
        assert win.error_banner.isHidden(), win.error_banner.text()
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)


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
            qapp, lambda: win.waveform._line.value() > 0.0, 5
        ), "the playhead does not follow the voice"
        assert win.btn_voice.text() == "■"
        win.stop_playback()
        assert win.playback.mode is None

        win.on_toggle_both()
        assert _pump_until(
            qapp,
            lambda: win.playback.playing and win.playback.mode == "both",
            5,
        ), "the combined mode must sound both sources"
        win.stop_playback()
        assert win.btn_both.text() == "🎹+🎙"
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

        win.spin_transpose.setValue(2)  # audible live transpose
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
        win.accuracy.reset()
        win._refresh_live_accuracy()
        assert win.page_live.label_accuracy.text() == tr("ui accuracy empty")
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
        win.slider_volume.setValue(70)
        win.on_piano_volume_changed(70)
        assert win.player is not None and len(win.player.signal) > 0
        peak70 = float(np.max(np.abs(win.player.signal)))
        win.slider_volume.setValue(35)  # half gain
        win.on_piano_volume_changed(35)
        peak35 = float(np.max(np.abs(win.player.signal)))
        assert peak35 == pytest.approx(peak70 / 2, rel=0.1)
        # the voice button toggles against the stored live signal
        assert win._live_signal is not None
    finally:
        win.close()


def test_back_home_stops_the_audio(qapp, monkeypatch):
    """Going back home stops the playback: the page changes, the sound too.

    The 2026-09-26 report ("when you leave the page for home, the audio
    keeps playing"): navigation lambdas only changed the page. One owner
    (`on_back_home`) stops BEFORE navigating — the test forces a running
    playback then goes back home.
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


def test_reinterpret_button_rereads_evidence_without_pyin(qapp, monkeypatch):
    """↻ re-run the analysis: the interpretation alone, never pYIN or decoding.

    The 2026-09-26 feedback asked for a button on the analyze page that
    re-runs ONLY the interpretation (via the cached observation) after a
    threshold adjustment. The proof is the same as the script's: the
    heavy inputs are replaced by raising guards — if the handler goes
    through them, the test falls.
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

        win.spin_min_note.setValue(80)
        win.on_reinterpret()

        assert win.result is not None and win.result is not before_result
        assert len(win.result.notes) >= before  # a reading happened, views refilled
        assert win.result.observation is observation  # same evidence object
        assert win.result.source == "test.wav"  # identity preserved
    finally:
        win.close()


def test_reinterpret_button_errors_without_evidence(qapp):
    """Without a cached observation, the button warns instead of crashing."""
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.result = AnalysisResult(
            source="x", duration_s=1.0, notes=[], tempo=None, key=None,
        )
        win.on_reinterpret()  # must show the banner, not raise
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
