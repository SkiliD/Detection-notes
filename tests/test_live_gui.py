"""Bout en bout du mode live GUI : micro factice -> capture -> analyse -> export.

Régression : après l'arrêt d'un enregistrement live, l'analyse finale était
bien calculée mais la vue levait une exception (valeur de vérité d'un tableau
NumPy) AVANT l'export — le piano roll restait vide et aucun fichier n'était
écrit, sans message d'erreur.
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

MELODY_HZ = [220.0, 261.63, 293.66, 329.63]  # La3 Do4 Ré4 Mi4
TONE_S = 0.35
GAP_S = 0.12
CAPTURE_SR = 44_100


class FakeMic:
    """`sounddevice.InputStream` de remplacement : rejoue une mélodie connue."""

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
        assert captured, "le micro factice n'a rien livré à la capture live"
        live_thread = win.live_thread
        assert live_thread is not None

        # pendant la capture : les notes closes sont déjà sur le piano roll
        # et sur la portée qui défile
        seg = live_thread.segmenter
        assert seg is not None and seg.notes, "aucune note détectée pendant le live"
        assert win.piano_live._bar_items, "le piano roll live est resté vide"
        assert win.staff_live.layout().notes, "la portée live est restée vide"

        win.on_stop_live()
        assert _pump_until(qapp, lambda: win.result is not None, timeout=60), (
            "aucune analyse finale après l'arrêt du live"
        )

        result = win.result
        assert result is not None
        assert result.notes, "l'analyse finale du live est vide"
        assert win._live_signal is not None and len(win._live_signal) > 0
        assert win._voice_signal() is win._live_signal
        assert exported, "aucun export après l'enregistrement live"
        res, stem, _ = exported[-1]
        assert res is win.result
        assert Path(stem).name.startswith("live_"), f"nom de fichier inattendu : {stem}"
        assert Path(stem).parent.parent.name == "prises_live", (
            f"la prise doit vivre dans prises_live/<stem>/ : {stem}"
        )

        # la page résultats montre la prise, et rien n'a été signalé en rouge
        assert win.stack.currentWidget() is win.page_analyze
        assert len(win.piano._bar_items) == len(result.notes)
        assert win.piano_keys._spans
        staff_notes = win.staff.layout().notes
        assert len(staff_notes) == len(result.notes), "la portée ne suit pas l'analyse"
        assert all(1 <= n.measure for n in staff_notes)
        assert win.error_banner.isHidden(), win.error_banner.text()
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)


def test_live_grid_is_the_take_published_one(qapp, monkeypatch):
    """La GUI consomme la Grille live publiée par LiveTake : elle ne
    recule plus elle-même le BPM approché (l'ancien code reconstruisait un
    BeatGrid(estimate_bpm_live(...)) à chaque changement de compte de
    notes dans le tick)."""
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
        ), "la grille live publiée par LiveTake n'est jamais arrivée à la GUI"
        # le staff live grave la grille consommée (le staff lit grid.bpm —
        # toute dérive de la publication se verrait ici)
        assert _pump_until(
            qapp, lambda: win.staff_live._grid is win._live_grid, timeout=5
        ), "le staff live n'a pas reçu la grille publiée"
        # la mesure publiée arrive au label de statut, à côté du compte de temps
        assert win._live_grid.meter.label in win.label_live_status.text(), (
            f"mesure absente du statut live : {win.label_live_status.text()}"
        )
        # arrêt propre : la prise va au bout de son analyse pendant CE test,
        # sinon l'événement en file se livre pendant le test suivant
        win.on_stop_live()
        assert _pump_until(qapp, lambda: win.result is not None, timeout=60)
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)


def test_note_finder_lights_the_sung_key_and_keeps_nothing(qapp, monkeypatch):
    """La section « trouver la note » n'a qu'un rôle : allumer la touche
    chantée. Pas d'analyse finale, pas d'export, rien de conservé."""
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
        assert _pump_until(qapp, lambda: bool(win.finder_keys._active), timeout=60), (
            "aucune touche allumée par le chercheur de note"
        )
        # la touche allumée est celle de la mélodie factice (La3 Do4 Ré4 Mi4),
        # jamais un numéro : le clavier du piano est la seule réponse affichée
        assert win.finder_keys._active <= {57, 60, 62, 64}
        assert win.label_finder_note.text() not in ("", tr("ui finder idle"))
        assert win.finder_history.count() > 0
        assert "🎹" in win.finder_history.item(0).text()

        win.on_close_finder()
        assert win.stack.currentWidget() is win.page_home
        assert win.finder_thread is None
        assert win.finder_keys._active == set()
        assert win.label_finder_note.text() == tr("ui finder idle")
        assert not exported, "le chercheur de note ne doit rien exporter"
        assert win.result is None, "le chercheur de note ne doit rien analyser"
    finally:
        if win.finder_thread is not None and win.finder_thread.isRunning():
            win.finder_thread.stop()
            win.finder_thread.wait(3000)


def test_metronome_follows_detected_bpm(qapp):
    """The optional metronome tracks the take's detected tempo, not a fixed one."""
    from v2m.core.metronome import Metronome
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        # 4 onsets 0.5 s apart -> ~120 BPM detected
        notes = []
        for i in range(5):
            notes.append(type("N", (), {"start": i * 0.5})())
        win.chk_metronome.setChecked(True)  # builds the Metronome
        try:
            assert win.metronome is not None
            win._follow_detected_bpm(120.0)
            assert win.metronome.bpm == 120.0
            assert win.spin_bpm.value() == 120
            # the spin box is a silent follower: dragging it still works
            win.spin_bpm.setValue(100)
            assert win.metronome.bpm == 100.0
        finally:
            win.chk_metronome.setChecked(False)
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
        ), "le playhead ne suit pas la voix"
        assert win.btn_voice.text() == "■"
        win.stop_playback()
        assert win.playback.mode is None

        win.on_toggle_both()
        assert _pump_until(
            qapp,
            lambda: win.playback.playing and win.playback.mode == "both",
            5,
        ), "le mode combiné doit sonner les deux sources"
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
            "la durée doit rester exacte pour ne pas désynchroniser le piano"
        )
        assert not np.allclose(voice_player.signal, old_signal)
        assert win.playback.playing, "la transposition ne doit pas couper l'écoute"
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
        assert win.label_live_accuracy.text() != ""
        win._on_live_note((60.1, 262.0))
        win._on_live_note((60.2, 263.0))
        txt = win.label_live_accuracy.text()
        # mean of 10 and 20 cents on C4: any language, no history-dependence
        assert ("C4 / Do4" in txt or "Do4 / C4" in txt) and "15" in txt
        win.accuracy.reset()
        win._refresh_live_accuracy()
        assert win.label_live_accuracy.text() == tr("ui accuracy empty")
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


def test_live_keyboard_marks_expected_pitch(qapp):
    from v2m.core.notes import Note
    from v2m.core.i18n import tr
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
        assert win.piano_live_keys._active_colors[60] == MATCH_COLOR

        win._on_live_note((62.0, 293.66))
        # wrong note: red however pure
        assert win.piano_live_keys._active_colors[62] == MISMATCH_COLOR
        # +20 cents on the right note: amber (NEAR), not a binary match
        win._on_live_note((60.2, 264.6))
        assert win.piano_live_keys._active_colors[60] == AMBER_COLOR
    finally:
        win.close()
