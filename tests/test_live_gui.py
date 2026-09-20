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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

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
            lambda: win.live_thread.elapsed() >= 1.8,
            timeout=60,
        )
        assert captured, "le micro factice n'a rien livré à la capture live"

        # pendant la capture : les notes closes sont déjà sur le piano roll
        # et sur la portée qui défile
        assert win.live_thread.seg.notes, "aucune note détectée pendant le live"
        assert win.piano_live._bar_items, "le piano roll live est resté vide"
        assert win.staff_live.layout().notes, "la portée live est restée vide"

        win.on_stop_live()
        assert _pump_until(qapp, lambda: win.result is not None, timeout=60), (
            "aucune analyse finale après l'arrêt du live"
        )

        assert win.result.notes, "l'analyse finale du live est vide"
        assert exported, "aucun export après l'enregistrement live"
        res, stem, _ = exported[-1]
        assert res is win.result
        assert str(stem).startswith("live_"), f"nom de fichier inattendu : {stem}"

        # la page résultats montre la prise, et rien n'a été signalé en rouge
        assert win.stack.currentWidget() is win.page_analyze
        assert len(win.piano._bar_items) == len(win.result.notes)
        assert win.piano_keys._spans
        staff_notes = win.staff.layout().notes
        assert len(staff_notes) == len(win.result.notes), "la portée ne suit pas l'analyse"
        assert all(1 <= n.measure for n in staff_notes)
        assert win.error_banner.isHidden(), win.error_banner.text()
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
