"""Calibration flow end-to-end: the fake mic SINGS, the card advances.

The pitches travel the REAL capture -> segmenter -> `pitchFrame` chain;
only the stream is fake. Pins the wiring: open the card, hold a note,
the round advances; the verdict reads the gates; save persists, forget
clears; noise never writes the file. (Core-level gates: test_mic_cal.)
Runs in-process like test_live_gui.py.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.slow


class HeldMic:
    """A fake `sd.InputStream` that SINGS one held tone at a time.

    `sing(hz, cents)` queues the next tone; silence in between, so a
    round advances only on a held note.
    """

    instances: list = []

    def __init__(self, samplerate, channels, blocksize, callback, **kwargs):  # noqa: ANN001
        self.rate = samplerate
        self.block = blocksize
        self.callback = callback
        self._tones: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        HeldMic.instances.append(self)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        quiet = np.zeros(self.block, np.float32).reshape(-1, 1)
        while not self._stop.is_set():
            if self._tones:
                hz, cents, seconds = self._tones.pop(0)
                self._emit(hz, cents, seconds)
            else:
                self.callback(quiet, self.block, None, None)
                time.sleep(0.005)

    def _emit(self, hz: float, cents: float, seconds: float) -> None:
        n = int(self.rate * seconds)
        t = np.arange(n) / self.rate
        f = hz * (2.0 ** (cents / 1200.0))
        tone = (0.35 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        for i in range(0, len(tone), self.block):
            if self._stop.is_set():
                return
            blk = tone[i : i + self.block]
            if len(blk) < self.block:
                blk = np.pad(blk, (0, self.block - len(blk)))
            self.callback(blk.reshape(-1, 1), self.block, None, None)
            time.sleep(0.005)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)

    def close(self) -> None:
        pass

    def sing(self, hz: float, cents: float = 0.0, seconds: float = 4.0) -> None:
        """Queue one held tone the collector will hear."""
        self._tones.append((hz, cents, seconds))


def _fake_sd(monkeypatch) -> None:
    """Patch `sd.InputStream` with the singing fake."""
    import sounddevice as sd

    HeldMic.instances = []
    monkeypatch.setattr(sd, "InputStream", HeldMic)


def _mic_of(win):
    """The fake stream the live take opened."""
    assert HeldMic.instances, "the fake mic never reached the capture"
    return HeldMic.instances[-1]


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _pump_until(app, predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def isolated_cal(tmp_path, monkeypatch):
    """The calibration file points at tmp."""
    from v2m.core import mic_cal

    cal = tmp_path / "cal.json"
    monkeypatch.setattr(mic_cal, "CAL_FILE", cal)
    yield cal


def test_calibration_flow_names_the_offset_and_persists(qapp, monkeypatch, isolated_cal):
    """A held, consistent voice names the device offset; saving persists it."""
    import sounddevice  # noqa: F401  (the patch target must be importable)

    from v2m.core import mic_cal
    from v2m.core import take_export as te
    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    OFFSET = -22.0  # a device reading ~22 cents flat
    exported: list[tuple] = []
    monkeypatch.setattr(te, "export_result", lambda res, stem, **kw: exported.append((res, stem)))
    _fake_sd(monkeypatch)

    win = mw.MainWindow()
    try:
        # the card is a HOME widget: show the window for real visibility
        win.show()
        # the collector rides a live capture
        win.on_start_live()
        live = win.live_thread
        assert live is not None
        assert _pump_until(
            qapp,
            lambda: HeldMic.instances and live.elapsed() >= 0.4,
            timeout=30,
        ), "the fake mic delivered nothing to the live capture"
        mic = _mic_of(win)

        win.stack.setCurrentWidget(win.page_home)
        win.page_home.btn_calibrate.click()  # opens the card + arms the collector
        assert win.page_home.calibration_card.isVisible()
        assert win._calibration_timer.isActive(), "an open card must arm the collector"

        card = win.page_home.calibration_card
        targets: list[float] = []
        for round_index in range(mic_cal.ROUNDS):
            assert card.prompt.text() == tr("ui calibrate target").format(
                n=round_index + 1, total=mic_cal.ROUNDS
            ), f"round {round_index + 1} does not name itself"
            # the test sings the card's own target
            target = card._targets[-1]
            targets.append(target)
            mic.sing(target, cents=OFFSET)
            assert _pump_until(
                qapp,
                lambda r=round_index: card._round == r + 1,
                timeout=30,
            ), f"round {round_index + 1} did not advance on the held note"

        # the verdict names the offset; saving opens up
        assert card.prompt.text() == tr("ui calibrate found").format(cents=OFFSET)
        assert card.btn_save.isVisible(), "a recordable verdict offers saving"
        assert mic_cal.read() is None, "nothing persists before the explicit save"

        card.btn_save.click()
        saved = mic_cal.read()
        assert saved is not None, "save must write the calibration file"
        assert saved["offset_cents"] == pytest.approx(OFFSET, abs=1.0)
        assert saved["n_used"] == mic_cal.ROUNDS
        assert saved["targets_hz"] == pytest.approx([round(t, 1) for t in targets], abs=0.1)
        assert card.btn_save.isHidden(), "the save action is spent once persisted"
        assert card.prompt.text() == tr("ui calibrate saved now")

        # the verdict stays a display fact (ADR 0032)
        assert card.verdict.text() == tr("ui calibrate context")

        # the opt-out clears the file
        card.btn_forget.click()
        assert not isolated_cal.exists(), "Forget must delete the calibration"
        assert card.prompt.text() == tr("ui calibrate forgotten")
        assert card.progress.text() == ""
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)
    assert not exported, "a calibration run must leave nothing exported"


def test_calibration_never_measures_the_singer(qapp, monkeypatch, isolated_cal):
    """A scattered run (flat/sharp alternating) offers NOTHING to save."""
    import sounddevice  # noqa: F401

    from v2m.core import mic_cal
    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    _fake_sd(monkeypatch)

    win = mw.MainWindow()
    try:
        win.show()
        win.on_start_live()
        live = win.live_thread
        assert live is not None
        assert _pump_until(
            qapp,
            lambda: HeldMic.instances and live.elapsed() >= 0.4,
            timeout=30,
        ), "the fake mic delivered nothing to the live capture"
        mic = _mic_of(win)

        win.stack.setCurrentWidget(win.page_home)
        win.page_home.btn_calibrate.click()
        card = win.page_home.calibration_card

        for round_index in range(mic_cal.ROUNDS):
            mic.sing(
                card._targets[-1],
                cents=(18.0 if round_index % 2 else -18.0),
            )
            assert _pump_until(
                qapp,
                lambda r=round_index: card._round == r + 1,
                timeout=30,
            ), f"scattered round {round_index + 1} did not advance"

        assert card.prompt.text() == tr("ui calibrate none"), (
            "an alternating voice names no device habit"
        )
        assert not card.btn_save.isVisible(), "noise must never offer a save"
        assert mic_cal.read() is None and not isolated_cal.exists(), (
            "a scattered run must not persist anything"
        )
        assert (
            tr("ui calibrate counts").format(used=mic_cal.ROUNDS, skipped=0) in card.progress.text()
        ), "every round landed on its target, so none is skipped"
    finally:
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)
