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
import types

import numpy as np
import pytest

from v2m.core import audio_io
from v2m.core.i18n import tr

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


class DeafMic:
    """A fake `sd.InputStream` whose start() fails (no device, busy...)."""

    def __init__(self, samplerate, channels, blocksize, callback, **kwargs):  # noqa: ANN001
        self.callback = callback

    def start(self) -> None:
        raise RuntimeError("device unavailable")

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def _note_of(hz: float) -> str:
    """The asked note the way the card names it."""
    from v2m.core.i18n import note_display
    from v2m.core.music import hz_to_midi

    return note_display(int(round(hz_to_midi(hz))))


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
            # the test sings the card's own target, which the card READS OUT
            target = card._targets[-1]
            targets.append(target)
            assert card.prompt.text() == tr("ui calibrate target").format(
                n=round_index + 1, total=mic_cal.ROUNDS, note=_note_of(target)
            ), f"round {round_index + 1} does not name its note"
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


def test_the_card_opens_its_own_mic_without_a_take(qapp, monkeypatch, isolated_cal):
    """The card opens the microphone ITSELF, and releases it on close.

    Only the record action started a capture, so a home-screen user who
    opened the card WITHOUT recording got a frozen prompt and read it as
    "the app cannot find my mic": the rounds advanced only while a live
    take happened to run, and nothing on screen said which of the two was
    missing. Opening the card must open a capture; closing it (the card's
    own Fermer) must give the device back.
    """
    import sounddevice  # noqa: F401  (the patch target must be importable)

    from v2m.core import mic_cal
    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    OFFSET = -18.0
    _fake_sd(monkeypatch)

    win = mw.MainWindow()
    try:
        win.show()
        win.stack.setCurrentWidget(win.page_home)
        # NO live take: the card is the only thing asking for the mic
        win.page_home.btn_calibrate.click()
        card = win.page_home.calibration_card
        assert card.isVisible()

        assert _pump_until(
            qapp,
            lambda: bool(HeldMic.instances) and win._calibration_thread is not None,
            timeout=30,
        ), "the card never opened a capture of its own"
        live = win._calibration_thread
        assert live is not None and live.isRunning()
        assert _pump_until(qapp, lambda: live.elapsed() >= 0.4, timeout=30), (
            "the card's capture delivered nothing"
        )
        mic = _mic_of(win)

        for round_index in range(mic_cal.ROUNDS):
            mic.sing(card._targets[-1], cents=OFFSET)
            assert _pump_until(qapp, lambda r=round_index: card._round == r + 1, timeout=30), (
                f"round {round_index + 1} did not advance without a take"
            )

        assert card.prompt.text() == tr("ui calibrate found").format(cents=OFFSET)
        assert card.btn_save.isVisible()
        # the card SHOWED what it heard (a dead mic and a wrong note must
        # not look the same)
        assert card.heard_line.isVisible()

        # closing the card releases the device and stops the tick
        card.btn_stop.click()
        assert not card.isVisible()
        assert win._calibration_thread is None, "the card's capture must be dropped"
        assert not win._calibration_timer.isActive()
        assert not live.isRunning()
    finally:
        if win._calibration_thread is not None and win._calibration_thread.isRunning():
            win._calibration_thread.stop()
            win._calibration_thread.wait(3000)
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)


def test_saved_input_selection_is_shared_by_capture_flows(qapp, monkeypatch, tmp_path):
    """Finder, calibration and live takes resolve the same persisted endpoint."""
    from v2m.core import i18n
    from v2m.gui import main_window as mw

    preference = tmp_path / "input-device.json"
    monkeypatch.setattr(mw, "INPUT_DEVICE_FILE", preference)
    monkeypatch.setattr(i18n, "LANG_FILE", tmp_path / "lang")
    monkeypatch.setattr(
        audio_io,
        "input_devices",
        lambda: [
            audio_io.InputDevice(4, "WASAPI\x1fUSB Mic", "USB Mic", "WASAPI"),
            audio_io.InputDevice(5, "WASAPI\x1fBuilt-in", "Built-in", "WASAPI"),
        ],
    )
    HeldMic.instances = []
    monkeypatch.setattr(
        audio_io,
        "sd",
        types.SimpleNamespace(
            InputStream=HeldMic,
            default=types.SimpleNamespace(device=(0, -1)),
        ),
    )

    win = mw.MainWindow()
    try:
        win.show()
        win.page_home._on_input_device_selected(
            win.page_home.input_device.findData("WASAPI\x1fUSB Mic")
        )
        assert preference.read_text(encoding="utf-8") == '"WASAPI\\u001fUSB Mic"'
        thread = win._new_input_thread()
        assert thread.device == 4
        assert thread.device_name == "USB Mic — WASAPI"

        win.page_home.btn_calibrate.click()
        assert win._calibration_thread is not None
        assert win._calibration_thread.device == 4
        assert win._calibration_timer.isActive()
        assert "USB Mic" not in win.page_live.label_input_device.text()
        assert win.page_home.calibration_card.prompt.text().startswith("Ronde 1/")
        from v2m.core.i18n import set_language

        set_language("en")
        win.retranslate()
        assert win.page_home.calibration_card.prompt.text().startswith("Round 1/")
        assert "Next capture: USB Mic — WASAPI" in win.page_home.input_device_status.text()
        assert "USB Mic" not in win.page_live.label_input_device.text()
        set_language("fr")
        win.retranslate()
        win.on_start_live()
        assert win.page_live.label_input_device.text() == tr("ui input device active").format(
            device="USB Mic — WASAPI"
        )
        assert win._calibration_thread is None
        assert not win._calibration_timer.isActive()
        assert win.page_home.calibration_card.isHidden()
        assert win.live_thread is not None and win.live_thread.device == 4

        # The active take keeps naming its open endpoint; a changed choice is
        # shown on Home as the endpoint that the next capture will open.
        win.page_home._on_input_device_selected(
            win.page_home.input_device.findData("WASAPI\x1fBuilt-in")
        )
        assert win.page_live.label_input_device.text() == tr("ui input device active").format(
            device="USB Mic — WASAPI"
        )
        assert win.page_home.input_device_status.text() == tr("ui input device next").format(
            device="Built-in — WASAPI"
        )
        win.live_thread.stop()
        win.live_thread.wait(3000)

        win.on_open_finder()
        assert win.page_finder.thread is not None
        assert win.page_finder.thread.device == 5
        win.on_close_finder()

        win.page_home.btn_calibrate.click()
        assert win._calibration_thread is not None
        assert win._calibration_thread.device == 5
        win.page_home.calibration_card.reset()
        assert win._calibration_thread is None
    finally:
        if win._calibration_thread is not None and win._calibration_thread.isRunning():
            win._calibration_thread.stop()
            win._calibration_thread.wait(3000)
        if win.page_finder.thread is not None and win.page_finder.thread.isRunning():
            win.page_finder.thread.stop()
            win.page_finder.thread.wait(3000)
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)
        win.close()


def test_a_dead_mic_is_said_on_the_card(qapp, monkeypatch, isolated_cal):
    """A capture that never opens must SAY so (never a frozen prompt)."""
    import sounddevice as sd

    from v2m.core import mic_cal
    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", DeafMic)

    win = mw.MainWindow()
    try:
        win.show()
        win.stack.setCurrentWidget(win.page_home)
        win.page_home.btn_calibrate.click()
        card = win.page_home.calibration_card

        assert _pump_until(qapp, lambda: card.error_line.isVisible(), timeout=30), (
            "a microphone that never opened stayed invisible"
        )
        # The card names the cause the way the page does — the driver's own
        # "device unavailable" is the sentence for an endpoint another
        # application holds, not a transport string on screen.
        assert card.error_line.text() == tr("ui mic busy")
        assert card.prompt.text() == tr("ui calibrate target").format(
            n=1, total=mic_cal.ROUNDS, note=_note_of(card._targets[-1])
        )
    finally:
        if win._calibration_thread is not None and win._calibration_thread.isRunning():
            win._calibration_thread.stop()
            win._calibration_thread.wait(3000)


def test_a_stale_hold_never_advances_a_round(qapp, monkeypatch, isolated_cal):
    """A round counts a pitch only while the note still sounds.

    The collector's 400 ms tick used the LAST pitch seen: a note the
    singer had already released (or a passing glide) stayed in the slot
    and could be graded — "a sustained pitch only" was a comment, not a
    rule. The pitch now carries its arrival time.
    """
    import time

    from v2m.core.i18n import tr
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.show()
        win.stack.setCurrentWidget(win.page_home)
        win.page_home.btn_calibrate.click()
        card = win.page_home.calibration_card
        assert card._round == 0

        # a pitch the singer holds: it arrives fresh
        win._calibration_hz = card._targets[-1]
        win._calibration_at = time.monotonic()
        win._calibration_feed_tick()
        assert card._round == 1, "a held note must advance its round"

        # ...and one that ARRIVED long ago: the note is over, no round
        win._calibration_hz = card._targets[-1]
        win._calibration_at = time.monotonic() - (mw.MainWindow.CALIBRATION_FRESH_S + 1.0)
        win._calibration_feed_tick()
        assert card._round == 1, "a released note must not be graded"
        assert card.heard_line.text() == tr("ui calibrate silent"), (
            "a released note must not be reported as heard"
        )
    finally:
        if win._calibration_thread is not None and win._calibration_thread.isRunning():
            win._calibration_thread.stop()
            win._calibration_thread.wait(3000)
        if win.live_thread is not None and win.live_thread.isRunning():
            win.live_thread.stop()
            win.live_thread.wait(3000)


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
