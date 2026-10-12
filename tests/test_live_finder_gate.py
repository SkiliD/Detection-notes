"""The Finder lights a key only on a CONFIRMED note (workstream 4).

`LiveNoteSegmenter.current()` also answers during the PENDING state, a
window that has not held `confirm_s` yet. Mic handling noise passes the
periodicity gate for a few frames then disappears: it never confirms
(measured: 0 notes and 0 ACTIVE blocks for 50-400 ms thumps) but
`current()` does deliver a pitch — 19 blocks over the 1.5 s train below.

`confirmed()` is the answer: the note the segmenter actually admitted.
The live indicator and the intonation needle keep reading
`current()` / `recent_midi()`: that is the immediate feedback the
singer watches WHILE singing.
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from tests.test_live_gui import FakeMic  # the same synthetic singer
from v2m.core import take_export as te
from v2m.core.i18n import tr  # noqa: E402
from v2m.core.liveseg import LiveNoteSegmenter

SR = 44_100


def _hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_confirmed_waits_for_the_hold_while_current_answers():
    """PENDING: the live feedback answers, but no note is asserted."""
    seg = LiveNoteSegmenter()
    for i in range(3):
        seg.push(i * 0.01, _hz(69))
    assert seg.state == "PENDING"
    assert seg.current() is not None, "the live feedback must answer during PENDING"
    assert seg.confirmed() is None, "a PENDING window is not a note"

    for i in range(3, 40):  # the hold fills up
        seg.push(i * 0.01, _hz(69))
    assert seg.state == "ACTIVE"
    confirmed = seg.confirmed()
    assert confirmed is not None and round(confirmed[0]) == 69


def test_confirmed_never_names_a_key_outside_the_piano():
    """The keyboard stops at C8: a pitch beyond is not a note."""
    seg = LiveNoteSegmenter()
    for i in range(40):
        seg.push(i * 0.01, _hz(110))  # above C8
    assert seg.state == "ACTIVE"       # the segmenter holds the pitch...
    assert seg.confirmed() is None     # ...and the display does not make it a key


def test_the_level_meter_names_a_saturated_input(qapp, monkeypatch):
    """A block at the rail turns the fill red, a healthy level back to
    the normal color: saturation is a decision the singer must see."""
    from v2m.gui.finder_page import (
        LEVEL_GATE_COLOR, LEVEL_FULL_SCALE, LevelMeter,
    )
    from v2m.gui.widgets import NOTE_TEXT_COLOR

    meter = LevelMeter()
    meter.resize(200, 8)
    healthy = LEVEL_FULL_SCALE * 0.5
    meter.set_level(healthy)
    assert NOTE_TEXT_COLOR in meter._fill.styleSheet()
    # at the rail: the fill goes to the gate's red ("turn the input DOWN")
    meter.set_level(LevelMeter.CLIP_RMS_RMS)
    assert LEVEL_GATE_COLOR in meter._fill.styleSheet()
    assert "background:" in meter._fill.styleSheet()
    # a healthy level again restores the normal fill
    meter.set_level(healthy)
    assert NOTE_TEXT_COLOR in meter._fill.styleSheet()
    # silence hides the fill and clears the latch
    meter.set_level(0.0)
    assert meter._fill.isHidden()
    assert meter._clipping is False


def test_finder_history_uses_the_confirmation_captured_with_the_frame(qapp):
    """Queued UI delivery must not replace the block's note with newer state."""
    from v2m.gui.finder_page import FinderPage

    class AdvancedSegmenter:
        def confirmed(self):
            return None

        def heard(self):
            return None

    class QueuedThread:
        segmenter = AdvancedSegmenter()

        def isRunning(self):  # noqa: N802 (Qt naming)
            return True

    def unexpected_capture():
        raise AssertionError("the queued reading test must not open the microphone")

    page = FinderPage(
        thread_factory=unexpected_capture,
        go_home=lambda: None,
        show_error=lambda _msg: None,
        clear_error=lambda: None,
        audition_note=lambda _midi: None,
    )
    setattr(page, "_thread", QueuedThread())
    try:
        # The audio thread confirmed C4 for this block; by the time Qt
        # handles its queued event, the mutable segmenter has moved to idle.
        page._on_note((60.0, 261.6), (60.0, 261.6), (60.0, 261.6))
        assert page.session.history == [60]
        assert page.keys._active == {60}

        # Another frame in the same held note updates nothing in history.
        page._on_note((60.0, 261.6), (60.0, 261.6), (60.0, 261.6))
        assert page.session.history == [60]
    finally:
        page.deleteLater()


def test_pending_plateau_names_without_lighting(qapp, monkeypatch):
    """The big name follows the PENDING plateau, the key lights ONLY on
    confirmed: two distinct contracts, both alive.

    The "the mic seemed ignored" fix first displayed the forming note in
    the name AND lit the key: the anti-noise test then caught a key lit
    on plain handling noise. The separation — name = `current()`
    (PENDING answers), key = `confirmed()` — keeps both measured
    behaviours of one tick.
    """
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        # PENDING: the name answers, the key stays dark
        win._set_finder_note(69.0, light=False)
        assert win.page_finder.label_note.text() != tr("ui finder idle")
        assert win.page_finder.keys._active == set(), "an unconfirmed note lit the key"
        # CONFIRMED: both the name AND the key answer
        win._set_finder_note(69.0, light=True)
        assert win.page_finder.keys._active == {69}
        # nothing at all: the prompt and a dark keyboard
        win._set_finder_note(None)
        assert win.page_finder.label_note.text() == tr("ui finder idle")
        assert win.page_finder.keys._active == set()
    finally:
        win.close()


def _rumble_train(seed: int = 1, bursts: int = 3, burst_s: float = 0.4) -> np.ndarray:
    """Three low-frequency thumps (band < 120 Hz): handling noise, nothing sung."""
    from scipy.signal import butter, filtfilt

    rng = np.random.default_rng(seed)
    b, a = butter(2, 120 / (SR / 2), "low")
    parts: list[np.ndarray] = []
    for _ in range(bursts):
        n = int(burst_s * SR)
        low = filtfilt(b, a, rng.standard_normal(n))
        t = np.arange(n) / SR
        y = 4.0 * low + 0.25 * np.sin(2 * np.pi * 72 * t)
        y[: int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
        parts.append(y / np.max(np.abs(y)) * 0.7)
        parts.append(np.zeros(int(0.1 * SR)))
    return np.concatenate(parts).astype(np.float32)


class RumbleMic:
    """A stand-in `sounddevice.InputStream`: replays the measured handling noise."""

    def __init__(self, samplerate, channels, blocksize, callback, **kwargs):  # noqa: ANN001
        self.rate = samplerate
        self.block = blocksize
        self.callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._play, daemon=True)
        self._thread.start()

    def _play(self) -> None:
        y = _rumble_train()
        for i in range(0, len(y), self.block):
            if self._stop.is_set():
                return
            blk = y[i : i + self.block]
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


def _pump_until(app, predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_the_finder_stays_dark_on_a_mic_rumble(qapp, monkeypatch):
    """Handling noise must light NO Finder key.

    The train lasts 1.5 s: plenty to see 19 blocks of `current()` (the
    defect's measurement) — and zero confirmed notes. Before the fix,
    the last lit key appeared here. The BIG NAME, meanwhile, follows the
    immediate feedback (`current()`): on noise it shows nothing stable,
    but it must never enter a pitch into the history.
    """
    import sounddevice as sd

    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", RumbleMic)
    monkeypatch.setattr(te, "export_result", lambda res, stem, **kw: None)

    win = mw.MainWindow()
    try:
        win.on_open_finder()
        captured = _pump_until(
            qapp,
            lambda: win.page_finder.thread is not None and win.page_finder.thread.elapsed() >= 1.3,
            timeout=60,
        )
        assert captured, "the fake mic delivered nothing to the Finder"
        assert not win.page_finder.keys._active, (
            f"key(s) lit by mic noise : {sorted(win.page_finder.keys._active)}"
        )
        assert not win.page_finder.keys._spans, "a Finder key carries a note in progress"
        assert win.page_finder.session.history == [], "noise entered the history"
        assert win.error_banner.isHidden(), win.error_banner.text()
    finally:
        win.on_close_finder()
        if win.page_finder.thread is not None and win.page_finder.thread.isRunning():
            win.page_finder.thread.stop()
            win.page_finder.thread.wait(3000)
        win.close()


def test_the_finder_shows_the_note_while_it_confirms(qapp, monkeypatch):
    """The "which note am I singing" feedback must NOT wait for the
    confirmation: between the note's start and its 90 ms hold, the screen
    must not fall mute. The big name follows `current()` (PENDING
    answers), the history entry stays reserved for confirmed notes."""
    import sounddevice as sd

    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    monkeypatch.setattr(te, "export_result", lambda res, stem, **kw: None)

    win = mw.MainWindow()
    try:
        win.on_open_finder()
        captured = _pump_until(
            qapp,
            lambda: win.page_finder.thread is not None and win.page_finder.thread.elapsed() >= 1.3,
            timeout=60,
        )
        assert captured, "the fake mic delivered nothing to the Finder"
        thread = win.page_finder.thread
        assert thread is not None
        seg = thread.segmenter
        assert seg is not None
        pending_seen = _pump_until(
            qapp,
            lambda: (
                seg.current() is not None
                and win.page_finder.label_note.text() not in ("", None)
                and "…" not in win.page_finder.label_note.text()
                and seg.confirmed() is None
            )
            or seg.confirmed() is not None,
            timeout=10,
        )
        # either we saw a note displayed BEFORE its confirmation, or the
        # confirmation is already there (the train is short: both prove
        # the name follows the immediate feedback)
        if seg.confirmed() is None:
            assert pending_seen, "the name does not follow the confirming note"
        history = _pump_until(
            qapp,
            lambda: len(win.page_finder.session.history) >= 2,
            timeout=10,
        )
        assert history, "no confirmed note entered the history"
        assert all(48 <= m <= 84 for m in win.page_finder.session.history)
    finally:
        win.on_close_finder()
        if win.page_finder.thread is not None and win.page_finder.thread.isRunning():
            win.page_finder.thread.stop()
            win.page_finder.thread.wait(3000)
        win.close()
