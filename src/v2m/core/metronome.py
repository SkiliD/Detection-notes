"""Optional metronome for the live page: a steady click at a fixed BPM.

Working on accuracy ("sing in tune") is only half of practice; the other half
is regularity ("sing in time"). The metronome synthesizes a short click (a
2.8 kHz burst with a fast decay — sharp enough to cut through a sung voice)
and ticks on a monotonic clock, so the beat does not drift with GUI load.
Every 4th beat is accented, which is why beats_per_measure exists: it counts
the pattern, nothing else.

core stays Qt-free: the beat loop is a plain function that asks an injected
`scheduler` for the next tick, so the scheduling backend (Qt timer in the
GUI, test clock anywhere else) is chosen by the owner, not by core.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

CLICK_HZ = 2800.0
CLICK_S = 0.045
ACCENT_GAIN = 1.6
CLICK_GAIN = 0.22

class TimerHandle(Protocol):
    """What a scheduler must hand back: a cancellable, KEPT-alive handle."""

    def stop(self) -> None: ...


# A scheduler: schedule(delay_ms, fire) arms the next `_fire` and returns
# its timer handle (the metronome keeps it — a garbage-collected timer
# dies silently after one beat). Any backend with this one-method shape
# fits — the GUI injects a QTimer-backed one, the tests a manual one.
Scheduler = Callable[[int, Callable[[], None]], TimerHandle]


def click(sr: int, accent: bool = False) -> np.ndarray:
    """One metronome click as float32 (sharp attack, ~45 ms decay)."""
    n = int(sr * CLICK_S)
    t = np.arange(n) / sr
    env = np.exp(-t / 0.012)
    tone = np.sin(2 * np.pi * CLICK_HZ * t) + 0.4 * np.sin(4 * np.pi * CLICK_HZ * t)
    return (CLICK_GAIN * (ACCENT_GAIN if accent else 1.0) * env * tone).astype(np.float32)


def qt_scheduler():
    """A QTimer-backed scheduler, for GUI owners (imports Qt lazily).

    Kept next to the scheduling policy (the wall-clock beat loop) but out of
    core's import tree: importing this module must not pull PyQt6, so the
    import happens at call time, when a GUI owner actually asks for it.
    """
    from PyQt6.QtCore import QTimer

    def schedule(delay_ms: int, fire: Callable[[], None]) -> TimerHandle:
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(fire)
        timer.start(delay_ms)
        return timer  # the caller keeps the handle alive (a GC'd timer dies)

    return schedule


class Metronome:
    """Beat scheduler: `on_beat(beat_number)` fires every 60/bpm seconds.

    The first beat of the pattern is accented; the callback receives the
    1-based beat index. `start()` resets to beat 1 — a practice session
    always begins on a downbeat.
    """

    def __init__(
        self,
        bpm: float,
        beats_per_measure: int = 4,
        scheduler: Scheduler | None = None,
    ):
        self.bpm = float(bpm)
        self.beats_per_measure = max(1, int(beats_per_measure))
        self.on_beat: Callable[[int, bool], None] | None = None
        self._scheduler = scheduler or qt_scheduler()
        self._timer = None
        self._beat = 0
        self._t0 = 0.0

    @property
    def period(self) -> float:
        return 60.0 / max(1.0, self.bpm)

    def start(self) -> None:
        """(Re)start from beat 1 and tick on a monotonic wall clock."""
        self.stop()
        self._beat = 0
        self._t0 = time.monotonic()
        self._fire()

    def _fire(self) -> None:
        self._beat += 1
        accent = (self._beat - 1) % self.beats_per_measure == 0
        if self.on_beat is not None:
            self.on_beat(self._beat, accent)
        # schedule the next tick against the WALL CLOCK, not a fixed timer
        # interval: GUI load would otherwise accumulate drift over a minute
        due = self._t0 + self._beat * self.period
        delay_ms = int(max(0.0, due - time.monotonic()) * 1000)
        self._timer = self._scheduler(delay_ms, self._fire)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        self._timer = None
