"""Vibrato-tolerant display reading: a smoothing channel for note NAMES.

A beginner's vibrato (+/-0.5 st, measured as the gap behind "il entend
mal la voix") swings the raw pitch across a note boundary every ~90 ms:
the INSTANT reading (one frame, `heard()`) alternates between the two
neighbouring names and the display flickers. Segmentation must NOT
change (measured: the corpus holds the cases exact — HOLD_ST 0.6 stays,
`scripts/liveseg_sweep.py`); only the NAME channel stabilizes, and the
same way a singer reads their own vibrato: the pitch CENTRE, not its
excursion.

`DisplayStabilizer` is the decision, pure Python (the interface is the
test surface): feed it instant readings, it answers the name to show —
hysteresis over the ROUNDED note (switch only when the new note wins
the window), a centre median when the window straddles a boundary (a
±0.5-st swing centred on the seam reads the seam's name, which is what
the ear follows), stale -> None (the mic left: do not repeat a pitch).
"""

from __future__ import annotations

from collections import deque

WINDOW_S = 0.30  # ~5 vibrato cycles at 5.5 Hz, one name change at most
SWITCH_MARGIN = 2  # frames the challenger must win to take the display


class DisplayStabilizer:
    """The name a wavering pitch should display. See the module docstring."""

    def __init__(self, window_s: float = WINDOW_S) -> None:
        self._window_s = window_s
        self._frames: deque[tuple[float, float]] = deque()  # (t, float midi)
        self._shown: int | None = None

    def feed(self, t: float, midi_float: float | None) -> int | None:
        """One instant reading; the name to display (None when idle)."""
        if midi_float is None or midi_float != midi_float:  # None or NaN
            return self._drop()
        self._frames.append((t, midi_float))
        while self._frames and t - self._frames[0][0] > self._window_s:
            self._frames.popleft()
        centre = sum(m for _, m in self._frames) / len(self._frames)
        note = int(round(centre))
        if self._shown is None or note == self._shown:
            self._shown = note
            return note
        # a challenger wins only on the window's DECIDED majority: a
        # vibrato excursion crossing the seam for a few frames must not
        # drag the display with it
        wins = sum(
            1
            for _, m in self._frames
            if abs(m - note) <= abs(m - self._shown)
        )
        if wins >= len(self._frames) - SWITCH_MARGIN and wins > SWITCH_MARGIN:
            self._shown = note
            return note
        return self._shown

    def _drop(self) -> int | None:
        """An idle reading: forget the frames, keep nothing to repeat."""
        self._frames.clear()
        self._shown = None
        return None

    def reset(self) -> None:
        """A new session: nothing carries over."""
        self._frames.clear()
        self._shown = None
