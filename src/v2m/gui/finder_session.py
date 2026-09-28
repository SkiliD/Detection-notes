"""The note finder's STATE, one deep module: what it heard, what it shows.

The finder page has two jobs that look like one and are not (measured,
user-reported): answering "which note am I singing RIGHT NOW" — a note
still confirming counts, or the singer reads "the mic is ignored" during
the 90 ms confirmation window — and remembering the CONFIRMED pitches —
a mic rumble slips a few periodic frames but never holds, so it lights
nothing and enters nothing. `FinderSession` owns both readings and the
recent-pitch history; the window renders what the session decides.

Pure Python: the interface is the test surface, no event loop. The
window keeps the thread and the widgets; this object owns the DECISIONS
(what enters the history, what the big name shows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.music import on_piano

HISTORY_LEN = 6  # distinct recent pitches, most recent last


@dataclass
class FinderSession:
    """The finder's readings and memory, decided in one place.

    - `feed(cur_midi, confirmed_midi)` — one segmenter tick: `cur_midi`
      is the immediate (PENDING answers too) reading, `confirmed_midi`
      the segmenter's confirmed one. Returns what to DISPLAY (the
      pending reading wins: it is the "the mic hears you" channel).
    - `reset()` — the finder restarted: history and last-confirmed
      pitch die with the session (the window re-creates it).
    - `history` / `display` — read-only views for the widgets.
    """

    history: list[int] = field(default_factory=list)
    _last_confirmed: int | None = None
    display: int | None = None

    def feed(self, cur_midi: int | None, confirmed_midi: int | None) -> int | None:
        """One tick: decide the display, keep the confirmed history."""
        # the big name answers from the IMMEDIATE reading (a confirming
        # note counts — hiding it read as "the mic is ignored")
        self.display = cur_midi
        # the history records CONFIRMED piano notes only, no repeats: a
        # rumble holds no confirmation window, so it enters nothing
        if (
            confirmed_midi is not None
            and on_piano(confirmed_midi)
            and confirmed_midi != self._last_confirmed
        ):
            self.history.append(confirmed_midi)
            self.history[:] = self.history[-HISTORY_LEN:]
            self._last_confirmed = confirmed_midi
        return self.display

    def reset(self) -> None:
        """The finder restarted: nothing is remembered."""
        self.history.clear()
        self._last_confirmed = None
        self.display = None
