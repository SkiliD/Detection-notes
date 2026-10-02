"""The note finder's STATE, one deep module: what it heard, what it shows.

The finder page has two jobs that look like one and are not:
answering "which note am I singing RIGHT NOW" — a note still confirming
counts — and remembering the CONFIRMED pitches — a mic rumble slips a
few periodic frames but never holds, so it lights nothing and enters
nothing. `FinderSession` owns both readings and the recent-pitch
history; the window renders what the session decides.

Repeats ARE notes: the history deduplicates a HOLD, not the pitch — a
confirmed reading that breaks (the note closed, `confirmed()` answers
None) makes the next confirmation a new entry, the same pitch included.
Each entry is a real re-articulation: re-confirming costs `confirm_s`
of held plateau (the segmenter's own bar), so a wobble cannot spam the
history — only a genuinely re-confirmed note does.

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

    Holds are one entry; repeats (a re-confirmation after the reading
    broke) are separate entries — see the module docstring.
    """

    history: list[int] = field(default_factory=list)
    _last_confirmed: int | None = None
    _confirmed_open: bool = False  # a confirmed reading is streaming now
    display: int | None = None
    last_reading: float | None = None  # tick time of the last segmenter reading

    def feed(self, cur_midi: int | None, confirmed_midi: int | None, t: float | None = None) -> int | None:
        """One tick: decide the display, keep the confirmed history."""
        # the big name answers from the IMMEDIATE reading (a confirming
        # note counts — hiding it read as "the mic is ignored")
        self.display = cur_midi
        if cur_midi is not None or confirmed_midi is not None:
            self.last_reading = t
        # a confirmed reading that broke means the note CLOSED (state left
        # ACTIVE — the segmenter only leaves it on attack/pitch change/gap):
        # the next confirmation is a new note, a repetition included
        if confirmed_midi is None and self._confirmed_open:
            self._confirmed_open = False
        # the history records CONFIRMED piano notes only: a hold is ONE
        # entry, a re-confirmation after a break is another (singing do-do
        # must show do twice). A rumble holds no confirmation window, so it
        # enters nothing.
        if (
            confirmed_midi is not None
            and on_piano(confirmed_midi)
            and (confirmed_midi != self._last_confirmed or not self._confirmed_open)
        ):
            self.history.append(confirmed_midi)
            self.history[:] = self.history[-HISTORY_LEN:]
            self._last_confirmed = confirmed_midi
        if confirmed_midi is not None:
            self._confirmed_open = True
        return self.display

    def note_idle(self) -> None:
        """One tick with NO reading at all: the mic heard nothing."""
        self.last_reading = None

    def reset(self) -> None:
        """The finder restarted: nothing is remembered."""
        self.history.clear()
        self._last_confirmed = None
        self.display = None
        self.last_reading = None
