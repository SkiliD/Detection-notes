"""The finder's readings, decided once: pending shows, confirmed remembers.

The big name answers from the PENDING reading; the history keeps
CONFIRMED piano notes only — a rumble holds no confirmation window, so
it lights nothing and enters nothing. A HOLD is one entry; a REPEAT
(the same pitch re-confirmed after the reading broke) is another.
These tests pin that split without Qt: the session's interface is the
test surface.
"""

from __future__ import annotations

from v2m.gui.finder_session import HISTORY_LEN, FinderSession


def test_display_reads_pending_history_reads_confirmed():
    s = FinderSession()
    # a note still confirming (pending): the NAME shows it, history waits
    assert s.feed(69, None) == 69
    assert s.history == []
    # confirmed: the history records it (once)
    s.feed(69, 69)
    assert s.history == [69]
    # a HOLD stays one entry (the confirmed reading keeps streaming)
    s.feed(69, 69)
    assert s.history == [69]


def test_repeated_notes_enter_the_history_twice():
    # a repeat is the same pitch re-confirmed after the reading broke
    # (the note closed between the two confirmations)
    s = FinderSession()
    s.feed(69, 69)
    s.feed(69, None)   # the confirmed reading broke: the note closed
    s.feed(69, 69)     # re-confirmed: a new note, the same pitch
    assert s.history == [69, 69]


def test_a_rumble_never_enters_the_history():
    s = FinderSession()
    # off-piano (or a rumble's nonsense midi): display may show it (the
    # immediate channel is honest), the history records nothing
    assert s.feed(120, 120) == 120
    assert s.history == []


def test_history_is_capped_and_keeps_the_most_recent():
    s = FinderSession()
    for m in range(60, 60 + HISTORY_LEN + 3):
        s.feed(m, m)
    assert len(s.history) == HISTORY_LEN
    assert s.history[-1] == 60 + HISTORY_LEN + 2  # the latest pitch last
    assert s.history[0] == 60 + 3  # the oldest survivors follow


def test_reset_forgets_everything():
    s = FinderSession()
    s.feed(69, 69)
    s.reset()
    assert s.history == [] and s.display is None
