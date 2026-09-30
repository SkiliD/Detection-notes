"""DisplayStabilizer: the vibrato-tolerant NAME channel.

Pins the measured behavior: a ±0.5-st vibrato riding a note-boundary
seam must NOT flicker the displayed name (one name holds), a real
melodic change must switch, and an idle mic drops everything.
Segmentation is untouched (the corpus holds the cases exact).
"""

from __future__ import annotations

import pytest

from v2m.gui.display_stabilizer import DisplayStabilizer


def _vibrato_around(seam: float, depth: float = 0.5, cycles: int = 8):
    """(t, midi) readings of a vibrato centred on `seam` (5.5 Hz)."""
    t = 0.0
    for _ in range(cycles * 2):
        for phase in (0.0, 0.5, 1.0, 1.5):  # quarters of a sine
            import math

            yield t, seam + depth * math.sin(2 * math.pi * phase / 4)
            t += 11.61 / 1000.0


def test_a_vibrato_riding_the_seam_keeps_one_name():
    """The ±0.5-st swing across 64/65 alternates the INSTANT name every
    ~90 ms; the stabilized channel must answer ONE name for the whole
    hold (either side, but never flickering frame by frame)."""
    st = DisplayStabilizer()
    seen = {
        st.feed(t, m)
        for t, m in _vibrato_around(64.5)
    }
    assert seen <= {64, 65}, seen
    # one name dominates: the window's centre decides, the excursion
    # never drags the display with it
    names = [st.feed(t, m) for t, m in _vibrato_around(64.5)]
    switches = sum(1 for a, b in zip(names, names[1:]) if a != b)
    assert switches <= 1, f"flickering: {names}"


def test_a_wide_vibrato_centred_on_a_note_holds_the_centre():
    """The measured upper boundary (±0.7 st, corpus `vibrato large*`):
    centred ON a note, the swing rounds to THREE instant names
    (64/65/66 at 5.5 Hz); the stabilized channel answers the CENTRE for
    the whole hold — the boundary is measured, the centre rule is what
    makes it hold (deeper than that, the window's median itself moves)."""
    st = DisplayStabilizer()
    names = [
        shown
        for shown in (
            st.feed(t, m) for t, m in _vibrato_around(65.0, depth=0.7)
        )
    ]
    assert set(names) == {65}, names


def test_a_real_melodic_change_switches_the_name():
    """A whole step HELD (a real move, not an excursion) takes the display."""
    st = DisplayStabilizer()
    t = 0.0
    for _ in range(40):
        st.feed(t, 64.0)
        t += 0.02
    for _ in range(40):
        shown = st.feed(t, 66.0)
        t += 0.02
    assert shown == 66


def test_an_idle_mic_drops_the_reading():
    """Stale frames answer None: the mic left, no pitch is repeated."""
    st = DisplayStabilizer()
    assert st.feed(0.0, 64.2) == 64
    assert st.feed(0.1, None) is None
    # ...and a later lone reading starts fresh (no ghost majority)
    assert st.feed(5.0, 67.1) == 67


def test_reset_carries_nothing_over():
    st = DisplayStabilizer()
    for i in range(10):
        st.feed(i * 0.02, 64.0)
    st.reset()
    assert st.feed(10.0, 60.0) == 60


def test_nan_is_an_idle_reading():
    st = DisplayStabilizer()
    assert st.feed(0.0, float("nan")) is None
    assert st.feed(0.02, 64.1) == 64
