"""A live take's render state has ONE owner: LiveRenderState.

Structural regression: the nine scattered attributes (`_live_signal`,
`_closed_live_notes`, `_live_record_audio`, the pitch cloud, the grid…)
were reset one by one in `on_start_live` — one miss and the PREVIOUS
take's value leaked into this one (measured: the exported WAV described
the earlier take). The take now owns one object and `on_start_live`
swaps it wholesale: a stale value is structurally impossible, no buffer
able to survive exists anymore. Tests simulating a mid-take state build
THIS object (a single entry point), no need to drive the whole window;
the `win._live_*` faces stay readable on top.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# Qt windows built and torn down: the module costs a few seconds —
# marked slow (deselect with `-m "not slow"`), it stays IN the suite by
# default.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_a_new_take_swaps_the_whole_render_state(qapp, monkeypatch):
    """A new take restarts from a FRESH state: signal, closed notes, pitch
    cloud, grid — nothing of the previous take survives, by construction
    (a new object), not by discipline (nine resets)."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        old = win._live
        # the PREVIOUS take left everything dirty
        old.signal = np.ones(1000, dtype=np.float32)
        old.closed_notes = [object()]
        old.pitch_t = [1.0]
        old.pitch_m = [69.0]
        old.t = 42.0
        old.record_audio = True

        win.on_start_live()

        assert win._live is not old, "the new take must own ITS state"
        assert win._live.signal is None
        assert win._live.closed_notes == []
        assert win._live.pitch_t == [] and win._live.pitch_m == []
        assert win._live.t == 0.0
    finally:
        win.on_stop_live()
        win.close()


def test_the_audio_box_choice_is_read_at_take_start(qapp, monkeypatch):
    """The audio box belongs to the TAKE (read at start): toggling mid-take
    does not change what this take will write — pinned on the owned state,
    not a free attribute."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.chk_record_audio.setChecked(True)
        win.on_start_live()
        assert win._live.record_audio is True
        # a mid-take toggle must not change what THIS take writes
        win.chk_record_audio.setChecked(False)
        assert win._live.record_audio is True
    finally:
        win.on_stop_live()
        win.close()


def test_the_familiar_faces_read_the_owned_state(qapp):
    """The `win._live_*` readers (and the tests using them) see the SAME
    state through the properties: write through the face, read through
    the object — one storage."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win._live_signal = np.zeros(4, dtype=np.float32)
        assert win._live.signal is win._live_signal
        win._closed_live_notes = [1, 2]
        assert win._live.closed_notes is win._closed_live_notes
        win._live_t = 3.5
        assert win._live.t == 3.5
        win._live_grid = mw.BeatGrid(90.0)
        assert win._live.grid.bpm == 90.0
    finally:
        win.close()


def test_the_live_tests_no_longer_need_the_whole_window(qapp):
    """A mid-take state is built in ONE line: the export test no longer has
    to replay the whole window to place a signal."""
    from v2m.gui import main_window as mw

    state = mw.LiveRenderState()
    state.signal = np.ones(8, dtype=np.float32)
    state.record_audio = True
    assert state.signal.shape == (8,)
    assert state.record_audio is True
    # a fresh state has nothing of a take in it
    fresh = mw.LiveRenderState()
    assert fresh.signal is None and fresh.closed_notes == []
