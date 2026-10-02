"""Metronome (core scheduling) + ClickPlayer (audio) + TuningMeter (cents)."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# core.metronome: the scheduling
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_click_shape_and_length():
    from v2m.core.metronome import CLICK_S, click

    sr = 22_050
    c = click(sr)
    plain, acc = click(sr, accent=False), click(sr, accent=True)
    assert len(c) == int(sr * CLICK_S)
    assert c.dtype == np.float32
    # the accent is the SAME sound, louder — never a different pitch
    assert 1.3 < float(np.max(np.abs(acc)) / np.max(np.abs(plain))) < 2.0
    assert float(np.max(np.abs(plain))) < 1.0


class _ManualTimer:
    def __init__(self, delay_ms, fire):
        self.delay_ms = delay_ms
        self._fire = fire
        self.stopped = False

    def fire(self):
        if not self.stopped:
            self._fire()

    def stop(self):
        self.stopped = True


class ManualScheduler:
    """A scheduler without any event loop: the test fires the armed tick."""

    def __init__(self):
        self.timers: list[_ManualTimer] = []

    def __call__(self, delay_ms, fire):
        timer = _ManualTimer(delay_ms, fire)
        self.timers.append(timer)
        return timer


def test_beats_fire_in_order_with_accent_on_one():
    """The beat loop itself is plain Python: no Qt, no event loop needed."""
    from v2m.core.metronome import Metronome

    sched = ManualScheduler()
    beats: list[tuple[int, bool]] = []
    m = Metronome(600.0, beats_per_measure=4, scheduler=sched)  # 10 Hz: fast to step
    m.on_beat = lambda b, a: beats.append((b, a))
    m.start()
    for _ in range(7):  # each fire rearms through the scheduler
        sched.timers[-1].fire()
    m.stop()
    assert [b for b, _ in beats][:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [a for _, a in beats[:4]] == [True, False, False, False]
    assert all(a for b, a in beats if (b - 1) % 4 == 0)


def test_beat_delays_are_scheduled_against_the_wall_clock():
    """The next tick is due at t0 + beat*period, not 'now + period': GUI load
    cannot accumulate drift. A late tick is compensated on the rearm."""
    from v2m.core.metronome import Metronome

    sched = ManualScheduler()
    m = Metronome(600.0, scheduler=sched)  # period 0.1 s
    m.start()
    time.sleep(0.15)  # a stalled GUI fires the first timer late
    assert sched.timers[0].delay_ms == pytest.approx(100, abs=30)
    sched.timers[0].fire()
    # the rearm compensates the lateness (a fixed +100 ms loop would not)
    assert sched.timers[1].delay_ms < 80
    m.stop()


# --------------------------------------------------------------------------- #
# gui.clicks: mixing the clicks into the stream
# --------------------------------------------------------------------------- #


def test_click_writer_mixes_overlapping_clicks(qapp):
    from v2m.gui.clicks import ClickPlayer

    player = ClickPlayer(sr=22_050)
    out = np.empty((4096, 1), dtype=np.float32)
    player._queue.append(np.ones(2048, dtype=np.float32))
    player._writer(out, 4096, None, None)
    assert np.all(out[:2048, 0] == 1.0)
    assert np.all(out[2048:, 0] == 0.0)  # no click left -> silence, no crash
    assert player._queue == []


def test_click_writer_carries_a_click_longer_than_the_block(qapp):
    from v2m.gui.clicks import ClickPlayer

    player = ClickPlayer(sr=22_050)
    out = np.empty((512, 1), dtype=np.float32)
    player._queue.append(np.ones(2048, dtype=np.float32))
    player._writer(out, 512, None, None)
    assert np.all(out[:, 0] == 1.0)
    assert len(player._queue) == 1  # the tail is re-queued, not dropped
    player._writer(out, 512, None, None)
    assert np.all(out[:, 0] == 1.0)  # ...and continues seamlessly


# --------------------------------------------------------------------------- #
# gui.widgets.TuningMeter: the cents offset
# --------------------------------------------------------------------------- #


def test_tuning_meter_clamps_and_ignores_nan(qapp):
    from v2m.gui.widgets import TuningMeter

    m = TuningMeter()
    m.set_cents(120.0)
    assert m._cents == 50.0
    m.set_cents(-120.0)
    assert m._cents == -50.0
    m.set_cents(float("nan"))
    assert m._cents is None
    m.set_cents(None)
    assert m._cents is None
    assert m.grab().isNull() is False  # paints without error offscreen


def test_live_indicator_publishes_cents(qapp):
    from v2m.gui.widgets import LiveIndicator

    ind = LiveIndicator()
    ind.update_note(60.2, 265.0)   # +20 cents above C4
    assert ind.tuning._cents == pytest.approx(20.0, abs=0.5)
    ind.update_note(None, float("nan"))
    assert ind.tuning._cents is None


# --------------------------------------------------------------------------- #
# widgets.tuning_color: one green/amber/red scale for the meter
# AND the keyboard key
# --------------------------------------------------------------------------- #


def test_tuning_color_scale(qapp):
    from v2m.gui.widgets import (
        AMBER_COLOR, MATCH_COLOR, MISMATCH_COLOR, TuningMeter, tuning_color,
    )

    assert tuning_color(0.0) == MATCH_COLOR
    assert tuning_color(10.0) == MATCH_COLOR
    assert tuning_color(20.0) == AMBER_COLOR
    assert tuning_color(-25.0) == AMBER_COLOR
    assert tuning_color(40.0) == MISMATCH_COLOR
    assert tuning_color(None) == MISMATCH_COLOR
    # the meter paints with the same function (no second scale)
    m = TuningMeter()
    m.set_cents(20.0)
    assert tuning_color(m._cents) == AMBER_COLOR


# --------------------------------------------------------------------------- #
# widgets.AccuracyHistory: per-note mean accuracy of the live take
# --------------------------------------------------------------------------- #


def test_recent_midi_answers_now_not_note_average():
    """The instantaneous pitch: correcting mid-note moves the needle now."""
    from v2m.core.liveseg import LiveNoteSegmenter
    from v2m.core.music import midi_to_hz

    seg = LiveNoteSegmenter()
    t = 0.0
    for _ in range(40):                     # 0.4 s settled on A4
        seg.push(t, float(midi_to_hz(69))); t += 0.01
    for _ in range(15):                     # the singer corrects to A4+40 cents
        seg.push(t, float(midi_to_hz(69)) * 2 ** (40 / 1200)); t += 0.01
    recent = seg.recent_midi()
    cur = seg.current()
    assert recent is not None and cur is not None
    whole = cur[0]
    assert (recent - 69) * 100 == pytest.approx(40.0, abs=6.0)
    assert (whole - 69) * 100 < 25.0        # the note median barely moves


def test_metronome_beats_follow_published_grid_meter(qapp):
    """The click pattern is the take's meter in displayed beats (3/4 -> 3).

    The meter is PUBLISHED inside the take's live grid (LiveTake detects it
    from the notes' accents; the GUI only reads grid.meter): a 3/4 take
    keeps 3 displayed beats, a 4/4 take then re-syncs the same metronome to
    4. Same values, another path.
    """
    from v2m.core.grid import BeatGrid, Meter
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win.page_live.chk_metronome.setChecked(True)
        metro = win.metronome
        assert metro is not None
        try:
            win._sync_metronome_to_take(BeatGrid(150.0, Meter(3, 4)))
            assert metro.beats_per_measure == 3
            win._sync_metronome_to_take(BeatGrid(120.0, Meter(4, 4)))
            assert metro.beats_per_measure == 4
        finally:
            win.page_live.chk_metronome.setChecked(False)
    finally:
        win.close()


def test_accuracy_history_running_mean_and_reset(qapp):
    from v2m.gui.widgets import AccuracyHistory

    h = AccuracyHistory()
    h.add(60, 20.0)
    h.add(60, 10.0)
    h.add(62, -30.0)
    assert h.means[60] == pytest.approx(15.0)
    assert h.means[62] == pytest.approx(30.0)
    assert list(h.means) == [60, 62]  # insertion order = singing order
    h.add(None, 5.0)      # no note: ignored
    h.add(64, None)
    h.add(64, float("nan"))
    assert 64 not in h.means
    h.reset()
    assert h.means == {}


def test_accuracy_history_keeps_repeated_note_points(qapp):
    from v2m.gui.widgets import AccuracyHistory

    h = AccuracyHistory()
    h.add(60, 10.0, onset=0.2)
    h.add(60, 20.0, onset=0.2)
    h.add(60, 30.0, onset=0.8)
    assert h.points == [(60, 15.0), (60, 30.0)]
    assert h.onsets == [0.2, 0.8]


def test_accuracy_plot_points_names_and_paint(qapp):
    from v2m.gui.widgets import AccuracyPlot

    plot = AccuracyPlot()
    assert plot._points == []  # empty state before any take
    plot.set_points([(60, 5.0), (64, 30.0), (67, 80.0)])
    assert [m for m, _ in plot._points] == [60, 64, 67]
    plot.set_points([])  # new take: back to the empty state
    assert plot._points == []
    # both states paint without error offscreen (empty card incl. guides)
    assert plot.grab().isNull() is False
    plot.set_points([(60, 5.0), (64, 30.0), (67, 80.0)])
    assert plot.grab().isNull() is False


def test_live_accuracy_line_and_plot_agree(qapp):
    """Line and plot read the same history: they can never disagree."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win._on_live_note((60.1, 262.0))
        win._on_live_note((62.4, 295.0))
        items = list(win.page_live.accuracy.means.items())
        assert win.page_live.accuracy_plot._points == items
        assert len(win.page_live.accuracy_plot._points) == 2
        # each point carries its note's onset: the click-to-seek data
        assert len(win.page_live.accuracy_plot._times) == 2
    finally:
        win.close()


def test_accuracy_plot_click_emits_note_onset(qapp):
    """Clicking a point emits its note's onset; a click off the points, none."""
    from v2m.gui.widgets import AccuracyPlot

    class _Pos:
        def __init__(self, x):
            self._x = x

        def x(self):
            return self._x

    class _Ev:
        def __init__(self, x):
            self._p = _Pos(x)

        def position(self):
            return self._p

    plot = AccuracyPlot()
    plot.set_points([(60, 5.0), (64, 30.0)], times=[1.0, 2.5])
    got = []
    plot.pointClicked.connect(got.append)
    # centre of the SECOND point, computed from the paint's own geometry
    pad, n = 16.0, 2
    ww = (plot.width() - 2 * pad) / n
    x_second = pad + (1 + 0.5) * ww
    plot.mousePressEvent(_Ev(x_second))
    assert got == [2.5]
    plot.mousePressEvent(_Ev(2.0))  # off the points: no emission
    assert got == [2.5]
    # without onsets the plot stays a plain summary (no crash, no signal)
    plot.set_points([(60, 5.0)])
    plot.mousePressEvent(_Ev(plot.width() / 2))
    assert got == [2.5]


def test_accuracy_point_click_seeks_results_roll(qapp):
    """After analysis, a plot click jumps to the results page playhead."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        win._accuracy_plot_target = "results"
        win._on_accuracy_point_clicked(0.5)
        assert win.stack.currentWidget() is win.page_results
    finally:
        win.close()


def test_metronome_module_is_qt_free():
    """core must not depend on Qt: importing core.metronome pulls no PyQt6.

    Proven in a SUBPROCESS: in-suite, earlier GUI tests have already loaded
    Qt into this process, so only a clean interpreter is an honest proof.
    The QTimer backend is imported lazily at call time (qt_scheduler), which
    the GUI smoke test exercises."""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"))
    code = (
        "import sys; import v2m.core.metronome; "
        "loaded = [m for m in sys.modules if m == 'PyQt6' or m.startswith('PyQt6.')]; "
        "assert not loaded, f'PyQt6 charge par core.metronome: {loaded}'; "
        "print('qt-free ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert out.returncode == 0, out.stderr
    assert "qt-free ok" in out.stdout
