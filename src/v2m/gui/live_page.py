"""The live page: widgets AND behavior, one owner.

A Page module, ADR-0010-shaped — the full shape now: this module builds
the live page's widgets (the roll, the staff, the indicator, the
metronome row, the recording box…) AND owns the behavior that drives
them. The window composes the page and keeps the `win.<widget>` faces
as properties over the page's attributes, so the 60+ external readers
(the tests, the smoke script) read the same objects at the same paths
without knowing who built them.

What lives here (was the window's live cluster):
- the widget build (moved from pages.build_live_page, unchanged layout);
- the 50 ms `_timer` — the ONE writer of the live render (`on_tick`:
  roll, staff, preview, cursor, pitch cloud, status);
- the note tick (`on_note`: indicator, instantaneous cents, accuracy
  history, key color by expected-vs-sung);
- the pitch cloud (`on_pitch`: the scrolling window's buffers);
- the silence watch (`watch_silence`, the 2 s advisory);
- the per-note accuracy line + plot (`refresh_accuracy`).

The state (buffers, clock, silence watch) lives on the take's
`LiveRenderState` — swapped WHOLESALE by `on_start_live`, so a stale
value from the previous take is structurally impossible. The page reads
it through one injected getter; the window keeps the `win._live_*`
faces over the same state.

The mic is faked in tests at the `sounddevice.InputStream` level; the
thread is injected as a zero-arg factory and the page never imports the
audio stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import time
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from ..core.accuracy import AccuracyHistory
from ..core.i18n import note_display_pair, tr
from ..core.music import on_piano
from .clicks import ClickPlayer
from .display_stabilizer import DisplayStabilizer
from .finder_page import LevelMeter
from .staff import StaffView
from .widgets import (
    ACCENT, AccuracyPlot, HINT_CARD_STYLE, LiveIndicator, MISMATCH_COLOR,
    PianoRoll, PianoView, cents_deviation, tuning_color,
)


def clock(t: float) -> str:
    """mm:ss, the live status clock (moved with its only readers)."""
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes:02d}:{seconds:02d}"

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow
    from .main_window import LiveThread, LiveRenderState

# silence longer than this during a capture is told to the singer (the
# "am I being heard?" check): a phrase gap of a take's own character
# sits well under it (~0.36 s on chant.wav, up to the 2 s rest case in
# the truth corpus) — past 2 s the mic is mute, slipped or facing away
LIVE_SILENCE_WARN_S = 2.0

# a weak mic is named BEFORE the take starts: nothing above the gate yet
# (the singer is settling) and the page says so after 3 s — with an
# action, not a judgment. Motivated by the 2026-09-29 corpus: that day's
# takes sat at RMS 0.019..0.062, 3-7x under the previous day's, and the
# weakest degraded the PITCH (12/22 labels off), not the note count — a
# level the singer must know about before wasting a take. More patient
# than the during-take watch (3 s vs 2 s): settling before singing is
# normal, silence DURING a take is not. One watch per phase: once one
# block clears the gate the take has started and `watch_silence` owns
# the rest; the state swaps wholesale with the take, so the next take
# re-arms it.
LIVE_LOW_LEVEL_WARN_S = 3.0


class LivePage(QWidget):
    """The live page: its widgets (built here) and its behavior (the tick)."""

    def __init__(
        self,
        window: "QMainWindow",
        state_getter: Callable[[], "LiveRenderState"],
        thread_getter: Callable[[], "LiveThread"],
        on_stop: Callable[[], None],
        toggle_metronome: Callable[[bool], None],
        metronome_bpm_changed: Callable[[int], None],
        expected_note: Callable[[float], "int | None"],
        silence_message: Callable[[str], None],
        gate_rms: Callable[[], float],
    ) -> None:
        super().__init__()
        self._win = window
        self._state = state_getter
        self._thread = thread_getter
        self._on_stop = on_stop
        self._toggle_metronome = toggle_metronome
        self._metronome_bpm_changed = metronome_bpm_changed
        self._expected_note = expected_note
        self._silence_message = silence_message
        self._gate_rms = gate_rms
        # the accuracy history is the PAGE's domain state (it is the only
        # writer, note by note) — the window consumes it for the archive
        self.accuracy = AccuracyHistory()
        # the big NAME channel is vibration-tolerant (a ±0.5-st beginner
        # vibrato must not flicker the name across the seam): the needle
        # stays instantaneous, the name takes the stabilizer's decision
        self._stabilizer = DisplayStabilizer()
        self._build()
        # the render tick: ONE owner (50 ms) — the audio thread only
        # publishes notes, so the roll stays in sync even if a block or
        # an event is late
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.on_tick)

    # ------------------------------------------------------------------
    # widgets (moved from pages.build_live_page, unchanged layout)
    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_stop_live = QPushButton("")
        self.btn_stop_live.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop_live.clicked.connect(self._on_stop)
        top.addWidget(self.btn_stop_live)
        self.label_status = QLabel("")
        top.addWidget(self.label_status, stretch=1)
        self.label_expected = QLabel("")
        self.label_expected.setStyleSheet("color: #9fb3c8; font-size: 9pt;")
        top.addWidget(self.label_expected)
        root.addLayout(top)

        self.label_hint = QLabel("")
        self.label_hint.setStyleSheet(HINT_CARD_STYLE)
        self.label_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.label_hint)

        # optional metronome: OFF by default — a singer opens it when
        # working on regularity, the status line shows the count during
        # capture
        metro = QHBoxLayout()
        self.chk_metronome = QCheckBox("")
        self.chk_metronome.toggled.connect(self._toggle_metronome)
        metro.addWidget(self.chk_metronome)
        self.spin_bpm = QSpinBox()
        self.spin_bpm.setRange(40, 208)
        self.spin_bpm.setValue(90)
        self.spin_bpm.valueChanged.connect(self._metronome_bpm_changed)
        metro.addWidget(self.spin_bpm)
        self.label_bpm = QLabel("")
        metro.addWidget(self.label_bpm)
        self.label_beat = QLabel("")
        self.label_beat.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
        metro.addWidget(self.label_beat, stretch=1)
        metro.addStretch(1)
        root.addLayout(metro)

        # optional: keep the take's AUDIO beside its artifacts, only in
        # live mode. The box is read when the take STARTS (a mid-take
        # toggle must not change what this take writes). ON by default:
        # the audio is the only way a failed take stays diagnosable —
        # measured on the 2026-09-29 corpus, where the takes that needed
        # auditing (2 notes live vs 18 final) had no WAV to audit.
        rec = QHBoxLayout()
        self.chk_record_audio = QCheckBox("")
        self.chk_record_audio.setChecked(True)
        rec.addWidget(self.chk_record_audio)
        rec.addStretch(1)
        root.addLayout(rec)

        mid = QHBoxLayout()
        self.live_indicator = LiveIndicator()
        mid.addWidget(self.live_indicator, stretch=1)
        # the mic level bar, the silence gate drawn on it (the finder's
        # LevelMeter pattern): "am I being heard?" is answerable BEFORE
        # starting a take, and during one by the same bar. Injected: the
        # gate is audio_io's fact, the page never re-derives it.
        self.level_meter = LevelMeter()
        self.level_meter.set_gate_rms(self._gate_rms())
        mid.addWidget(self.level_meter)
        self.piano_live = PianoRoll()
        mid.addWidget(self.piano_live, stretch=3)
        root.addLayout(mid, stretch=2)

        # per-note mean deviation so far: the progress signal of the take
        self.label_accuracy = QLabel("")
        self.label_accuracy.setStyleSheet("color: #8a97a5; font-size: 9pt;")
        root.addWidget(self.label_accuracy)
        # ...and the same deviations over the take's timeline (singing order)
        self.accuracy_plot = AccuracyPlot()
        root.addWidget(self.accuracy_plot)

        self.staff_live = StaffView(live=True)
        root.addWidget(self.staff_live, stretch=2)
        self.piano_live_keys = PianoView()
        root.addWidget(self.piano_live_keys)
        self.clicks = ClickPlayer()

    def retranslate(self) -> None:
        """The page's user-facing strings (the window's retranslate calls it)."""
        self.btn_stop_live.setText(tr("ui stop live"))
        self.label_hint.setText(tr("ui live hint"))
        self.chk_metronome.setText(tr("ui metronome"))
        self.spin_bpm.setToolTip(tr("ui metronome tooltip"))
        self.label_bpm.setText(tr("ui bpm unit"))
        if self.chk_metronome.isChecked():
            self.label_beat.setText(tr("ui beat count").format(beat="—"))
        else:
            self.label_beat.setText("")
        self.chk_record_audio.setText(tr("ui record audio"))
        self.chk_record_audio.setToolTip(tr("ui record audio tooltip"))

    # ------------------------------------------------------------------
    # lifecycle the window drives
    # ------------------------------------------------------------------
    def start_timer(self) -> None:
        self._timer.start()

    def stop_timer(self) -> None:
        self._timer.stop()

    def timer_active(self) -> bool:
        """The controller reads this fact, it never reaches into the timer."""
        return self._timer.isActive()

    def begin_take(self) -> None:
        """A new take: drop the previous pitch cloud and its curve."""
        live = self._state()
        live.pitch_t = []
        live.pitch_m = []
        live.t = 0.0
        self.accuracy.reset()
        self._stabilizer.reset()

    # ------------------------------------------------------------------
    # the 50 ms render tick (single owner of the live rendering)
    # ------------------------------------------------------------------
    def on_tick(self) -> None:
        """Live follow (~20 fps): render the take, scroll, move the cursor."""
        thr = self._thread()
        if thr is None or thr.segmenter is None:
            return
        live = self._state()
        piano_live = self.piano_live
        staff_live = self.staff_live
        t = live.t or thr.elapsed()
        notes = thr.segmenter.notes
        preview = thr.segmenter.active_note(t)
        piano_live.set_notes(notes)
        piano_live.set_live_preview(preview)
        midis = [n.midi for n in notes[-64:]]
        if preview is not None:
            midis.append(preview.midi)
        piano_live.set_live_pitch_range(midis)
        piano_live.set_live_cursor(t)
        staff_live.set_live(notes, preview, grid=live.grid)
        self._refresh_expected(t)
        # the big counter answers "is it following me?" at a glance
        self.live_indicator.set_note_count(len(notes))
        self.label_status.setText(
            tr("ui live progress").format(
                t=clock(t), notes=len(notes), meter=live.grid.meter.label
            )
        )

    def _refresh_expected(self, t: float) -> None:
        expected = self._expected_note(t)
        if expected is None:
            self.label_expected.setText(tr("ui expected none"))
        else:
            self.label_expected.setText(
                tr("ui expected note").format(note=note_display_pair(expected))
            )

    # ------------------------------------------------------------------
    # event handlers (the thread's signals)
    # ------------------------------------------------------------------
    def on_block(self, t: float, rms: float) -> None:
        """Telemetry from each captured block: clock, level, the phase's watch."""
        self._state().t = t
        self.live_indicator.update_rms(rms)
        self.level_meter.set_level(rms)
        if self._state().heard_once:
            self.watch_silence(t, rms)    # the take has started: phrase gaps
        else:
            self.watch_low_level(t, rms)  # pre-take: name a weak mic early

    def watch_low_level(self, t: float, rms: float) -> None:
        """Name a mute or weak mic BEFORE the take starts (3 s under the gate).

        The take STARTS at its first block above the silence gate — before
        that the singer is settling, and the during-take watch must not
        speak (its wording judges a take that has not begun). Past 3 s of
        nothing above the gate since the page opened, the status bar names
        the mic so it can be checked BEFORE singing. The level meter
        already shows the level per block; this is the escalation that
        reads itself. A voiced block disarms it for the whole take (the
        state swaps per take, so the next take re-arms it).
        """
        live = self._state()
        if rms > self._gate_rms():
            live.heard_once = True
            live.low_since = None
            return
        if live.low_since is None:
            live.low_since = t
            return
        silent_for = t - live.low_since
        if silent_for < LIVE_LOW_LEVEL_WARN_S:
            return
        self._silence_message(
            tr("ui live low mic").format(s=int(silent_for)),
        )

    def watch_silence(self, t: float, rms: float) -> None:
        """Warn the singer when the capture hears NOTHING for over 2 s.

        A voiced block resets the clock silently. Past the threshold the
        status bar says WHEN the sound stopped and for HOW LONG — checked
        at block rate (~46 ms), so the message stops growing with the
        next block and the take itself is never touched (an advisory,
        not a fault). State lives on LiveRenderState: one swap per take.
        """
        live = self._state()
        if rms > self._gate_rms():
            live.silence_since = None
            return
        if live.silence_since is None:
            live.silence_since = t
            return
        silent_for = t - live.silence_since
        if silent_for < LIVE_SILENCE_WARN_S:
            return
        self._silence_message(
            tr("ui live silence").format(s=int(silent_for)),
        )

    def on_note(self, cur) -> None:
        """Render the recognition state published by the live engine."""
        midi = int(round(cur[0])) if cur else None
        hz = cur[1] if cur else float("nan")
        # the deviation shown is the INSTANTANEOUS one (last smoothed
        # frames): correcting the pitch mid-note must move the needle now
        thr = self._thread()
        seg = thr.segmenter if thr is not None else None
        inst = seg.recent_midi() if seg is not None else None
        cents_inst = cents_deviation(inst) if inst is not None else None
        # the NAME comes from the stabilized reading (a ±0.5-st vibrato
        # must not flicker the name across the seam every ~90 ms); the
        # hz line and the tuning needle stay INSTANTANEOUS — correcting
        # the pitch mid-note must move the needle now. The stabilizer
        # state is the live tick's clock (wall time), not the audio t.
        shown = self._stabilizer.feed(time.monotonic(), cur[0] if cur else None)
        shown_float = cur[0] if cur is not None else None
        if shown is not None:
            shown_float = float(shown) if cur is None else self._blend(shown, cur[0])
        self.live_indicator.update_note(shown_float, hz, cents_inst)
        midi = midi if midi is not None and on_piano(midi) else None
        cents = cents_deviation(cur[0]) if cur else None
        onset = None
        if seg is not None:
            onset = seg.note_start if seg.note_start is not None else seg.pending_start
        self.accuracy.add(midi, cents, onset=onset if onset is not None else self._state().t)
        # the key carries the tuning feedback: a wrong note is red however
        # pure, a right/unknown one is green/amber/red by its cents error
        color = None
        if midi is not None:
            expected = self._expected_note(self._state().t)
            if expected is not None and midi != expected:
                color = MISMATCH_COLOR
            else:
                # instantaneous when the stream provides it, note median
                # otherwise (no info must not read as "out of tune")
                color = tuning_color(cents_inst if cents_inst is not None else cents)
        self.piano_live_keys.set_pressed(
            {midi} if midi is not None else (),
            colors=({midi: color} if color is not None else None),
        )
        self.refresh_accuracy()

    @staticmethod
    def _blend(shown: int, midi_float: float) -> float:
        """The stabilized name carrying the REAL excursion for the meter.

        `update_note` derives the cents from the float midi against its
        nearest note; clamping the excursion at ±0.5 st keeps the needle
        honest at vibrato's extremes (a ±0.5-st swing reads ±50 ¢) while
        the NAME stays the stabilized one. `cents` (the instantaneous
        argument) already overrides it when the stream provides one —
        this only fixes the fallback path.
        """
        excursion = max(-0.5, min(0.5, midi_float - shown))
        return shown + excursion

    def refresh_accuracy(self) -> None:
        """Per-note mean deviation so far: the text line and the plot.

        Both views read the same `AccuracyHistory`, so they can never
        disagree; the line shows the last six notes, the plot the whole
        take in singing order. Each plot point carries its note's onset
        so a click seeks the roll there (live) or to the results
        playhead (after the final analysis).
        """
        items = list(self.accuracy.points)
        self.accuracy_plot.set_points(items, times=self.accuracy.onsets)
        if not items:
            self.label_accuracy.setText(tr("ui accuracy empty"))
            return
        parts = "  ".join(
            f"{note_display_pair(m)} {v:.0f}¢" for m, v in reversed(items[-6:])
        )
        self.label_accuracy.setText(tr("ui accuracy title") + " " + parts)

    def on_pitch(self, t: float, hz: float) -> None:
        """The pitch cloud: buffers + the scrolling window's trim."""
        live = self._state()
        if hz == hz and hz > 0:  # NaN check
            live.pitch_t.append(t)
            live.pitch_m.append(69 + 12 * np.log2(hz / 440.0))
            # keep the pitch cloud inside the scrolling window
            while live.pitch_t and t - live.pitch_t[0] > self.piano_live.LIVE_SPAN:
                live.pitch_t.pop(0)
                live.pitch_m.pop(0)
            if live.curve is not None:
                live.curve.setData(live.pitch_t, live.pitch_m)

    def on_failed(self, msg: str) -> None:
        self._timer.stop()
        self.label_status.setText(tr("ui mic preparing"))
        self._silence_message(msg)  # the window's error surface (same route)
