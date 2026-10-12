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

import time
from typing import TYPE_CHECKING, Callable

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.accuracy import AccuracyHistory
from ..core.i18n import note_display_pair, tr
from ..core.intonation_live import LiveIntonation
from ..core.liveseg import LiveNote
from ..core.music import on_piano
from .clicks import ClickPlayer
from .display_stabilizer import DisplayStabilizer
from .finder_page import LevelMeter
from .staff import StaffView
from .widgets import (
    ACCENT,
    HINT_CARD_STYLE,
    MISMATCH_COLOR,
    AccuracyPlot,
    LiveIndicator,
    PianoRoll,
    PianoView,
    cents_deviation,
    intonation_lines,
    action_button_style,
    tuning_color,
)


def clock(t: float) -> str:
    """mm:ss, the live status clock (moved with its only readers)."""
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes:02d}:{seconds:02d}"


if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow

    from .main_window import LiveRenderState, LiveThread

# silence longer than this during a capture is told to the singer (the
# "am I being heard?" check): a phrase gap of a take's own character
# sits well under it (~0.36 s on chant.wav, up to the 2 s rest case in
# the truth corpus) — past 2 s the mic is mute, slipped or facing away
LIVE_SILENCE_WARN_S = 2.0

# a weak mic is named BEFORE the take starts: nothing above the gate yet
# (the singer is settling) and the page says so after 3 s — with an
# action, not a judgment. Measured: a level 3-7x under the usual one
# degraded the PITCH labels, not the note count — a level the singer
# must know about before wasting a take. More patient
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
        on_review_back: Callable[[], None],
        audition_note: Callable[[int], None],
        stop_audition: Callable[[], None],
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
        self._on_review_back = on_review_back
        self._audition_note = audition_note
        self._stop_audition = stop_audition
        self._replacement_mode = False
        self._reviewing = False
        # the captured roll's audition readout: the note the singer is
        # hearing and the stop action that ends it (results-page pattern)
        self._note_audition: tuple[int, int, int] | None = None
        self._comparison_notes: tuple[list, set[int]] | None = None
        self._comparison_visible = True
        # the accuracy history is the PAGE's domain state (it is the only
        # writer, note by note) — the window consumes it for the archive
        self.accuracy = AccuracyHistory()
        # the take-so-far intonation feedback (share of in-tune notes +
        # the cumulative lean): fed at `note_closed`, read in the render
        # tick — one core owner, the report's own evidence table
        self.intonation = LiveIntonation()
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
        self.btn_stop_live.setStyleSheet(action_button_style(primary=True))
        self.btn_stop_live.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop_live.clicked.connect(self._on_stop)
        top.addWidget(self.btn_stop_live)
        # shown only while REVIEWING a stopped take's frozen roll (the way
        # back to the final analysis, which a fresh take hides)
        self.btn_review_back = QPushButton("")
        self.btn_review_back.clicked.connect(self._on_review_back)
        self.btn_review_back.setVisible(False)
        top.addWidget(self.btn_review_back)
        self.label_status = QLabel("")
        top.addWidget(self.label_status, stretch=1)
        self.label_expected = QLabel("")
        self.label_expected.setStyleSheet("color: #9fb3c8; font-size: 9pt;")
        top.addWidget(self.label_expected)
        # the frozen roll's audition: which CAPTURED note is sounding, and
        # the stop action (hidden until Space picks a note — Qt skips it)
        self.label_live_audition = QLabel("")
        self.label_live_audition.setVisible(False)
        top.addWidget(self.label_live_audition)
        self.btn_live_audition_stop = QPushButton("")
        self.btn_live_audition_stop.clicked.connect(self._stop_audition_cb)
        self.btn_live_audition_stop.setVisible(False)
        top.addWidget(self.btn_live_audition_stop)
        root.addLayout(top)

        self.label_hint = QLabel("")
        self.label_hint.setStyleSheet(HINT_CARD_STYLE)
        self.label_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.label_hint)

        # Recording controls share one card so the optional settings stay
        # visually secondary to the live reading and the stop action.
        self.box_controls = QGroupBox("")
        controls = QVBoxLayout(self.box_controls)
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
        controls.addLayout(metro)

        # optional: keep the take's AUDIO beside its artifacts, only in
        # live mode. The box is read when the take STARTS (a mid-take
        # toggle must not change what this take writes). ON by default:
        # the audio is the only way a failed take stays diagnosable.
        rec = QHBoxLayout()
        self.chk_record_audio = QCheckBox("")
        self.chk_record_audio.setChecked(True)
        rec.addWidget(self.chk_record_audio)
        rec.addStretch(1)
        controls.addLayout(rec)
        root.addWidget(self.box_controls)

        self.box_monitor = QGroupBox("")
        mid = QHBoxLayout(self.box_monitor)
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
        # Space on the frozen roll auditions a CAPTURED note; the window
        # slices the take's audio at that note (the roll never holds it)
        self.piano_live.audition_requested.connect(self._audition_note_cb)
        mid.addWidget(self.piano_live, stretch=3)
        root.addWidget(self.box_monitor, stretch=2)

        # Both tuning views share a card: the latest notes and the whole-take
        # trend answer different questions, but belong to the same feedback.
        self.box_feedback = QGroupBox("")
        feedback = QVBoxLayout(self.box_feedback)
        self.label_accuracy = QLabel("")
        self.label_accuracy.setStyleSheet("color: #8a97a5; font-size: 9pt;")
        feedback.addWidget(self.label_accuracy)
        self.accuracy_plot = AccuracyPlot()
        feedback.addWidget(self.accuracy_plot)
        self.label_intonation = QLabel("")
        self.label_intonation.setStyleSheet("color: #8a97a5; font-size: 9pt;")
        feedback.addWidget(self.label_intonation)
        root.addWidget(self.box_feedback)

        self.staff_live = StaffView(live=True)
        root.addWidget(self.staff_live, stretch=2)
        self.piano_live_keys = PianoView()
        root.addWidget(self.piano_live_keys)
        self.clicks = ClickPlayer()

        # Keyboard traversal in task order: control the take, set what it
        # records, then walk the notes it captured. The mirror keyboard
        # (`piano_live_keys`) is deliberately NOT a Tab stop: it only draws
        # what is sounding and has no action of its own.
        focus_order = (
            self.btn_stop_live,
            self.btn_review_back,
            self.chk_metronome,
            self.spin_bpm,
            self.chk_record_audio,
            self.piano_live,
            self.btn_live_audition_stop,
        )
        self._focus_order = focus_order
        for first, second in zip(focus_order, focus_order[1:], strict=False):
            QWidget.setTabOrder(first, second)

    def retranslate(self) -> None:
        """The page's user-facing strings (the window's retranslate calls it)."""
        self.btn_stop_live.setText(
            tr("ui finish replacement") if self._replacement_mode else tr("ui stop live")
        )
        self.label_hint.setText(
            tr("ui replacement prompt") if self._replacement_mode else tr("ui live hint")
        )
        self.chk_metronome.setText(tr("ui metronome"))
        self.spin_bpm.setToolTip(tr("ui metronome tooltip"))
        self.label_bpm.setText(tr("ui bpm unit"))
        if self.chk_metronome.isChecked():
            self.label_beat.setText(tr("ui beat count").format(beat="—"))
        else:
            self.label_beat.setText("")
        self.box_controls.setTitle(tr("ui live controls"))
        self.box_monitor.setTitle(tr("ui live monitor"))
        self.box_feedback.setTitle(tr("ui live feedback"))
        self.chk_record_audio.setText(tr("ui record audio"))
        self.chk_record_audio.setToolTip(tr("ui record audio tooltip"))
        self.btn_review_back.setText(tr("ui back to results"))
        self.btn_review_back.setToolTip(tr("ui back to results tooltip"))
        self.btn_live_audition_stop.setText(tr("ui piano stop audition"))
        self.btn_live_audition_stop.setToolTip(tr("ui piano stop audition tooltip"))
        self._render_note_audition()  # the readout follows the language
        self.refresh_intonation()  # the feedback line follows the language
        if self._reviewing:
            self.label_status.setText(tr("ui live roll review"))
        if self._replacement_mode:
            self.label_hint.setText(tr("ui replacement prompt"))
            self.btn_stop_live.setText(tr("ui finish replacement"))

    # ------------------------------------------------------------------
    # the frozen roll's audition readout (Space on the captured roll)
    # ------------------------------------------------------------------
    def _audition_note_cb(self, index: int) -> None:
        """Space picked a captured note: ask the window to play its slice."""
        self._audition_note(index)

    def _stop_audition_cb(self) -> None:
        """The readout's stop action shares the window's ONE stop path."""
        self._stop_audition()

    def show_note_audition(self, index: int, midi: int, count: int) -> None:
        """Identify the captured note sounding from Space and expose its stop."""
        self._note_audition = (index, midi, count)
        self._render_note_audition()

    def stop_note_audition(self) -> None:
        """Hide the note-specific readout when its audition ends."""
        self._note_audition = None
        self._render_note_audition()

    def _render_note_audition(self) -> None:
        """Render the current live-audition fact in the active language.

        The stop action sits at its permanent place in the page's ONE
        focus order; Qt skips it while hidden, so showing or hiding the
        readout never relinks the chain.
        """
        current = self._note_audition
        self.label_live_audition.setVisible(current is not None)
        self.btn_live_audition_stop.setVisible(current is not None)
        if current is not None:
            index, midi, count = current
            self.label_live_audition.setText(
                tr("ui piano audition label").format(
                    note=note_display_pair(midi), i=index + 1, n=count
                )
            )

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

    def set_replacement_mode(self, active: bool) -> None:
        """Show the replacement-specific prompt and finish action."""
        self._replacement_mode = active
        self.retranslate()

    def set_comparison_notes(self, notes, moved=None) -> None:
        """Remember the final reading beside the frozen captured roll."""
        self._comparison_notes = (list(notes), set(moved or ()))
        self._render_comparison()

    def set_comparison_visible(self, visible: bool) -> None:
        """Toggle only the final-reading overlay, leaving captured notes intact."""
        self._comparison_visible = visible
        self._render_comparison()

    def _render_comparison(self) -> None:
        """Render the cached final reading if it is enabled and available."""
        if self._comparison_visible and self._comparison_notes is not None:
            notes, moved = self._comparison_notes
            self.piano_live.set_comparison_notes(notes, moved=moved)
        else:
            self.piano_live.clear_comparison()

    def enter_review(self) -> None:
        """Freeze-frame the last take: the roll as it was captured.

        The stopped take's closed notes stay on this page's roll; review
        makes that reading reachable from the results page and offers the
        way back to the final analysis.
        """
        self._reviewing = True
        self.btn_review_back.setVisible(True)
        self.label_status.setText(tr("ui live roll review"))

    def exit_review(self) -> None:
        """Leave the frozen roll (a fresh take or the back action)."""
        if not self._reviewing:
            return
        self._reviewing = False
        self.btn_review_back.setVisible(False)

    def review_active(self) -> bool:
        """Whether the page is showing the frozen roll for a stopped take."""
        return self._reviewing

    # ------------------------------------------------------------------
    # the named surface (ADR 0039): the window calls verbs, it never
    # reaches into this page's widgets
    # ------------------------------------------------------------------
    def begin_take_render(self, *, notation: bool) -> None:
        """A starting take: the roll locked pick-less, in notation mode,
        the staff cleared — the window keeps the derived curve handle."""
        self.piano_live.set_selectable(False)  # a running take is never picked
        self.piano_live.begin_live()
        self.piano_live.set_notation_mode(notation)
        self.staff_live.clear()

    def end_take_render(self) -> None:
        """The take is over: closed notes stay frozen, now pickable."""
        self.piano_live.set_live_preview(None)
        self.piano_live.set_selectable(True)
        self.piano_live_keys.set_pressed()
        self.staff_live.clear()

    def clear_preview(self) -> None:
        """Drop the sounding note's preview bar (an aborted feed)."""
        self.piano_live.set_live_preview(None)

    def set_cursor(self, t: float) -> None:
        """Move the live roll's follow cursor (the 50 ms tick normally owns it)."""
        self.piano_live.set_live_cursor(max(0.0, t))

    def announce(self, text: str) -> None:
        """The page's status line (the window supplies the translated sentence)."""
        self.label_status.setText(text)

    def show_beat(self, text: str) -> None:
        """The metronome's beat counter line ("" = hidden)."""
        self.label_beat.setText(text)

    def set_bpm_enabled(self, on: bool) -> None:
        """The metronome's BPM control follows the click's state."""
        self.spin_bpm.setEnabled(on)

    def set_bpm_silently(self, bpm: float) -> None:
        """Seed the BPM spin without firing its valueChanged (the take's
        published grid feeds the metronome; the spin stays the user's
        override — a silent update, never a program click)."""
        self.spin_bpm.blockSignals(True)
        self.spin_bpm.setValue(int(round(bpm)))
        self.spin_bpm.blockSignals(False)

    def bpm(self) -> float:
        """The BPM spin's current value (the metronome reads it)."""
        return self.spin_bpm.value()

    def record_audio_checked(self) -> bool:
        """Whether the take should write its WAV (read at STOP, never live)."""
        return self.chk_record_audio.isChecked()

    def begin_take(self) -> None:
        """A new take: drop the previous pitch cloud and its curve."""
        live = self._state()
        live.pitch_t = []
        live.pitch_m = []
        live.t = 0.0
        self._comparison_notes = None
        self._comparison_visible = True
        self.piano_live.clear_comparison()
        self.accuracy.reset()
        self.intonation.reset()
        self.refresh_intonation()
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
            tr("ui live progress").format(t=clock(t), notes=len(notes), meter=live.grid.meter.label)
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
            self.watch_silence(t, rms)  # the take has started: phrase gaps
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
        parts = "  ".join(f"{note_display_pair(m)} {v:.0f}¢" for m, v in reversed(items[-6:]))
        self.label_accuracy.setText(tr("ui accuracy title") + " " + parts)

    def on_live_note_closed(self, note: LiveNote) -> None:
        """One closed live note enters the intonation feedback.

        The core mints the step (a note IS closed); the page only feeds
        the accumulator and repaints. Reading through the render tick
        keeps the single-writer rule for the live render.
        """
        self.intonation.add(note)
        self.refresh_intonation()

    def refresh_intonation(self) -> None:
        """Render the current summary with the report's shared painter.

        A new take starts with zero notes, and `intonation_lines` renders
        that empty summary as no text.
        """
        self.label_intonation.setText(" · ".join(intonation_lines(self.intonation.summary())))

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
