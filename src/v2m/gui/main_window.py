"""Main window of the Voix -> MIDI application."""

from __future__ import annotations

import copy
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import audio_io, capture_failure, driver_fx
from ..core.audio_io import LiveCapture
from ..core.compare import invalidate as invalidate_breakdown
from ..core.compare import match as compare_match
from ..core.continuity import coherence_report
from ..core.edits import (
    CorrectPitchIntent,
    DeleteNotesIntent,
    EditHistory,
    EditMove,
    EditOutcome,
    EditState,
    MoveNoteIntent,
    NoteEditIntent,
    apply_note_edit,
    snapshot_state,
)
from ..core.grid import BeatGrid
from ..core.i18n import get_language, note_display_pair, set_language, tr
from ..core.live_session import LiveEvent, LiveTake
from ..core.liveseg import LiveNoteSegmenter
from ..core.metronome import Metronome, qt_scheduler
from ..core.pipeline import AnalysisResult, AnalyzeOptions, FileAnalysis
from ..core.playback import NotePlayer, PlaybackSession
from ..core.take_export import ExportOptions, ExportWarning, write_take
from ..core.takes import (
    automatic_reading_dir,
    corrected_readings_dir,
    live_export_dir,
)
from . import widgets
from .analysis_workers import AnalysisThread, FullReanalysisThread, SignalAnalysisThread
from .analysis_workflow import AnalysisWorkflow, WorkflowPhase
from .audition_controller import AuditionController, AuditionRequest
from .finder_page import FinderPage
from .home_page import HomePage
from .live_page import LIVE_SILENCE_WARN_S, LivePage
from .live_session_controller import LiveSessionController
from .results_page import ResultsPage
from .widgets import ErrorBanner

# the opening size, the floor and the position come from ONE policy module:
# the window reads the screen it runs on and applies the plan it returns
# (the size and floor are no longer window constants)
from .window_geometry import WindowPlan, window_plan

AUDIO_EXTS = audio_io.AUDIO_EXTS  # single source of truth (core.audio_io)
EXPORT_DIRECTORY_FILE = Path.home() / ".v2m_export_dir"
INPUT_DEVICE_FILE = Path.home() / ".v2m_input_device"
THEME_FILE = Path.home() / ".v2m_theme"


def audio_filter() -> str:
    """File-dialog filter in the current language.

    Built on demand: `tr()` must run after the persisted language is
    restored, so this cannot be a module constant.
    """
    return tr("ui audio filter")


LiveFrame = tuple[float, float] | None


class LiveRenderState:
    """EVERYTHING the GUI renders from one live take, owned as a whole.

    The take's signal, closed notes, audio choice, pitch cloud and published
    grid travel together in one state object. `on_start_live` replaces the
    whole object, so no buffer or fact from the prior take can survive. The
    tests fake a mid-take state by building
    THIS object instead of driving the whole window; the window keeps the
    familiar `win._live_*` faces as thin properties over it.
    """

    def __init__(self) -> None:
        self.signal: np.ndarray | None = None  # the take's audio (float32)
        self.record_audio = False  # the box, read at take START
        self.closed_notes: list = []  # handed over by on_stop_live
        self.last_take_notes: list = []  # for the report's live_vs_final
        self.pitch_t: list[float] = []  # the live pitch cloud (x)
        self.pitch_m: list[float] = []  # ...and its pitches (y)
        self.curve = None  # the cloud's plot item
        self.t = 0.0  # capture time of the last block
        self.grid = BeatGrid(120.0)  # until the take publishes its own
        self.accuracy_target = "live"  # where a plot click seeks
        self.silence_since: float | None = None  # silence watch (ui live silence)
        # the PRE-TAKE low-level watch (ui live low mic): the mic's first
        # clear block arms the take (heard_once) and disarms this — the
        # take is then judged by `watch_silence`. One swap per take: the
        # watch re-arms itself when a new take replaces the state.
        self.heard_once = False  # any block above the gate yet
        self.low_since: float | None = None  # pre-take low-level watch


class LiveThread(QThread):
    """Qt adapter around the shared :class:`LiveTake` lifecycle.

    Recognition lives in `core.live_session`; this thread only shuttles its
    typed events onto Qt signals so the GUI thread can render them. The
    final analysis also runs here (it costs seconds), and its result lands
    on `analysisFinished` with the signal bytes the results page needs.
    """

    blockStats = pyqtSignal(float, float)
    pitchFrame = pyqtSignal(float, float)
    noteFrame = pyqtSignal(object)
    finderFrame = pyqtSignal(object, object, object)
    gridChanged = pyqtSignal(object)  # BeatGrid: the take's live grid policy
    liveNoteClosed = pyqtSignal(object)  # one closed LiveNote (intonation feed)
    captureOpened = pyqtSignal(bool)  # the take opened (True = raw/exclusive)
    analysisFinished = pyqtSignal(object, bytes)
    failed = pyqtSignal(str, bytes)  # the error + the take's signal (the trace reads it)

    def __init__(
        self,
        device: int | None = None,
        device_name: str | None = None,
        exclusive: bool = False,
        stream_factory: "audio_io.StreamFactory | None" = None,
    ):
        super().__init__()
        self._stop = False
        self.device = device
        self.device_name = device_name or "System default"
        self.exclusive = exclusive
        # None = the real microphone; a test injects an opener that refuses,
        # exactly as `LiveTake` allows (never a patch of `audio_io.sd`)
        self.stream_factory = stream_factory
        self.take: LiveTake | None = None
        # A capture that never OPENED is the microphone's own failure and earns
        # the page's names; a later failure is the take's. `failure_key` is set
        # only in the first case, and only there does the window rewrite the
        # sentence.
        self.started = False
        self.failure_key = ""

    def _on_take_event(self, event: LiveEvent) -> None:
        """Forward one LiveTake event to its Qt signal (worker thread)."""
        if event.kind == "block":
            self.blockStats.emit(event.time_s, event.rms)
        elif event.kind == "pitch" and event.pitch is not None:
            self.pitchFrame.emit(*event.pitch)
        elif event.kind == "note":
            self.noteFrame.emit(event.pitch)
            self.finderFrame.emit(event.pitch, event.confirmed_pitch, event.heard_pitch)
        elif event.kind == "note_closed":
            # the grid is published by its single owner (LiveTake); the GUI
            # consumes it, it never re-derives the approximate BPM itself
            self.gridChanged.emit(event.grid)
            if event.note is not None:
                self.liveNoteClosed.emit(event.note)
        elif event.kind == "failed" and event.error:
            # the take's signal rides along: the failure diagnostic (the
            # trace) needs the audio to name the folder and the duration
            sig = event.signal.tobytes() if event.signal is not None else b""
            self.failed.emit(event.error, sig)

    def run(self) -> None:
        self.take = LiveTake(
            on_event=self._on_take_event,
            device=self.device,
            exclusive=self.exclusive,
            stream_factory=self.stream_factory,
        )  # default thresholds (pipeline)
        try:
            self.take.start()
            self.started = True
            # what the capture really OPENED (a refused raw open falls back
            # silently): the page must name the mode it actually got
            self.captureOpened.emit(self.take.raw_capture)
            while not self._stop:
                self.msleep(50)
            self.take.stop()
            result, signal = self.take.finalize()
            self.analysisFinished.emit(result, signal.tobytes())
        except Exception as error:
            # a finalization failure was already published as a "failed"
            # event by LiveTake — don't report the same error twice
            if self.take is None or not self.take.error_emitted:
                # a stream that never opened is the microphone's failure, and
                # only it earns the page's three names (see `capture_failure`)
                if not self.started:
                    self.failure_key = capture_failure.capture_failure_key(error)
                    # the page keeps the same pair: the sentence names the cause,
                    # the console keeps what the driver actually said
                    print(f"[live] capture failed: {error}")
                # no event carried the signal (the failure happened outside
                # the take): the diagnostic degrades to "no audio", never a
                # guess about the take's content
                self.failed.emit(str(error), b"")

    @property
    def segmenter(self) -> LiveNoteSegmenter | None:
        return self.take.segmenter if self.take is not None else None

    @property
    def capture(self) -> LiveCapture | None:
        return self.take.capture if self.take is not None else None

    @property
    def raw_capture(self) -> bool:
        return self.take.raw_capture if self.take is not None else False

    def stop(self) -> None:
        self._stop = True

    def elapsed(self) -> float:
        return self.take.elapsed if self.take is not None else 0.0


class MainWindow(QMainWindow):
    playbackFailed = pyqtSignal(str)
    result: AnalysisResult | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        get_language()  # restore the persisted choice before building the UI
        self.setWindowTitle(tr("ui window title"))
        # The window opens FOR the screen it runs on (gui/window_geometry.py
        # owns the policy): its default 1280x780 hangs past the edges of a
        # scaled laptop screen, and its old fixed 900x560 floor left the
        # window taller than the desktop with no way to shrink it back on.
        self._screen_hooked = False
        self._apply_screen_plan(self._screen_plan())
        self.setAcceptDrops(True)

        self.result = None  # last AnalysisResult
        self._live = LiveRenderState()
        self.analyze_thread: AnalysisThread | None = None
        self.live_thread: LiveThread | None = None
        # the note finder's state lives IN the page (finder_page.FinderPage:
        # session + thread) — the window keeps no finder mirror
        self.source_path: str | None = None
        self._signal_cache: dict[str, np.ndarray] = {}  # source -> decoded audio
        # the accuracy history lives IN the live page (its only writer);
        # the archive below consumes it through the page
        self._analysis_started_at: float = 0.0  # cache-status timing (perf_counter)
        self._replay_worker = None  # ReplayWorker: the take's own cross-check
        self._workflow = AnalysisWorkflow()
        self._signal_analysis_thread: SignalAnalysisThread | None = None
        self._full_reanalysis_thread: FullReanalysisThread | None = None
        # the note edits' undo/redo: whole-take states (core.edits owns the
        # history; this window snapshots its facts and applies restorations)
        self._edit_history = EditHistory()
        # The first completed live interpretation is held separately from
        # later readings and edits so export can preserve it unchanged.
        self._automatic_live_result: AnalysisResult | None = None
        self._automatic_live_signal: np.ndarray | None = None
        # the take in progress, ONE owner: stem (minted once), signal (the
        # event's bytes), replay queue (survives a new take), sidecar and
        # timer decisions — this window is its Qt adapter (applies its
        # Verdicts)
        self.take_ctl = LiveSessionController()
        self.metronome = None  # created when the user opts in
        self.playback = PlaybackSession(audio_io.SR, on_error=self._on_playback_error)
        self.player: NotePlayer | None = None
        self._voice_player: NotePlayer | None = None
        # the short auditions' lifecycle has ONE owner: identity, the
        # Space toggle, replacements and the drag's two-clip sequence live
        # in the controller — this window only prepares clips and applies
        # the returned transitions to the pages' note-only readouts
        self._auditions = AuditionController(self.playback)
        self.playbackFailed.connect(self._show_playback_error)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)  # ~30 fps playhead updates
        self._play_timer.timeout.connect(self._on_play_tick)
        # the calibration card's collector: one held pitch per round
        self._calibration_timer = QTimer(self)
        self._calibration_timer.setInterval(400)
        self._calibration_timer.timeout.connect(self._calibration_feed_tick)
        self._calibration_hz = None
        # the pitch's ARRIVAL time: a round counts only while the note is
        # still sounding (a stale value would grade a note that ended)
        self._calibration_at = 0.0
        # when the CURRENT pitch settled: a round needs the note to hold
        # (below CALIBRATION_HOLD_S), never one frame
        self._calibration_stable_since = 0.0
        # the card's OWN mic, opened when no take is capturing
        self._calibration_thread: LiveThread | None = None
        self._selected_input_key: str | None = self._read_input_device_key()
        self._theme = self._read_theme()
        widgets.set_theme(self._theme)
        self._active_input_label = ""
        self._input_device_fell_back = False
        # Ctrl+Z / Ctrl+Y (StandardKey.Redo covers Ctrl+Shift+Z too): the
        # handlers no-op when there is nothing to restore
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self.on_undo_edit)
        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self.on_redo_edit)
        # Del removes the SELECTED range as one undoable edit; the handler
        # itself guards (another page, no selection → silent no-op), and a
        # focused line edit still eats the key for its own text
        delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self)
        delete_shortcut.activated.connect(self.on_delete_selection)

        # Single error banner shown above every page (no modal error popups).
        self.stack = QStackedWidget()
        self.error_banner = ErrorBanner()
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.error_banner)
        root.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        self.page_home = HomePage(
            open_file=self.on_open_file,
            start_live=self.on_start_live,
            open_finder=self.on_open_finder,
            toggle_language=self.on_toggle_language,
            snapshot_factory=self._coherence_snapshot,
            show_status=lambda message, timeout: self.statusBar().showMessage(message, timeout),
            on_calibrate_toggled=self._on_calibrate_toggled,
            input_devices=self._input_device_choices,
            driver_note=driver_fx.enhancement_note,
            selected_input_key=self._selected_input_key,
            input_device_changed=self._on_input_device_changed,
            theme=self._theme,
            theme_changed=self._on_theme_changed,
        )
        # The results page is a Page module (ADR 0010's full shape): it owns
        # its widgets (waveform, staff, roll, keyboard, options, play
        # buttons) and its rendering behavior; the window injects the user
        # actions that drive window state and keeps the FACTS (result,
        # audio cache, playback session, exports). Widget readers go
        # through `page_results.<widget>`.
        self.page_results = ResultsPage(
            on_seek=self.on_seek,
            correct_note=self.on_correct_note,
            note_moved=self.on_note_moved,
            replace_phrase=self.on_start_phrase_replacement,
            finish_replacement=self.on_finish_phrase_replacement,
            toggle_play=self.on_toggle_play,
            toggle_voice=self.on_toggle_voice,
            toggle_both=self.on_toggle_both,
            piano_volume_changed=self.on_piano_volume_changed,
            transpose_changed=self.on_transpose_changed,
            export_again=self.on_export_again,
            reinterpret=self.on_reinterpret,
            full_reanalyze=self.on_full_reanalyze,
            open_file=self.on_open_file,
            back_home=self.on_back_home,
            undo_edit=self.on_undo_edit,
            redo_edit=self.on_redo_edit,
            jump_edit_state=self.on_jump_edit_state,
            accept_replacement=self._on_accept_replacement,
            cancel_replacement=self._on_cancel_replacement,
            audition_replacement=self.on_audition_replacement,
            audition_original=self.on_audition_original,
            show_live_roll=self.on_show_live_roll,
            open_archived_comparison=self.on_open_archived_comparison,
            audition_note=self.on_audition_note,
            audition_pitch=self.on_audition_key,
            stop_audition=self.stop_playback,
            toggle_live_comparison=self.on_toggle_live_comparison,
        )
        # The live page is a Page module (ADR 0010's full shape): it owns
        # its widgets AND its behavior — the window composes it, injecting
        # the callables it cannot own. All widget readers (tests, smoke
        # script, the window itself) go through `page_live.<widget>`;
        # `_live_timer` is the page's timer.
        self.page_live = LivePage(
            self,
            state_getter=lambda: self._live,
            thread_getter=lambda: self.live_thread,
            on_stop=self.on_stop_live,
            toggle_metronome=self.on_metronome_toggled,
            metronome_bpm_changed=self.on_metronome_bpm_changed,
            expected_note=self._expected_midi,
            silence_message=lambda msg: self.statusBar().showMessage(msg, 2_500),
            error_message=lambda msg: self.show_error(msg, timeout=10_000),
            gate_rms=lambda: audio_io.GATE_RMS,
            on_review_back=self.on_exit_live_roll,
            audition_note=self.on_audition_live_note,
            stop_audition=self.stop_playback,
        )
        # The finder page is a Page module (ADR 0010's shape): it owns its
        # widgets AND its behavior — window dependencies injected (where
        # home is, where errors go), thread injected as a factory. The
        # window composes; it keeps no finder mirror.
        self.page_finder = FinderPage(
            thread_factory=self._new_finder_input_thread,
            go_home=lambda: self.stack.setCurrentWidget(self.page_home),
            show_error=self._on_error,
            clear_error=self.error_banner.clear,
            audition_note=self.on_audition_key,
        )
        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_results)
        self.stack.addWidget(self.page_live)
        self.stack.addWidget(self.page_finder)
        self.retranslate()
        self._refresh_input_device_status()

    # Pages are Page modules (gui/*_page.py); MainWindow owns the window
    # state (result, cache, playback, captures) and the user callbacks.

    # ------------------------------------------------------------------
    # The screen the window runs on. `gui/window_geometry.py` owns the
    # POLICY (how big a window opens on a given screen, and how far it may
    # shrink); these methods are only the adapter that reads the screen and
    # applies the returned plan.
    # ------------------------------------------------------------------
    def _screen_plan(self, screen=None) -> WindowPlan:
        screen = screen or self.screen() or QApplication.primaryScreen()
        rect = screen.availableGeometry()
        return window_plan(rect.width(), rect.height(), rect.x(), rect.y())

    def _apply_screen_plan(self, plan: WindowPlan) -> None:
        """Open at the plan's floor, size and position."""
        self.setMinimumSize(plan.min_width, plan.min_height)
        self.resize(plan.width, plan.height)
        self.move(plan.x, plan.y)

    def _adapt_floor_to_screen(self, screen) -> None:
        """Re-clamp the FLOOR to the screen the window now sits on.

        Dragging the window onto a smaller monitor must leave it shrinkable
        there; back on a large one the floor returns to its full height. The
        window's own size and position stay as the user left them — the move
        was their decision, not ours.
        """
        plan = self._screen_plan(screen)
        self.setMinimumSize(plan.min_width, plan.min_height)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's own name
        super().showEvent(event)
        # one hook per window: the screen the user drags the window onto is
        # only reachable through the window handle, which exists from here
        handle = self.windowHandle()
        if handle is not None and not self._screen_hooked:
            self._screen_hooked = True
            handle.screenChanged.connect(self._adapt_floor_to_screen)

    # ------------------------------------------------------------------
    # The live render state, read through its familiar faces: one owner
    # (LiveRenderState), nine thin properties. Tests fake a mid-take state
    # by building the ONE object; a stale value from a previous take is
    # structurally impossible (on_start_live swaps the whole object).
    # ------------------------------------------------------------------
    @property
    def _live_signal(self):
        return self._live.signal

    @_live_signal.setter
    def _live_signal(self, value) -> None:
        self._live.signal = value

    @property
    def _live_record_audio(self) -> bool:
        return self._live.record_audio

    @_live_record_audio.setter
    def _live_record_audio(self, value) -> None:
        self._live.record_audio = value

    @property
    def _closed_live_notes(self) -> list:
        return self._live.closed_notes

    @_closed_live_notes.setter
    def _closed_live_notes(self, value) -> None:
        self._live.closed_notes = value

    @property
    def _last_take_live_notes(self) -> list:
        return self._live.last_take_notes

    @_last_take_live_notes.setter
    def _last_take_live_notes(self, value) -> None:
        self._live.last_take_notes = value

    @property
    def _live_pitch_t(self) -> list:
        return self._live.pitch_t

    @_live_pitch_t.setter
    def _live_pitch_t(self, value) -> None:
        self._live.pitch_t = value

    @property
    def _live_pitch_m(self) -> list:
        return self._live.pitch_m

    @_live_pitch_m.setter
    def _live_pitch_m(self, value) -> None:
        self._live.pitch_m = value

    @property
    def _live_curve(self):
        return self._live.curve

    @_live_curve.setter
    def _live_curve(self, value) -> None:
        self._live.curve = value

    @property
    def _live_t(self) -> float:
        return self._live.t

    @_live_t.setter
    def _live_t(self, value) -> None:
        self._live.t = value

    @property
    def _live_grid(self) -> BeatGrid:
        return self._live.grid

    @_live_grid.setter
    def _live_grid(self, value) -> None:
        self._live.grid = value

    @property
    def _accuracy_plot_target(self) -> str:
        return self._live.accuracy_target

    @_accuracy_plot_target.setter
    def _accuracy_plot_target(self, value) -> None:
        self._live.accuracy_target = value

    # ------------------------------------------------------------------
    # Language (fr / en, persisted)
    # ------------------------------------------------------------------
    def on_toggle_language(self) -> None:
        """Toggle fr <-> en and retranslate the whole UI immediately."""
        set_language("en" if get_language() == "fr" else "fr")
        self.retranslate()

    def _read_theme(self) -> str:
        """Load the saved theme; malformed or missing values keep dark mode."""
        try:
            value = THEME_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return "dark"
        return value if value in widgets.THEMES else "dark"

    def _write_theme(self, theme: str) -> None:
        """Persist the selected theme best-effort."""
        try:
            THEME_FILE.write_text(theme, encoding="utf-8")
        except OSError:
            pass

    def _on_theme_changed(self, theme: str) -> None:
        """Apply the selected theme immediately and keep all page facts in sync."""
        if theme not in widgets.THEMES:
            raise ValueError(f"unknown theme: {theme}")
        app = QApplication.instance()
        previous_theme = self._theme
        self._theme = theme
        try:
            widgets.set_theme(theme)
            if app is not None:
                app.setStyleSheet(widgets.theme_stylesheet())
            self.page_home.set_theme(theme)
        except Exception:
            self._theme = previous_theme
            widgets.set_theme(previous_theme)
            self.page_home.set_theme(previous_theme)
            if app is not None:
                app.setStyleSheet(widgets.theme_stylesheet())
            raise
        self._write_theme(theme)
        self._refresh_theme_views()

    def _refresh_theme_views(self) -> None:
        """Repaint custom-drawn and locally styled widgets from current tokens."""
        from . import staff

        for widget in self.findChildren(QWidget):
            if isinstance(widget, widgets.PianoRoll):
                widget.setBackground(widgets.BG_APP)
                # pyqtgraph's own words follow the palette too, or a theme switch
                # leaves the pitch rows and the seconds in the other theme's ink.
                widgets.ink_plot(widget)
                widget.set_notes(widget._last_notes, force=True)
                widget.set_live_preview(getattr(widget, "_live_preview", None))
                widget._render_selection()
            elif isinstance(widget, widgets.WaveformView):
                widget.setBackground(widgets.BG_APP)
                widgets.ink_plot(widget)
            elif isinstance(widget, staff.StaffView):
                widget.update()
            elif isinstance(widget, widgets.PianoView):
                widget.update()
            elif isinstance(widget, widgets.TuningMeter):
                widget.update()
            elif isinstance(widget, widgets.AccuracyPlot):
                widget.update()
            elif isinstance(widget, widgets.LiveIndicator):
                widget.set_theme()
        self.page_home.refresh_theme()
        self.page_home.retranslate()
        self.page_finder.refresh_theme()
        self.page_live.refresh_theme()
        self.page_results.refresh_theme()

    def retranslate(self) -> None:
        """Refresh every window string in the current language."""
        self.setWindowTitle(tr("ui window title"))
        self.page_home.retranslate()
        self.page_home.set_theme(self._theme)
        self.page_home.refresh_theme()
        self.page_finder.retranslate()
        self.page_results.retranslate()
        self.page_live.retranslate()
        self._refresh_live_accuracy()
        self._refresh_expected_note(self._live_t)
        for view in (self.page_results.staff, self.page_live.staff_live):
            view.setToolTip(tr("ui staff tooltip"))
        for roll in (self.page_results.piano, self.page_live.piano_live):
            roll.set_axis_labels(tr("ui axis note"), tr("ui axis time"))
        if (
            self.live_thread is None or not self.live_thread.isRunning()
        ) and not self.page_live.review_active():
            self.page_live.announce(tr("ui mic preparing"))
            self.page_live._set_input_state("ui mic waiting")

    # ------------------------------------------------------------------
    # Errors (inline red banner)
    # ------------------------------------------------------------------
    def show_error(self, message: str, timeout: int = 0) -> None:
        """Show an error in the red banner; auto-hide after `timeout` ms if set."""
        self.error_banner.show_error(message)
        if timeout:
            QTimer.singleShot(timeout, self.error_banner.clear)

    def _on_error(self, msg: str) -> None:
        self.show_error(msg, timeout=8_000)

    def _on_playback_error(self, message: str) -> None:
        """Publish an audio-thread failure to the GUI thread."""
        self.playbackFailed.emit(message)

    def _show_playback_error(self, message: str) -> None:
        self.show_error(tr("err ui playback").format(e=message), timeout=8_000)

    # ------------------------------------------------------------------
    # Playback (ear-check of the analysis, synced playhead). The session
    # and its clock live HERE (a window fact); the page renders it.
    # ------------------------------------------------------------------
    def _piano_gain(self) -> float:
        """Slider 0-100 -> render gain (70 = neutral, 0 = mute)."""
        return self.page_results.piano_gain()

    def _rebuild_piano(self) -> None:
        """Re-render the piano at the current gain; resume it if it sounded.

        Only the piano source is rebuilt: a running voice keeps sounding,
        so the volume can be tuned live during the piano-over-voice mode.
        An active short audition is left untouched: its clip stays staged
        and sounding, and the new gain applies on the next start (the
        approved spec — volume never interrupts a decision clip).
        """
        was_playing = self.playback.playing
        pos = self.playback.position() if was_playing else 0.0
        mode = self.playback.mode
        if self.result is None:
            return
        self.player = NotePlayer(
            self.result.notes,
            sr=audio_io.SR,
            gain=self._piano_gain(),
            on_error=self._on_playback_error,
        )
        if self._auditions.active:
            self._sync_play_buttons()
            return
        self.playback.set_sources(
            piano=self.player.signal,
            voice=self._voice_player.signal if self._voice_player is not None else None,
        )
        if was_playing and mode:
            self.playback.play(mode, pos)
            self._play_timer.start()
        self._sync_play_buttons()

    def on_piano_volume_changed(self, _value: int) -> None:
        self._rebuild_piano()

    def on_transpose_changed(self, _value: int) -> None:
        """Rebuild both audible sources without re-running analysis."""
        shared_was_playing = self.playback.playing
        shared_pos = self.playback.position() if shared_was_playing else 0.0
        shared_mode = self.playback.mode
        self._rebuild_piano()
        voice_player = self._make_voice_player()
        if voice_player is None:
            return
        self._voice_player = voice_player
        if self._auditions.active:
            return  # the audition keeps sounding; settings apply next start
        if shared_was_playing and shared_mode:
            self.playback.set_sources(
                piano=self.player.signal if self.player is not None else None,
                voice=voice_player.signal,
            )
            self.playback.play(shared_mode, shared_pos)
            self._play_timer.start()

    # Playback. ONE player for the three sources (piano, voice, both):
    # `playback` owns the clock and the output stream, the tick renders
    # from it, so the playhead follows whichever mode is sounding. Every
    # new playback starts by stopping the previous one.

    def _voice_signal(self) -> np.ndarray | None:
        """Return the original audio for either a live or file analysis."""
        if self.result is None:
            return None
        signal = (
            self._live_signal
            if self.result.source == "live"
            else self._signal_cache.get(self.result.source)
        )
        return signal if signal is not None and len(signal) > 0 else None

    def _sync_play_buttons(self) -> None:
        self.page_results.sync_play_buttons(self.playback.mode if self.playback.playing else None)

    def _make_voice_player(self) -> NotePlayer | None:
        """A rendered voice source at the live transposition, or None without a take."""
        signal = self._voice_signal()
        if signal is None:
            return None
        return NotePlayer(
            signal=signal,
            sr=audio_io.SR,
            transpose=self.page_results.transpose_st(),
            on_error=self._on_playback_error,
        )

    def _start(self, kind: str, t: float = 0.0) -> None:
        """Start one source at `t`, stopping every other playback first."""
        if self.result is None:
            return
        self.stop_playback()
        voice_player: NotePlayer | None = self._voice_player
        if kind != "piano":
            voice_player = self._make_voice_player()
            self._voice_player = voice_player
        self.playback.set_sources(
            piano=self.player.signal if self.player is not None else None,
            voice=voice_player.signal if voice_player is not None else None,
        )
        if not self.playback.play(kind, t):
            return
        self._play_timer.start()
        self._sync_play_buttons()

    def on_toggle_play(self) -> None:
        # an active short audition never reads as "piano is playing": the
        # button always cancels it and starts THIS button's whole-take mode
        if self._auditions.active:
            self._start("piano")
            return
        if self.playback.playing and self.playback.mode == "piano":
            self.stop_playback()
            return
        self._start("piano")

    def on_toggle_voice(self) -> None:
        """Play the original recording through the shared timeline."""
        if self._auditions.active:
            self._start("voice")
            return
        if self.playback.playing and self.playback.mode == "voice":
            self.stop_playback()
            return
        self._start("voice")

    def on_toggle_both(self) -> None:
        """Piano over the voice (comparative listening), or stop both."""
        if self._auditions.active:
            self._start("both")
            return
        if self.playback.playing:
            self.stop_playback()
            return
        self._start("both")

    def on_seek(self, t: float) -> None:
        """Double-click on the piano roll: play from there (or just move).

        Every exit resolves what was sounding: an audition's readout must
        not outlive the tick that observes it.
        """
        if self.result is None or self.player is None:
            return
        self._play_timer.stop()
        self.page_results.set_playhead(t, follow=False)
        self.page_results.set_wave_position(t)
        # the TAKE's length: `playback.duration` is whatever was loaded
        # last — an audition leaves its short slice there, which would make
        # every later click read as "the end"
        if t >= self.result.duration_s - 0.05:
            self.stop_playback()
            return
        # Preserve the source selected by the user, including piano + voice.
        self._start(self.playback.mode or "piano", t)

    def _on_play_tick(self) -> None:
        if not self.playback.playing:
            self._play_timer.stop()
            self._sync_play_buttons()
            # ONE completion path for every audition kind: the controller
            # advances a sequence (the drag's second clip) or spends the
            # state; the returned facts name the readout to clear
            transition = self._auditions.note_clip_completed()
            self._apply_audition_transition(transition)
            if transition.active is not None:
                # the sequence's next clip is sounding: keep observing it
                self._play_timer.start()
            self.page_results.set_keys_playhead(-1.0)
            return
        t = self.playback.position()
        duration = self.playback.duration
        self.page_results.set_playhead(t)
        self.page_results.set_wave_position(t)
        self.page_results.set_keys_playhead(t)
        self.page_results.set_play_pos(t, duration)
        self._sync_play_buttons()

    def stop_playback(self) -> None:
        self._play_timer.stop()
        if self._auditions.active:
            # the cancel owns the session's stop while an audition sounds:
            # ONE stop per stop request, the identity cleared, and the
            # returned fact names the readout to clear
            self._apply_audition_transition(self._auditions.cancel())
        else:
            self.playback.stop()
        self.page_results.reset_play_buttons()

    def on_back_home(self) -> None:
        """Leave the analyze page: the playback must NOT keep sounding.

        One owner for the navigation back, and it stops the audio first —
        a page switch alone would leave the PlaybackSession running.
        """
        self.stop_playback()
        self.stack.setCurrentWidget(self.page_home)

    def on_show_live_roll(self) -> None:
        """Review the frozen live roll: the notes as the take captured them.

        The stopped take's closed notes stay on the live page's roll after
        the final analysis (`on_stop_live` hands them over); this is the way
        back to that reading from the results page, without starting a new
        take. Only a live take has one; a replacement capture is mid-flight,
        so it takes precedence over the detour.
        """
        if self.result is None or self.result.source != "live":
            return
        if self._workflow.phase in (
            WorkflowPhase.SPLICE_ANALYSIS,
            WorkflowPhase.REPLACEMENT_CAPTURE,
        ):
            return
        if self._workflow.phase is WorkflowPhase.REPLACEMENT_DECISION:
            self._abandon_pending_replacement()
        self.stop_playback()
        self.page_live.enter_review()
        self.stack.setCurrentWidget(self.page_live)

    def on_toggle_live_comparison(self, visible: bool) -> None:
        """Show or hide the final reading on the captured live roll."""
        self.page_live.set_comparison_visible(visible)

    def on_exit_live_roll(self) -> None:
        """Leave the frozen roll and return to the final analysis."""
        self.page_live.exit_review()
        if self.result is not None:
            self.stack.setCurrentWidget(self.page_results)

    def _current_analysis_signal(self) -> np.ndarray | None:
        """Return the exact audio buffer behind the currently shown result."""
        result = self.result
        if result is None:
            return None
        if result.source == "live":
            return self._live_signal
        return self._signal_cache.get(result.source)

    def _start_full_reanalysis(
        self,
        *,
        signal: np.ndarray,
        source: AnalysisResult,
    ) -> None:
        """Start pYIN again from retained audio without replacing the old view."""
        context = self._workflow.begin_full_reanalysis(source, signal)
        if context is None:
            return
        self.error_banner.clear()
        self.stop_playback()
        self.page_results.set_full_reanalysis(True)
        options = AnalyzeOptions(
            min_note_s=self.page_results.min_note_ms() / 1000.0,
            min_conf=AnalyzeOptions().min_conf,
        )
        worker = FullReanalysisThread(context.signal, source.sr, source.source, options)
        self._full_reanalysis_thread = worker
        worker.progress.connect(
            lambda fraction, message, token=context.token: self._on_full_reanalysis_progress(
                token, fraction, message
            )
        )
        worker.finishedOk.connect(
            lambda result, token=context.token: self._on_full_reanalysis_done(result, token)
        )
        worker.failed.connect(
            lambda error, token=context.token: self._on_full_reanalysis_failed(error, token)
        )
        worker.finished.connect(
            lambda current=worker: self._on_full_reanalysis_thread_finished(current)
        )
        worker.start()

    def on_full_reanalyze(self) -> None:
        """Rebuild pYIN evidence from audio instead of re-reading the observation."""
        source = self.result
        signal = self._current_analysis_signal()
        if source is None or signal is None or len(signal) == 0:
            self.show_error(tr("err full reanalysis audio"), timeout=6_000)
            return
        self._start_full_reanalysis(signal=signal, source=source)

    def _on_full_reanalysis_progress(
        self, token: int, fraction: float, message: str
    ) -> None:
        if self._workflow.is_current(token, WorkflowPhase.FULL_REANALYSIS):
            self.page_results.set_full_reanalysis_progress(fraction, message)

    def _on_full_reanalysis_thread_finished(self, worker: FullReanalysisThread) -> None:
        """Release only the worker whose run method just returned."""
        if self._full_reanalysis_thread is worker and not worker.isRunning():
            self._full_reanalysis_thread = None

    def _on_full_reanalysis_done(self, result: AnalysisResult, token: int) -> None:
        """Publish a completed fresh reading, preserving audio and take facts."""
        transition = self._workflow.accept_current(token, WorkflowPhase.FULL_REANALYSIS)
        if transition is None or transition.full_reanalysis is None:
            return
        context = transition.full_reanalysis
        source = context.source
        signal = context.signal
        result.source = source.source

        if source.source == "live":
            self._live_signal = signal
            self._signal_cache = {"live": signal}
        else:
            self._signal_cache = {source.source: signal}
        self.stop_playback()
        self._edit_history.clear()
        self._sync_edit_history()
        previous = source
        self.result = result
        self.player = NotePlayer(result.notes, sr=result.sr, gain=self._piano_gain())
        self._show_result(result, signal)
        self._publish_reading_diff(previous, result)
        if source.source == "live":
            self._mark_live_final_diff()
            from ..core.compare import breakdown, read_block

            prior = read_block(getattr(source, "live_vs_final", None))
            if self._last_take_live_notes:
                result.live_vs_final = breakdown(
                    self._last_take_live_notes,
                    result.notes,
                    result.grid,
                    dropped=prior.dropped if prior is not None else None,
                    drain_min=prior.drain_min if prior is not None else None,
                )
        self.page_results.finish_full_reanalysis()
        self.statusBar().showMessage(
            tr("ui status full reanalyzed").format(n=len(result.notes)), 8_000
        )

    def _on_full_reanalysis_failed(self, error: str, token: int) -> None:
        """Keep the prior reading and edits if a fresh evidence pass fails."""
        transition = self._workflow.fail_full_reanalysis(token)
        if transition is None or transition.full_reanalysis is None:
            return
        context = transition.full_reanalysis
        self.result = context.source
        self._show_result(context.source, context.signal)
        self.page_results.finish_full_reanalysis()
        self.show_error(tr("err full reanalysis failed").format(e=error), timeout=8_000)

    # ------------------------------------------------------------------
    # the before/after reading diff (core.reading_diff renders nothing:
    # the window decides WHEN two readings are comparable)
    # ------------------------------------------------------------------
    def _publish_reading_diff(self, before, after) -> None:
        """Compute and show one before/after comparison for the SAME audio.

        `before` is the reading that was SHOWN, `after` the one that just
        succeeded. If comparison fails, the previous diagnostic stays cleared:
        it must never be shown beside a reading it does not describe, nor cost
        the reading itself.
        """
        # The current reading has already been replaced by the caller. Drop
        # the previous diagnostic first: if this optional comparison fails,
        # it must not remain beside a reading it does not describe.
        self._clear_reading_diff()
        try:
            from ..core.reading_diff import compare_readings

            comparison = compare_readings(before, after)
        except Exception:
            return  # a diagnostic may never cost the reading it describes
        self.page_results.set_reading_diff(comparison)

    def _clear_reading_diff(self) -> None:
        """The shown presentation no longer describes this take's state
        (a new take/file, a manual edit, a replacement decision): drop it."""
        self.page_results.clear_reading_diff()

    def on_reinterpret(self) -> None:
        """Re-read the take's CACHED evidence with the current thresholds.

        The observation (cleaned curve + attacks) is already on the result:
        `interpret_observation` re-reads it in milliseconds — no decode, no
        pYIN. The min-note spin decides how long a sung note must hold
        before it counts; the transpose spin still applies at export. The
        re-read only refills the VIEWS: the user explicitly asked for a
        different reading, nothing is exported until they export again.
        """
        res = self.result
        if self._workflow.phase is not WorkflowPhase.IDLE:
            return
        if res is None or getattr(res, "observation", None) is None:
            self.show_error(tr("err reinterpret none"), timeout=6_000)
            return
        # a re-reading replaces the whole interpretation: the edit history's
        # snapshots describe the previous reading, so they start over
        self._edit_history.clear()
        self._sync_edit_history()
        self.stop_playback()
        opts = AnalyzeOptions(min_note_s=self.page_results.min_note_ms() / 1000.0)
        observation = res.observation
        assert observation is not None  # guarded above; narrows for the reader
        try:
            from ..core.pipeline import interpret_observation

            reinterpreted = interpret_observation(
                observation,
                tempo=res.tempo,
                source=res.source,
                opts=opts,
                report=lambda f, m: None,
            )
        except Exception as e:
            self.show_error(tr("err reinterpret failed").format(e=e), timeout=8_000)
            return
        # keep the take's identity and export payload, swap the reading
        reinterpreted.source = res.source
        reinterpreted.grid = res.grid
        previous = res
        self.result = reinterpreted
        self._show_result(reinterpreted)
        self._publish_reading_diff(previous, reinterpreted)

        self.statusBar().showMessage(
            tr("ui status reinterpreted").format(n=len(reinterpreted.notes)), 8_000
        )

    # ------------------------------------------------------------------
    # File analysis
    # ------------------------------------------------------------------
    def on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("ui choose file"), "", audio_filter())
        if not path:
            return
        self.start_analysis(path)

    def start_analysis(self, path: str) -> None:
        """Validate the input early, then run the analysis in a thread."""
        if self._workflow.phase is WorkflowPhase.FULL_REANALYSIS:
            return
        self.error_banner.clear()
        p = Path(path)
        if not p.exists():
            self.show_error(tr("err ui missing").format(path=path))
            self.stack.setCurrentWidget(self.page_home)
            return
        if p.suffix.lower() not in AUDIO_EXTS:
            self.show_error(
                tr("err ui unsupported format").format(
                    ext=p.suffix or "?",
                    fmts=", ".join(e.lstrip(".") for e in AUDIO_EXTS),
                )
            )
            return
        try:
            if p.stat().st_size == 0:
                self.show_error(tr("err ui empty").format(name=p.name))
                return
        except OSError as e:
            self.show_error(tr("err ui unreadable").format(e=e))
            return

        was_capturing_replacement = (
            self._workflow.phase is WorkflowPhase.REPLACEMENT_CAPTURE
        )
        self._abandon_pending_replacement()
        if was_capturing_replacement:
            self.on_stop_live()
        self.page_results.begin_analysis()
        self.stack.setCurrentWidget(self.page_results)
        self.source_path = path
        self.page_results.set_source_name(p.name)
        self.analyze_thread = AnalysisThread(path, AnalyzeOptions())
        self.analyze_thread.progress.connect(self._on_progress)
        self.analyze_thread.finishedOk.connect(self._on_analysis_done)
        self.analyze_thread.failed.connect(self._on_error)
        self.analyze_thread.start()
        self._analysis_started_at = time.perf_counter()

    def _on_progress(self, f: float, msg: str) -> None:
        self.page_results.set_progress(int(f * 100))
        name = Path(self.source_path).name if self.source_path else tr("ui no file")
        self.page_results.set_file_label(f"{name} — {msg}")

    def _coherence_snapshot(self):
        """The core snapshot is the one aggregation pass for every view."""
        from ..core import coherence as coh

        return coh.build_snapshot(coh.LIVE_TAKES_DIR)

    def on_show_coherence(self) -> None:
        """Thin public delegation to the home page's coherence view."""
        self.page_home.show_coherence()

    def _coherence_summary(self, res) -> str:
        """The one-line coherence verdict for the status bar (or '')."""
        try:
            observation = getattr(res, "observation", None)
            if observation is None:
                return ""
            block = coherence_report(observation, res.notes)
            checked, n = block["checked"], len(block["suspects"])
            if n:
                return tr("ui coherence suspects").format(n=n, checked=checked)
            return tr("ui coherence clean").format(checked=checked)
        except Exception:
            return ""

    def _on_analysis_done(self, outcome: FileAnalysis) -> None:
        res = outcome.analysis
        # a new file is a new take: its old splice decision is already
        # abandoned when the validated analysis starts.
        self.page_results.clear_reading_diff()
        self._edit_history.clear()
        self._sync_edit_history()
        self.result = res
        # retain the exact decoded samples used for this analysis: waveform
        # rendering must not reopen the source file.
        self._signal_cache = {res.source: outcome.audio}
        self._live_signal = None  # a file view must never show a live take
        self.stop_playback()
        self.player = NotePlayer(
            res.notes,
            sr=audio_io.SR,
            gain=self._piano_gain(),
        )
        self._show_result(res, outcome.audio)
        # The analysis remains editable; the export button writes artifacts.
        summary = self._coherence_summary(res)
        if summary:
            self.statusBar().showMessage(tr("ui status coherence").format(summary=summary), 10_000)
        # Keep the cache explanation last: before the file outcome carried
        # this fact, its queued signal ran after this handler and the
        # coherence summary. A synchronous write here would be overwritten.
        if outcome.cache_used:
            elapsed_ms = 1000.0 * (time.perf_counter() - self._analysis_started_at)
            self.statusBar().showMessage(tr("ui status cache").format(ms=elapsed_ms), 8_000)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def _export_results(
        self,
        save_path: Path | None = None,
        format_suffix: str | None = None,
    ) -> None:
        if self.result is None:
            return
        if self._workflow.phase in (
            WorkflowPhase.REPLACEMENT_DECISION,
            WorkflowPhase.SPLICE_ANALYSIS,
            WorkflowPhase.FULL_REANALYSIS,
        ):
            return  # a splice decision or fresh evidence pass is in progress
        res = self.result
        stem = self._export_stem(res)
        manually_corrected = bool(getattr(res, "manual_pitch_corrections", False))
        archive_sidecar = (
            self._stage_live_export(res, stem, manually_corrected)
            if res.source == "live"
            else False
        )
        archive_signal = (
            self._automatic_live_signal
            if manually_corrected and self._automatic_live_signal is not None
            else self._live_signal
        )
        options = ExportOptions(
            quantize_s=(res.quantize_step(16) if self.page_results.quantize_checked() else None),
            transpose=self.page_results.transpose_st(),
            record_audio=self._live_record_audio,
            pdf=(self.page_results.pdf_checked() or format_suffix == ".pdf"),
            live_notes=self._last_take_live_notes,
            manual_corrections=manually_corrected,
            automatic_result=self._automatic_live_result if res.source == "live" else None,
            archive_corrected_reading=manually_corrected and res.source == "live",
            archive_sidecar=archive_sidecar,
            save_path=save_path,
            format_suffix=format_suffix,
        )
        # ONE call owns the whole sequence (WAV -> artifacts -> PDF, the
        # sidecar archive, the retention trim); the window only RENDERS the
        # receipt — it never re-derives what was written.
        receipt = write_take(res, stem, signal=archive_signal, options=options)
        if receipt.failed:
            if receipt.pdf_status in ("missing_tool", "failed"):
                self._show_export_warning(ExportWarning("pdf", receipt.warning_detail("pdf")))
            else:
                detail = receipt.warning_detail("export")
                self.show_error(tr("err ui export failed").format(e=detail), timeout=8_000)
            return
        for warning in receipt.warnings:
            self._show_export_warning(warning)
        if receipt.pruned:
            self.statusBar().showMessage(
                tr("ui retention pruned").format(
                    n=len(receipt.pruned), stems=", ".join(receipt.pruned)
                ),
                8_000,
            )
        files = ", ".join(receipt.file_names())
        # the action acknowledges itself where the eye already is
        self.page_results.ack_export()
        self.statusBar().showMessage(tr("ui status exported").format(files=files), 10_000)
        self._refresh_exported_coherence()
        self._refresh_archive_comparison()
        # the take's WAV now exists on disk: its queued cross-check may
        # start — the controller's guards still hold (a running capture or
        # replay keeps it waiting); a file analysis's export pumps an empty
        # queue, the call is free
        self._pump_replay_queue()
        if receipt.saved_path is not None:
            self.statusBar().showMessage(
                tr("ui status saved file").format(path=receipt.saved_path), 10_000
            )

    def _export_filters(self) -> list[tuple[str, str]]:
        """The formats this installation can actually write, in display order."""
        formats = [
            (tr("ui format midi"), ".mid"),
            (tr("ui format musicxml"), ".musicxml"),
            (tr("ui format json"), ".json"),
            (tr("ui format csv"), ".csv"),
        ]
        # PDF selection is always visible: the format remains a useful
        # choice even before MuseScore/LilyPond is installed, and the
        # export receipt can explain the missing tool without producing
        # a partial output file.
        formats.insert(2, (tr("ui format pdf"), ".pdf"))
        return formats

    def _last_export_directory(self) -> Path | None:
        """Return the remembered destination while it still names a directory."""
        try:
            saved = EXPORT_DIRECTORY_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not saved:
            return None
        directory = Path(saved).expanduser()
        return directory if directory.is_dir() else None

    def _remember_export_directory(self, directory: Path) -> None:
        """Persist the selected folder without making settings a condition of export."""
        try:
            EXPORT_DIRECTORY_FILE.write_text(str(directory), encoding="utf-8")
        except OSError:
            # The export itself must still work when the preference location is read-only.
            pass

    def _choose_export_destination(self, res: AnalysisResult) -> tuple[Path, str] | None:
        """Ask for a filename and format, reopening in the last selected folder."""
        formats = self._export_filters()
        if res.source == "live":
            stem = self._export_stem(res)
            initial_dir = Path.cwd()
            default_name = stem.name
        else:
            source = Path(res.source)
            initial_dir = source.parent if str(source.parent) else Path.cwd()
            default_name = source.stem
        remembered_dir = self._last_export_directory()
        if remembered_dir is not None:
            initial_dir = remembered_dir
        default_filter, default_suffix = formats[0]
        suggested = initial_dir / f"{default_name}{default_suffix}"
        filters = ";;".join(label for label, _suffix in formats)
        selected, selected_filter = QFileDialog.getSaveFileName(
            self,
            tr("ui save export"),
            str(suggested),
            filters,
            initialFilter=default_filter,
        )
        if not selected:
            return None
        suffix_by_filter = dict((label, suffix) for label, suffix in formats)
        target = Path(selected).expanduser()
        suffix = suffix_by_filter.get(selected_filter)
        if suffix is None:
            suffix = next(
                (candidate for _label, candidate in formats if target.suffix.lower() == candidate),
                default_suffix,
            )
        if target.suffix.lower() != suffix:
            target = target.with_suffix(suffix)
        self._remember_export_directory(target.parent)
        return target, suffix

    def _show_export_warning(self, warning: ExportWarning) -> None:
        """Render one non-fatal export warning (the receipt carries values)."""
        if warning.kind == "wav":
            self.show_error(tr("err ui wav failed").format(e=warning.detail), timeout=8_000)
        elif warning.kind == "pdf":
            self.show_error(tr("err ui pdf").format(e=warning.detail), timeout=6_000)

    def _archived_live_readings(self) -> list[tuple[str, AnalysisResult]]:
        """Read the current take's JSON archives as comparison-only results."""
        take_dir = self.take_ctl.take_dir()
        stem_name = self.take_ctl.take_stem()
        if take_dir is None or stem_name is None:
            return []

        candidates: list[tuple[str, Path]] = []
        automatic_json = automatic_reading_dir(take_dir) / f"{stem_name}.json"
        if automatic_json.is_file():
            candidates.append((tr("ui compare automatic"), automatic_json))

        corrected_parent = corrected_readings_dir(take_dir)
        if corrected_parent.is_dir():
            corrected_dirs = sorted(
                (path for path in corrected_parent.glob("v*") if path.is_dir()),
                key=lambda path: int(path.name[1:]) if path.name[1:].isdigit() else -1,
            )
            for path in corrected_dirs:
                version = path.name[1:]
                if not version.isdigit():
                    continue
                report_path = path / f"{stem_name}.json"
                if report_path.is_file():
                    candidates.append(
                        (tr("ui compare corrected").format(version=int(version)), report_path)
                    )

        readings: list[tuple[str, AnalysisResult]] = []
        for label, report_path in candidates:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(report, dict) or not isinstance(report.get("notes"), list):
                    continue
                from ..core.notes import Note

                notes = []
                for item in report["notes"]:
                    if not isinstance(item, dict):
                        raise ValueError("invalid note row")
                    start = float(item["start_s"])
                    end = float(item["end_s"])
                    midi = int(item["midi"])
                    confidence = float(item.get("confidence", 0.0))
                    if not np.isfinite((start, end, confidence)).all() or end < start:
                        raise ValueError("invalid note timing")
                    notes.append(Note(start, end, midi, confidence))
                tempo_data = report.get("tempo")
                bpm = float(tempo_data["bpm"]) if isinstance(tempo_data, dict) else 120.0
                if not np.isfinite(bpm) or bpm <= 0:
                    bpm = 120.0
                grid = BeatGrid(bpm)
                duration = max(
                    float(report.get("duration_s", 0.0)),
                    max((note.end for note in notes), default=0.0),
                )
                if not np.isfinite(duration) or duration < 0:
                    raise ValueError("invalid reading duration")
                result = AnalysisResult(
                    source="live",
                    duration_s=duration,
                    notes=notes,
                    tempo=None,
                    key=None,
                    grid=grid,
                )
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
            readings.append((label, result))
        return readings

    def _refresh_archive_comparison(self) -> None:
        """Publish whether the current live take has two readable archives."""
        available = len(self._archived_live_readings()) >= 2
        self.page_results.set_archive_comparison_available(available)

    def on_open_archived_comparison(self) -> None:
        """Compare two immutable reading archives without touching the active result."""
        readings = self._archived_live_readings()
        if len(readings) < 2:
            self.page_results.set_archive_comparison_available(False)
            return
        from ..core.reading_diff import compare_readings

        labels = tuple(label for label, _result in readings)
        self.page_results.show_archived_comparison(
            labels,
            lambda before, after: compare_readings(readings[before][1], readings[after][1]),
        )

    def _export_stem(self, res: AnalysisResult) -> Path:
        """The artifact prefix for one export.

        A live take names its folder ONCE: the controller mints at the first
        export and later re-exports — the Export button — reuse it, because
        re-minting a take name per call scattered one take's artifacts over
        several timestamped folders. One sub-folder per take; `res.source`
        itself must stay "live" — the signal cache and the voice button test
        for that exact value.
        """
        src = Path(self.take_ctl.export_stem() if res.source == "live" else res.source)
        stem = src.with_suffix("") if src.suffix else src
        if res.source == "live":
            stem = live_export_dir(stem) / stem.name
        return stem

    def _stage_live_export(self, res: AnalysisResult, stem: Path, manually_corrected: bool) -> bool:
        """Hand a live export to the controller; answer the sidecar decision.

        The controller owns the take's stem and the sidecar decision (ADR
        0010): a live export queues its cross-check replay here and gets back
        the stem to archive the live/final sidecar under. The pump at the END
        of `_export_results` is the only replay starter — the WAV does not
        exist yet at this point, and a replay started now would find none.
        Returns True when the sidecar is due.
        """
        audit_block = getattr(res, "live_vs_final", None) or {}
        archive_stem = self.take_ctl.handle_export(
            {} if manually_corrected else audit_block,
            replay=not manually_corrected,
        )
        if manually_corrected:
            # the recorded audio no longer describes the edited notes: a
            # queued or running cross-check is taken back
            self.take_ctl.discard_replay(archive_stem or stem.name)
            worker = self._replay_worker
            if worker is not None and worker.isRunning():
                worker_stem = Path(worker.stem).name
                if self.take_ctl.discard_replay(worker_stem):
                    worker.stop()
        return bool(archive_stem)

    def _refresh_exported_coherence(self) -> None:
        """Refresh the session verdict from one evidence snapshot after export."""
        if self.result is None or self.result.source != "live":
            return
        try:
            snapshot = self._coherence_snapshot()
            if snapshot.sessions and snapshot.sessions[-1]["checked"] is not None:
                row = snapshot.sessions[-1]
                self.statusBar().showMessage(
                    tr("ui coherence session verdict").format(
                        suspects=row["suspects"],
                        checked=row["checked"],
                        takes=row["takes"],
                    ),
                    10_000,
                )
            self._announce_regression(snapshot)
        except Exception:
            pass

    def on_export_again(self) -> None:
        if self.result is None or self._workflow.phase is not WorkflowPhase.IDLE:
            return
        destination = self._choose_export_destination(self.result)
        if destination is None:
            return
        self._export_results(*destination)

    # ------------------------------------------------------------------
    # Live
    # ------------------------------------------------------------------
    def on_start_live(self) -> None:
        if self._workflow.phase is WorkflowPhase.REPLACEMENT_DECISION:
            self._abandon_pending_replacement()
        if self._workflow.phase not in (
            WorkflowPhase.IDLE,
            WorkflowPhase.REPLACEMENT_CAPTURE,
        ):
            return
        if self._full_reanalysis_thread is not None or (
            self.live_thread is not None and self.live_thread.isRunning()
        ):
            return
        self.error_banner.clear()
        # a fresh capture ends any review of the previous take's frozen roll
        self._apply_audition_transition(self._auditions.cancel())
        self.page_live.exit_review()
        # the take opens its OWN microphone: the calibration card's is
        # released first (two streams on one device is a failure, and the
        # card is off screen from here on — its flow is over)
        if self._calibration_thread is not None or self._calibration_timer.isActive():
            self.page_home.calibration_card.setVisible(False)
            self._on_calibrate_toggled(False)
        self.stack.setCurrentWidget(self.page_live)
        self._live_pitch_t = []
        self._live_pitch_m = []
        self._live_t = 0.0
        # the NEW take owns every live buffer: keeping the previous take's
        # signal here made `_on_live_analysis_done` keep it too (its
        # "recover from bytes only when None" guard), so the WAV export, the
        # voice playback and the waveform cache described the PREVIOUS take
        # while the analysis described this one — the two artifacts of one
        # button press told two different stories
        # the NEW take owns its whole render state: one swap, nothing left
        # to forget (the previous take's buffers are gone with the object)
        self._live = LiveRenderState()
        self.page_live.accuracy.reset()
        replacing = self._workflow.phase is WorkflowPhase.REPLACEMENT_CAPTURE
        self.page_live.set_replacement_mode(replacing)
        # the audio box is read HERE (before the capture starts): the choice
        # belongs to the take, not to the moment its export happens
        replacement = self._workflow.replacement
        self._live.record_audio = (
            replacement.record_audio
            if replacing and replacement is not None
            else self.page_live.chk_record_audio.isChecked()
        )
        # the NEW capture is the DSP priority: a replay running (its
        # real-time wait loop costs the same drain as the capture) is
        # BROKEN HERE, not joined — the worker skips the rest of its take
        # and exits on its own thread. Its stem is NOT re-queued: the
        # cross-check is a nicety, and keeping it queued would keep
        # competing with live captures
        worker = self._replay_worker
        if worker is not None and worker.isRunning():
            worker.stop()
        # A phrase replacement remains the SAME take: retain its export
        # identity and queued work. A fresh take starts a new controller state.
        if not replacing:
            self._automatic_live_result = None
            self._automatic_live_signal = None
            # a fresh take abandons the previous take's edit history;
            # its before/after diff dies with the previous take's reading
            self._edit_history.clear()
            self._sync_edit_history()
            self.page_results.clear_reading_diff()
            self.take_ctl.begin_take()
        else:
            self.take_ctl.begin_replacement()
        self._refresh_live_accuracy()
        self._refresh_expected_note(0.0)
        # reset the live state BEFORE the status label: the meter shown at
        # start must be the new take's fallback, not the previous take's
        self._live_grid = BeatGrid(120.0)  # until the take publishes its own
        self.page_live.announce(
            tr("ui live progress").format(t="00:00", notes=0, meter=self._live_grid.meter.label)
        )
        self.page_live._set_input_state("ui mic waiting")
        self.page_home.set_input_device_status(
            self._active_input_label or tr("ui input device default name"),
            self._input_device_fell_back,
        )
        # live pitch curve on the scrolling piano roll (drop the previous take's)
        self.page_live.begin_take_render(notation=True)
        if self._live_curve is not None:
            self.page_live.piano_live.removeItem(self._live_curve)
        self._live_curve = self.page_live.piano_live.plot(
            [], [], pen=None, symbol="o", symbolSize=3, symbolBrush=(120, 200, 255, 180)
        )
        self.live_thread = self._new_input_thread()
        self.page_live.set_input_device_status(
            self.live_thread.device_name, self._input_device_fell_back
        )
        self.page_live.set_driver_effects(False)  # the previous take's fact
        self.live_thread.blockStats.connect(self.page_live.on_block)
        # _on_live_pitch is the pitch's ONE route: cloud rendering AND the
        # calibration card's collector (a direct page connect feeds nothing)
        self.live_thread.pitchFrame.connect(self._on_live_pitch)
        self.live_thread.noteFrame.connect(self.page_live.on_note)
        self.live_thread.gridChanged.connect(self._on_live_grid)
        self.live_thread.liveNoteClosed.connect(self.page_live.on_live_note_closed)
        self.live_thread.captureOpened.connect(self._on_live_capture_opened)
        replacement = self._workflow.replacement
        replacement_token = replacement.token if replacing and replacement is not None else None
        self.live_thread.analysisFinished.connect(
            lambda result, signal, token=replacement_token: self._on_live_analysis_done(
                result, signal, token
            )
        )
        self.live_thread.failed.connect(
            lambda error, signal, token=replacement_token: self._live_thread_failed(
                error, signal, token
            )
        )
        self.live_thread.start()
        self.page_live.start_timer()
        # clicking a plot point seeks the live roll to that note
        try:
            self.page_live.accuracy_plot.pointClicked.disconnect()
        except TypeError:
            pass
        self.page_live.accuracy_plot.pointClicked.connect(self._on_accuracy_point_clicked)

    # The live page's behavior lives in gui/live_page.py (LivePage): the
    # 50 ms tick, the note render, the pitch cloud, the silence watch and
    # the accuracy history. The window keeps the `_on_live_*` names the
    # metronome wire-up and the tests use — thin delegations; the timer
    # itself is created and owned by the page.
    LIVE_SILENCE_WARN_S = LIVE_SILENCE_WARN_S  # re-exported for the tests
    # a calibration round counts a pitch only this fresh: the pitch route
    # reports ~every 50 ms while a note sounds, so anything older means the
    # singer stopped (the tick would otherwise grade a note that ended)
    CALIBRATION_FRESH_S = 0.25
    # ...and only after it has HELD this long. The analysis window is ~23 ms,
    # so one frame straddling the change to the next tone blends both: on the
    # real chain at 44.1 kHz, where the new tone sits at -22, it reads -14.19
    # cents on an abrupt switch, -34.84 across a 200 ms glide and -36.42
    # across a 150 ms crossfade (a so-called seam), and that frame lives one
    # hop (~6 ms).
    CALIBRATION_HOLD_S = 0.10
    # the frame-to-frame step under which a pitch counts as the SAME held
    # note: vibrato moves a few cents per frame, a note change moves ~100
    CALIBRATION_STABLE_CENTS = 25.0

    def _on_live_block(self, t: float, rms: float) -> None:
        self.page_live.on_block(t, rms)

    def _watch_live_silence(self, t: float, rms: float) -> None:
        self.page_live.watch_silence(t, rms)

    def _on_live_tick(self) -> None:
        self.page_live.on_tick()

    def _on_live_note(self, cur) -> None:
        self.page_live.on_note(cur)

    def _on_live_pitch(self, t: float, hz: float) -> None:
        self.page_live.on_pitch(t, hz)
        if self.page_home.calibration_card.isVisible():
            # a SUSTAINED pitch only: a passing glide never counts
            self._remember_calibration_pitch(hz)

    def _remember_calibration_pitch(self, hz: float) -> None:
        """One heard pitch, stamped: the tick drops it once it goes stale.

        A pitch that JUMPED from the previous frame restarts the hold clock:
        the splice frame of a note change is not the note the round is after,
        and grading it charged the round with the window's blend of the two
        tones. A steady note (vibrato included) keeps its clock running.
        """
        now = time.monotonic()
        previous = self._calibration_hz
        self._calibration_hz = hz
        self._calibration_at = now
        jumped = (
            previous is None
            or not math.isfinite(previous)
            or previous <= 0
            or not math.isfinite(hz)
            or hz <= 0
            or abs(1200.0 * math.log2(hz / previous)) > self.CALIBRATION_STABLE_CENTS
        )
        if jumped:
            self._calibration_stable_since = now

    def _calibration_round_started(self) -> None:
        """Arms one calibration round: the next held pitch feeds it."""
        self._calibration_hz = None
        self._calibration_stable_since = 0.0

    def _on_calibrate_toggled(self, on: bool) -> None:
        """The card's open/close starts/stops the round collector AND its mic."""
        if on:
            self.page_home.calibration_card.start()
            self._calibration_round_started()
            self._start_calibration_capture()
            self._calibration_timer.start()
        else:
            self._calibration_timer.stop()
            self._stop_calibration_capture()

    def _start_calibration_capture(self) -> None:
        """Open a microphone for the rounds when no take is capturing.

        The rounds read the pitch of a RUNNING capture: with none, the
        card sat on its prompt with zero input, and the user read it as a
        microphone the app cannot find (only the record action started the
        take's thread). A live take already running is REUSED — two
        streams on one device is the failure this avoids.
        """
        if self.live_thread is not None and self.live_thread.isRunning():
            return
        if self._calibration_thread is not None and self._calibration_thread.isRunning():
            return
        thread = self._new_input_thread()
        self.page_home.set_input_device_status(thread.device_name, self._input_device_fell_back)
        thread.pitchFrame.connect(self._on_calibration_pitch)
        thread.failed.connect(self._calibration_failed)
        thread.captureOpened.connect(self._on_calibration_capture_opened)
        self._calibration_thread = thread
        thread.start()

    def _stop_calibration_capture(self) -> None:
        """Drop the CARD's mic (the take's own thread is never touched)."""
        thread = self._calibration_thread
        self._calibration_thread = None
        self._calibration_hz = None
        self.page_home.calibration_card.set_driver_effects(False)
        if thread is not None and thread.isRunning():
            thread.stop()
            thread.wait(6000)

    def _on_calibration_pitch(self, t: float, hz: float) -> None:
        """The card's own capture: the collector's only pitch source."""
        self._remember_calibration_pitch(hz)

    def _calibration_failed(self, msg: str, signal: bytes) -> None:
        """A microphone that never opened is SAID on the card."""
        self._calibration_hz = None
        self._calibration_stable_since = 0.0
        card = self.page_home.calibration_card
        if card.isVisible():
            self._calibration_timer.stop()
            thread = self._calibration_thread
            card.set_error(
                capture_failure.capture_failure_message(
                    msg, thread.failure_key if thread is not None else ""
                )
            )

    def _calibration_feed_tick(self) -> None:
        """Feed the armed round a pitch only once the note has HELD.

        The tick can land on any frame, and the analysis window is wide enough
        to straddle two tones: on the real chain at 44.1 kHz, where the new
        tone sits at -22, the frame at the change reads -14.19 cents on an
        abrupt switch and -36.42 across a 150 ms crossfade, so a round that
        graded one averaged -20.4 or -24.9 over four clean rounds at -22.
        A splice frame lives one hop
        (~6 ms) and the onset frame one window; neither survives
        `CALIBRATION_HOLD_S`, while a note the singer holds does.
        """
        card = self.page_home.calibration_card
        if not card.isVisible():
            return
        hz = self._calibration_hz
        now = time.monotonic()
        if hz is not None and now - self._calibration_at > self.CALIBRATION_FRESH_S:
            hz = None  # the note ended: a stale hold is not evidence
        card.set_heard(hz)  # before the gate: the card shows what it hears
        if hz is None or now - self._calibration_stable_since < self.CALIBRATION_HOLD_S:
            return
        if card.feed(hz):
            self._calibration_round_started()

    def _refresh_live_accuracy(self) -> None:
        self.page_live.refresh_accuracy()

    def _on_live_failed(self, msg: str) -> None:
        self.page_live.on_failed(msg)

    def _on_live_grid(self, grid) -> None:
        """Consume the take's live grid (published by its owner, LiveTake)."""
        self._live_grid = grid
        self._sync_metronome_to_take(grid)

    def _on_accuracy_point_clicked(self, t: float) -> None:
        """A plot point is a way back into the take: seek the right roll.

        During capture the live roll follows the wall clock, so only the
        results page can show the past; once the take is analysed, the
        point seeks the results playhead directly.
        """
        if self._accuracy_plot_target == "results":
            self.stack.setCurrentWidget(self.page_results)
            self.page_results.set_playhead(t)
            return
        if self.live_thread is not None and self.live_thread.isRunning() and t < self._live_t:
            self.stack.setCurrentWidget(self.page_live)
            self.page_live.set_cursor(t)

    def _on_live_analysis_done(
        self, res, signal_bytes: bytes = b"", replacement_token: int | None = None
    ) -> None:
        """Show the completed live analysis and retain it until explicit export."""
        if replacement_token is not None and not self._workflow.is_current(
            replacement_token, WorkflowPhase.REPLACEMENT_CAPTURE
        ):
            return
        # the live timer is the RENDER of the CURRENT capture: the take the
        # singer has ALREADY started (its analysis arrives queued) keeps
        # recording while its roll, counter and cursor must stay live. The
        # CONTROLLER decides (it also queues this take's cross-check
        # replay): the adapter applies.
        verdict = self.take_ctl.handle_analysis(
            capture_running=(self.live_thread is not None and self.live_thread.isRunning()),
            signal_bytes=signal_bytes,  # the event's bytes AS THEY ARE: an
            # EMPTY payload CLEARS the buffer (the controller's rule) —
            # `or None` here would turn an empty event into "keep the
            # previous take's signal": the stale-WAV family
        )
        if verdict.keep_timer:
            self.page_live.start_timer()  # a newer capture owns the view: keep it
        else:
            self.page_live.stop_timer()
            self.page_live.clear_preview()
        if (
            replacement_token is not None
            and self._workflow.is_current(
                replacement_token, WorkflowPhase.REPLACEMENT_CAPTURE
            )
        ):
            self._finish_phrase_replacement(res, signal_bytes)
            return
        if replacement_token is not None:
            return
        self._edit_history.clear()
        self._sync_edit_history()
        self.result = res
        sig = self.take_ctl.signal()
        self._live_signal = np.frombuffer(sig, dtype=np.float32).copy() if sig else None
        self._automatic_live_signal = (
            np.asarray(self._live_signal, dtype=np.float32).copy()
            if self._live_signal is not None
            else None
        )
        # The result object and note list stay detached from future edits.
        automatic_result = copy.copy(res)
        automatic_result.notes = [copy.copy(note) for note in res.notes]
        self._automatic_live_result = automatic_result
        if self._automatic_live_signal is not None and hasattr(automatic_result, "sr"):
            automatic_result.duration_s = (
                len(self._automatic_live_signal) / automatic_result.sr
            )
        # LIVE vs FINAL, made visible: the notes the live segmenter had
        # confirmed are compared to the offline reading of the same take
        # through the ONE policy (`core.compare.match`) — every MOVED note
        # (pitch changed, or onset beyond the window) keeps its live bar on
        # the roll, outlined in ACCENT: the singer sees what the final
        # analysis corrected, instead of wondering why the final picture
        # looks different from what they watched. The comparison runs on the
        # live thread's segmenter (already closed: the timer is stopped,
        # nothing will mutate it mid-pass).
        try:
            self._mark_live_final_diff()
        except Exception:  # the outlines are a nicety, never a lost take
            pass
        self._signal_cache = (
            {"live": np.asarray(self._live_signal, dtype=np.float32)}
            if self._live_signal is not None
            else {}
        )
        self.source_path = "live"
        self._accuracy_plot_target = "results"
        # Export FIRST: presentation and playback setup must not prevent the
        # take's data from reaching its artifact writer.
        self.stop_playback()
        self.player = NotePlayer(
            res.notes,
            sr=audio_io.SR,
            gain=self._piano_gain(),
            on_error=self._on_playback_error,
        )
        self.playback.set_sources(piano=self.player.signal, voice=self._live_signal)
        try:
            self._show_result(res, self._live_signal)
        except Exception as e:  # a broken view must not swallow the take
            self.show_error(str(e), timeout=8_000)
        self.statusBar().showMessage(
            tr("ui status live ready").format(
                bpm=res.tempo.bpm, key=res.key.best.label, n=len(res.notes)
            ),
            12_000,
        )
        # A fresh live result waits for explicit export before its replay
        # enters the archive or the coherence journal.

    def _pump_replay_queue(self) -> None:
        """Start the next queued replay, ONLY when nothing competes for DSP.

        A replay's real-time DSP (the same `LiveCapture` -> YIN ->
        segmenter cost as a capture) competes with the LIVE capture's
        drain thread when both run. The CONTROLLER owns the queue and the
        decision (its `pump_replay` reads the adapter's facts: capture
        timer, worker thread); this adapter only applies — start the
        named worker, re-pump from `on_stop_live` and from the worker's
        own `finished` signal.
        """
        if (
            self._workflow.phase
            in (
                WorkflowPhase.SPLICE_ANALYSIS,
                WorkflowPhase.FULL_REANALYSIS,
                WorkflowPhase.REPLACEMENT_DECISION,
            )
            or self._signal_analysis_thread is not None
        ):
            return
        stem_name = self.take_ctl.pump_replay(
            capture_running=self.live_thread is not None and self.live_thread.isRunning(),
            timer_running=self.page_live.timer_active(),
            worker_running=(self._replay_worker is not None and self._replay_worker.isRunning()),
        )
        if not stem_name:
            return
        from .replay_worker import ReplayWorker

        self._replay_worker = ReplayWorker(live_export_dir(stem_name) / stem_name)
        # the worker writes the sidecar from its own thread (the archive IS
        # the product); the GUI only echoes the verdict and chains the next
        self._replay_worker.done.connect(self._on_replay_done)
        self._replay_worker.failed.connect(lambda msg: self.statusBar().showMessage(msg, 4_000))
        self._replay_worker.finished.connect(self._on_replay_worker_finished)
        self._replay_worker.start()

    def _on_replay_worker_finished(self) -> None:
        """One replay ended: tell the controller, try the next queued one."""
        self.take_ctl.replay_finished()
        self._pump_replay_queue()

    def _mark_live_final_diff(self) -> None:
        """Show the final reading BESIDE the captured one, on the live roll.

        The final notes live on `self.result` (set just above); the live
        ones on the segmenter the take just closed. A note "moved" when its
        pitch changed or its onset drifted beyond one 16th at the take's
        grid (the segmentation tolerance), judged against the NEAREST final
        note — a merge or a split shows as its moved parts, which is the
        honest picture of what the offline pass corrected. A failure here
        must never cost the take already exported (it runs after)."""
        try:
            seg = self.live_thread.segmenter if self.live_thread else None
            res = self.result
            # the closed take's notes, handed over by on_stop_live when the
            # capture thread went away before the queued analysis arrived
            live_notes = list(seg.notes) if seg is not None else self._closed_live_notes
            if not live_notes or res is None or not res.notes:
                return
            # the ONE matching policy (core.compare): same window, same
            # reasons, as the JSON's live_vs_final breakdown and the audit
            # tools — the GUI mark can never disagree with the report
            moves = compare_match(live_notes, res.notes, res.grid)
            self._last_take_live_notes = live_notes  # for the report block
            # the final reading drawn BESIDE the captured one: the notes the
            # ONE policy flagged carry the accent, so a moved note reads as
            # two bars at two heights in one glance (no outline to decode)
            moved = {i for i, note in enumerate(res.notes) if any(mv.final is note for mv in moves)}
            self.page_live.set_comparison_notes(res.notes, moved=moved)
        except Exception:  # a display nicety must never break the results
            pass

    def _finish_phrase_replacement(self, replacement_result, signal_bytes: bytes) -> None:
        """Splice the sung phrase, then rebuild evidence for the whole take."""
        context = self._workflow.replacement
        if context is None:
            return
        context = self._workflow.begin_splice_analysis(context.token)
        if context is None:
            return
        original = context.source
        original_signal = context.signal
        if original_signal is None:
            return
        first, last = context.first, context.last
        self._live_record_audio = context.record_audio
        self.page_live.set_replacement_mode(False)
        self.page_results.set_replacement_active(False)
        self.page_results.set_replacement_analyzing(True)
        self._last_take_live_notes = []
        try:
            from ..core.edits import splice_phrase

            replacement_signal = np.frombuffer(signal_bytes, dtype=np.float32).copy()
            notes, merged_signal, delta = splice_phrase(
                original.notes,
                original_signal,
                replacement_result.notes,
                replacement_signal,
                first,
                last,
                original.sr,
                replacement_sr=replacement_result.sr,
            )
            phrase_window = (
                original.notes[first].start,
                original.notes[last].end + delta,
            )
        except Exception as error:
            transition = self._workflow.fail_replacement(context.token)
            if transition is not None and transition.replacement is not None:
                self._apply_edit_state(transition.replacement.before)
                self._live_record_audio = transition.replacement.record_audio
            self.page_results.set_replacement_analyzing(False)
            self.show_error(tr("err replacement failed").format(e=error), timeout=8_000)
            self._pump_replay_queue()
            return

        if not self._workflow.set_splice_window(context.token, phrase_window):
            transition = self._workflow.fail_replacement(context.token)
            if transition is not None and transition.replacement is not None:
                self._apply_edit_state(transition.replacement.before)
                self._live_record_audio = transition.replacement.record_audio
            self.page_results.set_replacement_analyzing(False)
            self._pump_replay_queue()
            return
        self.result = original
        self._live_signal = original_signal
        self.take_ctl.set_signal(original_signal.tobytes())
        self._show_result(original, original_signal)
        self.page_results.label_play_pos.setText(tr("ui replacement analyzing"))
        self.page_results.set_replacement_analyzing(True)
        self._clear_reading_diff()
        worker = SignalAnalysisThread(merged_signal, original.sr, source="live")
        self._signal_analysis_thread = worker
        token = context.token
        worker.finishedOk.connect(
            lambda result, signal, token=token, current=worker: self._on_replacement_analysis_done(
                result, signal, token, current
            )
        )
        worker.failed.connect(
            lambda error, token=token, current=worker: self._on_replacement_analysis_failed(
                error, token, current
            )
        )
        worker.finished.connect(
            lambda current=worker: self._on_replacement_thread_finished(current)
        )
        worker.start()

    def _on_replacement_thread_finished(self, worker: SignalAnalysisThread) -> None:
        """Release a completed worker without clearing a newer worker reference."""
        if self._signal_analysis_thread is worker:
            self._signal_analysis_thread = None

    def _on_replacement_analysis_done(
        self,
        result: AnalysisResult,
        signal: np.ndarray,
        token: int | None = None,
        worker: SignalAnalysisThread | None = None,
    ) -> None:
        """Publish the spliced take and hand the decision to the singer.

        The splice is NOT applied yet: the views show the spliced take and
        the singer auditions it (the phrase clip, or the whole take with
        the play buttons), then keeps it — the splice becomes an undoable
        edit — or cancels it — the take returns to what it was.
        """
        transition = (
            self._workflow.accept_current(token, WorkflowPhase.SPLICE_ANALYSIS)
            if token is not None
            else None
        )
        if transition is None or transition.replacement is None:
            return
        if worker is not None and self._signal_analysis_thread is worker:
            self._signal_analysis_thread = None
        result.source = transition.replacement.source.source
        self.page_results.set_replacement_analyzing(False)
        invalidate_breakdown(result)
        self._last_take_live_notes = []
        self.stop_playback()
        self.result = result
        self._live_signal = np.asarray(signal, dtype=np.float32)
        self.take_ctl.set_signal(self._live_signal.tobytes())
        self._signal_cache = {result.source: self._live_signal}

        self.player = NotePlayer(
            result.notes,
            sr=result.sr,
            gain=self._piano_gain(),
            on_error=self._on_playback_error,
        )
        self.playback.set_sources(piano=self.player.signal, voice=self._live_signal)
        self.page_results.set_replacement_pending(True)
        self._show_result(result, self._live_signal)
        self.statusBar().showMessage(tr("ui replacement pending"), 8_000)
        # Nothing is exported and no stale pre-decision replay may run
        # before the singer keeps or cancels the splice.

    def _on_accept_replacement(self) -> None:
        """Keep the spliced take: the decision becomes an undoable edit."""
        transition = self._workflow.resolve_replacement(accept=True)
        if transition is None or transition.resolution is None or self.result is None:
            return
        resolution = transition.resolution
        self.page_results.set_replacement_pending(False)
        if resolution.record_history:
            self._edit_history.record(resolution.replacement.before)
            self.result.manual_pitch_corrections = True
            if self._automatic_live_result is None:
                before = resolution.replacement.before.result
                before = resolution.replacement.before.result
                automatic_result = copy.copy(before)
                automatic_result.notes = [copy.copy(note) for note in before.notes]
                automatic_result.manual_pitch_corrections = False
                self._automatic_live_result = automatic_result
            if (
                self._automatic_live_signal is None
                and resolution.replacement.before.signal is not None
            ):
                self._automatic_live_signal = np.asarray(
                    resolution.replacement.before.signal, dtype=np.float32
                ).copy()
        self._sync_edit_history()
        self.statusBar().showMessage(
            tr("ui replacement accepted").format(n=len(self.result.notes)), 8_000
        )
        self._pump_replay_queue()

    def _on_cancel_replacement(self) -> None:
        """Refuse the splice: the take returns to what it was."""
        transition = self._workflow.resolve_replacement(accept=False)
        if transition is None or transition.resolution is None:
            return
        self.page_results.set_replacement_pending(False)
        self._apply_edit_state(transition.resolution.replacement.before)
        self._sync_edit_history()
        self.statusBar().showMessage(tr("ui replacement cancelled"), 6_000)
        self._pump_replay_queue()

    def _abandon_pending_replacement(self) -> None:
        """Drop an unaccepted splice silently, restoring the take it edited.

        A pending decision belongs to its take: a NEW take (or a new file
        analysis) abandons it — the pre-replacement state is restored
        WITHOUT refilling the views (the caller is on its way elsewhere)
        and the edit history is untouched (nothing was ever applied).
        """
        transition = self._workflow.abandon_replacement()
        if transition is None or transition.resolution is None:
            return
        self.page_results.set_replacement_pending(False)
        self.page_results.set_replacement_active(False)
        self.page_results.set_replacement_analyzing(False)
        self.page_live.set_replacement_mode(False)
        self._live_record_audio = transition.resolution.replacement.record_audio
        self._apply_edit_state(transition.resolution.replacement.before, show=False)

    def on_audition_replacement(self) -> None:
        """Play the re-sung phrase, as the spliced take will sound there."""
        replacement = self._workflow.replacement
        signal = self._live_signal
        if (
            replacement is None
            or replacement.phrase_window is None
            or signal is None
            or self.result is None
        ):
            return
        phrase_start, phrase_end = replacement.phrase_window
        sr = self.result.sr
        lo = max(0, int(round(phrase_start * sr)))
        hi = min(len(signal), int(round(phrase_end * sr)))
        if hi <= lo:
            return
        self._submit_audition(
            AuditionRequest(
                kind="phrase_replacement",
                identity=("phrase_replacement", phrase_start, phrase_end),
                clips=(np.asarray(signal[lo:hi], dtype=np.float32),),
                mode="voice",
            ),
            show=lambda: None,  # a phrase audition has no readout
        )
        self._sync_play_buttons()  # the button state follows the sounding clip

    def _submit_audition(self, request: AuditionRequest, *, show) -> bool:
        """Submit-then-apply: the window resolved the clip, the controller
        decides EVERYTHING — identity, toggle, replace, silence on a failed
        start — and the returned facts name the readout to clear. The
        window never pre-checks the active identity and never pre-stops:
        the Space toggle-off IS the controller's decision on a repeated
        same-identity submit (the approved spec puts the toggle behind this
        one entry). `show` renders this kind's page readout only while the
        audition stayed active; a toggle-off or a failed start leaves no
        timer running — the observer only watches a sounding clip. The
        sounding clip is replaced by the adapter's own stage/play: no
        separate cancel is needed for a REPLACE.
        """
        transition = self._auditions.start(request)
        self._apply_audition_transition(transition)
        if transition.active is not None:
            show()
            self._play_timer.start()
        else:
            # toggle-off, failure, or an idle controller: nothing to observe
            self._play_timer.stop()
        self._sync_play_buttons()  # the button state follows the sounding clip
        return transition.active is not None

    def _apply_audition_transition(self, transition) -> None:
        """The ONE place the controller's facts reach the pages: the kind
        names the readout to clear; the other kinds have none."""
        if transition.previous_kind == "note":
            self.page_results.stop_note_audition()
        elif transition.previous_kind == "live_note":
            self.page_live.stop_note_audition()

    def on_audition_note(self, index: int) -> None:
        """Play or stop ONE note's own segment (Space on table or roll).

        The clip is the take's CURRENT audio sliced between that note's
        onset and offset — the singer hears the capture before deleting it.
        A file take reads its cached signal, a live one its take signal.
        The controller owns identity: the same note's second Space toggles
        off, a different note replaces (its readout is the orphan cleared).
        """
        result = self.result
        signal = self._voice_signal()
        if result is None or signal is None:
            return
        if not 0 <= index < len(result.notes):
            return
        note = result.notes[index]
        sr = result.sr
        lo = max(0, int(round(note.start * sr)))
        hi = min(len(signal), int(round(note.end * sr)))
        if hi <= lo:
            return
        self._submit_audition(
            AuditionRequest(
                kind="note",
                identity=("result", index),
                clips=(np.asarray(signal[lo:hi], dtype=np.float32),),
                mode="voice",
            ),
            show=lambda: self.page_results.show_note_audition(index, note.midi, len(result.notes)),
        )

    # ------------------------------------------------------------------
    # the moved note's before/after audition (the drag's companion)
    # ------------------------------------------------------------------
    def _rendered_note(self, note) -> np.ndarray:
        """One note as the piano renders it (the take's own engine)."""
        from ..core.playback import render_notes

        return render_notes([note], sr=audio_io.SR, gain=self._piano_gain())

    def _start_move_audition(self, before, after) -> None:
        """Play the old note then the moved one, one clip each.

        The window prepares the two rendered clips; the controller owns the
        sequence (the 33 ms tick's completion starts the second clip) and
        every other start or stop cancels it — one owner of what sounds.
        """
        self._submit_audition(
            AuditionRequest(
                kind="drag",
                identity=("drag", after.start, after.midi),
                clips=(self._rendered_note(before), self._rendered_note(after)),
                mode="piano",
            ),
            show=lambda: None,  # the drag comparison has no readout
        )

    def on_audition_live_note(self, index: int) -> None:
        """Play or stop ONE CAPTURED note's segment (Space on the live roll).

        The frozen roll holds the take's captured reading
        (`_last_take_live_notes`), not the final analysis, so the clip is
        cut from that note's own onset/offset on the take's signal — the
        singer hears what the capture wrote before deciding to edit it. A
        second Space on the same note stops it; the readout lives on the
        live page, the page the review is shown on.
        """
        signal = self._live_signal
        notes = self._last_take_live_notes
        if signal is None or not notes or not 0 <= index < len(notes):
            return
        note = notes[index]
        sr = audio_io.SR
        lo = max(0, int(round(note.start * sr)))
        hi = min(len(signal), int(round(note.end * sr)))
        if hi <= lo:
            return
        self._submit_audition(
            AuditionRequest(
                kind="live_note",
                identity=("captured", index),
                clips=(np.asarray(signal[lo:hi], dtype=np.float32),),
                mode="voice",
            ),
            show=lambda: self.page_live.show_note_audition(index, note.midi, len(notes)),
        )

    def on_audition_original(self) -> bool:
        """Play the selected phrase's ORIGINAL audio, before re-singing it.

        The clip is cut from the take's CURRENT signal (the live take's, or
        a file's cached audio) between the first selected note's onset and
        the last one's offset — the same span a replacement would splice,
        so the singer can compare the two auditions like for like.
        """
        signal = self._voice_signal()
        if signal is None or self.result is None:
            return False
        indices = self.page_results._selected_note_indices()
        if not indices:
            return False
        notes = self.result.notes
        if indices[-1] >= len(notes):
            return False
        sr = self.result.sr
        lo = max(0, int(round(notes[indices[0]].start * sr)))
        hi = min(len(signal), int(round(notes[indices[-1]].end * sr)))
        if hi <= lo:
            return False
        return self._submit_audition(
            AuditionRequest(
                kind="phrase_original",
                identity=("phrase_original", indices[0], indices[-1]),
                clips=(np.asarray(signal[lo:hi], dtype=np.float32),),
                mode="voice",
            ),
            show=lambda: None,  # a phrase audition has no readout
        )

    def _edit_snapshot(self, action: str = "correction") -> EditState | None:
        """The whole current take state (the edit history's before/after unit).

        `action` names the edit that is ABOUT to overwrite this state — it
        only feeds the history's timeline labels.
        """
        res = self.result
        if res is None:
            return None
        signal = self._live_signal if res.source == "live" else self._signal_cache.get(res.source)
        return snapshot_state(res, signal, self._last_take_live_notes, action=action)

    def _sync_edit_history(self) -> None:
        """Push the history's branches and whole timeline to the results page."""
        res = self.result
        current = self._edit_snapshot() if res is not None else None
        if current is not None:
            self.page_results.set_edit_states(self._edit_history.entries(current))
        self.page_results.set_edit_history_state(
            self._edit_history.can_undo, self._edit_history.can_redo
        )

    def _edit_jumps_allowed(self) -> bool:
        """The shared guard for every note edit and every history move.

        Deleting, undoing, redoing and the state dropdown all answer NO at
        the same moments: no take, a splice awaiting its verdict, another
        page, a capture re-singing the phrase, or a splice re-analyzing.
        One owner of the rule, so the dropdown can never reach a state
        undo would refuse and Del can never fire where an edit cannot.
        """
        if self.result is None or self._workflow.phase is not WorkflowPhase.IDLE:
            return False
        # the shortcut is window-wide: it only acts where the edits live
        if self.stack.currentWidget() is not self.page_results:
            return False
        if self._signal_analysis_thread is not None:
            return False  # a capture or full reanalysis is in progress
        return True

    def _move_through_history(self, steps: int) -> None:
        """Apply `steps` undo (+) or redo (-) moves as ONE view restoration.

        The core move carries the acknowledgement key as data; this adapter
        only applies the state and renders that key on the status bar.
        """
        current = self._edit_snapshot()
        if current is None:
            return
        move = self._edit_history.walk(current, steps)
        if move is None:
            return
        self._apply_history_move(move)

    def _apply_history_move(self, move: EditMove) -> None:
        """Land one core move: swap its state in, refresh the timeline
        browser, acknowledge it — the ONE route for every edit move."""
        self._apply_edit_state(move.state)
        self._sync_edit_history()
        self._ack_edit(move.acknowledgement_key)

    def _ack_edit(self, key: str, **fields) -> None:
        """Announce one note edit (undo, redo, jump, deletion, correction).

        ONE route for every acknowledgement, so the wording, the timeout and
        the status bar itself live in one place: the handlers pass the i18n
        key and the fields it formats, never a message string.
        """
        self.statusBar().showMessage(tr(key).format(**fields), 6_000)

    def _publish_note_edit(
        self,
        intent: NoteEditIntent,
        *,
        acknowledgement_key: str,
        **acknowledgement_fields,
    ) -> EditOutcome:
        """Commit one direct edit, then publish its result to every GUI reader.

        The core transaction can fail while validating the intent or rebuilding
        derived facts. Keep every presentation change after it returns so a
        rejected candidate cannot clear the reading diff or disturb playback.
        """
        result = self.result
        if result is None:
            raise RuntimeError("there is no current result to edit")

        outcome = apply_note_edit(
            result,
            self._edit_history,
            signal=self._current_analysis_signal(),
            live_notes=self._last_take_live_notes,
            intent=intent,
        )
        self._clear_reading_diff()
        self.stop_playback()
        self.result = outcome.result
        self.player = NotePlayer(
            outcome.result.notes,
            sr=outcome.result.sr,
            gain=self._piano_gain(),
            on_error=self._on_playback_error,
        )
        self.playback.set_sources(
            piano=self.player.signal,
            voice=self._voice_signal(),
        )
        self._show_result(outcome.result)
        self._sync_edit_history()
        self._ack_edit(acknowledgement_key, **acknowledgement_fields)
        return outcome

    def on_undo_edit(self) -> None:
        """Restore the take as it was before the last applied note edit."""
        if not self._edit_jumps_allowed():
            return
        self._move_through_history(1)

    def on_redo_edit(self) -> None:
        """Reapply the last undone note edit."""
        if not self._edit_jumps_allowed():
            return
        self._move_through_history(-1)

    def on_jump_edit_state(self, row_index: int) -> None:
        """Restore a state chosen in the results page's dropdown."""
        if not self._edit_jumps_allowed():
            return
        current = self._edit_snapshot()
        if current is None:
            return
        move = self._edit_history.jump(current, row_index)
        if move is None:
            return
        self._apply_history_move(move)

    def _apply_edit_state(self, state: EditState, *, show: bool = True) -> None:
        """Swap one whole-take state in (an undo, a redo, a cancelled splice)."""
        res = state.result
        # a restored state IS another reading: the diff it described is gone
        self._clear_reading_diff()
        self.stop_playback()
        self.result = res
        if res.source == "live":
            self._live_signal = (
                np.asarray(state.signal, dtype=np.float32) if state.signal is not None else None
            )
            self._signal_cache = (
                {"live": np.asarray(self._live_signal, dtype=np.float32)}
                if self._live_signal is not None
                else {}
            )
            if self._live_signal is not None:
                # the take's exportable audio follows the restored state
                self.take_ctl.set_signal(self._live_signal.tobytes())
        else:
            self._live_signal = None  # a file view must never show a live take
            self._signal_cache = (
                {res.source: np.asarray(state.signal, dtype=np.float32)}
                if state.signal is not None
                else {}
            )
        self._last_take_live_notes = list(state.live_notes)
        self.player = NotePlayer(
            res.notes,
            sr=res.sr,
            gain=self._piano_gain(),
            on_error=self._on_playback_error,
        )
        self.playback.set_sources(piano=self.player.signal, voice=self._voice_signal())
        if show:
            self._show_result(res, state.signal)

    def _on_replacement_analysis_failed(
        self,
        error: str,
        token: int | None = None,
        worker: SignalAnalysisThread | None = None,
    ) -> None:
        """Restore the original take only for the current failed analysis."""
        transition = self._workflow.fail_replacement(token) if token is not None else None
        if transition is None or transition.replacement is None:
            return
        if worker is not None and self._signal_analysis_thread is worker:
            self._signal_analysis_thread = None
        replacement = transition.replacement
        self.page_results.set_replacement_analyzing(False)
        self.page_results.set_replacement_active(False)
        self.page_live.set_replacement_mode(False)
        self._clear_reading_diff()
        self._live_record_audio = replacement.record_audio
        self._apply_edit_state(replacement.before)
        self._show_result(replacement.source, replacement.signal)
        self.take_ctl.set_signal(
            replacement.signal.tobytes() if replacement.signal is not None else b""
        )
        self.show_error(tr("err replacement failed").format(e=error), timeout=8_000)
        self._pump_replay_queue()

    def on_correct_note(self, index: int, midi: int) -> None:
        """Correct one selected note and publish its committed take state."""
        result = self.result
        if result is None:
            return
        try:
            self._publish_note_edit(
                CorrectPitchIntent(index=index, midi=midi),
                acknowledgement_key="ui status note corrected",
                note=note_display_pair(midi),
            )
        except Exception as error:
            self.show_error(tr("err note correction failed").format(e=error), timeout=8_000)

    def on_note_moved(self, index: int, start_s: float, end_s: float, midi: int) -> None:
        """One dragged bar lands as ONE recorded edit (the roll's drag).

        The roll clamps the preview to the same bounds the core enforces,
        so a release that reaches here is already legal; the guard still
        answers NO where no edit may happen (another page, a splice, a
        re-singing capture) and the bar simply snaps back.
        """
        current = self.result
        if current is None or not self._edit_jumps_allowed():
            return
        try:
            outcome = self._publish_note_edit(
                MoveNoteIntent(index=index, start_s=start_s, end_s=end_s, midi=midi),
                acknowledgement_key="ui status note moved",
                note=note_display_pair(midi),
            )
            result = outcome.result
            # the drag's companion, LAST so the player rebuild above cannot
            # kill it: the OLD note, then the moved one — read from the
            # snapshot the edit records, so an undo-then-replay answers
            # exactly what the release played
            before_notes = outcome.before.result.notes
            if index < len(before_notes):
                self._start_move_audition(before_notes[index], result.notes[index])
        except Exception as error:
            self.show_error(tr("err note move failed").format(e=error), timeout=8_000)

    def on_delete_selection(self) -> None:
        """Delete the selected notes as ONE undoable edit (the Del key).

        No selection stays silent — Del belongs to other widgets too. The
        shared edit transaction records one whole before-state, so one undo
        restores every deleted note at once; a deletion that would empty the
        take is refused by the core and announced without changing history.
        """
        result = self.result
        if result is None or not self._edit_jumps_allowed():
            return
        indices = self.page_results._selected_note_indices()
        if not indices:
            return
        try:
            self._publish_note_edit(
                DeleteNotesIntent(indices=tuple(indices)),
                acknowledgement_key="ui status notes deleted",
                n=len(indices),
            )
        except Exception as error:
            self.show_error(tr("err note deletion failed").format(e=error), timeout=8_000)

    def on_start_phrase_replacement(self) -> None:
        """Start a live capture that replaces the selected contiguous notes."""
        result = self.result
        signal = self._voice_signal()
        indices = self.page_results._selected_note_indices()
        if result is None or result.source != "live" or signal is None:
            self.show_error(tr("err replacement live only"), timeout=6_000)
            return
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            self.show_error(tr("err replacement selection"), timeout=6_000)
            return
        state = snapshot_state(result, signal, self._last_take_live_notes, action="replacement")
        replacement = self._workflow.begin_replacement(
            state, indices[0], indices[-1], self._live_record_audio
        )
        if replacement is None:
            return
        self.page_results.set_replacement_active(True)
        try:
            self.on_start_live()
        except Exception as error:
            self._cancel_phrase_replacement(str(error))
            return
        self._live_record_audio = replacement.record_audio
        self.page_live.set_replacement_mode(True)
        self.page_live.announce(tr("ui replacement prompt"))

    def on_finish_phrase_replacement(self) -> None:
        """Stop singing the replacement; the live finalizer handles its audio."""
        if self._workflow.phase is not WorkflowPhase.REPLACEMENT_CAPTURE:
            return
        self.on_stop_live()

    def on_audition_key(self, midi: int) -> None:
        """Play one short piano tone for a chosen key.

        The finder's click and the results page's pitch preview share this
        ONE tone: choosing a key must be heard before it is committed.
        Another choice or leaving stops the previous tone (the controller
        replaces it — a repeated click restarts, it never toggles off).
        """
        if midi <= 0:
            return
        from ..core.notes import Note
        from ..core.playback import render_notes

        tone = Note(0.0, 0.85, int(midi), 0.95, strength=0.8)
        self._submit_audition(
            AuditionRequest(
                kind="key",
                identity=int(midi),
                clips=(render_notes([tone], sr=audio_io.SR, gain=0.8),),
                mode="piano",
            ),
            show=lambda: None,  # a key tone has no readout
        )

    def _show_result(self, res, signal: np.ndarray | None = None) -> None:
        """Show a fresh result on the results page (file or live take)."""
        self.stack.setCurrentWidget(self.page_results)
        self.page_results.set_result(
            res,
            signal
            if signal is not None
            else self._signal_cache.get(res.source)
            if res.source != "live"
            else self._live_signal,
        )
        self._refresh_archive_comparison()

    def on_stop_live(self) -> None:
        self.page_live.exit_review()
        self.page_live.stop_timer()
        self.page_live.clear_preview()
        # the take is over: its closed notes sit frozen on the roll — the
        # singer can now pick them read-only (the next take clears it all)
        self.page_live.end_take_render()
        if self.metronome is not None:
            self.metronome.stop()
            self.page_live.show_beat(tr("ui beat count").format(beat="—"))
        if self.live_thread is not None:
            # the final analysis arrives QUEUED (AnalysisThread), i.e. AFTER
            # this handler returns and the capture reference is gone: the
            # live-vs-final diff mark must still see what the singer watched,
            # so the closed segmenter hands over its notes right here — in
            # EVERY path (a mic that ends its stream on its own reaches this
            # handler with a thread that isRunning() False, and its notes
            # still belong to the take being finalized)
            seg = getattr(self.live_thread, "segmenter", None)
            if seg is not None and seg.notes:
                self._closed_live_notes = list(seg.notes)
            if self.live_thread.isRunning():
                self.page_live.announce(tr("ui stopping"))
                self.live_thread.stop()
                # the capture thread also drains its queued blocks: give it
                # room before the reference goes away. A Ctrl+C at the
                # launching console lands HERE (the wait is where the GUI
                # thread sits) — `_install_console_quit` turns it into a
                # clean window close instead of a process abort.
                self.live_thread.wait(6000)
            dropped = (
                self.live_thread.take.dropped_blocks if (self.live_thread.take is not None) else 0
            )
            if dropped:
                # the honesty counter, read where the eye is: the live note
                # count reads hollow (blocks never reached the DSP), the WAV
                # keeps everything — the announcement makes the gap legible
                self.page_live.announce(tr("ui dropped blocks").format(n=dropped))
            self.live_thread = None
            # a take whose replay was DEFERRED (its analysis arrived while
            # the next capture was already running) starts HERE, where no
            # capture competes for the drain thread
            if not self._workflow.replacement_pending:
                try:
                    self._pump_replay_queue()
                except Exception:
                    pass
        else:
            self.stack.setCurrentWidget(self.page_home)

    def _cancel_phrase_replacement(self, error: str) -> None:
        """Restore the original take when replacement capture cannot finish."""
        context = self._workflow.replacement
        transition = self._workflow.fail_replacement(context.token) if context else None
        if transition is None or transition.replacement is None:
            return
        replacement = transition.replacement
        self.page_live.set_replacement_mode(False)
        self.page_results.set_replacement_analyzing(False)
        self.page_results.set_replacement_active(False)
        self._clear_reading_diff()
        self._live_record_audio = replacement.record_audio
        self._apply_edit_state(replacement.before, show=False)
        self.take_ctl.set_signal(
            replacement.signal.tobytes() if replacement.signal is not None else b""
        )
        self._show_result(replacement.source, replacement.signal)
        self.show_error(tr("err replacement failed").format(e=error), timeout=8_000)
        self._pump_replay_queue()

    def _live_thread_failed(
        self, error: str, signal_bytes: bytes = b"", replacement_token: int | None = None
    ) -> None:
        """The take's finalization failed: announce it, leave the trace.

        ONE place for the failure event (before, only the status label saw
        it). The trace is the diagnosable half (ADR 0031): a take that
        dies here has no analysis to export — nothing was minted, nothing
        written — and its folder would stay empty and unexplained. The
        controller owns the folder, the failure vocabulary and the write;
        error is announced on the status label (the same route the page
        owned); the trace is the diagnosable half (ADR 0031): a take that
        dies here has no analysis to export — nothing was minted, nothing
        written — and its folder would stay empty and unexplained. The
        event's signal is stored FIRST (the controller learns the take's
        audio here, exactly as the success path does): the trace reads it
        to name the folder and the duration. The controller owns the
        failure vocabulary and the write; the window only decides THAT the
        take failed. Best-effort: the failure is already announced, the
        trace must never raise a second error. NOT called on a normal
        stop: the stop path's queued analysis is still in flight — a
        stop-time trace would brand a healthy take failed before its
        export arrived.
        """
        if replacement_token is not None:
            if not self._workflow.is_current(
                replacement_token, WorkflowPhase.REPLACEMENT_CAPTURE
            ):
                return
            self._cancel_phrase_replacement(error)
            return
        if self._workflow.replacement_pending:
            self._cancel_phrase_replacement(error)
            return
        thread = self.live_thread
        self.page_live.on_failed(
            capture_failure.capture_failure_message(
                error, thread.failure_key if thread is not None else ""
            )
        )
        self.take_ctl.set_signal(signal_bytes)
        try:
            self.take_ctl.fail_take()
        except Exception:
            pass

    def _on_replay_done(self, row: dict) -> None:
        """The take's own replay finished: echo the cross-check where the eye is.

        The worker already archived the row (`<stem>.live_vs_final.json`,
        replaced over the export-time block); the singer reads the verdict —
        live vs final, the contour rate — without opening any file. A
        failure stays in the status bar (a diagnostic, not an app error).
        """
        try:
            self.statusBar().showMessage(
                tr("ui replay done").format(
                    live=row.get("live_notes", 0),
                    final=row.get("final_notes", 0),
                    rate=float(row.get("rate", 0.0)) * 100,
                ),
                8_000,
            )
        except Exception:  # a nicety on top of an archived result
            pass

    def _announce_regression(self, snapshot=None) -> None:
        """Warn when the real-time health DEGRADED against the previous week.

        The journal was just refreshed with this take's audit: `regression_report`
        compares the last two weeks' contour rates (the ONE threshold,
        `coherence.REGRESSION_DELTA`); the banner shows the measured points,
        never a judgment. Silent when healthy, when the data is missing, or
        when a lone take cannot back the comparison.
        """
        snapshot = snapshot or self._coherence_snapshot()
        reg = snapshot.regression
        if reg is None:
            return
        self.show_error(
            tr("ui regression warning").format(
                delta=reg["delta_points"],
                last=reg["last_rate"] * 100,
                prev=reg["prev_rate"] * 100,
            ),
            timeout=12_000,
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt's own name
        """Stop the background replay before the window dies.

        A replay costs the take's duration (real-time delivery): closing
        the app must break its wait loop (`stop()`), never join a running
        capture for the whole take again — and never leave a worker thread
        keeping the process alive after the window is gone.
        """
        worker = self._replay_worker
        if worker is not None:
            worker.stop()
            worker.wait(2_000)
        full_reanalysis = self._full_reanalysis_thread
        if full_reanalysis is not None and full_reanalysis.isRunning():
            # pYIN has no cancellation seam: finish before Qt destroys the
            # receiver, otherwise a still-running QThread outlives the window.
            full_reanalysis.wait()
        live = self.live_thread
        if live is not None and live.isRunning():
            live.stop()
            live.wait(2_000)  # the drain is bounded; the app is quitting anyway
        calibration = self._calibration_thread
        if calibration is not None and calibration.isRunning():
            calibration.stop()
            calibration.wait(2_000)
        self.take_ctl.close()  # the session's queue dies with the window
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Metronome (optional): a steady click to practice regularity
    # ------------------------------------------------------------------
    def on_metronome_toggled(self, on: bool) -> None:
        """Start/stop the click; the spin box follows the same state."""
        self.page_live.set_bpm_enabled(on)
        if on:
            self.page_live.clicks.start()
            # the take's live grid (published by LiveTake) seeds the tempo
            bpm = self._live_grid.bpm
            if bpm and bpm != 120.0:
                self.page_live.set_bpm_silently(bpm)
            self.metronome = Metronome(self.page_live.bpm(), scheduler=qt_scheduler())
            self.metronome.on_beat = self._on_metronome_beat
            self.metronome.start()
        else:
            if self.metronome is not None:
                self.metronome.stop()
                self.metronome = None
            self.page_live.clicks.stop()
            self.page_live.show_beat("")

    def on_metronome_bpm_changed(self, bpm: int) -> None:
        """A tempo change applies immediately, without restarting the count."""
        if self.metronome is not None:
            self.metronome.bpm = bpm

    def _follow_detected_bpm(self, bpm: float) -> None:
        """The metronome tracks the take's detected tempo as it refines.

        The spin box stays the user's override: it is updated silently, so
        the user can still drag it away and be heard.
        """
        self.page_live.set_bpm_silently(bpm)
        if self.metronome is not None:
            self.metronome.bpm = bpm

    def _sync_metronome_to_take(self, grid) -> None:
        """Tempo AND beat pattern follow the take's published live grid.

        Both facts are PUBLISHED by LiveTake's grid policy: the tempo is the
        grid's bpm, the pattern is the grid's meter in DISPLAYED beats — the
        grid's rule, shared with the staff: 4/4 -> 4, 3/4 -> 3, 6/8 -> 2
        (dotted beats). Accenting the first of two dotted beats is what a
        6/8 conductor shows, not 6 eighth clicks.
        """
        if self.metronome is None or grid is None or not grid.bpm:
            return
        self._follow_detected_bpm(grid.bpm)
        self.metronome.beats_per_measure = grid.meter.n_slots

    def _on_metronome_beat(self, beat: int, accent: bool) -> None:
        self.page_live.clicks.play(accent)
        self.page_live.show_beat(tr("ui beat count").format(beat=beat))

    # ------------------------------------------------------------------
    # Note finder: "which note is this?" on a piano keyboard, nothing else.
    # The page is a Page module (gui/finder_page.py): it owns its widgets,
    # its state (session + thread) and its behavior. The window keeps the
    # public method names the home button and the tests use — thin
    # delegations, no finder logic left here.
    # ------------------------------------------------------------------
    def _read_input_device_key(self) -> str | None:
        """Load the device preference without making startup depend on it."""
        try:
            value = json.loads(INPUT_DEVICE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, str) else None

    def _write_input_device_key(self, key: str | None) -> None:
        """Persist the user's selection best-effort."""
        try:
            if key is None:
                INPUT_DEVICE_FILE.unlink(missing_ok=True)
            else:
                INPUT_DEVICE_FILE.write_text(json.dumps(key), encoding="utf-8")
        except OSError:
            pass

    def _input_device_choices(self) -> list[tuple[str, str]]:
        """Return a stable key and localized label for every input endpoint."""
        return [(device.key, device.label) for device in audio_io.input_devices()]

    def _on_input_device_changed(self, key: str | None) -> None:
        """Keep the shared selection on the window and apply it next capture."""
        self._selected_input_key = key
        self._write_input_device_key(key)
        self._refresh_input_device_status()

    def _resolve_input_device(self) -> audio_io.ResolvedInputDevice:
        return audio_io.resolve_input_device(self._selected_input_key)

    def _refresh_input_device_status(self) -> None:
        resolved = self._resolve_input_device()
        self._active_input_label = resolved.name
        self._input_device_fell_back = resolved.fell_back
        self.page_home.set_input_device_status(resolved.name, resolved.fell_back)

    def _on_live_capture_opened(self, raw: bool) -> None:
        """Report the endpoint the capture ACTUALLY opened, and its processing.

        A raw (exclusive) open bypasses the driver's effects chain, so nothing
        is said; a shared one means the chain may have filtered the take
        before the app heard it — a fact no evidence INSIDE the take carries.
        """
        thread = self.live_thread
        name = thread.device_name if thread is not None else ""
        self.page_live.set_driver_effects(not raw and driver_fx.driver_effects_active(name))

    def _new_input_thread(self) -> LiveThread:
        """Resolve and announce the endpoint once for this capture."""
        resolved = self._resolve_input_device()
        self._active_input_label = resolved.name
        self._input_device_fell_back = resolved.fell_back
        if resolved.fell_back:
            message = tr("ui input device missing").format(device=resolved.name)
            self.statusBar().showMessage(message, 8_000)
        return LiveThread(
            device=resolved.index,
            device_name=resolved.name,
            # a driver effects chain is bypassed by the raw mode when the
            # endpoint offers one; a refusal falls back INSIDE the capture
            exclusive=driver_fx.driver_effects_active(resolved.name),
        )

    def _new_finder_input_thread(self) -> LiveThread:
        """Build the finder capture and label its actually resolved device."""
        thread = self._new_input_thread()
        self.page_finder.set_input_device_status(thread.device_name, self._input_device_fell_back)
        thread.captureOpened.connect(self._on_finder_capture_opened)
        return thread

    def _on_calibration_capture_opened(self, raw: bool) -> None:
        """The card's own capture: name its endpoint's chain only if it applied."""
        thread = self._calibration_thread
        name = thread.device_name if thread is not None else ""
        self.page_home.calibration_card.set_driver_effects(
            not raw and driver_fx.driver_effects_active(name)
        )

    def _on_finder_capture_opened(self, raw: bool) -> None:
        """The finder's capture: a raw open already bypassed the chain."""
        thread = self.page_finder.thread
        name = thread.device_name if thread is not None else ""
        self.page_finder.set_driver_effects(not raw and driver_fx.driver_effects_active(name))

    def on_open_finder(self) -> None:
        self.stack.setCurrentWidget(self.page_finder)
        self.page_finder.open()

    def on_toggle_finder(self) -> None:
        self.page_finder.toggle()

    def on_close_finder(self) -> None:
        self.page_finder.leave()

    def _set_finder_note(self, midi_float, light: bool = True) -> None:
        self.page_finder.set_note(midi_float, light)

    def _expected_midi(self, t: float) -> int | None:
        if self.result is None:
            return None
        for note in self.result.notes:
            if note.start <= t < note.end:
                return note.midi
        return None

    def _refresh_expected_note(self, t: float) -> None:
        expected = self._expected_midi(t)
        if expected is None:
            self.page_live.label_expected.setText(tr("ui expected none"))
        else:
            self.page_live.label_expected.setText(
                tr("ui expected note").format(note=note_display_pair(expected))
            )

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if urls and any(u.toLocalFile().lower().endswith(AUDIO_EXTS) for u in urls):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(AUDIO_EXTS):
                self.start_analysis(path)
                break


def run_gui() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Voix → MIDI")
    win = MainWindow()
    app.setStyleSheet(widgets.theme_stylesheet())
    win.show()
    _install_console_quit(app, win)
    sys.exit(app.exec())


def _install_console_quit(app: QApplication, win: MainWindow) -> None:
    """Ctrl+C at the launching console closes the window instead of raising
    KeyboardInterrupt.

    A KeyboardInterrupt raised inside a Qt slot is left unhandled and PyQt
    aborts the whole process; the handler below runs between bytecode
    instructions on the GUI thread, so routing it through the window's own
    `close()` (closeEvent stops the workers) quits the session cleanly.
    """

    def _quit_on_sigint(signum: int, _frame: object) -> None:
        if signum == signal.SIGINT:
            win.close()

    signal.signal(signal.SIGINT, _quit_on_sigint)
