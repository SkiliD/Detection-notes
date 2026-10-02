"""Main window of the Voix -> MIDI application."""

from __future__ import annotations

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

from ..core import audio_io
from ..core.audio_io import LiveCapture
from ..core.continuity import coherence_report
from ..core.edits import EDIT_ACTIONS, EditHistory, EditState, delete_notes, snapshot_state
from ..core.grid import BeatGrid
from ..core.i18n import get_language, note_display_pair, set_language, tr
from ..core.live_session import LiveEvent, LiveTake
from ..core.liveseg import LiveNoteSegmenter
from ..core.metronome import Metronome, qt_scheduler
from ..core.music import on_piano
from ..core.pdf_export import PdfExportError
from ..core.pipeline import AnalysisResult, AnalyzeOptions, FileAnalysis
from ..core.compare import match as compare_match
from ..core.playback import NotePlayer, PlaybackSession
from ..core.report import engrave_pdf, export_result
from ..core import takes
from ..core.takes import live_export_dir, live_take_name
from .finder_page import FinderPage
from .live_page import LIVE_SILENCE_WARN_S, LivePage
from .live_session_controller import LiveSessionController
from .home_page import HomePage
from .results_page import ResultsPage
from .widgets import (
    STYLESHEET,
    ErrorBanner,
)

AUDIO_EXTS = audio_io.AUDIO_EXTS  # single source of truth (core.audio_io)


def audio_filter() -> str:
    """File-dialog filter in the current language.

    Built on demand: `tr()` must run after the persisted language is
    restored, so this cannot be a module constant.
    """
    return tr("ui audio filter")


LiveFrame = tuple[float, float] | None


class LiveRenderState:
    """EVERYTHING the GUI renders from one live take, owned as a whole.

    The take's signal, the closed notes handed over for the diff mark, the
    audio-box choice read at take start, the pitch cloud, the published
    grid: these used to be nine loose attributes on the window, reset one
    by one in `on_start_live` — one forgotten reset was the previous take's
    value leaking into this one (measured: the exported WAV described the
    PREVIOUS take). Now the take OWNS one object and `on_start_live` swaps
    it wholesale: a stale value is structurally impossible, there is no
    buffer left to survive. The tests fake a mid-take state by building
    THIS object instead of driving the whole window; the window keeps the
    familiar `win._live_*` faces as thin properties over it.
    """

    def __init__(self) -> None:
        self.signal: np.ndarray | None = None      # the take's audio (float32)
        self.record_audio = False                  # the box, read at take START
        self.closed_notes: list = []               # handed over by on_stop_live
        self.last_take_notes: list = []            # for the report's live_vs_final
        self.pitch_t: list[float] = []             # the live pitch cloud (x)
        self.pitch_m: list[float] = []             # ...and its pitches (y)
        self.curve = None                          # the cloud's plot item
        self.t = 0.0                               # capture time of the last block
        self.grid = BeatGrid(120.0)                # until the take publishes its own
        self.accuracy_target = "live"              # where a plot click seeks
        self.silence_since: float | None = None    # silence watch (ui live silence)
        # the PRE-TAKE low-level watch (ui live low mic): the mic's first
        # clear block arms the take (heard_once) and disarms this — the
        # take is then judged by `watch_silence`. One swap per take: the
        # watch re-arms itself when a new take replaces the state.
        self.heard_once = False                    # any block above the gate yet
        self.low_since: float | None = None        # pre-take low-level watch


class AnalysisThread(QThread):
    """Run one file analysis and publish its complete outcome."""

    progress = pyqtSignal(float, str)
    finishedOk = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source: str, opts: AnalyzeOptions):
        super().__init__()
        self.source = source
        self.opts = opts

    def run(self) -> None:
        # pYIN only has to finish EVENTUALLY: below the capture's priority,
        # so finalizing the previous take cannot starve the next one's live
        # drain (see gui/priority.py)
        from .priority import lower_background_priority

        lower_background_priority()
        try:
            from ..core.pipeline import analyze_file

            outcome = analyze_file(
                self.source,
                opts=self.opts,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.finishedOk.emit(outcome)
        except Exception as e:
            self.failed.emit(str(e))


class SignalAnalysisThread(QThread):
    """Analyze an edited in-memory take without blocking the GUI thread."""

    finishedOk = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, signal: np.ndarray, sr: int, source: str):
        super().__init__()
        self.signal = np.asarray(signal, dtype=np.float32)
        self.sr = sr
        self.source = source

    def run(self) -> None:
        from .priority import lower_background_priority

        lower_background_priority()
        try:
            from ..core.pipeline import AnalyzeOptions, LIVE_CONF, analyze_signal

            result = analyze_signal(
                self.signal,
                self.sr,
                source=self.source,
                opts=AnalyzeOptions(min_conf=LIVE_CONF),
            )
            self.finishedOk.emit(result, self.signal)
        except Exception as error:
            self.failed.emit(str(error))


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
    gridChanged = pyqtSignal(object)  # BeatGrid: the take's live grid policy
    analysisFinished = pyqtSignal(object, bytes)
    failed = pyqtSignal(str, bytes)  # the error + the take's signal (the trace reads it)

    def __init__(self):
        super().__init__()
        self._stop = False
        self.take: LiveTake | None = None

    def _on_take_event(self, event: LiveEvent) -> None:
        """Forward one LiveTake event to its Qt signal (worker thread)."""
        if event.kind == "block":
            self.blockStats.emit(event.time_s, event.rms)
        elif event.kind == "pitch" and event.pitch is not None:
            self.pitchFrame.emit(*event.pitch)
        elif event.kind == "note":
            self.noteFrame.emit(event.pitch)
        elif event.kind == "note_closed":
            # the grid is published by its single owner (LiveTake); the GUI
            # consumes it, it never re-derives the approximate BPM itself
            self.gridChanged.emit(event.grid)
        elif event.kind == "failed" and event.error:
            # the take's signal rides along: the failure diagnostic (the
            # trace) needs the audio to name the folder and the duration
            sig = event.signal.tobytes() if event.signal is not None else b""
            self.failed.emit(event.error, sig)

    def run(self) -> None:
        self.take = LiveTake(on_event=self._on_take_event)  # default thresholds (pipeline)
        try:
            self.take.start()
            while not self._stop:
                self.msleep(50)
            self.take.stop()
            result, signal = self.take.finalize()
            self.analysisFinished.emit(result, signal.tobytes())
        except Exception as error:
            # a finalization failure was already published as a "failed"
            # event by LiveTake — don't report the same error twice
            if self.take is None or not self.take.error_emitted:
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
        self.resize(1280, 780)
        self.setAcceptDrops(True)

        self.result = None            # last AnalysisResult
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
        self._phrase_replacement: tuple[
            AnalysisResult, np.ndarray, int, int, bool
        ] | None = None
        self._replacement_source_path: str | None = None
        self._replacement_pending_analysis = False
        self._signal_analysis_thread: SignalAnalysisThread | None = None
        # the note edits' undo/redo: whole-take states (core.edits owns the
        # history; this window snapshots its facts and applies restorations)
        self._edit_history = EditHistory()
        # the re-sung phrase awaiting the singer's verdict: (before-state,
        # phrase start s, phrase end s in the merged audio). Export waits,
        # and the views show the spliced take — the decision is informed.
        self._replacement_decision: tuple | None = None
        # the take in progress, ONE owner: stem (minted once), signal (the
        # event's bytes), replay queue (survives a new take), sidecar and
        # timer decisions — this window is its Qt adapter (applies its
        # Verdicts)
        self.take_ctl = LiveSessionController()
        self.metronome = None       # created when the user opts in
        self.playback = PlaybackSession(audio_io.SR, on_error=self._on_playback_error)
        self.player: NotePlayer | None = None
        self._voice_player: NotePlayer | None = None
        self._finder_audition = np.zeros(0, dtype=np.float32)
        self.playbackFailed.connect(self._show_playback_error)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)  # ~30 fps playhead updates
        self._play_timer.timeout.connect(self._on_play_tick)
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
            show_status=lambda message, timeout: self.statusBar().showMessage(
                message, timeout
            ),
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
            replace_phrase=self.on_start_phrase_replacement,
            finish_replacement=self.on_finish_phrase_replacement,
            toggle_play=self.on_toggle_play,
            toggle_voice=self.on_toggle_voice,
            toggle_both=self.on_toggle_both,
            piano_volume_changed=self.on_piano_volume_changed,
            transpose_changed=self.on_transpose_changed,
            export_again=self.on_export_again,
            reinterpret=self.on_reinterpret,
            open_file=self.on_open_file,
            back_home=self.on_back_home,
            undo_edit=self.on_undo_edit,
            redo_edit=self.on_redo_edit,
            jump_edit_state=self.on_jump_edit_state,
            accept_replacement=self._on_accept_replacement,
            cancel_replacement=self._on_cancel_replacement,
            audition_replacement=self.on_audition_replacement,
            audition_original=self.on_audition_original,
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
            gate_rms=lambda: audio_io.GATE_RMS,
        )
        # The finder page is a Page module (ADR 0010's shape): it owns its
        # widgets AND its behavior — window dependencies injected (where
        # home is, where errors go), thread injected as a factory. The
        # window composes; it keeps no finder mirror.
        self.page_finder = FinderPage(
            thread_factory=LiveThread,
            go_home=lambda: self.stack.setCurrentWidget(self.page_home),
            show_error=self._on_error,
            clear_error=self.error_banner.clear,
            audition_note=self.on_audition_finder_note,
        )
        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_results)
        self.stack.addWidget(self.page_live)
        self.stack.addWidget(self.page_finder)
        self.retranslate()

    # Pages are Page modules (gui/*_page.py); MainWindow owns the window
    # state (result, cache, playback, captures) and the user callbacks.

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

    def retranslate(self) -> None:
        """Refresh every window string in the current language."""
        self.setWindowTitle(tr("ui window title"))
        self.page_home.retranslate()
        self.page_finder.retranslate()
        self.page_results.retranslate()
        self.page_live.retranslate()
        self._refresh_live_accuracy()
        self._refresh_expected_note(self._live_t)
        for view in (self.page_results.staff, self.page_live.staff_live):
            view.setToolTip(tr("ui staff tooltip"))
        for roll in (self.page_results.piano, self.page_live.piano_live):
            roll.set_axis_labels(tr("ui axis note"), tr("ui axis time"))
        if self.live_thread is None or not self.live_thread.isRunning():
            self.page_live.label_status.setText(tr("ui mic preparing"))

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
        """
        was_playing = self.playback.playing
        pos = self.playback.position() if was_playing else 0.0
        mode = self.playback.mode
        if self.result is None:
            return
        self.player = NotePlayer(
            self.result.notes, sr=audio_io.SR, gain=self._piano_gain(),
            on_error=self._on_playback_error,
        )
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
        self.page_results.sync_play_buttons(
            self.playback.mode if self.playback.playing else None
        )

    def _make_voice_player(self) -> NotePlayer | None:
        """A rendered voice source at the live transposition, or None without a take."""
        signal = self._voice_signal()
        if signal is None:
            return None
        return NotePlayer(
            signal=signal, sr=audio_io.SR,
            transpose=self.page_results.spin_transpose.value(),
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
        if self.playback.playing and self.playback.mode == "piano":
            self.stop_playback()
            return
        self._start("piano")

    def on_toggle_voice(self) -> None:
        """Play the original recording through the shared timeline."""
        if self.playback.playing and self.playback.mode == "voice":
            self.stop_playback()
            return
        self._start("voice")

    def on_toggle_both(self) -> None:
        """Piano over the voice (comparative listening), or stop both."""
        if self.playback.playing:
            self.stop_playback()
            return
        self._start("both")

    def on_seek(self, t: float) -> None:
        """Double-click on the piano roll: play from there (or just move)."""
        if self.result is None or self.player is None:
            return
        self._play_timer.stop()
        self.page_results.set_playhead(t, follow=False)
        self.page_results.set_wave_position(t)
        duration = self.playback.duration or self.player.duration
        if t >= duration - 0.05:
            return
        # Preserve the source selected by the user, including piano + voice.
        self._start(self.playback.mode or "piano", t)

    def _on_play_tick(self) -> None:
        if not self.playback.playing:
            self._play_timer.stop()
            self._sync_play_buttons()
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
        self.playback.stop()
        self.page_results.reset_play_buttons()

    def on_back_home(self) -> None:
        """Leave the analyze page: the playback must NOT keep sounding.

        One owner for the navigation back, and it stops the audio first —
        a page switch alone would leave the PlaybackSession running.
        """
        self.stop_playback()
        self.stack.setCurrentWidget(self.page_home)

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
        if res is None or getattr(res, "observation", None) is None:
            self.show_error(tr("err reinterpret none"), timeout=6_000)
            return
        # a re-reading replaces the whole interpretation: the edit history's
        # snapshots describe the previous reading, so they start over
        self._edit_history.clear()
        self._sync_edit_history()
        self.stop_playback()
        opts = AnalyzeOptions(
            min_note_s=self.page_results.spin_min_note.value() / 1000.0
        )
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
        self.result = reinterpreted
        self._show_result(reinterpreted)

        self.statusBar().showMessage(
            tr("ui status reinterpreted").format(n=len(reinterpreted.notes)), 8_000
        )

    # ------------------------------------------------------------------
    # File analysis
    # ------------------------------------------------------------------
    def on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("ui choose file"), "", audio_filter()
        )
        if not path:
            return
        self.start_analysis(path)

    def start_analysis(self, path: str) -> None:
        """Validate the input early, then run the analysis in a thread."""
        self.error_banner.clear()
        self.page_results.begin_analysis()
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
        # a new file is a new take: an unaccepted splice dies with the take
        # it edited, and the edit history never crosses takes
        self._abandon_pending_replacement()
        self._edit_history.clear()
        self._sync_edit_history()
        self.result = res
        # retain the exact decoded samples used for this analysis: waveform
        # rendering must not reopen the source file.
        self._signal_cache = {res.source: outcome.audio}
        self._live_signal = None  # a file view must never show a live take
        self.stop_playback()
        self.player = NotePlayer(
            res.notes, sr=audio_io.SR, gain=self._piano_gain(),
        )
        self._show_result(res, outcome.audio)
        # The analysis remains editable; the export button writes artifacts.
        summary = self._coherence_summary(res)
        if summary:
            self.statusBar().showMessage(
                tr("ui status coherence").format(summary=summary), 10_000
            )
        # Keep the cache explanation last: before the file outcome carried
        # this fact, its queued signal ran after this handler and the
        # coherence summary. A synchronous write here would be overwritten.
        if outcome.cache_used:
            elapsed_ms = 1000.0 * (time.perf_counter() - self._analysis_started_at)
            self.statusBar().showMessage(
                tr("ui status cache").format(ms=elapsed_ms), 8_000
            )

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def _export_results(self) -> None:
        if self.result is None:
            return
        if self._replacement_decision is not None:
            return  # the spliced take awaits its keep/cancel decision
        res = self.result
        # a live take names its folder ONCE (the controller mints at the
        # first export, later re-exports — the Export button — reuse it):
        # re-minting `live_take_name()` per call scattered one take's
        # artifacts over several timestamped folders
        src = Path(
            self.take_ctl.export_stem() if res.source == "live" else res.source
        )
        stem = src.with_suffix("") if src.suffix else src
        if res.source == "live":
            # one sub-folder per take; `res.source` itself must stay "live":
            # the signal cache and the voice button test for that exact value
            stem = live_export_dir(stem) / stem.name
        quantize_s = (
            res.quantize_step(16)
            if self.page_results.chk_quantize.isChecked()
            else None
        )
        pdf_written = False
        audio_written = False
        # the take's OWN audio first, when it was asked for: it is the
        # rawest form of the user's data, and a later failure must not cost
        # it (live only — a file analysis already reads its audio from the
        # input path). Its failure is a WARNING, never a lost take: the
        # analysis artifacts below are written regardless — the singer keeps
        # their transcription even when the disk refuses the WAV.
        if (
            res.source == "live"
            and self._live_record_audio
            and self._live_signal is not None
            and len(self._live_signal)
        ):
            try:
                audio_io.save_file(self._live_signal, str(stem) + ".wav", audio_io.SR)
                audio_written = True
            except Exception as e:
                self.show_error(tr("err ui wav failed").format(e=e), timeout=8_000)
        try:
            export_result(
                res, stem, quantize_s=quantize_s,
                transpose=self.page_results.spin_transpose.value(),
                # the take's live reading, for the report's live_vs_final
                # breakdown (the gap decomposed: ±1 roundings, moves,
                # absorbed glides) — a file analysis hands over none
                live_notes=(
                    self._last_take_live_notes
                    if res.source == "live"
                    and not getattr(res, "manual_pitch_corrections", False)
                    else None
                ),
            )
            if self.page_results.chk_pdf.isChecked():
                engrave_pdf(stem)  # optional, external tool — after the artifacts
                pdf_written = True
        except PdfExportError as e:
            # the other artifacts were written; only the PDF is missing
            self.show_error(tr("err ui pdf").format(e=e), timeout=6_000)
        except Exception as e:  # never crash the UI on a failed export
            self.show_error(tr("err ui export failed").format(e=e), timeout=8_000)
            return
        # the take's artifact set has ONE owner (takes.has_artifacts, the
        # artifact contract): the status line lists what the export
        # ACTUALLY wrote — each extension is confirmed on disk first, so a
        # half-written export (disk full mid-file) can never be announced
        # as complete; the conditional artifacts follow the same rule
        extras = []
        if audio_written:
            extras.append(".wav")
        if pdf_written:
            extras.append(".pdf")
        exts = takes.has_artifacts(stem, tuple(extras))
        # the take's live/final decomposition, archived BESIDE the artifacts:
        # the replay tool and the multi-week trend read the gap back per
        # take without re-running anything (live only — a file analysis has
        # no live reading). Written AFTER the four data artifacts: a failed
        # archive must never cost them.
        if res.source == "live":
            # the controller records the stem this take exported under (the
            # sidecar, the replay and a later re-export all name it) and
            # decides whether a sidecar is due (ONE writer, ONE stem)
            manually_corrected = getattr(res, "manual_pitch_corrections", False)
            audit_block = getattr(res, "live_vs_final", None) or {}
            archive_stem = self.take_ctl.handle_export(
                audit_block if not manually_corrected else {},
                replay=not manually_corrected,
            )
            if manually_corrected:
                self.take_ctl.discard_replay(archive_stem or stem.name)
                worker = self._replay_worker
                if worker is not None and worker.isRunning():
                    worker_stem = Path(worker.stem).name
                    if self.take_ctl.discard_replay(worker_stem):
                        worker.stop()
            archive = (
                self._archive_sidecar(archive_stem, audit_block)
                if archive_stem and not manually_corrected
                else None
            )
            if archive:
                exts.append(takes.SIDECAR_SUFFIX)
            elif manually_corrected:
                from ..core.takes import sidecar_path

                try:
                    sidecar_path(stem.parent, stem.name).unlink(missing_ok=True)
                except OSError:
                    pass  # stale audit removal is best-effort after the export
            # the retention lives at export time: the take folder was just
            # written, so the disk's size is current. The oldest take
            # folders are trimmed WHOLE under the documented budget — the
            # journal keeps their audits, the trend survives (its rule).
            try:
                from ..core import coherence as coh

                pruned = coh.prune_takes(coh.LIVE_TAKES_DIR)
                if pruned:
                    self.statusBar().showMessage(
                        tr("ui retention pruned").format(n=len(pruned), stems=", ".join(pruned)),
                        8_000,
                    )
            except Exception:  # a trim failure must never cost the take
                pass
        files = ", ".join(f"{stem.name}{ext}" for ext in exts)
        # the action acknowledges itself where the eye already is
        self.page_results.ack_export()
        self.statusBar().showMessage(tr("ui status exported").format(files=files), 10_000)
        self._refresh_exported_coherence()
        # the take's WAV now exists on disk: its queued cross-check may
        # start — the controller's guards still hold (a running capture or
        # replay keeps it waiting); a file analysis's export pumps an empty
        # queue, the call is free
        self._pump_replay_queue()

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
        self._export_results()

    # ------------------------------------------------------------------
    # Live
    # ------------------------------------------------------------------
    def on_start_live(self) -> None:
        self.error_banner.clear()
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
        self.page_live.set_replacement_mode(self._phrase_replacement is not None)
        # the audio box is read HERE (before the capture starts): the choice
        # belongs to the take, not to the moment its export happens
        self._live.record_audio = self.page_live.chk_record_audio.isChecked()
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
        if self._phrase_replacement is None:
            # a fresh take abandons an unaccepted splice and forgets the
            # previous take's edit history (its states belong to that take)
            self._abandon_pending_replacement()
            self._edit_history.clear()
            self._sync_edit_history()
            self.take_ctl.begin_take()
        else:
            self.take_ctl.begin_replacement()
        self._refresh_live_accuracy()
        self._refresh_expected_note(0.0)
        # reset the live state BEFORE the status label: the meter shown at
        # start must be the new take's fallback, not the previous take's
        self._live_grid = BeatGrid(120.0)  # until the take publishes its own
        self.page_live.label_status.setText(
            tr("ui live progress").format(
                t="00:00", notes=0, meter=self._live_grid.meter.label
            )
        )
        # live pitch curve on the scrolling piano roll (drop the previous take's)
        self.page_live.piano_live.set_selectable(False)  # a running take is never picked
        self.page_live.piano_live.begin_live()
        self.page_live.piano_live.set_notation_mode(True)
        self.page_live.staff_live.clear()
        if self._live_curve is not None:
            self.page_live.piano_live.removeItem(self._live_curve)
        self._live_curve = self.page_live.piano_live.plot(
            [], [], pen=None, symbol="o", symbolSize=3, symbolBrush=(120, 200, 255, 180)
        )
        self.live_thread = LiveThread()
        self.live_thread.blockStats.connect(self.page_live.on_block)
        self.live_thread.pitchFrame.connect(self.page_live.on_pitch)
        self.live_thread.noteFrame.connect(self.page_live.on_note)
        self.live_thread.gridChanged.connect(self._on_live_grid)
        self.live_thread.analysisFinished.connect(self._on_live_analysis_done)
        self.live_thread.failed.connect(self._live_thread_failed)
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
            self.page_live.piano_live.set_live_cursor(max(0.0, t))

    def _on_live_analysis_done(self, res, signal_bytes: bytes = b"") -> None:
        """Show the completed live analysis and retain it until explicit export."""
        # a NEW take's analysis never inherits an unaccepted splice (a
        # pending decision belongs to the take it edited)
        self._abandon_pending_replacement()
        # the live timer is the RENDER of the CURRENT capture: the take the
        # singer has ALREADY started (its analysis arrives queued) keeps
        # recording while its roll, counter and cursor must stay live. The
        # CONTROLLER decides (it also queues this take's cross-check
        # replay): the adapter applies.
        verdict = self.take_ctl.handle_analysis(
            capture_running=(
                self.live_thread is not None and self.live_thread.isRunning()
            ),
            signal_bytes=signal_bytes,  # the event's bytes AS THEY ARE: an
            # EMPTY payload CLEARS the buffer (the controller's rule) —
            # `or None` here would turn an empty event into "keep the
            # previous take's signal": the stale-WAV family
        )
        if verdict.keep_timer:
            self.page_live.start_timer()  # a newer capture owns the view: keep it
        else:
            self.page_live.stop_timer()
        self.page_live.piano_live.set_live_preview(None)
        if self._phrase_replacement is not None:
            self._finish_phrase_replacement(res, signal_bytes)
            return
        self._edit_history.clear()
        self._sync_edit_history()
        self.result = res
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
        sig = self.take_ctl.signal()
        self._live_signal = (
            np.frombuffer(sig, dtype=np.float32).copy() if sig else None
        )
        self._signal_cache = (
            {"live": np.asarray(self._live_signal, dtype=np.float32)}
            if self._live_signal is not None
            else {}
        )
        self.source_path = "live"
        self._accuracy_plot_target = "results"
        # Export FIRST: the take is the user's data, a cosmetic hiccup below
        # must never cost it (it used to silently skip the export entirely).
        self.stop_playback()
        self.player = NotePlayer(
            res.notes, sr=audio_io.SR, gain=self._piano_gain(),
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
        if self._replacement_pending_analysis or self._signal_analysis_thread is not None:
            return
        stem_name = self.take_ctl.pump_replay(
            capture_running=self.live_thread is not None
            and self.live_thread.isRunning(),
            timer_running=self.page_live.timer_active(),
            worker_running=(
                self._replay_worker is not None and self._replay_worker.isRunning()
            ),
        )
        if not stem_name:
            return
        from .replay_worker import ReplayWorker

        self._replay_worker = ReplayWorker(live_export_dir(stem_name) / stem_name)
        # the worker writes the sidecar from its own thread (the archive IS
        # the product); the GUI only echoes the verdict and chains the next
        self._replay_worker.done.connect(self._on_replay_done)
        self._replay_worker.failed.connect(
            lambda msg: self.statusBar().showMessage(msg, 4_000)
        )
        self._replay_worker.finished.connect(self._on_replay_worker_finished)
        self._replay_worker.start()

    def _on_replay_worker_finished(self) -> None:
        """One replay ended: tell the controller, try the next queued one."""
        self.take_ctl.replay_finished()
        self._pump_replay_queue()

    def _mark_live_final_diff(self) -> None:
        """Outline the notes the final analysis moved, on the live roll.

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
            from .widgets import ACCENT  # the one accent, already the palette's

            # the ONE matching policy (core.compare): same window, same
            # reasons, as the JSON's live_vs_final breakdown and the audit
            # tools — the GUI mark can never disagree with the report
            moves = compare_match(live_notes, res.notes, res.grid)
            self._last_take_live_notes = live_notes  # for the report block
            for mv in moves:
                self.page_live.piano_live.outline_note(
                    mv.live.start, mv.live.end, mv.live.midi, ACCENT
                )
        except Exception:  # a display nicety must never break the results
            pass

    def _finish_phrase_replacement(self, replacement_result, signal_bytes: bytes) -> None:
        """Splice the sung phrase, then rebuild evidence for the whole take."""
        context = self._phrase_replacement
        self._phrase_replacement = None
        self.page_live.set_replacement_mode(False)
        self.page_results.set_replacement_analyzing(True)
        if context is None:
            self._replacement_pending_analysis = False
            return
        original, original_signal, first, last, record_audio = context
        self._live_record_audio = record_audio
        self._replacement_source_path = original.source
        # the state the decision will offer to restore: the take as it was
        # when the singer chose to re-sing (captured BEFORE the live-notes
        # hand-over is cleared below). The action travels with the state:
        # the history's timeline labels this row "replacement".
        before = snapshot_state(
            original, original_signal, self._last_take_live_notes, action="replacement"
        )
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
            # the audition window: where the replaced span sits in the MERGED
            # audio (the selection's start, its span stretched by delta)
            phrase_window = (
                original.notes[first].start,
                original.notes[last].end + delta,
            )
        except Exception as error:
            self.result = original
            self._replacement_pending_analysis = False
            self._live_signal = original_signal
            self.take_ctl.set_signal(original_signal.tobytes())
            self._show_result(original, original_signal)
            self.show_error(tr("err replacement failed").format(e=error), timeout=8_000)
            self._pump_replay_queue()
            return
        self.result = original
        self._live_signal = original_signal
        self.take_ctl.set_signal(original_signal.tobytes())
        self._show_result(original, original_signal)
        self.page_results.label_play_pos.setText(tr("ui replacement analyzing"))
        self.page_results.set_replacement_analyzing(True)
        self._replacement_pending_analysis = True
        self._replacement_decision = (before, phrase_window[0], phrase_window[1])
        self._signal_analysis_thread = SignalAnalysisThread(
            merged_signal, original.sr, source="live"
        )
        self._signal_analysis_thread.finishedOk.connect(self._on_replacement_analysis_done)
        self._signal_analysis_thread.failed.connect(self._on_replacement_analysis_failed)
        self._signal_analysis_thread.start()

    def _on_replacement_analysis_done(self, result, signal: np.ndarray) -> None:
        """Publish the spliced take and hand the decision to the singer.

        The splice is NOT applied yet: the views show the spliced take and
        the singer auditions it (the phrase clip, or the whole take with
        the play buttons), then keeps it — the splice becomes an undoable
        edit — or cancels it — the take returns to what it was.
        """
        self._signal_analysis_thread = None
        self._replacement_pending_analysis = False
        self.page_results.set_replacement_analyzing(False)
        if self._replacement_source_path is not None:
            result.source = self._replacement_source_path
        self._replacement_source_path = None
        result.live_vs_final = None
        self._last_take_live_notes = []
        self.stop_playback()
        self.result = result
        self._live_signal = np.asarray(signal, dtype=np.float32)
        self.take_ctl.set_signal(self._live_signal.tobytes())
        self._signal_cache = {result.source: self._live_signal}

        self.player = NotePlayer(
            result.notes, sr=result.sr, gain=self._piano_gain(),
            on_error=self._on_playback_error,
        )
        self.playback.set_sources(piano=self.player.signal, voice=self._live_signal)
        if self._replacement_decision is None:
            # no decision context (the splice came from another path): the
            # spliced take is shown, the singer keeps or re-exports as today
            self._show_result(result, self._live_signal)
            return
        self.page_results.set_replacement_pending(True)
        self._show_result(result, self._live_signal)
        self.statusBar().showMessage(tr("ui replacement pending"), 8_000)
        # Nothing is exported and no stale pre-decision replay may run
        # before the singer keeps or cancels the splice.

    def _on_accept_replacement(self) -> None:
        """Keep the spliced take: the decision becomes an undoable edit."""
        decision = self._replacement_decision
        if decision is None or self.result is None:
            return
        self._replacement_decision = None
        self.page_results.set_replacement_pending(False)
        self._edit_history.record(decision[0])
        self._sync_edit_history()
        self.statusBar().showMessage(
            tr("ui replacement accepted").format(n=len(self.result.notes)), 8_000
        )
        self._pump_replay_queue()

    def _on_cancel_replacement(self) -> None:
        """Refuse the splice: the take returns to what it was."""
        decision = self._replacement_decision
        if decision is None:
            return
        self._replacement_decision = None
        self.page_results.set_replacement_pending(False)
        self._apply_edit_state(decision[0])
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
        decision = self._replacement_decision
        if decision is None:
            return
        self._replacement_decision = None
        self.page_results.set_replacement_pending(False)
        self._apply_edit_state(decision[0], show=False)

    def on_audition_replacement(self) -> None:
        """Play the re-sung phrase, as the spliced take will sound there."""
        decision = self._replacement_decision
        signal = self._live_signal
        if decision is None or signal is None or self.result is None:
            return
        _before, phrase_start, phrase_end = decision
        sr = self.result.sr
        lo = max(0, int(round(phrase_start * sr)))
        hi = min(len(signal), int(round(phrase_end * sr)))
        if hi <= lo:
            return
        self.stop_playback()
        self.playback.set_sources(
            voice=np.asarray(signal[lo:hi], dtype=np.float32)
        )
        self.playback.play("voice")
        self._sync_play_buttons()  # the button state follows the sounding clip

    def on_audition_original(self) -> None:
        """Play the selected phrase's ORIGINAL audio, before re-singing it.

        The clip is cut from the take's CURRENT signal (the live take's, or
        a file's cached audio) between the first selected note's onset and
        the last one's offset — the same span a replacement would splice,
        so the singer can compare the two auditions like for like.
        """
        signal = self._voice_signal()
        if signal is None or self.result is None:
            return
        indices = self.page_results._selected_note_indices()
        if not indices:
            return
        notes = self.result.notes
        if indices[-1] >= len(notes):
            return
        sr = self.result.sr
        lo = max(0, int(round(notes[indices[0]].start * sr)))
        hi = min(len(signal), int(round(notes[indices[-1]].end * sr)))
        if hi <= lo:
            return
        self.stop_playback()
        self.playback.set_sources(
            voice=np.asarray(signal[lo:hi], dtype=np.float32)
        )
        self.playback.play("voice")
        self._sync_play_buttons()  # the button state follows the sounding clip

    def _edit_snapshot(self, action: str = "correction") -> EditState | None:
        """The whole current take state (the edit history's before/after unit).

        `action` names the edit that is ABOUT to overwrite this state — it
        only feeds the history's timeline labels.
        """
        res = self.result
        if res is None:
            return None
        signal = (
            self._live_signal
            if res.source == "live"
            else self._signal_cache.get(res.source)
        )
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
        if self.result is None or self._replacement_decision is not None:
            return False
        # the shortcut is window-wide: it only acts where the edits live
        if self.stack.currentWidget() is not self.page_results:
            return False
        if self._phrase_replacement is not None or self._signal_analysis_thread is not None:
            return False  # a capture is re-singing, or the splice is re-analyzing
        return True

    def _steps_to_history_index(self, row_index: int) -> int:
        """Signed undo(+)/redo(-) step count that lands on the timeline row.

        `entries` is [undo stack ..., CURRENT, redo stack ...]: an undo row
        sits that many steps back from the current state; a redo row sits
        that many redo steps FORWARD (row `undo_depth` IS the current row).
        Nothing is lost either way — each stepped move re-archives the
        state it leaves, so the timeline keeps every state the jump passed
        over.
        """
        undo_depth = self._edit_history.undo_depth
        return undo_depth - row_index  # undo rows positive, redo rows negative

    def _move_through_history(self, steps: int) -> bool:
        """Apply `steps` undo (+) or redo (-) moves as ONE view restoration."""
        if steps == 0:
            return False
        current = self._edit_snapshot()
        if current is None:
            return False
        state = current
        undone = redone = 0
        for _ in range(abs(steps)):
            if steps > 0:
                nxt = self._edit_history.undo(state)
                if nxt is None:
                    break
                state = nxt
                undone += 1
            else:
                nxt = self._edit_history.redo(state)
                if nxt is None:
                    break
                state = nxt
                redone += 1
        if undone + redone == 0:
            return False
        self._apply_edit_state(state)
        self._sync_edit_history()
        message = "ui edit undone" if undone else "ui edit redone"
        self.statusBar().showMessage(tr(message), 6_000)
        return True

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
        """Restore a past state chosen in the results page's dropdown.

        The jump is a stepped undo (rows above the current are undo moves;
        below is impossible — the current state is the last row), replayed
        through the SAME undo machinery so one owner keeps the branch
        bookkeeping and every step lands on a state the history archived.
        """
        if not self._edit_jumps_allowed():
            return
        states = self.page_results._edit_states
        if not 0 <= row_index < len(states) or states[row_index][0] is None:
            return
        self._move_through_history(self._steps_to_history_index(row_index))

    def _apply_edit_state(self, state: EditState, *, show: bool = True) -> None:
        """Swap one whole-take state in (an undo, a redo, a cancelled splice)."""
        res = state.result
        self.stop_playback()
        self.result = res
        if res.source == "live":
            self._live_signal = (
                np.asarray(state.signal, dtype=np.float32)
                if state.signal is not None
                else None
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
            res.notes, sr=res.sr, gain=self._piano_gain(),
            on_error=self._on_playback_error,
        )
        self.playback.set_sources(piano=self.player.signal, voice=self._voice_signal())
        if show:
            self._show_result(res, state.signal)


    def _on_replacement_analysis_failed(self, error: str) -> None:
        """Keep the original result visible if rebuilt evidence cannot be read."""
        self._signal_analysis_thread = None
        self._replacement_pending_analysis = False
        self._replacement_decision = None
        self.page_results.set_replacement_analyzing(False)
        if self.result is not None:
            if self._replacement_source_path is not None:
                self.result.source = self._replacement_source_path
                self._replacement_source_path = None
            self.take_ctl.set_signal(
                self._live_signal.tobytes() if self._live_signal is not None else b""
            )
            self._show_result(self.result, self._live_signal)
        else:
            self._replacement_source_path = None
        self.show_error(tr("err replacement failed").format(e=error), timeout=8_000)
        self._pump_replay_queue()

    def on_correct_note(self, index: int, midi: int) -> None:
        """Correct one selected note and refresh the score's dependent facts."""
        if self.result is None:
            return
        before = self._edit_snapshot()
        try:
            from ..core.edits import correct_note_pitch

            correct_note_pitch(self.result, index, midi)
            self.stop_playback()
            self.player = NotePlayer(
                self.result.notes, sr=self.result.sr, gain=self._piano_gain(),
                on_error=self._on_playback_error,
            )
            self.playback.set_sources(piano=self.player.signal, voice=self._voice_signal())
            self._show_result(self.result)
            if before is not None:
                self._edit_history.record(before)
            self._sync_edit_history()
        except Exception as error:
            self.show_error(tr("err note correction failed").format(e=error), timeout=8_000)

    def on_delete_selection(self) -> None:
        """Delete the selected notes as ONE undoable edit (the Del key).

        No selection stays silent — Del belongs to other widgets too. The
        edit follows the correction path (snapshot, mutate, re-show,
        record), so one undo restores every deleted note at once; a
        deletion that would empty the take is refused by the core and
        announced, history untouched.
        """
        result = self.result
        if result is None or not self._edit_jumps_allowed():
            return
        indices = self.page_results._selected_note_indices()
        if not indices:
            return
        before = self._edit_snapshot(action="deletion")
        try:
            delete_notes(result, indices)
            self.stop_playback()
            self.player = NotePlayer(
                result.notes, sr=result.sr, gain=self._piano_gain(),
                on_error=self._on_playback_error,
            )
            self.playback.set_sources(piano=self.player.signal, voice=self._voice_signal())
            self._show_result(result)
            if before is not None:
                self._edit_history.record(before)
            self._sync_edit_history()
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
        self._phrase_replacement = (
            result,
            signal.copy(),
            indices[0],
            indices[-1],
            self._live_record_audio,
        )
        self._replacement_pending_analysis = True
        self.page_results.set_replacement_active(True)
        try:
            self.on_start_live()
        except Exception as error:
            self._cancel_phrase_replacement(str(error))
            return
        self._live_record_audio = self._phrase_replacement[4]
        self.page_live.set_replacement_mode(True)
        self.page_live.label_status.setText(tr("ui replacement prompt"))

    def on_finish_phrase_replacement(self) -> None:
        """Stop singing the replacement; the live finalizer handles its audio."""
        if self._phrase_replacement is None:
            return
        self.on_stop_live()

    def on_audition_finder_note(self, midi: int) -> None:
        """Play one short piano tone; another click or leaving stops it."""
        self._play_timer.stop()
        self.playback.stop()
        if midi <= 0:
            return
        from ..core.notes import Note
        from ..core.playback import render_notes

        self._finder_audition = np.zeros(0, dtype=np.float32)
        tone = Note(0.0, 0.85, int(midi), 0.95, strength=0.8)
        self._finder_audition = render_notes([tone], sr=audio_io.SR, gain=0.8)
        self.playback.set_sources(piano=self._finder_audition)
        self.playback.play("piano")

    def _show_result(self, res, signal: np.ndarray | None = None) -> None:
        """Show a fresh result on the results page (file or live take)."""
        self.stack.setCurrentWidget(self.page_results)
        name = Path(res.source).name if res.source != "live" else tr("ui live file")
        self.page_results.set_source_name(name)
        self.page_results.set_progress(100)
        self.page_results.set_result(
            res,
            signal if signal is not None
            else self._signal_cache.get(res.source)
            if res.source != "live"
            else self._live_signal,
        )

    def on_stop_live(self) -> None:
        self.page_live.stop_timer()
        self.page_live.piano_live.set_live_preview(None)
        # the take is over: its closed notes sit frozen on the roll — the
        # singer can now pick them read-only (the next take clears it all)
        self.page_live.piano_live.set_selectable(True)
        self.page_live.piano_live_keys.set_pressed()
        self.page_live.staff_live.clear()
        if self.metronome is not None:
            self.metronome.stop()
            self.page_live.label_beat.setText(tr("ui beat count").format(beat="—"))
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
                self.page_live.label_status.setText(tr("ui stopping"))
                self.live_thread.stop()
                # the capture thread also drains its queued blocks: give it
                # room before the reference goes away
                self.live_thread.wait(6000)
            dropped = self.live_thread.take.dropped_blocks if (
                self.live_thread.take is not None
            ) else 0
            if dropped:
                # the honesty counter, read where the eye is: the live note
                # count reads hollow (blocks never reached the DSP), the WAV
                # keeps everything — the announcement makes the gap legible
                self.page_live.label_status.setText(
                    tr("ui dropped blocks").format(n=dropped)
                )
            self.live_thread = None
            # a take whose replay was DEFERRED (its analysis arrived while
            # the next capture was already running) starts HERE, where no
            # capture competes for the drain thread
            if not self._replacement_pending_analysis:
                try:
                    self._pump_replay_queue()
                except Exception:
                    pass
        else:
            self.stack.setCurrentWidget(self.page_home)

    def _cancel_phrase_replacement(self, error: str) -> None:
        """Restore the original take when replacement capture cannot finish."""
        context = self._phrase_replacement
        self._phrase_replacement = None
        self._replacement_pending_analysis = False
        self.page_live.set_replacement_mode(False)
        self.page_results.set_replacement_analyzing(False)
        self.page_results.set_replacement_active(False)
        if context is not None:
            original, signal, _first, _last, record_audio = context
            self.result = original
            self._live_signal = signal
            self._live_record_audio = record_audio
            self.take_ctl.set_signal(signal.tobytes())
            self._show_result(original, signal)
        self.show_error(tr("err replacement failed").format(e=error), timeout=8_000)
        self._pump_replay_queue()

    def _live_thread_failed(self, error: str, signal_bytes: bytes = b"") -> None:
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
        if self._phrase_replacement is not None:
            self._cancel_phrase_replacement(error)
            return
        self.page_live.on_failed(error)
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
        self.take_ctl.close()  # the session's queue dies with the window
        super().closeEvent(event)

    def _archive_sidecar(self, stem_name: str, block: dict) -> str | None:
        """Archive ONE take's live/final decomposition beside its artifacts.

        The ONE writer of the per-take sidecar
        (`<prises_live>/<stem>/<stem>.live_vs_final.json`): `_export_results`
        calls it with the take's OWN stem and the result's attached block
        (the LiveTake computed it from the segmenter), and the replay worker
        replaces it with the cross-check row once its replay finishes. The
        same numbers then stay readable by `scripts/replay_takes.py`, the
        coherence page and the weekly trend without parsing the whole
        report. Returns the path written, None when there was nothing to
        write (best-effort: a failed archive never costs the take).
        """
        if not stem_name or not block:
            return None
        from ..core.explain import write_breakdown_file

        return write_breakdown_file(live_export_dir(stem_name) / stem_name, block)

    # ------------------------------------------------------------------
    # Metronome (optional): a steady click to practice regularity
    # ------------------------------------------------------------------
    def on_metronome_toggled(self, on: bool) -> None:
        """Start/stop the click; the spin box follows the same state."""
        self.page_live.spin_bpm.setEnabled(on)
        if on:
            self.page_live.clicks.start()
            # the take's live grid (published by LiveTake) seeds the tempo
            bpm = self._live_grid.bpm
            if bpm and bpm != 120.0:
                self.page_live.spin_bpm.blockSignals(True)
                self.page_live.spin_bpm.setValue(int(round(bpm)))
                self.page_live.spin_bpm.blockSignals(False)
            self.metronome = Metronome(self.page_live.spin_bpm.value(), scheduler=qt_scheduler())
            self.metronome.on_beat = self._on_metronome_beat
            self.metronome.start()
        else:
            if self.metronome is not None:
                self.metronome.stop()
                self.metronome = None
            self.page_live.clicks.stop()
            self.page_live.label_beat.setText("")

    def on_metronome_bpm_changed(self, bpm: int) -> None:
        """A tempo change applies immediately, without restarting the count."""
        if self.metronome is not None:
            self.metronome.bpm = bpm

    def _follow_detected_bpm(self, bpm: float) -> None:
        """The metronome tracks the take's detected tempo as it refines.

        The spin box stays the user's override: it is updated silently, so
        the user can still drag it away and be heard.
        """
        self.page_live.spin_bpm.blockSignals(True)
        self.page_live.spin_bpm.setValue(int(round(bpm)))
        self.page_live.spin_bpm.blockSignals(False)
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
        self.page_live.label_beat.setText(tr("ui beat count").format(beat=beat))

    # ------------------------------------------------------------------
    # Note finder: "which note is this?" on a piano keyboard, nothing else.
    # The page is a Page module (gui/finder_page.py): it owns its widgets,
    # its state (session + thread) and its behavior. The window keeps the
    # public method names the home button and the tests use — thin
    # delegations, no finder logic left here.
    # ------------------------------------------------------------------
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
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
