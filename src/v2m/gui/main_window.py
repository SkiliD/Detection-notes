"""Main window of the Voix -> MIDI application."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import audio_io
from ..core.accuracy import AccuracyHistory, record_take
from ..core.audio_io import LiveCapture
from ..core.continuity import coherence_report
from ..core.grid import BeatGrid
from ..core.i18n import get_language, note_display_pair, set_language, tr
from ..core.live_session import LiveEvent, LiveTake
from ..core.liveseg import LiveNoteSegmenter
from ..core.metronome import Metronome, qt_scheduler
from ..core.music import on_piano
from ..core.pdf_export import PdfExportError
from ..core.pipeline import AnalysisResult, AnalyzeOptions
from ..core.compare import match as compare_match
from ..core.playback import NotePlayer, PlaybackSession
from ..core.report import engrave_pdf, export_result
from ..core import takes
from ..core.takes import live_export_dir, live_take_name
from .finder_page import FinderPage
from .live_page import LIVE_SILENCE_WARN_S, LivePage
from .live_session_controller import LiveSessionController
from .pages import build_analyze_page, build_home_page
from .widgets import (
    MISMATCH_COLOR,
    NOTE_TEXT_COLOR,
    STYLESHEET,
    ErrorBanner,
    action_button_style,
    cents_deviation,
    tuning_color,
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


def clock(t: float) -> str:
    """Elapsed seconds as mm:ss (live status line)."""
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"



class AnalysisThread(QThread):
    """Run one analysis from a file or an already captured signal."""

    progress = pyqtSignal(float, str)
    finishedOk = pyqtSignal(object, bytes)
    failed = pyqtSignal(str)
    # whether the take's own observation cache served the analysis (True)
    # or pYIN ran (False): the GUI tells the user WHY an analysis was instant
    cacheUsed = pyqtSignal(bool)

    def __init__(self, source: str, opts: AnalyzeOptions, signal: np.ndarray | None = None):
        super().__init__()
        self.source = source
        self.opts = opts
        self.signal = signal
        self.used_cache = False

    def run(self) -> None:
        try:
            from ..core.pipeline import analyze_file, analyze_signal

            if self.signal is not None:
                y = self.signal
                res = analyze_signal(
                    y,
                    audio_io.SR,
                    source=self.source,
                    opts=self.opts,
                    progress=lambda f, m: self.progress.emit(f, m),
                )
            else:
                # FILE path: through analyze_file, which reads the take's
                # own cached observation (<take>.observation.npz) when the
                # audio has not changed — the pYIN pass is skipped and the
                # interpretation reads the same evidence. A file path is a
                # take on disk; a captured signal is not (its observation
                # cache does not exist).
                y = audio_io.load_file(self.source)
                from ..core.pipeline import cached_observation

                self.used_cache = cached_observation(
                    self.source, y, audio_io.SR, self.opts
                ) is not None
                res = analyze_file(
                    self.source,
                    opts=self.opts,
                    progress=lambda f, m: self.progress.emit(f, m),
                )
            self.finishedOk.emit(res, np.asarray(y, dtype=np.float32).tobytes())
            self.cacheUsed.emit(self.used_cache)
        except Exception as e:
            self.failed.emit(str(e))


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
    failed = pyqtSignal(str)

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
            self.failed.emit(event.error)

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
                self.failed.emit(str(error))

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
    label_file: QLabel
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
        self.accuracy = AccuracyHistory()  # per-note mean |cents|, live take
        self._last_take_entry: dict | None = None  # persisted accuracy entry
        self._analysis_started_at: float = 0.0  # cache-status timing (perf_counter)
        self._replay_worker = None  # ReplayWorker: the take's own cross-check
        # the take in progress, ONE owner: stem (minted once), signal (the
        # event's bytes), replay queue (survives a new take), sidecar and
        # timer decisions — the deep module the 2026-09-27 regressions
        # pointed at; this window is its Qt adapter (applies its Verdicts)
        self.take_ctl = LiveSessionController()
        self.metronome = None       # created when the user opts in
        self.playback = PlaybackSession(audio_io.SR, on_error=self._on_playback_error)
        self.player: NotePlayer | None = None
        self._voice_player: NotePlayer | None = None
        self.playbackFailed.connect(self._show_playback_error)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)  # ~30 fps playhead updates
        self._play_timer.timeout.connect(self._on_play_tick)

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

        self.page_home = build_home_page(self)
        self.page_analyze = build_analyze_page(self)
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
            piano_volume_changed=self.on_piano_volume_changed,
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
        )
        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_analyze)
        self.stack.addWidget(self.page_live)
        self.stack.addWidget(self.page_finder)
        self.retranslate()

    # Pages are assembled in `gui.pages`; MainWindow owns their state and callbacks.

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
        self.results.retranslate()

    def retranslate(self) -> None:
        """Refresh every window string in the current language."""
        self.setWindowTitle(tr("ui window title"))
        self._title.setText("<h1>🎤 Voix → MIDI</h1>")
        self._subtitle.setText(tr("ui app subtitle"))
        self._steps.setText(tr("ui steps"))
        self.btn_file.setText(tr("ui open file"))
        self.btn_live.setText(tr("ui sing live"))
        self.btn_lang.setText(
            tr("ui lang button") if get_language() == "fr" else "🌐  Langue : English"
        )
        self._hint.setText(tr("ui drop hint"))
        self.btn_back.setText(tr("ui back home"))
        if self.source_path:
            self.label_file.setText(Path(self.source_path).name)
        else:
            self.label_file.setText(tr("ui no file"))
        self._grp_opts.setTitle(tr("ui export options"))
        self.chk_quantize.setText(tr("ui quantize"))
        self._lbl_min_note.setText(tr("ui min note"))
        self.btn_reinterpret.setText(tr("ui reinterpret"))
        self.btn_reinterpret.setToolTip(tr("ui reinterpret tooltip"))
        self._lbl_transpose.setText(tr("ui transpose"))
        self.chk_notation.setText(tr("ui notation mode"))
        self.chk_pdf.setText(tr("ui pdf option"))
        self.chk_pdf.setToolTip(tr("ui pdf tooltip"))
        self.lbl_pdf_note.setVisible(not self.chk_pdf.isEnabled())
        self.lbl_pdf_note.setText(tr("ui pdf badge"))
        self.btn_export.setText(tr("ui re-export"))
        self.btn_open.setText(tr("ui open another"))
        self.btn_coherence.setText(tr("ui coherence sessions btn"))
        self.btn_play.setText("▶")
        self.btn_play.setToolTip(tr("ui play tooltip"))
        self.btn_voice.setText("🎙")
        self.btn_voice.setToolTip(tr("ui play voice"))
        self.btn_both.setText(tr("ui play both"))
        self.btn_both.setToolTip(tr("ui play both tooltip"))
        self.label_volume.setText(tr("ui volume"))
        self.slider_volume.setToolTip(tr("ui volume tooltip"))
        self.label_play_pos.setText(tr("ui play pos"))
        self.page_live.btn_stop_live.setText(tr("ui stop live"))
        self.btn_finder.setText(tr("ui finder button"))
        self.page_finder.retranslate()
        self.page_live.retranslate()
        self._refresh_live_accuracy()
        self._refresh_expected_note(self._live_t)
        for view in (self.staff, self.page_live.staff_live):
            view.setToolTip(tr("ui staff tooltip"))
        for roll in (self.piano, self.page_live.piano_live):
            roll.set_axis_labels(tr("ui axis note"), tr("ui axis time"))
        if self.live_thread is None or not self.live_thread.isRunning():
            self.page_live.label_status.setText(tr("ui mic preparing"))
        self.results.retranslate()

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
    # Playback (ear-check of the analysis, synced playhead)
    # ------------------------------------------------------------------
    def _piano_gain(self) -> float:
        """Slider 0-100 -> render gain (70 = neutral, 0 = mute)."""
        return self.slider_volume.value() / 70.0

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


    # ------------------------------------------------------------------
    # Playback. ONE player for the three sources (piano, voice, both):
    # `playback` owns the clock and the output stream, the tick renders
    # from it, so the playhead follows whichever mode is sounding. Every
    # new playback starts by stopping the previous one.
    # ------------------------------------------------------------------

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
        mode = self.playback.mode if self.playback.playing else None
        self.btn_play.setText("■" if mode == "piano" else "▶")
        self.btn_voice.setText("■" if mode == "voice" else "🎙")
        self.btn_both.setText("■" if mode == "both" else "🎹+🎙")

    def _make_voice_player(self) -> NotePlayer | None:
        """A rendered voice source at the live transposition, or None without a take."""
        signal = self._voice_signal()
        if signal is None:
            return None
        return NotePlayer(
            signal=signal, sr=audio_io.SR, transpose=self.spin_transpose.value(),
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
        self.piano.set_playhead(t, follow=False)
        self.waveform.set_position(t)
        duration = self.playback.duration or self.player.duration
        if t >= duration - 0.05:
            return
        # Preserve the source selected by the user, including piano + voice.
        self._start(self.playback.mode or "piano", t)

    def _on_play_tick(self) -> None:
        if not self.playback.playing:
            self._play_timer.stop()
            self._sync_play_buttons()
            self.piano_keys.set_playhead(-1.0)
            return
        t = self.playback.position()
        duration = self.playback.duration
        self.piano.set_playhead(t)
        self.waveform.set_position(t)
        self.piano_keys.set_playhead(t)
        self.label_play_pos.setText(
            f"{int(t // 60):02d}:{int(t % 60):02d} / {int(duration // 60):02d}:{int(duration % 60):02d}"
        )
        self._sync_play_buttons()

    def stop_playback(self) -> None:
        self._play_timer.stop()
        self.playback.stop()
        self.btn_play.setText("▶")
        self.btn_voice.setText("🎙")
        self.btn_both.setText("🎹+🎙")
        self.piano_keys.set_playhead(-1.0)  # all keys up

    def on_back_home(self) -> None:
        """Leave the analyze page: the playback must NOT keep sounding.

        The user report ("when you leave the page and come back home, the
        audio keeps playing"): every lambda that only switched pages left the
        PlaybackSession running — the page changed, the sound didn't. One
        owner for the navigation back, and it stops the audio first.
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
        self.stop_playback()
        opts = AnalyzeOptions(min_note_s=self.spin_min_note.value() / 1000.0)
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
        self.btn_play.setText("▶")
        self.label_play_pos.setText(tr("ui analyzing file"))
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

        self.stack.setCurrentWidget(self.page_analyze)
        self.source_path = path
        self.label_file.setText(p.name)
        self.progress.setValue(0)
        opts = AnalyzeOptions()  # default thresholds, one policy (pipeline)
        self.analyze_thread = AnalysisThread(path, opts)
        self.analyze_thread.progress.connect(self._on_progress)
        self.analyze_thread.finishedOk.connect(self._on_analysis_done)
        self.analyze_thread.failed.connect(self._on_error)
        self.analyze_thread.cacheUsed.connect(self._on_cache_used)
        self.analyze_thread.start()
        self._analysis_started_at = time.perf_counter()

    def _on_progress(self, f: float, msg: str) -> None:
        self.progress.setValue(int(f * 100))
        name = Path(self.source_path).name if self.source_path else tr("ui no file")
        self.label_file.setText(f"{name} — {msg}")

    def _on_cache_used(self, used: bool) -> None:
        """Say WHY an analysis was instant (the take's evidence was reused)."""
        if not used:
            return
        elapsed_ms = 1000.0 * (time.perf_counter() - self._analysis_started_at)
        self.statusBar().showMessage(
            tr("ui status cache").format(ms=elapsed_ms), 8_000
        )

    def on_show_coherence(self) -> None:
        """Aggregate the user's takes into the per-session table, on screen.

        The SAME core (`core.coherence`) the CLI script reads: one policy,
        two front-ends. Best-effort — a read-only or empty folder shows the
        invite, never an error banner (this is a viewer, not an action).
        """
        from ..core import coherence as coh

        text = ""
        try:
            root = coh.LIVE_TAKES_DIR
            takes = coh.takes_under(root)
            if not takes:
                self.coherence_text.setVisible(True)
                self.coherence_text.setPlainText(tr("ui coherence none"))
                return
            coh.update_journal(root, takes)
            text = coh.session_table_text(takes, root=root)
            # the practice-WINDOW view (morning/afternoon/evening): when is
            # the singer best? The windows read the same audits — no new
            # evidence — and only the sung windows appear.
            windows = coh.practice_time_report(takes)
            if windows:
                text += "\n" + tr("ui practice windows title") + "\n"
                for w in windows:
                    rate = (
                        f"{w['suspect_rate']:.1f}" if w["suspect_rate"] is not None else "-"
                    )
                    contours = (
                        f"{w['contour_rate']*100:.0f}%" if w["contour_rate"] is not None else "-"
                    )
                    text += tr("ui practice window row").format(
                        window=tr(f"time {w['window']}"), takes=w["takes"],
                        rate=rate, contours=contours,
                    ) + "\n"
        except Exception as e:  # a viewer must never break the home page
            text = tr("err ui unreadable").format(e=e)
        self.coherence_text.setVisible(True)
        self.coherence_text.setPlainText(text)
        try:
            self._refresh_health_chart()
        except Exception:  # the chart is a nicety, the table is the product
            pass

    def _refresh_health_chart(self) -> None:
        """Draw the weekly contour-rate curve from the JOURNAL.

        One point per week (the same rows `session_table_text` lists as
        text), the rate in percent on y — lower is better. The X positions
        are the weeks' index (categorical): the labels live in the tooltip
        and in the click-through, not on an axis, because ISO weeks are not
        a continuous time axis. Replaces the ASCII sparkline ON THIS PAGE —
        the CLI keeps its text trends.
        """
        from ..core import coherence as coh

        weeks = coh.contour_trend(coh.LIVE_TAKES_DIR)
        self.health_plot.clear()
        if not weeks:
            self.health_plot.setVisible(False)
            return
        self._health_weeks = weeks
        xs = list(range(len(weeks)))
        ys = [w["contour_rate"] * 100.0 for w in weeks]
        # the rate line, then each week as a clickable point (ACCENT = the
        # app's one accent, already the outlines' and the counter's color)
        self.health_plot.plot(
            xs, ys, pen=pg.mkPen("#4da3d8", width=2), antialias=True,
        )
        scatter = pg.ScatterPlotItem(
            xs, ys, size=9, brush=pg.mkBrush("#4da3d8"), pen=pg.mkPen("#0e1218"),
        )
        self.health_plot.addItem(scatter)
        self.health_plot.setXRange(-0.5, len(weeks) - 0.5, padding=0)
        self.health_plot.setYRange(0, 100, padding=0)
        self.health_plot.setVisible(True)

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

    def _on_analysis_done(self, res, signal_bytes: bytes) -> None:
        self.result = res
        # keep the decoded signal so re-display never re-decodes the file
        self._signal_cache = {res.source: np.frombuffer(signal_bytes, dtype=np.float32)}
        self._live_signal = None  # a file view must never show a live take
        self.stop_playback()
        self.player = NotePlayer(
            res.notes, sr=audio_io.SR, gain=self._piano_gain(),
        )
        self._show_result(res)
        self._export_results()
        # the coherence verdict in the status bar too: it is the one line
        # the singer reads without opening the JSON (results panel carries
        # the same verdict as a row)
        summary = self._coherence_summary(res)
        if summary:
            self.statusBar().showMessage(
                tr("ui status coherence").format(summary=summary), 10_000
            )

    def _on_notation_toggled(self, checked: bool) -> None:
        self.piano.set_notation_mode(checked)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def _export_results(self) -> None:
        if self.result is None:
            return
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
        quantize_s = res.quantize_step(16) if self.chk_quantize.isChecked() else None
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
                res, stem, quantize_s=quantize_s, transpose=self.spin_transpose.value(),
                # the take's live reading, for the report's live_vs_final
                # breakdown (the gap decomposed: ±1 roundings, moves,
                # absorbed glides) — a file analysis hands over none
                live_notes=(
                    self._last_take_live_notes if res.source == "live" else None
                ),
            )
            if self.chk_pdf.isChecked():
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
            archive_stem = self.take_ctl.handle_export(
                getattr(res, "live_vs_final", None) or {}
            )
            archive = (
                self._archive_sidecar(archive_stem, getattr(res, "live_vs_final", None) or {})
                if archive_stem
                else None
            )
            if archive:
                exts.append(takes.SIDECAR_SUFFIX)
            # the retention lives at export time: the take folder was just
            # written, so the disk's size is current. The oldest take
            # folders are trimmed WHOLE under the documented budget — the
            # journal keeps their audits, the trend survives (its rule).
            try:
                from ..core import coherence as coh

                pruned = coh.prune_takes(coh.LIVE_TAKES_DIR)
                if pruned:
                    self.statusBar().showMessage(
                        tr("ui retention pruned").format(n=len(pruned), stems=", ".join(pruned)),                        8_000,
                    )
            except Exception:  # a trim failure must never cost the take
                pass
        files = ", ".join(f"{stem.name}{ext}" for ext in exts)
        # the action acknowledges itself where the eye already is
        self.label_play_pos.setText(tr("ui exported ok"))
        self.statusBar().showMessage(tr("ui status exported").format(files=files), 10_000)
        # the take's WAV now exists on disk: its queued cross-check may
        # start — the controller's guards still hold (a running capture or
        # replay keeps it waiting); a file analysis's export pumps an empty
        # queue, the call is free
        self._pump_replay_queue()

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
        self.accuracy.reset()
        # the audio box is read HERE (before the capture starts): the choice
        # belongs to the take, not to the moment its export happens
        self._live.record_audio = self.page_live.chk_record_audio.isChecked()
        # the NEW capture is the DSP priority: a replay running (its
        # real-time wait loop costs the same drain as the capture) is
        # BROKEN HERE, not joined — the worker skips the rest of its take
        # and exits on its own thread. Its stem is NOT re-queued: the
        # cross-check is a nicety, a deferral that starved live captures
        # is the 2026-09-27 regression this whole queue exists to prevent
        worker = self._replay_worker
        if worker is not None and worker.isRunning():
            worker.stop()
        # the NEW take owns every buffer through the controller (stem,
        # signal): the previous take's name must never survive into this
        # take's sidecar or replay — but the replay QUEUE survives, its
        # entries belong to already-exported takes
        self.take_ctl.begin_take()
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
        self.live_thread.failed.connect(self.page_live.on_failed)
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
            self.stack.setCurrentWidget(self.page_analyze)
            self.piano.set_playhead(t)
            return
        if self.live_thread is not None and self.live_thread.isRunning() and t < self._live_t:
            self.stack.setCurrentWidget(self.page_live)
            self.page_live.piano_live.set_live_cursor(max(0.0, t))

    def _on_live_analysis_done(self, res, signal_bytes: bytes = b"") -> None:
        """The shared LiveTake finished its final analysis: show and export it.

        Export FIRST: the take is the user's data, a cosmetic hiccup below
        must never cost it (it used to silently skip the export entirely).
        """
        # the live timer is the RENDER of the CURRENT capture: the 2026-09-27
        # chained-takes regression stopped it unconditionally here, so the
        # take the singer had ALREADY started (its analysis arrives queued)
        # kept recording while its roll, counter and cursor froze — the
        # "the mic stopped detecting me" feel. The CONTROLLER decides (it
        # also queues this take's cross-check replay): the adapter applies.
        verdict = self.take_ctl.handle_analysis(
            capture_running=(
                self.live_thread is not None and self.live_thread.isRunning()
            ),
            signal_bytes=signal_bytes,  # the event's bytes AS THEY ARE: an
            # EMPTY payload CLEARS the buffer (the controller's rule) —
            # `or None` here once turned an empty event into "keep the
            # previous take's signal", the 2026-09-26 stale-WAV family
        )
        if verdict.keep_timer:
            self.page_live.start_timer()  # a newer capture owns the view: keep it
        else:
            self.page_live.stop_timer()
        self.page_live.piano_live.set_live_preview(None)
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
        self._export_results()
        self.stop_playback()
        # the SESSION's coherence verdict, right after this take's export:
        # the singer reads the session's health without running the script
        # (same core as the home-page button and the CLI script)
        try:
            from ..core import coherence as coh

            session_takes = coh.takes_under(coh.LIVE_TAKES_DIR)
            if session_takes:
                rows = coh.session_rows(session_takes)
                if rows:
                    last = rows[-1]
                    if last["checked"] is not None:
                        self.statusBar().showMessage(
                            tr("ui coherence session verdict").format(
                                suspects=last["suspects"],
                                checked=last["checked"],
                                takes=last["takes"],
                            ),
                            10_000,
                        )
        except Exception:
            pass  # the take is already exported; the verdict must never cost it
        self.player = NotePlayer(
            res.notes, sr=audio_io.SR, gain=self._piano_gain(),
            on_error=self._on_playback_error,
        )
        self.playback.set_sources(piano=self.player.signal, voice=self._live_signal)
        try:
            self._show_result(res)
        except Exception as e:  # a broken view must not swallow the take
            self.show_error(str(e), timeout=8_000)
        self.statusBar().showMessage(
            tr("ui status live done").format(
                bpm=res.tempo.bpm, key=res.key.best.label, n=len(res.notes)
            ),
            12_000,
        )
        # archive this take's tuning history (progress across sessions);
        # a failure here must never cost the take already exported
        dur = len(self._live_signal) / audio_io.SR if self._live_signal is not None else 0.0
        self._last_take_entry = record_take(
            self.accuracy,
            duration_s=dur,
            bpm=res.tempo.bpm,
        )
        # the take's live/final gap is already archived BESIDE its artifacts
        # (by `_export_results`, the ONE writer of the sidecar); the take is
        # then REPLAYED through the real chain (`ReplayWorker`, its own
        # thread) to cross-check the engine on its own audio: the audit row
        # lands in `<stem>.live_vs_final.json`, which `scripts/replay_takes.py`
        # and the weekly health trend read back. Best-effort: the take is
        # already exported, a replay failure must never cost it.
        try:
            # the finished take's cross-check goes through the controller:
            # it QUEUES it and decides (from the adapter's facts) whether it
            # may start now or must wait for the next quiet moment
            self._pump_replay_queue()
        except Exception:
            self._replay_worker = None  # no replay: the take is already archived
        # the multi-week REAL-TIME HEALTH regression, announced once the
        # session's journal was refreshed above: a deterioration of the
        # contour rate is the one number that says "the live view is drifting
        # from the final pass" — the singer reads it where they stand
        try:
            self._announce_regression()
        except Exception:  # a warning must never cost the take
            pass

    def _pump_replay_queue(self) -> None:
        """Start the next queued replay, ONLY when nothing competes for DSP.

        THE 2026-09-27 regression this pump guards: a replay's real-time DSP
        (the same `LiveCapture` -> YIN -> segmenter cost as a capture)
        starved the LIVE capture's drain thread when both ran — on the four
        chained takes of that evening the live counter confirmed 2-12 notes
        while replaying the same WAVs confirmed 17-23. The CONTROLLER owns
        the queue and the decision (its `pump_replay` reads the adapter's
        facts: capture timer, worker thread); this adapter only applies —
        start the named worker, re-pump from `on_stop_live` and from the
        worker's own `finished` signal.
        """
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

    def _show_result(self, res) -> None:
        """Fill the analysis page from a fresh result (file or live take)."""
        self.stack.setCurrentWidget(self.page_analyze)
        name = Path(res.source).name if res.source != "live" else tr("ui live file")
        self.label_file.setText(name)
        self.progress.setValue(100)
        self.results.set_results(res)
        signal = self._live_signal
        if res.source == "live" and signal is not None:
            self.waveform.set_signal(signal, audio_io.SR)
        elif res.source in self._signal_cache:
            self.waveform.set_signal(self._signal_cache[res.source], audio_io.SR)
        self.piano.set_notation_mode(self.chk_notation.isChecked())
        self.piano.set_notes(res.notes)
        self.piano_keys.set_notes(res.notes)
        self.piano.set_pitch_curve(res.pitch)
        self.staff.set_score(
            res.notes,
            # an incomplete result (notes-only re-read, no tempo) shows on the
            # rendered default grid: set_score's `BeatGrid(None)` owns that
            # decision, never build a grid from a None tempo here
            grid=(res.grid if res.grid is not None
                  else BeatGrid(res.tempo.bpm) if res.tempo is not None else None),
            key=getattr(res.key, "best", None),
            positions=getattr(res, "note_positions", None),
            rests=getattr(res, "rests", None),
        )

    def on_stop_live(self) -> None:
        self.page_live.stop_timer()
        self.page_live.piano_live.set_live_preview(None)
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
            self.live_thread = None
            # a take whose replay was DEFERRED (its analysis arrived while
            # the next capture was already running — the chained-takes case
            # that starved the live DSP, 2026-09-27) starts HERE, where no
            # capture competes for the drain thread
            try:
                self._pump_replay_queue()
            except Exception:
                pass
        else:
            self.stack.setCurrentWidget(self.page_home)

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

    def _on_health_clicked(self, ev) -> None:
        """A week was clicked on the health chart: show ITS replayed takes.

        The mapping is positional (the click's x is rounded to the nearest
        week index) and the drill-down reads the JOURNAL's own stems — the
        same rows the text table lists, never a folder guess. Double-click
        handling is left to pyqtgraph: any accepted click drills down.
        """
        import pyqtgraph as pg

        if getattr(ev, "button", lambda: None)() != pg.QtCore.Qt.MouseButton.LeftButton:
            return
        weeks = getattr(self, "_health_weeks", None)
        if not weeks:
            self.statusBar().showMessage(tr("ui health click none"), 5_000)
            return
        try:
            pos = self.health_plot.getPlotItem().vb.mapSceneToView(ev.scenePos())
            idx = int(round(pos.x()))
        except Exception:
            idx = -1
        if not 0 <= idx < len(weeks):
            self.statusBar().showMessage(tr("ui health click none"), 5_000)
            return
        week = weeks[idx]
        stems = week.get("stems") or []
        if not stems:
            self.statusBar().showMessage(tr("ui health click none"), 5_000)
            return
        self.statusBar().showMessage(
            tr("ui health click").format(week=week["week"], n=len(stems)), 8_000
        )
        # the drill-down shows the WEEK'S OWN takes through the session
        # table's reader: stems -> audits -> one text block, plus each
        # take's own line (the compact table shows sessions, the singer
        # clicked to see TAKES — the existing i18n formats render them)
        try:
            from ..core import coherence as coh

            take_audits = coh.takes_under(coh.LIVE_TAKES_DIR)
            by_stem = {t.stem: t for t in take_audits}
            week_takes = [by_stem[s] for s in stems if s in by_stem]
            missing = len(stems) - len(week_takes)
            detail = coh.session_table_text(week_takes)
            lines = [tr("ui health click").format(week=week["week"], n=len(stems)), ""]
            lines.extend(detail.splitlines())
            lines.append("")
            for t in week_takes:
                checked = f"{t.checked}" if t.checked is not None else "-"
                rate = f"{t.contour_rate*100:.0f}%" if t.contour_rate is not None else "-"
                if t.verify_ok is False:
                    lines.append(tr("coh take divergent").format(
                        stem=t.stem, checked=checked, rate=rate))
                elif not t.suspects:
                    lines.append(tr("coh take clean").format(
                        stem=t.stem, checked=checked) + f"  ({rate})")
                else:
                    lines.append(tr("coh take suspects").format(
                        stem=t.stem, checked=checked, rate=rate,
                        n=len(t.suspects)))
            if missing:
                lines.append(
                    tr("ui health click none") if missing == len(stems) else
                    f"  ({missing})"
                )
            self.coherence_text.setVisible(True)
            self.coherence_text.setPlainText("\n".join(lines))
        except Exception:  # a read-only drill-down must never break the page
            pass

    def _announce_regression(self) -> None:
        """Warn when the real-time health DEGRADED against the previous week.

        The journal was just refreshed with this take's audit: `regression_report`
        compares the last two weeks' contour rates (the ONE threshold,
        `coherence.REGRESSION_DELTA`); the banner shows the measured points,
        never a judgment. Silent when healthy, when the data is missing, or
        when a lone take cannot back the comparison.
        """
        from ..core import coherence as coh

        weeks = coh.contour_trend(coh.LIVE_TAKES_DIR)
        reg = coh.regression_report(weeks)
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
