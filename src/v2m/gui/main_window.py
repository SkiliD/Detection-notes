"""Main window of the Voix -> MIDI application."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
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
from ..core.audio_io import LiveCapture
from ..core.grid import BeatGrid
from ..core.i18n import get_language, note_display_pair, set_language, tr
from ..core.liveseg import LiveNoteSegmenter
from ..core.accuracy import AccuracyHistory, record_take
from ..core.live_session import LiveEvent, LiveTake
from ..core.metronome import Metronome, qt_scheduler
from ..core.music import on_piano
from ..core.pdf_export import PdfExportError
from ..core.pipeline import AnalysisResult, AnalyzeOptions
from ..core.playback import NotePlayer, PlaybackSession
from ..core.report import engrave_pdf, export_result, live_export_dir, live_take_name
from .pages import build_analyze_page, build_finder_page, build_home_page, build_live_page
from .widgets import (
    MISMATCH_COLOR, NOTE_TEXT_COLOR, STYLESHEET, ErrorBanner,
    action_button_style, cents_deviation, tuning_color,
)

AUDIO_EXTS = audio_io.AUDIO_EXTS  # single source of truth (core.audio_io)

# The finder's one answer, in two states: a recognised note is large and in
# the accent tint, the invite to sing stays calm — the same waiting style the
# rest of the app uses (`#6a7480`, italic) so it never reads as a result.
FINDER_NOTE_STYLE = f"font-size: 34pt; font-weight: bold; color: {NOTE_TEXT_COLOR};"
FINDER_IDLE_STYLE = "font-size: 15pt; font-style: italic; color: #6a7480;"


def audio_filter() -> str:
    """File-dialog filter in the current language.

    Built on demand: `tr()` must run after the persisted language is
    restored, so this cannot be a module constant.
    """
    return tr("ui audio filter")


LiveFrame = tuple[float, float] | None


def clock(t: float) -> str:
    """Elapsed seconds as mm:ss (live status line)."""
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


class AnalysisThread(QThread):
    """Run one analysis from a file or an already captured signal."""

    progress = pyqtSignal(float, str)
    finishedOk = pyqtSignal(object, bytes)
    failed = pyqtSignal(str)

    def __init__(self, source: str, opts: AnalyzeOptions, signal: np.ndarray | None = None):
        super().__init__()
        self.source = source
        self.opts = opts
        self.signal = signal

    def run(self) -> None:
        try:
            y = self.signal if self.signal is not None else audio_io.load_file(self.source)
            from ..core.pipeline import analyze_signal

            res = analyze_signal(
                y,
                audio_io.SR,
                source=self.source,
                opts=self.opts,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.finishedOk.emit(res, np.asarray(y, dtype=np.float32).tobytes())
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
    label_live_status: QLabel
    piano_live: Any
    result: AnalysisResult | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        get_language()  # restore the persisted choice before building the UI
        self.setWindowTitle(tr("ui window title"))
        self.resize(1280, 780)
        self.setAcceptDrops(True)

        self.result = None            # last AnalysisResult
        self.analyze_thread: AnalysisThread | None = None
        self.live_thread: LiveThread | None = None
        self.finder_thread: LiveThread | None = None   # note finder (no take kept)
        self.source_path: str | None = None
        self._signal_cache: dict[str, np.ndarray] = {}  # source -> decoded audio
        self._live_signal: np.ndarray | None = None
        self._live_pitch_t: list[float] = []
        self._live_pitch_m: list[float] = []
        self._live_curve = None
        self._live_t = 0.0          # capture time of the last audio block
        self._live_grid = BeatGrid(120.0)  # replaced by the take's published grid
        self._finder_history: list[int] = []
        self._last_finder_midi: int | None = None
        self.accuracy = AccuracyHistory()  # per-note mean |cents|, live take
        self._accuracy_plot_target: str = "live"  # where a plot click seeks
        self._last_take_entry: dict | None = None  # persisted accuracy entry
        self.metronome = None       # created when the user opts in
        self.playback = PlaybackSession(audio_io.SR, on_error=self._on_playback_error)
        self.player: NotePlayer | None = None
        self._voice_player: NotePlayer | None = None
        self.playbackFailed.connect(self._show_playback_error)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)  # ~30 fps playhead updates
        self._play_timer.timeout.connect(self._on_play_tick)
        # live follow: ~20 fps scrolling window + cursor (the audio thread is
        # far too uneven to drive the view directly)
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(50)
        self._live_timer.timeout.connect(self._on_live_tick)

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
        self.page_live = build_live_page(self)
        self.page_finder = build_finder_page(self)
        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_analyze)
        self.stack.addWidget(self.page_live)
        self.stack.addWidget(self.page_finder)
        self.retranslate()

    # Pages are assembled in `gui.pages`; MainWindow owns their state and callbacks.

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
        self._lbl_transpose.setText(tr("ui transpose"))
        self.chk_notation.setText(tr("ui notation mode"))
        self.chk_pdf.setText(tr("ui pdf option"))
        self.chk_pdf.setToolTip(tr("ui pdf tooltip"))
        self.lbl_pdf_note.setVisible(not self.chk_pdf.isEnabled())
        self.lbl_pdf_note.setText(tr("ui pdf badge"))
        self.btn_export.setText(tr("ui re-export"))
        self.btn_open.setText(tr("ui open another"))
        self.btn_play.setText("▶")
        self.btn_play.setToolTip(tr("ui play tooltip"))
        self.btn_voice.setText("🎙")
        self.btn_voice.setToolTip(tr("ui play voice"))
        self.btn_both.setText(tr("ui play both"))
        self.btn_both.setToolTip(tr("ui play both tooltip"))
        self.label_volume.setText(tr("ui volume"))
        self.slider_volume.setToolTip(tr("ui volume tooltip"))
        self.label_play_pos.setText(tr("ui play pos"))
        self.btn_stop_live.setText(tr("ui stop live"))
        self.btn_finder.setText(tr("ui finder button"))
        self.btn_finder_back.setText(tr("ui back home"))
        self.label_finder_hint.setText(tr("ui finder hint"))
        self.label_finder_history.setText(tr("ui finder history"))
        self._refresh_finder_history()
        self.live_indicator.retranslate()
        self.label_live_hint.setText(tr("ui live hint"))
        self._refresh_live_accuracy()
        self.chk_metronome.setText(tr("ui metronome"))
        self.label_bpm.setText(tr("ui bpm unit"))
        self.spin_bpm.setToolTip(tr("ui metronome tooltip"))
        if self.chk_metronome.isChecked():
            self.label_beat.setText(tr("ui beat count").format(beat="—"))
        else:
            self.label_beat.setText("")
        self._refresh_expected_note(self._live_t)
        self._refresh_finder_state()
        for view in (self.staff, self.staff_live):
            view.setToolTip(tr("ui staff tooltip"))
        for roll in (self.piano, self.piano_live):
            roll.set_axis_labels(tr("ui axis note"), tr("ui axis time"))
        if self.live_thread is None or not self.live_thread.isRunning():
            self.label_live_status.setText(tr("ui mic preparing"))
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
        self.analyze_thread.start()

    def _on_progress(self, f: float, msg: str) -> None:
        self.progress.setValue(int(f * 100))
        name = Path(self.source_path).name if self.source_path else tr("ui no file")
        self.label_file.setText(f"{name} — {msg}")

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

    def _on_notation_toggled(self, checked: bool) -> None:
        self.piano.set_notation_mode(checked)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def _export_results(self) -> None:
        if self.result is None:
            return
        res = self.result
        src = Path(res.source if res.source != "live" else live_take_name())
        stem = src.with_suffix("") if src.suffix else src
        if res.source == "live":
            # one sub-folder per take; `res.source` itself must stay "live":
            # the signal cache and the voice button test for that exact value
            stem = live_export_dir(stem) / stem.name
        quantize_s = res.quantize_step(16) if self.chk_quantize.isChecked() else None
        pdf_written = False
        try:
            export_result(
                res, stem, quantize_s=quantize_s, transpose=self.spin_transpose.value()
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
        exts = [".mid", ".musicxml", ".json", ".csv"] + ([".pdf"] if pdf_written else [])
        files = ", ".join(f"{stem.name}{ext}" for ext in exts)
        # the action acknowledges itself where the eye already is
        self.label_play_pos.setText(tr("ui exported ok"))
        self.statusBar().showMessage(tr("ui status exported").format(files=files), 10_000)

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
        self.accuracy.reset()
        self._accuracy_plot_target = "live"
        self._refresh_live_accuracy()
        self._refresh_expected_note(0.0)
        # reset the live state BEFORE the status label: the meter shown at
        # start must be the new take's fallback, not the previous take's
        self._live_grid = BeatGrid(120.0)  # until the take publishes its own
        self.label_live_status.setText(
            tr("ui live progress").format(
                t="00:00", notes=0, meter=self._live_grid.meter.label
            )
        )
        # live pitch curve on the scrolling piano roll (drop the previous take's)
        self.piano_live.begin_live()
        self.piano_live.set_notation_mode(True)
        self.staff_live.clear()
        if self._live_curve is not None:
            self.piano_live.removeItem(self._live_curve)
        self._live_curve = self.piano_live.plot(
            [], [], pen=None, symbol="o", symbolSize=3, symbolBrush=(120, 200, 255, 180)
        )
        self.live_thread = LiveThread()
        self.live_thread.blockStats.connect(self._on_live_block)
        self.live_thread.pitchFrame.connect(self._on_live_pitch)
        self.live_thread.noteFrame.connect(self._on_live_note)
        self.live_thread.gridChanged.connect(self._on_live_grid)
        self.live_thread.analysisFinished.connect(self._on_live_analysis_done)
        self.live_thread.failed.connect(self._on_live_failed)
        self.live_thread.start()
        self._live_timer.start()
        # clicking a plot point seeks the live roll to that note
        try:
            self.accuracy_plot.pointClicked.disconnect()
        except TypeError:
            pass
        self.accuracy_plot.pointClicked.connect(self._on_accuracy_point_clicked)

    def _on_live_block(self, t: float, rms: float) -> None:
        """Telemetry from each captured block: capture time + input level."""
        self._live_t = t
        self.live_indicator.update_rms(rms)

    def _on_live_grid(self, grid) -> None:
        """Consume the take's live grid (published by its owner, LiveTake)."""
        self._live_grid = grid
        self._sync_metronome_to_take(grid)

    def _on_live_tick(self) -> None:
        """Live follow (~20 fps): render the take, scroll, move the cursor.

        Single owner of the live rendering: the audio thread only publishes
        notes, so the roll stays in sync even if a block or an event is late.
        """
        thr = self.live_thread
        if thr is None or thr.segmenter is None:
            return
        t = self._live_t or thr.elapsed()
        notes = thr.segmenter.notes
        preview = thr.segmenter.active_note(t)
        self.piano_live.set_notes(notes)
        self.piano_live.set_live_preview(preview)
        midis = [n.midi for n in notes[-64:]]
        if preview is not None:
            midis.append(preview.midi)
        self.piano_live.set_live_pitch_range(midis)
        self.piano_live.set_live_cursor(t)
        self.staff_live.set_live(notes, preview, grid=self._live_grid)
        self._refresh_expected_note(t)
        self.label_live_status.setText(
            tr("ui live progress").format(
                t=clock(t), notes=len(notes), meter=self._live_grid.meter.label
            )
        )

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
            self.piano_live.set_live_cursor(max(0.0, t))

    def _on_live_note(self, cur: LiveFrame) -> None:
        """Render the recognition state published by the live engine."""
        midi = int(round(cur[0])) if cur else None
        hz = cur[1] if cur else float("nan")
        # the deviation shown is the INSTANTANEOUS one (last smoothed
        # frames): correcting the pitch mid-note must move the needle now
        seg = self.live_thread.segmenter if self.live_thread else None
        inst = seg.recent_midi() if seg is not None else None
        cents_inst = cents_deviation(inst) if inst is not None else None
        self.live_indicator.update_note(cur[0] if cur else None, hz, cents_inst)
        midi = midi if midi is not None and on_piano(midi) else None
        cents = cents_deviation(cur[0]) if cur else None
        onset = None
        if seg is not None:
            onset = seg.note_start if seg.note_start is not None else seg.pending_start
        self.accuracy.add(midi, cents, onset=onset if onset is not None else self._live_t)
        # the key carries the tuning feedback: a wrong note is red however
        # pure, a right/unknown one is green/amber/red by its cents error
        color = None
        if midi is not None:
            expected = self._expected_midi(self._live_t)
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
        self._refresh_live_accuracy()

    def _refresh_live_accuracy(self) -> None:
        """Per-note mean deviation so far: the text line and the plot.

        Both views read the same `AccuracyHistory`, so they can never
        disagree; the line shows the last six notes, the plot the whole take
        in singing order. Each plot point carries its note's onset so a
        click seeks the roll there (live) or to the results playhead (after
        the final analysis).
        """
        items = list(self.accuracy.points)
        self.accuracy_plot.set_points(items, times=self.accuracy.onsets)
        if not items:
            self.label_live_accuracy.setText(tr("ui accuracy empty"))
            return
        parts = "  ".join(
            f"{note_display_pair(m)} {v:.0f}¢" for m, v in reversed(items[-6:])
        )
        self.label_live_accuracy.setText(tr("ui accuracy title") + " " + parts)

    def _on_live_pitch(self, t: float, hz: float) -> None:
        if hz == hz and hz > 0:  # NaN check
            self._live_pitch_t.append(t)
            self._live_pitch_m.append(69 + 12 * np.log2(hz / 440.0))
            # keep the pitch cloud inside the scrolling window
            while self._live_pitch_t and t - self._live_pitch_t[0] > self.piano_live.LIVE_SPAN:
                self._live_pitch_t.pop(0)
                self._live_pitch_m.pop(0)
            if self._live_curve is not None:
                self._live_curve.setData(self._live_pitch_t, self._live_pitch_m)

    def _on_live_failed(self, msg: str) -> None:
        self._live_timer.stop()
        self.label_live_status.setText(tr("ui mic preparing"))
        self._on_error(msg)

    def _on_live_analysis_done(self, res, signal_bytes: bytes = b"") -> None:
        """The shared LiveTake finished its final analysis: show and export it.

        Export FIRST: the take is the user's data, a cosmetic hiccup below
        must never cost it (it used to silently skip the export entirely).
        """
        self._live_timer.stop()
        self.piano_live.set_live_preview(None)
        self.result = res
        # the LiveTake owns the captured signal; recover it from the event
        # bytes when the GUI thread never saw the recording itself
        if self._live_signal is None and signal_bytes:
            self._live_signal = np.frombuffer(signal_bytes, dtype=np.float32).copy()
        self._signal_cache = {"live": np.asarray(self._live_signal, dtype=np.float32)} if self._live_signal is not None else {}
        self.source_path = "live"
        self._accuracy_plot_target = "results"
        # Export FIRST: the take is the user's data, a cosmetic hiccup below
        # must never cost it (it used to silently skip the export entirely).
        self._export_results()
        self.stop_playback()
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
        self._last_take_entry = record_take(
            self.accuracy,
            duration_s=len(self._live_signal) / audio_io.SR if self._live_signal is not None else 0.0,
            bpm=res.tempo.bpm,
        )

    def _show_result(self, res) -> None:
        """Fill the analysis page from a fresh result (file or live take)."""
        self.stack.setCurrentWidget(self.page_analyze)
        self.label_file.setText(tr("ui live file") if res.source == "live" else Path(res.source).name)
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
            grid=(res.grid if res.grid is not None else BeatGrid(res.tempo.bpm)),
            key=getattr(res.key, "best", None),
            positions=getattr(res, "note_positions", None),
            rests=getattr(res, "rests", None),
        )

    def on_stop_live(self) -> None:
        self._live_timer.stop()
        self.piano_live.set_live_preview(None)
        self.piano_live_keys.set_pressed()
        self.staff_live.clear()
        if self.metronome is not None:
            self.metronome.stop()
            self.label_beat.setText(tr("ui beat count").format(beat="—"))
        if self.live_thread is not None and self.live_thread.isRunning():
            self.label_live_status.setText(tr("ui stopping"))
            self.live_thread.stop()
            # the capture thread also drains its queued blocks: give it room
            self.live_thread.wait(6000)
            self.live_thread = None
        else:
            self.stack.setCurrentWidget(self.page_home)

    # ------------------------------------------------------------------
    # Metronome (optional): a steady click to practice regularity
    # ------------------------------------------------------------------
    def on_metronome_toggled(self, on: bool) -> None:
        """Start/stop the click; the spin box follows the same state."""
        self.spin_bpm.setEnabled(on)
        if on:
            self.clicks.start()
            # the take's live grid (published by LiveTake) seeds the tempo
            bpm = self._live_grid.bpm
            if bpm and bpm != 120.0:
                self.spin_bpm.blockSignals(True)
                self.spin_bpm.setValue(int(round(bpm)))
                self.spin_bpm.blockSignals(False)
            self.metronome = Metronome(self.spin_bpm.value(), scheduler=qt_scheduler())
            self.metronome.on_beat = self._on_metronome_beat
            self.metronome.start()
        else:
            if self.metronome is not None:
                self.metronome.stop()
                self.metronome = None
            self.clicks.stop()
            self.label_beat.setText("")

    def on_metronome_bpm_changed(self, bpm: int) -> None:
        """A tempo change applies immediately, without restarting the count."""
        if self.metronome is not None:
            self.metronome.bpm = bpm

    def _follow_detected_bpm(self, bpm: float) -> None:
        """The metronome tracks the take's detected tempo as it refines.

        The spin box stays the user's override: it is updated silently, so
        the user can still drag it away and be heard.
        """
        self.spin_bpm.blockSignals(True)
        self.spin_bpm.setValue(int(round(bpm)))
        self.spin_bpm.blockSignals(False)
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
        self.clicks.play(accent)
        self.label_beat.setText(tr("ui beat count").format(beat=beat))

    # ------------------------------------------------------------------
    # Note finder: "which note is this?" on a piano keyboard, nothing else
    # ------------------------------------------------------------------
    def on_open_finder(self) -> None:
        self.error_banner.clear()
        self.stack.setCurrentWidget(self.page_finder)
        if not self._finder_running():
            self.on_toggle_finder()

    def on_toggle_finder(self) -> None:
        if self._finder_running():
            self._stop_finder()
            return
        self.error_banner.clear()
        self._set_finder_note(None)
        self._finder_history.clear()
        self._last_finder_midi = None
        self._refresh_finder_history()
        self.finder_thread = LiveThread()
        self.finder_thread.noteFrame.connect(self._on_finder_note)
        self.finder_thread.failed.connect(self._on_finder_failed)
        self.finder_thread.start()
        self._refresh_finder_state()

    def on_close_finder(self) -> None:
        """Leave the finder: the capture is dropped, never analysed nor exported."""
        self._stop_finder()
        self.stack.setCurrentWidget(self.page_home)

    def _finder_running(self) -> bool:
        return self.finder_thread is not None and self.finder_thread.isRunning()

    def _stop_finder(self) -> None:
        if self.finder_thread is not None and self.finder_thread.isRunning():
            self.finder_thread.stop()
            self.finder_thread.wait(6000)
        self.finder_thread = None
        self._refresh_finder_state()

    def _refresh_finder_state(self) -> None:
        """The finder's chrome for the current language and mic state.

        One writer for the three things that must agree: the button (its
        label, and its weight — the inviting action is the filled one), the
        state line, and the note area, which falls back on the invite when
        nothing is being sung.
        """
        running = self._finder_running()
        self.btn_finder_toggle.setText(
            tr("ui finder stop") if running else tr("ui finder start")
        )
        self.btn_finder_toggle.setStyleSheet(action_button_style(primary=not running))
        self.label_finder_state.setText(tr("ui finder live") if running else tr("ui finder off"))
        if not running:
            self._set_finder_note(None)

    def _on_finder_note(self, cur: LiveFrame) -> None:
        """Render the note and retain only distinct recent pitches."""
        midi = int(round(cur[0])) if cur else None
        self._set_finder_note(midi)
        if midi is not None and on_piano(midi) and midi != self._last_finder_midi:
            self._finder_history.append(midi)
            self._finder_history = self._finder_history[-6:]
            self._last_finder_midi = midi
            self._refresh_finder_history()

    def _refresh_finder_history(self) -> None:
        self.finder_history.clear()
        if not self._finder_history:
            self.finder_history.addItem(QListWidgetItem(tr("ui finder history empty")))
            return
        for midi in reversed(self._finder_history):
            self.finder_history.addItem(QListWidgetItem(f"🎹  {note_display_pair(midi)}"))

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
            self.label_live_expected.setText(tr("ui expected none"))
        else:
            self.label_live_expected.setText(
                tr("ui expected note").format(note=note_display_pair(expected))
            )

    def _set_finder_note(self, midi_float) -> None:
        """Show the note as a name and a lit key; nothing outside the piano.

        The invite and the answer do not share a style: a 34 pt accent name is
        an answer, and a first-run page must not shout one it does not have.
        """
        midi = None if midi_float is None else int(round(float(midi_float)))
        if midi is None or not on_piano(midi):
            self.label_finder_note.setText(tr("ui finder idle"))
            self.label_finder_note.setStyleSheet(FINDER_IDLE_STYLE)
            self.finder_keys.set_pressed()
            return
        self.label_finder_note.setText(note_display_pair(midi))
        self.label_finder_note.setStyleSheet(FINDER_NOTE_STYLE)
        self.finder_keys.set_pressed({midi})

    def _on_finder_failed(self, msg: str) -> None:
        self._stop_finder()
        self._on_error(msg)

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
