"""Main window of the Voix -> MIDI application."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import audio_io
from ..core.audio_io import LiveCapture
from ..core.figures import label_notes
from ..core.grid import BeatGrid
from ..core.i18n import get_language, note_display_pair, set_language, tr
from ..core.liveseg import LiveNoteSegmenter, LivePitchStream
from ..core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI, hz_to_midi, on_piano
from ..core.pdf_export import PdfExportError, find_notation_tool
from ..core.pipeline import AnalyzeOptions
from ..core.playback import NotePlayer
from ..core.report import engrave_pdf, export_result
from ..core.tempo import estimate_bpm_live
from .staff import StaffView
from .widgets import (
    HINT_CARD_STYLE,
    NOTE_TEXT_COLOR,
    STYLESHEET,
    ErrorBanner,
    LiveIndicator,
    PianoRoll,
    PianoView,
    ResultsPanel,
    WaveformView,
    action_button_style,
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


def clock(t: float) -> str:
    """Elapsed seconds as mm:ss (live status line)."""
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


class AnalyzeThread(QThread):
    """Analyze an audio file in a background thread."""

    progress = pyqtSignal(float, str)
    finishedOk = pyqtSignal(object, bytes)  # result + raw signal (for the waveform)
    failed = pyqtSignal(str)

    def __init__(self, path: str, opts: AnalyzeOptions):
        super().__init__()
        self.path = path
        self.opts = opts

    def run(self) -> None:
        try:
            y = audio_io.load_file(self.path)  # decoded exactly once
            res = self._analyze(y)
            self.finishedOk.emit(res, np.asarray(y, dtype=np.float32).tobytes())
        except Exception as e:
            self.failed.emit(str(e))

    def _analyze(self, y):
        from ..core.pipeline import analyze_signal

        return analyze_signal(
            y,
            audio_io.SR,
            source=self.path,
            opts=self.opts,
            progress=lambda f, m: self.progress.emit(f, m),
        )


class LiveThread(QThread):
    """Microphone capture + real-time pitch tracking.

    The capture itself (device stream, block queue, worker thread) lives in
    `core.audio_io.LiveCapture`, which keeps YIN out of the audio callback.
    This thread only owns the segmenter and the finished signal.
    """

    blockStats = pyqtSignal(float, float)
    pitchFrame = pyqtSignal(float, float)
    recordingFinished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, min_conf: float = 0.5):
        super().__init__()
        self.min_conf = min_conf
        self._stop = False
        self.capture: LiveCapture | None = None
        self.seg: LiveNoteSegmenter | None = None
        self.stream: LivePitchStream | None = None

    def run(self) -> None:
        self.stream = None
        self.seg = None
        self.capture = LiveCapture(on_block=self._process_block, sr=audio_io.SR)
        try:
            self.capture.start()
        except Exception as e:
            self.failed.emit(tr("err ui mic").format(e=e))
            return

        while not self._stop:
            self.msleep(50)
        self.capture.stop()
        self.recordingFinished.emit(self.capture.signal())

    def _process_block(self, block: np.ndarray, t: float) -> None:
        """Pitch-track one captured block and feed the live chain."""
        if self.stream is None:
            self.stream = LivePitchStream(self.capture.samplerate, gate_rms=audio_io.GATE_RMS)
            self.seg = self.stream.seg
        closed = self.stream.elapsed_notes()
        rms, frames = self.stream.feed(block, t)
        self.blockStats.emit(t, rms)
        for ft, hz in frames:
            self.pitchFrame.emit(ft, hz)
        if self.seg is not None and self.stream.elapsed_notes() > closed:
            # Label figures at note END (approximate BPM from the note
            # onsets so far) so the closed note shows its figure as
            # soon as it lands on the rolling piano roll.
            bpm = estimate_bpm_live(self.seg.notes)
            if bpm:
                label_notes(self.seg.notes, BeatGrid(bpm))

    def stop(self) -> None:
        self._stop = True

    def elapsed(self) -> float:
        return self.capture.elapsed() if self.capture is not None else 0.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        get_language()  # restore the persisted choice before building the UI
        self.setWindowTitle(tr("ui window title"))
        self.resize(1280, 780)
        self.setAcceptDrops(True)

        self.result = None            # last AnalysisResult
        self.analyze_thread: AnalyzeThread | None = None
        self.live_thread: LiveThread | None = None
        self.finder_thread: LiveThread | None = None   # note finder (no take kept)
        self.source_path: str | None = None
        self._signal_cache: dict[str, np.ndarray] = {}  # source -> decoded audio
        self._live_signal: np.ndarray | None = None
        self._live_pitch_t: list[float] = []
        self._live_pitch_m: list[float] = []
        self._live_curve = None
        self._live_t = 0.0          # capture time of the last audio block
        self.player: NotePlayer | None = None
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

        self.page_home = self._build_home_page()
        self.page_analyze = self._build_analyze_page()
        self.page_live = self._build_live_page()
        self.page_finder = self._build_finder_page()
        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_analyze)
        self.stack.addWidget(self.page_live)
        self.stack.addWidget(self.page_finder)
        self.retranslate()

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    def _build_home_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch(2)
        self._title = QLabel("<h1>🎤 Voix → MIDI</h1>")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._title)
        self._subtitle = QLabel("")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._subtitle)
        lay.addSpacing(22)

        # first-run guidance: three numbered steps, always visible
        self._steps = QLabel("")
        self._steps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._steps.setWordWrap(True)
        self._steps.setMaximumWidth(720)
        self._steps.setStyleSheet(HINT_CARD_STYLE)
        steps_lay = QHBoxLayout()
        steps_lay.addStretch(1)
        steps_lay.addWidget(self._steps)
        steps_lay.addStretch(1)
        lay.addLayout(steps_lay)
        lay.addSpacing(18)

        self.btn_file = QPushButton("")
        self.btn_file.setMinimumHeight(54)
        self.btn_file.setStyleSheet(action_button_style(primary=True))
        self.btn_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_file.clicked.connect(self.on_open_file)
        lay.addWidget(self.btn_file)

        self.btn_live = QPushButton("")
        self.btn_live.setMinimumHeight(54)
        self.btn_live.setStyleSheet(action_button_style(primary=False))
        self.btn_live.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_live.clicked.connect(self.on_start_live)
        lay.addWidget(self.btn_live)

        # note finder: the only job of this page is "which note is this?"
        self.btn_finder = QPushButton("")
        self.btn_finder.setMinimumHeight(54)
        self.btn_finder.setStyleSheet(action_button_style(primary=False))
        self.btn_finder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_finder.clicked.connect(self.on_open_finder)
        lay.addWidget(self.btn_finder)

        # language selector: click to toggle fr <-> en, persisted on disk
        self.btn_lang = QPushButton("")
        self.btn_lang.setMinimumHeight(40)
        self.btn_lang.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_lang.clicked.connect(self.on_toggle_language)
        lang_lay = QHBoxLayout()
        lang_lay.addStretch(1)
        lang_lay.addWidget(self.btn_lang)
        lang_lay.addStretch(1)
        lay.addLayout(lang_lay)

        self._hint = QLabel("")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setStyleSheet("color: #7f8b99;")
        lay.addWidget(self._hint)
        lay.addStretch(3)
        return w

    def _build_analyze_page(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)

        # top bar
        top = QHBoxLayout()
        self.btn_back = QPushButton("")
        self.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_home))
        top.addWidget(self.btn_back)
        self.label_file = QLabel("")
        self.label_file.setStyleSheet("font-weight: bold;")
        top.addWidget(self.label_file, stretch=1)
        root.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        # center: views + side panel
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.waveform = WaveformView()
        self.piano = PianoRoll()
        self.piano.seek_requested.connect(self.on_seek)
        self.staff = StaffView()
        left.addWidget(self.waveform, stretch=1)
        left.addWidget(self.staff, stretch=2)   # the score itself
        left.addWidget(self.piano, stretch=2)

        # playback bar: play/stop the analysis + live playhead position
        pb = QHBoxLayout()
        self.btn_play = QPushButton("")
        self.btn_play.setMinimumWidth(64)
        self.btn_play.clicked.connect(self.on_toggle_play)
        self.label_play_pos = QLabel("")
        pb.addWidget(self.btn_play)
        pb.addWidget(self.label_play_pos, stretch=1)
        left.addLayout(pb)
        # keyboard view: detected notes light their keys while playing
        self.piano_keys = PianoView()
        left.addWidget(self.piano_keys)
        body.addLayout(left, stretch=3)

        right = QVBoxLayout()
        self.results = ResultsPanel()
        right.addWidget(self.results, stretch=1)

        opts = QGroupBox("")
        form = QGridLayout(opts)
        self._grp_opts = opts

        self.chk_quantize = QCheckBox("")
        form.addWidget(self.chk_quantize, 0, 0, 1, 2)
        self._lbl_transpose = QLabel("")
        form.addWidget(self._lbl_transpose, 1, 0)
        self.spin_transpose = QSpinBox()
        self.spin_transpose.setRange(-48, 48)
        form.addWidget(self.spin_transpose, 1, 1)
        self.chk_notation = QCheckBox("")
        self.chk_notation.toggled.connect(self._on_notation_toggled)
        form.addWidget(self.chk_notation, 2, 0, 1, 2)
        self.chk_pdf = QCheckBox("")
        self.chk_pdf.setEnabled(find_notation_tool() is not None)
        form.addWidget(self.chk_pdf, 3, 0, 1, 2)
        # badge explaining WHY the PDF option is off (first-run clarity)
        self.lbl_pdf_note = QLabel("")
        self.lbl_pdf_note.setStyleSheet("color: #7f8b99; font-size: 8pt; font-style: italic;")
        self.lbl_pdf_note.setWordWrap(True)
        form.addWidget(self.lbl_pdf_note, 4, 0, 1, 2)
        right.addWidget(opts)

        self.btn_export = QPushButton("")
        self.btn_export.clicked.connect(self.on_export_again)
        right.addWidget(self.btn_export)

        self.btn_open = QPushButton("")
        self.btn_open.clicked.connect(self.on_open_file)
        right.addWidget(self.btn_open)
        body.addLayout(right, stretch=1)
        root.addLayout(body, stretch=1)
        return w

    def _build_live_page(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        top = QHBoxLayout()
        self.btn_stop_live = QPushButton("")
        self.btn_stop_live.clicked.connect(self.on_stop_live)
        top.addWidget(self.btn_stop_live)
        self.label_live_status = QLabel("")
        top.addWidget(self.label_live_status, stretch=1)
        root.addLayout(top)

        mid = QHBoxLayout()
        self.live_indicator = LiveIndicator()
        mid.addWidget(self.live_indicator, stretch=1)
        self.piano_live = PianoRoll()
        mid.addWidget(self.piano_live, stretch=3)
        root.addLayout(mid, stretch=2)
        # the score as it is being sung, scrolling under the staff
        self.staff_live = StaffView(live=True)
        root.addWidget(self.staff_live, stretch=2)
        # recognition read as keys, not as numbers: the keys of the note
        # being sung light up here, under the same keyboard as the results
        self.piano_live_keys = PianoView()
        root.addWidget(self.piano_live_keys)
        return w

    def _build_finder_page(self) -> QWidget:
        """Note finder: the live note, on a piano keyboard, and nothing else.

        No staff, no roll, no analysis, no export — the take is not kept.
        """
        w = QWidget()
        root = QVBoxLayout(w)
        top = QHBoxLayout()
        self.btn_finder_back = QPushButton("")
        self.btn_finder_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_finder_back.clicked.connect(self.on_close_finder)
        top.addWidget(self.btn_finder_back)
        self.btn_finder_toggle = QPushButton("")
        self.btn_finder_toggle.setMinimumHeight(38)
        self.btn_finder_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_finder_toggle.clicked.connect(self.on_toggle_finder)
        top.addWidget(self.btn_finder_toggle)
        # the button says what a click DOES; this line says what the app IS
        # doing — the mic is open, or it is not (an empty page must explain
        # itself rather than look broken)
        self.label_finder_state = QLabel("")
        self.label_finder_state.setStyleSheet("color: #8a97a5; font-size: 9pt;")
        top.addWidget(self.label_finder_state)
        top.addStretch(1)
        root.addLayout(top)

        self.label_finder_note = QLabel("")
        self.label_finder_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_finder_note.setMinimumHeight(66)  # no jump when it appears
        root.addWidget(self.label_finder_note)
        # first-run guidance, in the same card as the home steps
        self.label_finder_hint = QLabel("")
        self.label_finder_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_finder_hint.setWordWrap(True)
        self.label_finder_hint.setMaximumWidth(560)
        self.label_finder_hint.setStyleSheet(HINT_CARD_STYLE)
        hint_lay = QHBoxLayout()
        hint_lay.addStretch(1)
        hint_lay.addWidget(self.label_finder_hint)
        hint_lay.addStretch(1)
        root.addLayout(hint_lay)
        root.addStretch(1)

        # the whole classical piano: the keys ARE the answer, and nothing
        # outside A0..C8 can ever light up
        self.finder_keys = PianoView(min_midi=PIANO_MIN_MIDI, max_midi=PIANO_MAX_MIDI)
        self.finder_keys.setMinimumHeight(150)
        root.addWidget(self.finder_keys, stretch=2)
        return w

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
        self.label_play_pos.setText(tr("ui play pos"))
        self.btn_stop_live.setText(tr("ui stop live"))
        self.btn_finder.setText(tr("ui finder button"))
        self.btn_finder_back.setText(tr("ui back home"))
        self.label_finder_hint.setText(tr("ui finder hint"))
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

    # ------------------------------------------------------------------
    # Playback (ear-check of the analysis, synced playhead)
    # ------------------------------------------------------------------
    def on_toggle_play(self) -> None:
        if self.result is None:
            return
        if self.player is not None and self.player.playing:
            self.stop_playback()
            return
        if self.player is None or self.player.signal is None:
            return
        self.btn_play.setText("■")
        self.player.play()
        self._play_timer.start()

    def on_seek(self, t: float) -> None:
        """Double-click on the piano roll: play from there (or just move)."""
        if self.result is None or self.player is None:
            return
        self.player.stop()
        self._play_timer.stop()
        self.piano.set_playhead(t, follow=False)
        self.waveform.set_position(t)
        if t < self.player.duration - 0.05:
            self.player.play_from(t)
            self.btn_play.setText("■")
            self._play_timer.start()

    def _on_play_tick(self) -> None:
        if self.player is None:
            self._play_timer.stop()
            return
        t = self.player.position()
        self.piano.set_playhead(t)
        self.waveform.set_position(t)
        self.piano_keys.set_playhead(t)
        self.label_play_pos.setText(
            f"{int(t // 60):02d}:{int(t % 60):02d} / {int(self.player.duration // 60):02d}:{int(self.player.duration % 60):02d}"
        )
        if not self.player.playing:
            self._play_timer.stop()
            self.btn_play.setText("▶")

    def stop_playback(self) -> None:
        self._play_timer.stop()
        if self.player is not None:
            self.player.stop()
        self.btn_play.setText("▶")
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
        opts = AnalyzeOptions(min_conf=0.6)
        self.analyze_thread = AnalyzeThread(path, opts)
        self.analyze_thread.progress.connect(self._on_progress)
        self.analyze_thread.finishedOk.connect(self._on_analysis_done)
        self.analyze_thread.failed.connect(self._on_error)
        self.analyze_thread.start()

    def _on_progress(self, f: float, msg: str) -> None:
        self.progress.setValue(int(f * 100))
        self.label_file.setText(f"{Path(self.source_path).name} — {msg}")

    def _on_analysis_done(self, res, signal_bytes: bytes) -> None:
        self.result = res
        # keep the decoded signal so re-display never re-decodes the file
        self._signal_cache = {res.source: np.frombuffer(signal_bytes, dtype=np.float32)}
        self._live_signal = None  # a file view must never show a live take
        self.stop_playback()
        self.player = NotePlayer(res.notes, sr=audio_io.SR)
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
        src = Path(res.source if res.source != "live" else f"live_{time.strftime('%Y%m%d_%H%M%S')}")
        stem = src.with_suffix("") if src.suffix else src
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
        self.label_live_status.setText(tr("ui live progress").format(t="00:00", notes=0))
        # live pitch curve on the scrolling piano roll (drop the previous take's)
        self.piano_live.begin_live()
        self.staff_live.clear()
        if self._live_curve is not None:
            self.piano_live.removeItem(self._live_curve)
        self._live_curve = self.piano_live.plot(
            [], [], pen=None, symbol="o", symbolSize=3, symbolBrush=(120, 200, 255, 180)
        )
        self.live_thread = LiveThread(min_conf=0.5)
        self.live_thread.blockStats.connect(self._on_live_block)
        self.live_thread.pitchFrame.connect(self._on_live_pitch)
        self.live_thread.recordingFinished.connect(self._on_live_finished)
        self.live_thread.failed.connect(self._on_live_failed)
        self.live_thread.start()
        self._live_timer.start()

    def _on_live_block(self, t: float, rms: float) -> None:
        """Telemetry from each captured block: capture time + input level."""
        self._live_t = t
        self.live_indicator.update_rms(rms)

    def _on_live_tick(self) -> None:
        """Live follow (~20 fps): render the take, scroll, move the cursor.

        Single owner of the live rendering: the audio thread only publishes
        notes, so the roll stays in sync even if a block or an event is late.
        """
        thr = self.live_thread
        if thr is None or thr.seg is None:
            return
        t = self._live_t or thr.elapsed()
        notes = thr.seg.notes
        preview = thr.seg.active_note(t)
        self.piano_live.set_notation_mode(True)
        self.piano_live.set_notes(notes)
        self.piano_live.set_live_preview(preview)
        midis = [n.midi for n in notes[-64:]]
        if preview is not None:
            midis.append(preview.midi)
        self.piano_live.set_live_pitch_range(midis)
        self.piano_live.set_live_cursor(t)
        self.staff_live.set_live(
            notes,
            preview,
            grid=BeatGrid(estimate_bpm_live(notes)),
        )
        self.label_live_status.setText(
            tr("ui live progress").format(t=clock(t), notes=len(notes))
        )

    def _on_live_pitch(self, t: float, hz: float) -> None:
        cur = self.live_thread.seg.current() if self.live_thread and self.live_thread.seg else None
        self.live_indicator.update_note(cur[0] if cur else None, hz)
        # the recognized note read as a KEY, not as a number
        midi = int(round(cur[0])) if cur else None
        self.piano_live_keys.set_pressed({midi} if midi is not None and on_piano(midi) else ())
        if hz == hz and hz > 0:  # NaN check
            self._live_pitch_t.append(t)
            self._live_pitch_m.append(69 + 12 * np.log2(hz / 440.0))
            # keep the pitch cloud inside the scrolling window
            while self._live_pitch_t and t - self._live_pitch_t[0] > self.piano_live.LIVE_SPAN:
                self._live_pitch_t.pop(0)
                self._live_pitch_m.pop(0)
            self._live_curve.setData(self._live_pitch_t, self._live_pitch_m)

    def _on_live_failed(self, msg: str) -> None:
        self._live_timer.stop()
        self.label_live_status.setText(tr("ui mic preparing"))
        self._on_error(msg)

    def _on_live_finished(self, y: np.ndarray) -> None:
        """Final analysis of the live recording."""
        self._live_timer.stop()
        self.piano_live.set_live_preview(None)
        self.label_live_status.setText(tr("ui analyzing"))
        self._live_signal = y
        if len(y) < audio_io.SR // 2:
            self.show_error(tr("err ui too short"))
            self.stack.setCurrentWidget(self.page_home)
            return

        class _FinalThread(QThread):
            finishedOk = pyqtSignal(object)
            failed = pyqtSignal(str)

            def run(inner_self) -> None:
                try:
                    from ..core.pipeline import analyze_signal

                    res = analyze_signal(
                        y,
                        audio_io.SR,
                        source="live",
                        opts=AnalyzeOptions(min_conf=0.5),
                        progress=lambda f, m: None,
                    )
                    inner_self.finishedOk.emit(res)
                except Exception as e:
                    inner_self.failed.emit(str(e))

        self._final_thread = _FinalThread()
        self._final_thread.finishedOk.connect(self._on_live_analysis_done)
        self._final_thread.failed.connect(self._on_error)
        self._final_thread.start()

    def _on_live_analysis_done(self, res) -> None:
        self.result = res
        self._signal_cache = {}  # live takes over; drop stale file caches
        self.source_path = "live"
        # Export FIRST: the take is the user's data, a cosmetic hiccup below
        # must never cost it (it used to silently skip the export entirely).
        self._export_results()
        self.stop_playback()
        self.player = NotePlayer(res.notes, sr=audio_io.SR)
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
            grid=res.grid or BeatGrid(res.tempo.bpm if res.tempo else None),
            key=getattr(res.key, "best", None),
            positions=getattr(res, "note_positions", None),
            rests=getattr(res, "rests", None),
        )

    def on_stop_live(self) -> None:
        self._live_timer.stop()
        self.piano_live.set_live_preview(None)
        self.piano_live_keys.set_pressed()
        self.staff_live.clear()
        if self.live_thread is not None and self.live_thread.isRunning():
            self.label_live_status.setText(tr("ui stopping"))
            self.live_thread.stop()
            # the capture thread also drains its queued blocks: give it room
            self.live_thread.wait(6000)
            self.live_thread = None
        else:
            self.stack.setCurrentWidget(self.page_home)

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
        self.finder_thread = LiveThread(min_conf=0.5)
        self.finder_thread.pitchFrame.connect(self._on_finder_pitch)
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
        if self._finder_running():
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

    def _on_finder_pitch(self, t: float, hz: float) -> None:
        """Pitch frame -> the key to light (the segmenter's median if it has one)."""
        seg = self.finder_thread.seg if self.finder_thread is not None else None
        cur = seg.current() if seg is not None else None
        if cur is not None:
            self._set_finder_note(cur[0])
        elif hz == hz and hz > 0:
            self._set_finder_note(float(hz_to_midi(hz)))

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
