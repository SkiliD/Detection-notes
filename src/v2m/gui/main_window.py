"""Fenêtre principale de l'application Voix -> MIDI."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import audio_io
from ..core.pipeline import AnalyzeOptions, export_result
from .widgets import LiveIndicator, PianoRoll, ResultsPanel, WaveformView

AUDIO_FILTER = "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;Tous les fichiers (*)"


class AnalyzeThread(QThread):
    """Analyse d'un fichier dans un thread."""

    progress = pyqtSignal(float, str)
    finishedOk = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: str, opts: AnalyzeOptions):
        super().__init__()
        self.path = path
        self.opts = opts

    def run(self) -> None:
        try:
            y = audio_io.load_file(self.path)
            res = self._analyze(y)
            self.finishedOk.emit(res)
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
    """Capture micro + pitch temps réel."""

    blockStats = pyqtSignal(float, float)
    pitchFrame = pyqtSignal(float, float)
    noteOn = pyqtSignal(float, int)
    noteOff = pyqtSignal(float, int)
    recordingFinished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, min_conf: float = 0.5):
        super().__init__()
        self.min_conf = min_conf
        self._stop = False
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self.seg = None
        self.yin = None

    def run(self) -> None:
        import sounddevice as sd

        from ..core.liveseg import LiveNoteSegmenter
        from ..core.pitch import YinBlock

        self.seg = LiveNoteSegmenter()
        self.yin = None
        samplerate = 44_100

        def callback(indata, frames, t_info, status):  # noqa: ANN001
            if status:
                print(f"[micro] {status}")
            mono = np.asarray(indata[:, 0], dtype=np.float32)
            self._chunks.append(mono.copy())
            rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
            t_block = sum(len(c) for c in self._chunks) / samplerate
            self.blockStats.emit(t_block, rms)
            if self.yin is None:
                self.yin = YinBlock(samplerate, window=2048, hop=1024)
            for ft, hz in self.yin.push(mono):
                if rms < 0.008:
                    hz = float("nan")
                ev = self.seg.push(ft, hz)
                self.pitchFrame.emit(ft, hz)
                if ev == "on" and self.seg.notes:
                    self.noteOn.emit(ft, self.seg.notes[-1].midi)
                elif ev == "off" and self.seg.notes:
                    self.noteOff.emit(ft, self.seg.notes[-1].midi)

        try:
            self._stream = sd.InputStream(
                samplerate=samplerate, channels=1, blocksize=1024, callback=callback
            )
            self._stream.start()
        except Exception as e:
            self.failed.emit(f"Micro indisponible : {e}")
            return

        while not self._stop:
            self.msleep(50)
        self._stream.stop()
        self._stream.close()

        y = self._signal()
        self.recordingFinished.emit(y)

    def stop(self) -> None:
        self._stop = True

    def elapsed(self) -> float:
        return sum(len(c) for c in self._chunks) / 44_100

    def _signal(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        y = np.concatenate(self._chunks).astype(np.float32)
        if 44_100 == audio_io.SR:
            return y
        n_out = int(len(y) * audio_io.SR / 44_100)
        idx = np.linspace(0, len(y) - 1, n_out)
        return np.interp(idx, np.arange(len(y)), y).astype(np.float32)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voix → MIDI — détection de notes, tempo et tonalité")
        self.resize(1280, 780)
        self.setAcceptDrops(True)

        self.result = None            # dernière AnalysisResult
        self.analyze_thread: AnalyzeThread | None = None
        self.live_thread: LiveThread | None = None
        self.source_path: str | None = None
        self._live_pitch_t: list[float] = []
        self._live_pitch_m: list[float] = []

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.page_home = self._build_home_page()
        self.page_analyze = self._build_analyze_page()
        self.page_live = self._build_live_page()
        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_analyze)
        self.stack.addWidget(self.page_live)

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    def _build_home_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch(1)
        title = QLabel("<h1>🎤 Voix → MIDI</h1>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        sub = QLabel("Détecte les notes, le tempo et la tonalité de votre voix,\npuis exporte un fichier MIDI + un rapport JSON/CSV.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        lay.addSpacing(30)

        btn_file = QPushButton("📁  Analyser un fichier audio")
        btn_file.setMinimumHeight(52)
        btn_file.setStyleSheet("font-size: 16px;")
        btn_file.clicked.connect(self.on_open_file)
        lay.addWidget(btn_file)

        btn_live = QPushButton("🎤  Chanter au micro (temps réel)")
        btn_live.setMinimumHeight(52)
        btn_live.setStyleSheet("font-size: 16px;")
        btn_live.clicked.connect(self.on_start_live)
        lay.addWidget(btn_live)

        hint = QLabel("Astuce : glissez-déposez un fichier audio dans cette fenêtre.")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #888;")
        lay.addWidget(hint)
        lay.addStretch(2)
        return w

    def _build_analyze_page(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)

        # barre du haut
        top = QHBoxLayout()
        self.btn_back = QPushButton("← Accueil")
        self.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_home))
        top.addWidget(self.btn_back)
        self.label_file = QLabel("Aucun fichier")
        self.label_file.setStyleSheet("font-weight: bold;")
        top.addWidget(self.label_file, stretch=1)
        root.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        # zone centrale : vues + panneau
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.waveform = WaveformView()
        self.piano = PianoRoll()
        left.addWidget(self.waveform, stretch=1)
        left.addWidget(self.piano, stretch=2)
        body.addLayout(left, stretch=3)

        right = QVBoxLayout()
        self.results = ResultsPanel()
        right.addWidget(self.results, stretch=1)

        opts = QGroupBox("Options d'export")
        form = QGridLayout(opts)
        self.chk_quantize = QCheckBox("Quantifier sur la grille 1/16")
        form.addWidget(self.chk_quantize, 0, 0, 1, 2)
        form.addWidget(QLabel("Transposition (demi-tons)"), 1, 0)
        self.spin_transpose = QSpinBox()
        self.spin_transpose.setRange(-48, 48)
        form.addWidget(self.spin_transpose, 1, 1)
        self.chk_notation = QCheckBox("Mode notation (figures colorées)")
        self.chk_notation.toggled.connect(self._on_notation_toggled)
        form.addWidget(self.chk_notation, 2, 0, 1, 2)
        right.addWidget(opts)

        btn_export = QPushButton("💾  Ré-exporter avec ces options")
        btn_export.clicked.connect(self.on_export_again)
        right.addWidget(btn_export)

        btn_open = QPushButton("📁  Ouvrir un autre fichier")
        btn_open.clicked.connect(self.on_open_file)
        right.addWidget(btn_open)
        body.addLayout(right, stretch=1)
        root.addLayout(body, stretch=1)
        return w

    def _build_live_page(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        top = QHBoxLayout()
        btn_back = QPushButton("← Arrêter / Accueil")
        btn_back.clicked.connect(self.on_stop_live)
        top.addWidget(btn_back)
        self.label_live_status = QLabel("Préparation du micro…")
        top.addWidget(self.label_live_status, stretch=1)
        root.addLayout(top)

        mid = QHBoxLayout()
        self.live_indicator = LiveIndicator()
        mid.addWidget(self.live_indicator, stretch=1)
        self.piano_live = PianoRoll()
        mid.addWidget(self.piano_live, stretch=3)
        root.addLayout(mid, stretch=1)
        return w

    # ------------------------------------------------------------------
    # Fichier
    # ------------------------------------------------------------------
    def on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choisir un fichier audio", "", AUDIO_FILTER)
        if not path:
            return
        self.start_analysis(path)

    def start_analysis(self, path: str) -> None:
        self.stack.setCurrentWidget(self.page_analyze)
        self.source_path = path
        self.label_file.setText(Path(path).name)
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

    def _on_analysis_done(self, res) -> None:
        self.result = res
        self.label_file.setText(Path(res.source).name)
        self.results.set_results(res)
        self.waveform.set_signal(audio_io.load_file(res.source), audio_io.SR)
        self.piano.set_notation_mode(self.chk_notation.isChecked())
        self.piano.set_notes(res.notes)
        self.piano.set_pitch_curve(res.pitch)
        self._export_results()

    def _on_notation_toggled(self, checked: bool) -> None:
        self.piano.set_notation_mode(checked)

    def _on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Erreur", msg)
        self.stack.setCurrentWidget(self.page_home)

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
        export_result(
            res,
            stem,
            quantize_s=quantize_s,
            transpose=self.spin_transpose.value(),
        )
        self.statusBar().showMessage(
            f"Exporté : {stem.name}.mid, {stem.name}.musicxml, {stem.name}.json, {stem.name}.csv",
            10_000,
        )

    def on_export_again(self) -> None:
        self._export_results()

    # ------------------------------------------------------------------
    # Live
    # ------------------------------------------------------------------
    def on_start_live(self) -> None:
        self.stack.setCurrentWidget(self.page_live)
        self.label_live_status.setText("🎤 Enregistrement… (chantez !)")
        self._live_pitch_t = []
        self._live_pitch_m = []
        # courbe live sur le piano roll
        self._live_curve = self.piano_live.plot(
            [], [], pen=None, symbol="o", symbolSize=3, symbolBrush=(120, 200, 255, 180)
        )
        self.live_thread = LiveThread(min_conf=0.5)
        self.live_thread.blockStats.connect(self.live_indicator.update_rms)
        self.live_thread.pitchFrame.connect(self._on_live_pitch)
        self.live_thread.noteOn.connect(self._on_live_note_on)
        self.live_thread.noteOff.connect(self._on_live_note_off)
        self.live_thread.recordingFinished.connect(self._on_live_finished)
        self.live_thread.failed.connect(self._on_error)
        self.live_thread.start()

    def _on_live_pitch(self, t: float, hz: float) -> None:
        cur = self.live_thread.seg.current() if self.live_thread and self.live_thread.seg else None
        self.live_indicator.update_note(cur[0] if cur else None, hz)
        if hz == hz and hz > 0:
            self._live_pitch_t.append(t)
            self._live_pitch_m.append(69 + 12 * np.log2(hz / 440.0))
            # garde ~15 s de fenêtre
            while self._live_pitch_t and t - self._live_pitch_t[0] > 15.0:
                self._live_pitch_t.pop(0)
                self._live_pitch_m.pop(0)
            self._live_curve.setData(self._live_pitch_t, self._live_pitch_m)

    def _on_live_note_on(self, t: float, midi: int) -> None:
        self.piano_live.set_notes(self.live_thread.seg.notes)

    def _on_live_note_off(self, t: float, midi: int) -> None:
        self.piano_live.set_notes(self.live_thread.seg.notes)

    def _on_live_finished(self, y: np.ndarray) -> None:
        """Analyse finale de l'enregistrement live."""
        self.label_live_status.setText("Analyse finale…")
        if len(y) < audio_io.SR // 2:
            QMessageBox.information(self, "Enregistrement court", "Enregistrement trop court pour être analysé.")
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
        self.source_path = "live"
        self.stack.setCurrentWidget(self.page_analyze)
        self.label_file.setText("Enregistrement live")
        self.progress.setValue(100)
        self.results.set_results(res)
        self.piano.set_notes(res.notes)
        self.piano.set_pitch_curve(res.pitch)
        self._export_results()
        QMessageBox.information(
            self,
            "Analyse terminée",
            f"Tempo : {res.tempo.bpm} BPM\nTonalité : {res.key.best.label}\nNotes : {len(res.notes)}\n\nFichiers exportés dans le dossier courant.",
        )

    def on_stop_live(self) -> None:
        if self.live_thread is not None and self.live_thread.isRunning():
            self.label_live_status.setText("Arrêt…")
            self.live_thread.stop()
            self.live_thread.wait(3000)
            self.live_thread = None
        else:
            self.stack.setCurrentWidget(self.page_home)

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if urls and any(u.toLocalFile().lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")) for u in urls):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")):
                self.start_analysis(path)
                break


def run_gui() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Voix → MIDI")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
