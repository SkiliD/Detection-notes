"""Workers Qt : analyse en arrière-plan, capture temps réel."""

from __future__ import annotations

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal

from ..core.audio_io import SR
from ..core.liveseg import LiveNoteSegmenter
from ..core.pipeline import AnalyzeOptions, analyze_signal
from ..core.pitch import YinBlock

GATE_RMS = 0.008


class AnalyzeWorker(QObject):
    """Analyse d'un fichier audio dans un thread séparé."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)   # AnalysisResult
    error = pyqtSignal(str)

    def __init__(self, path: str, opts: AnalyzeOptions | None = None):
        super().__init__()
        self.path = path
        self.opts = opts or AnalyzeOptions()

    def run(self) -> None:
        try:
            from ..core import audio_io

            y = audio_io.load_file(self.path)
            res = analyze_signal(
                y,
                SR,
                source=self.path,
                opts=self.opts,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.finished.emit(res)
        except Exception as e:  # pragma: no cover
            self.error.emit(str(e))


class LiveWorker(QObject):
    """Capture micro + pitch temps réel, relais vers la GUI.

    Les blocs arrivent via le callback sounddevice (thread audio) ; chaque
    frame de pitch est relayée par un signal Qt (thread-safe). L'analyse
    finale (tempo, tonalité, notes) est déclenchée à l'arrêt.
    """

    blockStats = pyqtSignal(float, float)       # (temps, rms)
    pitchFrame = pyqtSignal(float, float)       # (temps, hz) — NaN si non voisé
    noteOn = pyqtSignal(float, int)             # (temps, midi)
    noteOff = pyqtSignal(float, int)            # (temps, midi)
    finishedRecording = pyqtSignal(object)      # np.ndarray (signal complet, SR)
    error = pyqtSignal(str)

    def __init__(self, samplerate: int = 44_100, min_conf: float = 0.5):
        super().__init__()
        self.samplerate = samplerate
        self.min_conf = min_conf
        self.seg = LiveNoteSegmenter()
        self.yin: YinBlock | None = None
        self._chunks: list[np.ndarray] = []
        self._stream = None

    # ---------------- capture ----------------
    def start(self) -> None:
        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                blocksize=1024,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self.error.emit(f"Micro indisponible : {e}")

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        y = self._concatenate()
        self.finishedRecording.emit(y)
        return y

    def elapsed(self) -> float:
        return sum(len(c) for c in self._chunks) / self.samplerate

    def _concatenate(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        y = np.concatenate(self._chunks).astype(np.float32)
        if self.samplerate == SR:
            return y
        ratio = SR / self.samplerate
        n_out = int(len(y) * ratio)
        idx = np.linspace(0, len(y) - 1, n_out)
        return np.interp(idx, np.arange(len(y)), y).astype(np.float32)

    # ---------------- callback audio (thread sounddevice) ----------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[micro] {status}")
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        self._chunks.append(mono.copy())
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        t_block = self.elapsed()
        self.blockStats.emit(t_block, rms)

        if self.yin is None:
            # hop 1024 : ~43 estimations/s, léger dans le callback audio
            self.yin = YinBlock(self.samplerate, window=2048, hop=1024)
        frames_out = self.yin.push(mono)
        for ft, hz in frames_out:
            if rms < GATE_RMS:
                hz = float("nan")
            ev = self.seg.push(ft, hz)
            self.pitchFrame.emit(ft, hz)
            if ev == "on" and self.seg.notes:
                self.noteOn.emit(ft, self.seg.notes[-1].midi)
            elif ev == "off" and self.seg.notes:
                self.noteOff.emit(ft, self.seg.notes[-1].midi)

    # ---------------- analyse finale ----------------
    def analyze_recording(self, y: np.ndarray, progress=None):
        """Analyse complète du signal enregistré (appeler hors thread audio)."""
        res = analyze_signal(
            y,
            SR,
            source="live",
            opts=AnalyzeOptions(min_conf=self.min_conf),
            progress=progress,
        )
        return res

    def live_notes(self):
        """Notes détectées en direct (pour pré-affichage avant analyse)."""
        return list(self.seg.notes)
