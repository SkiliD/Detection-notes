"""Background analysis workers for the GUI.

The analysis workers all yield CPU to the live capture and publish failures
through the same Qt signal. Their success payloads and pipeline options remain
specific to each operation, so those contracts stay in the concrete classes.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..core.pipeline import AnalyzeOptions
from .priority import lower_background_priority


class BackgroundAnalysisThread(QThread):
    """Run one background analysis with shared priority and error handling."""

    failed = pyqtSignal(str)

    def run(self) -> None:
        """Lower this worker's priority, then convert analysis errors to a signal."""
        lower_background_priority()
        try:
            self._run_analysis()
        except Exception as error:
            self.failed.emit(str(error))

    def _run_analysis(self) -> None:
        """Perform the specialization's pipeline call and emit its result."""
        raise NotImplementedError


class AnalysisThread(BackgroundAnalysisThread):
    """Run one file analysis and publish its complete outcome."""

    progress = pyqtSignal(float, str)
    finishedOk = pyqtSignal(object)

    def __init__(self, source: str, opts: AnalyzeOptions):
        super().__init__()
        self.source = source
        self.opts = opts

    def _run_analysis(self) -> None:
        from ..core.pipeline import analyze_file

        outcome = analyze_file(
            self.source,
            opts=self.opts,
            progress=lambda fraction, message: self.progress.emit(fraction, message),
        )
        self.finishedOk.emit(outcome)


class SignalAnalysisThread(BackgroundAnalysisThread):
    """Analyze an edited in-memory take without blocking the GUI thread."""

    finishedOk = pyqtSignal(object, object)

    def __init__(self, signal: np.ndarray, sr: int, source: str):
        super().__init__()
        self.signal = np.asarray(signal, dtype=np.float32)
        self.sr = sr
        self.source = source

    def _run_analysis(self) -> None:
        from ..core.pipeline import LIVE_CONF, analyze_signal
        from ..core.pipeline import AnalyzeOptions as PipelineOptions

        result = analyze_signal(
            self.signal,
            self.sr,
            source=self.source,
            opts=PipelineOptions(min_conf=LIVE_CONF),
        )
        self.finishedOk.emit(result, self.signal)


class FullReanalysisThread(BackgroundAnalysisThread):
    """Rebuild vocal evidence from the retained audio, then interpret it."""

    progress = pyqtSignal(float, str)
    finishedOk = pyqtSignal(object)

    def __init__(self, signal: np.ndarray, sr: int, source: str, opts: AnalyzeOptions):
        super().__init__()
        self.signal = np.asarray(signal, dtype=np.float32).copy()
        self.sr = sr
        self.source = source
        self.opts = opts

    def _run_analysis(self) -> None:
        from ..core.pipeline import analyze_signal

        result = analyze_signal(
            self.signal,
            self.sr,
            source=self.source,
            opts=self.opts,
            progress=lambda fraction, message: self.progress.emit(fraction, message),
        )
        self.finishedOk.emit(result)
