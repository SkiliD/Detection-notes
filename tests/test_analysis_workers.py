"""Shared lifecycle and public signal contracts for GUI analysis workers."""

from __future__ import annotations

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QApplication

from v2m.core.pipeline import LIVE_CONF, AnalyzeOptions
from v2m.gui import analysis_workers as workers


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class ProbeWorker(workers.BackgroundAnalysisThread):
    """Expose the shared run method without invoking the audio pipeline."""

    completed = pyqtSignal(str)

    def __init__(self, failure: Exception | None = None):
        super().__init__()
        self.failure = failure
        self.calls = 0

    def _run_analysis(self) -> None:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        self.completed.emit("done")


def test_shared_worker_lowers_priority_and_emits_success(qapp, monkeypatch):
    priority_threads = []
    monkeypatch.setattr(
        workers,
        "lower_background_priority",
        lambda: priority_threads.append(threading.get_ident()),
    )
    worker = ProbeWorker()
    analysis_threads = []
    original_run_analysis = worker._run_analysis

    def record_analysis_thread():
        analysis_threads.append(threading.get_ident())
        original_run_analysis()

    worker._run_analysis = record_analysis_thread
    success = []
    failures = []
    worker.completed.connect(success.append)
    worker.failed.connect(failures.append)

    worker.start()
    assert worker.wait(5_000), "the worker thread did not finish"
    qapp.processEvents()

    assert len(priority_threads) == 1
    assert priority_threads == analysis_threads
    assert priority_threads[0] != threading.get_ident()
    assert worker.calls == 1
    assert success == ["done"]
    assert failures == []


def test_shared_worker_converts_analysis_errors_to_failure_signal(monkeypatch):
    monkeypatch.setattr(workers, "lower_background_priority", lambda: None, raising=False)
    worker = ProbeWorker(ValueError("pipeline failed"))
    success = []
    failures = []
    worker.completed.connect(success.append)
    worker.failed.connect(failures.append)

    worker.run()

    assert worker.calls == 1
    assert success == []
    assert failures == ["pipeline failed"]


def test_worker_signals_inherit_common_failure_contract():
    assert workers.AnalysisThread.failed is workers.BackgroundAnalysisThread.failed
    assert workers.SignalAnalysisThread.failed is workers.BackgroundAnalysisThread.failed
    assert workers.FullReanalysisThread.failed is workers.BackgroundAnalysisThread.failed


def test_file_worker_forwards_pipeline_progress_and_outcome(monkeypatch):
    expected = object()
    captured = {}

    def analyze_file(source, *, opts, progress):
        captured.update(source=source, opts=opts)
        progress(0.25, "stage")
        return expected

    monkeypatch.setattr("v2m.core.pipeline.analyze_file", analyze_file)
    worker = workers.AnalysisThread("voice.wav", AnalyzeOptions())
    progress = []
    completed = []
    worker.progress.connect(lambda fraction, message: progress.append((fraction, message)))
    worker.finishedOk.connect(completed.append)

    worker._run_analysis()

    assert captured == {"source": "voice.wav", "opts": worker.opts}
    assert progress == [(0.25, "stage")]
    assert completed == [expected]


def test_signal_worker_preserves_float32_buffer_and_emits_result(monkeypatch):
    expected = object()
    captured = {}

    def analyze_signal(signal, sr, *, source, opts):
        captured.update(signal=signal, sr=sr, source=source, opts=opts)
        return expected

    monkeypatch.setattr("v2m.core.pipeline.analyze_signal", analyze_signal)
    original = np.arange(8, dtype=np.float64)
    worker = workers.SignalAnalysisThread(original, 22050, "live")
    completed = []
    worker.finishedOk.connect(lambda result, signal: completed.append((result, signal)))

    worker._run_analysis()

    assert worker.signal.dtype == np.float32
    assert captured["signal"] is worker.signal
    assert captured["sr"] == 22050
    assert captured["source"] == "live"
    assert captured["opts"].min_conf == LIVE_CONF
    assert completed == [(expected, worker.signal)]


def test_full_reanalysis_copies_signal_and_forwards_progress(monkeypatch):
    expected = object()
    captured = {}

    def analyze_signal(signal, sr, *, source, opts, progress):
        captured.update(signal=signal, sr=sr, source=source, opts=opts)
        progress(0.75, "reading")
        return expected

    monkeypatch.setattr("v2m.core.pipeline.analyze_signal", analyze_signal)
    original = np.arange(8, dtype=np.float32)
    options = AnalyzeOptions(min_note_s=0.08)
    worker = workers.FullReanalysisThread(original, 22050, "take.wav", options)
    progress = []
    completed = []
    worker.progress.connect(lambda fraction, message: progress.append((fraction, message)))
    worker.finishedOk.connect(completed.append)

    original[0] = -99.0
    worker._run_analysis()

    assert worker.signal[0] == 0.0
    assert captured == {
        "signal": worker.signal,
        "sr": 22050,
        "source": "take.wav",
        "opts": options,
    }
    assert progress == [(0.75, "reading")]
    assert completed == [expected]
