"""Qt-free transition tests for the GUI analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from v2m.core.edits import EditState
from v2m.core.pipeline import AnalysisResult
from v2m.gui.analysis_workflow import AnalysisWorkflow, WorkflowPhase


@dataclass
class Reading:
    source: str
    notes: list


def _state(source: str = "live") -> EditState:
    signal = np.arange(12, dtype=np.float32)
    result = cast(AnalysisResult, Reading(source, [object(), object()]))
    return EditState(result, signal, ("live note",), "replacement")


def test_full_reanalysis_preserves_context_and_rejects_stale_completions():
    workflow = AnalysisWorkflow()
    before = _state("take.wav")
    signal = np.arange(6, dtype=np.float32)

    context = workflow.begin_full_reanalysis(before.result, signal)

    assert context is not None
    assert context.source is before.result
    assert np.array_equal(context.signal, signal)
    assert context.signal is not signal
    assert workflow.phase is WorkflowPhase.FULL_REANALYSIS
    assert workflow.begin_replacement(_state(), 0, 0, False) is None
    signal[0] = -1
    assert context.signal[0] == 0

    completed = workflow.complete_full_reanalysis(context.token)

    assert completed is not None
    assert completed.full_reanalysis is context
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.complete_full_reanalysis(context.token) is None


def test_full_reanalysis_failure_returns_the_same_restore_facts():
    workflow = AnalysisWorkflow()
    context = workflow.begin_full_reanalysis(_state("take.wav").result, np.ones(3))
    assert context is not None

    failed = workflow.fail_full_reanalysis(context.token)

    assert failed is not None and failed.full_reanalysis is context
    assert failed.full_reanalysis is not None
    assert failed.full_reanalysis.source.source == "take.wav"
    assert workflow.phase is WorkflowPhase.IDLE


def test_replacement_capture_analysis_accept_and_cancel_have_distinct_history_intents():
    workflow = AnalysisWorkflow()
    before = _state()
    replacement = workflow.begin_replacement(before, 0, 1, record_audio=True)

    assert replacement is not None
    assert workflow.phase is WorkflowPhase.REPLACEMENT_CAPTURE
    assert workflow.replacement_pending
    assert workflow.begin_full_reanalysis(before.result, np.ones(3)) is None

    analysis = workflow.begin_splice_analysis(replacement.token)
    assert analysis is replacement
    assert workflow.replacement_pending
    assert workflow.phase is WorkflowPhase.SPLICE_ANALYSIS
    assert workflow.set_splice_window(replacement.token, (0.25, 1.75))

    pending = workflow.complete_replacement_analysis(replacement.token)
    assert pending is not None
    assert pending.phase is WorkflowPhase.REPLACEMENT_DECISION
    assert pending.replacement is not None
    assert pending.replacement.before is before
    assert pending.replacement.phrase_window == (0.25, 1.75)

    accepted = workflow.resolve_replacement(accept=True)
    assert accepted is not None and accepted.resolution is not None
    assert accepted.resolution.record_history
    assert not accepted.resolution.show_restored_state
    assert accepted.resolution.replacement.before is before
    assert workflow.resolve_replacement(accept=True) is None

    another = workflow.begin_replacement(before, 0, 0, record_audio=False)
    assert another is not None
    assert workflow.begin_splice_analysis(another.token) is not None
    assert workflow.set_splice_window(another.token, (0.1, 0.5))
    assert workflow.complete_replacement_analysis(another.token) is not None

    cancelled = workflow.resolve_replacement(accept=False)
    assert cancelled is not None and cancelled.resolution is not None
    assert not cancelled.resolution.record_history
    assert cancelled.resolution.show_restored_state
    assert cancelled.resolution.replacement.before is before


def test_replacement_failure_and_abandonment_invalidate_old_tokens():
    workflow = AnalysisWorkflow()
    before = _state()
    failed_context = workflow.begin_replacement(before, 0, 0, False)
    assert failed_context is not None

    failed = workflow.fail_replacement(failed_context.token)
    assert failed is not None and failed.replacement is failed_context
    assert workflow.phase is WorkflowPhase.IDLE
    assert workflow.fail_replacement(failed_context.token) is None

    abandoned_context = workflow.begin_replacement(before, 0, 0, True)
    assert abandoned_context is not None
    abandoned = workflow.abandon_replacement()
    assert abandoned is not None and abandoned.resolution is not None
    assert not abandoned.resolution.record_history
    assert not abandoned.resolution.show_restored_state

    next_context = workflow.begin_full_reanalysis(_state("new.wav").result, np.ones(4))
    assert next_context is not None
    assert workflow.complete_replacement_analysis(abandoned_context.token) is None
    assert workflow.fail_replacement(abandoned_context.token) is None
    assert workflow.phase is WorkflowPhase.FULL_REANALYSIS
    assert workflow.is_current(next_context.token, WorkflowPhase.FULL_REANALYSIS)
    assert not workflow.is_current(abandoned_context.token, WorkflowPhase.FULL_REANALYSIS)
