"""Lifecycle decisions for full reanalysis and live phrase replacement.

The window owns workers and applies the returned facts. Keeping phase changes,
restore context and completion tokens here makes queued results safe to apply
without making this module depend on Qt or the analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray

    from ..core.edits import EditState
    from ..core.pipeline import AnalysisResult


class WorkflowPhase(Enum):
    """Exclusive operation currently owned by the coordinator."""

    IDLE = auto()
    FULL_REANALYSIS = auto()
    REPLACEMENT_CAPTURE = auto()
    SPLICE_ANALYSIS = auto()
    REPLACEMENT_DECISION = auto()


@dataclass(frozen=True)
class FullReanalysis:
    """The prior reading and retained audio while a full re-read runs."""

    token: int
    source: AnalysisResult
    signal: ndarray


@dataclass(frozen=True)
class Replacement:
    """Original take and selection needed to finish or restore a replacement."""

    token: int
    before: EditState
    first: int
    last: int
    record_audio: bool
    phrase_window: tuple[float, float] | None = None

    @property
    def source(self) -> AnalysisResult:
        return self.before.result

    @property
    def signal(self) -> ndarray | None:
        return self.before.signal


@dataclass(frozen=True)
class ReplacementResolution:
    """Facts for the adapter to accept, cancel or abandon a splice."""

    replacement: Replacement
    record_history: bool
    show_restored_state: bool


@dataclass(frozen=True)
class WorkflowTransition:
    """Facts returned for one accepted completion or failure."""

    phase: WorkflowPhase
    full_reanalysis: FullReanalysis | None = None
    replacement: Replacement | None = None
    resolution: ReplacementResolution | None = None


class AnalysisWorkflow:
    """Own exclusive workflow phases and the context needed to resolve them.

    Tokens identify the operation that emitted an asynchronous completion.
    Abandoning or completing an operation invalidates its token, so an old
    queued signal cannot mutate a later take or its result view.
    """

    def __init__(self) -> None:
        self._phase = WorkflowPhase.IDLE
        self._next_token = 0
        self._active_token: int | None = None
        self._full: FullReanalysis | None = None
        self._replacement: Replacement | None = None

    @property
    def phase(self) -> WorkflowPhase:
        return self._phase

    @property
    def replacement(self) -> Replacement | None:
        return self._replacement

    @property
    def replacement_pending(self) -> bool:
        return self._phase in (WorkflowPhase.REPLACEMENT_CAPTURE, WorkflowPhase.SPLICE_ANALYSIS)

    def begin_full_reanalysis(
        self, source: AnalysisResult, signal: ndarray
    ) -> FullReanalysis | None:
        """Start an exclusive full re-read and retain an immutable audio copy."""
        if self._phase is not WorkflowPhase.IDLE or len(signal) == 0:
            return None
        token = self._new_token()
        context = FullReanalysis(token, source, signal.copy())
        self._phase = WorkflowPhase.FULL_REANALYSIS
        self._full = context
        return context

    def begin_replacement(
        self,
        before: EditState,
        first: int,
        last: int,
        record_audio: bool,
    ) -> Replacement | None:
        """Start a contiguous live replacement while preserving its full state."""
        if (
            self._phase is not WorkflowPhase.IDLE
            or before.result.source != "live"
            or before.signal is None
            or len(before.signal) == 0
            or first < 0
            or last < first
            or last >= len(before.result.notes)
        ):
            return None
        token = self._new_token()
        context = Replacement(token, before, first, last, record_audio)
        self._phase = WorkflowPhase.REPLACEMENT_CAPTURE
        self._replacement = context
        return context

    def begin_splice_analysis(self, token: int) -> Replacement | None:
        """Move the current capture to the synchronous splice/analysis phase."""
        if not self._matches(token, WorkflowPhase.REPLACEMENT_CAPTURE):
            return None
        self._phase = WorkflowPhase.SPLICE_ANALYSIS
        return self._replacement

    def set_splice_window(self, token: int, phrase_window: tuple[float, float]) -> bool:
        """Keep the analyzed splice's audition window with its restore state."""
        if not self._matches(token, WorkflowPhase.SPLICE_ANALYSIS):
            return False
        replacement = self._replacement
        if replacement is None:
            return False
        self._replacement = replace(replacement, phrase_window=phrase_window)
        return True

    def complete_full_reanalysis(self, token: int) -> WorkflowTransition | None:
        """Consume one current successful full-reanalysis completion."""
        if not self._matches(token, WorkflowPhase.FULL_REANALYSIS):
            return None
        context = self._full
        self._clear()
        if context is None:
            return None
        return WorkflowTransition(WorkflowPhase.IDLE, full_reanalysis=context)

    def fail_full_reanalysis(self, token: int) -> WorkflowTransition | None:
        """Consume a failed full re-read and return the prior reading/audio."""
        return self.complete_full_reanalysis(token)

    def complete_replacement_analysis(self, token: int) -> WorkflowTransition | None:
        """Enter the explicit keep/cancel decision after analysis succeeds."""
        if not self._matches(token, WorkflowPhase.SPLICE_ANALYSIS):
            return None
        replacement = self._replacement
        if replacement is None or replacement.phrase_window is None:
            return None
        self._phase = WorkflowPhase.REPLACEMENT_DECISION
        self._active_token = None
        return WorkflowTransition(WorkflowPhase.REPLACEMENT_DECISION, replacement=replacement)

    def fail_replacement(self, token: int) -> WorkflowTransition | None:
        """Return the saved take after splice or analysis failure."""
        if token != self._active_token or self._phase not in (
            WorkflowPhase.REPLACEMENT_CAPTURE,
            WorkflowPhase.SPLICE_ANALYSIS,
        ):
            return None
        replacement = self._replacement
        self._clear()
        if replacement is None:
            return None
        return WorkflowTransition(WorkflowPhase.IDLE, replacement=replacement)

    def resolve_replacement(self, *, accept: bool) -> WorkflowTransition | None:
        """Resolve the pending decision; acceptance alone requests history."""
        if self._phase is not WorkflowPhase.REPLACEMENT_DECISION or self._replacement is None:
            return None
        replacement = self._replacement
        self._clear()
        resolution = ReplacementResolution(
            replacement=replacement,
            record_history=accept,
            show_restored_state=not accept,
        )
        return WorkflowTransition(WorkflowPhase.IDLE, resolution=resolution)

    def abandon_replacement(self) -> WorkflowTransition | None:
        """Invalidate an in-flight or pending replacement without recording it."""
        if (
            self._phase
            not in (
                WorkflowPhase.REPLACEMENT_CAPTURE,
                WorkflowPhase.SPLICE_ANALYSIS,
                WorkflowPhase.REPLACEMENT_DECISION,
            )
            or self._replacement is None
        ):
            return None
        replacement = self._replacement
        self._clear()
        resolution = ReplacementResolution(
            replacement=replacement,
            record_history=False,
            show_restored_state=False,
        )
        return WorkflowTransition(WorkflowPhase.IDLE, resolution=resolution)

    def is_current(self, token: int, phase: WorkflowPhase) -> bool:
        """Tell the adapter whether progress belongs to the active operation."""
        return self._matches(token, phase)

    def accept_current(self, token: int, phase: WorkflowPhase) -> WorkflowTransition | None:
        """Consume one valid asynchronous completion for its current phase."""
        if not self._matches(token, phase):
            return None
        if phase is WorkflowPhase.FULL_REANALYSIS:
            return self.complete_full_reanalysis(token)
        if phase is WorkflowPhase.SPLICE_ANALYSIS:
            return self.complete_replacement_analysis(token)
        return None

    def _new_token(self) -> int:
        self._next_token += 1
        self._active_token = self._next_token
        return self._next_token

    def _matches(self, token: int, phase: WorkflowPhase) -> bool:
        return self._active_token == token and self._phase is phase

    def _clear(self) -> None:
        self._phase = WorkflowPhase.IDLE
        self._active_token = None
        self._full = None
        self._replacement = None
