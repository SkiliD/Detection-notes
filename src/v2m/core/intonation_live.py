"""Cumulative intonation evidence for the active live take.

Closed live notes carry residual deviation in semitones (`LiveNote.dev`);
the report summarizes the same fact from `Note.deviation`. Snapshot each
closed note in that report's input shape so its thresholds, registers and
summary policy remain the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import intonation
from .liveseg import LiveNote


@dataclass(frozen=True)
class _LiveNoteView:
    """The two note facts read by `intonation.summarize`."""

    midi: int
    deviation: float


@dataclass
class LiveIntonation:
    """Own the closed-note snapshots for one live take."""

    _notes: list[_LiveNoteView] = field(default_factory=list)

    def add(self, note: LiveNote) -> None:
        """Snapshot a closed note as the report's summary input."""
        self._notes.append(_LiveNoteView(midi=note.midi, deviation=float(note.dev)))

    def reset(self) -> None:
        """Forget evidence when the page starts a new take."""
        self._notes.clear()

    def summary(self) -> dict:
        """Summarize the take-so-far using the report's policy."""
        return intonation.summarize(self._notes)
