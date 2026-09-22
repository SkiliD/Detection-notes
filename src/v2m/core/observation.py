"""Intermediate vocal evidence kept separate from musical interpretation."""

from __future__ import annotations

from dataclasses import dataclass

from .onsets import Attacks
from .pitch import PitchTrack


@dataclass(frozen=True)
class VoiceObservation:
    """Pitch and attack evidence from one signal, before Notes are interpreted."""

    pitch: PitchTrack
    attacks: Attacks | None
    duration_s: float
