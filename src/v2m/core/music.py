"""Utilitaires musicaux de base."""

from __future__ import annotations

import numpy as np

A4_HZ = 440.0
NOTE_NAMES_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def hz_to_midi(hz):
    """Fréquence(s) Hz -> numéro MIDI flottant (NaN transmis tel quel)."""
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=float) / A4_HZ)


def midi_to_hz(midi):
    """Numéro(s) MIDI -> fréquence(s) Hz."""
    return A4_HZ * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def midi_to_name(midi: int) -> str:
    """Numéro MIDI -> nom de note, ex. 69 -> 'A4'."""
    m = int(round(midi))
    return f"{NOTE_NAMES_SHARP[m % 12]}{m // 12 - 1}"
