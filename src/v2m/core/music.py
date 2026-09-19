"""Core music utilities: pitch conversions and note naming."""

from __future__ import annotations

import numpy as np

A4_HZ = 440.0

# English (sharp) note names, C-based
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# French (Latin) solfège names, same pitch-class order.
NOTE_NAMES_FR = ("Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si")

# kept as a backwards-compatible alias used by earlier revisions
NOTE_NAMES_SHARP = NOTE_NAMES


def hz_to_midi(hz):
    """Frequency in Hz -> float MIDI number (NaN passes through)."""
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=float) / A4_HZ)


def midi_to_hz(midi):
    """MIDI number -> frequency in Hz."""
    return A4_HZ * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def _named(midi: int, names: tuple[str, ...]) -> str:
    m = int(round(midi))
    return f"{names[m % 12]}{m // 12 - 1}"


def midi_to_name(midi: int) -> str:
    """MIDI number -> English note name, e.g. 69 -> 'A4'."""
    return _named(midi, NOTE_NAMES)


def midi_to_name_fr(midi: int) -> str:
    """MIDI number -> French note name, e.g. 69 -> 'La4'."""
    return _named(midi, NOTE_NAMES_FR)


def format_note_fr(midi: int) -> str:
    """'La4/A4' — French name first, English kept alongside (display helper)."""
    return f"{midi_to_name_fr(midi)} / {midi_to_name(midi)}"
