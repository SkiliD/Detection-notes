"""Core music utilities: pitch conversions and note naming."""

from __future__ import annotations

import numpy as np

A4_HZ = 440.0

# English (sharp) note names, C-based
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# French (Latin) sol-fa names, same pitch-class order.
NOTE_NAMES_FR = ("Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si")

# kept as a backwards-compatible alias used by earlier revisions
NOTE_NAMES_SHARP = NOTE_NAMES

# The 88 keys of a classical piano: A0 (MIDI 21) to C8 (108). Nothing outside
# them is a note a piano has, so no recognition path may produce one — the
# staff, the keyboard view and the export all assume a piano key.
PIANO_MIN_MIDI = 21
PIANO_MAX_MIDI = 108


def on_piano(midi) -> bool:
    """True when a MIDI number (float allowed) is a classical piano key.

    Undecidable input is not a key either — None, NaN, or a string that is
    not even a number: the point is a guarantee, not a best effort.
    """
    try:
        value = float(midi)
    except (TypeError, ValueError):
        return False
    return PIANO_MIN_MIDI <= value <= PIANO_MAX_MIDI


def sane_bpm(bpm, default: float = 120.0) -> float:
    """A tempo a grid can divide by: seconds <-> beats needs bpm > 0.

    Steps that engrave or export divide by the tempo, so a missing or
    failed detection (None, 0, NaN) falls back on a plausible default
    instead of raising ZeroDivisionError inside a finished analysis.
    """
    try:
        value = float(bpm)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) and value > 0 else default


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
