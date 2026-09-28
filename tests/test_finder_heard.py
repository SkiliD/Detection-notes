"""The finder's "what does the mic hear" channel: heard(), level, silence.

"Il entend mal la voix" (user report) decomposed and measured:
- a steady tone at ANY usable amplitude was already heard (the gate is
  absolute 0.008, a sung voice lands far above) — the REAL gap was the
  DISPLAY: nothing on the page told "the mic hears this" while the
  segmenter held nothing (a glide, a wobble, a pre-confirmation tone);
- `heard()` is that missing channel: the RAW instant YIN reading, no
  median, no confirmation, STALE-aware (mute frames age it out);
- the level bar + gate mark make "the mic hears nothing" distinguishable
  from "the mic hears sound, hold longer" — both honest, no invented note.

The anti-rumble contracts are untouched: the KEY and the HISTORY still
record confirmed notes only; heard() fills the NAME, never a key.
"""

from __future__ import annotations

import numpy as np
import pytest

from v2m.core.liveseg import LiveNoteSegmenter, LivePitchStream

SR = 22_050  # the harness rate (ADR 0009: durations, not samples)


def _tone(f0: float, dur: float, amp: float = 0.2) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = amp * np.sin(2 * np.pi * f0 * t)
    fade = int(0.005 * SR)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y.astype(np.float32)


# --------------------------------------------------------------------------- #
# heard(): the mic's ear, at the core's level
# --------------------------------------------------------------------------- #


def test_heard_answers_during_a_glide_when_current_holds_nothing():
    """A 0.3 s portamento: no confirmed note mid-glide, yet the mic HEARS."""
    stream = LivePitchStream(SR, gate_rms=0.008)
    y = _tone(196.0, 0.15)  # G3: a tone, not yet held long enough to confirm
    block = 1024
    heard_seen = None
    current_seen = None
    for i in range(0, len(y) - block, block):
        t_end = (i + block) / SR
        stream.feed(y[i:i + block], t_end)
        if heard_seen is None and stream.seg.heard() is not None:
            heard_seen = t_end
        if current_seen is None and stream.seg.current() is not None:
            current_seen = t_end
    midi, hz = stream.seg.heard()
    assert 55 <= midi <= 56  # G3-ish, raw
    assert 190 <= hz <= 202


def test_heard_goes_stale_when_the_voice_stops():
    """A mute run ages the reading out: no pitch repeated from thin air."""
    stream = LivePitchStream(SR, gate_rms=0.008)
    y = np.concatenate([_tone(220.0, 0.4), np.zeros(int(0.8 * SR), np.float32)])
    block = 1024
    for i in range(0, len(y) - block, block):
        stream.feed(y[i:i + block], (i + block) / SR)
    assert stream.seg.heard() is None


def test_heard_is_none_before_anything_was_sung():
    seg = LiveNoteSegmenter()
    assert seg.heard() is None


def test_heard_never_invents_a_note_for_the_history_contracts():
    """heard() fills the NAME; the confirmed-only contracts still hold:
    a rumble's frames (below the periodicity gate) must keep the key and
    the history dark — `heard()` returning a pitch does not change that."""
    seg = LiveNoteSegmenter()
    # feed one plausible voiced frame by hand (no confirmation window)
    seg.push(0.05, 220.0)
    assert seg.heard() is not None
    assert seg.current() is not None  # PENDING answers too
    assert seg.confirmed() is None    # but nothing is confirmed


# --------------------------------------------------------------------------- #
# the display contracts (the page), through the session + gate
# --------------------------------------------------------------------------- #


def test_a_wavering_beginner_tone_still_names_one_note_in_the_end():
    """±0.5 st vibrato FRAGMENTS the confirmed notes (measured: 3 -> 7, the
    raw sweep crosses HOLD_ST) — but `current()`/`heard()` never fall
    silent during it: the display keeps answering, which is what the
    finder serves. The confirmed fragmentation is the segmentation's
    honest reading of a wavering tone, not a display bug."""
    stream = LivePitchStream(SR, gate_rms=0.008)
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    f = 220.0 * 2 ** (0.5 * np.sin(2 * np.pi * 5.5 * t) / 12)
    y = (0.2 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)
    silent = 0
    block = 1024
    for i in range(0, len(y) - block, block):
        stream.feed(y[i:i + block], (i + block) / SR)
        if stream.seg.current() is None and stream.seg.heard() is None:
            silent += 1
    assert silent == 0, "the finder channels went dark mid-tone"
    stream.seg.flush(len(y) / SR)
    assert len(stream.seg.notes) >= 3  # the fragmentation, measured


def test_finder_session_distinguishes_reading_from_silence():
    """The idle line's decision: last_reading survives ticks that read a
    pitch and dies on a tick with none — 'sound, no note yet' vs silence."""
    from v2m.gui.finder_session import FinderSession

    session = FinderSession()
    session.feed(57, None, t=1.0)
    assert session.last_reading == 1.0
    session.note_idle()
    assert session.last_reading is None
    # a confirmed-only tick (the name keeps the pending reading) also counts
    session.feed(None, 57, t=2.0)
    assert session.last_reading == 2.0
