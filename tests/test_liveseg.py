"""Tests du segmenteur temps réel (hystérésis)."""

import numpy as np

from v2m.core.liveseg import LiveNoteSegmenter
from v2m.core.music import midi_to_hz


def _feed(seg: LiveNoteSegmenter, hz: float, start: float, dur: float, step: float = 0.01):
    t = start
    while t < start + dur:
        seg.push(t, hz)
        t += step


def test_note_confirmed_and_closed():
    seg = LiveNoteSegmenter()
    _feed(seg, midi_to_hz(69), 0.0, 0.30)
    seg.flush(0.35)
    assert len(seg.notes) == 1
    assert seg.notes[0].midi == 69
    assert seg.notes[0].duration >= 0.25


def test_two_separate_notes():
    seg = LiveNoteSegmenter()
    _feed(seg, midi_to_hz(69), 0.0, 0.30)
    _feed(seg, float("nan"), 0.30, 0.20)   # silence
    _feed(seg, midi_to_hz(72), 0.50, 0.30)
    seg.flush(0.85)
    assert len(seg.notes) == 2
    assert [n.midi for n in seg.notes] == [69, 72]


def test_short_blip_ignored():
    """Un parasite de 40 ms ne crée pas de note."""
    seg = LiveNoteSegmenter()
    _feed(seg, midi_to_hz(69), 0.0, 0.40)
    _feed(seg, midi_to_hz(81), 0.40, 0.04)  # parasite
    _feed(seg, midi_to_hz(69), 0.44, 0.30)
    seg.flush(0.80)
    assert len(seg.notes) == 1
    assert seg.notes[0].midi == 69


def test_nan_ignored():
    seg = LiveNoteSegmenter()
    _feed(seg, float("nan"), 0.0, 0.30)
    seg.flush(0.30)
    assert seg.notes == []
