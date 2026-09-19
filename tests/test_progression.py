"""Tests for measure-aligned chord progressions (chords.detect_chord_progression)."""

from v2m.core.chords import detect_chord_progression
from v2m.core.meter import Meter
from v2m.core.notes import Note

BPM = 120.0
BEAT = 60.0 / BPM  # 0.5 s
M4 = Meter(4, 4)


def _note_at(t: float, dur: float, pc: int, base: int = 60) -> Note:
    return Note(start=t, end=t + dur, midi=base + pc, confidence=0.9)


def _progression_notes(chords: list[set[int]], beats_per_chord: int = 4) -> list[Note]:
    """One chord per span: root+third+fifth sounded on every beat of the span."""
    notes: list[Note] = []
    t = 0.0
    for pcs in chords:
        for _ in range(beats_per_chord):
            for pc in sorted(pcs):
                notes.append(_note_at(t, BEAT * 0.9, pc))
            t += BEAT
    return notes


def test_c_major_followed_by_f_and_g():
    # measures: C | F | G  (root position triads, roots in the bass)
    chords = [{0, 4, 7}, {5, 9, 12}, {7, 11, 14}]
    notes = _progression_notes(chords)
    num_measures = 3
    prog = detect_chord_progression(notes, BPM, M4, num_measures)
    labels = [c.label for _, _, c in prog]
    assert "C" in labels
    assert "F" in labels
    assert "G" in labels
    # spans are measure-aligned and non-overlapping
    for (m0a, m1a, _), (m0b, m1b, _) in zip(prog, prog[1:]):
        assert m1a == m0b


def test_consecutive_equal_chords_merge():
    chords = [{0, 4, 7}, {0, 4, 7}, {5, 9, 12}]
    notes = _progression_notes(chords)
    prog = detect_chord_progression(notes, BPM, M4, 3)
    labels = [c.label for _, _, c in prog]
    assert labels.count("C") == 1
    c_span = next((m0, m1) for m0, m1, c in prog if c.label == "C")
    assert c_span == (1, 3)  # measures 1-2 merged


def test_measure_without_enough_notes_is_skipped():
    notes = _progression_notes([{0, 4, 7}], beats_per_chord=8)
    # measures 3-4 have no notes at all
    prog = detect_chord_progression(notes, BPM, M4, 4)
    for m0, _, _ in prog:
        assert m0 <= 2


def test_default_grid_without_meter():
    notes = _progression_notes([{0, 4, 7}, {7, 11, 14}])
    prog = detect_chord_progression(notes, BPM, None, 2)
    assert prog
    labels = [c.label for _, _, c in prog]
    assert "C" in labels or "G" in labels


def test_empty_notes_returns_empty():
    assert detect_chord_progression([], BPM, M4, 4) == []
