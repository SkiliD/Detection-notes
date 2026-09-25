"""Tests chords.py: template scoring, bass bonus, global and segmented chords."""

from v2m.core.chords import (
    CHORD_TEMPLATES,
    ChordCandidate,
    ChordResult,
    detect_chord_segments,
    detect_chords,
)
from v2m.core.notes import Note


def _chord(midi_weights: dict[int, float]) -> ChordResult:
    """A detection the test itself asserts: fails with a clear message on None."""
    result = detect_chords(_chord_notes(midi_weights))
    assert result is not None, "accordeur indécis sur un accord clair"
    return result


def _chord_notes(midi_weights: dict[int, float]) -> list[Note]:
    """Build notes from {absolute_midi: duration}; order defines the bass."""
    notes = []
    for i, (midi, dur) in enumerate(sorted(midi_weights.items())):
        start = i * 0.5
        notes.append(Note(start=start, end=start + dur, midi=midi, confidence=0.9))
    return notes


def test_major_triad_recognized():
    notes = _chord_notes({60: 1.0, 64: 1.0, 67: 1.0})  # C4 E4 G4
    res = _chord({60: 1.0, 64: 1.0, 67: 1.0})
    assert res.best.label == "C"
    assert res.best.score > 0.9


def test_minor_triad_recognized_with_bass_a():
    # A2 in the bass: root-position Am, not Fmaj7 (whose bass would be F)
    res = _chord({45: 1.2, 60: 1.0, 64: 1.0})  # A2 C4 E4
    assert res.best.label == "Am"


def test_dominant_seventh_beats_plain_triad():
    # G B D F: the 7th disqualifies a plain G triad (weight outside template)
    res = _chord({55: 1.0, 59: 1.0, 62: 1.0, 65: 1.0})
    assert res.best.label == "G7"


def test_bass_note_prefers_root_position():
    # C strong + E/G: root position wins. A bare E-G dyad is genuinely
    # ambiguous (Em or C/E): the detector may stay undecided, but when it
    # does answer with the E bass, Em must win over C/E.
    notes_with_c = _chord_notes({60: 1.0, 64: 0.2, 67: 1.0})
    decided = detect_chords(notes_with_c)
    assert decided is not None and decided.best.label == "C"
    notes_bass_e = _chord_notes({52: 1.0, 55: 1.0, 64: 1.0, 67: 1.0})
    res = detect_chords(notes_bass_e)
    if res is not None:
        assert res.best.label == "Em"


def test_too_few_notes_returns_none():
    notes = _chord_notes({60: 1.0, 64: 1.0})
    assert detect_chords(notes) is None


def test_single_pitch_class_returns_none():
    # three notes but all the same pitch: no chord can be named
    notes = [Note(start=i, end=i + 0.9, midi=60, confidence=0.9) for i in range(3)]
    assert detect_chords(notes) is None


def test_alternatives_have_distinct_roots():
    res = _chord({60: 1.0, 64: 1.0, 67: 1.0, 69: 0.4})  # C major + a bit of A
    roots = [c.root for c in res.alternatives]
    assert res.best.root not in roots
    assert len(roots) == len(set(roots))


def test_segments_cover_long_recording():
    # 4 s of a C major arpeggio -> at least one segment labeled C
    notes = []
    t = 0.0
    for pc in (0, 4, 7, 0, 4, 7, 0, 4):
        notes.append(Note(start=t, end=t + 0.45, midi=60 + pc, confidence=0.9))
        t += 0.5
    segs = detect_chord_segments(notes, segment_s=2.0)
    assert segs
    assert any(c.label == "C" for _, _, c in segs)
    for t0, t1, _ in segs:
        assert t1 > t0


def test_segments_are_adjacent_windows_not_overlapping_ones():
    """Régression : les fenêtres se recouvraient de moitié (pas de 1 s), donc
    chaque accord sortait deux fois dans le JSON et dans le tableau."""
    notes = []
    t = 0.0
    for pc in (0, 4, 7, 0, 4, 7, 0, 4):
        notes.append(Note(start=t, end=t + 0.45, midi=60 + pc, confidence=0.9))
        t += 0.5

    segs = detect_chord_segments(notes, segment_s=2.0)

    assert [(t0, t1) for t0, t1, _ in segs] == [(0.0, 2.0), (2.0, 3.95)]
    for (t0, t1, _), (n0, _, _) in zip(segs, segs[1:]):
        assert n0 == t1, "les fenêtres doivent se toucher, jamais se recouvrir"
        assert t1 - t0 == 2.0


def test_segments_empty_for_short_input():
    # fewer than min_notes (3): every window is skipped
    notes = _chord_notes({60: 0.5, 64: 0.5})
    assert detect_chord_segments(notes, segment_s=2.0) == []


def test_all_templates_have_distinct_intervals():
    seen = {v for v in CHORD_TEMPLATES.values()}
    assert len(seen) == len(CHORD_TEMPLATES)


def test_candidate_label_formatting():
    c = ChordCandidate("F#", "m7", 0.8)
    assert c.label == "F#m7"
