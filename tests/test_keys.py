"""Tests tonalité : gammes connues -> candidates attendues."""

from v2m.core.keys import detect_key, pitch_class_histogram
from v2m.core.notes import Note


def _notes_from_midis(midis_durs: list[tuple[int, float]]) -> list[Note]:
    out = []
    t = 0.0
    for midi, dur in midis_durs:
        out.append(Note(start=t, end=t + dur, midi=midi, confidence=0.9))
        t += dur + 0.05
    return out


def test_c_major_scale():
    # gamme de Do majeur, Do insiste (tonique + dominante longues)
    midis = [60, 62, 64, 65, 67, 69, 71, 72]
    durs = [0.6, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.6]
    notes = _notes_from_midis(list(zip(midis, durs)))
    res = detect_key(notes)
    assert res.best.tonic == "C"
    assert res.best.mode == "major"
    assert res.best.score > 0.5


def test_a_minor():
    # gamme de La mineur, La insiste
    midis = [57, 59, 60, 62, 64, 65, 67, 69]
    durs = [0.6, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.6]
    notes = _notes_from_midis(list(zip(midis, durs)))
    res = detect_key(notes)
    assert res.best.tonic == "A"
    assert res.best.mode == "minor"


def test_alternatives_present():
    midis = [60, 62, 64, 65, 67, 69, 71, 72]
    notes = _notes_from_midis([(m, 0.3) for m in midis])
    res = detect_key(notes)
    assert 1 <= len(res.alternatives) <= 2
    # le score décroît
    scores = [res.best.score] + [a.score for a in res.alternatives]
    assert scores == sorted(scores, reverse=True)


def test_histogram_weighted_by_duration():
    notes = [
        Note(start=0, end=1.0, midi=60, confidence=0.9),   # C
        Note(start=1, end=1.1, midi=67, confidence=0.9),   # G court
    ]
    h = pitch_class_histogram(notes)
    assert h[0] > h[7] * 5  # C >> G


def test_empty_notes():
    res = detect_key([])
    assert res.best.score == 0.0
