"""Key tests: known scales -> expected candidates."""

from v2m.core.keys import detect_key, pitch_class_histogram, scale_pitch_classes, snap_to_key
from v2m.core.notes import Note


def _notes_from_midis(midis_durs: list[tuple[int, float]]) -> list[Note]:
    out = []
    t = 0.0
    for midi, dur in midis_durs:
        out.append(Note(start=t, end=t + dur, midi=midi, confidence=0.9))
        t += dur + 0.05
    return out


def test_c_major_scale():
    # C major scale, C insisted on (long tonic + dominant)
    midis = [60, 62, 64, 65, 67, 69, 71, 72]
    durs = [0.6, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.6]
    notes = _notes_from_midis(list(zip(midis, durs)))
    res = detect_key(notes)
    assert res.best.tonic == "C"
    assert res.best.mode == "major"
    assert res.best.score > 0.5


def test_a_minor():
    # A minor scale, A insisted on
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
    # the score decreases
    scores = [res.best.score] + [a.score for a in res.alternatives]
    assert scores == sorted(scores, reverse=True)


def test_histogram_weighted_by_duration():
    notes = [
        Note(start=0, end=1.0, midi=60, confidence=0.9),   # C
        Note(start=1, end=1.1, midi=67, confidence=0.9),   # short G
    ]
    h = pitch_class_histogram(notes)
    assert h[0] > h[7] * 5  # C >> G


def test_empty_notes():
    res = detect_key([])
    assert res.best.score == 0.0


# ---------------------------------------------------------------------------
# snap_to_key: the key snap (residual deviation + context)
# ---------------------------------------------------------------------------


def _one(midi: int, dev: float) -> Note:
    return Note(start=0.0, end=0.5, midi=midi, confidence=0.9, deviation=dev)


def test_scale_pitch_classes_covers_the_key():
    s = scale_pitch_classes(detect_key(_notes_from_midis([(60, 0.4), (64, 0.4), (67, 0.4)])).best)
    assert len(s) == 7 and 0 in s and 7 in s  # C major: C and G present, F# absent
    assert 6 not in s


def test_ambiguous_out_of_scale_note_resolves_toward_its_lean():
    """An F# sung NEARLY in tune (deviation -0.4: leaning down) in C major
    joins F, the degree it leans toward — the key-snap rule. A frank F#
    (deviation ~0) stays chromatic."""
    key = detect_key(_notes_from_midis([(60, 0.4), (64, 0.4), (67, 0.4)])).best
    flat_fsharp = [_one(66, -0.4)]  # 40 cents flat of F#3 -> leans to F
    assert snap_to_key(flat_fsharp, key) == 1
    assert flat_fsharp[0].midi == 65  # F3
    assert flat_fsharp[0].deviation == 0.0
    sharp_fsharp = [_one(66, 0.05)]  # a chromatic shade: untouched
    assert snap_to_key(sharp_fsharp, key) == 0
    assert sharp_fsharp[0].midi == 66


def test_mistuned_scale_degree_is_re_anchored():
    """A scale degree sung off (deviation >= ±0.3) is re-anchored on its
    degree: the key says so, the lean was vocal noise. The same shade
    under the threshold stays a shade."""
    key = detect_key(_notes_from_midis([(60, 0.4), (64, 0.4), (67, 0.4)])).best
    mistuned = [_one(65, 0.45)]  # F sung 45 cents sharp: a scale degree anyway
    assert snap_to_key(mistuned, key) == 1
    assert mistuned[0].midi == 65 and mistuned[0].deviation == 0.0
    shade = [_one(65, 0.10)]
    assert snap_to_key(shade, key) == 0
    assert shade[0].deviation == 0.10  # kept: expression, not mistune


def test_snap_without_key_or_without_deviation_is_a_noop():
    notes = [_one(66, -0.4)]
    assert snap_to_key(notes, None) == 0
    assert notes[0].midi == 66
    bare = [Note(start=0.0, end=0.5, midi=66, confidence=0.9)]  # no deviation field value
    key = detect_key(_notes_from_midis([(60, 0.4), (64, 0.4), (67, 0.4)])).best
    assert snap_to_key(bare, key) == 0  # default dev 0.0: below the band
    assert bare[0].midi == 66
