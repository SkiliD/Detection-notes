"""Notation tests: pitch spelling, key signatures, durations, layout."""

from v2m.core.liveseg import LiveNote
from v2m.core.grid import BeatGrid, Meter
from v2m.core.notation import (
    accidental_needed,
    key_fifths,
    key_signature,
    layout_staff,
    notated_duration,
    notated_rest_duration,
    spell,
)
from v2m.core.notes import Note


def _note(midi, start=0.0, dur_s=0.5, figure=None):
    return Note(start=start, end=start + dur_s, midi=midi, confidence=0.9, figure=figure)


GRID = BeatGrid(120.0)


# ------------------------------------------------------------------ spelling
def test_spell_diatonic_note_keeps_its_letter():
    p = spell(60, fifths=0)          # C4 in C major
    assert (p.letter, p.octave, p.alter, p.step) == ("C", 4, 0, 28)


def test_spell_chromatic_note_leans_sharp_in_sharp_keys():
    p = spell(66, fifths=0)          # F#4, not Gb4, in C major
    assert (p.letter, p.alter, p.step) == ("F", 1, 31)
    assert spell(61, fifths=0).letter == "C"      # C#4
    assert spell(61, fifths=0).alter == 1


def test_spell_chromatic_note_leans_flat_in_flat_keys():
    p = spell(66, fifths=-1)         # F major: Gb4
    assert (p.letter, p.alter, p.step) == ("G", -1, 32)
    assert spell(61, fifths=-1).letter == "D"     # Db4 in F major


def test_spell_follows_the_key_signature():
    b_flat = spell(70, fifths=-2)    # Bb4 in Bb major
    assert (b_flat.letter, b_flat.alter, b_flat.octave) == ("B", -1, 4)
    f_sharp = spell(66, fifths=1)    # F#4 is in G major: no accidental needed
    assert (f_sharp.letter, f_sharp.alter) == ("F", 1)


def test_spell_key_letter_sung_natural_stays_natural():
    """A key letter sung natural is spelled NATURAL, not a neighbour.

    Measured: in G major, an F natural fell into the chromatic fallback
    and came out E# (the nearest altered neighbour) - the staff engraved
    a wrong pitch instead of an F with a natural. The natural is ALWAYS
    the winning spelling when the key's letter exists.
    """
    f_nat = spell(65, fifths=1)      # F natural in G major (F# of the key undone)
    assert (f_nat.letter, f_nat.alter) == ("F", 0)
    assert accidental_needed(0, "F", 1)   # and it needs a natural sign

    b_nat = spell(71, fifths=-1)     # B natural in F major (Bb of the key undone)
    assert (b_nat.letter, b_nat.alter) == ("B", 0)
    assert accidental_needed(0, "B", -1)

    # non-regression of the neighbour spellings the fallback already got right
    assert spell(61, fifths=0).letter == "C"     # C# in C major
    assert spell(61, fifths=-1).letter == "D"    # Db in F major


def test_spell_octave_belongs_to_the_letter():
    # B#3 and Cb4 sit at the same pitch as C4 (60) but on different steps
    assert spell(60, fifths=0).octave == 4
    assert spell(59, fifths=0).letter == "B"
    assert spell(59, fifths=0).octave == 3


# ------------------------------------------------------------------ signature
def test_key_fifths_from_labels_and_modes():
    assert key_fifths(None) == 0
    assert key_fifths("C major") == 0
    assert key_fifths("G major") == 1
    assert key_fifths("F major") == -1
    assert key_fifths("F# major") == 6
    assert key_fifths("A minor") == 0     # relative major of C
    assert key_fifths("E minor") == 1
    assert key_fifths("D minor") == -1
    assert key_fifths("C minor") == -3


def test_key_signature_steps_follow_the_engraving_order():
    assert key_signature(0) == []
    assert key_signature(2) == [(38, 1), (35, 1)]        # F#5 then C#5
    assert key_signature(-3) == [(34, -1), (37, -1), (33, -1)]  # Bb4 Eb5 Ab4


def test_accidental_needed_only_outside_the_key():
    assert accidental_needed(1, "F", 0) is True          # F# in C major
    assert accidental_needed(1, "F", 1) is False         # F# is in G major
    assert accidental_needed(-1, "B", -1) is False       # Bb is in F major
    assert accidental_needed(1, "F", -1) is True


# ----------------------------------------------------------------- durations
def test_notated_duration_uses_the_detected_figure():
    w = notated_duration(_note(60, figure="noire"), grid=GRID)
    assert (w.type_, w.beats, w.dots) == ("quarter", 1.0, 0)
    w = notated_duration(_note(60, figure="croche pointée"), grid=GRID)
    assert (w.type_, w.beats, w.dots) == ("eighth", 0.75, 1)
    w = notated_duration(_note(60, figure="triolet de croche"), grid=GRID)
    assert w.tuplet == (3, 2) and w.type_ == "eighth"


def test_notated_duration_falls_back_to_a_drawable_value():
    # 0.7 beats is closest to a dotted eighth (0.75), not to a quarter
    w = notated_duration(_note(60, dur_s=0.35), grid=GRID)
    assert (w.type_, w.beats, w.dots) == ("eighth", 0.75, 1)
    # half a second at 120 BPM = one quarter beat
    assert notated_duration(_note(60, dur_s=0.5), grid=GRID).beats == 1.0
    # a very short blip never becomes a 64th: the smallest value floor applies
    assert notated_duration(_note(60, dur_s=0.01), grid=GRID).beats >= 0.25


# -------------------------------------------------------------------- layout
def test_layout_places_notes_on_the_detected_grid():
    notes = [_note(60, 0.0), _note(62, 0.5), _note(64, 2.0)]
    positions = [(1, 1.0), (1, 2.0), (2, 1.0)]
    lay = layout_staff(notes, grid=GRID, key="C major", positions=positions)
    assert [n.onset for n in lay.notes] == [0.0, 1.0, 4.0]
    assert [n.measure for n in lay.notes] == [1, 1, 2]
    assert lay.beats_per_measure == 4.0
    assert lay.measures == 2
    assert lay.signature == []
    assert lay.key_tonic == "C" and lay.key_mode == "major"


def test_layout_honours_compound_meter_units():
    notes = [_note(60, 0.0), _note(62, 0.75)]
    lay = layout_staff(
        notes, grid=BeatGrid(120.0, Meter(6, 8)), positions=[(1, 1.0), (1, 2.0)]
    )
    assert lay.beats_per_measure == 3.0   # two dotted-quarter beats
    assert lay.slot_quarters == 1.5
    assert lay.notes[1].onset == 1.5
    assert lay.meter_label == "6/8"


def test_layout_chains_durations_without_a_grid():
    """Live take: no positions, the notes are chained by their figures."""
    notes = [
        _note(60, 0.0, dur_s=0.5, figure="noire"),
        _note(62, 0.5, dur_s=0.25, figure="croche"),
        _note(64, 0.75, dur_s=0.25, figure="croche"),
    ]
    lay = layout_staff(notes, grid=GRID)
    assert [n.onset for n in lay.notes] == [0.0, 1.0, 1.5]
    assert [n.written.type_ for n in lay.notes] == ["quarter", "eighth", "eighth"]
    assert lay.measures == 1
    assert lay.total_beats == 2.0


def test_layout_appends_the_preview_without_touching_the_notes():
    notes = [_note(60, 0.0, dur_s=0.5, figure="noire")]
    preview = LiveNote(start=0.5, end=0.9, midi=67)      # being sung right now
    lay = layout_staff(notes, grid=GRID, key="C major", preview=preview)
    assert len(lay.notes) == 2
    assert len(notes) == 1                               # the model is untouched
    last = lay.notes[-1]
    assert last.preview is True and last.onset == 1.0
    assert last.step == 32 and last.alter == 0           # G4
    assert lay.notes[0].preview is False


def test_layout_engraves_accidentals_outside_the_key():
    notes = [_note(66, 0.0), _note(67, 0.5)]             # F# then G, in C major
    lay = layout_staff(notes, grid=GRID, key="C major")
    assert lay.notes[0].accidental is True
    assert lay.notes[1].accidental is False


def test_layout_of_an_empty_take_is_empty():
    lay = layout_staff([], grid=GRID)
    assert lay.notes == [] and lay.measures == 0 and lay.total_beats == 0.0


# ------------------------------------------------------------------ silences
def test_notated_rest_duration_follows_the_figure():
    from v2m.core.rests import Rest

    w = notated_rest_duration(Rest(start=0.0, end=0.5, figure="soupir"), grid=GRID)
    assert (w.type_, w.beats, w.dots) == ("quarter", 1.0, 0)
    # without a figure (unknown tempo): the nearest drawable value
    w = notated_rest_duration(Rest(start=0.0, end=4.0), BeatGrid(60.0))
    assert (w.type_, w.beats) == ("whole", 4.0)


def test_layout_engraves_rests_between_phrases_on_the_grid():
    """Grid path: the rest fills the written gap left by the figures (the
    tempo slowed during the breath); the positions do not move."""
    from v2m.core.rests import Rest

    notes = [_note(60, 0.0), _note(62, 2.0)]
    positions = [(1, 1.0), (2, 1.0)]
    rests = [Rest(start=0.5, end=1.0, figure="soupir")]
    lay = layout_staff(notes, grid=GRID, positions=positions, rests=rests)
    assert [n.onset for n in lay.notes] == [0.0, 4.0]
    assert len(lay.rests) == 1
    r = lay.rests[0]
    assert (r.shape, r.onset, r.measure) == ("soupir", 1.0, 1)
    assert r.written.type_ == "quarter"


def test_layout_chains_rests_so_phrases_are_separated():
    """Chained path (live): the rest CONSUMES written time — the next
    phrase starts after the breath instead of being glued to it."""
    from v2m.core.rests import Rest

    notes = [
        _note(60, 0.0, dur_s=0.5, figure="noire"),
        _note(67, 1.5, dur_s=0.5, figure="noire"),
    ]
    rests = [Rest(start=0.5, end=1.5, figure="pause")]
    lay = layout_staff(notes, grid=GRID, rests=rests)
    assert [n.onset for n in lay.notes] == [0.0, 5.0]     # 1 note + 4 beats of rest
    assert [r.onset for r in lay.rests] == [1.0]
    assert lay.rests[0].shape == "pause"
    assert lay.total_beats == 6.0


def test_rest_without_figure_falls_back_to_the_closest_value():
    from v2m.core.rests import Rest

    notes = [
        _note(60, 0.0, dur_s=0.5, figure="noire"),
        _note(62, 3.5, dur_s=0.5, figure="noire"),
    ]
    rests = [Rest(start=0.5, end=2.5)]          # 2 s à 120 BPM = 4 temps
    lay = layout_staff(notes, grid=GRID, rests=rests)
    r = lay.rests[0]
    assert r.shape is None
    assert (r.written.type_, r.written.beats) == ("whole", 4.0)
