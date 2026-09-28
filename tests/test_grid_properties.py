"""Grid properties, written BEFORE the Grid existed.

The metric frame — seconds ↔ beats conversion, measure position, snapping
onto the grid — used to be copy-pasted across seven modules. These tests
do not describe that dispersion: they pin what every consumer must keep
doing once it is one module (`grid.BeatGrid`, see CONTEXT.md "Grille").
It is the refactor's non-regression contract: each property can then be
re-read through the grid without changing its statement.

Two commitments:

- the properties go through the consumers' PUBLIC interfaces (figures,
  meter, musicxml, midi_export), never by re-implementing their
  arithmetic inside the test;
- they hold at SEVERAL tempos: tempo invariance is the property, not a
  result at 120 BPM. A `171.02 BPM` shows up everywhere here because
  that is the tempo actually detected on `chant.wav`.

A take's snapping is a member of the Grid (`BeatGrid.snap_notes`): the
snapping properties below query it directly, whereas they used to go
through a private function of the export. What the export makes of it
remains held by the net, `test_reference_artifacts.py`. The last
property does the same for `BeatGrid.written`, the choice of the
written value of a figure-less duration: it queries the Grid AND the
Staff, which must use it instead of re-rounding at home.
"""

import io

import numpy as np
import pytest
from v2m.core.figures import label_notes
from v2m.core.grid import CATALOGUE, BeatGrid, Meter
from v2m.core.midi_export import build_midi
from v2m.core.notation import notated_duration
from v2m.core.notes import Note

BPM = (60.0, 120.0, 171.02)  # incl. the tempo actually detected on chant.wav


def _note(midi, start, dur_s):
    return Note(start=start, end=start + dur_s, midi=midi, confidence=0.9)


def _score_step(bpm):
    """The score's snap step: the sixteenth (a quarter of a beat)."""
    return BeatGrid(bpm).step_seconds(per_beat=4.0)


def _midi_bytes(notes, grid, quantize_s):
    buf = io.BytesIO()
    build_midi(notes, grid, None, quantize_s=quantize_s).save(file=buf)
    return buf.getvalue()


# ------------------------------------------------ seconds <-> beats, round-trip
@pytest.mark.parametrize("bpm", BPM)
@pytest.mark.parametrize("beats,figure", [(4.0, "ronde"), (2.0, "blanche"),
                                         (1.0, "noire"), (0.5, "croche")])
def test_a_note_of_k_beats_is_labeled_k(bpm, beats, figure):
    """A duration of k beats carries the k-beat figure, at any tempo: that is
    the seconds → beats conversion, seen by the figures consumer."""
    notes = [_note(60, 0.0, beats * 60.0 / bpm)]
    label_notes(notes, BeatGrid(bpm))
    assert notes[0].figure == figure


@pytest.mark.parametrize("bpm", BPM)
@pytest.mark.parametrize("meter,beats", [(Meter(4, 4), 4.0), (Meter(3, 4), 3.0),
                                        (Meter(6, 8), 3.0)])
def test_a_measure_lasts_its_quarter_beats(bpm, meter, beats):
    """A measure lasts its number of quarter beats at any tempo — a 6/8, three."""
    assert BeatGrid(bpm, meter).measure_seconds == pytest.approx(beats * 60.0 / bpm)


@pytest.mark.parametrize("bpm", BPM)
def test_a_sixteenth_is_a_quarter_of_a_beat(bpm):
    """The quantization step is a divided beat, not a number of seconds."""
    assert BeatGrid(bpm).step_seconds(per_beat=4.0) * bpm / 60.0 == pytest.approx(0.25)
    assert BeatGrid(bpm).step_seconds(per_beat=1.0) * bpm / 60.0 == pytest.approx(1.0)


# ------------------------------------------------------------- positions
@pytest.mark.parametrize("meter", [None, Meter(3, 4), Meter(4, 4), Meter(6, 8)])
def test_positions_rebuild_the_note_start_and_never_go_back(meter):
    """(measure, beat) suffices to recover the start: the position is a
    conversion, not decoration — and the notes' order is kept."""
    bpm = 120.0
    grid = BeatGrid(bpm, meter)
    measure_s, slot_s = grid.measure_seconds, grid.slot_seconds
    slots = grid.meter.n_slots
    notes = [_note(60, t, 0.4) for t in (0.0, 1.7, 2.0, 3.3, 5.5)]

    positions = grid.positions(notes)
    assert [m for m, _ in positions] == sorted(m for m, _ in positions)
    for note, (measure, beat) in zip(notes, positions):
        assert 1.0 <= beat <= slots + 1.0
        rebuilt = (measure - 1) * measure_s + (beat - 1) * slot_s
        assert rebuilt == pytest.approx(note.start, abs=0.005 * slot_s)


def test_a_note_on_a_boundary_opens_the_next_measure():
    grid = BeatGrid(120.0, Meter(4, 4))
    measure_s = grid.measure_seconds
    notes = [_note(60, 0.0, 0.5), _note(62, measure_s, 0.5), _note(64, 2 * measure_s, 0.5)]
    assert grid.positions(notes) == [(1, 1.0), (2, 1.0), (3, 1.0)]
    assert grid.measures(notes) == 3


def test_measures_cover_the_last_note_that_overhangs():
    grid = BeatGrid(120.0, Meter(4, 4))
    notes = [_note(60, 0.0, 0.5), _note(62, grid.measure_seconds - 0.1, 0.4)]
    assert grid.measures(notes) == 2  # overhangs 0.3 s into the 2nd


# ------------------------------------------------------------- snapping
def test_snapping_is_idempotent_and_lands_on_the_grid():
    """Snapping twice is snapping once: that is what lets a consumer use an
    already-aligned result without moving it."""
    bpm = 171.02
    step = _score_step(bpm)
    notes = [_note(60, 0.0, 0.31), _note(62, 0.37, 0.19), _note(64, 1.29, 0.44)]

    once = BeatGrid(bpm).snap_notes(notes)
    twice = BeatGrid(bpm).snap_notes(once)
    assert [(n.start, n.end) for n in twice] == [(n.start, n.end) for n in once]
    for snapped in (once, twice):
        for n in snapped:
            assert n.start == pytest.approx(round(n.start / step) * step)
            assert n.end == pytest.approx(round(n.end / step) * step)


def test_snapping_never_moves_a_boundary_more_than_one_division():
    """A snap that jumps a whole division would be a shift, not a rounding:
    swept over pseudo-random starts and durations."""
    bpm = 171.02
    step = _score_step(bpm)
    rng = np.random.default_rng(0)
    notes = [
        _note(60, float(t), float(d))
        for t, d in zip(rng.uniform(0.0, 12.0, 200), rng.uniform(0.08, 1.2, 200))
    ]
    for note, snapped in zip(notes, BeatGrid(bpm).snap_notes(notes)):
        assert abs(snapped.start - note.start) <= step / 2 + 1e-9
        assert abs(snapped.end - note.end) <= step + 1e-9


def test_midi_quantization_leaves_an_already_aligned_note_alone():
    """The quantization step must move nothing on a note already on its
    grid: quantizing or not then gives the same file."""
    grid = BeatGrid(120.0)
    step = _score_step(grid.bpm)
    notes = [_note(60, 0.0, 4 * step), _note(62, 2 * step, step), _note(64, 5 * step, 2 * step)]
    assert _midi_bytes(notes, grid, quantize_s=step) == _midi_bytes(notes, grid, quantize_s=None)


# --------------------------------------------------- written durations
@pytest.mark.parametrize("bpm", BPM)
def test_a_duration_on_a_written_value_is_written_as_is(bpm):
    """The nearest neighbour of an already-writable duration is itself: the
    bound rounds without drifting, at any tempo."""
    grid = BeatGrid(bpm)
    for figure in CATALOGUE:
        if figure.tuplet is None and not figure.export_only:
            assert grid.written(grid.seconds(figure.beats)) == figure


def test_at_equal_distance_the_plain_figure_wins():
    """The catalogue's order breaks the ties: 1.25 beats sits 0.25 from a
    quarter as from a dotted quarter, and 5 beats 1 from a whole as from a
    dotted whole; both are written as plain figures."""
    grid = BeatGrid(120.0)
    assert grid.written(grid.seconds(1.25)).name == "noire"
    assert grid.written(grid.seconds(5.0)).name == "ronde"


def test_a_duration_exactly_on_a_dotted_value_gets_it():
    """The dotted sixteenth (0.375 beats) is a catalogue value: a duration
    landing exactly on it is no longer folded onto the neighbouring eighth."""
    grid = BeatGrid(120.0)
    assert grid.written(grid.seconds(0.375)).name == "double-croche pointée"


@pytest.mark.parametrize("bpm", BPM)
def test_a_written_duration_stays_within_the_catalogue(bpm):
    """Nothing shorter than a sixteenth, nothing longer than a dotted whole
    — null, negative, huge and undecidable durations included."""
    grid = BeatGrid(bpm)
    for seconds in (-3.0, 0.0, 1e-9, 0.2, 137.0, float("nan")):
        figure = grid.written(seconds)
        assert 0.25 <= figure.beats <= 6.0
        assert figure.dots in (0, 1)


def test_a_longer_duration_is_never_written_shorter():
    """The written value grows with the sung duration: swept over 300 durations."""
    grid = BeatGrid(171.02)
    beats = [grid.written(grid.seconds(b)).beats for b in np.linspace(0.0, 6.5, 300)]
    assert beats == sorted(beats)


def test_the_staff_takes_its_fallback_duration_from_the_grid():
    """The Staff no longer re-rounds at home: the written duration of a note
    without a figure is exactly what the Grid announces."""
    grid = BeatGrid(171.02)
    for dur_s in (100.0, 1.1, 0.5, 0.35, 0.01):
        w = notated_duration(_note(60, 0.0, dur_s), grid)
        assert w == grid.written(dur_s)
