"""Propriétés de la Grille, écrites AVANT que la Grille n'existe.

Le référentiel métrique — conversion secondes ↔ temps, position en mesure,
arrimage sur la grille — est aujourd'hui recopié dans sept modules. Ces tests
ne décrivent pas cette dispersion : ils fixent ce que tout consommateur doit
continuer de faire quand elle sera un seul module (`grid.BeatGrid`, voir
CONTEXT.md « Grille »). C'est le contrat de non-régression du refactor : chaque
propriété se relira ensuite à travers la grille sans changer d'énoncé.

Deux partis pris :

- les propriétés passent par les interfaces PUBLIQUES des consommateurs
  (figures, meter, musicxml, midi_export), jamais par une recopie de leur
  arithmétique dans le test ;
- elles tiennent à PLUSIEURS tempos : c'est l'invariance par tempo qui est la
  propriété, pas un résultat à 120 BPM. Un `171,02 BPM` traîne partout ici
  parce que c'est le tempo réellement détecté sur `chant.wav`.

L'arrimage d'une prise est un membre de la Grille (`BeatGrid.snap_notes`) :
les propriétés d'arrimage ci-dessous l'interrogent donc directement, alors
qu'elles passaient par une fonction privée de l'export. Ce que l'export en
fait reste tenu par le filet, `test_reference_artifacts.py`. La dernière
propriété fait de même pour `BeatGrid.written`, le choix de la valeur écrite
d'une durée sans figure : elle interroge la Grille ET la Portée, qui doit s'en
servir au lieu de réarrondir chez elle.
"""

import io

import numpy as np
import pytest
from v2m.core.figures import label_notes
from v2m.core.grid import CATALOGUE, BeatGrid, Meter
from v2m.core.midi_export import build_midi
from v2m.core.notation import notated_duration
from v2m.core.notes import Note

BPM = (60.0, 120.0, 171.02)  # dont le tempo réellement détecté sur chant.wav


def _note(midi, start, dur_s):
    return Note(start=start, end=start + dur_s, midi=midi, confidence=0.9)


def _score_step(bpm):
    """Pas d'arrimage de la partition : le seizième (un quart de temps)."""
    return BeatGrid(bpm).step_seconds(per_beat=4.0)


def _midi_bytes(notes, grid, quantize_s):
    buf = io.BytesIO()
    build_midi(notes, grid, None, quantize_s=quantize_s).save(file=buf)
    return buf.getvalue()


# ------------------------------------------------ secondes ↔ temps, aller-retour
@pytest.mark.parametrize("bpm", BPM)
@pytest.mark.parametrize("beats,figure", [(4.0, "ronde"), (2.0, "blanche"),
                                         (1.0, "noire"), (0.5, "croche")])
def test_a_note_of_k_beats_is_labeled_k(bpm, beats, figure):
    """Une durée de k temps porte la figure de k temps, à tout tempo : c'est la
    conversion secondes → temps, vue par le consommateur des figures."""
    notes = [_note(60, 0.0, beats * 60.0 / bpm)]
    label_notes(notes, BeatGrid(bpm))
    assert notes[0].figure == figure


@pytest.mark.parametrize("bpm", BPM)
@pytest.mark.parametrize("meter,beats", [(Meter(4, 4), 4.0), (Meter(3, 4), 3.0),
                                        (Meter(6, 8), 3.0)])
def test_a_measure_lasts_its_quarter_beats(bpm, meter, beats):
    """Une mesure dure son nombre de noires à tout tempo — un 6/8, trois."""
    assert BeatGrid(bpm, meter).measure_seconds == pytest.approx(beats * 60.0 / bpm)


@pytest.mark.parametrize("bpm", BPM)
def test_a_sixteenth_is_a_quarter_of_a_beat(bpm):
    """Le pas de quantification est un temps divisé, pas un nombre de secondes."""
    assert BeatGrid(bpm).step_seconds(per_beat=4.0) * bpm / 60.0 == pytest.approx(0.25)
    assert BeatGrid(bpm).step_seconds(per_beat=1.0) * bpm / 60.0 == pytest.approx(1.0)


# ------------------------------------------------------------- positions
@pytest.mark.parametrize("meter", [None, Meter(3, 4), Meter(4, 4), Meter(6, 8)])
def test_positions_rebuild_the_note_start_and_never_go_back(meter):
    """(mesure, temps) suffit à retrouver le début : la position est une
    conversion, pas un décor — et l'ordre des notes est conservé."""
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
    assert grid.measures(notes) == 2  # déborde de 0,3 s dans la 2e


# ------------------------------------------------------------- arrimage
def test_snapping_is_idempotent_and_lands_on_the_grid():
    """Arrimer deux fois, c'est arrimer une fois : c'est ce qui autorise un
    consommateur à se servir d'un résultat déjà aligné sans le déplacer."""
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
    """Un arrimage qui saute une division entière serait un décalage, pas un
    arrondi : balayé sur des débuts et durées pseudo-aléatoires."""
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
    """Le pas de quantification ne doit rien déplacer sur une note déjà sur sa
    grille : quantifier ou non donne alors le même fichier."""
    grid = BeatGrid(120.0)
    step = _score_step(grid.bpm)
    notes = [_note(60, 0.0, 4 * step), _note(62, 2 * step, step), _note(64, 5 * step, 2 * step)]
    assert _midi_bytes(notes, grid, quantize_s=step) == _midi_bytes(notes, grid, quantize_s=None)


# --------------------------------------------------- durées écrites
@pytest.mark.parametrize("bpm", BPM)
def test_a_duration_on_a_written_value_is_written_as_is(bpm):
    """Le plus proche voisin d'une durée déjà écrivable, c'est elle-même : la
    borne arrondit sans dériver, à tout tempo."""
    grid = BeatGrid(bpm)
    for figure in CATALOGUE:
        if figure.tuplet is None and not figure.export_only:
            assert grid.written(grid.seconds(figure.beats)) == figure


def test_at_equal_distance_the_plain_figure_wins():
    """L'ordre du catalogue départage les égalités : 1,25 temps est à 0,25
    d'une noire comme d'une noire pointée, et 5 temps à 1 d'une ronde comme
    d'une ronde pointée ; les deux s'écrivent en figure simple."""
    grid = BeatGrid(120.0)
    assert grid.written(grid.seconds(1.25)).name == "noire"
    assert grid.written(grid.seconds(5.0)).name == "ronde"


def test_a_duration_exactly_on_a_dotted_value_gets_it():
    """La double-croche pointée (0,375 temps) est une valeur du catalogue : une
    durée qui tombe pile dessus n'est plus rabattue sur la croche voisine."""
    grid = BeatGrid(120.0)
    assert grid.written(grid.seconds(0.375)).name == "double-croche pointée"


@pytest.mark.parametrize("bpm", BPM)
def test_a_written_duration_stays_within_the_catalogue(bpm):
    """Rien de plus court qu'un seizième, rien de plus long qu'une ronde
    pointée — durée nulle, négative, énorme ou non décidable comprises."""
    grid = BeatGrid(bpm)
    for seconds in (-3.0, 0.0, 1e-9, 0.2, 137.0, float("nan")):
        figure = grid.written(seconds)
        assert 0.25 <= figure.beats <= 6.0
        assert figure.dots in (0, 1)


def test_a_longer_duration_is_never_written_shorter():
    """La valeur écrite croît avec la durée chantée : balayé sur 300 durées."""
    grid = BeatGrid(171.02)
    beats = [grid.written(grid.seconds(b)).beats for b in np.linspace(0.0, 6.5, 300)]
    assert beats == sorted(beats)


def test_the_staff_takes_its_fallback_duration_from_the_grid():
    """La Portée ne réarrondit plus chez elle : la durée écrite d'une note sans
    figure est exactement ce que la Grille annonce."""
    grid = BeatGrid(171.02)
    for dur_s in (100.0, 1.1, 0.5, 0.35, 0.01):
        w = notated_duration(_note(60, 0.0, dur_s), grid)
        assert w == grid.written(dur_s)
