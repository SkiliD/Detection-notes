"""Tests du segmenteur temps réel (hystérésis, attaques, transitions)."""

import numpy as np
import pytest

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


def test_out_of_piano_notes_are_never_kept():
    """Un piano n'a que 88 touches (A0..C8) : une hauteur en dehors n'entre ni
    dans la prise ni dans le preview — l'erreur d'octave reste visible, la note
    n'existe pas."""
    from v2m.core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI

    seg = LiveNoteSegmenter()
    _feed(seg, midi_to_hz(PIANO_MIN_MIDI - 3), 0.0, 0.30)   # sous A0
    _feed(seg, float("nan"), 0.30, 0.10)
    _feed(seg, midi_to_hz(PIANO_MAX_MIDI + 5), 0.40, 0.30)  # au-dessus de C8
    _feed(seg, float("nan"), 0.70, 0.10)
    _feed(seg, midi_to_hz(PIANO_MIN_MIDI), 0.80, 0.30)      # A0, la touche extrême
    seg.flush(1.15)
    assert [n.midi for n in seg.notes] == [PIANO_MIN_MIDI]

    seg2 = LiveNoteSegmenter()
    _feed(seg2, midi_to_hz(PIANO_MAX_MIDI + 5), 0.0, 0.30)
    assert seg2.active_note(0.25) is None


def test_active_note_shows_the_note_being_sung():
    """Pendant qu'une note est tenue, le preview la décrit sans la stocker."""
    seg = LiveNoteSegmenter()
    _feed(seg, midi_to_hz(69), 0.0, 0.20)
    assert seg.notes == []  # rien n'est stocké avant la fin de la note
    prev = seg.active_note(0.25)
    assert prev is not None
    assert prev.midi == 69
    assert prev.start == pytest.approx(0.0, abs=0.05)
    assert prev.end == pytest.approx(0.25)
    # toujours pas stockée : le preview est jetable
    assert seg.notes == []


def test_active_note_none_when_idle():
    seg = LiveNoteSegmenter()
    assert seg.active_note(0.1) is None
    _feed(seg, float("nan"), 0.0, 0.20)
    assert seg.active_note(0.25) is None
    _feed(seg, midi_to_hz(60), 0.30, 0.30)
    seg.flush(0.65)
    assert seg.active_note(0.70) is None  # note closed -> plus rien en cours


def test_active_note_end_never_before_start():
    """Un tick d'affichage en retard ne doit pas produire une barre inversée."""
    seg = LiveNoteSegmenter()
    _feed(seg, midi_to_hz(69), 0.50, 0.20)
    prev = seg.active_note(0.10)
    assert prev is not None and prev.end > prev.start


def test_nan_ignored():
    seg = LiveNoteSegmenter()
    _feed(seg, float("nan"), 0.0, 0.30)
    seg.flush(0.30)
    assert seg.notes == []


# --------------------------------------------------------------------------- #
# Attaques et transitions (voir notes.py pour les mêmes règles hors ligne)
# --------------------------------------------------------------------------- #


def _feed_glide(
    seg,
    t0: float,
    glide_s: float,
    hold_s: float,
    start: int,
    target: int,
    step: float = 0.02,
    attack_first: bool = False,
):
    """Portamento de `start` vers `target` pendant `glide_s`, puis tenue."""
    n = int(glide_s / step)
    for i in range(n):
        w = (i + 1) / n
        attack = attack_first and i == 0
        seg.push(t0 + (i + 1) * step, midi_to_hz(start + w * (target - start)), attack=attack)
    t = t0 + glide_s
    while t < t0 + glide_s + hold_s:
        t += step
        seg.push(t, midi_to_hz(target))
    return t


def test_a_glide_does_not_become_a_note():
    """Un portamento ne doit être confirmé qu'à l'arrivée, une seule fois.

    Régression mesurée : avec une fenêtre de confirmation de 80 ms (deux
    trames), un pas de l'échelle passait pour une note tenue et un glissando
    de 60 vers 62 fabriquait des notes à 61 et 63.

    Limite assumée : un portamento PLUS LENT que ~3 demi-tons par seconde
    (0,6 s pour 2 demi-tons) est transcrit en deux notes, puisqu'aucune
    fenêtre de 140 ms ne permet alors de distinguer la glisse de la tenue.
    """
    seg = LiveNoteSegmenter()
    t = _feed_glide(seg, 0.0, 0.12, 0.6, 60, 62)
    seg.flush(t + 0.2)
    assert [n.midi for n in seg.notes] == [62]


def test_attack_dates_the_onset_of_a_note():
    """Avec l'attaque, la note commence au portamento — pas à son arrivée."""
    without = LiveNoteSegmenter()
    t = _feed_glide(without, 0.0, 0.4, 0.4, 60, 67)
    without.flush(t + 0.2)

    with_attack = LiveNoteSegmenter()
    t = _feed_glide(with_attack, 0.0, 0.4, 0.4, 60, 67, attack_first=True)
    with_attack.flush(t + 0.2)

    assert [n.midi for n in with_attack.notes] == [67]
    assert with_attack.notes[0].start == pytest.approx(0.0, abs=0.03)
    # sans preuve d'attaque la note ne peut être datée qu'à l'arrivée de la
    # hauteur : c'est le repli documenté du temps réel
    assert [n.midi for n in without.notes] == [67]
    assert without.notes[0].start > 0.35


def test_re_attack_splits_a_held_note():
    """Même hauteur relancée plus fort : deux notes, pas une tenue."""
    seg = LiveNoteSegmenter()
    t = 0.0
    while t < 0.4:
        t += 0.02
        seg.push(t, midi_to_hz(64))
    seg.push(t + 0.02, midi_to_hz(64), attack=True)
    t += 0.02
    while t < 0.9:
        t += 0.02
        seg.push(t, midi_to_hz(64))
    seg.flush(1.0)
    assert [n.midi for n in seg.notes] == [64, 64]


def test_a_fresh_note_is_not_split_by_a_late_attack():
    """Une attaque parasite juste après l'onset ne coupe pas la note"""
    seg = LiveNoteSegmenter()
    t = 0.0
    while t < 0.5:
        t += 0.02
        seg.push(t, midi_to_hz(60), attack=(t > 0.06 and t < 0.08))
    seg.flush(0.6)
    assert len(seg.notes) == 1
