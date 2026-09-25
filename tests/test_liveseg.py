"""Tests du segmenteur temps réel (hystérésis, attaques, transitions)."""

import numpy as np
import pytest

from v2m.core.liveseg import LiveNoteSegmenter
from v2m.core.music import midi_to_hz


def _hz(midi) -> float:
    """midi_to_hz returns a 0-d array; the segmenter expects a plain float."""
    return float(midi_to_hz(midi))


def _feed(seg: LiveNoteSegmenter, hz: float, start: float, dur: float, step: float = 0.01):
    t = start
    while t < start + dur:
        seg.push(t, hz)
        t += step


def test_note_confirmed_and_closed():
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.30)
    seg.flush(0.35)
    assert len(seg.notes) == 1
    assert seg.notes[0].midi == 69
    assert seg.notes[0].duration >= 0.25


def test_two_separate_notes():
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.30)
    _feed(seg, float("nan"), 0.30, 0.20)   # silence
    _feed(seg, _hz(72), 0.50, 0.30)
    seg.flush(0.85)
    assert len(seg.notes) == 2
    assert [n.midi for n in seg.notes] == [69, 72]


def test_attack_strength_lands_on_the_closed_note():
    """La force de l'attaque voyage avec le flag et finit sur la note.

    Régression : `LiveNote` n'avait pas de champ `strength`, donc
    `detect_meter` (étape accents 3/4 vs 4/4) levait AttributeError sur la
    première prise live de six notes — et le motif du métronome restait 4/4.
    La force (0..1, normalisée sur l'attaque la plus forte de la prise)
    est l'équivalent vivant de `Note.strength` hors ligne.
    """
    seg = LiveNoteSegmenter()
    seg.push(0.0, _hz(69), attack=True, strength=1.0)
    t = 0.01
    while t < 0.30:
        seg.push(t, _hz(69))
        t += 0.01
    seg.flush(0.35)
    assert len(seg.notes) == 1
    assert seg.notes[0].strength == pytest.approx(1.0)
    # a second note opened WITHOUT a seen attack carries no accent
    seg.push(0.40, _hz(69))
    t = 0.41
    while t < 0.70:
        seg.push(t, _hz(69))
        t += 0.01
    seg.flush(0.75)
    assert len(seg.notes) == 2
    assert seg.notes[1].strength == 0.0


def test_staged_attack_keeps_its_strength():
    """Une attaque sur frame muette garde sa force jusqu'à la note.

    Régression : l'attaque était stage (son temps), mais la force voyait en
    paramètre du frame voisé suivant — où elle valait déjà 0.0. Sur une vraie
    prise chantée (consonnes sur frames muettes), tous les downbeats lisaient
    0.0 et la détection de mesure perdait ses accents.
    """
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(64), 0.0, 0.40)
    seg.push(0.41, float("nan"), attack=True, strength=1.0)  # consonne, muet
    _feed(seg, _hz(64), 0.45, 0.40)                          # la note re-attaquée
    seg.flush(0.90)
    assert len(seg.notes) == 2
    assert seg.notes[1].articulated
    assert seg.notes[1].strength == pytest.approx(1.0)


def test_short_blip_ignored():
    """Un parasite de 40 ms ne crée pas de note."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.40)
    _feed(seg, _hz(81), 0.40, 0.04)  # parasite
    _feed(seg, _hz(69), 0.44, 0.30)
    seg.flush(0.80)
    assert len(seg.notes) == 1
    assert seg.notes[0].midi == 69


def test_repetition_same_pitch_stays_two_notes():
    """Reprendre la même note (aa-aa) sans attaque vue: DEUX notes.

    Régression: le merge du flush recollait deux notes pleine longueur
    séparées d'un vrai silence (60 ms) dès que l'attaque n'avait pas été
    détectée — chaque répétition douce finissait en une note unique.
    """
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.30)
    _feed(seg, float("nan"), 0.30, 0.06)   # re-prise de la même note
    _feed(seg, _hz(69), 0.36, 0.30)
    seg.flush(0.70)
    assert len(seg.notes) == 2
    assert [n.duration for n in seg.notes] == pytest.approx([0.30, 0.30], abs=0.05)


def test_repetition_after_longer_gap_stays_two_notes():
    """Un silence d'une longueur de note entre deux répétitions est une phrase."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.30)
    _feed(seg, float("nan"), 0.30, 0.25)
    _feed(seg, _hz(69), 0.55, 0.30)
    seg.flush(0.90)
    assert len(seg.notes) == 2


def test_out_of_piano_notes_are_never_kept():
    """Un piano n'a que 88 touches (A0..C8) : une hauteur en dehors n'entre ni
    dans la prise ni dans le preview — l'erreur d'octave reste visible, la note
    n'existe pas."""
    from v2m.core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI

    seg = LiveNoteSegmenter()
    _feed(seg, _hz(PIANO_MIN_MIDI - 3), 0.0, 0.30)   # sous A0
    _feed(seg, float("nan"), 0.30, 0.10)
    _feed(seg, _hz(PIANO_MAX_MIDI + 5), 0.40, 0.30)  # au-dessus de C8
    _feed(seg, float("nan"), 0.70, 0.10)
    _feed(seg, _hz(PIANO_MIN_MIDI), 0.80, 0.30)      # A0, la touche extrême
    seg.flush(1.15)
    assert [n.midi for n in seg.notes] == [PIANO_MIN_MIDI]

    seg2 = LiveNoteSegmenter()
    _feed(seg2, _hz(PIANO_MAX_MIDI + 5), 0.0, 0.30)
    assert seg2.active_note(0.25) is None


def test_active_note_shows_the_note_being_sung():
    """Pendant qu'une note est tenue, le preview la décrit sans la stocker."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.0, 0.20)
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
    _feed(seg, _hz(60), 0.30, 0.30)
    seg.flush(0.65)
    assert seg.active_note(0.70) is None  # note closed -> plus rien en cours


def test_active_note_end_never_before_start():
    """Un tick d'affichage en retard ne doit pas produire une barre inversée."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(69), 0.50, 0.20)
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
        seg.push(t0 + (i + 1) * step, _hz(start + w * (target - start)), attack=attack)
    t = t0 + glide_s
    while t < t0 + glide_s + hold_s:
        t += step
        seg.push(t, _hz(target))
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
        seg.push(t, _hz(64))
    seg.push(t + 0.02, _hz(64), attack=True)
    t += 0.02
    while t < 0.9:
        t += 0.02
        seg.push(t, _hz(64))
    seg.flush(1.0)
    assert [n.midi for n in seg.notes] == [64, 64]


def test_repeated_same_pitch_through_the_full_chain():
    """« Ta-ta-ta-ta » sur un même Mi doit donner QUATRE notes (~0,25 s).

    Régression mesurée : le tracker de niveau ne revoyait qu'une attaque sur
    quatre (la baseline restait au niveau des notes précédentes), et les
    attaques tombées sur une frame muette étaient avalées — le merge du flush
    recollait alors tout en une note de 1,1 s. Chaîne complète (blocs 1024,
    comme le micro)."""
    from voicetruth import SungNote, synthesize
    from v2m.core.liveseg import LivePitchStream

    truth = [SungNote(64, 0.2, 0.25), SungNote(64, 0.5, 0.25),
             SungNote(64, 0.8, 0.25), SungNote(64, 1.1, 0.25)]
    y = synthesize(truth, consonant_s=0.05, glide_s=0.03)
    stream = LivePitchStream(22_050, gate_rms=0.008)
    for i in range(0, len(y) - 1024, 1024):
        stream.feed(y[i : i + 1024], (i + 1024) / 22_050)
    stream.seg.flush(len(y) / 22_050)
    notes = stream.seg.notes
    assert [n.midi for n in notes] == [64, 64, 64, 64]
    assert all(n.duration < 0.45 for n in notes)  # ni tenue unique ni fusion


def test_split_attack_survives_a_pending_mute_run():
    """L'attaque d'une répétition splitée survit au reset PENDING-sur-mute.

    Une split même hauteur (consonne en rafale) ouvre la répétition PENDING
    avec `pending_attack` = la coupure ; si le burst s'étend sur les frames
    suivantes, elles sont MUETTES et le reset PENDING-sur-mute effaçait
    l'attaque sans la mettre en attente — la répétition repartait datée à
    sa confirmation (60 ms de trou mesuré entre les deux moitiés) et sans
    preuve d'articulation, donc recollable par le merge du flush.
    """
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(64), 0.0, 0.40)              # note tenue (confirmée ~0.10)
    seg.push(0.41, float("nan"), attack=True)   # burst : la tenue est en rupture
    seg.push(0.42, float("nan"))                # (silence < MIN_GAP_HOLD_S)
    seg.push(0.43, _hz(64), attack=True)        # frame voisée : la split coupe ici
    seg.push(0.44, float("nan"))                # la queue du burst : PENDING muet
    seg.push(0.45, float("nan"))
    _feed(seg, _hz(64), 0.46, 0.40)             # la voyelle de la répétition
    seg.flush(0.90)
    assert [n.midi for n in seg.notes] == [64, 64]
    assert seg.notes[1].articulated
    assert seg.notes[1].start == pytest.approx(0.40, abs=0.02)
    assert seg.notes[1].start >= seg.notes[0].end - 0.02  # pas de trou


def test_staged_attack_dates_the_new_note():
    """Une attaque vue sur une frame muette (consonne sans voix) doit dater
    la note suivante : sinon `flush` fusionne la répétition comme un blip."""
    seg = LiveNoteSegmenter()
    _feed(seg, _hz(64), 0.0, 0.40)
    seg.push(0.41, float("nan"), attack=True)   # consonne sur frame muette
    _feed(seg, _hz(64), 0.45, 0.40)      # la note re-attaquée
    seg.flush(0.90)
    assert len(seg.notes) == 2
    assert seg.notes[1].articulated
    assert seg.notes[1].start == pytest.approx(0.41, abs=0.05)


def test_a_fresh_note_is_not_split_by_a_late_attack():
    """Une attaque parasite juste après l'onset ne coupe pas la note"""
    seg = LiveNoteSegmenter()
    t = 0.0
    while t < 0.5:
        t += 0.02
        seg.push(t, _hz(60), attack=(t > 0.06 and t < 0.08))
    seg.flush(0.6)
    assert len(seg.notes) == 1
