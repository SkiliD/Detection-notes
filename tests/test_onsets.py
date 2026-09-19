"""Tests onsets.py : attaques hors ligne et en flux.

Ce que ces tests épinglent, ce sont les deux erreurs mesurées : le détecteur
comptait 19 « attaques » sur une seule note tenue avec vibrato (le vibrato
déplace les harmoniques, donc le flux), et un niveau-seul ne détectait rien
du tout sur une ligne legato (l'amplitude ne bouge pas d'une note à l'autre).
"""

import numpy as np
import pytest
from conftest import SR
from voicetruth import SungNote, sequence, synthesize

from v2m.core.onsets import AttackTracker, detect_attacks

CONSONANT = 0.05
BLOCK = 1024


def _held(midi: int = 57, **kw) -> np.ndarray:
    """Une note tenue de 2 s, chantée à partir de 0,2 s."""
    return synthesize([SungNote(midi, 0.2, 1.8)], **kw)


def _stream_attacks(y: np.ndarray, sr: int = SR, block: int = BLOCK) -> list[float]:
    """Fait passer le signal dans le tracker temps réel (comme le fait le GUI)."""
    tracker = AttackTracker()
    out: list[float] = []
    for i in range(0, len(y) - block, block):
        t = (i + block) / sr
        if tracker.push(t, y[i : i + block]):
            out.append(round(t, 3))
    return out


# --------------------------------------------------------------------------- #
# Hors ligne
# --------------------------------------------------------------------------- #


def test_a_held_note_with_vibrato_gives_one_attack():
    """Régression : 19 attaques (une par cycle de vibrato) au lieu d'une."""
    attacks = detect_attacks(_held(vibrato_st=0.4, consonant_s=CONSONANT), SR)
    assert len(attacks) == 1
    assert attacks.times[0] == pytest.approx(0.2, abs=0.08)


def test_a_held_note_without_consonant_still_has_its_onset():
    attacks = detect_attacks(_held(), SR)
    assert len(attacks) == 1
    assert attacks.near(0.2, 0.08) is not None


def test_every_sung_onset_gets_an_attack():
    truth = sequence([60, 62, 64, 65, 67, 72], dur_s=0.6, start_s=0.2)
    y = synthesize(truth, glide_s=0.09, consonant_s=CONSONANT)
    attacks = detect_attacks(y, SR)
    for note in truth:
        assert attacks.near(note.start_s, 0.10) is not None


def test_no_attack_falls_inside_a_sung_note():
    """Un faux positif DANS une note est ce qui fabrique des micro-notes."""
    truth = sequence([60, 62, 64, 65, 67, 72], dur_s=0.6, start_s=0.2)
    y = synthesize(truth, glide_s=0.09, consonant_s=CONSONANT, vibrato_st=0.3)
    attacks = detect_attacks(y, SR)
    inside = [
        t
        for t in attacks.times
        if any(n.start_s + 0.10 < t < n.start_s + n.dur_s for n in truth)
    ]
    assert inside == []


def test_silence_has_no_attack():
    assert len(detect_attacks(np.zeros(SR, dtype=np.float32), SR)) == 0


def test_strengths_are_normalized_and_ordered():
    truth = sequence([60, 62, 64, 65], dur_s=0.5, start_s=0.2)
    attacks = detect_attacks(synthesize(truth, consonant_s=CONSONANT), SR)
    assert len(attacks) >= 2
    assert np.all(np.diff(attacks.times) > 0)  # chronologique
    assert attacks.strengths.max() == pytest.approx(1.0)
    assert np.all((attacks.strengths > 0) & (attacks.strengths <= 1.0))


# --------------------------------------------------------------------------- #
# En flux (temps réel)
# --------------------------------------------------------------------------- #


def test_streaming_tracker_fires_on_each_articulated_note():
    """Le tracker live suit le NIVEAU : chaque note reprise après un silence
    (ou relancée plus fort) est une attaque."""
    truth = sequence([60, 62, 64, 65], dur_s=0.5, gap_s=0.2, start_s=0.2)
    y = synthesize(truth, glide_s=0.0, consonant_s=CONSONANT)
    hits = _stream_attacks(y)
    for note in truth:
        assert any(abs(h - note.start_s) <= 0.14 for h in hits), note


def test_streaming_tracker_stays_quiet_inside_a_held_vibrato_note():
    """Le test que le flux spectral échouait : un vibrato n'est pas une attaque.

    L'attaque de la note elle-même est légitime ; c'est la suite qui doit
    rester muette (le détecteur en trouvait 11 dans la tenue).
    """
    hits = _stream_attacks(_held(vibrato_st=0.4, consonant_s=CONSONANT))
    assert all(abs(h - 0.2) <= 0.14 for h in hits), hits


def test_streaming_tracker_is_conservative_on_legato():
    """Une ligne legato à niveau constant ne déclenche rien après la première
    note (choix assumé : mieux vaut une attaque manquée qu'un faux positif
    qui couperait une note tenue en deux)."""
    truth = sequence([60, 62, 64, 65, 67, 72], dur_s=0.6, start_s=0.2)
    y = synthesize(truth, glide_s=0.09, consonant_s=CONSONANT)
    hits = _stream_attacks(y)
    assert all(abs(h - 0.2) <= 0.14 for h in hits), hits


def test_streaming_tracker_does_not_fire_on_silence():
    assert _stream_attacks(np.zeros(SR, dtype=np.float32)) == []


def test_streaming_tracker_primes_on_the_first_block():
    """La première trame amorce la ligne de base au lieu de déclencher."""
    tracker = AttackTracker()
    block = np.full(BLOCK, 0.25, dtype=np.float32)  # déjà fort
    assert tracker.push(0.05, block) is False
    assert tracker.baseline_rms > 0
    tracker.reset()
    assert tracker.baseline_rms == 0
    assert tracker.push(0.05, block) is False  # de nouveau l'amorçage


def test_streaming_tracker_keeps_a_minimum_gap():
    """Deux blocs forts d'affilée ne comptent que pour une attaque."""
    tracker = AttackTracker()
    quiet = np.full(BLOCK, 0.002, dtype=np.float32)
    loud = np.full(BLOCK, 0.25, dtype=np.float32)
    for _ in range(6):
        tracker.push(0.0, quiet)
    assert tracker.push(0.1, loud) is True
    assert tracker.push(0.11, loud) is False  # < min_gap_s
