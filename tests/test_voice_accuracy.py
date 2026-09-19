"""Justesse de la transcription : chaque note chantée doit être retrouvée.

Le harnais `voicetruth` synthétise une voix dont on connaît toutes les notes
(portamentos, consonnes, vibrato, bruit) et compare ce que la chaîne détecte à
cette vérité. Ces tests fixent les seuils de non-régression :

- la segmentation HORS LIGNE est exacte sur les 19 cas — avant la détection
  d'attaques et les règles de transition, un glissando de 90 ms transformait
  6 notes chantées en 10, et une tenue avec vibrato comptait 19 « attaques » ;
- la preuve d'attaque ne dégrade jamais la géométrie seule ;
- aucune note n'est inventée À L'INTÉRIEUR d'une note chantée, le défaut qui
  rendait la partition illogique ;
- le chemin temps réel, sans regard en arrière, retrouve exactement les
  phrases articulées.

Limites connues du temps réel (documentées dans AGENTS.md, non testées ici) :
les doubles-croches (120 ms) sont fusionnées, et un portamento très long
(250 ms) décale l'attaque de la note d'arrivée.
"""

from functools import lru_cache

import numpy as np
import pytest
from conftest import SR
from voicetruth import Accuracy, SungNote, detect_notes, measure, sequence, synthesize
from v2m.core.liveseg import LivePitchStream
from v2m.core.notes import Note
from v2m.core.pipeline import analyze_signal

MIDIS = [60, 62, 64, 65, 67, 72]
GAMME = [60, 62, 64, 65, 67, 69, 71, 72]
BLOCK = 1024
GATE = 0.008

# nom -> (notes chantées, réglages du synthétiseur)
CASES: dict[str, tuple[list[SungNote], dict]] = {
    "consonnes": (sequence(MIDIS, dur_s=0.6, start_s=0.2), dict(glide_s=0.09, consonant_s=0.05)),
    "legato": (sequence(MIDIS, dur_s=0.6, start_s=0.2), dict(glide_s=0.09, consonant_s=0.0)),
    "détaché": (sequence(MIDIS, dur_s=0.6, gap_s=0.15, start_s=0.2), dict(glide_s=0.09, consonant_s=0.05)),
    "silences longs": (
        sequence([60, 64, 67], dur_s=0.5, gap_s=0.5, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05),
    ),
    "vibrato": (
        sequence(MIDIS, dur_s=0.6, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05, vibrato_st=0.3),
    ),
    "bruit": (
        sequence(MIDIS, dur_s=0.6, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05, noise=0.03),
    ),
    "bruit et vibrato": (
        sequence(MIDIS, dur_s=0.6, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05, vibrato_st=0.3, noise=0.03),
    ),
    "croches": (sequence(GAMME, dur_s=0.2, start_s=0.2), dict(glide_s=0.03, consonant_s=0.04)),
    "doubles": (sequence([60, 62, 64, 65], dur_s=0.12, start_s=0.2), dict(glide_s=0.02, consonant_s=0.03)),
    "note courte": (
        [SungNote(60, 0.2, 0.6), SungNote(67, 0.8, 0.15), SungNote(72, 0.95, 0.6)],
        dict(glide_s=0.03, consonant_s=0.05),
    ),
    "glissando long": (
        [SungNote(60, 0.2, 0.8), SungNote(67, 1.0, 0.8)],
        dict(glide_s=0.25, consonant_s=0.05),
    ),
    "tenue vibrato": ([SungNote(57, 0.2, 1.5)], dict(vibrato_st=0.4, consonant_s=0.05)),
    "voix grave": (
        sequence([45, 47, 48, 50, 52], dur_s=0.8, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05),
    ),
    "voix haute": (
        sequence([69, 71, 72, 74, 76], dur_s=0.8, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05),
    ),
    "gamme liée": (sequence(GAMME, dur_s=0.4, start_s=0.2), dict(glide_s=0.06, consonant_s=0.04)),
    "descendante liée": (
        sequence(list(reversed(GAMME)), dur_s=0.4, start_s=0.2),
        dict(glide_s=0.06, consonant_s=0.04),
    ),
    "saut d'octave": (
        [SungNote(60, 0.2, 0.6), SungNote(72, 0.8, 0.6)],
        dict(glide_s=0.05, consonant_s=0.05),
    ),
    "répétition": ([SungNote(64, 0.2, 0.7), SungNote(64, 1.2, 0.7)], dict(consonant_s=0.05)),
}

# le temps réel retrouve exactement ces phrases-là
LIVE_CASES = [
    "consonnes",
    "legato",
    "détaché",
    "silences longs",
    "croches",
    "gamme liée",
    "descendante liée",
    "voix grave",
    "voix haute",
    "bruit et vibrato",
    "saut d'octave",
    "répétition",
]
LIVE_TOL_S = 0.20

# le temps réel n'a pas de regard en arrière : ces cas extrêmes (doubles
# croches de 120 ms, portamento de 250 ms, note de 150 ms précédée d'un
# portamento) sont fusionnés ou décalés — jamais faux en hauteur.
LIVE_KNOWN_LIMITS = {"doubles", "glissando long", "note courte"}


@lru_cache(maxsize=None)
def _signal(name: str) -> np.ndarray:
    truth, kwargs = CASES[name]
    return synthesize(truth, **kwargs)


@lru_cache(maxsize=None)
def _offline(name: str, with_attacks: bool) -> Accuracy:
    truth, _ = CASES[name]
    return measure(detect_notes(_signal(name), SR, with_attacks=with_attacks), truth)


@lru_cache(maxsize=None)
def _live(name: str) -> Accuracy:
    truth, _ = CASES[name]
    stream = LivePitchStream(SR, gate_rms=GATE)
    y = _signal(name)
    for i in range(0, len(y) - BLOCK, BLOCK):
        stream.feed(y[i : i + BLOCK], (i + BLOCK) / SR)
    stream.seg.flush(len(y) / SR)
    return measure(stream.seg.notes, truth, tol_s=LIVE_TOL_S)


# --------------------------------------------------------------------------- #
# Le harnais lui-même
# --------------------------------------------------------------------------- #


def test_the_harness_catches_a_lost_note():
    """Sans quoi les tests ci-dessous ne prouveraient rien."""
    truth, _ = CASES["consonnes"]
    acc = measure([], truth)
    assert acc.matched == 0 and acc.missing == len(truth) and not acc.exact


def test_the_harness_catches_a_note_in_the_middle_of_another():
    """Le défaut d'origine : une micro-note coincée dans une note chantée."""
    truth, _ = CASES["tenue vibrato"]
    detected = [
        Note(start=truth[0].start_s, end=truth[0].start_s + truth[0].dur_s, midi=truth[0].midi, confidence=1.0),
        Note(start=truth[0].start_s + 0.5, end=truth[0].start_s + 0.58, midi=truth[0].midi + 1, confidence=1.0),
    ]
    acc = measure(detected, truth)
    assert acc.matched == 1 and acc.extra == 1 and acc.intruders == 1
    assert not acc.exact


# --------------------------------------------------------------------------- #
# Hors ligne
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(CASES))
def test_offline_transcription_is_exact(name: str):
    acc = _offline(name, True)
    assert acc.exact, acc.line()


@pytest.mark.parametrize("name", sorted(CASES))
def test_attack_evidence_never_makes_it_worse(name: str):
    with_attacks = _offline(name, True)
    geometry = _offline(name, False)
    assert with_attacks.matched >= geometry.matched
    assert with_attacks.extra <= geometry.extra
    assert with_attacks.intruders == 0


@pytest.mark.parametrize("name", ["legato", "consonnes", "glissando long", "saut d'octave", "répétition"])
def test_geometry_alone_is_still_strict(name: str):
    """Sans preuve d'attaque, la règle géométrique du glissando doit suffire
    partout où elle est sans ambiguïté (le pipeline fournit toujours des
    attaques, mais un appelant peut ne pas en avoir)."""
    acc = _offline(name, False)
    assert acc.exact, acc.line()


def test_onsets_and_pitches_are_close():
    for name in ("consonnes", "gamme liée", "croches", "voix grave"):
        acc = _offline(name, True)
        assert acc.worst_start_ms <= 80.0, (name, acc.line())  # < ATTACK_TOL_S
        assert acc.worst_pitch_st == 0.0, (name, acc.line())


# --------------------------------------------------------------------------- #
# Temps réel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", LIVE_CASES)
def test_live_chain_reads_the_sung_phrase(name: str):
    acc = _live(name)
    assert acc.exact, acc.line()
    assert acc.worst_start_ms <= 120.0, acc.line()


def test_live_chain_never_invents_notes_inside_a_sung_note():
    for name in CASES:
        if name in LIVE_KNOWN_LIMITS:
            continue  # voir test_live_limits_are_bounded
        assert _live(name).intruders == 0, name


def test_live_limits_are_bounded():
    """Caractérise les limites assumées du temps réel, pour qu'elles restent
    bornées : une note perdue ou fusionnée, jamais une hauteur inventée."""
    for name in LIVE_KNOWN_LIMITS:
        truth, _ = CASES[name]
        stream = LivePitchStream(SR, gate_rms=GATE)
        y = _signal(name)
        for i in range(0, len(y) - BLOCK, BLOCK):
            stream.feed(y[i : i + BLOCK], (i + BLOCK) / SR)
        stream.seg.flush(len(y) / SR)
        sung = {n.midi for n in truth}
        assert len(stream.seg.notes) <= len(truth), name
        assert all(n.midi in sung for n in stream.seg.notes), name


# --------------------------------------------------------------------------- #
# Respirations : phrases articulées -> silences dans le modèle
# --------------------------------------------------------------------------- #


def test_breaths_between_phrases_become_rests():
    """« Détaché » = 6 notes en 6 phrases ; chaque respiration doit devenir
    un silence (l'intro incluse) et aucune ne peut chevaucher une note.
    Les figures sont soit None (gap trop court pour une figure à ce tempo,
    la portée l'engrave quand même d'après sa durée), soit du catalogue."""
    y = _signal("détaché")
    pipeline = analyze_signal(y, SR)
    rests = pipeline.rests
    assert len(rests) >= 5, [(r.start, r.end, r.figure) for r in rests]
    catalog = {"pause", "demi-pause", "soupir", "demi-soupir", "quart de soupir"}
    catalog |= {f"{f} pointée" for f in catalog}
    for r in rests:
        assert r.figure is None or r.figure in catalog, r.figure
    # aucune respiration ne chevauche une note
    notes = pipeline.notes
    for r in rests:
        for n in notes:
            assert r.end <= n.start + 0.02 or r.start >= n.end - 0.02, (r, n)


def test_long_breath_is_labeled_a_quarter_rest():
    """Une vraie respiration (0,6 s à 80 BPM = 0,8 temps) doit être étiquetée
    « soupir » — c'est la figure qu'un éditeur grave entre deux phrases."""
    y = synthesize(sequence([60, 62, 64], dur_s=0.6, gap_s=0.6, start_s=0.2),
                   glide_s=0.09, consonant_s=0.05)
    rests = analyze_signal(y, SR).rests
    figures = [r.figure for r in rests if r.figure]
    assert "soupir" in figures, [(r.start, r.end, r.figure) for r in rests]


def test_rearticulated_note_on_same_pitch_is_split():
    """Deux « la » séparés par 150 ms de silence : une note unique trahirait
    une attaque avalée — la chaîne doit les séparer (comportement live)."""
    from voicetruth import synthesize as _syn

    truth = [
        SungNote(start_s=0.2, dur_s=0.6, midi=60),
        SungNote(start_s=0.95, dur_s=0.6, midi=60),
    ]
    y = _syn(truth, glide_s=0.09, consonant_s=0.05)
    detected = detect_notes(y, SR)
    acc = measure(detected, truth)
    assert acc.exact, acc.line()
