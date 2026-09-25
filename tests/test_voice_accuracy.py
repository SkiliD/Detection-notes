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
from pathlib import Path

import numpy as np
import pytest
from conftest import SR
from voicetruth import (
    Accuracy,
    SungNote,
    add_breath,
    detect_notes,
    measure,
    sequence,
    synthesize,
)
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
    "saut octave haut": (
        [SungNote(72, 0.2, 0.6), SungNote(84, 0.8, 0.6)],
        dict(glide_s=0.05, consonant_s=0.05),
    ),
    "soprano gamme": (
        sequence([79, 81, 83, 84, 86, 88], dur_s=0.6, start_s=0.2),
        dict(glide_s=0.09, consonant_s=0.05),
    ),
    "glissando descendant": (
        [SungNote(67, 0.2, 0.8), SungNote(60, 1.0, 0.8)],
        dict(glide_s=0.25, consonant_s=0.05),
    ),
    "vibrato profond": (
        [SungNote(64, 0.2, 0.82), SungNote(67, 1.02, 0.8)],
        dict(glide_s=0.09, consonant_s=0.05, vibrato_st=0.6),
    ),
    "trille": (
        [SungNote(64, 0.2, 0.35), SungNote(65, 0.55, 0.35),
         SungNote(64, 0.9, 0.35), SungNote(65, 1.25, 0.35)],
        dict(glide_s=0.02, consonant_s=0.03),
    ),
    "répétition": ([SungNote(64, 0.2, 0.7), SungNote(64, 1.2, 0.7)], dict(consonant_s=0.05)),
    "répétition aa-aa": (
        sequence([69, 69, 69, 69], dur_s=0.30, gap_s=0.15, start_s=0.2),
        dict(consonant_s=0, glide_s=0, noise=0.02),
    ),
}

# le temps réel retrouve exactement ces phrases-là (2026-09-25 : les trois
# nouveaux cas de vérité — glissando descendant, vibrato profond, soprano —
# y entrent ; ils passaient déjà à l'arrivée, la chaîne live les tient)
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
    "répétition aa-aa",
    "glissando descendant",
    "vibrato profond",
    "soprano gamme",
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


@pytest.mark.parametrize(
    "name",
    ["legato", "consonnes", "glissando long", "saut d'octave", "répétition", "répétition aa-aa"],
)
def test_geometry_alone_is_still_strict(name: str):
    """Sans preuve d'attaque, la règle géométrique du glissando doit suffire
    partout où elle est sans ambiguïté (le pipeline fournit toujours des
    attaques, mais un appelant peut ne pas en avoir)."""
    acc = _offline(name, False)
    assert acc.exact, acc.line()


def test_onsets_and_pitches_are_close():
    for name in ("consonnes", "gamme liée", "croches", "voix grave"):
        acc = _offline(name, True)
        # la note détectée démarre sur la consonne, pas sur la stabilisation
        assert acc.worst_start_ms <= 80.0, (name, acc.line())
        assert acc.worst_pitch_st == 0.0, (name, acc.line())


def test_high_octave_reaches_the_pitch_ceiling():
    """Les notes AU-DESSUS de l'ancien plafond de lecture (C6) restent justes.

    L'ancien FMAX (1050 Hz) tombait EXACTEMENT sur le C6 chanté : pYIN n'avait
    aucune marge au-dessus et repliait la soprano une octave plus bas (mesuré :
    un 84 chanté lu 72, une ligne 79-88 lue 67-76). Un C7 de marge lit ces
    notes sans rien changer en dessous (chant.wav/demo.wav inchangés).
    """
    from v2m.core.pitch import FMAX

    assert FMAX > 440.0 * 2 ** ((84 - 69) / 12)  # au-dessus du C6 chanté
    for name in ("saut octave haut", "soprano gamme"):
        acc = _offline(name, True)
        assert acc.exact, (name, acc.line())


def test_descending_glide_leaves_no_duplicate():
    """Un glissando DESCENDANT se lit comme celui qui monte : la note de départ
    se termine au passage de main (la consonne de l'arrivée y appartient) et
    aucune fragment dupliqué ne survit. Deux pièges, tous deux mesurés : la
    ré-attaque de l'arrivée coupait la note étirée à sa propre rampe (le
    doublon), et la borne stricte de _split_between_at_attack rejetait
    l'attaque tombée SUR la frontière (l'arrivée démarrait 150 ms tard).
    """
    acc = _offline("glissando descendant", True)
    assert acc.exact, acc.line()
    detected = detect_notes(_signal("glissando descendant"), SR)
    arrival = [n for n in detected if n.midi == 60]
    assert len(arrival) == 1 and arrival[0].start <= 1.03, [
        (n.midi, round(n.start, 2)) for n in detected
    ]


def test_arrival_pitch_reads_the_plateau_not_the_ramp():
    """La hauteur d'une note arrivant en glissando se lit sur son PLATEAU,
    pas sur la médiane de la rampe qui y mène (mesuré : 62 lu 61, 67 lu 66,
    72 lu 71 quand la condition du forced_pitch était inversée — 25 tests
    rouges). La règle : médiane de queue si l'arrivée s'est posée.
    """
    # le corpus gèle déjà la lecture juste ; ce cas pique la mécanique : une
    # ligne legato dont les notes 2/3/5 arrivent par portamento de 90 ms
    acc = _offline("legato", True)
    assert acc.exact and acc.worst_pitch_st == 0.0, acc.line()


def test_deep_vibrato_and_trill_stay_exact():
    """Vibrato profond (±0,6 st) et trille rapide restent exacts : le vibrato
    ne se fragmente pas (le median 9 trames suit la oscillation) et un trille
    de 350 ms ne se fusionne pas."""
    for name in ("vibrato profond", "trille"):
        acc = _offline(name, True)
        assert acc.exact, (name, acc.line())


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


def test_dotted_pulse_take_publishes_6_8_offline():
    """Un vrai signal sur temps pointés obtient son 6/8 HORS LIGNE, comme en
    direct — le tempo est corrigé du temps pointé vers la noire (×1,5) par
    la même preuve d'alternance (meter.dotted_pulse_tempo), appliquée après
    le raffinage. Temps pointés à 90 BPM : noire à 135, repli d'octave dans
    le range ; les forces alternent (note forte de temps pointé, faible de
    contretemps) pour armer la preuve d'alternance.
    """
    quarter = 60.0 / 135.0            # noire attendue à 135 BPM
    dotted = 1.5 * quarter            # le temps pointé chanté
    notes = []
    for i in range(9):                # 9 temps pointés = 4.5 mesures 6/8
        notes.append(SungNote(60 + (i % 3), start_s=i * dotted,
                              dur_s=dotted * 0.85,
                              amp=(1.0 if i % 2 == 0 else 0.45)))
    # articulation par attaque de niveau : les bursts de consonne ne sont
    # pas scalés par note, leur loudness constante brouillerait la force
    y = synthesize(notes, consonant_s=0)
    res = analyze_signal(y, SR)
    assert res.meter is not None and res.meter.label == "6/8"
    assert res.tempo is not None and res.tempo.bpm == pytest.approx(135.0, abs=4.0)


def test_breaths_between_phrases_become_rests():
    """« Détaché » = 6 notes en 6 phrases ; chaque respiration doit devenir
    un silence (l'intro incluse) et aucune ne peut chevaucher une note.
    Les figures sont soit None (gap trop court pour une figure à ce tempo,
    la portée l'engrave quand même d'après sa durée), soit du catalogue."""
    y = _signal("détaché")
    pipeline = analyze_signal(y, SR)
    rests = pipeline.rests or []
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
    rests = analyze_signal(y, SR).rests or []
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


# --------------------------------------------------------------------------- #
# Erreurs humaines : souffle, voix douce, timing négligent
# --------------------------------------------------------------------------- #


def test_a_breathy_voice_does_not_sprout_extra_notes():
    """Une voix SOUFFLÉE (beaucoup d'air) ne doit pas se couvrir de
    micro-notes : mesuré, les ondulations du souffle produisaient jusqu'à 12
    fragments à l'intérieur des notes chantées. Le plancher relatif de force
    des attaques (MIN_STRENGTH) rejette le bruit (≤ 0,06) et garde les
    consonnes (≥ 0,86)."""
    truth = sequence(MIDIS, dur_s=0.6, start_s=0.2)
    for noise in (0.05, 0.08, 0.10):
        y = synthesize(truth, glide_s=0.09, consonant_s=0.05, noise=noise)
        acc = measure(detect_notes(y, SR), truth)
        assert acc.exact, f"noise={noise}: {acc.line()}"


def test_breaths_never_become_notes():
    """Les respirations ne deviennent JAMAIS des notes, sous aucune forme :
    ni une reprise d'air dans le gap d'une ligne liée (le souffle 90 ms
    précède chaque réattaque), ni une respiration isolée dans un silence
    long (qui doit rester un SILENCE, pas une note de bruit), ni le
    souffle continu d'une tenue longue. Mesuré : un souffle 3× plus fort
    que le contrat ci-dessus (0,10 absolu) fragmente réellement une tenue —
    c'est hors contrat, le harnais le documente plutôt que de le subir."""
    # 1) ligne liée : le chanteur reprend l'air entre deux notes liées
    truth = sequence([60, 64, 67, 72], dur_s=0.5, gap_s=0.15, start_s=0.2)
    y = synthesize(truth, SR, glide_s=0.09, consonant_s=0.0)
    for n in truth[1:]:  # le souffle précède chaque réattaque
        add_breath(y, n.start_s - 0.06)
    acc = measure(detect_notes(y, SR), truth)
    assert acc.exact, f"ligne liée + reprises d'air: {acc.line()}"

    # 2) respiration isolée dans un silence long : silence, pas note
    truth2 = [SungNote(60, 0.2, 0.5), SungNote(67, 1.4, 0.5)]
    y2 = synthesize(truth2, SR, glide_s=0.0, consonant_s=0.0)
    add_breath(y2, 0.8, dur_s=0.15, amp=0.3, seed=9)
    acc2 = measure(detect_notes(y2, SR), truth2)
    assert acc2.exact, f"souffle isolé: {acc2.line()}"

    # 3) souffle continu du harnais (0,10 = le contrat gelé) sur des tenues
    #    LONGUES : jamais de micro-note à l'intérieur
    for dur in (1.0, 2.4):
        truth3 = [SungNote(60, 0.2, dur), SungNote(67, 0.2 + dur + 0.3, 1.0)]
        y3 = synthesize(truth3, SR, glide_s=0.0, consonant_s=0.05, noise=0.10)
        acc3 = measure(detect_notes(y3, SR), truth3)
        assert acc3.exact, f"tenue {dur}s soufflée: {acc3.line()}"


def test_a_quiet_voice_is_still_transcribed():
    """Chanter DOUCEMENT (amp 0,05) ne doit ni perdre les notes ni tomber
    sous les seuils de silence."""
    truth = sequence(MIDIS, dur_s=0.6, start_s=0.2)
    y = synthesize(truth, glide_s=0.09, consonant_s=0.05, amp=0.05)
    assert measure(detect_notes(y, SR), truth).exact


def test_careless_timing_is_still_transcribed():
    """Durées et respirations IRRÉGULIÈRES (chanson Humaine) : 5/5 notes.
    Aucun des deux chemins ne suppose un tempo régulier."""
    truth = [
        SungNote(60, 0.20, 0.35), SungNote(62, 0.70, 0.70),
        SungNote(64, 1.55, 0.28), SungNote(65, 1.95, 0.55),
        SungNote(67, 2.65, 0.90),
    ]
    y = synthesize(truth, glide_s=0.09, consonant_s=0.05)
    assert measure(detect_notes(y, SR), truth).exact


def test_live_confirmation_latency_stays_under_120ms():
    """La confirmation live a été abaissée (140 -> 100 -> 90 ms) pour
    l'affichage : la robustesse doit tenir — vibrato, legato, croches,
    répétitions. Le plancher mesuré est 0.09 : en dessous (0.06-0.08), le
    cas « bruit et vibrato » échoue sporadiquement selon la graine.
    """
    stream = LivePitchStream(SR, gate_rms=GATE)
    assert stream.seg.confirm_s == pytest.approx(0.09)
    for name in ("vibrato", "legato", "croches", "répétition"):
        acc = _live(name)
        assert acc.exact, f"{name}: {acc.line()}"


# ---------------------------------------------------------------------------
# Dérive sur la VRAIE prise : l'empreinte dorée rejouée à chaque run
# ---------------------------------------------------------------------------


def test_real_take_segmentation_does_not_drift():
    """Filet automatique contre la dérive de segmentation sur chant.wav :
    le fingerprint (--compare du diagnostic) est rejoué ICI, à chaque run
    de la suite, contre l'empreinte enregistrée à côté de la prise
    (<audio>.golden.json). Toute différence (tempo, mesure, tonalité,
    chaque note, chaque respiration) fait échouer CE test — plus besoin de
    penser à lancer le script. La prise est gitignorée : skip absente.
    """
    from v2m.core import audio_io
    from v2m.core.pipeline import AnalyzeOptions, analyze_signal
    from scripts.score_accuracy import (  # type: ignore[import-not-found]
        compare_fingerprints,
        load_golden,
        segmentation_fingerprint,
    )

    audio = Path(__file__).resolve().parents[1] / "chant.wav"
    if not audio.exists():
        pytest.skip("chant.wav absent (gitignore, régénérable) — dérive non rejouée")
    golden = load_golden(str(audio))
    if golden is None:
        pytest.skip("aucune empreinte dorée pour chant.wav — lancer le script --golden une fois")
    res = analyze_signal(
        audio_io.load_file(str(audio)), audio_io.SR,
        source=str(audio), opts=AnalyzeOptions(),
    )
    diffs = compare_fingerprints(golden, segmentation_fingerprint(res))
    assert not diffs, "dérive de segmentation sur chant.wav :\n" + "\n".join(diffs)
