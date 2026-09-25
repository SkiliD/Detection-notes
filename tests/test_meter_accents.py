"""Tests for 3/4 vs 4/4 discrimination from onset accents (meter.py)."""

from v2m.core.meter import detect_meter
from v2m.core.notes import Note

BPM = 120.0
BEAT = 60.0 / BPM  # 0.5 s


def _make_meter_notes(beat_pcs: list[tuple[int, ...]], strengths: list[float],
                      base: int = 60) -> list[Note]:
    """One chord per beat: the whole attack shares the beat's strength.

    A strummed/hummed chord attacks together: every note of the beat gets
    the beat's accent (the beat, not the individual note, is accented).
    """
    notes: list[Note] = []
    t = 0.0
    for i, pcs in enumerate(beat_pcs):
        s = strengths[i]
        for pc in pcs:
            notes.append(Note(start=t, end=t + BEAT * 0.9,
                              midi=base + pc, confidence=0.9, strength=s))
        t += BEAT
    return notes


def test_waltz_detected_as_3_4():
    # 8 bars of 3/4: strong downbeat every 3 beats
    beat_pcs: list[tuple[int, ...]] = []
    strengths: list[float] = []
    for _ in range(8):
        beat_pcs += [(0, 4, 7), (7,), (7,)]      # I - V - V pattern
        strengths += [1.0, 0.3, 0.3]
    notes = _make_meter_notes(beat_pcs, strengths)
    meter = detect_meter(notes, BPM)
    assert meter is not None
    assert meter.label == "3/4"


def test_common_time_detected_as_4_4():
    # 6 bars of 4/4: strong downbeat every 4 beats
    beat_pcs: list[tuple[int, ...]] = []
    strengths: list[float] = []
    for _ in range(6):
        beat_pcs += [(0, 4, 7), (7,), (9,), (7,)]  # I - V - vi - V
        strengths += [1.0, 0.3, 0.3, 0.3]
    notes = _make_meter_notes(beat_pcs, strengths)
    meter = detect_meter(notes, BPM)
    assert meter is not None
    assert meter.label == "4/4"


def _pulse(strengths: list[float]) -> list[Note]:
    """Une pulsation de notes (une par temps) répétant le profil donné.

    Répété jusqu'à 12 notes : la lecture parité du 2/4 exige que CHAQUE
    phase de la grille 4 temps soit assez peuplée (MIN_DOWNBEATS) pour
    que son absence ne masque pas le motif.
    """
    n_reps = -(-12 // len(strengths))  # ceil(12 / len)
    return _make_meter_notes([(0,)] * (len(strengths) * n_reps), strengths * n_reps)


def test_march_pattern_is_2_4():
    """Forces alternées strictes (période 2) : la mesure est 2/4.

    Les grilles emboîtées ne peuvent pas séparer 2/4 de 4/4 (barre moitié) ;
    la période des forces le peut — c'est la preuve qui casse le tie. Les
    deux temps forts doivent être comparables : un 4/4 à hiérarchie (t1
    primaire, t3 secondaire) garde le 4/4.
    """
    meter = detect_meter(_pulse([1.0, 0.15]), 120.0)
    assert meter is not None and meter.label == "2/4"


def test_unequal_strong_beats_stay_4_4():
    """Temps forts inégaux (t1 primaire, t3 secondaire plus faible) : la
    hiérarchie entre les phases paires est la signature 4 temps — pas 2/4.
    """
    meter = detect_meter(_pulse([1.0, 0.3, 0.6, 0.3]), 120.0)
    assert meter is not None and meter.label == "4/4"


def test_backbeat_keeps_4_4():
    """Pulsation forte sur les temps FAIBLES : pas de 2/4 — les barres
    démarreraient sur un temps faible et la grille n'a pas de phase de
    mesure ; le défaut 4/4 est non destructeur.
    """
    meter = detect_meter(_pulse([0.15, 1.0]), 120.0)
    assert meter is not None and meter.label == "4/4"


def test_flat_accents_stay_undecided():
    # every beat equally strong: timing cannot decide, None keeps 4/4 default
    beat_pcs = [(0, 4, 7)] * 12
    strengths = [0.6] * 12
    notes = _make_meter_notes(beat_pcs, strengths)
    assert detect_meter(notes, BPM) is None


def test_strength_is_filled_by_pipeline():
    from v2m.core.onsets import compute_onset_strengths

    notes = [Note(start=0.0, end=0.4, midi=60, confidence=0.9),
             Note(start=0.5, end=0.9, midi=64, confidence=0.9)]
    import numpy as np

    sr = 22050
    y = np.zeros(int(1.0 * sr), dtype=np.float32)
    # loud burst at t=0, quiet at t=0.5 -> relative strengths differ
    y[: int(0.1 * sr)] = 0.9
    y[int(0.5 * sr): int(0.55 * sr)] = 0.1
    compute_onset_strengths(notes, y, sr)
    assert 0.0 <= notes[0].strength <= 1.0
    assert 0.0 <= notes[1].strength <= 1.0
    assert notes[0].strength > notes[1].strength
