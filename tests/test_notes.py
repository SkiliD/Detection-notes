"""Tests segmentation : 3 notes synthétiques -> 3 événements."""

import sys
from pathlib import Path

import numpy as np

from conftest import SR, silence, sine_note
from v2m.core.notes import track_to_notes
from v2m.core.pitch import track_pitch_offline


def _three_notes_signal() -> np.ndarray:
    return np.concatenate(
        [
            sine_note(220.0, 0.5),   # A3
            silence(0.15),
            sine_note(261.63, 0.5),  # C4
            silence(0.15),
            sine_note(329.63, 0.5),  # E4
        ]
    )


def test_three_notes_detected():
    y = _three_notes_signal()
    track = track_pitch_offline(y, SR)
    notes = track_to_notes(track, SR)
    assert len(notes) == 3
    midis = [n.midi for n in notes]
    assert midis == [57, 60, 64]  # A3, C4, E4


def test_note_times_reasonable():
    y = _three_notes_signal()
    track = track_pitch_offline(y, SR)
    notes = track_to_notes(track, SR)
    # la première note commence près de 0
    assert notes[0].start < 0.15
    # durée de chaque note ~0.5 s (± tolérance)
    for n in notes:
        assert 0.3 < n.duration < 0.75
    # ordre chronologique
    assert notes[0].end <= notes[1].start + 0.1


def test_silence_gives_no_notes():
    track = track_pitch_offline(np.zeros(SR * 2, dtype=np.float32), SR)
    notes = track_to_notes(track, SR)
    assert notes == []


def test_vibrato_single_note():
    """Une note avec vibrato (±30 cents) reste une seule note."""
    dur = 1.0
    t = np.arange(int(dur * SR)) / SR
    hz = 440.0 * 2 ** (0.3 * np.sin(2 * np.pi * 5.5 * t) / 12)  # ±30 cents
    env = np.minimum(1, np.minimum(t / 0.01, (dur - t) / 0.01))
    y = (0.5 * np.sin(2 * np.pi * np.cumsum(hz) / SR) * env).astype(np.float32)
    track = track_pitch_offline(y, SR)
    notes = track_to_notes(track, SR)
    assert len(notes) == 1


# ---------------------------------------------------------------------------
# Stabilisation du f0 : le masque de dérivée, et le pitch lu sur le plateau
# ---------------------------------------------------------------------------


def test_clean_pitch_marks_transitions_and_estimates_on_plateaus():
    """Le masque `transition` (dérivée du f0) couvre les rampes du
    portamento, PAS les plateaux ni les pics du vibrato : le pitch d'une
    note arrivant en glissando se lit sur son PLATEAU (médiane des trames
    stables), jamais sur la rampe qui y mène (mesuré : 66.09 lu pour un
    67 chanté quand la rampe restait dans l'estimation)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from voicetruth import SungNote, synthesize

    truth = [SungNote(60, 0.2, 0.7), SungNote(67, 0.9, 0.7)]
    y = synthesize(truth, glide_s=0.12, consonant_s=0.0)  # LONG portamento
    from v2m.core.onsets import detect_attacks

    track = track_pitch_offline(y, SR)
    notes = track_to_notes(track, SR, attacks=detect_attacks(y, SR))
    assert [n.midi for n in notes] == [60, 67]
    assert track.transition is not None
    voiced = ~np.isnan(track.hz)
    assert 0.0 < float(track.transition[voiced].mean()) < 0.5


def test_tuning_offset_pulls_a_flat_take_back_on_the_grid():
    """Un chanteur systématiquement 30 cents bas : sans correction, chaque
    note ronde autour du mauvais demi-ton ; l'offset global (médiane des
    déviations, borné ±0.35) replace la prise AVANT l'arrondi. L'offset
    mesuré part dans `tuning_out` (le JSON le rapporte en cents)."""
    from voicetruth import SungNote, synthesize

    CENTS = -0.30
    midis = [60, 62, 64, 65, 67]
    truth = [SungNote(m, 0.2 + i * 0.6, 0.55) for i, m in enumerate(midis)]
    y = synthesize(truth, glide_s=0.0, consonant_s=0.05)
    # detune the WHOLE take by -30 cents (a consistently flat singer) by
    # playing it slower: rate r shifts the pitch by 12*log2(r) st
    sr = SR
    idx = np.arange(len(y)) / sr
    rate = 2.0 ** (CENTS / 12.0)  # < 1: slower playback, lower pitch
    src_positions = idx * rate
    y_flat = np.interp(src_positions, idx, y).astype(np.float32)

    track = track_pitch_offline(y_flat, sr)
    tuning: dict = {}
    notes = track_to_notes(track, sr, attacks=None, tuning_out=tuning)
    assert len(notes) == len(midis)
    assert tuning["tuning_offset_st"] == pytest_approx_cents(CENTS)
    assert [n.midi for n in notes] == midis  # the flat take reads ON the grid


def pytest_approx_cents(cents: float) -> float:
    """Tolerance wrapper: `cents` in SEMITONES, measured within 8 cents."""
    import pytest

    return pytest.approx(cents, abs=0.08)
    assert notes[0].midi == 69
