"""The consonant corpus: voiced re-strikes read, unvoiced bursts never note.

A sung ta-ta-ta is a series of wide-band consonant bursts carrying the
articulation. Over a CONTINUOUS tone (one breath, no gap) each burst cuts
a re-articulated note; in SILENCE the same bursts produce attacks but
never a note. pYIN's confidence collapses through a burst (vprob 0.01
while the voice is continuous) — the chain still reads correctly because
clean_pitch interpolates the voiced gaps.
"""

from __future__ import annotations

import numpy as np

from voicetruth import SungNote, add_consonant, consonant_line, synthesize
from v2m.core.notes import ATTACK_TOL_S, interpret
from v2m.core.observation import VoiceObservation

SR = 22_050


def _read(y: np.ndarray):
    obs = VoiceObservation.from_signal(y, SR, use_attacks=True)
    return obs, interpret(obs)


def _orphans(obs: VoiceObservation, starts: list[float]) -> list[float]:
    """Attacks no note starts within ATTACK_TOL_S of (the audit's sign)."""
    assert obs.attacks is not None  # every reader here builds with attacks
    return [
        float(t)
        for t in obs.attacks.times
        if not any(abs(float(t) - s) <= ATTACK_TOL_S for s in starts)
    ]


def test_tata_on_a_continuous_tone_cuts_each_re_strike():
    """One held tone re-struck by four consonant bursts at the onsets:
    four notes, all the same pitch, each starting at its burst."""
    y = synthesize([SungNote(60, 0.1, 1.5)], consonant_s=0)
    for t in (0.1, 0.6, 1.0, 1.4):
        add_consonant(y, t)
    obs, notes = _read(y)
    assert [n.midi for n in notes] == [60, 60, 60, 60]
    for t, n in zip((0.1, 0.6, 1.0, 1.4), notes):
        assert abs(n.start - t) <= ATTACK_TOL_S, (
            f"note {n.start:.2f} does not sit at its burst {t:.2f}"
        )


def test_tata_with_gaps_reads_one_note_per_burst():
    """Ta-ta-ta separated by short gaps: three notes, no orphan attack —
    every burst dates the note it starts."""
    y = consonant_line([60, 60, 60], dur_s=0.30, gap_s=0.15)
    obs, notes = _read(y)
    assert [n.midi for n in notes] == [60, 60, 60]
    assert _orphans(obs, [n.start for n in notes]) == []


def test_silent_bursts_produce_attacks_but_never_a_note():
    """The same consonant bursts in silence: attacks are detected (the
    audit sign lists every one as orphan) but no note is engraved."""
    y = np.zeros(int(1.6 * SR), dtype=np.float32)
    for t in (0.2, 0.6, 1.0, 1.4):
        add_consonant(y, t)
    obs, notes = _read(y)
    assert notes == []
    assert _orphans(obs, []) == [0.2, 0.6, 1.0, 1.4]


def test_vprob_collapses_through_a_burst_yet_the_curve_survives():
    """The measured evidence an unvoiced rule would have to beat: pYIN's
    confidence reads 0.01 INSIDE a continuous voice right at the burst,
    yet the pitch curve stays voiced through it (clean_pitch interpolates
    the doubt) — confidence alone cannot separate voice from silence."""
    y = synthesize([SungNote(60, 0.1, 1.2)], consonant_s=0)
    add_consonant(y, 0.6)
    obs, _ = _read(y)
    mid = (obs.pitch.times >= 0.55) & (obs.pitch.times <= 0.65)
    voiced_mid = mid & ~np.isnan(obs.pitch.hz)
    assert voiced_mid.any()  # the curve bridges the burst
    assert (obs.pitch.confidence[voiced_mid] < 0.10).any()  # the doubt is real
    hz_mid = obs.pitch.hz[voiced_mid]
    assert (np.abs(hz_mid - 261.63) / 261.63 < 0.02).all()  # ...and on pitch
