"""Vérité terrain : une voix synthétique dont on connaît toutes les notes.

Un sinus pur ne teste rien d'utile ici : il n'a ni transition, ni consonne, ni
vibrato — donc rien de ce qui fait échouer la segmentation. Ce module fabrique
un signal « chanté » (harmoniques glottiques, portamento entre les notes,
consonnes large bande, vibrato, bruit de fond) dont les événements sont connus
exactement, puis compare ce que le pipeline détecte à cette vérité.

Deux usages :

    from voicetruth import sequence, synthesize, detect_notes, measure
    truth = sequence([60, 62, 64])              # noires liées, 100 BPM
    acc = measure(detect_notes(synthesize(truth)), truth)
    assert acc.exact

`scripts/score_accuracy.py` s'en sert pour noter une vraie prise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 22_050


@dataclass(frozen=True)
class SungNote:
    """Une note effectivement chantée (la référence)."""

    midi: int
    start_s: float
    dur_s: float

    @property
    def hz(self) -> float:
        return 440.0 * 2.0 ** ((self.midi - 69) / 12.0)


def sequence(
    midis,
    start_s: float = 0.0,
    dur_s: float = 0.6,
    gap_s: float = 0.0,
    legato_s: float = 0.0,
) -> list[SungNote]:
    """Notes successives : `legato_s` > 0 les enchaîne (portamento, pas de silence)."""
    step = dur_s + gap_s - legato_s
    return [SungNote(m, start_s + i * step, dur_s) for i, m in enumerate(midis)]


def _harmonic_voice(f0: np.ndarray, sr: int, f_max: float = 4000.0) -> np.ndarray:
    """Somme harmonique à pente glottique (1/k^1.4), bornée vers 4 kHz."""
    phase = 2.0 * np.pi * np.cumsum(f0) / sr
    base = float(np.mean(f0))
    n_harm = max(1, int(f_max / max(base, 1.0)))
    out = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        out += np.sin(k * phase) / k ** 1.4
    return out


def synthesize(
    notes: list[SungNote],
    sr: int = SR,
    glide_s: float = 0.09,
    consonant_s: float = 0.05,
    consonant_amp: float = 1.4,
    vibrato_st: float = 0.0,
    vibrato_hz: float = 5.5,
    noise: float = 0.0,
    amp: float = 0.35,
    seed: int = 0,
) -> np.ndarray:
    """Signal mono float32 portant `notes`.

    `glide_s` : durée du portamento à l'attaque (0 = attaque nette). Le
        portamento n'existe qu'entre notes LIÉES : après un silence, le
        chanteur réattaque la note (c'est ce que fait un vrai détaché).
    `consonant_s` : durée du burst large bande à l'attaque (0 = aucune).
    `vibrato_st` : profondeur du vibrato en demi-tons (±valeur), 0 = tenue.
    `noise` : amplitude du bruit de fond ajouté partout.
    """
    if not notes:
        return np.zeros(0, dtype=np.float32)
    end = max(n.start_s + n.dur_s for n in notes) + 0.3
    y = np.zeros(int(end * sr), dtype=np.float64)
    rng = np.random.default_rng(seed)
    prev_hz: float | None = None
    prev_end = 0.0
    phase0 = 0.0

    for note in notes:
        i0 = int(note.start_s * sr)
        ln = max(1, int(note.dur_s * sr))
        t = np.arange(ln) / sr
        hz = note.hz
        f0 = np.full(ln, hz)
        legato = note.start_s <= prev_end + 0.05
        if prev_hz and glide_s > 0 and legato:
            g = min(int(glide_s * sr), ln)
            f0[:g] = np.linspace(prev_hz, hz, g)  # portamento linéaire en Hz
        if vibrato_st > 0:
            depth = np.clip((t - 0.12) / 0.12, 0.0, 1.0)  # vibrato qui s'installe
            f0 = f0 * 2.0 ** (vibrato_st * np.sin(2 * np.pi * vibrato_hz * t) * depth / 12.0)

        # phase continue d'une note à l'autre : pas de clic à la jointure
        phase = phase0 + 2.0 * np.pi * np.cumsum(f0) / sr
        phase0 = float(phase[-1] + 2.0 * np.pi * f0[-1] / sr)
        body = _harmonic_voice(f0, sr) * _envelope(t, ln / sr)

        if consonant_s > 0:
            cn = min(int(consonant_s * sr), ln)
            burst = rng.standard_normal(cn) * _consonant_envelope(cn, sr)
            body[:cn] += consonant_amp * burst
        y[i0 : i0 + ln] += body
        prev_hz = hz
        prev_end = note.start_s + note.dur_s

    y *= amp
    if noise > 0:
        y += noise * amp * rng.standard_normal(len(y))
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * amp).astype(np.float32)


def _envelope(t: np.ndarray, dur_s: float) -> np.ndarray:
    """Attaque 25 ms, relâchement 40 ms — pas de clic aux extrémités."""
    env = np.ones_like(t)
    a = min(int(0.025 * len(t) / max(dur_s, 1e-9)), len(t) // 2)
    r = min(int(0.040 * len(t) / max(dur_s, 1e-9)), len(t) // 2)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def _consonant_envelope(n: int, sr: int) -> np.ndarray:
    """Burst percussif : très court, décroissance rapide."""
    t = np.arange(n) / sr
    return np.exp(-t / max(0.012, 1e-9))


# --------------------------------------------------------------------------- #
# Mesure
# --------------------------------------------------------------------------- #


@dataclass
class Accuracy:
    """Ce que la chaîne a fait des notes chantées."""

    truth_n: int
    detected_n: int
    matched: int
    missing: int
    extra: int
    intruders: int          # notes en trop À L'INTÉRIEUR d'une note chantée
    worst_start_ms: float
    worst_pitch_st: float

    @property
    def exact(self) -> bool:
        """Toutes les notes trouvées, aucune en trop (donc pas de fragment)."""
        return self.matched == self.truth_n and self.extra == 0

    def line(self) -> str:
        return (
            f"{self.matched}/{self.truth_n} notes, {self.extra} en trop "
            f"(dont {self.intruders} à l'intérieur), écart attaque ≤ "
            f"{self.worst_start_ms:.0f} ms"
        )


def detect_notes(y: np.ndarray, sr: int = SR, with_attacks: bool = True):
    """Chaîne complète hauteur -> attaques -> segmentation."""
    from v2m.core.notes import track_to_notes
    from v2m.core.onsets import detect_attacks
    from v2m.core.pitch import track_pitch_offline

    track = track_pitch_offline(y, sr)
    attacks = detect_attacks(y, sr) if with_attacks else None
    return track_to_notes(track, sr, attacks=attacks)


def measure(
    detected,
    truth: list[SungNote],
    tol_s: float = 0.12,
    pitch_tol_st: float = 0.5,
) -> Accuracy:
    """Apparie les notes détectées aux notes chantées (même attaque, même hauteur).

    Un fragment de transition (note en trop entre deux vraies notes) est
    compté comme `intruders` : c'est exactement le défaut que la détection
    d'attaques doit supprimer, et il se cacherait dans un simple compte de
    notes total.
    """
    used = [False] * len(detected)
    matched = 0
    worst_start = 0.0
    worst_pitch = 0.0
    for ev in truth:
        best_j: int | None = None
        best_dt = tol_s
        for j, d in enumerate(detected):
            if used[j] or d.midi != ev.midi:
                continue
            dt = abs(d.start - ev.start_s)
            if dt <= best_dt:
                best_j, best_dt = j, dt
        if best_j is None:
            continue
        used[best_j] = True
        matched += 1
        worst_start = max(worst_start, best_dt * 1000.0)
        worst_pitch = max(worst_pitch, abs(detected[best_j].midi - ev.midi))

    intruders = 0
    for j, d in enumerate(detected):
        if used[j]:
            continue
        for ev in truth:
            if ev.start_s - tol_s <= d.start <= ev.start_s + ev.dur_s + tol_s:
                intruders += 1
                break

    return Accuracy(
        truth_n=len(truth),
        detected_n=len(detected),
        matched=matched,
        missing=len(truth) - matched,
        extra=sum(1 for j in range(len(detected)) if not used[j]),
        intruders=intruders,
        worst_start_ms=round(worst_start, 1),
        worst_pitch_st=round(worst_pitch, 2),
    )


def describe(acc: Accuracy) -> str:
    """Ligne lisible pour les rapports de script."""
    flag = "OK " if acc.exact else "→  "
    return f"{flag}{acc.line()}"
