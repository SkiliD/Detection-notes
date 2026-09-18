"""Détection du tempo (BPM) et des positions de temps."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

BPM_RANGE = (60.0, 180.0)


@dataclass
class TempoResult:
    bpm: float
    beats: np.ndarray  # temps (s) des temps estimés


def detect_tempo(
    y: np.ndarray,
    sr: int,
    bpm_range: tuple[float, float] = BPM_RANGE,
) -> TempoResult:
    """Tempo via onset strength + beat tracking, avec repli x2 / ÷2.

    Sur une voix seule (peu de percussif), l'estimateur peut se tromper
    d'un facteur 2 : on contraint au range demandé.
    """
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, start_bpm=120.0, tightness=100.0
    )
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # repli : si le BPM est hors range, essayer x2 et /2
    lo, hi = bpm_range
    if not (lo <= bpm <= hi):
        candidates = [bpm, bpm * 2, bpm / 2, bpm * 2 / 3, bpm * 3 / 2]
        valid = [b for b in candidates if lo <= b <= hi]
        if valid:
            bpm = valid[0]
        else:
            bpm = float(np.clip(bpm, lo, hi))
    return TempoResult(bpm=round(bpm, 1), beats=np.asarray(beat_times, dtype=float))


def refine_bpm_with_onsets(
    onsets: np.ndarray,
    bpm: float,
    span_rel: float = 0.08,
    step: float = 0.0005,
    min_hits: int = 3,
) -> float:
    """Raffine un BPM en maximisant la cohérence de phase des attaques.

    `librosa.beat.beat_track` peut renvoyer un tempo arrondi (ex. 117,5
    pour un vrai 120) : les frontières de notes finissent alors hors de
    toute grille (0,5 s = 1,378 temps à 117,5) et le rendu en figures de
    notes se fragmente. On scanne bpm±span_rel (relatif) ; pour chaque
    candidat, la grille k·période est alignée par moindres carrés sur les
    attaques (fenêtre ±15 % de période) et on garde le meilleur score :
    hits − 0,5·écart quadratique. Pas de pénalité de proximité : dans un
    span de ±8 %, les demi/double tempo équivalents sont hors de portée.
    """
    if bpm <= 0 or len(onsets) < min_hits:
        return float(bpm)
    best_bpm, best_score = float(bpm), -np.inf
    lo, hi = bpm * (1.0 - span_rel), bpm * (1.0 + span_rel)
    for cand in np.arange(lo, hi + step / 2, step):
        c_period = 60.0 / cand
        # alignement de phase par moindres carrés (sans recalculer hits)
        k = np.round(onsets / c_period)
        targets = k * c_period
        resid = onsets - targets
        # fenêtre serrée : un BPM faux de 1 % accumule ~75 ms de dérive
        # en 9 temps et perd la majorité de ses hits
        tol = max(0.045, 0.15 * c_period)
        good = np.abs(resid) <= tol
        n_hits = int(good.sum())
        if n_hits < min_hits:
            continue
        rms = float(np.sqrt(np.mean(resid[good] ** 2)))
        score = n_hits - 0.5 * rms
        if score > best_score:
            best_score, best_bpm = score, float(cand)

    # polissage : régression linéaire temps = k·période + offset sur les
    # attaques en phase → période exacte (ex. 120,75 → 120,00)
    p = 60.0 / best_bpm
    k = np.round(onsets / p)
    tol = max(0.045, 0.15 * p)
    inl = np.abs(onsets - k * p) <= tol
    if inl.sum() >= min_hits:
        kk = k[inl]
        tt = onsets[inl]
        slope = float(np.polyfit(kk, tt, 1)[0])
        if slope > 0.05:
            refined = 60.0 / slope
            # garde-fou : rester dans le span relatif du scan
            if bpm * (1 - span_rel - 0.02) <= refined <= bpm * (1 + span_rel + 0.02):
                best_bpm = refined
    return best_bpm

