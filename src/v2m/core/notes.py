"""Nettoyage de la courbe de hauteur et segmentation en notes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .music import hz_to_midi

DEFAULT_MIN_CONF = 0.6
DEFAULT_MIN_NOTE_S = 0.060   # 60 ms
DEFAULT_GAP_S = 0.050        # trous < 50 ms interpolés
DEFAULT_SEMITONES = 0.5      # rupture de note si |Δmidi| > 0.5
NEAR_MERGE_ST = 0.4          # fusion des segments dont les médianes sont proches


@dataclass
class Note:
    """Une note segmentée, prête pour l'export MIDI."""

    start: float   # secondes
    end: float     # secondes
    midi: int      # numéro MIDI (arrondi)
    confidence: float  # 0..1, moyenne de confiance
    figure: str | None = None  # figure musicale (cf. figures.py), None si hors grille

    @property
    def duration(self) -> float:
        return self.end - self.start


def clean_pitch(track, min_conf: float = DEFAULT_MIN_CONF) -> None:
    """Filtre médian anti-octave-jumps + masquage des frames peu fiables.

    Modifie track.hz / track.confidence en place.
    """
    hz = track.hz.copy()
    voiced = ~np.isnan(hz) & (track.confidence >= min_conf)
    if voiced.sum() < 3:
        track.hz = np.where(voiced, hz, np.nan)
        track.confidence = np.where(voiced, track.confidence, 0.0)
        return

    midi = hz_to_midi(hz)
    # filtre médian (fenêtre impaire ~ 5 frames) pour éliminer les sauts d'octave
    med = midi.copy()
    k = 2
    idx = np.arange(len(midi))
    for i in idx[voiced]:
        lo, hi = max(0, i - k), min(len(midi), i + k + 1)
        seg = midi[lo:hi][~np.isnan(midi[lo:hi])]
        if len(seg):
            med[i] = np.median(seg)
    # re-filtrage : on garde les frames proches de la médiane (< 1 demi-ton)
    ok = voiced & (np.abs(midi - med) < 1.0)
    ok &= voiced & (med >= 21.0) & (med <= 108.0)  # A0..C8
    # lissage vibrato : médiane 9 frames (~90 ms) — réduit le vibrato tout
    # en PRÉSERVANT les transitions nettes entre notes (la médiane ne floute
    # pas les fronts raides, contrairement à une moyenne).
    smoothed = _median_filter_nan(med, k=4)
    smoothed = np.where(np.isnan(med), np.nan, smoothed)
    track.hz = np.where(ok, 440.0 * 2.0 ** ((smoothed - 69.0) / 12.0), np.nan)
    track.confidence = np.where(ok, track.confidence, 0.0)


def _median_filter_nan(x: np.ndarray, k: int = 4) -> np.ndarray:
    """Filtre médian (fenêtre 2k+1) ignorant les NaN, préserve les fronts."""
    n = len(x)
    if k < 1 or n == 0:
        return x
    out = np.array(x, dtype=float, copy=True)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        seg = x[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = float(np.median(seg))
    return out


def track_to_notes(
    track,
    sr: int,
    min_conf: float = DEFAULT_MIN_CONF,
    min_note_s: float = DEFAULT_MIN_NOTE_S,
    gap_s: float = DEFAULT_GAP_S,
    semitones: float = DEFAULT_SEMITONES,
) -> list[Note]:
    """Segmente la courbe de hauteur nettoyée en notes (start, end, midi)."""
    clean_pitch(track, min_conf=min_conf)
    midi = hz_to_midi(track.hz)
    times = track.times

    voiced = ~np.isnan(midi)
    if not voiced.any():
        return []

    # interpolation des trous courts
    idx = np.flatnonzero(voiced)
    gaps = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1:
            gap = (times[b] - times[a])
            if gap <= gap_s:
                gaps.append((a, b))
    for a, b in gaps:
        span = b - a
        for j in range(a + 1, b):
            w = (j - a) / span
            midi[j] = (1 - w) * midi[a] + w * midi[b]
            voiced[j] = True

    # segmentation : la référence de note est la médiane courante du segment
    # (hystérésis avec confirmation) — les segments sont gardés en médiane
    # flottante pour permettre la fusion des artefacts avant l'arrondi
    segs: list[list[float]] = []  # [start, end, median_float, conf]
    seg_start: int | None = None
    seg_frames: list[int] = []

    def close_segment(end_frame: int) -> None:
        nonlocal seg_start, seg_frames
        if seg_start is None or not seg_frames:
            return
        seg_midi = midi[seg_frames]
        med_midi = float(np.median(seg_midi))
        t_start = float(times[seg_start])
        t_end = float(times[end_frame])
        if t_end - t_start >= min_note_s:
            segs.append(
                [t_start, t_end, med_midi,
                 float(np.nanmean(track.confidence[seg_frames]))]
            )
        seg_start = None
        seg_frames = []

    prev = idx[0]
    seg_start = idx[0]
    seg_frames = [idx[0]]
    ref = float(midi[idx[0]])
    pending: int | None = None
    confirm = 3  # frames consécutives hors bande pour confirmer une coupure
    for i in idx[1:]:
        if (i - prev) > 1:  # trou non comblé par l'interpolation
            close_segment(prev)
            seg_start = i
            seg_frames = [i]
            ref = float(midi[i])
            pending = None
            prev = i
            continue
        m = float(midi[i])
        if abs(m - ref) > semitones:
            # hors bande par rapport à la RÉFÉRENCE de la note (pas à la
            # frame précédente) : une rampe de transition déclenche dès la
            # première marche réelle
            if pending is None:
                pending = i
            if i - pending + 1 >= confirm:
                close_segment(pending - 1)
                seg_start = pending
                seg_frames = list(range(pending, i + 1))
                ref = float(np.median(midi[seg_frames]))
                pending = None
        else:
            if pending is not None:
                # fausse alerte : on ré-absorbe les frames pendantes
                seg_frames.extend(range(pending, i + 1))
                pending = None
            else:
                seg_frames.append(i)
            ref = float(np.median(midi[seg_frames]))
        prev = i
    close_segment(prev)

    # 1) fusion des segments contigus dont les médianes sont proches :
    #    réintègre les fragments de transition (courbe pYIN qui "rampe"
    #    entre deux notes) dans la note qui les suit ou les précède
    fused: list[list[float]] = []
    for s in segs:
        if (
            fused
            and s[0] - fused[-1][1] <= gap_s
            and abs(s[2] - fused[-1][2]) < NEAR_MERGE_ST
        ):
            d1 = fused[-1][1] - fused[-1][0]
            d2 = s[1] - s[0]
            fused[-1][2] = (fused[-1][2] * d1 + s[2] * d2) / (d1 + d2)
            fused[-1][1] = s[1]
            fused[-1][3] = max(fused[-1][3], s[3])
        else:
            fused.append(list(s))

    notes = [
        Note(start=s[0], end=s[1], midi=int(round(s[2])), confidence=s[3])
        for s in fused
    ]

    # 2) fusion des notes consécutives de même hauteur séparées par un trou court
    merged: list[Note] = []
    for n in notes:
        if (
            merged
            and n.midi == merged[-1].midi
            and n.start - merged[-1].end <= gap_s
        ):
            merged[-1].end = n.end
            merged[-1].confidence = max(merged[-1].confidence, n.confidence)
        else:
            merged.append(n)

    # recalage temporel : centrer les débuts/fins de note sur la trame
    frame_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    for n in merged:
        n.start = max(0.0, n.start - frame_dt / 2)
        n.end = n.end + frame_dt / 2
    return merged
