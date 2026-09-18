"""Pipeline d'analyse complet : audio -> notes + tempo + tonalité -> MIDI/rapports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import audio_io
from .figures import figures_summary, label_notes
from .keys import KeyResult, detect_key
from .midi_export import build_midi, quantize_step_s
from .music import midi_to_name
from .musicxml import save_musicxml
from .notes import Note, track_to_notes
from .pitch import PitchTrack, track_pitch_offline
from .tempo import TempoResult, detect_tempo, refine_bpm_with_onsets


@dataclass
class AnalysisResult:
    source: str
    duration_s: float
    notes: list[Note]
    tempo: Optional[TempoResult]
    key: Optional[KeyResult]
    pitch: Optional[PitchTrack] = None
    sr: int = audio_io.SR

    def quantize_step(self, division: int = 16) -> float:
        if self.tempo and self.tempo.bpm > 0:
            return quantize_step_s(self.tempo.bpm, division)
        return 0.0


@dataclass
class AnalyzeOptions:
    min_conf: float = 0.6
    min_note_s: float = 0.060
    semitones: float = 0.6
    bpm_range: tuple[float, float] = (60.0, 180.0)


def analyze_signal(
    y: np.ndarray,
    sr: int,
    source: str = "signal",
    opts: AnalyzeOptions | None = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> AnalysisResult:
    """Analyse complète d'un signal mono float32.

    `progress(fraction, message)` est appelé régulièrement (GUI/CLI).
    """
    opts = opts or AnalyzeOptions()
    report = progress or (lambda f, m: None)

    report(0.05, "Détection du tempo…")
    tempo = detect_tempo(y, sr, bpm_range=opts.bpm_range)

    report(0.20, "Suivi de la hauteur (pYIN)…")
    track = track_pitch_offline(y, sr)

    report(0.75, "Segmentation en notes…")
    notes = track_to_notes(
        track,
        sr,
        min_conf=opts.min_conf,
        min_note_s=opts.min_note_s,
        semitones=opts.semitones,
    )

    report(0.90, "Détection de la tonalité…")
    key = detect_key(notes)

    report(0.92, "Raffinement du tempo…")
    if tempo and tempo.bpm > 0 and len(notes) >= 3:
        # beat_track renvoie un BPM arrondi (ex. 117,5 pour un vrai 120) :
        # on le raffine sur les attaques des notes pour reconstituer une
        # grille exacte, indispensable aux figures et à la partition.
        refined = refine_bpm_with_onsets(
            np.array([n.start for n in notes], dtype=float), tempo.bpm
        )
        t0 = float(tempo.beats[0]) if len(tempo.beats) else 0.0
        grid = np.arange(t0, len(y) / sr, 60.0 / refined)
        tempo = TempoResult(bpm=round(refined, 2), beats=grid)

    report(0.95, "Étiquetage des figures de notes…")
    if tempo and tempo.bpm > 0:
        label_notes(notes, tempo.bpm)

    report(1.0, "Terminé")
    return AnalysisResult(
        source=source,
        duration_s=len(y) / sr,
        notes=notes,
        tempo=tempo,
        key=key,
        pitch=track,
        sr=sr,
    )


def analyze_file(
    path: str,
    opts: AnalyzeOptions | None = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> AnalysisResult:
    y = audio_io.load_file(path)
    return analyze_signal(y, audio_io.SR, source=path, opts=opts, progress=progress)


def export_result(
    res: AnalysisResult,
    stem: str | Path,
    quantize_s: float | None = None,
    transpose: int = 0,
    with_pitch_curve: bool = False,
) -> dict:
    """Écrit les quatre artefacts (.mid/.musicxml/.json/.csv) du préfixe `stem`.

    Le dict de rapport JSON a un seul schéma pour tous les contextes (CLI,
    GUI, live) : source, tempo, tonalité, notes avec figures, récapitulatif
    des figures. `with_pitch_curve` ajoute la courbe de hauteur (usage CLI).
    Retourne le dict écrit, pour affichage ou tests.
    """
    if res.tempo is None or res.key is None:
        raise ValueError("export_result exige une analyse complète (tempo + tonalité)")
    stem = Path(stem)

    mid = build_midi(
        res.notes,
        bpm=res.tempo.bpm,
        key=res.key,
        quantize_s=quantize_s,
        transpose=transpose,
    )
    mid.save(str(stem) + ".mid")
    save_musicxml(res.notes, res.tempo.bpm, res.key, str(stem) + ".musicxml")

    rep = {
        "source": res.source,
        "duration_s": round(res.duration_s, 3),
        "tempo": {"bpm": res.tempo.bpm},
        "key": {
            "best": res.key.best.label,
            "score": round(res.key.best.score, 4),
            "alternatives": [
                {"label": a.label, "score": round(a.score, 4)} for a in res.key.alternatives
            ],
        },
        "n_notes": len(res.notes),
        "figures_summary": figures_summary(res.notes),
        "notes": [
            {
                "start_s": round(n.start, 4),
                "end_s": round(n.end, 4),
                "duration_s": round(n.duration, 4),
                "midi": n.midi,
                "name": midi_to_name(n.midi),
                "confidence": round(n.confidence, 3),
                "figure": n.figure,
            }
            for n in res.notes
        ],
        "quantized": bool(quantize_s),
        "transpose_semitones": transpose,
    }
    if with_pitch_curve and res.pitch is not None:
        rep["pitch_curve_hz"] = [
            round(float(v), 2) if v == v else None  # NaN -> null
            for v in res.pitch.hz[::2]
        ]
        rep["pitch_times_s"] = [round(float(t), 4) for t in res.pitch.times[::2]]
    with open(str(stem) + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    with open(str(stem) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "start_s", "end_s", "duration_s", "midi", "name",
                    "confidence", "figure"])
        for i, n in enumerate(res.notes):
            w.writerow([i, round(n.start, 4), round(n.end, 4), round(n.duration, 4),
                        n.midi, midi_to_name(n.midi), round(n.confidence, 3),
                        n.figure or ""])
    return rep
