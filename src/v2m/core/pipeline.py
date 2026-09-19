"""Full analysis pipeline: audio -> notes + tempo + key + chords -> exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import audio_io
from .chords import (
    ChordResult,
    detect_chord_progression,
    detect_chord_segments,
    detect_chords,
)
from .figures import label_notes
from .keys import KeyResult, detect_key
from .meter import Meter, count_measures, detect_meter, note_positions
from .midi_export import quantize_step_s
from .notes import Note, compute_onset_strengths, track_to_notes
from .onsets import detect_attacks
from .pitch import PitchTrack, track_pitch_offline
from .rests import Rest, detect_rests
from .tempo import TempoResult, detect_tempo, refine_bpm_with_onsets


@dataclass
class AnalysisResult:
    source: str
    duration_s: float
    notes: list[Note]
    tempo: Optional[TempoResult]
    key: Optional[KeyResult]
    pitch: Optional[PitchTrack] = None
    meter: Optional[Meter] = None
    note_positions: Optional[list] = None  # (measure, beat) per note
    num_measures: int = 0
    chord: Optional[ChordResult] = None
    chord_segments: Optional[list] = None  # (start_s, end_s, ChordCandidate) — legacy 2 s
    chord_progression: Optional[list] = None  # (m0, m1, ChordCandidate) per measure
    rests: Optional[list[Rest]] = None  # breaths/phrase gaps (lead-in included)
    sr: int = audio_io.SR

    def quantize_step(self, division: int = 16) -> float:
        """Seconds per grid division (16 = sixteenth) at the detected tempo."""
        if self.tempo and self.tempo.bpm > 0:
            return quantize_step_s(self.tempo.bpm, division)
        return 0.0


@dataclass
class AnalyzeOptions:
    min_conf: float = 0.6
    min_note_s: float = 0.060
    semitones: float = 0.6
    bpm_range: tuple[float, float] = (60.0, 180.0)
    use_attacks: bool = True  # attack evidence: tells a note from a glide


def analyze_signal(
    y: np.ndarray,
    sr: int,
    source: str = "signal",
    opts: AnalyzeOptions | None = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> AnalysisResult:
    """Full analysis of a mono float32 signal.

    `progress(fraction, message)` is called regularly (GUI/CLI).
    """
    opts = opts or AnalyzeOptions()
    report = progress or (lambda f, m: None)

    report(0.05, "Détection du tempo…")  # GUI messages stay user-facing
    tempo = detect_tempo(y, sr, bpm_range=opts.bpm_range)

    report(0.20, "Suivi de la hauteur (pYIN)…")
    track = track_pitch_offline(y, sr)

    report(0.75, "Segmentation en notes…")
    # attacks (consonants, glottal stops, vowel changes) are what separates a
    # real note start from a legato glide: the pitch curve alone cannot
    attacks = detect_attacks(y, sr) if opts.use_attacks else None
    notes = track_to_notes(
        track,
        sr,
        min_conf=opts.min_conf,
        min_note_s=opts.min_note_s,
        semitones=opts.semitones,
        attacks=attacks,
    )

    report(0.90, "Détection de la tonalité…")
    key = detect_key(notes)

    report(0.91, "Détection des accords…")
    chord = detect_chords(notes)
    chord_segments = detect_chord_segments(notes) if chord else []

    report(0.92, "Raffinement du tempo…")
    if tempo and tempo.bpm > 0 and len(notes) >= 3:
        # beat_track returns a rounded BPM (e.g. 117.5 for a true 120):
        # refine it against note onsets to rebuild an exact grid, required
        # by figure labeling and the score.
        refined = refine_bpm_with_onsets(
            np.array([n.start for n in notes], dtype=float), tempo.bpm
        )
        t0 = float(tempo.beats[0]) if len(tempo.beats) else 0.0
        grid = np.arange(t0, len(y) / sr, 60.0 / refined)
        tempo = TempoResult(bpm=round(refined, 2), beats=grid)

    report(0.95, "Étiquetage des figures de notes…")
    meter: Optional[Meter] = None
    positions: Optional[list] = None
    num_measures = 0
    progression: list = []
    if tempo and tempo.bpm > 0:
        label_notes(notes, tempo.bpm)
        # accents (relative onset strengths) feed the 3/4 vs 4/4 decision
        compute_onset_strengths(notes, y, sr)
        meter = detect_meter(notes, tempo.bpm)
        # positions on the effective grid: detected meter, else rendered 4/4
        positions = note_positions(notes, tempo.bpm, meter)
        num_measures = count_measures(notes, tempo.bpm, meter)
        progression = detect_chord_progression(notes, tempo.bpm, meter, num_measures)

    report(0.96, "Détection des respirations…")
    # breaths between phrases, written as rests on the score. An addition,
    # never a prerequisite: any failure here degrades to no rests at all
    # instead of costing the analysis the user already waited for.
    try:
        rests = detect_rests(notes, tempo.bpm if tempo else None)
    except Exception:  # noqa: BLE001 - see comment above
        rests = []

    report(1.0, "Terminé")
    return AnalysisResult(
        source=source,
        duration_s=len(y) / sr,
        notes=notes,
        tempo=tempo,
        key=key,
        pitch=track,
        meter=meter,
        note_positions=positions,
        num_measures=num_measures,
        chord=chord,
        chord_segments=chord_segments,
        chord_progression=progression,
        rests=rests,
        sr=sr,
    )


def analyze_file(
    path: str,
    opts: AnalyzeOptions | None = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> AnalysisResult:
    y = audio_io.load_file(path)
    return analyze_signal(y, audio_io.SR, source=path, opts=opts, progress=progress)
