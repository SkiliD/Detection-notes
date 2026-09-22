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
from .grid import BeatGrid, Meter
from .i18n import tr
from .keys import KeyResult, detect_key
from .meter import detect_meter
from .notes import Note, compute_onset_strengths, track_to_notes
from .observation import VoiceObservation
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
    grid: Optional[BeatGrid] = None  # the metric frame: seconds <-> beats
    sr: int = audio_io.SR
    observation: VoiceObservation | None = None

    def quantize_step(self, division: int = 16) -> float:
        """Seconds per grid division (16 = sixteenth) at the detected tempo."""
        if self.grid is not None:
            return self.grid.step_seconds(per_beat=division / 4.0)
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

    report(0.05, tr("progress tempo"))  # GUI messages stay user-facing
    tempo = detect_tempo(y, sr, bpm_range=opts.bpm_range)

    report(0.20, tr("progress pitch"))
    track = track_pitch_offline(y, sr)

    report(0.75, tr("progress notes"))
    # attacks (consonants, glottal stops, vowel changes) are what separates a
    # real note start from a legato glide: the pitch curve alone cannot
    attacks = detect_attacks(y, sr) if opts.use_attacks else None
    observation = VoiceObservation(
        pitch=PitchTrack(
            times=track.times.copy(),
            hz=track.hz.copy(),
            confidence=track.confidence.copy(),
        ),
        attacks=attacks,
        duration_s=len(y) / sr,
    )
    notes = track_to_notes(
        track,
        sr,
        min_conf=opts.min_conf,
        min_note_s=opts.min_note_s,
        semitones=opts.semitones,
        attacks=attacks,
    )

    report(0.90, tr("progress key"))
    key = detect_key(notes)

    report(0.91, tr("progress chords"))
    chord = detect_chords(notes)
    chord_segments = detect_chord_segments(notes) if chord else []

    report(0.92, tr("progress refine"))
    if tempo and tempo.bpm > 0 and len(notes) >= 3:
        # beat_track returns a rounded BPM (e.g. 117.5 for a true 120):
        # refine it against note onsets to rebuild an exact grid, required
        # by figure labeling and the score.
        refined = refine_bpm_with_onsets(
            np.array([n.start for n in notes], dtype=float), tempo.bpm
        )
        t0 = float(tempo.beats[0]) if len(tempo.beats) else 0.0
        beat_times = np.arange(t0, len(y) / sr, 60.0 / refined)
        tempo = TempoResult(bpm=round(refined, 2), beats=beat_times)

    report(0.95, tr("progress figures"))
    meter: Optional[Meter] = None
    positions: Optional[list] = None
    num_measures = 0
    progression: list = []
    grid: Optional[BeatGrid] = None
    if tempo and tempo.bpm > 0:
        # the figures only need the beat, so they are labeled as soon as the
        # tempo is known; the measure-aware grid comes right after, once the
        # accent evidence has decided the meter
        grid = BeatGrid(tempo.bpm)
        label_notes(notes, grid)
        # accents (relative onset strengths) feed the 3/4 vs 4/4 decision
        compute_onset_strengths(notes, y, sr)
        meter = detect_meter(notes, tempo.bpm)
        grid = grid.with_meter(meter)
        positions = grid.positions(notes)
        num_measures = grid.measures(notes)
        progression = detect_chord_progression(notes, grid, num_measures)

    report(0.96, tr("progress rests"))
    # breaths between phrases, written as rests on the score. An addition,
    # never a prerequisite: any failure here degrades to no rests at all
    # instead of costing the analysis the user already waited for.
    try:
        rests = detect_rests(notes, grid)
    except (TypeError, ValueError, AttributeError):
        rests = []

    report(1.0, tr("progress done"))
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
        grid=grid,
        sr=sr,
        observation=observation,
    )


def analyze_file(
    path: str,
    opts: AnalyzeOptions | None = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> AnalysisResult:
    y = audio_io.load_file(path)
    return analyze_signal(y, audio_io.SR, source=path, opts=opts, progress=progress)
