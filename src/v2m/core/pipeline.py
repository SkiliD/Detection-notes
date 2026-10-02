"""Full analysis pipeline: audio -> notes + tempo + key + chords -> exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
from .keys import KeyResult, detect_key, snap_to_key
from .meter import detect_meter, dotted_pulse_tempo
from .notes import DEFAULT_MIN_NOTE_S, DEFAULT_SEMITONES, Note, interpret
from .observation import FORMAT_VERSION, VoiceObservation
from . import takes
from .onsets import compute_onset_strengths
from .pitch import PitchTrack
from .rests import Rest, detect_rests
from .tempo import BPM_RANGE, TempoResult, detect_tempo, refine_bpm_with_onsets


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
    tuning_offset_st: float = 0.0  # the take's systematic offset, absorbed before rounding
    # the live/final decomposition (core.compare.breakdown), attached by
    # LiveTake.finalize when the take has a live reading: ONE source for
    # the JSON block, the results-panel explainer and the per-take archive
    live_vs_final: Optional[dict] = None
    manual_pitch_corrections: bool = False

    def quantize_step(self, division: int = 16) -> float:
        """Seconds per grid division (16 = sixteenth) at the detected tempo."""
        if self.grid is not None:
            return self.grid.step_seconds(per_beat=division / 4.0)
        return 0.0


@dataclass
class FileAnalysis:
    """One file analysis and the exact decoded audio that produced it.

    `cache_used` reports whether the file's validated Observation cache
    served this reading. The audio is returned as mono float32 at the
    analysis sample rate, so a presentation adapter can draw its waveform
    without decoding the source a second time.
    """

    analysis: AnalysisResult
    audio: np.ndarray
    cache_used: bool


# One policy for the analysis thresholds: the dataclass, the CLI parsers,
# the GUI and LiveTake all read these — a default written twice had already
# started to drift (the GUI hard-coded its own 0.6).
FILE_CONF = 0.6  # offline analysis: the whole file refines the pitch
LIVE_CONF = 0.5  # live capture: keep faint notes while tracking


@dataclass
class AnalyzeOptions:
    min_conf: float = FILE_CONF
    min_note_s: float = DEFAULT_MIN_NOTE_S  # notes.py owns the floor
    # notes.py owns this threshold too: the hysteresis band is calibrated on
    # the truth corpus at 0.5 (every test and the chant.wav golden
    # fingerprint read it) — a 0.6 written here once detached the offline
    # defaults from the tested ones and lost one note of chant.wav (28 @
    # 155.1 BPM, the documented regression signature, vs 29 @ ~171).
    semitones: float = DEFAULT_SEMITONES
    # the fallback range is tempo's own constant: pipeline only aggregates it
    bpm_range: tuple[float, float] = BPM_RANGE
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
    # ONE piece of evidence per signal: the cleaned pitch curve, the attacks
    # (consonants, glottal stops, vowel changes — what separates a real note
    # start from a legato glide), the frame clock and the accent envelope.
    # Built once here, consumed by the interpretation below, and kept on the
    # result: a take can be reinterpreted without decoding or re-running pYIN
    # (scripts/reinterpret.py).
    observation = VoiceObservation.from_signal(
        y, sr, min_conf=opts.min_conf, use_attacks=opts.use_attacks
    )
    return interpret_observation(
        observation, tempo=tempo, source=source, opts=opts, report=report
    )


def interpret_observation(
    observation: VoiceObservation,
    tempo: Optional[TempoResult],
    source: str,
    opts: AnalyzeOptions,
    report: Callable[[float, str], None],
) -> AnalysisResult:
    """Read the musical facts from READY evidence — no decode, no pYIN.

    This is the interpretation seam: `analyze_signal` fills the observation
    from a signal and calls this, and `scripts/reinterpret.py` calls it on a
    saved observation to re-read the SAME evidence with other thresholds.
    `tempo` is the caller's (a signal's beat tracker, an existing report, or
    None when neither exists — notes, and only notes, are produced then).
    """
    duration_s = observation.duration_s
    report(0.75, tr("progress notes"))
    notes = interpret(
        observation,
        min_note_s=opts.min_note_s,
        semitones=opts.semitones,
    )
    # the take's own tuning reference is measured inside interpret (on the
    # plateau pitches, before rounding) and left on the observation it read:
    # one number, reported in the JSON as cents
    tuning_offset = observation.tuning_offset_st

    report(0.90, tr("progress key"))
    key = detect_key(notes)
    # context-aware pitch snap (AGENTS.md b): a note that sits almost
    # exactly between two semitones resolves toward whichever neighbour IS
    # a degree of the detected key — the global tuning offset (notes.py)
    # cannot see a mistuned degree, the scale can. No-op without a key.
    if key is not None:
        snap_to_key(notes, key.best)

    report(0.91, tr("progress chords"))
    chord = detect_chords(notes)
    chord_segments = detect_chord_segments(notes) if chord else []

    report(0.92, tr("progress refine"))
    # the onset strengths feed BOTH the dotted-pulse tempo correction below
    # and the meter detection: compute them once, before the tempo is
    # finalized (they used to be filled after the refine — the correction
    # then read all-zero strengths and never fired on a real take)
    compute_onset_strengths(observation, notes)
    if tempo and tempo.bpm > 0 and len(notes) >= 3:
        # beat_track returns a rounded BPM (e.g. 117.5 for a true 120):
        # refine it against note onsets to rebuild an exact grid, required
        # by figure labeling and the score.
        refined = refine_bpm_with_onsets(
            np.array([n.start for n in notes], dtype=float), tempo.bpm
        )
        # AFTER the refine, not before: beat_track lands on an INTEGER
        # multiple of the dotted pulse (the onsets live on its grid), so the
        # refine stays exact on that grid. The correction is the one the
        # live estimate applies (meter.dotted_pulse_tempo) — a compound take
        # publishes the same tempo live and in the final analysis.
        refined = dotted_pulse_tempo(notes, refined)
        t0 = float(tempo.beats[0]) if len(tempo.beats) else 0.0
        beat_times = np.arange(t0, duration_s, 60.0 / refined)
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
        # accents (relative onset strengths) feed the 2/4 vs 3/4 vs 4/4
        # decision — already computed above, before the tempo correction
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
        tuning_offset_st=tuning_offset,
        source=source,
        duration_s=duration_s,
        notes=notes,
        tempo=tempo,
        key=key,
        # the observation's curve is the ONLY one: it is already cleaned (the
        # transition mask included), so the result does not carry a second,
        # unused copy of the raw curve
        pitch=observation.pitch,
        meter=meter,
        note_positions=positions,
        num_measures=num_measures,
        chord=chord,
        chord_segments=chord_segments,
        chord_progression=progression,
        rests=rests,
        grid=grid,
        sr=observation.sample_rate,
        observation=observation,
    )


def analyze_file(
    path: str,
    opts: AnalyzeOptions | None = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> FileAnalysis:
    """Analyze one audio file and return its result, audio, and cache fact.

    Decode and cache validation happen once here. Callers that need the
    original samples (for example, to draw the waveform) use this same
    buffer instead of decoding the file again.
    """
    opts = opts or AnalyzeOptions()
    y = audio_io.load_file(path)
    # The take's own cached observation (<take>.observation.npz, written by
    # a previous analysis or by scripts/reinterpret.py --save) skips the
    # pYIN pass when the audio has not changed since: `matches_take` checks
    # the sample rate, the decoded duration and the recorded size+mtime. A
    # stale or absent cache falls back to the full evidence build — the
    # cache can only ever SAVE time, never change a decision: same evidence
    # object, same interpretation.
    observation = cached_observation(path, y, audio_io.SR, opts)
    cache_used = observation is not None
    report = progress or (lambda f, m: None)
    report(0.05, tr("progress tempo"))
    if observation is None:
        report(0.20, tr("progress pitch"))
        observation = VoiceObservation.from_signal(
            y, audio_io.SR, min_conf=opts.min_conf, use_attacks=opts.use_attacks
        )
        # auto-save: THIS analysis becomes the NEXT one's cache (the pYIN
        # cost is paid once per take, not once per analysis). Best-effort:
        # a read-only folder must never cost the analysis the user waited
        # for. `matches_take` will validate it through the audio's own
        # size+mtime, recorded here from the file on disk.
        try:
            stat = Path(path).stat()
            observation.source_size = stat.st_size
            observation.source_mtime = stat.st_mtime
            observation.save(takes.evidence_path(path))
        except OSError:
            pass
    result = interpret_observation(
        observation,
        tempo=detect_tempo(y, audio_io.SR, bpm_range=opts.bpm_range),
        source=path,
        opts=opts,
        report=progress or (lambda f, m: None),
    )
    return FileAnalysis(analysis=result, audio=y, cache_used=cache_used)


def cached_observation(
    path: str,
    y: np.ndarray,
    sr: int,
    opts: AnalyzeOptions,
) -> Optional[VoiceObservation]:
    """A take's cached observation, or None when it must be (re)computed.

    The cache does not merely SAVE time: it must read the SAME evidence
    `from_signal` would have built. Any divergence — attacks absent
    (`use_attacks=False`), a modified take, a different rate, a `.npz` of
    another layout — and the cache is ignored: better to re-run pYIN (the
    cost we wanted to avoid) than to transcribe from wrong evidence.
    `use_attacks` compares the PRESENCE of attacks (None <-> not-None),
    never their count: an empty count is legitimate evidence.
    """
    cache = takes.evidence_path(path)
    if not cache.exists():
        return None
    try:
        # a cache of another LAYOUT (an older or future .npz) is refused
        # HERE, never in `VoiceObservation.load`: the reinterpret script
        # must stay able to read evidence one version old, while the cache
        # cannot afford to guess what the arrays mean
        with np.load(cache, allow_pickle=False) as data:
            version = (
                int(data["format_version"])
                if "format_version" in set(data.files)
                else 0
            )
        if version != FORMAT_VERSION:
            return None
        observation = VoiceObservation.load(cache)
    except (OSError, ValueError, KeyError):
        # an unreadable cache is an absent cache: NEVER let a corrupt file
        # cost the user their analysis
        return None
    if observation.matches_take(path, sr, len(y) / sr):
        if opts.use_attacks != (observation.attacks is not None):
            return None
        return observation
    return None
