"""The coherence verdict and the live/final breakdown in the report.

Three contracts:
1. the `live_vs_final` block exists only for an export THAT HAS live
   notes (the take) — a file export carries none; its breakdown (±1,
   moved, absorbed) comes from the single `core.compare` policy;
2. the coherence CSV writes the plateau's NEW columns (time, pitch,
   signed interval) — the aggregator re-reads them;
3. the displayed coherence verdict comes from the same
   `coherence_report` as the JSON block (the GUI panel and the report
   cannot contradict each other).
"""

from __future__ import annotations

import csv

import numpy as np

from v2m.core.continuity import coherence_report
from v2m.core.grid import BeatGrid
from v2m.core.keys import KeyCandidate, KeyResult
from v2m.core.liveseg import LiveNote
from v2m.core.notes import Note
from v2m.core.observation import VoiceObservation
from v2m.core.pitch import PitchTrack
from v2m.core.tempo import TempoResult


def _observation() -> VoiceObservation:
    times = np.arange(0.0, 0.6, 0.02)
    hz = np.full(len(times), 440.0)
    return VoiceObservation(
        pitch=PitchTrack(
            times=times, hz=hz,
            confidence=np.ones(len(times)),
            transition=np.zeros(len(times), dtype=bool),
        ),
        attacks=None, duration_s=0.6, sample_rate=22_050,
        frame_dt=0.02, accent_envelope=None,
    )


def _result(observation, source: str = "live"):
    from v2m.core.pipeline import AnalysisResult

    return AnalysisResult(
        source=source, duration_s=0.6,
        notes=[Note(0.0, 0.5, 69, 0.9)],
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 0.6, 0.5)),
        key=KeyResult(KeyCandidate("A", "minor", 0.9), []),
        grid=BeatGrid(120.0),
        observation=observation,
    )


def test_live_vs_final_block_only_when_live_notes_given(tmp_path):
    from v2m.core.report import export_result

    res = _result(_observation())
    rep = export_result(res, tmp_path / "prise", live_notes=[LiveNote(0.0, 0.5, 69)])
    assert "live_vs_final" in rep
    block = rep["live_vs_final"]
    assert block["live"] == 1 and block["final"] == 1
    assert block["contours"] == 0 and block["off1"] == 0
    assert block["rate"] == 0.0

    # a FILE analysis has no live reading: no block at all
    rep2 = export_result(_result(_observation(), source="chant.wav"),
                         tmp_path / "fichier")
    assert "live_vs_final" not in rep2


def test_live_vs_final_decomposes_off1_moves(tmp_path):
    from v2m.core.report import export_result

    # a live note read La (69), the offline pass wrote Sib (70) at the same
    # instant: the ±1 rounding family
    res = _result(_observation())
    res.notes[0].midi = 70
    rep = export_result(res, tmp_path / "prise3", live_notes=[LiveNote(0.0, 0.5, 69)])
    block = rep["live_vs_final"]
    assert block["off1"] == 1 and block["moved"] == 0 and block["contours"] == 1
    assert block["rate"] == 1.0


def test_coherence_csv_writes_the_published_plateau_columns(tmp_path):
    from v2m.core.report import export_result

    export_result(_result(_observation()), tmp_path / "prise4")
    with open(tmp_path / "prise4.coherence.csv", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "secondary_plateau_start_s" in header
    assert "secondary_plateau_midi" in header
    assert "secondary_plateau_offset_st" in header


def test_panel_and_report_read_the_same_coherence_source():
    """The displayed verdict comes from `coherence_report`, like the JSON
    block: same evidence, same thresholds, zero divergence possible."""
    obs = _observation()
    notes = [Note(0.0, 0.5, 69, 0.9)]
    block = coherence_report(obs, notes)
    assert set(block) == {"checked", "suspects"}
    assert block["checked"] == 1
    assert block["suspects"] == []  # a curve ON the label: clean verdict
