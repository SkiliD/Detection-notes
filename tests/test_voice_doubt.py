"""The voice-doubt fact: thin voicing evidence is PUBLISHED, never masked.

pYIN's confidence is a measurement (ADR 0005), so a note whose plateau frames
mostly sat under `VOICE_DOUBT_CONFIDENCE` is still a note: exported, editable,
counted. The reading only says the evidence under it was thin — on the note
(JSON + notes table) and in the audit count — and the marking pass itself is
read-only, which is what keeps frozen segmentation references out of it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from v2m.core.audit import voice_doubt_count
from v2m.core.notes import VOICE_DOUBT_CONFIDENCE, interpret
from v2m.core.observation import VoiceObservation
from v2m.core.pitch import PitchTrack

DT = 0.02  # one frame = 20 ms (the offline hop)


def _observation(conf: float, frames: int = 100) -> VoiceObservation:
    """A fabricated C4 plateau: no audio, no pYIN, no cleanup."""
    times = np.arange(frames) * DT
    return VoiceObservation(
        pitch=PitchTrack(
            times=times,
            hz=np.full(frames, 261.63),
            confidence=np.full(frames, conf),
        ),
        attacks=None,
        duration_s=frames * DT,
        sample_rate=22_050,
        frame_dt=DT,
    )


def test_a_note_on_thin_voicing_is_flagged_but_kept():
    """The doubt NEVER costs the note: no confidence mask exists here."""
    notes = interpret(_observation(0.45))
    assert [n.midi for n in notes] == [60]
    assert notes[0].voice_doubt is True
    assert voice_doubt_count(type("Result", (), {"notes": notes})()) == 1


def test_a_well_voiced_note_carries_no_doubt():
    notes = interpret(_observation(0.90))
    assert notes[0].voice_doubt is False
    assert voice_doubt_count(type("Result", (), {"notes": notes})()) == 0


def test_the_line_itself_is_not_a_doubt():
    """The boundary is a MEASUREMENT line: AT it, the evidence still counts."""
    notes = interpret(_observation(VOICE_DOUBT_CONFIDENCE))
    assert notes[0].voice_doubt is False


def test_marking_the_doubt_never_moves_a_note(monkeypatch):
    """The pass reads the track and writes one boolean — geometry untouched."""
    import v2m.core.notes as notes_mod

    reference = [(n.start, n.end, n.midi) for n in interpret(_observation(0.45))]
    monkeypatch.setattr(notes_mod, "_mark_voice_doubt", lambda notes, track: None)
    unflagged = interpret(_observation(0.45))
    assert [(n.start, n.end, n.midi) for n in unflagged] == reference


def test_the_report_carries_the_doubt(tmp_path):
    """A reader of the JSON sees the thin evidence instead of a sure note."""
    import json

    from v2m.core import i18n
    from v2m.core.grid import BeatGrid
    from v2m.core.keys import KeyCandidate, KeyResult
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.core.report import export_result
    from v2m.core.tempo import TempoResult

    i18n.set_language("fr")
    result = AnalysisResult(
        source="thin.wav",
        duration_s=1.5,
        notes=[
            Note(0.1, 0.4, 60, 0.45, voice_doubt=True),
            Note(0.5, 0.8, 64, 0.9),
        ],
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 1.5, 0.5)),
        key=KeyResult(KeyCandidate("C", "major", 0.9), []),
        grid=BeatGrid(120.0),
        pitch=None,
    )
    export_result(result, tmp_path / "thin")
    report = json.loads((tmp_path / "thin.json").read_text(encoding="utf-8"))
    assert [n["voice_doubt"] for n in report["notes"]] == [True, False]


def _live_note(quality: float):
    """Hold C4 for 0.8 s at a given periodicity, then let the mute run close it."""
    from v2m.core.liveseg import LiveNoteSegmenter

    seg = LiveNoteSegmenter()
    t = 0.0
    for _ in range(40):
        seg.push(t, 261.63, quality=quality)
        t += 0.02
    for _ in range(10):
        seg.push(t, float("nan"), quality=quality)
        t += 0.02
    return seg.notes[-1]


def test_a_live_note_on_borderline_periodicity_is_flagged():
    """The live twin of the offline doubt: published at closure, never a cut."""
    note = _live_note(0.30)  # under MIN_PITCH_QUALITY, above the doubt line
    assert note.voice_doubt is True
    assert note.midi == 60
    assert note.duration > 0.5


def test_a_clean_live_note_carries_no_doubt():
    assert _live_note(0.05).voice_doubt is False


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_notes_table_names_the_doubt_in_words(qapp):
    """A per-note fact, in words: a colour alone would not carry it."""
    from v2m.core.grid import BeatGrid
    from v2m.core.i18n import set_language, tr
    from v2m.core.keys import KeyCandidate, KeyResult
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.core.tempo import TempoResult
    from v2m.gui.widgets import ResultsPanel

    set_language("fr")
    panel = ResultsPanel()
    panel.set_results(
        AnalysisResult(
            source="thin.wav",
            duration_s=1.5,
            notes=[
                Note(0.1, 0.4, 60, 0.45, voice_doubt=True),
                Note(0.5, 0.8, 64, 0.9),
            ],
            tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 1.5, 0.5)),
            key=KeyResult(KeyCandidate("C", "major", 0.9), []),
            grid=BeatGrid(120.0),
            pitch=None,
        )
    )
    table = panel._notes_table
    assert table.item(0, 4).text() == tr("ui doubt voice")
    assert table.item(1, 4).text() == "—"


def test_the_roll_outlines_a_doubtful_note_without_moving_it(qapp):
    """A dashed outline: the cue is a SHAPE, and the bar keeps its geometry."""
    from PyQt6.QtCore import Qt

    from v2m.core.notes import Note
    from v2m.gui.widgets import PianoRoll

    roll = PianoRoll()
    doubtful = Note(0.1, 0.4, 60, 0.45, voice_doubt=True)
    sure = Note(0.5, 0.8, 64, 0.9)
    roll.set_notes([doubtful, sure])
    pens = [item.opts["pen"].style() for item in roll._bar_items]
    assert pens[0] == Qt.PenStyle.DashLine
    assert pens[1] == Qt.PenStyle.NoPen
    assert [(n.start, n.end, n.midi) for n in (doubtful, sure)] == [
        (0.1, 0.4, 60),
        (0.5, 0.8, 64),
    ]


def test_a_repaint_keeps_the_doubt_outline(qapp):
    """The cue survives the selection repaint (it used to be reset to NoPen)."""
    from PyQt6.QtCore import Qt

    from v2m.core.notes import Note
    from v2m.gui.widgets import PianoRoll

    roll = PianoRoll()
    doubtful = Note(0.1, 0.4, 60, 0.45, voice_doubt=True)
    roll.set_notes([doubtful])
    roll.set_notes([doubtful], force=True)  # the rebuild repaints every bar
    assert roll._bar_items[0].opts["pen"].style() == Qt.PenStyle.DashLine
