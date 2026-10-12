"""The live/final marking goes through the SINGLE policy (core.compare).

The side-by-side comparison must not own its own matching rule: the GUI
mark and the `live_vs_final` JSON block read the same `match` — same
window (grid sixteenth OR dating lead), same reasons. The test builds a
final result shifted by a semitone and checks the WHOLE final reading is
drawn beside the captured one, with only the moved note accented.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from typing import Any, cast

import pytest

from v2m.core.liveseg import LiveNote
from v2m.core.notes import Note


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class _Seg:
    def __init__(self, notes):
        self.notes = notes


class _Thread:
    def __init__(self, notes):
        self._seg = _Seg(notes)

    @property
    def segmenter(self):
        return self._seg


def test_the_comparison_accents_only_the_moved_notes(qapp):
    from PyQt6.QtCore import Qt

    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        live = [LiveNote(0.0, 0.5, 69), LiveNote(0.6, 1.1, 71)]
        win.live_thread = cast(Any, _Thread(live))
        win._closed_live_notes = []
        win.result = _res()
        win._mark_live_final_diff()
        roll = win.page_live.piano_live
        # the final reading is drawn WHOLE beside the captured one (2 bars),
        # and only the moved note (final 72 vs live 71) carries the accent
        assert len(roll._compare_items) == 2
        assert roll._compare_items[0].opts["pen"].style() == Qt.PenStyle.NoPen
        assert roll._compare_items[1].opts["pen"].style() == Qt.PenStyle.SolidLine
    finally:
        win.live_thread = None
        win.result = None
        win.close()


def _res():
    from v2m.core.grid import BeatGrid
    from v2m.core.keys import KeyCandidate, KeyResult
    from v2m.core.pipeline import AnalysisResult
    from v2m.core.tempo import TempoResult

    return AnalysisResult(
        source="live", duration_s=1.2,
        notes=[Note(0.0, 0.5, 69, 0.9), Note(0.6, 1.1, 72, 0.9)],
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 1.2, 0.5)),
        key=KeyResult(KeyCandidate("A", "minor", 0.9), []),
        grid=BeatGrid(120.0),
    )

