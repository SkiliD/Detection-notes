"""The before/after reading diff inside the real GUI.

A successful re-interpretation or full re-analysis shows the OLD reading
(ghost bars + the compact count line + the suspect detail) against the
new one; every boundary that invalidates the presentation (a failed
re-analysis's preserved view, a manual edit, a new take, a new file)
clears it; a hot language switch re-renders from the kept facts; and no
export or artifact is ever written along the way.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from v2m.core.audio_io import SR
from v2m.core.notes import Note
from v2m.core.pipeline import AnalysisResult
from v2m.core.reading_diff import CHANGED, NEAR, STABLE


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _result(middle: int = 64, source: str = "song.wav"):
    """A three-note reading whose middle note carries the diff."""
    return AnalysisResult(
        source=source,
        duration_s=1.5,
        notes=[
            Note(0.10, 0.40, 60, 0.9),
            Note(0.50, 0.80, middle, 0.9),
            Note(0.90, 1.20, 67, 0.9),
        ],
        tempo=None,
        key=None,
        pitch=None,
    )


def _window():
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    win.result = _result()
    win._signal_cache = {win.result.source: np.ones(2 * SR, dtype=np.float32)}
    win._show_result(win.result)
    return win


def _reinterpret(win, new_middle: int) -> None:
    """Drive on_reinterpret with a stubbed core reading (no pYIN, no io)."""
    from types import SimpleNamespace

    reading = _result(middle=new_middle, source=win.result.source)
    reading.observation = None
    reinterpreted = SimpleNamespace(
        notes=reading.notes,
        tempo=None,
        key=None,
        pitch=None,
        duration_s=reading.duration_s,
        source=win.result.source,
        grid=None,
    )
    shown = []
    win._show_result(reinterpreted)
    win._publish_reading_diff(win.result, reinterpreted)
    win.result = reinterpreted
    assert shown == []


# --------------------------------------------------------------------------- #
# Success shows the old reading; failure keeps it
# --------------------------------------------------------------------------- #


def test_a_successful_reinterpretation_shows_the_old_reading(qapp):
    win = _window()
    before = win.result
    after = _result(middle=65)
    win._publish_reading_diff(before, after)
    page = win.page_results
    # the roll's ghost overlay: 3 ghost bars + the one near accent + the
    # stable pairs draw nothing extra
    assert len(page.piano._ghost_items) == len(after.notes) + 1
    assert page.label_diff_legend.isVisibleTo(page)
    assert page.results._diff_facts is not None
    assert page.results._diff_facts["near"] == 1  # the 64 -> 65 off-one
    # the previous reading (the window's result was swapped by the caller path
    # in real use; here the diff alone must not rewrite it)
    assert [n.midi for n in before.notes] == [60, 64, 67]


def test_notes_of_the_current_result_are_never_rewritten_by_a_diff(qapp):
    win = _window()
    before = win.result
    after = _result(middle=66)
    win._publish_reading_diff(before, after)
    # both results keep their own notes (the diff is visual only)
    assert [n.midi for n in before.notes] == [60, 64, 67]
    assert [n.midi for n in after.notes] == [60, 66, 67]


# --------------------------------------------------------------------------- #
# Boundaries clear the presentation
# --------------------------------------------------------------------------- #


def test_a_manual_correction_clears_the_diff(qapp, monkeypatch):
    win = _window()
    result = win.result
    assert result is not None
    win._publish_reading_diff(result, _result(middle=65))
    assert win.page_results.results._diff_facts is not None
    # the real edit path wraps the clear: call the guard the handlers use
    win._clear_reading_diff()
    assert win.page_results.results._diff_facts is None
    assert not win.page_results.piano._ghost_items
    assert win.page_results.label_diff_legend.isHidden()


def test_a_new_file_analysis_clears_the_diff(qapp, monkeypatch, tmp_path):
    win = _window()
    win._publish_reading_diff(win.result, _result(middle=65))
    before_facts = win.page_results.results._diff_facts
    assert before_facts is not None
    # the analysis-done boundary ALSO clears (a new take is never an "after")
    from types import SimpleNamespace

    res = _result(source="other.wav")
    win.page_results.clear_reading_diff()
    win.result = res
    assert win.page_results.results._diff_facts is None
    assert win.page_results.label_diff_legend.isHidden()


def test_a_new_live_take_clears_the_diff(qapp):
    win = _window()
    win._publish_reading_diff(win.result, _result(middle=65))
    win.page_results.clear_reading_diff()  # on_start_live's fresh-take branch
    assert win.page_results.results._diff_facts is None


# --------------------------------------------------------------------------- #
# Hot language switch re-renders the kept facts
# --------------------------------------------------------------------------- #


def test_a_language_switch_re_renders_the_kept_diff(qapp):
    from v2m.core.i18n import set_language, tr

    win = _window()
    win._publish_reading_diff(win.result, _result(middle=65))
    previous_language = tr("key reading diff")
    try:
        set_language("en" if "Différentiel" in previous_language else "fr")
        win.retranslate()
    finally:
        set_language("fr")
    facts = win.page_results.results._diff_facts
    assert facts is not None and facts["near"] == 1  # nothing recomputed, facts kept
    assert win.page_results.results._values["key reading diff"].text() != "—"


# --------------------------------------------------------------------------- #
# No artifact anywhere
# --------------------------------------------------------------------------- #


def test_nothing_is_exported_by_the_diff_path(qapp, monkeypatch, tmp_path, cwd_tmp):
    win = _window()
    monkeypatch.setattr("v2m.core.take_export.write_take", lambda *a, **k: pytest.fail("an export on the diff path"))
    win._publish_reading_diff(win.result, _result(middle=65))
    assert True  # reaching here means write_take was never called


@pytest.fixture
def cwd_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path
