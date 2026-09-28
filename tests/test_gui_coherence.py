"""The coherence verdict in the GUI: visible without running a script.

Goal (the draft's retained intention): at the end of an analysis — live
take OR file — the singer sees the take's coherence verdict on the
results screen. Today that verdict only existed in the JSON and the
CSV; the results panel now carries a "Cohérence" row reading the SAME
`coherence_report` as the JSON block (zero divergence possible), and
the home page carries the follow-up button aggregating the takes into a
sessions table (the `core.coherence` core, the same one as the CLI
script).

Offscreen environment like the other GUI tests.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from v2m.core.grid import BeatGrid
from v2m.core.keys import KeyCandidate, KeyResult
from v2m.core.notes import Note
from v2m.core.observation import VoiceObservation
from v2m.core.pitch import PitchTrack
from v2m.core.tempo import TempoResult


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _observation(hz_override=None) -> VoiceObservation:
    """The curve FOLLOWS the notes (69 then 71): a healthy hold is not a
    suspect — note 2 labeled 71 must not read note 1's curve."""
    times = np.arange(0.0, 1.2, 0.02)
    hz = np.where(times < 0.6, 440.0, 440.0 * 2 ** (2 / 12.0))  # A3 then B3
    if hz_override is not None:
        hz = hz_override(times, hz)
    return VoiceObservation(
        pitch=PitchTrack(
            times=times, hz=hz,
            confidence=np.ones(len(times)),
            transition=np.zeros(len(times), dtype=bool),
        ),
        attacks=None, duration_s=1.2, sample_rate=22_050,
        frame_dt=0.02, accent_envelope=None,
    )


def _result(source: str = "live"):
    from v2m.core.pipeline import AnalysisResult

    return AnalysisResult(
        source=source, duration_s=1.2,
        notes=[Note(0.0, 0.5, 69, 0.9), Note(0.6, 1.1, 71, 0.9)],
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 1.2, 0.5)),
        key=KeyResult(KeyCandidate("A", "minor", 0.9), []),
        grid=BeatGrid(120.0),
        observation=_observation(),
    )


def test_results_panel_shows_the_take_coherence_verdict(qapp):
    """A clean take: the "Cohérence" row says "clean" (propre) with the
    audited note count — no script, no JSON."""
    from v2m.gui.widgets import ResultsPanel

    panel = ResultsPanel()
    res = _result("live")
    panel.set_results(res)
    text = panel._values["key coherence"].text()
    assert "propre" in text or "clean" in text
    assert "2" in text  # the audited count is on the line


def test_results_panel_shows_the_suspects_verdict(qapp):
    """A note hiding a secondary plateau: the row says ⚠ with the count —
    the doubt is made visible where the singer looks."""
    from v2m.gui.widgets import ResultsPanel

    # the second note (71) hides a plateau at +2 st (73) for 0.3 s
    def lift(times, hz):
        hz = hz.copy()
        inside = (times >= 0.7) & (times <= 1.0)
        hz[inside] = 440.0 * 2 ** (4 / 12.0)
        return hz

    res = _result("live")
    res.observation = _observation(lift)
    panel = ResultsPanel()
    panel.set_results(res)
    text = panel._values["key coherence"].text()
    assert "⚠" in text and "1" in text


def test_results_panel_shows_na_without_evidence(qapp):
    """A reinterpretation without observation: "unavailable"
    (indisponible), never a false "clean"."""
    from v2m.gui.widgets import ResultsPanel

    res = _result("chant.wav")
    res.observation = None
    panel = ResultsPanel()
    panel.set_results(res)
    assert "indisponible" in panel._values["key coherence"].text()


def test_home_page_button_aggregates_the_sessions(qapp, monkeypatch, tmp_path):
    """The follow-up button: same core as the script, table on the page —
    the user opens no terminal."""
    from v2m.core import coherence as coh
    from v2m.gui import main_window as mw

    # a clean take + a suspect one in a fake takes folder
    d1 = tmp_path / "live_20260927_090000"
    d1.mkdir()
    (d1 / "live_20260927_090000.coherence.csv").write_text(
        "start_s,duration_s,midi,inner_range_st,secondary_plateau_s,"
        "secondary_plateau_start_s,secondary_plateau_midi,"
        "secondary_plateau_offset_st\n", encoding="utf-8",
    )
    (d1 / "live_20260927_090000.json").write_text(
        '{"coherence": {"checked": 6, "suspects": []}}', encoding="utf-8"
    )
    d2 = tmp_path / "live_20260927_091000"
    d2.mkdir()
    (d2 / "live_20260927_091000.coherence.csv").write_text(
        "start_s,duration_s,midi,inner_range_st,secondary_plateau_s,"
        "secondary_plateau_start_s,secondary_plateau_midi,"
        "secondary_plateau_offset_st\n"
        "1.0,0.6,57,2.0,0.20,0.2,55,2.0\n", encoding="utf-8",
    )
    (d2 / "live_20260927_091000.json").write_text(
        '{"coherence": {"checked": 4, "suspects": []}}', encoding="utf-8"
    )

    monkeypatch.setattr(coh, "LIVE_TAKES_DIR", tmp_path)
    win = mw.MainWindow()
    try:
        win.stack.setCurrentWidget(win.page_home)
        win.on_show_coherence()
        # QStackedWidget subtlety (AGENTS.md): a child of the CURRENT page
        # still reports isVisible()==False on the offscreen platform; the
        # explicit setVisible state is `isHidden()`
        assert not win.coherence_text.isHidden()
        text = win.coherence_text.toPlainText()
        assert "20260927" in text
        # the take count and the suspects land on the table
        assert "1" in text
        win.coherence_text.setVisible(False)
    finally:
        win.close()


def test_home_page_button_with_no_takes_shows_the_invite(qapp, monkeypatch, tmp_path):
    from v2m.core import coherence as coh
    from v2m.gui import main_window as mw

    monkeypatch.setattr(coh, "LIVE_TAKES_DIR", tmp_path)
    win = mw.MainWindow()
    try:
        win.stack.setCurrentWidget(win.page_home)
        win.on_show_coherence()
        assert not win.coherence_text.isHidden()
        assert "Aucune prise" in win.coherence_text.toPlainText() or \
            "No take" in win.coherence_text.toPlainText()
    finally:
        win.close()
