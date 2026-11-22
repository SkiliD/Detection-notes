"""The window follows the current language — including outside the widgets.

File dialogs and result labels use the active language for titles,
filters, keys and figures.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core import i18n
from v2m.core.i18n import set_language, tr


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _restore_language(monkeypatch, tmp_path):
    """Sandbox the language and theme preferences for each GUI test."""
    from v2m.gui import main_window, widgets

    monkeypatch.setattr(i18n, "LANG_FILE", tmp_path / "lang")
    original_theme_file = main_window.THEME_FILE
    monkeypatch.setattr(main_window, "THEME_FILE", tmp_path / "theme")
    widgets.set_theme("dark")
    yield
    set_language("fr")
    widgets.set_theme("dark")
    main_window.THEME_FILE = original_theme_file


def test_saved_theme_is_applied_and_selection_persists(qapp, monkeypatch, tmp_path):
    from v2m.gui import main_window as mw
    from v2m.gui import widgets

    theme_file = tmp_path / "theme"
    theme_file.write_text("light", encoding="utf-8")
    monkeypatch.setattr(mw, "THEME_FILE", theme_file)
    window = mw.MainWindow()
    try:
        assert window._theme == "light"
        assert widgets.current_theme() == "light"
        assert window.page_home.theme_selector.currentData() == "light"

        window._on_theme_changed("dark")

        assert theme_file.read_text(encoding="utf-8") == "dark"
        assert widgets.current_theme() == "dark"
        assert window.page_home.theme_selector.currentData() == "dark"
        assert qapp.styleSheet() == widgets.theme_stylesheet()
    finally:
        window.close()


def test_file_dialog_speaks_the_current_language(qapp, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from v2m.gui import main_window as mw

    seen: dict[str, str] = {}

    def fake_get_open_file_name(parent, title, directory, selected_filter):  # noqa: ANN001
        seen["title"] = title
        seen["filter"] = selected_filter
        return "", ""

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(fake_get_open_file_name)
    )
    win = mw.MainWindow()
    try:
        set_language("fr")
        win.on_open_file()
        assert seen["title"] == tr("ui choose file")
        assert "Tous les fichiers" in seen["filter"]

        set_language("en")
        win.on_open_file()
        assert seen["title"] == "Choose an audio file"
        assert "All files" in seen["filter"]

        save_dialog: dict[str, str] = {}

        def fake_get_save_file_name(
            parent, caption, directory, file_filter, initialFilter, options=None
        ):  # noqa: ANN001
            # the fake mirrors PyQt6's real signature with NO **kwargs: an
            # invalid keyword (e.g. the PyQt5-era selectedFilter) raises
            # TypeError here exactly as the real binding does
            save_dialog.update(
                title=caption,
                filename=directory,
                filters=file_filter,
                initial_filter=initialFilter,
            )
            return "", ""

        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", staticmethod(fake_get_save_file_name)
        )
        win._choose_export_destination(
            mw.AnalysisResult(
                source="/music/song.wav",
                duration_s=1.0,
                notes=[],
                tempo=None,
                key=None,
            )
        )
        assert save_dialog["title"] == "Save export"
        assert save_dialog["filename"].endswith("song.mid")
        assert "MIDI (*.mid)" in save_dialog["filters"]
        assert "MusicXML score (*.musicxml)" in save_dialog["filters"]
        assert save_dialog["initial_filter"] == save_dialog["filters"].split(";;")[0]

        set_language("fr")
        win._choose_export_destination(
            mw.AnalysisResult(
                source="/music/chanson.wav",
                duration_s=1.0,
                notes=[],
                tempo=None,
                key=None,
            )
        )
        assert save_dialog["title"] == "Enregistrer l'export"
        assert "Partition MusicXML (*.musicxml)" in save_dialog["filters"]
    finally:
        win.close()


def test_file_analysis_outcome_preserves_audio_and_cache_status(qapp, monkeypatch):
    """The file outcome's samples reach the waveform and its cache fact
    remains visible after the completion handler's other status messages."""
    import numpy as np

    from v2m.core.pipeline import AnalysisResult, FileAnalysis
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    audio = np.linspace(-0.2, 0.2, 64, dtype=np.float32)
    result = AnalysisResult(
        source="take.wav", duration_s=64 / 22_050, notes=[], tempo=None, key=None,
    )
    outcome = FileAnalysis(analysis=result, audio=audio, cache_used=True)
    win._analysis_started_at = 0.0
    seen_audio = []
    real_set_signal = win.page_results.waveform.set_signal

    def _capture_audio(signal, sr):
        seen_audio.append(signal)
        real_set_signal(signal, sr)

    monkeypatch.setattr(win.page_results.waveform, "set_signal", _capture_audio)
    monkeypatch.setattr(win, "_export_results", lambda: None)
    monkeypatch.setattr(win, "_coherence_summary", lambda _res: "coherence verdict")

    try:
        win._on_analysis_done(outcome)
        assert win._signal_cache["take.wav"] is audio
        assert seen_audio == [audio]
        cache_prefix = tr("ui status cache").split("{", 1)[0].strip()
        assert win.statusBar().currentMessage().startswith(cache_prefix)
    finally:
        win.close()


def test_both_languages_define_the_same_keys():
    """A key added on one side only would fall back to the key (i.e. English)."""
    assert set(i18n.STRINGS["fr"]) == set(i18n.STRINGS["en"])


def test_results_panel_translates_key_label_and_figures(qapp):
    """The panel shows translated labels, not the canonical storage.

    Key labels and figures use translated display text while the report
    keeps canonical values internally.
    """
    from types import SimpleNamespace

    from v2m.gui.widgets import ResultsPanel

    note = SimpleNamespace(midi=60, figure="noire", start=0.0, end=0.5)
    res = SimpleNamespace(
        source="take.wav",
        duration_s=1.0,
        tempo=None,
        meter=None,
        key=SimpleNamespace(
            best=SimpleNamespace(label="C major", score=0.91),
            alternatives=[SimpleNamespace(label="A minor", score=0.74)],
        ),
        chord=None,
        notes=[note],
    )
    panel = ResultsPanel()
    try:
        set_language("fr")
        panel.set_results(res)
        # the value slots ARE the i18n keys (SLOT_KEYS is the key tuple),
        # so the tr() key guard sees every panel label statically
        assert panel._values["key key"].text() == "C majeur"
        assert panel._values["key alternatives"].text() == "A mineur"
        assert panel._values["key figures"].text() == "noire ×1"

        set_language("en")
        panel.set_results(res)
        assert panel._values["key key"].text() == "C major"
        assert panel._values["key figures"].text() == "quarter ×1"
    finally:
        panel.deleteLater()


def test_results_panel_shows_the_tuning_offset_with_advice(qapp):
    """The global tuning offset uses measured tuning language at each band.

    In tune <= 10 c, to watch <= 25 c, else show direction and measured
    cents without inferring a singer's physical state. Without an offset,
    the line stays at '—'.
    """
    from types import SimpleNamespace

    from v2m.gui.widgets import ResultsPanel

    res = SimpleNamespace(
        source="take.wav", duration_s=1.0, tempo=None, meter=None,
        key=None, chord=None, notes=[],
        tuning_offset_cents=-14.0,
    )
    panel = ResultsPanel()
    try:
        set_language("fr")
        panel.set_results(res)
        txt = panel._values["key tuning offset"].text()
        assert "-14,0 c" in txt and "bas" in txt, txt
        # in tune: no injunction, just the encouragement
        panel.set_results(SimpleNamespace(**{**res.__dict__, "tuning_offset_cents": 5.0}))
        assert "juste" in panel._values["key tuning offset"].text()
        # A strong deviation gives a direction and the measured cents only.
        panel.set_results(SimpleNamespace(**{**res.__dict__, "tuning_offset_cents": 31.0}))
        txt = panel._values["key tuning offset"].text()
        assert "nettement aigu (+31 c)" in txt
        assert "détendez" not in txt
        # a take without an offset measure reads 0.0 (the absorbed offset
        # default) — the line stays informative, never empty
        panel.set_results(SimpleNamespace(**{**res.__dict__, "tuning_offset_cents": None}))
        assert "+0,0 c" in panel._values["key tuning offset"].text()
    finally:
        panel.deleteLater()


def test_results_panel_shows_the_per_take_intonation_report(qapp):
    """The intonation row reads the SAME `intonation.summarize` the JSON
    block carries: share of in-tune notes, a named tendency (only when
    MIN_NOTES notes and MIN_CENTS back it), and the register leans."""
    from types import SimpleNamespace

    from v2m.core.i18n import set_language
    from v2m.gui.widgets import ResultsPanel

    # 12 notes, all sharp-leaning: each register band holds MIN_NOTES notes,
    # so the mid and high leans are announced while the low band (a 2.5 c
    # mean) stays silent — the summary invents nothing
    notes = [
        SimpleNamespace(midi=55, deviation=0.02, figure="noire"),
        SimpleNamespace(midi=57, deviation=0.03, figure="noire"),
        SimpleNamespace(midi=59, deviation=0.02, figure="noire"),
        SimpleNamespace(midi=60, deviation=0.03, figure="noire"),
        SimpleNamespace(midi=61, deviation=0.05, figure="noire"),
        SimpleNamespace(midi=63, deviation=0.10, figure="noire"),
        SimpleNamespace(midi=64, deviation=0.15, figure="noire"),
        SimpleNamespace(midi=65, deviation=0.25, figure="noire"),
        SimpleNamespace(midi=67, deviation=0.28, figure="noire"),
        SimpleNamespace(midi=69, deviation=0.30, figure="noire"),
        SimpleNamespace(midi=71, deviation=0.30, figure="noire"),
        SimpleNamespace(midi=72, deviation=0.28, figure="noire"),
    ]
    res = SimpleNamespace(
        source="take.wav", duration_s=1.0, tempo=None, meter=None,
        key=None, chord=None, notes=notes,
    )
    panel = ResultsPanel()
    try:
        set_language("fr")
        panel.set_results(res)
        txt = panel._values["key intonation"].text()
        assert "50 % de notes justes" in txt, txt
        assert "tendance" in txt and "au-dessus" in txt, txt
        assert "médiums" in txt and "+14 c" in txt, txt
        assert "aigus" in txt and "+29 c" in txt, txt
        assert "graves" not in txt, txt  # the low band's mean is 2.5 c: silent
        assert "position : fin +26 c" in txt, txt
        assert "début" not in txt, txt  # the first half averages 4 c: silent

        set_language("en")
        panel.set_results(res)
        txt = panel._values["key intonation"].text()
        assert "50% in tune" in txt, txt
        assert "above" in txt and "mid +14 c" in txt, txt
        assert "end +26 c" in txt, txt
    finally:
        set_language("fr")
        panel.deleteLater()


def test_results_panel_names_the_halves_of_a_drifting_take(qapp):
    """A take that starts flat and ends sharp: the take-wide mean is zero, so
    the tendency reads steady and ONLY the position line locates the drift.

    Both halves are announced — that is the whole point of the line.
    """
    from types import SimpleNamespace

    from v2m.core.i18n import set_language
    from v2m.gui.widgets import ResultsPanel

    notes = [
        SimpleNamespace(midi=60 + index, deviation=deviation, figure="noire")
        for index, deviation in enumerate([-0.30] * 4 + [0.30] * 4)
    ]
    res = SimpleNamespace(
        source="take.wav", duration_s=1.0, tempo=None, meter=None,
        key=None, chord=None, notes=notes,
    )
    panel = ResultsPanel()
    try:
        set_language("fr")
        panel.set_results(res)
        txt = panel._values["key intonation"].text()
        assert "stable" in txt, txt
        assert "position : début -30 c · fin +30 c" in txt, txt

        set_language("en")
        panel.set_results(res)
        txt = panel._values["key intonation"].text()
        assert "start -30 c · end +30 c" in txt, txt
    finally:
        set_language("fr")
        panel.deleteLater()


def test_results_panel_intonation_stays_silent_without_evidence(qapp):
    """No notes, no report: the row shows the placeholder, never a
    fabricated percentage."""
    from types import SimpleNamespace

    from v2m.core.i18n import set_language
    from v2m.gui.widgets import ResultsPanel

    panel = ResultsPanel()
    try:
        set_language("fr")
        panel.set_results(
            SimpleNamespace(
                source="take.wav", duration_s=1.0, tempo=None, meter=None,
                key=None, chord=None, notes=[],
            )
        )
        assert panel._values["key intonation"].text() == "—"
    finally:
        set_language("fr")
        panel.deleteLater()
