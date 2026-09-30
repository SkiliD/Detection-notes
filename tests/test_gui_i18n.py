"""The window follows the current language — including outside the widgets.

Regression: `QFileDialog.getOpenFileName` received a title and a filter
hardcoded in French; `tr()` never saw them, so the selection dialog
stayed French even in English.
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
    """Sandbox the language state: never touch the real ~/.v2m_lang."""
    monkeypatch.setattr(i18n, "LANG_FILE", tmp_path / "lang")
    yield
    set_language("fr")


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

    Regression: `set_results` placed `res.key.best.label` ('C major') and
    the figures ('noire') raw — English leaked into a French UI and
    French into an English UI, while the staff already translates via
    `tr_key_label`.
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
    """The global tuning offset displays with an intonation advice (user
    request): in tune <= 10 c, to watch <= 25 c, else a "détendez"
    advice; the adjective follows the sign (sharp/flat). Without an
    offset, the line stays at '—'."""
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
        assert "-14.0 c" in txt and "bas" in txt, txt
        # in tune: no injunction, just the encouragement
        panel.set_results(SimpleNamespace(**{**res.__dict__, "tuning_offset_cents": 5.0}))
        assert "juste" in panel._values["key tuning offset"].text()
        # clearly flat: the "détendez" advice
        panel.set_results(SimpleNamespace(**{**res.__dict__, "tuning_offset_cents": 31.0}))
        assert "détendez" in panel._values["key tuning offset"].text()
        # a take without an offset measure reads 0.0 (the absorbed offset
        # default) — the line stays informative, never empty
        panel.set_results(SimpleNamespace(**{**res.__dict__, "tuning_offset_cents": None}))
        assert "+0.0 c" in panel._values["key tuning offset"].text()
    finally:
        panel.deleteLater()
