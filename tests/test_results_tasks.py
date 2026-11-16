"""The results page by TASK: listen, view, adjust, export.

The page used to be one undifferentiated column (play symbols, an export
block, five edit buttons, one "options" box mixing export settings with
the minimum-note threshold that drove the re-read). These tests pin the
organisation a singer follows:

- the controls sit in three named groups, each holding its own actions;
- the playback buttons NAME what they play (Piano / Voice / Together);
- the reading surface is a choice of three named views, one shown;
- the advanced settings start collapsed and split Export from
  Transcription, because "minimum note" re-reads the evidence;
- the export action says what it will write.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qapp):
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    try:
        yield win.page_results
    finally:
        win.close()


def _inside(group, widget) -> bool:
    return group.isAncestorOf(widget)


# --------------------------------------------------------------------------- #
# The three task groups hold their own actions
# --------------------------------------------------------------------------- #


def test_the_controls_are_grouped_by_task(page):
    from v2m.core.i18n import tr

    assert page.group_listen.title() == tr("ui group listen")
    assert page.group_adjust.title() == tr("ui group adjust")
    assert page.group_export.title() == tr("ui group export")

    for widget in (page.btn_play, page.btn_voice, page.btn_both, page.slider_volume):
        assert _inside(page.group_listen, widget)
    for widget in (
        page.btn_undo,
        page.btn_redo,
        page.btn_replace_phrase,
        page.btn_audition_original,
        page.combo_edit_states,
    ):
        assert _inside(page.group_adjust, widget)
    for widget in (page.btn_export, page.chk_pdf, page.options_toggle):
        assert _inside(page.group_export, widget)
    # a control belongs to ONE group (the old single options box mixed them)
    assert not _inside(page.group_export, page.btn_undo)
    assert not _inside(page.group_adjust, page.spin_min_note)


def test_rows_that_leave_the_task_read_as_links(page):
    """Leaving the take is a quiet row, not a second full-width button.

    Four bordered rows (export, all tools, details, another file) carried the
    same weight as the action the singer must press; the leaving rows are now
    text links, and the primary action stays the only plate.
    """
    for link in (page.btn_guide_skip, page.tools_toggle, page.details_toggle, page.btn_open):
        assert "border: none" in link.styleSheet(), link.text()
        assert "background: transparent" in link.styleSheet(), link.text()
    assert "border: 2px solid transparent" in page.btn_play.styleSheet()


# --------------------------------------------------------------------------- #
# Listening: named sources, one writer for the state prefix
# --------------------------------------------------------------------------- #


def test_the_playback_actions_name_what_they_play(page):
    from v2m.core.i18n import tr

    page.sync_play_buttons(None)
    assert page.btn_play.text() == f"▶ {tr('ui play piano')}"
    assert page.btn_voice.text() == f"▶ {tr('ui play voice')}"
    assert page.btn_both.text() == f"▶ {tr('ui play both')}"
    # the words, not a symbol the user has to learn
    assert "🎙" not in page.btn_voice.text()
    assert "🎹" not in page.btn_both.text()

    page.sync_play_buttons("voice")
    assert page.btn_voice.text().startswith("■")
    assert page.btn_play.text().startswith("▶")
    assert tr("ui play voice") in page.btn_voice.text()


def test_a_language_switch_keeps_the_sounding_square(page):
    """One writer: the prefix is state, the label is language."""
    from v2m.core.i18n import set_language, tr

    page.sync_play_buttons("both")
    try:
        set_language("en")
        page.retranslate()
        assert page.btn_both.text() == f"■ {tr('ui play both')}"
        assert page.btn_play.text() == f"▶ {tr('ui play piano')}"
    finally:
        set_language("fr")
        page.retranslate()
    assert page.btn_both.text() == f"■ {tr('ui play both')}"


# --------------------------------------------------------------------------- #
# Viewing: one named surface at a time
# --------------------------------------------------------------------------- #


def test_the_view_switcher_shows_one_named_surface(page):
    from v2m.gui.results_page import VIEW_NOTES, VIEW_STAFF, VIEW_WAVE

    assert page.current_view() == VIEW_NOTES
    assert page.view_stack.currentWidget() is page.piano
    assert page.btn_view_notes.isChecked()
    assert not page.btn_view_staff.isChecked()

    page.set_view(VIEW_STAFF)
    assert page.view_stack.currentWidget() is page.staff
    assert page.btn_view_staff.isChecked()
    assert not page.btn_view_notes.isChecked()  # exclusive: one chip at a time

    page.set_view(VIEW_WAVE)
    assert page.view_stack.currentWidget() is page.waveform

    with pytest.raises(ValueError):
        page.set_view("spectrogram")


def test_the_notation_option_belongs_to_the_notes_view(page):
    from v2m.gui.results_page import VIEW_NOTES, VIEW_STAFF

    page.set_view(VIEW_NOTES)
    assert not page.notation_row.isHidden()
    page.set_view(VIEW_STAFF)
    assert page.notation_row.isHidden()
    page.set_view(VIEW_NOTES)
    assert not page.notation_row.isHidden()


# --------------------------------------------------------------------------- #
# Adjusting: the advanced settings split the two stages
# --------------------------------------------------------------------------- #


def test_the_advanced_settings_split_export_from_transcription(page):
    from v2m.core.i18n import tr

    assert page.options_host.isHidden()  # collapsed until asked for
    page.options_toggle.setChecked(True)
    assert not page.options_host.isHidden()
    assert page.options_toggle.text() == tr("ui hide advanced settings")
    page.options_toggle.setChecked(False)
    assert page.options_host.isHidden()
    assert page.options_toggle.text() == tr("ui advanced settings")

    assert page.grp_export_opts.title() == tr("ui export options")
    assert page.grp_transcription.title() == tr("ui opts transcription")
    # what changes the WRITTEN file
    assert _inside(page.grp_export_opts, page.chk_quantize)
    assert _inside(page.grp_export_opts, page.spin_transpose)
    # what RE-READS the evidence — the old box filed it as an export option
    assert _inside(page.grp_transcription, page.spin_min_note)
    assert _inside(page.grp_transcription, page.btn_reinterpret)
    assert _inside(page.grp_transcription, page.btn_full_reanalyze)
    assert not _inside(page.grp_export_opts, page.spin_min_note)


# --------------------------------------------------------------------------- #
# Exporting: the action says what it writes
# --------------------------------------------------------------------------- #


def test_archive_comparison_button_is_live_only_and_requires_archives(page, qapp):
    from v2m.core.i18n import tr
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult

    page.set_archive_comparison_available(True)
    page.set_result(
        AnalysisResult(
            source="song.wav",
            duration_s=1.0,
            notes=[Note(0.0, 0.4, 60, 0.9)],
            tempo=None,
            key=None,
        )
    )
    assert page.btn_compare_readings.isHidden()

    page.set_result(
        AnalysisResult(
            source="live",
            duration_s=1.0,
            notes=[Note(0.0, 0.4, 60, 0.9)],
            tempo=None,
            key=None,
        )
    )
    page.set_archive_comparison_available(True)
    assert not page.btn_compare_readings.isHidden()
    assert page.btn_compare_readings.text() == tr("ui compare archived readings")

    page.set_result(
        AnalysisResult(
            source="song.wav",
            duration_s=1.0,
            notes=[Note(0.0, 0.4, 60, 0.9)],
            tempo=None,
            key=None,
        )
    )
    assert page.btn_compare_readings.isHidden()
    assert not page._archive_comparison_available


def test_the_export_action_says_what_it_writes(page):
    from v2m.core.i18n import set_language, tr

    assert page.btn_export.text() == tr("ui export")
    tooltip = page.btn_export.toolTip()
    for written in ("MIDI", "MusicXML", "JSON"):
        assert written in tooltip, tooltip
    try:
        set_language("en")
        page.retranslate()
        assert page.btn_export.text() == "Export the transcription"
        page.options_toggle.setChecked(True)
        assert page.options_toggle.text() == "− Hide settings"
    finally:
        set_language("fr")
        page.retranslate()
