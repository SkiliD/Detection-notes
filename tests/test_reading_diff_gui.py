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


# --------------------------------------------------------------------------- #
# Success shows the old reading; failure keeps it
# --------------------------------------------------------------------------- #


def test_a_successful_reinterpretation_shows_the_old_reading(qapp):
    win = _window()
    before = win.result
    assert before is not None
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


def test_archived_live_readings_open_in_a_separate_comparison(qapp, monkeypatch, tmp_path):
    import json

    from v2m.core import takes
    from v2m.core.takes import automatic_reading_dir, corrected_reading_dir

    monkeypatch.setattr(takes, "LIVE_TAKES_DIR", tmp_path / "prises_live")
    monkeypatch.setattr(
        "v2m.gui.live_session_controller.live_take_name",
        lambda: "live_20261027_120000",
    )
    win = _window()
    captured = {}
    win.page_results.show_archived_comparison = lambda labels, compare: captured.update(
        labels=labels, compare=compare
    )
    try:
        stem = win.take_ctl.export_stem()
        win.take_ctl.handle_export({}, replay=False)
        take_dir = takes.live_export_dir(stem)
        automatic_json = automatic_reading_dir(take_dir) / f"{stem}.json"
        corrected_json = corrected_reading_dir(take_dir, 1) / f"{stem}.json"
        automatic_json.parent.mkdir(parents=True)

        def report(midis):
            return {
                "duration_s": 1.5,
                "tempo": {"bpm": 120.0},
                "notes": [
                    {
                        "start_s": start,
                        "end_s": end,
                        "midi": midi,
                        "confidence": 0.9,
                    }
                    for start, end, midi in (
                        (0.1, 0.4, midis[0]),
                        (0.5, 0.8, midis[1]),
                    )
                ],
            }

        automatic_json.write_text(json.dumps(report([60, 64])), encoding="utf-8")
        live_result = _result(source="live")
        win.result = live_result
        win._show_result(live_result)
        assert win.page_results.btn_compare_readings.isHidden()

        corrected_json.parent.mkdir(parents=True)
        corrected_json.write_text(json.dumps(report([60, 65])), encoding="utf-8")
        win._refresh_archive_comparison()
        assert not win.page_results.btn_compare_readings.isHidden()
        win.on_open_archived_comparison()

        assert captured["labels"] == ("Automatique", "Correction v001")
        comparison = captured["compare"](0, 1)
        assert comparison.count("near") == 1
        assert win.result is live_result
        assert [note.midi for note in live_result.notes] == [60, 64, 67]
    finally:
        win.close()


def test_archive_comparison_dialog_compares_selected_readings(qapp):
    from v2m.core.i18n import tr
    from v2m.core.reading_diff import compare_readings
    from v2m.gui.results_page import ArchiveComparisonDialog

    dialog = ArchiveComparisonDialog(
        ("Automatic", "Correction v001"),
        lambda before, after: compare_readings(
            _result(middle=64 if before == 0 else 65),
            _result(middle=64 if after == 0 else 65),
        ),
        None,
    )
    try:
        dialog.before.setCurrentIndex(0)
        dialog.after.setCurrentIndex(1)
        dialog.compare_button.click()

        assert "1" in dialog.summary.text()
        assert tr("ui compare archive no coherence") in dialog.note.text()
        assert "arrondi proche" in dialog.details.toPlainText()
        assert [note.midi for note in dialog.before_roll._last_notes] == [60, 64, 67]
        assert [note.midi for note in dialog.after_roll._last_notes] == [60, 65, 67]
        before_range = dialog.before_roll.getPlotItem().vb.viewRange()
        after_range = dialog.after_roll.getPlotItem().vb.viewRange()
        assert before_range == after_range

        dialog.after.setCurrentIndex(0)
        dialog.compare_button.click()
        assert dialog.summary.text() == tr("ui compare choose two")
        assert not dialog.after_roll._last_notes
    finally:
        dialog.close()


def test_notes_of_the_current_result_are_never_rewritten_by_a_diff(qapp):
    win = _window()
    before = win.result
    assert before is not None
    after = _result(middle=66)
    win._publish_reading_diff(before, after)
    # both results keep their own notes (the diff is visual only)
    assert [n.midi for n in before.notes] == [60, 64, 67]
    assert [n.midi for n in after.notes] == [60, 66, 67]


def test_failed_comparison_drops_the_previous_reading_diff(qapp, monkeypatch):
    """A new reading never keeps a diagnostic computed for an older reading."""
    import v2m.core.reading_diff as reading_diff

    win = _window()
    before = win.result
    assert before is not None
    win._publish_reading_diff(before, _result(middle=65))
    assert win.page_results.results._diff_facts is not None
    assert win.page_results.piano._ghost_items

    current = _result(middle=65)
    win.result = current
    win._show_result(current)

    def fail_comparison(*_args, **_kwargs):
        raise RuntimeError("comparison unavailable")

    monkeypatch.setattr(reading_diff, "compare_readings", fail_comparison)

    win._publish_reading_diff(current, _result(middle=66))

    assert win.page_results._reading_diff is None
    assert win.page_results.results._diff_facts is None
    assert not win.page_results.piano._ghost_items
    assert win.page_results.label_diff_legend.isHidden()
    # A diagnostic failure cannot replace or otherwise affect the reading.
    assert win.result is current
    assert [note.midi for note in current.notes] == [60, 65, 67]
    assert [note.midi for note in before.notes] == [60, 64, 67]


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


@pytest.mark.parametrize(
    ("operation", "expected_intent", "expected_action", "acknowledgement_key"),
    [
        ("correct", "CorrectPitchIntent", "correction", "ui status note corrected"),
        ("move", "MoveNoteIntent", "move", "ui status note moved"),
        ("delete", "DeleteNotesIntent", "deletion", "ui status notes deleted"),
    ],
)
def test_direct_note_edits_share_successful_publication(
    qapp, monkeypatch, operation, expected_intent, expected_action, acknowledgement_key
):
    """Every direct edit shares publication after one successful transaction."""
    from v2m.gui import main_window as mw

    win = _window()
    before = win.result
    assert before is not None
    win.stack.setCurrentWidget(win.page_results)
    win._publish_reading_diff(before, _result(middle=65))
    assert win.page_results.results._diff_facts is not None
    if operation == "delete":
        win.page_results.results._notes_table.selectRow(1)

    calls = []
    actual_apply = mw.apply_note_edit

    def capture_transaction(result, history, **kwargs):
        assert win.page_results.results._diff_facts is not None
        outcome = actual_apply(result, history, **kwargs)
        publication_order.append("transaction")
        calls.append((result, history, kwargs["intent"], outcome))
        return outcome

    monkeypatch.setattr(mw, "apply_note_edit", capture_transaction)
    publication_order = []
    auditions = []
    original_clear = win._clear_reading_diff
    original_show = win._show_result

    def track_clear():
        publication_order.append("clear")
        original_clear()

    def track_show(result, signal=None):
        publication_order.append("show")
        original_show(result, signal)

    monkeypatch.setattr(win, "_clear_reading_diff", track_clear)
    monkeypatch.setattr(win, "_show_result", track_show)
    original_audition = win._start_move_audition

    def track_audition(before_note, after_note):
        publication_order.append("audition")
        auditions.append((before_note, after_note))
        original_audition(before_note, after_note)

    monkeypatch.setattr(win, "_start_move_audition", track_audition)
    try:
        if operation == "correct":
            win.on_correct_note(1, 65)
        elif operation == "move":
            win.on_note_moved(1, 0.55, 0.85, 65)
        else:
            win.on_delete_selection()

        assert len(calls) == 1
        original, history, intent, outcome = calls[0]
        assert original is before
        assert history is win._edit_history
        assert type(intent).__name__ == expected_intent
        published = win.result
        assert published is outcome.result
        assert published is not None and published is not before
        if operation == "correct":
            assert published.notes[1].midi == 65
        elif operation == "move":
            assert (published.notes[1].start, published.notes[1].end) == (0.55, 0.85)
        else:
            assert [note.midi for note in published.notes] == [60, 67]
        assert win.page_results.results._diff_facts is None
        expected_order = ["transaction", "clear", "show"]
        if operation == "move":
            expected_order.append("audition")
        assert publication_order == expected_order
        assert win._edit_history.undo_depth == 1
        from v2m.core.i18n import note_display_pair, tr

        assert win.page_results.combo_edit_states.itemText(0) == tr(expected_action)

        if operation == "delete":
            expected_message = tr(acknowledgement_key).format(n=1)
        else:
            expected_message = tr(acknowledgement_key).format(note=note_display_pair(65))
        assert win.statusBar().currentMessage() == expected_message
        assert win.error_banner.isHidden()
        assert outcome.before.result.notes[1].midi == 64
        if operation == "move":
            assert win._auditions.active_kind == "drag"
            assert len(auditions) == 1
            assert auditions[0][0] is outcome.before.result.notes[1]
            assert auditions[0][1] is published.notes[1]
    finally:
        win.close()


def test_failed_note_edit_preserves_published_result_history_and_diff(qapp, monkeypatch):
    """A core failure is reported without publishing or clearing presentation."""
    from v2m.core.edits import snapshot_state
    from v2m.core.i18n import tr

    win = _window()
    before = win.result
    assert before is not None
    win.stack.setCurrentWidget(win.page_results)
    win.page_results.results._notes_table.selectRow(1)
    win._edit_history.record(snapshot_state(_result(60), None, action="move"))
    win._edit_history.record(snapshot_state(_result(62), None))
    assert win._edit_history.undo(snapshot_state(before, None)) is not None
    win._sync_edit_history()
    win._publish_reading_diff(before, _result(middle=65))
    diff = win.page_results.results._diff_facts
    assert diff is not None
    history_rows = win._edit_history.entries(snapshot_state(before, None))
    history_identity = [
        (id(state) if state is not None else None, action) for state, action in history_rows
    ]

    def fail_after_candidate_mutation(candidate, notes, **_kwargs):
        candidate.notes = notes
        candidate.key = "candidate only"
        raise RuntimeError("derived facts failed")

    from v2m.core import edits

    monkeypatch.setattr(edits, "refresh_result_facts", fail_after_candidate_mutation)

    try:
        win.on_correct_note(1, 65)

        assert win.result is before
        assert win.page_results.results._diff_facts is diff
        restored_rows = win._edit_history.entries(snapshot_state(before, None))
        assert [
            (id(state) if state is not None else None, action) for state, action in restored_rows
        ] == history_identity
        assert win._edit_history.can_undo and win._edit_history.can_redo
        assert tr("err note correction failed").split("{e}")[0] in win.error_banner.text()
    finally:
        win.close()


def test_a_new_file_analysis_clears_the_diff(qapp, monkeypatch, tmp_path):
    win = _window()
    win._publish_reading_diff(win.result, _result(middle=65))
    before_facts = win.page_results.results._diff_facts
    assert before_facts is not None
    # the analysis-done boundary ALSO clears (a new take is never an "after")
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


def test_the_legend_follows_the_notes_view(qapp):
    """Ghost bars live on the roll: their legend goes with that surface.

    Reading the score or the waveform with a legend about bar colours
    would explain something that is not on screen; coming back to the
    notes view brings the comparison back with it.
    """
    from v2m.gui.results_page import VIEW_NOTES, VIEW_STAFF

    win = _window()
    page = win.page_results
    win._publish_reading_diff(win.result, _result(middle=65))
    assert page.label_diff_legend.isVisibleTo(page)

    page.set_view(VIEW_STAFF)
    assert page.label_diff_legend.isHidden()
    page.set_view(VIEW_NOTES)
    assert page.label_diff_legend.isVisibleTo(page)
    # the comparison itself is kept, never recomputed by switching views
    assert page.results._diff_facts is not None


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
    monkeypatch.setattr(
        "v2m.core.take_export.write_take",
        lambda *a, **k: pytest.fail("an export on the diff path"),
    )
    win._publish_reading_diff(win.result, _result(middle=65))
    assert True  # reaching here means write_take was never called


@pytest.fixture
def cwd_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path
