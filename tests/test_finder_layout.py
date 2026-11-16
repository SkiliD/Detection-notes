"""The note finder reads as one answer, and every answer is playable.

Section 5 of the interface study, pinned as behavior:
- the name, the tuning needle and the mic's reception share ONE card
  (three unrelated rows read as three unrelated questions);
- the needle's scale is NAMED (plus grave / juste / plus aigu) while the
  measured cents stay in the detail line;
- the history is a row of playable chips, not an empty box — clicking or
  pressing Space on a chip hears that note again, and Tab reaches it;
- the keyboard shows a span CHOSEN explicitly and never recentres under
  the pointer.

The product contracts this must not weaken: a provisional reading shows
its NAME but lights no key and enters no chip; the history records
confirmed notes only.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _page(played: list[int] | None = None):
    from v2m.gui.finder_page import FinderPage

    played = played if played is not None else []

    def _no_capture():
        raise AssertionError("building the page must not open the mic")

    return FinderPage(
        thread_factory=_no_capture,
        go_home=lambda: None,
        show_error=lambda _msg: None,
        clear_error=lambda: None,
        audition_note=played.append,
    )


def _host(page) -> QWidget:
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(page)
    host.show()
    return host


def _confirm(page, midi: int, cents: float = 0.0) -> None:
    """One block whose reading is CONFIRMED at `midi` (the finder's path)."""
    page._on_note(
        (midi + cents / 100.0, 440.0),
        confirmed=(float(midi), 440.0),
        heard=(float(midi), 440.0),
    )


# --------------------------------------------------------------------------- #
# One central answer
# --------------------------------------------------------------------------- #


def test_the_name_the_needle_and_the_mic_state_share_one_card(qapp):
    """The page's answer is one composition, not three loose rows."""
    page = _page()
    try:
        page.retranslate()
        answer_facts = (
            page.label_note,
            page.tuning,
            page.label_tuning,
            page.level,
            page.label_heard,
        )
        for widget in answer_facts:
            assert widget.parentWidget() is page.box_answer, type(widget).__name__
        from v2m.core.i18n import tr

        assert page.box_answer.title() == tr("ui finder answer")
        # a bounded reading column: the needle keeps a readable scale on a
        # wide window instead of stretching across it
        assert page.box_answer.maximumWidth() <= 640
    finally:
        page.deleteLater()


def test_the_needle_scale_names_both_directions_in_the_active_language(qapp):
    """A dot left of centre reads "plus grave" without knowing the cents."""
    from v2m.core.i18n import get_language, set_language, tr

    page = _page()
    previous = get_language()
    try:
        set_language("fr")
        page.retranslate()
        assert page.label_tuning_low.text() == tr("ui tuning lower")
        assert page.label_tuning_centre.text() == tr("ui tuning centre")
        assert page.label_tuning_high.text() == tr("ui tuning higher")
        set_language("en")
        page.retranslate()
        assert page.label_tuning_low.text() == tr("ui tuning lower")
        assert "♭" in page.label_tuning_low.text()
        assert "♯" in page.label_tuning_high.text()
        # the measured cents stay in the detail line, never replaced by words
        page.set_tuning(-32.0)
        assert "-32" in page.label_tuning.text()
    finally:
        set_language(previous)
        page.deleteLater()


# --------------------------------------------------------------------------- #
# The history is a row of playable chips
# --------------------------------------------------------------------------- #


def test_a_confirmed_note_becomes_a_playable_chip(qapp):
    """Clicking a chip auditions that note; a hold is one chip."""
    played: list[int] = []
    page = _page(played)
    try:
        page.retranslate()
        assert not page.history_chips
        assert page.label_history_empty.text(), "an empty history shows no invitation"
        assert not page.label_history_empty.isHidden()

        _confirm(page, 60)
        _confirm(page, 60)  # the same hold, still one entry
        assert [chip.text() for chip in page.history_chips] == ["Do4"]
        assert page.label_history_empty.isHidden()

        page.history_chips[0].click()
        assert played == [60]
    finally:
        page.deleteLater()


def test_a_chip_answers_the_keyboard(qapp):
    """Space on the focused chip hears it — the history keeps its route."""
    played: list[int] = []
    page = _page(played)
    host = _host(page)
    try:
        page.retranslate()
        _confirm(page, 62)
        _confirm(page, 64)
        chips = page.history_chips
        assert len(chips) == 2

        # a chip created into a shown layout becomes visible on the next
        # event pass; only then can the keyboard reach it (Qt refuses a
        # hidden widget). Tab's route to it is pinned in
        # `test_live_finder_keyboard.py`; this pins the chip's own answer.
        qapp.processEvents()
        assert chips[1].isVisible()
        QTest.keyClick(chips[1], Qt.Key.Key_Space)
        qapp.processEvents()
        assert played and played[-1] == 64
    finally:
        host.close()


def test_the_chips_are_rebuilt_only_when_the_readings_change(qapp):
    """The page refreshes per audio block: identical chips must not churn.

    Rebuilding them at 20 Hz would destroy the widget the singer just
    focused (or clicked), so an unchanged history keeps its widgets.
    """
    page = _page()
    try:
        page.retranslate()
        _confirm(page, 60)
        first = page.history_chips[0]
        _confirm(page, 60)
        _confirm(page, 60)
        assert page.history_chips[0] is first
        _confirm(page, 62)
        assert [chip.text() for chip in page.history_chips] == ["Do4", "Ré4"]
    finally:
        page.deleteLater()


def test_a_growing_history_leaves_no_stale_chip_painted(qapp):
    """A removed layout item keeps painting until it leaves the widget tree.

    Measured: a chip taken out of the layout but still parented drew a
    640x480 band across the page, hiding the answer card.
    """
    from PyQt6.QtWidgets import QPushButton

    page = _page()
    try:
        page.retranslate()
        _confirm(page, 60)
        _confirm(page, 62)
        _confirm(page, 64)
        painted = page.history_row.findChildren(QPushButton)
        assert len(painted) == len(page.history_chips) == 3
    finally:
        page.deleteLater()


def test_a_provisional_reading_never_enters_the_history(qapp):
    """The name answers early; the chip waits for confirmation."""
    page = _page()
    try:
        page.retranslate()
        page._on_note((65.0, 349.2), confirmed=None, heard=(65.0, 349.2))
        assert page.label_note.text() != ""
        assert not page.history_chips, "an unconfirmed reading entered the history"
        assert page.keys._active == set(), "an unconfirmed reading lit a key"
    finally:
        page.deleteLater()


# --------------------------------------------------------------------------- #
# The keyboard's span is chosen, never guessed
# --------------------------------------------------------------------------- #


def test_the_range_selector_lists_named_spans_and_keeps_the_choice(qapp):
    from v2m.core.i18n import get_language, set_language

    page = _page()
    previous = get_language()
    try:
        set_language("fr")
        page.retranslate()
        labels = [page.combo_range.itemText(i) for i in range(page.combo_range.count())]
        assert len(labels) == 3
        assert all("Do" in label or "La" in label for label in labels)
        assert page.keys.min_midi == 48 and page.keys.max_midi == 84  # voice span

        page.combo_range.setCurrentIndex(2)
        qapp.processEvents()
        assert (page.keys.min_midi, page.keys.max_midi) == (21, 108)

        set_language("en")
        page.retranslate()
        assert page.combo_range.currentIndex() == 2, "a language switch reset the span"
        assert "A0" in page.combo_range.currentText()
        assert (page.keys.min_midi, page.keys.max_midi) == (21, 108)
    finally:
        set_language(previous)
        page.deleteLater()


def test_changing_the_span_never_moves_a_key_under_the_pointer(qapp):
    """The lit key and the staged cursor are kept; nothing recentres.

    A recentre on every note would slide the keys under a mouse that is
    about to click one — the span stays where the singer put it.
    """
    page = _page()
    try:
        page.retranslate()
        _confirm(page, 60)
        page.keys.set_keyboard_cursor(60)

        page.combo_range.setCurrentIndex(2)
        qapp.processEvents()
        assert page.keys._active == {60}, "the lit key did not survive the span change"
        assert page.keys.keyboard_cursor == 60, "the staged cursor did not survive"

        # a cursor outside the new span is DROPPED, never slid to an edge
        page.keys.set_keyboard_cursor(60)
        page.combo_range.setCurrentIndex(1)  # C2–C5: 60 is still inside
        assert page.keys.keyboard_cursor == 60
        page.keys.set_keyboard_cursor(30)  # outside every voice span
        page.combo_range.setCurrentIndex(0)
        assert page.keys.keyboard_cursor is None
    finally:
        page.deleteLater()


def test_a_note_outside_the_shown_span_still_names_and_lights(qapp):
    """The span is a VIEW: the page's answer does not depend on it.

    Choosing a narrow keyboard must never silence a note the mic heard —
    the roll would look deaf instead of looking narrow.
    """
    page = _page()
    try:
        page.retranslate()
        page.combo_range.setCurrentIndex(0)  # C3–C6
        _confirm(page, 33)  # A1, far below the shown keys
        assert page.label_note.text() != ""
        assert page.keys._active == {33}
        assert [chip.text() for chip in page.history_chips] == ["La1"]
    finally:
        page.deleteLater()
