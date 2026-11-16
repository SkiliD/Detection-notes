"""The note finder says whether the sung note is in tune or off.

A tuning needle and named advice line sit under the note name, using the
same cents scale as the report
(`tuning_advice` / `tuning_color`, one shared vocabulary), driven by the
same reading that names the note.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QVBoxLayout, QWidget


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _page():
    from v2m.gui.finder_page import FinderPage

    def _no_capture():
        raise AssertionError("building the page must not open the mic")

    return FinderPage(
        thread_factory=_no_capture,
        go_home=lambda: None,
        show_error=lambda _msg: None,
        clear_error=lambda: None,
        audition_note=lambda _midi: None,
    )


class _FakeSegmenter:
    """Answers the two readings `FinderPage._on_note` consumes."""

    def __init__(self, confirmed=None, heard=None):
        self._confirmed = confirmed
        self._heard = heard

    def confirmed(self):
        return self._confirmed

    def heard(self, max_age_s: float = 0.5):
        return self._heard


class _FakeThread:
    def __init__(self, segmenter):
        self.segmenter = segmenter

    def isRunning(self) -> bool:  # noqa: N802 (Qt naming)
        return True


def test_tuning_advice_reads_the_shared_scale(qapp):
    """On the note, to watch, or off — and whether it is sharp or flat."""
    from v2m.core.i18n import tr
    from v2m.gui.widgets import tuning_advice

    assert tuning_advice(0.0) == tr("ui tuning ok")
    assert tuning_advice(-9.9) == tr("ui tuning ok")  # inside the in-tune band
    assert tr("ui tuning sharp") in tuning_advice(18.0)
    assert tr("ui tuning flat") in tuning_advice(-18.0)
    assert tr("ui tuning flat") in tuning_advice(-40.0)


def test_clear_tuning_advice_reports_direction_and_measured_cents(qapp):
    """The strong correction names the measured deviation without guessing why."""
    from v2m.core.i18n import get_language, set_language, tr
    from v2m.gui.widgets import tuning_advice

    previous = get_language()
    try:
        set_language("fr")
        assert tuning_advice(40.0) == tr("ui tuning off").format(
            dir=tr("ui tuning sharp"), cents=40.0
        )
        assert "40 c" in tuning_advice(40.0)
        assert tuning_advice(-40.0) == tr("ui tuning off").format(
            dir=tr("ui tuning flat"), cents=-40.0
        )
        assert "détendez" not in tuning_advice(40.0)

        set_language("en")
        assert tuning_advice(40.0) == tr("ui tuning off").format(
            dir=tr("ui tuning sharp"), cents=40.0
        )
        assert "40 c" in tuning_advice(40.0)
        assert "relax" not in tuning_advice(40.0)
    finally:
        set_language(previous)


def test_set_tuning_names_the_verdict_and_keeps_a_colored_meter(qapp):
    """The text answers just/off; the needle retains the shared cents scale."""
    from v2m.core.i18n import tr
    from v2m.gui.widgets import (
        MATCH_COLOR,
        MISMATCH_COLOR,
        TEXT,
        tuning_advice,
        tuning_color,
    )

    page = _page()
    try:
        page.set_tuning(3.0)
        assert tr("ui tuning ok") in page.label_tuning.text()
        assert "¢" in page.label_tuning.text()
        assert TEXT in page.label_tuning.styleSheet()
        assert page.tuning._cents == 3.0

        page.set_tuning(60.0)
        assert tuning_advice(60.0) in page.label_tuning.text()
        assert TEXT in page.label_tuning.styleSheet()
        assert tuning_color(60.0) == MISMATCH_COLOR
        assert page.tuning._cents == 50.0  # the meter clamps to its ±50 rail

        page.set_tuning(None)
        assert page.label_tuning.text() == ""
        assert page.tuning._cents is None
        assert MATCH_COLOR != MISMATCH_COLOR
    finally:
        page.deleteLater()


def test_the_tuning_follows_the_note_being_sung(qapp, monkeypatch):
    """`_on_note` derives the verdict from the same reading it names."""
    page = _page()
    monkeypatch.setattr(
        page,
        "_thread",
        _FakeThread(_FakeSegmenter(confirmed=(60.0, 261.6), heard=(60.0, 261.6))),
    )
    try:
        # 60.3 is 30 cents sharp of C4: named C4, flagged off
        page._on_note((60.3, 261.6))
        from v2m.core.i18n import note_display_pair, tr

        assert page.label_note.text() == note_display_pair(60)
        assert tr("ui tuning sharp") in page.label_tuning.text()
        assert page.tuning._cents is not None and page.tuning._cents > 0

        # a dead-on reading clears the reproach
        page._on_note((60.0, 261.6))
        assert page.label_tuning.text().endswith(tr("ui tuning ok"))
    finally:
        page.deleteLater()


def test_no_reading_hides_the_verdict(qapp, monkeypatch):
    """A quiet mic says nothing — no stale cents left hanging."""
    page = _page()
    monkeypatch.setattr(page, "_thread", _FakeThread(_FakeSegmenter()))
    try:
        page.set_tuning(12.0)
        assert page.label_tuning.text()
        page._on_note(None)
        assert page.label_tuning.text() == ""
        assert page.tuning._cents is None
    finally:
        page.deleteLater()


def test_the_verdict_is_re_rendered_on_a_language_switch(qapp, monkeypatch):
    """The advice follows the language while a note is still sung."""
    from v2m.core.i18n import get_language, set_language, tr

    page = _page()
    monkeypatch.setattr(page, "_thread", _FakeThread(_FakeSegmenter()))
    previous = get_language()
    try:
        set_language("fr")
        page.set_tuning(-40.0)
        assert tr("ui tuning flat") in page.label_tuning.text()
        set_language("en")
        page.retranslate()
        assert tr("ui tuning flat") in page.label_tuning.text()
    finally:
        set_language(previous)
        page.deleteLater()


def test_the_page_renders_with_the_tuning_row(qapp):
    """Sanity: the new row does not break the page's layout/render."""
    page = _page()
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(page)
    host.show()
    try:
        assert not host.grab().isNull()
        assert page.tuning.parent() is not None
    finally:
        host.close()
