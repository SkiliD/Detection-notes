"""The home page is a MENU, not a form: one bounded, centered column.

Two layout regressions pinned here (both render fine and both read badly):

1. The three take-off buttons used to stretch FULL BLEED across the window:
   a wall of equal bars reads as a form to fill in, not as three choices.
   They now share one column capped at ``ACTION_COLUMN_MAX`` and centered.
2. The language and coherence buttons used to each take their own
   full-width row below the actions — five stacked bars of the same weight.
   They now share ONE quiet footer row, side by side, half the action
   height: settings are not take-off points.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core import coherence as coh
from v2m.gui.home_page import ACTION_COLUMN_MAX


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qapp):
    from v2m.gui.home_page import HomePage

    home = HomePage(
        open_file=lambda: None,
        start_live=lambda: None,
        open_finder=lambda: None,
        toggle_language=lambda: None,
        snapshot_factory=lambda: coh.EvidenceSnapshot(
            root=coh.LIVE_TAKES_DIR,
            takes=[],
            sessions=[],
            days=[],
            practice_windows=[],
            health_weeks=[],
            regression=None,
            journal={},
        ),
        show_status=lambda *_: None,
    )
    home.retranslate()  # labels size the buttons; an empty button collapses
    home.resize(1280, 780)
    home.show()
    for _ in range(3):
        qapp.processEvents()
    yield home
    home.close()


def _actions(page):
    return [page.btn_file, page.btn_live, page.btn_finder]


def test_the_three_actions_share_one_bounded_centered_column(page):
    """Equal width, capped at the column measure, centered in the window."""
    width = page.btn_file.width()
    assert 0 < width <= ACTION_COLUMN_MAX, f"action width {width} escapes the column"
    for button in _actions(page):
        assert button.width() == width, "the actions must read as one column"
    # coordinates live in the action host's local space: map to the page
    left = page.btn_file.mapTo(page, page.btn_file.rect().topLeft()).x()
    right = left + width
    assert left > 0 and right < page.width(), "the column must not touch the edges"
    # centered: equal margins on both sides (2 px of rounding tolerance)
    assert abs(left - (page.width() - right)) <= 2


def test_the_quiet_controls_share_one_footer_row(page):
    """Language + coherence side by side, smaller than the actions."""
    same_row = abs(page.btn_lang.y() - page.btn_coherence.y()) <= 2
    assert same_row, "the quiet controls belong on one row"
    assert page.btn_lang.height() < page.btn_file.height(), (
        "footer controls must be quieter than the take-off actions"
    )
    # one centered row of two modest controls — never full-width bars:
    # each button stays well under half the window (they used to each
    # stretch edge to edge in their own row)
    lang_left = page.btn_lang.mapTo(page, page.btn_lang.rect().topLeft()).x()
    right = page.btn_coherence.mapTo(page, page.btn_coherence.rect().topLeft()).x()
    right += page.btn_coherence.width()
    assert page.btn_lang.width() < page.width() / 2
    assert page.btn_coherence.width() < page.width() / 2
    # centered as a trio: equal margins on both sides (2 px of rounding)
    # — the calibration toggle joined the quiet row (2026-10-12 design)
    right = page.btn_calibrate.mapTo(page, page.btn_calibrate.rect().topLeft()).x()
    right += page.btn_calibrate.width()
    assert abs(lang_left - (page.width() - right)) <= 2


def test_the_footer_row_sits_below_the_action_column(page):
    """Order: actions first, follow-ups after — never interleaved."""
    bottom = page.btn_finder.mapTo(page, page.btn_finder.rect().bottomLeft()).y()
    footer_top = page.btn_lang.mapTo(page, page.btn_lang.rect().topLeft()).y()
    assert footer_top >= bottom - 2
