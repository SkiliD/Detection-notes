"""The home page is a MASTHEAD, not a form.

The page holds one bounded measure, centred in the window, whose type runs
flush left: a heavy rule, the headline, the deck, a hairline, the three-step
standfirst, a second hairline, then the actions. The primary take-off spans
the measure and the two secondary ones share a row under it (a column once
their translated labels no longer fit). Language, appearance and follow-ups
share the quiet footer below the actions, so settings read as secondary
controls.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core import coherence as coh
from v2m.core.i18n import set_language
from v2m.gui import main_window, widgets
from v2m.gui.home_page import ACTION_COLUMN_MAX


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def driver_note():
    """What the OS reports for the next capture; a test sets `value`."""

    class _Note:
        value: str | None = None

        def __call__(self, name: str) -> str | None:
            return self.value

    return _Note()


@pytest.fixture
def page(qapp, tmp_path, monkeypatch, driver_note):
    from v2m.gui.home_page import HomePage

    monkeypatch.setattr(main_window, "THEME_FILE", tmp_path / "theme")
    widgets.set_theme("dark")
    qapp.setStyleSheet(widgets.theme_stylesheet())

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
        driver_note=driver_note,
    )
    set_language("fr")
    home.retranslate()  # labels size the buttons; an empty button collapses
    home.resize(1280, 780)
    home.show()
    for _ in range(3):
        qapp.processEvents()
    yield home
    home.close()
    set_language("fr")
    widgets.set_theme("dark")


def test_the_home_names_driver_effects_for_the_next_capture(page, driver_note):
    """A capture the driver may process is named under the chosen endpoint."""
    driver_note.value = "Le pilote Windows publie une chaîne d'effets pour ce micro"
    page.set_input_device_status("Microphone (RODE NT-USB)")
    assert page.input_device_note.isVisible()
    assert page.input_device_note.text() == driver_note.value

    # an endpoint nobody reported, or one the OS cannot identify: no note
    driver_note.value = None
    page.set_input_device_status("Microphone (Camo)")
    assert not page.input_device_note.isVisible()


def test_singing_is_the_first_primary_action(page):
    from v2m.gui.widgets import action_button_style

    # the primary action and the pair live in different hosts: compare page
    # coordinates, never each button's local y
    live_top = page.btn_live.mapTo(page, page.btn_live.rect().topLeft()).y()
    file_top = page.btn_file.mapTo(page, page.btn_file.rect().topLeft()).y()
    finder_top = page.btn_finder.mapTo(page, page.btn_finder.rect().topLeft()).y()
    assert live_top < file_top
    # the two secondary take-offs share one row, file before finder
    assert file_top == finder_top
    assert page.btn_file.x() < page.btn_finder.x()
    assert page.btn_live.styleSheet() == action_button_style(primary=True)
    assert page.btn_file.styleSheet() == action_button_style(primary=False)
    assert page.btn_live.nextInFocusChain() is page.btn_file
    assert page.btn_file.nextInFocusChain() is page.btn_finder


@pytest.mark.parametrize("language", ["fr", "en"])
def test_home_invitation_and_next_microphone_retranslate(page, language):
    from v2m.core.i18n import set_language, tr

    set_language(language)
    page.set_input_device_status("Studio mic", fell_back=True)
    page.retranslate()
    assert page.title.text() == tr("ui home title")
    assert page.subtitle.text() == tr("ui app subtitle")
    assert page.steps.text() == tr("ui steps")
    assert page.input_device_status.text() == tr("ui input device fallback next").format(
        device="Studio mic"
    )
    assert page.input_device.accessibleName() == tr("ui home microphone")
    assert page.calibration_card.isHidden()
    assert page.coherence_text.isHidden()
    set_language("fr")
    page.retranslate()


def test_home_actions_keep_their_own_navigation_callbacks(page):
    from v2m.gui.home_page import HomePage

    calls = []
    home = HomePage(
        open_file=lambda: calls.append("file"),
        start_live=lambda: calls.append("live"),
        open_finder=lambda: calls.append("finder"),
        toggle_language=lambda: None,
        snapshot_factory=page._snapshot_factory,
        show_status=lambda *_: None,
    )
    try:
        assert calls == []
        for button in (home.btn_live, home.btn_file, home.btn_finder):
            button.click()
        assert calls == ["live", "file", "finder"]
    finally:
        home.close()


def test_the_home_is_a_masthead_with_one_left_edge(page):
    """Heavy rule, headline, deck, hairline, standfirst, hairline, actions.

    A page has ONE left edge: centering each block on its own axis is what
    made the screen read as a stack of panels.
    """
    from PyQt6.QtCore import Qt

    measure_left = page._measure.mapTo(page, page._measure.rect().topLeft()).x()
    for label in (page.title, page.subtitle, page.steps):
        assert bool(label.alignment() & Qt.AlignmentFlag.AlignLeft), "flush left"
        assert label.mapTo(page, label.rect().topLeft()).x() == measure_left
    title_top = page.title.mapTo(page, page.title.rect().topLeft()).y()
    deck_bottom = page.subtitle.mapTo(page, page.subtitle.rect().bottomLeft()).y()
    steps_bottom = page.steps.mapTo(page, page.steps.rect().bottomLeft()).y()
    deck_rule = page._deck_rule.mapTo(page, page._deck_rule.rect().bottomLeft()).y()
    actions_rule = page._actions_rule.mapTo(
        page, page._actions_rule.rect().bottomLeft()
    ).y()
    assert page._masthead_rule.mapTo(page, page._masthead_rule.rect().topLeft()).y() < (
        title_top
    )
    assert deck_bottom <= deck_rule <= steps_bottom
    assert steps_bottom <= actions_rule <= page.btn_live.mapTo(
        page, page.btn_live.rect().topLeft()
    ).y()
    # the top rule carries weight, the separators stay hairlines
    assert page._masthead_rule.height() > page._deck_rule.height()


def test_the_primary_action_spans_the_measure_and_the_pair_splits_it(page):
    """One measure: the primary plate spans it, the pair halves it."""
    width = page._measure.width()
    assert 0 < width <= ACTION_COLUMN_MAX, f"measure {width} escapes the column"
    assert page.btn_live.width() == width
    assert page.btn_file.width() == page.btn_finder.width()
    assert abs(page.btn_file.width() + page.btn_finder.width() + 12 - width) <= 2
    # coordinates live in the button's local space: map to the page
    left = page.btn_live.mapTo(page, page.btn_live.rect().topLeft()).x()
    right = left + page.btn_live.width()
    assert left > 0 and right < page.width(), "the column must not touch the edges"
    # centered: equal margins on both sides (2 px of rounding tolerance)
    assert abs(left - (page.width() - right)) <= 2


@pytest.mark.parametrize("language", ["fr", "en"])
def test_the_secondary_pair_stacks_when_its_labels_do_not_fit(page, language):
    """A narrow window drops the pair to a column instead of overflowing."""
    from PyQt6.QtWidgets import QApplication

    set_language(language)
    page.retranslate()
    page.resize(320, 700)
    for _ in range(4):
        page._arrange_footer()
        page.scroll.widget().layout().activate()
        page.scroll.widget().adjustSize()
        QApplication.processEvents()
    assert not page._actions_paired
    assert page.btn_file.mapTo(page, page.btn_file.rect().topLeft()).y() < (
        page.btn_finder.mapTo(page, page.btn_finder.rect().topLeft()).y()
    )
    assert page.btn_file.width() == page._measure.width()
    assert page.btn_finder.width() == page._measure.width()
    set_language("fr")
    page.retranslate()


def test_the_quiet_controls_share_one_footer_row(page):
    """Language and follow-ups form one row, on the headline's left edge."""
    buttons = (page.btn_lang, page.btn_coherence, page.btn_calibrate)
    assert {button.styleSheet() for button in buttons} == {widgets.quiet_button_style()}
    assert len({button.y() for button in buttons}) == 1
    assert all(button.height() < page.btn_file.height() for button in buttons)
    left = buttons[0].mapTo(page, buttons[0].rect().topLeft()).x()
    measure_left = page._measure.mapTo(page, page._measure.rect().topLeft()).x()
    assert abs(left - measure_left) <= 2, "the strip starts on the column's edge"


def test_the_footer_row_sits_below_the_action_column(page):
    """Order: actions first, follow-ups after — never interleaved."""
    bottom = page.btn_finder.mapTo(page, page.btn_finder.rect().bottomLeft()).y()
    footer_top = page.btn_lang.mapTo(page, page.btn_lang.rect().topLeft()).y()
    assert footer_top >= bottom - 2

@pytest.mark.parametrize("language", ["fr", "en"])
@pytest.mark.parametrize("width", [320, 360, 400, 480, 640, 800, 1024])
def test_home_controls_fit_on_narrow_and_laptop_windows(page, language, width):
    """Translated controls fit without overflow and keep the footer compact."""
    from PyQt6.QtWidgets import QApplication

    set_language(language)
    page.retranslate()
    page.resize(width, 700)
    for _ in range(4):
        page._arrange_footer()
        page.scroll.widget().layout().activate()
        page.scroll.widget().adjustSize()
        page.scroll.updateGeometry()
        QApplication.processEvents()
    assert page.scroll.horizontalScrollBar().maximum() == 0
    # one row while the labels fit; the strip spans the measure either way
    available = page.width() - 18 - page.scroll.verticalScrollBar().sizeHint().width()
    assert page._footer_row.width() == min(ACTION_COLUMN_MAX, available)
    footer_left = page._footer_row.mapTo(page, page._footer_row.rect().topLeft()).x()
    measure_left = page._measure.mapTo(page, page._measure.rect().topLeft()).x()
    assert abs(footer_left - measure_left) <= 2
    for widget in (
        page._actions_host,
        page.btn_live,
        page.btn_file,
        page.btn_finder,
        page._footer_row,
        page.btn_lang,
        page.btn_coherence,
        page.btn_calibrate,
        page.input_device,
        page.input_device_status,
    ):
        rect = widget.rect()
        top_left = widget.mapTo(page, rect.topLeft())
        assert top_left.x() >= 0, (language, width, widget, top_left.x())
        assert top_left.x() + rect.width() <= page.width(), (
            language,
            width,
            widget,
            top_left.x(),
            rect.width(),
        )
    assert page.input_device.width() > 0
    assert page.input_device_status.wordWrap()
    assert page.input_device_status.maximumWidth() <= width
