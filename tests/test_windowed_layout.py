"""The app in a WINDOWED window: the pages scroll instead of clipping, and no
labeled control of the results page leaves its column.

The window-layout invariants:

- the window is sized FOR the screen it runs on (`gui/window_geometry.py`):
  it opens inside the available rectangle, grows up to a comfort ceiling,
  and its floor follows a smaller screen instead of blocking a shrink;
  no page demands a tall window of its own;
- home, live and finder carry their whole content in `widgets.page_scroll`,
  so a short window reaches the bottom of the page instead of clipping it;
- the results body is side by side only while both columns fit their floors
  (`LEFT_COLUMN_MIN_W` / `RIGHT_COLUMN_MIN_W`), stacked below that, and its
  controls column never needs a horizontal scrollbar to reach a control.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QScrollArea

from v2m.core.audio_io import SR


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _result():
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult

    return AnalysisResult(
        source="song.wav",
        duration_s=1.5,
        notes=[Note(0.1, 0.4, 60, 0.9), Note(0.5, 0.8, 64, 0.9)],
        tempo=None,
        key=None,
        pitch=None,
    )


@pytest.fixture
def win(qapp):
    """A real window showing a two-note take, laid out like the app's."""
    from v2m.gui import main_window as mw

    window = mw.MainWindow()
    result = _result()
    window.result = result
    window._signal_cache = {result.source: np.ones(2 * SR, dtype=np.float32)}
    window._show_result(result)
    window.show()
    for _ in range(4):
        qapp.processEvents()
    yield window
    window.close()


def _settle(qapp):
    for _ in range(4):
        qapp.processEvents()


def _resize(win, qapp, width: int, height: int):
    win.resize(width, height)
    _settle(qapp)


def _page(win, name: str):
    page = getattr(win, f"page_{name}")
    win.stack.setCurrentWidget(page)
    return page


def _shown_views(page) -> list[str]:
    """The reading surfaces actually painted, in switcher order."""
    return [
        name
        for name, surface in (
            ("notes", page.piano),
            ("staff", page.staff),
            ("wave", page.waveform),
        )
        if surface.isVisible()
    ]


def test_the_plan_fits_the_window_to_the_screen_it_runs_on():
    """The opening size and the floor both come FROM the screen.

    The plan uses logical pixels from Qt's `availableGeometry()` after
    scaling. On a 1366×768 laptop at 150% scaling, the opening size and
    minimum are constrained to the available client area, with the floor
    clamped so the window can always fit the screen.
    """
    from v2m.gui.window_geometry import (
        FLOOR_H,
        FLOOR_W,
        MAX_H,
        MAX_W,
        window_plan,
    )

    scaled_laptop = window_plan(911, 512)  # 1366x768 at 150 %
    assert (scaled_laptop.width, scaled_laptop.height) == (887, 440)
    assert (scaled_laptop.min_width, scaled_laptop.min_height) == (887, 440)

    laptop = window_plan(1093, 614)  # 1366x768 at 125 %
    assert (laptop.width, laptop.height) == (909, 542)
    assert (laptop.min_width, laptop.min_height) == (FLOOR_W, 542)

    big = window_plan(1920, 1040)
    assert (big.width, big.height) == (MAX_W, 823)  # grows, up to the ceiling
    assert big.y == 108 and big.x == 160  # centered, not top-left

    huge = window_plan(3840, 2120)  # the ceiling holds
    assert (huge.width, huge.height) == (MAX_W, MAX_H)
    assert (huge.min_width, huge.min_height) == (FLOOR_W, FLOOR_H)

    # the property, over real screen sizes: the window always fits, and the
    # floor is always reachable
    for width, height in (
        (800, 600),
        (1024, 600),
        (1280, 800),
        (1366, 768),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
        (1080, 1920),  # portrait: the window must not demand the width
    ):
        plan = window_plan(width, height)
        assert plan.width <= width and plan.height <= height, (width, height)
        assert plan.height >= plan.min_height and plan.width >= plan.min_width
        assert plan.x >= 0 and plan.y >= 0, (width, height)
        assert plan.x + plan.width <= width and plan.y + plan.height <= height, (
            width,
            height,
        )


def test_the_window_opens_from_the_plan_of_its_own_screen(win, qapp):
    """The window is not a fixed size: it opens where its own screen says."""
    from v2m.gui.window_geometry import window_plan

    rect = qapp.primaryScreen().availableGeometry()
    plan = window_plan(rect.width(), rect.height(), rect.x(), rect.y())
    assert (win.minimumWidth(), win.minimumHeight()) == (
        plan.min_width,
        plan.min_height,
    )
    assert (win.width(), win.height()) == (plan.width, plan.height)
    assert win.width() <= rect.width() and win.height() <= rect.height()
    assert (win.x(), win.y()) == (plan.x, plan.y)


def test_the_floor_follows_the_screen_the_window_sits_on(win, qapp):
    """A window dragged onto a smaller monitor stays shrinkable there.

    The fixed floor was the trap: on a 911x512 desktop the 560 px minimum
    height exceeded the screen itself, so the window could not be made to
    fit at all. Moving back to a large screen restores the full floor.
    """
    from PyQt6.QtCore import QRect

    from v2m.gui.window_geometry import FLOOR_H, FLOOR_W

    class _Screen:
        def __init__(self, width: int, height: int) -> None:
            self._rect = QRect(0, 0, width, height)

        def availableGeometry(self) -> QRect:  # noqa: N802 - Qt's own name
            return self._rect

    win._adapt_floor_to_screen(_Screen(1920, 1040))
    assert (win.minimumWidth(), win.minimumHeight()) == (FLOOR_W, FLOOR_H)
    win._adapt_floor_to_screen(_Screen(911, 512))
    assert (win.minimumWidth(), win.minimumHeight()) == (887, 440)
    _resize(win, qapp, 400, 300)  # below the floor: Qt clamps, never breaks
    assert (win.width(), win.height()) == (887, 440)
    win._adapt_floor_to_screen(_Screen(1920, 1040))
    assert (win.minimumWidth(), win.minimumHeight()) == (FLOOR_W, FLOOR_H)


def test_no_page_demands_a_tall_window(win, qapp):
    """A page's own minimum must not decide the window's height: the stacked
    pages scroll (a 719 px floor was the live page's whole content)."""
    _resize(win, qapp, 900, 560)
    for name in ("home", "live", "finder"):
        page = _page(win, name)
        _settle(qapp)
        assert page.minimumSizeHint().height() <= 300, name
        assert isinstance(page.scroll, QScrollArea), name


def test_live_primary_view_keeps_the_roll_and_details_remain_reachable(win, qapp):
    """During singing the primary view is the finish action, the note and
    input level, AND the roll that draws the take; opening details adds the
    score, tuning history and settings on top of it.
    """
    _resize(win, qapp, 900, 560)
    page = _page(win, "live")
    _settle(qapp)

    assert page.details_panel.isHidden()
    for widget in (page.btn_stop_live, page.box_monitor):
        assert widget.isVisible(), type(widget).__name__
        top = widget.mapTo(page.scroll.viewport(), QPoint(0, 0))
        assert top.y() >= -1 and top.y() + widget.height() <= page.scroll.viewport().height() + 1, (
            type(widget).__name__,
            top.y(),
            widget.height(),
            page.scroll.viewport().height(),
        )
    # the roll is shown with the take, at its readable floor, above the
    # disclosure — never folded into it
    assert page.piano_live.isVisible()
    assert page.piano_live.height() >= page.piano_live.minimumHeight()
    assert page.box_monitor.y() < page.piano_live.y() < page.details_toggle.y()

    page.details_toggle.click()
    _settle(qapp)
    assert not page.details_panel.isHidden()
    assert page.scroll.verticalScrollBar().maximum() > 0
    page.scroll.verticalScrollBar().setValue(page.scroll.verticalScrollBar().maximum())
    _settle(qapp)
    viewport = page.scroll.viewport()
    for widget in (page.piano_live, page.staff_live):
        top = widget.mapTo(viewport, QPoint(0, 0))
        assert top.y() < viewport.height() and top.y() + widget.height() > 0, (
            type(widget).__name__,
            top.y(),
            widget.height(),
            viewport.height(),
        )
    keyboard_top = page.piano_live_keys.mapTo(viewport, QPoint(0, 0))
    assert keyboard_top.y() >= -1
    assert keyboard_top.y() + page.piano_live_keys.height() <= viewport.height() + 1


def test_every_stacked_page_keeps_its_bottom_reachable(win, qapp):
    """At the floor, the last widget of each page is inside its own content
    AND inside the viewport once the page is scrolled down."""
    _resize(win, qapp, 900, 560)
    probes = {
        "home": ("btn_lang", "calibration_card"),
        "live": ("btn_stop_live", "details_toggle"),
        "finder": ("btn_toggle", "keys"),
    }
    for name, (top_name, bottom_name) in probes.items():
        page = _page(win, name)
        _settle(qapp)
        content = page.scroll.widget()
        bottom = getattr(page, bottom_name)
        if bottom.isHidden():
            bottom = getattr(page, top_name)
        inside = bottom.mapTo(content, QPoint(0, 0)).y() + bottom.height()
        assert inside <= content.height() + 1, f"{name}: content clips its bottom"
        bar = page.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        _settle(qapp)
        viewport = page.scroll.viewport()
        top = bottom.mapTo(viewport, QPoint(0, 0))
        assert top.y() >= -1 and top.y() + bottom.height() <= viewport.height() + 1, (
            f"{name}: {bottom_name} is unreachable after scrolling"
        )


def test_the_live_page_keeps_a_readable_roll_and_still_grows(win, qapp):
    """The roll must never be squeezed to a sliver to make the page fit (it
    read 49 px tall at the floor before), and it must still take the extra
    height a TALL window offers (the scrolled content stretches above its
    own natural height)."""
    page = _page(win, "live")
    _resize(win, qapp, 900, 560)
    page.details_toggle.click()
    _settle(qapp)
    at_floor = page.piano_live.height()
    assert at_floor >= page.piano_live.minimumHeight(), (
        f"the roll is below its readable floor: {at_floor} logical px"
    )
    _resize(win, qapp, 1600, 1400)
    _settle(qapp)
    tall = page.piano_live.height()
    assert tall > at_floor, f"the roll no longer grows with the window: {at_floor} → {tall}"
    assert page.staff_live.height() > 0


def test_the_scroll_surfaces_paint_the_app_background(win, qapp):
    """The new scroll surfaces are plain QWidgets: the app palette must reach
    their viewport, not Qt's default white (a sheet that stopped applying
    after the wrapping would be visible here and nowhere else)."""
    from v2m.gui.widgets import BG_APP, STYLESHEET

    qapp.setStyleSheet(STYLESHEET)
    _resize(win, qapp, 900, 560)
    for name in ("home", "live", "finder"):
        page = _page(win, name)
        _settle(qapp)
        pixel = page.scroll.viewport().grab().toImage().pixelColor(3, 3)
        actual = f"{pixel.red():02x}{pixel.green():02x}{pixel.blue():02x}"
        assert actual == BG_APP.lstrip("#"), f"{name}: viewport paints {actual}"


def test_the_results_body_chooses_a_split_for_available_space(win, qapp):
    """Normal layouts preserve the responsive floors; compact layout gets a
    vertical splitter. WHICH surface fills the reading area is the view
    switcher's choice, so it never changes with the arrangement."""
    from PyQt6.QtCore import Qt

    from v2m.gui.results_page import (
        BODY_SIDE_BY_SIDE_MIN_W,
        COMPACT_BODY_MAX_H,
        COMPACT_ROLL_MIN_H,
        COMPACT_SETTINGS_MIN_H,
        LEFT_COLUMN_MIN_W,
        RIGHT_COLUMN_MIN_W,
        STACKED_LEFT_COLUMN_MIN_W,
    )

    assert (BODY_SIDE_BY_SIDE_MIN_W, LEFT_COLUMN_MIN_W, RIGHT_COLUMN_MIN_W) == (1180, 520, 620)
    assert STACKED_LEFT_COLUMN_MIN_W == 460
    page = _page(win, "results")
    _resize(win, qapp, 1400, 900)
    assert page._body_mode == "side-by-side"
    assert page.body.orientation() == Qt.Orientation.Horizontal
    assert page.left_host.minimumWidth() == LEFT_COLUMN_MIN_W
    assert page.right_scroll.minimumWidth() == RIGHT_COLUMN_MIN_W
    assert _shown_views(page) == ["notes"]  # one surface, the chosen one

    _resize(win, qapp, 900, 620)
    assert page._body_mode == "stacked"
    assert page.body.orientation() == Qt.Orientation.Vertical
    assert page.right_scroll.minimumWidth() == 0
    assert page.left_host.minimumWidth() == STACKED_LEFT_COLUMN_MIN_W
    assert _shown_views(page) == ["notes"]

    _resize(win, qapp, 900, 560)
    assert page.body.height() <= COMPACT_BODY_MAX_H
    assert page._body_mode == "compact"
    assert page.body.orientation() == Qt.Orientation.Vertical
    assert page.left_host.minimumHeight() == COMPACT_ROLL_MIN_H
    assert page.right_scroll.minimumHeight() == COMPACT_SETTINGS_MIN_H
    assert _shown_views(page) == ["notes"]


def test_compact_results_split_starts_at_sixty_five_thirty_five_and_can_move(win, qapp):
    """The splitter reserves a readable roll and settings pane, and resizing
    its handle changes and then preserves the user's preferred share."""
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    page = _page(win, "results")
    _resize(win, qapp, 909, 560)
    assert page._body_mode == "compact"
    sizes = page.body.sizes()
    assert abs(sizes[0] / sum(sizes) - 0.65) < 0.03, sizes
    assert sizes[0] >= page.left_host.minimumHeight()
    assert sizes[1] >= page.right_scroll.minimumHeight()

    handle = page.body.handle(1)
    start = handle.rect().center()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(handle, start - QPoint(0, 35), 100)
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=start - QPoint(0, 35))
    _settle(qapp)
    adjusted = page.body.sizes()
    assert adjusted[1] > sizes[1], (sizes, adjusted)
    preferred_share = adjusted[0] / sum(adjusted)

    _resize(win, qapp, 909, 580)
    resized = page.body.sizes()
    assert abs(resized[0] / sum(resized) - preferred_share) < 0.03, (
        preferred_share,
        resized,
    )
    _resize(win, qapp, 909, 700)
    assert page._body_mode == "stacked"
    assert _shown_views(page) == ["notes"]


def test_the_chosen_view_survives_every_arrangement(win, qapp):
    """The switcher, never the window height, decides what the body shows.

    The staff and waveform used to fold away on a short window: the
    LAYOUT overrode a reading the user had asked for. Now the arrangement
    only sets floors and the split, and the compact pane still shows the
    chosen surface whole.
    """
    page = _page(win, "results")
    _resize(win, qapp, 1400, 900)
    page.btn_view_staff.click()
    assert page.current_view() == "staff"
    assert _shown_views(page) == ["staff"]

    _resize(win, qapp, 900, 560)  # compact
    assert page._body_mode == "compact"
    assert _shown_views(page) == ["staff"]
    assert page.left_host.height() >= page.left_host.minimumHeight()

    page.btn_view_wave.click()
    assert _shown_views(page) == ["wave"]
    page.btn_view_notes.click()
    assert _shown_views(page) == ["notes"]
    # the roll's own display option belongs to the notes view
    assert not page.notation_row.isHidden()
    page.btn_view_staff.click()
    assert page.notation_row.isHidden()


def test_the_results_controls_never_leave_their_column(win, qapp):
    """Every labeled control of the results page sits inside the controls
    viewport at every window size — no horizontal scrollbar to reach one."""
    page = _page(win, "results")
    for width, height in ((1920, 1000), (1280, 780), (1000, 620), (900, 560)):
        _resize(win, qapp, width, height)
        _settle(qapp)
        scroll = page.right_scroll
        viewport_w = scroll.viewport().width()
        assert scroll.horizontalScrollBar().maximum() == 0, f"{width}x{height}"
        for name in (
            "btn_export",
            "btn_undo",
            "btn_replace_phrase",
            "btn_audition_original",
            "btn_full_reanalyze",
            "btn_open",
        ):
            button = getattr(page, name)
            if not button.isVisible():
                continue  # a hidden control has no geometry to judge
            left = button.mapTo(scroll.viewport(), QPoint(0, 0)).x()
            assert left >= 0 and left + button.width() <= viewport_w + 1, (
                f"{name} leaves the column at {width}x{height}"
            )


def test_results_details_table_remains_readable_at_compact_heights(win, qapp):
    """Opening diagnostics must leave a real viewport for their note rows.

    The details area scrolls its own content independently so the table
    retains a useful viewport even when its natural layout is taller than the
    available panel height. The table stays a useful size
    and can be brought fully into view at every window height.
    """
    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult

    page = win.page_results
    notes = [Note(i * 0.35, i * 0.35 + 0.25, 60 + i % 5, 0.9) for i in range(16)]
    result = AnalysisResult(
        source="sample.wav",
        duration_s=6.0,
        notes=notes,
        tempo=None,
        key=None,
        pitch=None,
    )
    win.result = result
    win._show_result(result)
    win.stack.setCurrentWidget(page)

    for width, height in ((900, 560), (1280, 780), (1920, 1000)):
        _resize(win, qapp, width, height)
        page.details_toggle.setChecked(True)
        _settle(qapp)

        table = page.results._notes_table
        assert table.viewport().height() >= 78, (
            f"the note table has no readable rows at {width}x{height}: "
            f"table={table.height()}, viewport={table.viewport().height()}"
        )
        assert table.maximumHeight() >= 280
        details_scroll = page.details_scroll
        details_bar = details_scroll.verticalScrollBar()
        assert details_bar.maximum() > 0

        details_bar.setValue(details_bar.maximum())
        _settle(qapp)
        table_top = table.mapTo(details_scroll.viewport(), QPoint(0, 0)).y()
        details_bar.setValue(details_bar.value() + table_top)
        _settle(qapp)
        table_top = table.mapTo(details_scroll.viewport(), QPoint(0, 0)).y()
        table_bottom = table_top + table.height()
        assert 0 <= table_top and table_bottom <= details_scroll.viewport().height(), (
            f"the note table cannot be scrolled fully into view at {width}x{height}: "
            f"table={table_top}..{table_bottom}, "
            f"viewport={details_scroll.viewport().height()}"
        )
        assert table.rowCount() == 16
        page.details_toggle.setChecked(False)
        _settle(qapp)
