"""The home page: its widgets and coherence presentation.

The page is a MASTHEAD, not a stack of centered blocks. One measure holds a
heavy rule, the headline, the deck, a hairline, the three-step standfirst and
a second hairline above the actions; the type is flush left inside that
measure so the page has one left edge instead of five centered axes. The
primary take-off spans the measure and the two secondary ones share a row
under it (a column when the translated labels do not fit).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QBoxLayout,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import coherence as coh
from ..core.i18n import format_number, tr
from . import widgets
from .calibration_card import CalibrationCard
from .widgets import action_button_style, page_scroll, quiet_button_style

# The reading measure: wide enough for the two secondary take-offs to share
# a row in both languages (their labels measure 599 px together) and for a
# headline to break on a phrase, narrow enough that the eye returns to the
# left edge without hunting — a column, not a full-width form.
ACTION_COLUMN_MAX = 640

# What `action_button_style` adds to a label's own width: 22 px of side padding
# plus the 1 px border, on each side.
ACTION_BUTTON_PADDING = 46

# What `quiet_button_style` adds to a label's own width: 14 px of side padding
# plus the 1 px border, on each side.
QUIET_BUTTON_PADDING = 30


def action_pair_fits(measure: int, button_widths: Sequence[int], spacing: int) -> bool:
    """Whether both secondary actions and their grid gap fit in `measure`."""
    return sum(button_widths) + spacing <= measure


def quiet_footer_fits(
    available: int,
    button_widths: Sequence[int],
    theme_width: int,
    spacing: int,
) -> bool:
    """Whether the quiet footer's four controls and three gaps fit in `available`."""
    return sum(button_widths) + theme_width + spacing * 3 <= available


def _rule(object_name: str, thickness: int) -> QFrame:
    """One horizontal rule, painted by the application stylesheet.

    The colour lives in the QSS (`QFrame#masthead` is the heavy rule over the
    headline, `QFrame#hairline` the thin separators), so a theme switch
    repaints every rule without restyling them one by one.
    """
    rule = QFrame()
    rule.setObjectName(object_name)
    rule.setFrameShape(QFrame.Shape.NoFrame)
    rule.setFixedHeight(thickness)
    return rule


def _chapeau_style() -> str:
    """The deck under the headline: one step down in size, still readable."""
    return f"color: {widgets.TEXT_DIM}; font-size: 11pt;"


def _sommaire_style() -> str:
    """The three-step standfirst: quiet, so it reads as a caption, not a menu."""
    return f"color: {widgets.TEXT_FAINT}; font-size: 9pt;"


class HomePage(QWidget):
    """Build the home screen and render the coherence evidence it receives."""

    def __init__(
        self,
        open_file: Callable[[], None],
        start_live: Callable[[], None],
        open_finder: Callable[[], None],
        toggle_language: Callable[[], None],
        snapshot_factory: Callable[[], coh.EvidenceSnapshot],
        show_status: Callable[[str, int], None],
        on_calibrate_toggled: Callable[[bool], None] | None = None,
        input_devices: Callable[[], list[tuple[str, str]]] | None = None,
        selected_input_key: str | None = None,
        input_device_changed: Callable[[str | None], None] | None = None,
        driver_note: Callable[[str], str | None] | None = None,
        theme: str = "dark",
        theme_changed: Callable[[str], None] | None = None,
    ) -> None:
        self._on_calibrate_toggled = on_calibrate_toggled or (lambda _on: None)
        super().__init__()
        self._snapshot_factory = snapshot_factory
        self._show_status = show_status
        self._input_devices = input_devices or (lambda: [])
        self._input_device_changed = input_device_changed or (lambda _key: None)
        self._driver_note = driver_note or (lambda _name: None)
        self._selected_input_key = selected_input_key
        self._theme_changed = theme_changed or (lambda _theme: None)
        self._theme = theme
        self._input_device_name = ""
        self._input_device_fell_back = False
        self._snapshot: coh.EvidenceSnapshot | None = None
        self._build(open_file, start_live, open_finder, toggle_language)

    def _build(
        self,
        open_file: Callable[[], None],
        start_live: Callable[[], None],
        open_finder: Callable[[], None],
        toggle_language: Callable[[], None],
    ) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(9, 9, 9, 9)
        layout.addStretch(2)

        # The reading column: ONE measure, every line flush left inside it.
        # Centering each block on its own gave the page five competing axes;
        # a heavy rule, a headline and one shared left edge read as a page.
        self._measure = QWidget()
        column = QVBoxLayout(self._measure)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        self._masthead_rule = _rule("masthead", 2)
        column.addWidget(self._masthead_rule)
        column.addSpacing(16)

        self.title = QLabel("")
        self.title.setStyleSheet(widgets.display_title_style(28.0))
        self.title.setWordWrap(True)
        self.title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        column.addWidget(self.title)
        column.addSpacing(6)

        self.subtitle = QLabel("")
        self.subtitle.setStyleSheet(_chapeau_style())
        self.subtitle.setWordWrap(True)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        column.addWidget(self.subtitle)
        column.addSpacing(14)

        self._deck_rule = _rule("hairline", 1)
        column.addWidget(self._deck_rule)
        column.addSpacing(10)

        self.steps = QLabel("")
        self.steps.setStyleSheet(_sommaire_style())
        self.steps.setWordWrap(True)
        self.steps.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        column.addWidget(self.steps)
        column.addSpacing(14)

        self._actions_rule = _rule("hairline", 1)
        column.addWidget(self._actions_rule)
        column.addSpacing(18)

        # The take-off actions keep one predictable order: the primary action
        # spans the measure, the two others share ONE row under it.
        self.btn_live = QPushButton("")
        self.btn_live.setMinimumHeight(54)
        self.btn_live.setStyleSheet(action_button_style(primary=True))
        self.btn_live.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_live.clicked.connect(start_live)

        self.btn_file = QPushButton("")
        self.btn_file.setMinimumHeight(46)
        self.btn_file.setStyleSheet(action_button_style(primary=False))
        self.btn_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_file.clicked.connect(open_file)

        self.btn_finder = QPushButton("")
        self.btn_finder.setMinimumHeight(46)
        self.btn_finder.setStyleSheet(action_button_style(primary=False))
        self.btn_finder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_finder.clicked.connect(open_finder)

        # Three full-width rows read as a menu of equals, so the secondary
        # take-offs pair up in ONE grid: the primary spans it, the two others
        # share the row beneath it and drop to their own rows when the
        # translated labels no longer fit. A grid keeps all three buttons
        # SIBLINGS, so the tab order stays the creation order.
        self._actions_grid = QGridLayout()
        self._actions_grid.setContentsMargins(0, 0, 0, 0)
        self._actions_grid.setHorizontalSpacing(12)
        self._actions_grid.setVerticalSpacing(12)
        self._actions_grid.setColumnStretch(0, 1)
        self._actions_grid.setColumnStretch(1, 1)
        self._actions_grid.addWidget(self.btn_live, 0, 0, 1, 2)
        self._actions_grid.addWidget(self.btn_file, 1, 0)
        self._actions_grid.addWidget(self.btn_finder, 1, 1)
        self._actions_host = QWidget()
        self._actions_host.setLayout(self._actions_grid)
        self._actions_paired = True
        column.addWidget(self._actions_host)

        # The measure joins the page LAST, once every line it carries (the
        # actions, the settings strip, the device line and the tip) is on it:
        # one left edge, no block centred on its own axis.
        column.addSpacing(22)

        # Quiet settings and follow-ups form a responsive row at wide sizes;
        # they stack when translated labels cannot fit side by side.
        self._footer = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._footer.setSizeConstraint(QBoxLayout.SizeConstraint.SetNoConstraint)
        self._footer.setContentsMargins(0, 0, 0, 0)
        self._footer.setSpacing(12)
        self.btn_lang = QPushButton("")
        self.btn_lang.setMinimumWidth(0)
        self.btn_lang.setMinimumHeight(40)
        self.btn_lang.setStyleSheet(quiet_button_style())
        self.btn_lang.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_lang.clicked.connect(toggle_language)
        self._footer.addWidget(self.btn_lang)
        # The selector names the active theme itself; its accessible name and
        # tooltip carry "appearance" without spending a label's width.
        self.theme_selector = QComboBox()
        self.theme_selector.setMinimumWidth(0)
        self.theme_selector.currentIndexChanged.connect(self._on_theme_selected)
        self._footer.addWidget(self.theme_selector)
        self.btn_coherence = QPushButton("")
        self.btn_coherence.setMinimumWidth(0)
        self.btn_coherence.setMinimumHeight(40)
        self.btn_coherence.setStyleSheet(quiet_button_style())
        self.btn_coherence.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coherence.clicked.connect(self.show_coherence)
        self._footer.addWidget(self.btn_coherence)
        self.btn_calibrate = QPushButton("")
        self.btn_calibrate.setMinimumWidth(0)
        self.btn_calibrate.setMinimumHeight(40)
        self.btn_calibrate.setStyleSheet(quiet_button_style())
        self.btn_calibrate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_calibrate.clicked.connect(self.show_calibration)
        self._footer.addWidget(self.btn_calibrate)
        self._footer_row = QWidget()
        self._footer_row.setLayout(self._footer)
        self._footer_row.setMinimumWidth(0)
        self._footer_row.setSizePolicy(
            self._footer_row.sizePolicy().Policy.Fixed,
            self._footer_row.sizePolicy().Policy.Fixed,
        )
        column.addWidget(self._footer_row)

        self.input_device = QComboBox()
        self.input_device.setMinimumWidth(0)
        self.input_device.setMaximumWidth(ACTION_COLUMN_MAX)
        self.input_device.setSizePolicy(
            self.input_device.sizePolicy().Policy.Ignored,
            self.input_device.sizePolicy().Policy.Fixed,
        )
        self.input_device.setAccessibleName(tr("ui home microphone"))
        self.input_device.currentIndexChanged.connect(self._on_input_device_selected)
        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        device_row.addStretch(1)
        device_row.addWidget(self.input_device)
        device_row.addStretch(1)
        device_row.setSizeConstraint(QBoxLayout.SizeConstraint.SetNoConstraint)
        self._device_row = device_row
        column.addLayout(device_row)

        self.input_device_status = QLabel("")
        self.input_device_status.setStyleSheet(f"color: {widgets.TEXT_DIM}; font-size: 9pt;")
        self.input_device_status.setWordWrap(True)
        self.input_device_status.setMinimumWidth(0)
        self.input_device_status.setMaximumWidth(ACTION_COLUMN_MAX)
        self.input_device_status.setSizePolicy(
            self.input_device_status.sizePolicy().Policy.Ignored,
            self.input_device_status.sizePolicy().Policy.Preferred,
        )
        column.addWidget(self.input_device_status)

        # The OS's own record, when the chosen endpoint publishes a driver
        # effects chain: a take the driver already filtered is a different
        # signal, and nothing INSIDE the take can say so.
        self.input_device_note = QLabel("")
        self.input_device_note.setStyleSheet(f"color: {widgets.TEXT_DIM}; font-size: 9pt;")
        self.input_device_note.setWordWrap(True)
        self.input_device_note.setMinimumWidth(0)
        self.input_device_note.setMaximumWidth(ACTION_COLUMN_MAX)
        self.input_device_note.setSizePolicy(
            self.input_device_note.sizePolicy().Policy.Ignored,
            self.input_device_note.sizePolicy().Policy.Preferred,
        )
        self.input_device_note.setVisible(False)
        column.addWidget(self.input_device_note)

        self.hint = QLabel("")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {widgets.TEXT_DIM};")
        self.hint.setMinimumWidth(0)
        self.hint.setSizePolicy(
            self.hint.sizePolicy().Policy.Ignored,
            self.hint.sizePolicy().Policy.Preferred,
        )
        column.addWidget(self.hint)

        layout.addWidget(self._measure, 0, Qt.AlignmentFlag.AlignHCenter)

        self.calibration_card = CalibrationCard(
            self, on_close=lambda: self._on_calibrate_toggled(False)
        )
        self.calibration_card.setVisible(False)
        layout.addWidget(self.calibration_card)
        layout.addSpacing(10)

        self.coherence_text = QPlainTextEdit("")
        self.coherence_text.setReadOnly(True)
        self.coherence_text.setVisible(False)
        self.coherence_text.setStyleSheet(
            f"background: {widgets.BG_SUNKEN}; color: {widgets.TEXT}; "
            f"border: 1px solid {widgets.BORDER}; border-radius: 8px; padding: 8px; font-size: 9pt;"
        )
        self.coherence_text.setMinimumHeight(180)
        layout.addWidget(self.coherence_text)

        self.health_plot = pg.PlotWidget()
        self.health_plot.setVisible(False)
        self.health_plot.setFixedHeight(140)
        self.health_plot.setMaximumWidth(640)
        self.health_plot.hideAxis("bottom")
        item = self.health_plot.getPlotItem()
        item.hideButtons()
        item.setMouseEnabled(x=False, y=False)
        item.setMenuEnabled(False)
        item.setTitle(tr("ui health chart title"), color=widgets.TEXT, size="9pt")
        for axis in ("left", "bottom"):
            item.getAxis(axis).setPen(pg.mkPen(widgets.BORDER))
            item.getAxis(axis).setTextPen(pg.mkPen(widgets.TEXT_DIM))
        item.vb.setBackgroundColor(widgets.BG_SUNKEN)
        self.health_plot.scene().sigMouseClicked.connect(self._on_health_clicked)
        layout.addWidget(self.health_plot, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch(2)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = page_scroll(content)
        outer.addWidget(self.scroll)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._arrange_footer()

    def _arrange_footer(self) -> None:
        """Keep translated secondary actions inside the home-page width."""
        if not self.btn_lang.text():
            return
        labels = (
            (self.btn_lang, tr("ui lang button")),
            (self.btn_coherence, tr("ui coherence sessions btn")),
            (self.btn_calibrate, tr("ui calibrate mic button")),
        )
        button_widths = [
            QFontMetrics(button.font()).horizontalAdvance(label) + QUIET_BUTTON_PADDING
            for button, label in labels
        ]
        # Measure the page, not the scroll viewport: during a resize Qt may
        # still report the previous (narrower) viewport, which stacked the row
        # on a wide window and never re-ran once the layout settled.
        available = max(
            0,
            self.width() - 18 - self.scroll.verticalScrollBar().sizeHint().width(),
        )
        measure = min(ACTION_COLUMN_MAX, available)
        stacked = not quiet_footer_fits(
            available,
            button_widths,
            self.theme_selector.sizeHint().width(),
            self._footer.spacing(),
        )
        # The strip spans the measure whether it folds or not: the settings
        # start on the headline's own left edge instead of drifting to the
        # centre of the page.
        width = measure
        self._footer_row.setMinimumWidth(0)
        self._footer_row.setMaximumWidth(width)
        self._footer_row.setSizePolicy(
            self._footer_row.sizePolicy().Policy.Fixed,
            self._footer_row.sizePolicy().Policy.Fixed,
        )
        self._footer.setDirection(
            QBoxLayout.Direction.TopToBottom if stacked else QBoxLayout.Direction.LeftToRight
        )
        self.theme_selector.setMaximumWidth(width)
        self._footer_row.setMinimumHeight(0)
        self._footer_row.setMaximumHeight(16777215)
        self._footer.invalidate()
        self._footer_row.layout().invalidate()
        self._footer_row.layout().setGeometry(self._footer_row.rect())
        self._device_row.invalidate()
        self._device_row.activate()
        for (button, label), button_width in zip(labels, button_widths, strict=True):
            button.setMinimumWidth(0)
            button.setMaximumWidth(width)
            if stacked:
                button.setText(self._wrapped_label(label, button, width - 24))
                button.setMinimumWidth(width)
                button.setMaximumWidth(width)
                button.setMinimumHeight(40)
                button.setMaximumHeight(16777215)
            else:
                button.setText(label)
                button.setMinimumWidth(button_width)
                button.setMaximumWidth(button_width)
                button.setMinimumHeight(40)
                button.setMaximumHeight(40)
        self._footer.invalidate()
        self._footer.activate()
        # The layout centres this row (its width is fixed above); placing it by
        # hand pinned the y read before the column re-flowed, which left the
        # row a few pixels over the action above it.
        self._footer_row.setSizePolicy(
            self._footer_row.sizePolicy().Policy.Fixed,
            self._footer_row.sizePolicy().Policy.Fixed,
        )
        self._footer_row.updateGeometry()
        self._footer_row.layout().setGeometry(self._footer_row.rect())
        # The column's width is the ONLY horizontal fact: the layouts inside
        # it place the headline, the rules and the actions against one edge.
        self._measure.setMinimumWidth(measure)
        self._measure.setMaximumWidth(measure)
        self._layout_action_pair(measure)
        self._measure.layout().invalidate()
        self._measure.layout().activate()
        self.input_device.setMinimumWidth(measure)
        self.input_device.setMaximumWidth(measure)
        self.input_device.setSizePolicy(
            self.input_device.sizePolicy().Policy.Fixed,
            self.input_device.sizePolicy().Policy.Fixed,
        )
        self.input_device_status.setMinimumWidth(0)
        self.input_device_status.setMaximumWidth(measure)
        self.input_device_status.setSizePolicy(
            self.input_device_status.sizePolicy().Policy.Ignored,
            self.input_device_status.sizePolicy().Policy.Preferred,
        )
        self.input_device.updateGeometry()
        self.input_device_status.updateGeometry()
        content = self.scroll.widget()
        content.layout().invalidate()
        content.updateGeometry()
        self.scroll.updateGeometry()

    def _layout_action_pair(self, measure: int) -> None:
        """Keep the two secondary take-offs in one row, or give each its own.

        Measured from the LIVE labels: a French label runs ~40 % longer than
        its English twin, so a fixed breakpoint would overflow the measure on
        one language and waste a row on the other.
        """
        button_widths = [
            QFontMetrics(button.font()).horizontalAdvance(button.text()) + ACTION_BUTTON_PADDING
            for button in (self.btn_file, self.btn_finder)
        ]
        paired = action_pair_fits(
            measure,
            button_widths,
            self._actions_grid.horizontalSpacing(),
        )
        self._set_action_pair(paired)

    def _set_action_pair(self, paired: bool) -> None:
        """Apply the row or column layout chosen by `action_pair_fits`."""
        if paired == self._actions_paired:
            return
        self._actions_paired = paired
        self._actions_grid.removeWidget(self.btn_file)
        self._actions_grid.removeWidget(self.btn_finder)
        if paired:
            self._actions_grid.addWidget(self.btn_file, 1, 0)
            self._actions_grid.addWidget(self.btn_finder, 1, 1)
        else:
            # each own row, both spanning the measure
            self._actions_grid.addWidget(self.btn_file, 1, 0, 1, 2)
            self._actions_grid.addWidget(self.btn_finder, 2, 0, 1, 2)

    @staticmethod
    def _wrapped_label(text: str, button: QPushButton, max_width: int) -> str:
        """Wrap secondary labels at word boundaries for narrow high-DPI views."""
        metrics = QFontMetrics(button.font())
        words = text.split()
        lines: list[str] = []
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if line and metrics.horizontalAdvance(candidate) > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        return "\n".join(lines)

    def retranslate(self) -> None:
        self.title.setText(tr("ui home title"))
        self.input_device.setAccessibleName(tr("ui home microphone"))
        self.subtitle.setText(tr("ui app subtitle"))
        self.steps.setText(tr("ui steps"))
        self.btn_file.setText(tr("ui open file"))
        self.btn_live.setText(tr("ui sing live"))
        self.btn_finder.setText(tr("ui finder button"))
        self.btn_lang.setText(tr("ui lang button"))
        self.theme_selector.setAccessibleName(tr("ui theme label"))
        self.theme_selector.setToolTip(tr("ui theme label"))
        self.theme_selector.blockSignals(True)
        self.theme_selector.clear()
        self.theme_selector.addItem(tr("ui theme dark"), "dark")
        self.theme_selector.addItem(tr("ui theme light"), "light")
        self.theme_selector.setCurrentIndex(max(0, self.theme_selector.findData(self._theme)))
        self.theme_selector.blockSignals(False)
        self.btn_coherence.setText(tr("ui coherence sessions btn"))
        self.btn_calibrate.setText(tr("ui calibrate mic button"))
        self.hint.setText(tr("ui drop hint"))
        self._refresh_input_devices()
        self._render_input_device_status()
        self.btn_coherence.setToolTip(tr("ui coherence sessions tooltip"))
        self.calibration_card.retranslate()
        self._arrange_footer()
        self.health_plot.getPlotItem().setTitle(
            tr("ui health chart title"), color=widgets.TEXT, size="9pt"
        )

    def _on_theme_selected(self, index: int) -> None:
        theme = self.theme_selector.itemData(index)
        if theme not in ("dark", "light") or theme == self._theme:
            return
        self._theme = theme
        self._theme_changed(theme)

    def refresh_theme(self) -> None:
        """Refresh the home page's inline palette and its plot axes."""
        self.title.setStyleSheet(widgets.display_title_style(28.0))
        self.subtitle.setStyleSheet(_chapeau_style())
        self.steps.setStyleSheet(_sommaire_style())
        self.btn_live.setStyleSheet(action_button_style(primary=True))
        self.btn_file.setStyleSheet(action_button_style(primary=False))
        self.btn_finder.setStyleSheet(action_button_style(primary=False))
        self.input_device_status.setStyleSheet(f"color: {widgets.TEXT_DIM}; font-size: 9pt;")
        self.input_device_note.setStyleSheet(f"color: {widgets.TEXT_DIM}; font-size: 9pt;")
        self.coherence_text.setStyleSheet(
            f"background: {widgets.BG_SUNKEN}; color: {widgets.TEXT}; "
            f"border: 1px solid {widgets.BORDER}; border-radius: 8px; padding: 8px; font-size: 9pt;"
        )
        self.hint.setStyleSheet(f"color: {widgets.TEXT_DIM};")
        item = self.health_plot.getPlotItem()
        for axis in ("left", "bottom"):
            item.getAxis(axis).setPen(pg.mkPen(widgets.BORDER))
            item.getAxis(axis).setTextPen(pg.mkPen(widgets.TEXT_DIM))
        item.vb.setBackgroundColor(widgets.BG_SUNKEN)
        item.setTitle(tr("ui health chart title"), color=widgets.TEXT, size="9pt")
        self._draw_health(self._snapshot.health_weeks if self._snapshot else [])
        self.btn_lang.setStyleSheet(quiet_button_style())
        self.btn_coherence.setStyleSheet(quiet_button_style())
        self.btn_calibrate.setStyleSheet(quiet_button_style())
        # The selector paints itself, and a widget-level sheet wins over the
        # application one for every property it names: without its own `:focus`
        # rule the shared `QComboBox:focus` ring never reached this Tab stop.
        # The ring pays for itself out of the padding (1 + 4 = 2 + 3), so the
        # box keeps its size hint while it holds focus.
        self.theme_selector.setStyleSheet(
            f"QComboBox {{ background: {widgets.BG_SUNKEN}; color: {widgets.TEXT}; "
            f"border: 1px solid {widgets.BORDER}; border-radius: 6px; padding: 4px; }}"
            f"QComboBox:focus {{ border: 2px dashed {widgets.ACCENT}; padding: 3px; }}"
        )

    def set_theme(self, theme: str) -> None:
        """Keep the selector aligned with the application theme fact."""
        if theme not in widgets.THEMES:
            raise ValueError(f"unknown theme: {theme}")
        self._theme = theme
        index = self.theme_selector.findData(theme)
        if index >= 0 and index != self.theme_selector.currentIndex():
            self.theme_selector.blockSignals(True)
            self.theme_selector.setCurrentIndex(index)
            self.theme_selector.blockSignals(False)

    def _refresh_input_devices(self) -> None:
        """Refresh the available-input list while preserving its stable key."""
        selected = self._selected_input_key
        devices = self._input_devices()
        self.input_device.blockSignals(True)
        self.input_device.clear()
        self.input_device.addItem(tr("ui input device default"), None)
        found_selected = selected is None
        for key, label in devices:
            self.input_device.addItem(label, key)
            found_selected = found_selected or key == selected
        if selected is not None and not found_selected:
            parts = selected.split("\x1f")
            missing_name = parts[1] if len(parts) > 1 else selected
            self.input_device.addItem(
                tr("ui input device unavailable").format(device=missing_name), selected
            )
        index = self.input_device.findData(selected)
        self.input_device.setCurrentIndex(index if index >= 0 else 0)
        self.input_device.blockSignals(False)
        self.input_device.setToolTip(tr("ui input device tooltip"))

    def _on_input_device_selected(self, index: int) -> None:
        key = self.input_device.itemData(index)
        self._selected_input_key = key
        self._input_device_changed(key)

    def set_input_device_status(self, name: str, fell_back: bool = False) -> None:
        """Show the endpoint selected for the next capture."""
        self._input_device_name = name
        self._input_device_fell_back = fell_back
        self._refresh_input_devices()
        self._render_input_device_status()

    def _render_input_device_status(self) -> None:
        key = (
            "ui input device fallback next"
            if self._input_device_fell_back
            else "ui input device next"
        )
        self.input_device_status.setText(tr(key).format(device=self._input_device_name))
        note = self._driver_note(self._input_device_name) or ""
        self.input_device_note.setText(note)
        self.input_device_note.setVisible(bool(note))

    def show_calibration(self) -> None:
        """Toggle the calibration card (the window wires the timer around it)."""
        self.calibration_card.setVisible(not self.calibration_card.isVisible())
        self.coherence_text.setVisible(False)
        self.health_plot.setVisible(False)
        self._on_calibrate_toggled(self.calibration_card.isVisible())

    def show_coherence(self) -> None:
        """Load one evidence snapshot, then render the table and chart."""
        try:
            snapshot = self._snapshot_factory()
        except Exception as error:
            self.coherence_text.setPlainText(tr("err ui unreadable").format(e=error))
            self.coherence_text.setVisible(True)
            return
        self._snapshot = snapshot
        if not snapshot.takes:
            self.coherence_text.setPlainText(tr("ui coherence none"))
            self.coherence_text.setVisible(True)
            self._draw_health(snapshot.health_weeks)
            return
        text = coh.session_table_text(
            snapshot.takes,
            rows=snapshot.sessions,
            weeks=snapshot.health_weeks,
        )
        if snapshot.practice_windows:
            text += "\n" + tr("ui practice windows title") + "\n"
            for window in snapshot.practice_windows:
                rate = (
                    format_number(window["suspect_rate"], 1)
                    if window["suspect_rate"] is not None
                    else "-"
                )
                contours = (
                    f"{format_number(round(window['contour_rate'] * 100), 0)}%"
                    if window["contour_rate"] is not None
                    else "-"
                )
                text += (
                    tr("ui practice window row").format(
                        window=tr(f"time {window['window']}"),
                        takes=window["takes"],
                        rate=rate,
                        contours=contours,
                    )
                    + "\n"
                )
        self.coherence_text.setPlainText(text)
        self.coherence_text.setVisible(True)
        self._draw_health(snapshot.health_weeks)

    def _draw_health(self, weeks: list[dict]) -> None:
        self.health_plot.clear()
        if not weeks:
            self.health_plot.setVisible(False)
            return
        xs = list(range(len(weeks)))
        ys = [week["contour_rate"] * 100.0 for week in weeks]
        self.health_plot.plot(xs, ys, pen=pg.mkPen(widgets.ACCENT, width=2), antialias=True)
        self.health_plot.addItem(
            pg.ScatterPlotItem(
                xs,
                ys,
                size=9,
                brush=pg.mkBrush(widgets.ACCENT),
                pen=pg.mkPen(widgets.BG_SUNKEN),
            )
        )
        self.health_plot.setXRange(-0.5, len(weeks) - 0.5, padding=0)
        self.health_plot.setYRange(0, 100, padding=0)
        self.health_plot.setVisible(True)

    def _on_health_clicked(self, event) -> None:
        """Show the selected week's takes from the same snapshot as the chart."""
        if getattr(event, "button", lambda: None)() != pg.QtCore.Qt.MouseButton.LeftButton:
            return
        snapshot = self._snapshot
        if snapshot is None or not snapshot.health_weeks:
            self._show_status(tr("ui health click none"), 5_000)
            return
        x = float("nan")
        try:
            x = self.health_plot.getPlotItem().vb.mapSceneToView(event.scenePos()).x()
            index = int(round(x))
        except Exception:
            index = -1
        if not x == x or abs(x - round(x)) > 0.5 or not 0 <= index < len(snapshot.health_weeks):
            self._show_status(tr("ui health click none"), 5_000)
            return
        week = snapshot.health_weeks[index]
        if not week.get("stems"):
            self._show_status(tr("ui health click none"), 5_000)
            return
        stems = week.get("stems") or []
        self._show_status(tr("ui health click").format(week=week["week"], n=len(stems)), 8_000)
        by_stem = {take.stem: take for take in snapshot.takes}
        week_takes = [by_stem[stem] for stem in stems if stem in by_stem]
        missing = len(stems) - len(week_takes)
        lines = [tr("ui health click").format(week=week["week"], n=len(stems)), ""]
        for take in week_takes:
            checked = f"{take.checked}" if take.checked is not None else "-"
            rate = (
                f"{format_number(round(take.contour_rate * 100), 0)}%"
                if take.contour_rate is not None
                else "-"
            )
            if take.verify_ok is False:
                lines.append(
                    tr("coh take divergent").format(stem=take.stem, checked=checked, rate=rate)
                )
            elif not take.suspects:
                lines.append(
                    tr("coh take clean").format(stem=take.stem, checked=checked) + f"  ({rate})"
                )
            else:
                lines.append(
                    tr("coh take suspects").format(
                        stem=take.stem, checked=checked, rate=rate, n=len(take.suspects)
                    )
                )
            for suspect in take.suspects:
                lines.append(
                    tr("coh suspect row").format(
                        midi=suspect["midi"],
                        span=format_number(suspect["inner_range_st"], 2),
                        dur=format_number(suspect["secondary_plateau_s"], 2),
                        at=(
                            tr("coh plateau at").format(
                                t=format_number(suspect["secondary_plateau_start_s"], 2)
                            )
                            if suspect.get("secondary_plateau_start_s") is not None
                            else ""
                        ),
                        offset=(
                            format_number(
                                suspect["secondary_plateau_offset_st"], 2, sign=True
                            )
                            + "st"
                            if suspect.get("secondary_plateau_offset_st") is not None
                            else "?"
                        ),
                        target=suspect.get("secondary_plateau_midi", "?"),
                    )
                )
        if missing:
            lines.append(tr("ui health click none") if missing == len(stems) else f"  ({missing})")
        self.coherence_text.setPlainText("\n".join(lines))
