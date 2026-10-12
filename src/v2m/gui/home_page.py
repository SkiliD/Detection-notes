"""The home page: its widgets and coherence presentation."""

from __future__ import annotations

from typing import Callable

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import coherence as coh
from ..core.i18n import tr
from .calibration_card import CalibrationCard
from .widgets import (
    ACCENT,
    BG_SUNKEN,
    BORDER,
    HINT_CARD_STYLE,
    TEXT,
    TEXT_DIM,
    action_button_style,
)

# The action column's measure: wide enough for the longest label in both
# languages, narrow enough to read as a menu instead of a form.
ACTION_COLUMN_MAX = 560
# The hero sentence and the steps card wrap on a comfortable line length.
HERO_MAX = 640
STEPS_MAX = 720


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
    ) -> None:
        self._on_calibrate_toggled = on_calibrate_toggled or (lambda _on: None)
        super().__init__()
        self._snapshot_factory = snapshot_factory
        self._show_status = show_status
        self._snapshot: coh.EvidenceSnapshot | None = None
        self._build(open_file, start_live, open_finder, toggle_language)

    def _build(
        self,
        open_file: Callable[[], None],
        start_live: Callable[[], None],
        open_finder: Callable[[], None],
        toggle_language: Callable[[], None],
    ) -> None:
        layout = QVBoxLayout(self)
        # Equal top/bottom stretches: the menu sits dead center instead of
        # drifting up and leaving a hollow third under the hint.
        layout.addStretch(2)

        # Hero: title over a one-line pitch, on a bounded measure so the
        # sentence wraps in two balanced lines instead of spanning the window.
        self.title = QLabel("<h1>🎤 Voix → MIDI</h1>")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)
        self.subtitle = QLabel("")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setWordWrap(True)
        self.subtitle.setMaximumWidth(HERO_MAX)
        layout.addWidget(self.subtitle, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(22)

        self.steps = QLabel("")
        self.steps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.steps.setWordWrap(True)
        self.steps.setMaximumWidth(STEPS_MAX)
        self.steps.setStyleSheet(HINT_CARD_STYLE)
        steps = QHBoxLayout()
        steps.addStretch(1)
        steps.addWidget(self.steps)
        steps.addStretch(1)
        layout.addLayout(steps)
        layout.addSpacing(24)

        # The three actions in ONE centered column of equal width: full-bleed
        # bars read as a form to fill, a bounded column reads as a menu of
        # choices, and the primary accent stays the first thing the eye finds.
        actions = QVBoxLayout()
        actions.setSpacing(12)
        self.btn_file = QPushButton("")
        self.btn_file.setMinimumHeight(54)
        self.btn_file.setStyleSheet(action_button_style(primary=True))
        self.btn_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_file.clicked.connect(open_file)
        actions.addWidget(self.btn_file)

        self.btn_live = QPushButton("")
        self.btn_live.setMinimumHeight(54)
        self.btn_live.setStyleSheet(action_button_style(primary=False))
        self.btn_live.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_live.clicked.connect(start_live)
        actions.addWidget(self.btn_live)

        self.btn_finder = QPushButton("")
        self.btn_finder.setMinimumHeight(54)
        self.btn_finder.setStyleSheet(action_button_style(primary=False))
        self.btn_finder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_finder.clicked.connect(open_finder)
        actions.addWidget(self.btn_finder)

        actions_host = QWidget()
        actions_host.setLayout(actions)
        actions_host.setMaximumWidth(ACTION_COLUMN_MAX)
        actions_row = QHBoxLayout()
        actions_row.addStretch(1)
        actions_row.addWidget(actions_host)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)
        layout.addSpacing(18)

        # The quiet controls share one footer row: they are settings and
        # follow-ups, not take-off points, so they sit side by side and
        # small instead of stacking three more full-width bars.
        footer = QHBoxLayout()
        footer.setSpacing(12)
        footer.addStretch(1)
        self.btn_lang = QPushButton("")
        self.btn_lang.setMinimumHeight(40)
        self.btn_lang.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_lang.clicked.connect(toggle_language)
        footer.addWidget(self.btn_lang)
        self.btn_coherence = QPushButton("")
        self.btn_coherence.setMinimumHeight(40)
        self.btn_coherence.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_coherence.clicked.connect(self.show_coherence)
        footer.addWidget(self.btn_coherence)
        self.btn_calibrate = QPushButton("")
        self.btn_calibrate.setMinimumHeight(40)
        self.btn_calibrate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_calibrate.clicked.connect(self.show_calibration)
        footer.addWidget(self.btn_calibrate)
        footer.addStretch(1)
        layout.addLayout(footer)

        # The calibration card: one guided flow (the approved design)
        # hidden under the footer until asked.
        self.calibration_card = CalibrationCard(self)
        self.calibration_card.setVisible(False)
        layout.addWidget(self.calibration_card)
        layout.addSpacing(10)

        self.coherence_text = QPlainTextEdit("")
        self.coherence_text.setReadOnly(True)
        self.coherence_text.setVisible(False)
        self.coherence_text.setStyleSheet(
            f"background: {BG_SUNKEN}; color: {TEXT}; border: 1px solid {BORDER};"
            "border-radius: 8px; padding: 8px; font-size: 9pt;"
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
        item.setTitle(tr("ui health chart title"), color=TEXT, size="9pt")
        for axis in ("left", "bottom"):
            item.getAxis(axis).setPen(pg.mkPen(BORDER))
            item.getAxis(axis).setTextPen(pg.mkPen(TEXT_DIM))
        item.vb.setBackgroundColor(BG_SUNKEN)
        self.health_plot.scene().sigMouseClicked.connect(self._on_health_clicked)
        layout.addWidget(self.health_plot, 0, Qt.AlignmentFlag.AlignHCenter)

        self.hint = QLabel("")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self.hint)
        layout.addStretch(2)

    def retranslate(self) -> None:
        self.title.setText("<h1>🎤 Voix → MIDI</h1>")
        self.subtitle.setText(tr("ui app subtitle"))
        self.steps.setText(tr("ui steps"))
        self.btn_file.setText(tr("ui open file"))
        self.btn_live.setText(tr("ui sing live"))
        self.btn_finder.setText(tr("ui finder button"))
        self.btn_lang.setText(tr("ui lang button"))
        self.btn_coherence.setText(tr("ui coherence sessions btn"))
        self.btn_calibrate.setText(tr("ui calibrate mic button"))
        self.hint.setText(tr("ui drop hint"))
        self.calibration_card.retranslate()
        self.health_plot.getPlotItem().setTitle(
            tr("ui health chart title"), color=TEXT, size="9pt"
        )

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
                    f"{window['suspect_rate']:.1f}"
                    if window["suspect_rate"] is not None
                    else "-"
                )
                contours = (
                    f"{window['contour_rate'] * 100:.0f}%"
                    if window["contour_rate"] is not None
                    else "-"
                )
                text += tr("ui practice window row").format(
                    window=tr(f"time {window['window']}"),
                    takes=window["takes"],
                    rate=rate,
                    contours=contours,
                ) + "\n"
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
        self.health_plot.plot(xs, ys, pen=pg.mkPen(ACCENT, width=2), antialias=True)
        self.health_plot.addItem(
            pg.ScatterPlotItem(
                xs,
                ys,
                size=9,
                brush=pg.mkBrush(ACCENT),
                pen=pg.mkPen(BG_SUNKEN),
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
        self._show_status(
            tr("ui health click").format(week=week["week"], n=len(stems)), 8_000
        )
        by_stem = {take.stem: take for take in snapshot.takes}
        week_takes = [by_stem[stem] for stem in stems if stem in by_stem]
        missing = len(stems) - len(week_takes)
        lines = [tr("ui health click").format(week=week["week"], n=len(stems)), ""]
        for take in week_takes:
            checked = f"{take.checked}" if take.checked is not None else "-"
            rate = f"{take.contour_rate * 100:.0f}%" if take.contour_rate is not None else "-"
            if take.verify_ok is False:
                lines.append(tr("coh take divergent").format(
                    stem=take.stem, checked=checked, rate=rate
                ))
            elif not take.suspects:
                lines.append(tr("coh take clean").format(
                    stem=take.stem, checked=checked
                ) + f"  ({rate})")
            else:
                lines.append(tr("coh take suspects").format(
                    stem=take.stem, checked=checked, rate=rate, n=len(take.suspects)
                ))
            for suspect in take.suspects:
                lines.append(tr("coh suspect row").format(
                    midi=suspect["midi"],
                    span=suspect["inner_range_st"],
                    dur=suspect["secondary_plateau_s"],
                    at=(
                        tr("coh plateau at").format(t=suspect["secondary_plateau_start_s"])
                        if suspect.get("secondary_plateau_start_s") is not None
                        else ""
                    ),
                    offset=(
                        f"{suspect['secondary_plateau_offset_st']:+.2f}st"
                        if suspect.get("secondary_plateau_offset_st") is not None else "?"
                    ),
                    target=suspect.get("secondary_plateau_midi", "?"),
                ))
        if missing:
            lines.append(tr("ui health click none") if missing == len(stems) else f"  ({missing})")
        self.coherence_text.setPlainText("\n".join(lines))
        self.coherence_text.setVisible(True)
