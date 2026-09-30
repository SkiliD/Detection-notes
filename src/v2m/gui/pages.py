"""Widget assembly for the application's HOME page.

The home page is the last builder here: the three feature pages are Page
modules (`gui/results_page.py`, `gui/live_page.py`, `gui/finder_page.py`)
that own their widgets and behavior — the window only composes them.
MainWindow owns state, navigation, capture, analysis, and rendering
callbacks.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

import pyqtgraph as pg

from ..core.i18n import tr
from .widgets import HINT_CARD_STYLE, action_button_style


def build_home_page(window) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addStretch(2)
    window._title = QLabel("<h1>🎤 Voix → MIDI</h1>")
    window._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(window._title)
    window._subtitle = QLabel("")
    window._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(window._subtitle)
    layout.addSpacing(22)

    window._steps = QLabel("")
    window._steps.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window._steps.setWordWrap(True)
    window._steps.setMaximumWidth(720)
    window._steps.setStyleSheet(HINT_CARD_STYLE)
    steps = QHBoxLayout()
    steps.addStretch(1)
    steps.addWidget(window._steps)
    steps.addStretch(1)
    layout.addLayout(steps)
    layout.addSpacing(18)

    window.btn_file = QPushButton("")
    window.btn_file.setMinimumHeight(54)
    window.btn_file.setStyleSheet(action_button_style(primary=True))
    window.btn_file.setCursor(Qt.CursorShape.PointingHandCursor)
    window.btn_file.clicked.connect(window.on_open_file)
    layout.addWidget(window.btn_file)

    window.btn_live = QPushButton("")
    window.btn_live.setMinimumHeight(54)
    window.btn_live.setStyleSheet(action_button_style(primary=False))
    window.btn_live.setCursor(Qt.CursorShape.PointingHandCursor)
    window.btn_live.clicked.connect(window.on_start_live)
    layout.addWidget(window.btn_live)

    window.btn_finder = QPushButton("")
    window.btn_finder.setMinimumHeight(54)
    window.btn_finder.setStyleSheet(action_button_style(primary=False))
    window.btn_finder.setCursor(Qt.CursorShape.PointingHandCursor)
    window.btn_finder.clicked.connect(window.on_open_finder)
    layout.addWidget(window.btn_finder)

    window.btn_lang = QPushButton("")
    window.btn_lang.setMinimumHeight(40)
    window.btn_lang.setCursor(Qt.CursorShape.PointingHandCursor)
    window.btn_lang.clicked.connect(window.on_toggle_language)
    language = QHBoxLayout()
    language.addStretch(1)
    language.addWidget(window.btn_lang)
    language.addStretch(1)
    layout.addLayout(language)

    # the coherence follow-up: one click aggregates the user's takes and
    # shows the per-session table (the CLI script's numbers, no terminal)
    window.btn_coherence = QPushButton("")
    window.btn_coherence.setMinimumHeight(40)
    window.btn_coherence.setCursor(Qt.CursorShape.PointingHandCursor)
    window.btn_coherence.clicked.connect(window.on_show_coherence)
    window.coherence_text = QPlainTextEdit("")
    window.coherence_text.setReadOnly(True)
    window.coherence_text.setVisible(False)
    window.coherence_text.setStyleSheet(
        "background: #0e1218; color: #cfd6dd; border: 1px solid #2c3644;"
        "border-radius: 8px; padding: 8px; font-size: 9pt;"
    )
    window.coherence_text.setMinimumHeight(180)
    layout.addWidget(window.btn_coherence)
    layout.addWidget(window.coherence_text)

    # the multi-week REAL-TIME HEALTH chart (pyqtgraph, replaces the ASCII
    # sparkline on this page): one point per week, the contour rate — lower
    # is better. Clicking a week shows THAT week's replayed takes (the
    # journal's click-through), never the folder's guess. Built hidden: a
    # page with no replayed take shows the invite instead of an empty plot.
    window.health_plot = pg.PlotWidget()
    window.health_plot.setVisible(False)
    window.health_plot.setFixedHeight(140)
    window.health_plot.setMaximumWidth(640)
    window.health_plot.hideAxis("bottom")
    window.health_plot.getPlotItem().hideButtons()
    health = window.health_plot.plotItem
    health.setMouseEnabled(x=False, y=False)
    health.setMenuEnabled(False)
    health.setTitle(tr("ui health chart title"), color="#cfd6dd", size="9pt")
    for axis in ("left", "bottom"):
        health.getAxis(axis).setPen(pg.mkPen("#2c3644"))
        health.getAxis(axis).setTextPen(pg.mkPen("#7f8b99"))
    vb = health.vb
    vb.setBackgroundColor("#0e1218")
    window.health_plot.scene().sigMouseClicked.connect(
        lambda ev: window._on_health_clicked(ev)
    )
    layout.addWidget(window.health_plot, 0, Qt.AlignmentFlag.AlignHCenter)

    window._hint = QLabel("")
    window._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window._hint.setStyleSheet("color: #7f8b99;")
    layout.addWidget(window._hint)
    layout.addStretch(3)
    return page
