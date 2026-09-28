"""Widget assembly for the application's GUI pages.

Page builders create widgets and connect user actions; MainWindow owns state,
navigation, capture, analysis, and rendering callbacks. The note finder is
NOT here: its page is a Page module (`gui/finder_page.py`, FinderPage) that
owns its widgets and behavior — the window only composes it.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

import pyqtgraph as pg

from ..core.pdf_export import find_notation_tool
from ..core.i18n import tr
from .staff import StaffView
from .widgets import (
    HINT_CARD_STYLE, PianoRoll, PianoView,
    ResultsPanel, WaveformView, action_button_style,
)


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


def build_analyze_page(window) -> QWidget:
    page = QWidget()
    root = QVBoxLayout(page)
    top = QHBoxLayout()
    window.btn_back = QPushButton("")
    window.btn_back.clicked.connect(window.on_back_home)
    top.addWidget(window.btn_back)
    window.label_file = QLabel("")
    window.label_file.setStyleSheet("font-weight: bold;")
    top.addWidget(window.label_file, stretch=1)
    root.addLayout(top)

    window.progress = QProgressBar()
    window.progress.setRange(0, 100)
    root.addWidget(window.progress)

    body = QHBoxLayout()
    left = QVBoxLayout()
    window.waveform = WaveformView()
    window.piano = PianoRoll()
    window.piano.seek_requested.connect(window.on_seek)
    window.staff = StaffView()
    left.addWidget(window.waveform, stretch=1)
    left.addWidget(window.staff, stretch=2)
    left.addWidget(window.piano, stretch=2)

    playback = QHBoxLayout()
    window.btn_play = QPushButton("")
    window.btn_play.setMinimumWidth(64)
    window.btn_play.clicked.connect(window.on_toggle_play)
    window.btn_voice = QPushButton("")
    window.btn_voice.setMinimumWidth(64)
    window.btn_voice.clicked.connect(window.on_toggle_voice)
    window.btn_both = QPushButton("")
    window.btn_both.setMinimumWidth(72)
    window.btn_both.clicked.connect(window.on_toggle_both)
    window.label_play_pos = QLabel("")
    playback.addWidget(window.btn_play)
    playback.addWidget(window.btn_voice)
    playback.addWidget(window.btn_both)
    playback.addWidget(window.label_play_pos, stretch=1)
    # the rendered piano's volume, for comparing it with the singer's voice
    window.label_volume = QLabel("")
    window.slider_volume = QSlider(Qt.Orientation.Horizontal)
    window.slider_volume.setRange(0, 100)
    window.slider_volume.setValue(70)
    window.slider_volume.setMaximumWidth(150)
    window.slider_volume.valueChanged.connect(window.on_piano_volume_changed)
    playback.addWidget(window.label_volume)
    playback.addWidget(window.slider_volume)
    left.addLayout(playback)
    window.piano_keys = PianoView()
    left.addWidget(window.piano_keys)
    body.addLayout(left, stretch=3)

    right = QVBoxLayout()
    window.results = ResultsPanel()
    right.addWidget(window.results, stretch=1)
    options = QGroupBox("")
    form = QGridLayout(options)
    window._grp_opts = options
    # the interpretation thresholds the "re-read" button uses: they change
    # HOW the take's evidence is read, never the evidence itself (pYIN does
    # not rerun — the cached observation is re-interpreted in milliseconds)
    window._lbl_min_note = QLabel("")
    form.addWidget(window._lbl_min_note, 5, 0)
    window.spin_min_note = QSpinBox()
    window.spin_min_note.setRange(40, 300)
    window.spin_min_note.setValue(60)
    window.spin_min_note.setSuffix(" ms")
    form.addWidget(window.spin_min_note, 5, 1)
    window.chk_quantize = QCheckBox("")
    form.addWidget(window.chk_quantize, 0, 0, 1, 2)
    window._lbl_transpose = QLabel("")
    form.addWidget(window._lbl_transpose, 1, 0)
    window.spin_transpose = QSpinBox()
    window.spin_transpose.setRange(-48, 48)
    window.spin_transpose.valueChanged.connect(window.on_transpose_changed)
    form.addWidget(window.spin_transpose, 1, 1)
    window.chk_notation = QCheckBox("")
    window.chk_notation.toggled.connect(window._on_notation_toggled)
    form.addWidget(window.chk_notation, 2, 0, 1, 2)
    window.chk_pdf = QCheckBox("")
    window.chk_pdf.setEnabled(find_notation_tool() is not None)
    form.addWidget(window.chk_pdf, 3, 0, 1, 2)
    window.lbl_pdf_note = QLabel("")
    window.lbl_pdf_note.setStyleSheet("color: #7f8b99; font-size: 8pt; font-style: italic;")
    window.lbl_pdf_note.setWordWrap(True)
    form.addWidget(window.lbl_pdf_note, 4, 0, 1, 2)
    right.addWidget(options)

    window.btn_export = QPushButton("")
    window.btn_export.clicked.connect(window.on_export_again)
    right.addWidget(window.btn_export)
    # re-read the take's evidence with the CURRENT thresholds (the spin
    # boxes above): interpretation only, the cached observation skips pYIN
    window.btn_reinterpret = QPushButton("")
    window.btn_reinterpret.clicked.connect(window.on_reinterpret)
    right.addWidget(window.btn_reinterpret)
    window.btn_open = QPushButton("")
    window.btn_open.clicked.connect(window.on_open_file)
    right.addWidget(window.btn_open)
    body.addLayout(right, stretch=1)
    root.addLayout(body, stretch=1)
    return page
