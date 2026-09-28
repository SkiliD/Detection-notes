"""Widget assembly for the application's GUI pages.

Page builders create widgets and connect user actions; MainWindow owns state,
navigation, capture, analysis, and rendering callbacks.
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
from ..core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI
from .clicks import ClickPlayer
from .staff import StaffView
from .widgets import (
    ACCENT, AccuracyPlot, HINT_CARD_STYLE, LiveIndicator, PianoRoll, PianoView,
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


def build_live_page(window) -> QWidget:
    page = QWidget()
    root = QVBoxLayout(page)
    top = QHBoxLayout()
    window.btn_stop_live = QPushButton("")
    window.btn_stop_live.clicked.connect(window.on_stop_live)
    top.addWidget(window.btn_stop_live)
    window.label_live_status = QLabel("")
    top.addWidget(window.label_live_status, stretch=1)
    window.label_live_expected = QLabel("")
    window.label_live_expected.setStyleSheet("color: #9fb3c8; font-size: 9pt;")
    top.addWidget(window.label_live_expected)
    root.addLayout(top)

    window.label_live_hint = QLabel("")
    window.label_live_hint.setStyleSheet(HINT_CARD_STYLE)
    window.label_live_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    root.addWidget(window.label_live_hint)

    # optional metronome: OFF by default — a singer opens it when working on
    # regularity, and the status line shows the count during capture
    metro = QHBoxLayout()
    window.chk_metronome = QCheckBox("")
    window.chk_metronome.toggled.connect(window.on_metronome_toggled)
    metro.addWidget(window.chk_metronome)
    window.spin_bpm = QSpinBox()
    window.spin_bpm.setRange(40, 208)
    window.spin_bpm.setValue(90)
    window.spin_bpm.valueChanged.connect(window.on_metronome_bpm_changed)
    metro.addWidget(window.spin_bpm)
    window.label_bpm = QLabel("")
    metro.addWidget(window.label_bpm)
    window.label_beat = QLabel("")
    window.label_beat.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
    metro.addWidget(window.label_beat, stretch=1)
    metro.addStretch(1)
    root.addLayout(metro)

    # optional: keep the take's AUDIO beside its artifacts, only in live mode.
    # The box is read when the take STARTS (a mid-take toggle must not change
    # what this take writes); label and tooltip are set in `retranslate`.
    rec = QHBoxLayout()
    window.chk_record_audio = QCheckBox("")
    rec.addWidget(window.chk_record_audio)
    rec.addStretch(1)
    root.addLayout(rec)

    mid = QHBoxLayout()
    window.live_indicator = LiveIndicator()
    mid.addWidget(window.live_indicator, stretch=1)
    window.piano_live = PianoRoll()
    mid.addWidget(window.piano_live, stretch=3)
    root.addLayout(mid, stretch=2)

    # per-note mean deviation so far: the progress signal of the take
    window.label_live_accuracy = QLabel("")
    window.label_live_accuracy.setStyleSheet("color: #8a97a5; font-size: 9pt;")
    root.addWidget(window.label_live_accuracy)
    # ...and the same deviations over the take's timeline (singing order)
    window.accuracy_plot = AccuracyPlot()
    root.addWidget(window.accuracy_plot)

    window.staff_live = StaffView(live=True)
    root.addWidget(window.staff_live, stretch=2)
    window.piano_live_keys = PianoView()
    root.addWidget(window.piano_live_keys)
    window.clicks = ClickPlayer()
    return page


def build_finder_page(window) -> QWidget:
    page = QWidget()
    root = QVBoxLayout(page)
    top = QHBoxLayout()
    window.btn_finder_back = QPushButton("")
    window.btn_finder_back.setCursor(Qt.CursorShape.PointingHandCursor)
    window.btn_finder_back.clicked.connect(window.on_close_finder)
    top.addWidget(window.btn_finder_back)
    window.btn_finder_toggle = QPushButton("")
    window.btn_finder_toggle.setMinimumHeight(38)
    window.btn_finder_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
    window.btn_finder_toggle.clicked.connect(window.on_toggle_finder)
    top.addWidget(window.btn_finder_toggle)
    window.label_finder_state = QLabel("")
    window.label_finder_state.setStyleSheet("color: #8a97a5; font-size: 9pt;")
    top.addWidget(window.label_finder_state)
    top.addStretch(1)
    root.addLayout(top)

    window.label_finder_note = QLabel("")
    window.label_finder_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.label_finder_note.setMinimumHeight(66)
    root.addWidget(window.label_finder_note)

    window.label_finder_hint = QLabel("")
    window.label_finder_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.label_finder_hint.setWordWrap(True)
    window.label_finder_hint.setMaximumWidth(560)
    window.label_finder_hint.setStyleSheet(HINT_CARD_STYLE)
    hint_lay = QHBoxLayout()
    hint_lay.addStretch(1)
    hint_lay.addWidget(window.label_finder_hint)
    hint_lay.addStretch(1)
    root.addLayout(hint_lay)

    window.label_finder_history = QLabel("")
    window.label_finder_history.setStyleSheet("color: #9fb3c8; font-weight: bold;")
    root.addWidget(window.label_finder_history)
    window.finder_history = QListWidget()
    window.finder_history.setMaximumHeight(112)
    window.finder_history.setStyleSheet(
        "background: #20262e; border: 1px solid #2e3742; border-radius: 6px;"
    )
    root.addWidget(window.finder_history)
    root.addStretch(1)

    window.finder_keys = PianoView(min_midi=PIANO_MIN_MIDI, max_midi=PIANO_MAX_MIDI)
    window.finder_keys.setMinimumHeight(150)
    root.addWidget(window.finder_keys, stretch=2)
    return page
