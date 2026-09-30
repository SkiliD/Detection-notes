"""The results page: one analysis shown, heard and re-read.

The third Page module (ADR 0010's pattern, after live and finder): the
page OWNS its widgets — waveform, staff, piano roll, keyboard, the
export options, the play buttons — and the rendering behavior around
them (refilling every view from a result, the playhead, the position
line). The window keeps the FACTS that outlive the page: the result,
the audio cache, the playback session and clock, the exports. The page
never touches them; the window never lays out a widget.

Dependencies are injected as callables (the user actions that drive
window state: seek, playback, export, reinterpret); the notation
checkbox stays page-internal — it only ever affected this page's own
piano roll. The public methods (`set_result`, the playhead writes,
`sync_play_buttons`, `retranslate`) are the page's whole face.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..core.audio_io import SR
from ..core.grid import BeatGrid
from ..core.i18n import tr
from .staff import StaffView
from .widgets import PianoRoll, PianoView, ResultsPanel, WaveformView

if TYPE_CHECKING:
    from ..core.pipeline import AnalysisResult


class ResultsPage(QWidget):
    """The analysis page's widgets (built here) and its rendering behavior."""

    def __init__(
        self,
        on_seek: Callable[[float], None],
        toggle_play: Callable[[], None],
        toggle_voice: Callable[[], None],
        toggle_both: Callable[[], None],
        piano_volume_changed: Callable[[int], None],
        transpose_changed: Callable[[int], None],
        export_again: Callable[[], None],
        reinterpret: Callable[[], None],
        open_file: Callable[[], None],
        back_home: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_seek = on_seek
        self._toggle_play = toggle_play
        self._toggle_voice = toggle_voice
        self._toggle_both = toggle_both
        self._piano_volume_changed = piano_volume_changed
        self._transpose_changed = transpose_changed
        self._export_again = export_again
        self._reinterpret = reinterpret
        self._open_file = open_file
        self._back_home = back_home
        # the file name remembered for retranslate (a progress line writes
        # the label WITHOUT replacing it — see set_file_label)
        self._file_name: str | None = None
        self._build()

    # ------------------------------------------------------------------
    # widgets (moved from pages.build_analyze_page, unchanged layout)
    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_back = QPushButton("")
        self.btn_back.clicked.connect(self._back_home)
        top.addWidget(self.btn_back)
        self.label_file = QLabel("")
        self.label_file.setStyleSheet("font-weight: bold;")
        top.addWidget(self.label_file, stretch=1)
        root.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        body = QHBoxLayout()
        left = QVBoxLayout()
        self.waveform = WaveformView()
        self.piano = PianoRoll()
        self.piano.seek_requested.connect(self._on_seek)
        self.staff = StaffView()
        left.addWidget(self.waveform, stretch=1)
        left.addWidget(self.staff, stretch=2)
        left.addWidget(self.piano, stretch=2)

        playback = QHBoxLayout()
        self.btn_play = QPushButton("")
        self.btn_play.setMinimumWidth(64)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_voice = QPushButton("")
        self.btn_voice.setMinimumWidth(64)
        self.btn_voice.clicked.connect(self._toggle_voice)
        self.btn_both = QPushButton("")
        self.btn_both.setMinimumWidth(72)
        self.btn_both.clicked.connect(self._toggle_both)
        self.label_play_pos = QLabel("")
        playback.addWidget(self.btn_play)
        playback.addWidget(self.btn_voice)
        playback.addWidget(self.btn_both)
        playback.addWidget(self.label_play_pos, stretch=1)
        # the rendered piano's volume, for comparing it with the singer's voice
        self.label_volume = QLabel("")
        self.slider_volume = QSlider(self._qt_orientation())
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(70)
        self.slider_volume.setMaximumWidth(150)
        self.slider_volume.valueChanged.connect(self._piano_volume_changed)
        playback.addWidget(self.label_volume)
        playback.addWidget(self.slider_volume)
        left.addLayout(playback)
        self.piano_keys = PianoView()
        left.addWidget(self.piano_keys)
        body.addLayout(left, stretch=3)

        right = QVBoxLayout()
        self.results = ResultsPanel()
        right.addWidget(self.results, stretch=1)
        options = QGroupBox("")
        form = QGridLayout(options)
        self._grp_opts = options
        # the interpretation thresholds the "re-read" button uses: they change
        # HOW the take's evidence is read, never the evidence itself (pYIN does
        # not rerun — the cached observation is re-interpreted in milliseconds)
        self._lbl_min_note = QLabel("")
        form.addWidget(self._lbl_min_note, 5, 0)
        self.spin_min_note = QSpinBox()
        self.spin_min_note.setRange(40, 300)
        self.spin_min_note.setValue(60)
        self.spin_min_note.setSuffix(" ms")
        form.addWidget(self.spin_min_note, 5, 1)
        self.chk_quantize = QCheckBox("")
        form.addWidget(self.chk_quantize, 0, 0, 1, 2)
        self._lbl_transpose = QLabel("")
        form.addWidget(self._lbl_transpose, 1, 0)
        self.spin_transpose = QSpinBox()
        self.spin_transpose.setRange(-48, 48)
        self.spin_transpose.valueChanged.connect(self._transpose_changed)
        form.addWidget(self.spin_transpose, 1, 1)
        self.chk_notation = QCheckBox("")
        self.chk_notation.toggled.connect(self._on_notation_toggled)
        form.addWidget(self.chk_notation, 2, 0, 1, 2)
        self.chk_pdf = QCheckBox("")
        from ..core.pdf_export import find_notation_tool

        self.chk_pdf.setEnabled(find_notation_tool() is not None)
        form.addWidget(self.chk_pdf, 3, 0, 1, 2)
        self.lbl_pdf_note = QLabel("")
        self.lbl_pdf_note.setStyleSheet(
            "color: #7f8b99; font-size: 8pt; font-style: italic;"
        )
        self.lbl_pdf_note.setWordWrap(True)
        form.addWidget(self.lbl_pdf_note, 4, 0, 1, 2)
        right.addWidget(options)

        self.btn_export = QPushButton("")
        self.btn_export.clicked.connect(self._export_again)
        right.addWidget(self.btn_export)
        # re-read the take's evidence with the CURRENT thresholds (the spin
        # boxes above): interpretation only, the cached observation skips pYIN
        self.btn_reinterpret = QPushButton("")
        self.btn_reinterpret.clicked.connect(self._reinterpret)
        right.addWidget(self.btn_reinterpret)
        self.btn_open = QPushButton("")
        self.btn_open.clicked.connect(self._open_file)
        right.addWidget(self.btn_open)
        body.addLayout(right, stretch=1)
        root.addLayout(body, stretch=1)

    @staticmethod
    def _qt_orientation():
        """The volume slider's horizontal orientation (import kept local)."""
        from PyQt6.QtCore import Qt

        return Qt.Orientation.Horizontal

    def _on_notation_toggled(self) -> None:
        """The checkbox drives THIS page's roll only — no window fact."""
        self.piano.set_notation_mode(self.chk_notation.isChecked())

    # ------------------------------------------------------------------
    # strings
    # ------------------------------------------------------------------
    def retranslate(self) -> None:
        """The page's user-facing strings (the window's retranslate calls it)."""
        self.btn_back.setText(tr("ui back home"))
        if self._file_name:
            self.label_file.setText(self._file_name)
        else:
            self.label_file.setText(tr("ui no file"))
        self._grp_opts.setTitle(tr("ui export options"))
        self.chk_quantize.setText(tr("ui quantize"))
        self._lbl_min_note.setText(tr("ui min note"))
        self.btn_reinterpret.setText(tr("ui reinterpret"))
        self.btn_reinterpret.setToolTip(tr("ui reinterpret tooltip"))
        self._lbl_transpose.setText(tr("ui transpose"))
        self.chk_notation.setText(tr("ui notation mode"))
        self.chk_pdf.setText(tr("ui pdf option"))
        self.chk_pdf.setToolTip(tr("ui pdf tooltip"))
        self.lbl_pdf_note.setVisible(not self.chk_pdf.isEnabled())
        self.lbl_pdf_note.setText(tr("ui pdf badge"))
        self.btn_export.setText(tr("ui re-export"))
        self.btn_open.setText(tr("ui open another"))
        self.btn_play.setText("▶")
        self.btn_play.setToolTip(tr("ui play tooltip"))
        self.btn_voice.setText("🎙")
        self.btn_voice.setToolTip(tr("ui play voice"))
        self.btn_both.setText(tr("ui play both"))
        self.btn_both.setToolTip(tr("ui play both tooltip"))
        self.label_volume.setText(tr("ui volume"))
        self.slider_volume.setToolTip(tr("ui volume tooltip"))
        self.label_play_pos.setText(tr("ui play pos"))

    # ------------------------------------------------------------------
    # view fills
    # ------------------------------------------------------------------
    def set_source_name(self, name: str) -> None:
        """Remember the take's display name (survives a retranslate)."""
        self._file_name = name
        self.label_file.setText(name)

    def set_file_label(self, text: str) -> None:
        """Transient label write (an analysis progress line passes through)."""
        self.label_file.setText(text)

    def begin_analysis(self) -> None:
        """Reset the views for a new analysis: buttons up, progress at 0."""
        self.btn_play.setText("▶")
        self.label_play_pos.setText(tr("ui analyzing file"))
        self.progress.setValue(0)

    def set_progress(self, percent: int) -> None:
        self.progress.setValue(percent)

    def ack_export(self) -> None:
        """The export's feedback, where the eye already is."""
        self.label_play_pos.setText(tr("ui exported ok"))

    def set_result(self, res: "AnalysisResult", signal: np.ndarray | None = None) -> None:
        """Fill every view from one analysis result (file or live take).

        `signal` is the decoded audio the waveform shows — the window's
        cache decides which (a file view must never show a live take);
        None leaves the waveform untouched.
        """
        self.results.set_results(res)
        if signal is not None and len(signal):
            self.waveform.set_signal(signal, SR)
        self.piano.set_notation_mode(self.chk_notation.isChecked())
        self.piano.set_notes(res.notes)
        self.piano_keys.set_notes(res.notes)
        self.piano.set_pitch_curve(res.pitch)
        self.staff.set_score(
            res.notes,
            # an incomplete result (notes-only re-read, no tempo) shows on the
            # rendered default grid: set_score's `BeatGrid(None)` owns that
            # decision, never build a grid from a None tempo here
            grid=(res.grid if res.grid is not None
                  else BeatGrid(res.tempo.bpm) if res.tempo is not None else None),
            key=getattr(res.key, "best", None),
            positions=getattr(res, "note_positions", None),
            rests=getattr(res, "rests", None),
        )

    # ------------------------------------------------------------------
    # playback rendering (the window owns the clock; the page draws it)
    # ------------------------------------------------------------------
    def piano_gain(self) -> float:
        """Slider 0-100 -> render gain (70 = neutral, 0 = mute)."""
        return self.slider_volume.value() / 70.0

    def set_playhead(self, t: float, follow: bool = True) -> None:
        self.piano.set_playhead(t, follow=follow)

    def set_wave_position(self, t: float) -> None:
        self.waveform.set_position(t)

    def set_keys_playhead(self, t: float) -> None:
        self.piano_keys.set_playhead(t)

    def set_play_pos(self, t: float, duration: float) -> None:
        """The mm:ss / mm:ss position line."""
        self.label_play_pos.setText(
            f"{int(t // 60):02d}:{int(t % 60):02d}"
            f" / {int(duration // 60):02d}:{int(duration % 60):02d}"
        )

    def sync_play_buttons(self, mode: str | None) -> None:
        """The three buttons reflect the sounding mode (None = stopped)."""
        self.btn_play.setText("■" if mode == "piano" else "▶")
        self.btn_voice.setText("■" if mode == "voice" else "🎙")
        self.btn_both.setText("■" if mode == "both" else "🎹+🎙")

    def reset_play_buttons(self) -> None:
        """Stopped state: every button up, every key released."""
        self.sync_play_buttons(None)
        self.piano_keys.set_playhead(-1.0)  # all keys up
