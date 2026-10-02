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
from PyQt6.QtCore import QItemSelection, QItemSelectionModel
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
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
        correct_note: Callable[[int, int], None],
        replace_phrase: Callable[[], None],
        finish_replacement: Callable[[], None],
        toggle_play: Callable[[], None],
        toggle_voice: Callable[[], None],
        toggle_both: Callable[[], None],
        piano_volume_changed: Callable[[int], None],
        transpose_changed: Callable[[int], None],
        export_again: Callable[[], None],
        reinterpret: Callable[[], None],
        open_file: Callable[[], None],
        back_home: Callable[[], None],
        undo_edit: Callable[[], None],
        redo_edit: Callable[[], None],
        jump_edit_state: Callable[[int], None],
        accept_replacement: Callable[[], None],
        cancel_replacement: Callable[[], None],
        audition_replacement: Callable[[], None],
        audition_original: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_seek = on_seek
        self._correct_note = correct_note
        self._replace_phrase = replace_phrase
        self._finish_replacement = finish_replacement
        self._toggle_play = toggle_play
        self._toggle_voice = toggle_voice
        self._toggle_both = toggle_both
        self._piano_volume_changed = piano_volume_changed
        self._transpose_changed = transpose_changed
        self._export_again = export_again
        self._reinterpret = reinterpret
        self._open_file = open_file
        self._back_home = back_home
        self._undo_edit = undo_edit
        self._redo_edit = redo_edit
        self._jump_edit_state = jump_edit_state
        self._accept_replacement = accept_replacement
        self._cancel_replacement = cancel_replacement
        self._audition_replacement = audition_replacement
        self._audition_original = audition_original
        # the file name remembered for retranslate (a progress line writes
        # the label WITHOUT replacing it — see set_file_label)
        self._file_name: str | None = None
        # the notes the table's rows index into (refilled by set_result —
        # the toolbar's selection readout needs their time spans)
        self._notes: list = []
        self._has_live_result = False
        self._replacement_analyzing = False
        self._replacement_pending_ui = False
        # the undo/redo enabled state is a HISTORY fact (the window's edit
        # history owns it): stored here so a selection change can re-apply
        # it without the window having to re-push it
        self._can_undo = False
        self._can_redo = False
        # the state dropdown's rows (the window's edit history owns them):
        # kept so _update_edit_controls can re-disable after any refill and
        # retranslate can relabel in place without a re-push
        self._edit_states: list[tuple] = []
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
        self.piano = PianoRoll(selectable=True)
        self.piano.seek_requested.connect(self._on_seek)
        self.piano.note_selection.connect(self._on_roll_selection)
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
        self.piano_keys.key_clicked.connect(self._on_key_clicked)
        left.addWidget(self.piano_keys)
        body.addLayout(left, stretch=3)

        right = QVBoxLayout()
        self.results = ResultsPanel()
        self.results._notes_table.setSelectionBehavior(
            self.results._notes_table.SelectionBehavior.SelectRows
        )
        self.results._notes_table.setSelectionMode(
            self.results._notes_table.SelectionMode.ExtendedSelection
        )
        self.results._notes_table.itemSelectionChanged.connect(self._on_table_selection)
        right.addWidget(self.results, stretch=1)
        self.label_edit_hint = QLabel("")
        self.label_edit_hint.setWordWrap(True)
        right.addWidget(self.label_edit_hint)
        edit_buttons = QHBoxLayout()
        self.btn_undo = QPushButton("")
        self.btn_undo.clicked.connect(self._undo_edit)
        self.btn_redo = QPushButton("")
        self.btn_redo.clicked.connect(self._redo_edit)
        self.btn_replace_phrase = QPushButton("")
        self.btn_replace_phrase.clicked.connect(self._replace_phrase)
        self.btn_audition_original = QPushButton("")
        self.btn_audition_original.clicked.connect(self._audition_original)
        self.btn_finish_replacement = QPushButton("")
        self.btn_finish_replacement.clicked.connect(self._finish_replacement)
        self.btn_finish_replacement.setVisible(False)
        edit_buttons.addWidget(self.btn_undo)
        edit_buttons.addWidget(self.btn_redo)
        edit_buttons.addWidget(self.btn_replace_phrase)
        edit_buttons.addWidget(self.btn_audition_original)
        edit_buttons.addWidget(self.btn_finish_replacement)
        # the toolbar's live readout: how many notes over which seconds —
        # the concrete "what am I about to delete/replace" next to the buttons
        edit_buttons.addStretch(1)
        self.label_selection = QLabel("")
        self.label_selection.setStyleSheet(
            "color: #7f8b99; font-size: 9pt;"
        )
        self.label_selection.setVisible(False)
        edit_buttons.addWidget(self.label_selection)
        right.addLayout(edit_buttons)
        # the take's edit history as one dropdown: every state (what it was
        # before, in the edit's words), the current one last. Choosing a row
        # jumps back to it (a stepped undo/redo — the history keeps ONE
        # owner of the branch bookkeeping); picking the current row does
        # nothing.
        self.combo_edit_states = QComboBox()
        self.combo_edit_states.currentIndexChanged.connect(self._on_state_chosen)
        right.addWidget(self.combo_edit_states)
        # the replacement's DECISION row: shown only while the spliced take
        # awaits the singer's verdict (audition, keep, cancel)
        pending_buttons = QHBoxLayout()
        self.btn_audition_replacement = QPushButton("")
        self.btn_audition_replacement.clicked.connect(self._audition_replacement)
        self.btn_accept_replacement = QPushButton("")
        self.btn_accept_replacement.clicked.connect(self._accept_replacement)
        self.btn_cancel_replacement = QPushButton("")
        self.btn_cancel_replacement.clicked.connect(self._cancel_replacement)
        for button in (
            self.btn_audition_replacement,
            self.btn_accept_replacement,
            self.btn_cancel_replacement,
        ):
            button.setVisible(False)
            pending_buttons.addWidget(button)
        right.addLayout(pending_buttons)
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
        self.btn_export.setText(tr("ui export"))
        self.btn_undo.setText(tr("ui undo"))
        self.btn_undo.setToolTip(tr("ui undo tooltip"))
        self.btn_redo.setText(tr("ui redo"))
        self.btn_redo.setToolTip(tr("ui redo tooltip"))
        self.btn_replace_phrase.setText(tr("ui replace phrase"))
        self.btn_audition_original.setText(tr("ui audition original"))
        self.btn_audition_original.setToolTip(tr("ui audition original tooltip"))
        self.btn_finish_replacement.setText(tr("ui finish replacement"))
        self.btn_audition_replacement.setText(tr("ui audition replacement"))
        self.btn_audition_replacement.setToolTip(tr("ui audition replacement tooltip"))
        self.btn_accept_replacement.setText(tr("ui accept replacement"))
        self.btn_cancel_replacement.setText(tr("ui cancel replacement"))
        self.label_edit_hint.setText(tr("ui edit hint"))
        self._update_selection_label()  # dynamic text: re-render in the new language
        for row_index, (_state, action) in enumerate(self._edit_states):
            self.combo_edit_states.setItemText(row_index, tr(action))
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

    def _selected_note_indices(self) -> list[int]:
        rows = sorted({index.row() for index in self.results._notes_table.selectedIndexes()})
        return rows

    def _update_selection_label(self) -> None:
        """The toolbar's readout: selected note count and their time span.

        Hidden with an empty selection; the rows index into `_notes` (both
        refilled together by `set_result`, so a stale row index is simply
        ignored rather than read against the wrong note).
        """
        indices = self._selected_note_indices()
        chosen = [self._notes[i] for i in indices if 0 <= i < len(self._notes)]
        if not chosen:
            self.label_selection.setVisible(False)
            return
        self.label_selection.setText(
            tr("ui selection label").format(
                n=len(chosen),
                t0=min(n.start for n in chosen),
                t1=max(n.end for n in chosen),
            )
        )
        self.label_selection.setVisible(True)

    def _on_table_selection(self) -> None:
        """The table owns the selection; the roll renders the same rows."""
        self.piano.set_selection(self._selected_note_indices())
        self._update_edit_controls()
        self._update_selection_label()

    def _on_roll_selection(self, indices: list[int]) -> None:
        """A roll gesture writes the table's selection (they stay one fact).

        The gesture's indices are contiguous (a click, or a drag between
        two bars), so one selection range covers them; the table's own
        `itemSelectionChanged` then runs `_on_table_selection` — the
        highlight and the edit controls refresh through the same path a
        table click takes, with no second bookkeeping.
        """
        table = self.results._notes_table
        model = table.model()
        if not indices:
            table.selectionModel().clearSelection()
            return
        table.selectionModel().select(
            QItemSelection(model.index(indices[0], 0), model.index(indices[-1], 0)),
            QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QItemSelectionModel.SelectionFlag.Rows,
        )

    def _update_edit_controls(self) -> None:
        if self._replacement_analyzing:
            self.btn_replace_phrase.setVisible(False)
            self.btn_audition_original.setEnabled(False)
            self.btn_finish_replacement.setVisible(False)
            self._set_pending_row_visible(False)
            self.btn_export.setEnabled(False)
            self.btn_reinterpret.setEnabled(False)
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)
            self.combo_edit_states.setEnabled(False)
            self.label_edit_hint.setText(tr("ui replacement analyzing"))
            return
        if self._replacement_pending_ui:
            # the spliced take awaits its verdict: export and the edit
            # history wait with it (the decision IS the next edit fact)
            self.btn_replace_phrase.setVisible(False)
            self.btn_audition_original.setEnabled(False)
            self.btn_finish_replacement.setVisible(False)
            self._set_pending_row_visible(True)
            self.btn_export.setEnabled(False)
            self.btn_reinterpret.setEnabled(False)
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)
            self.combo_edit_states.setEnabled(False)
            self.label_edit_hint.setText(tr("ui replacement pending"))
            return
        self._set_pending_row_visible(False)
        indices = self._selected_note_indices()
        contiguous = bool(indices) and indices == list(range(indices[0], indices[-1] + 1))
        self.btn_replace_phrase.setEnabled(contiguous)
        self.btn_replace_phrase.setVisible(
            not self.btn_finish_replacement.isVisible() and self._current_result_is_live()
        )
        self.btn_audition_original.setEnabled(contiguous)
        self.btn_audition_original.setVisible(
            not self.btn_finish_replacement.isVisible() and self._current_result_is_live()
        )
        self.btn_export.setEnabled(not self.btn_finish_replacement.isVisible())
        self.btn_reinterpret.setEnabled(not self.btn_finish_replacement.isVisible())
        self.btn_undo.setEnabled(self._can_undo)
        self.btn_redo.setEnabled(self._can_redo)
        self.combo_edit_states.setEnabled(True)
        self.label_edit_hint.setText(
            tr("ui edit selection").format(first=indices[0] + 1, last=indices[-1] + 1)
            if contiguous else tr("ui edit hint")
        )

    def _current_result_is_live(self) -> bool:
        return self._has_live_result

    def _on_key_clicked(self, midi: int) -> None:
        indices = self._selected_note_indices()
        if len(indices) == 1:
            self._correct_note(indices[0], midi)

    def set_replacement_active(self, active: bool) -> None:
        self._replacement_analyzing = False
        self.btn_replace_phrase.setVisible(not active)
        self.btn_finish_replacement.setVisible(active)
        self._update_edit_controls()

    def set_replacement_analyzing(self, analyzing: bool) -> None:
        self._replacement_analyzing = analyzing
        self._update_edit_controls()

    def set_replacement_pending(self, pending: bool) -> None:
        """The spliced take awaits the singer's verdict (keep / cancel)."""
        self._replacement_pending_ui = pending
        self._update_edit_controls()

    def set_edit_history_state(self, can_undo: bool, can_redo: bool) -> None:
        """Reflect the edit history's branches (the window owns the history)."""
        self._can_undo = can_undo
        self._can_redo = can_redo
        if not self._replacement_analyzing and not self._replacement_pending_ui:
            self.btn_undo.setEnabled(can_undo)
            self.btn_redo.setEnabled(can_redo)

    def set_edit_states(self, states: list[tuple]) -> None:
        """Refill the state dropdown from the window's history timeline.

        `states` rows are (EditState | None, action) — the None row is the
        current state, always last (the window's `_sync_edit_history` builds
        them). The window-block guard mirrors set_edit_history_state: during
        a replacement's analyzing/pending phases the dropdown is disabled by
        _update_edit_controls, so the refill only stores the rows.
        """
        self._edit_states = list(states)
        combo = self.combo_edit_states
        combo.blockSignals(True)
        try:
            combo.clear()
            for _state, action in self._edit_states:
                combo.addItem(tr(action))
            # the dropdown sits on the CURRENT row (the None reference,
            # wherever the timeline puts it), never on a jumpable state
            current_row = next(
                (i for i, (state, _a) in enumerate(self._edit_states) if state is None),
                max(0, combo.count() - 1),
            )
            combo.setCurrentIndex(current_row)
        finally:
            combo.blockSignals(False)
        self._update_edit_controls()

    def _on_state_chosen(self, row_index: int) -> None:
        """Hand the chosen row to the window (a jump = stepped undo/redo).

        The current state's row (the None reference, always last) jumps to
        nothing; a row whose action is disabled comes through only when the
        dropdown itself was enabled — the window's guards are the authority.
        """
        if not 0 <= row_index < len(self._edit_states):
            return
        state, _action = self._edit_states[row_index]
        if state is None:
            return
        self._jump_edit_state(row_index)

    def _set_pending_row_visible(self, visible: bool) -> None:
        for button in (
            self.btn_audition_replacement,
            self.btn_accept_replacement,
            self.btn_cancel_replacement,
        ):
            button.setVisible(visible)

    def set_result(self, res: "AnalysisResult", signal: np.ndarray | None = None) -> None:
        """Fill every view from one analysis result (file or live take).

        `signal` is the decoded audio the waveform shows — the window's
        cache decides which (a file view must never show a live take);
        None leaves the waveform untouched.
        """
        self._has_live_result = res.source == "live"
        was_replacing = self.btn_finish_replacement.isVisible()
        self.results.set_results(res)
        self._notes = list(res.notes)
        self.results._notes_table.clearSelection()
        self.piano.set_selection([])  # a new result starts with no highlight
        self._update_selection_label()
        self.set_replacement_active(was_replacing)
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
