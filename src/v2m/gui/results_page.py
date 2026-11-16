"""The results page: one analysis shown, heard and re-read.

The third Page module (ADR 0010's pattern, after live and finder): the
page OWNS its widgets — waveform, staff, piano roll, keyboard, the
export options, the play buttons — and the rendering behavior around
them (refilling every view from a result, the playhead, the position
line). The window keeps the FACTS that outlive the page: the result,
the audio cache, the playback session and clock, the exports. The page
never touches them; the window never lays out a widget.

The controls are organised BY TASK, in the order a take is reviewed:
listen (one NAMED source per action and one position line), adjust
(the actions on the selected notes, undo/redo, the edit history),
export (the one primary action, its PDF option, then the advanced
settings split into Export and Transcription — the two stages a
setting can belong to). The reading surface is a CHOICE as well:
notes, score or waveform, one at a time, so no window height has to
stack three views and squeeze the roll.

Dependencies are injected as callables (the user actions that drive
window state: seek, playback, export, reinterpret); the notation
checkbox stays page-internal — it only ever affected this page's own
piano roll. The public methods (`set_result`, `set_view`, the playhead
writes, `sync_play_buttons`, `retranslate`) are the page's whole face.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from PyQt6.QtCore import QEvent, QItemSelectionModel, Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.audio_io import SR
from ..core.continuity import coherence_report
from ..core.grid import BeatGrid
from ..core.i18n import note_display_pair, tr, tr_key_label
from ..core.notes import Note
from . import widgets
from .staff import StaffView
from .widgets import (
    PianoRoll,
    PianoView,
    ResultsPanel,
    WaveformView,
    action_button_style,
    link_button_style,
)

if TYPE_CHECKING:
    from ..core.pipeline import AnalysisResult
    from ..core.reading_diff import ReadingComparison

# The body's two arrangements, from the columns' own floors: the roll side
# keeps LEFT_COLUMN_MIN_W and the controls RIGHT_COLUMN_MIN_W, so side by
# side needs BODY_SIDE_BY_SIDE_MIN_W — below that the body STACKS (roll
# first, controls under it) and no control has to be reached through a
# horizontal scrollbar.
BODY_SIDE_BY_SIDE_MIN_W = 1180
# The two columns' floors: the roll side must stay readable (its plot widgets
# have no natural width, so the floor is the whole guarantee), the controls
# column must fit its widest single control without clipping.
LEFT_COLUMN_MIN_W = 520
RIGHT_COLUMN_MIN_W = 620
# Stacked, only one of them constrains the width: the roll side keeps a
# readable floor, the controls column takes the page's width.
STACKED_LEFT_COLUMN_MIN_W = 460
# (the edit actions stand in ONE COLUMN, never a row: five labeled buttons
# in a row need ~1234 px of column width, more than a windowed window's
# controls column gives — stacked, each keeps its own label)
# A short results view gives the roll and settings a vertical, user-adjustable
# split. Hiding the waveform and staff makes the roll useful at this height.
COMPACT_BODY_MAX_H = 520
COMPACT_ROLL_MIN_H = 195
COMPACT_SETTINGS_MIN_H = 169
COMPACT_ROLL_SHARE = 0.65

# The page's three reading surfaces, in the order the switcher lists them.
# ONE of them is shown (a view is a CHOICE, not a stacked column), and the
# notes roll is the default because that is where a take is corrected.
VIEW_NOTES = "notes"
VIEW_STAFF = "staff"
VIEW_WAVE = "wave"
VIEWS: tuple[str, ...] = (VIEW_NOTES, VIEW_STAFF, VIEW_WAVE)

# A view tab names the surface it shows: a serif-adjacent label underlined
# by the accent when chosen, the way a printed score marks its movements.
# The underline is always drawn (transparent when idle) so switching views
# never moves the row by a pixel.
def _view_chip_style() -> str:
    """Build the surface-choice style from the active palette."""
    return (
        # the tokens are read THROUGH the module: a `from .widgets import`
        # copy keeps the palette it was imported with, so a theme switch left
        # the tabs in the previous theme's colors
        "QPushButton {"
        " background: transparent; border: none; border-radius: 0;"
        " border-bottom: 2px solid transparent;"
        f" padding: 6px 10px 5px 10px; color: {widgets.TEXT_DIM}; font-size: 9.5pt;"
        "}"
        f"QPushButton:hover {{ color: {widgets.TEXT}; }}"
        f"QPushButton:focus {{ border-bottom: 2px dashed {widgets.ACCENT}; }}"
        f"QPushButton:checked {{ color: {widgets.TEXT};"
        f" border-bottom: 2px solid {widgets.ACCENT}; font-weight: 600; }}"
    )


class ArchiveComparisonDialog(QDialog):
    """Choose two saved readings and render their read-only note diff."""

    def __init__(
        self,
        labels: tuple[str, ...],
        compare: Callable[[int, int], "ReadingComparison"],
        parent: QWidget | None,
    ) -> None:
        super().__init__(parent)
        self._compare = compare
        self.setMinimumSize(620, 420)
        layout = QVBoxLayout(self)
        choices = QFormLayout()
        self.before = QComboBox()
        self.after = QComboBox()
        for index, label in enumerate(labels):
            self.before.addItem(label, index)
            self.after.addItem(label, index)
        self.after.setCurrentIndex(len(labels) - 1)
        choices.addRow(tr("ui compare before"), self.before)
        choices.addRow(tr("ui compare after"), self.after)
        layout.addLayout(choices)
        self.note = QLabel(tr("ui compare archive no coherence"))
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.compare_button = QPushButton(tr("ui compare action"))
        self.compare_button.clicked.connect(self._show_comparison)
        layout.addWidget(self.compare_button)
        self.summary = QLabel(tr("ui diff none"))
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.roll_headers = QHBoxLayout()
        self.before_title = QLabel()
        self.after_title = QLabel()
        self.roll_headers.addWidget(self.before_title)
        self.roll_headers.addWidget(self.after_title)
        layout.addLayout(self.roll_headers)
        self.rolls = QHBoxLayout()
        self.before_roll = PianoRoll()
        self.after_roll = PianoRoll()
        for roll in (self.before_roll, self.after_roll):
            roll.setMinimumHeight(220)
            roll.set_axis_labels(tr("key note col"), tr("key time col"))
            self.rolls.addWidget(roll, stretch=1)
        self.before_roll.getPlotItem().setXLink(self.after_roll)
        self.before_roll.getPlotItem().setYLink(self.after_roll)
        layout.addLayout(self.rolls, stretch=1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(120)
        self.details.setMaximumHeight(180)
        layout.addWidget(self.details)
        self.before_roll.hide_playhead()
        self.after_roll.hide_playhead()
        self.setWindowTitle(tr("ui compare archived readings title"))
        self._render_roll_titles()
        self._show_comparison()

    def _render_roll_titles(self) -> None:
        """Translate the two labels from the selectors' current values."""
        self.before_title.setText(str(self.before.currentText()))
        self.after_title.setText(str(self.after.currentText()))

    def _show_comparison(self) -> None:
        before_index = self.before.currentData()
        after_index = self.after.currentData()
        if before_index == after_index:
            self.summary.setText(tr("ui compare choose two"))
            self.details.clear()
            self.before_roll.set_notes([])
            self.after_roll.set_notes([])
            self._render_roll_titles()
            return
        try:
            comparison = self._compare(int(before_index), int(after_index))
        except Exception:
            self.summary.setText(tr("ui compare archive failed"))
            self.details.clear()
            self.before_roll.set_notes([])
            self.after_roll.set_notes([])
            self._render_roll_titles()
            return
        self.summary.setText(
            tr("ui diff archive compact").format(
                stable=comparison.count("stable"),
                changed=comparison.count("changed"),
                near=comparison.count("near"),
                missing=len(comparison.missing),
                added=len(comparison.added),
            )
        )
        before_notes = [
            Note(item.start, item.end, item.midi, 1.0) for item in comparison.before_notes
        ]
        after_notes = [pair.after for pair in comparison.pairs]
        after_notes.extend(comparison.added)
        after_notes.sort(key=lambda note: (note.start, note.index))
        self.before_roll.set_notes(before_notes, force=True)
        self.after_roll.set_notes(
            [Note(item.start, item.end, item.midi, 1.0) for item in after_notes], force=True
        )
        duration = max(
            0.5,
            max((note.end for note in (*before_notes, *after_notes)), default=0.5),
        )
        self.before_roll.getPlotItem().setXRange(0.0, duration, padding=0)
        pitches = [note.midi for note in (*before_notes, *after_notes)]
        low_pitch = max(float(PianoRoll.MIDI_MIN), min(pitches, default=64) - 6.0)
        high_pitch = min(float(PianoRoll.MIDI_MAX), max(pitches, default=64) + 6.0)
        if high_pitch - low_pitch < 12.0:
            low_pitch = max(float(PianoRoll.MIDI_MIN), high_pitch - 12.0)
            high_pitch = min(float(PianoRoll.MIDI_MAX), low_pitch + 12.0)
        self.before_roll.getPlotItem().setYRange(low_pitch, high_pitch, padding=0)
        self._render_roll_titles()

        rows: list[str] = []
        for pair in comparison.pairs:
            if pair.status == "stable":
                continue
            reasons = ", ".join(tr(f"diff reason {reason}") for reason in pair.reasons) or "—"
            rows.append(
                tr("ui diff pair row").format(
                    i=pair.before.index + 1,
                    before=note_display_pair(pair.before.midi),
                    after=note_display_pair(pair.after.midi),
                    status=tr(f"diff status {pair.status}"),
                    reasons=reasons,
                )
            )
        for note in comparison.missing:
            rows.append(
                tr("ui diff missing row").format(
                    i=note.index + 1,
                    before=note_display_pair(note.midi),
                    status=tr("diff status missing"),
                )
            )
        for note in comparison.added:
            rows.append(
                tr("ui diff added row").format(
                    i=note.index + 1,
                    after=note_display_pair(note.midi),
                    status=tr("diff status added"),
                )
            )
        self.details.setPlainText("\n".join(rows) if rows else tr("ui compare no changes"))


class ResultsPage(QWidget):
    """The analysis page's widgets (built here) and its rendering behavior."""

    def __init__(
        self,
        on_seek: Callable[[float], None],
        correct_note: Callable[[int, int], None],
        note_moved: Callable[[int, float, float, int], None],
        replace_phrase: Callable[[], None],
        finish_replacement: Callable[[], None],
        toggle_play: Callable[[], None],
        toggle_voice: Callable[[], None],
        toggle_both: Callable[[], None],
        piano_volume_changed: Callable[[int], None],
        transpose_changed: Callable[[int], None],
        export_again: Callable[[], None],
        reinterpret: Callable[[], None],
        full_reanalyze: Callable[[], None],
        open_file: Callable[[], None],
        back_home: Callable[[], None],
        undo_edit: Callable[[], None],
        redo_edit: Callable[[], None],
        jump_edit_state: Callable[[int], None],
        accept_replacement: Callable[[], None],
        cancel_replacement: Callable[[], None],
        audition_replacement: Callable[[], None],
        audition_original: Callable[[], bool],
        show_live_roll: Callable[[], None],
        open_archived_comparison: Callable[[], None],
        audition_note: Callable[[int], None],
        audition_pitch: Callable[[int], None],
        stop_audition: Callable[[], None],
        toggle_live_comparison: Callable[[bool], None],
    ) -> None:
        super().__init__()
        self.setMinimumSize(0, 0)
        self._body_mode: str | None = None
        self._compact_roll_share = COMPACT_ROLL_SHARE
        self._on_seek = on_seek
        self._correct_note = correct_note
        # the roll's drag-to-move (ONE recorded edit, like the correction)
        self._note_moved_cb = note_moved
        self._replace_phrase = replace_phrase
        self._finish_replacement = finish_replacement
        self._toggle_play = toggle_play
        self._toggle_voice = toggle_voice
        self._toggle_both = toggle_both
        self._piano_volume_changed = piano_volume_changed
        self._transpose_changed = transpose_changed
        self._export_again = export_again
        self._reinterpret = reinterpret
        self._full_reanalyze = full_reanalyze
        self._open_file = open_file
        self._back_home = back_home
        self._undo_edit = undo_edit
        self._redo_edit = redo_edit
        self._jump_edit_state = jump_edit_state
        self._accept_replacement = accept_replacement
        self._cancel_replacement = cancel_replacement
        self._audition_replacement = audition_replacement
        self._audition_original = audition_original
        self._show_live_roll = show_live_roll
        self._open_archived_comparison = open_archived_comparison
        self._audition_note_cb = audition_note
        # plays ONE candidate pitch while a key is being chosen (before Enter)
        self._audition_pitch = audition_pitch
        self._stop_audition_cb = stop_audition
        self._toggle_live_comparison_cb = toggle_live_comparison
        self._note_audition: tuple[int, int, int] | None = None
        # the sounding playback mode (None = stopped): ONE writer for the
        # three buttons' text — sync_play_buttons stores the fact and the
        # renderer applies it, so a language switch and a mode change never
        # disagree about the square/arrow prefix
        self._play_mode: str | None = None
        # the file name remembered for retranslate (a progress line writes
        # the label WITHOUT replacing it — see set_file_label)
        self._file_name: str | None = None
        # the shown reading surface (a page-internal display fact: the
        # window reads it through current_view() if it ever needs to)
        self._view = VIEW_NOTES
        # the notes the table's rows index into (refilled by set_result —
        # the toolbar's selection readout needs their time spans)
        self._notes: list = []
        self._has_live_result = False
        self._archive_comparison_available = False
        self._summary_facts: tuple[int, float, float | None, str | None] | None = None
        self._replacement_analyzing = False
        self._replacement_pending_ui = False
        self._full_reanalysis_active = False
        # the before/after reading diff kept across a retranslate (the core
        # comparison object, or None before any re-analysis succeeded)
        self._reading_diff = None
        # the undo/redo enabled state is a HISTORY fact (the window's edit
        # history owns it): stored here so a selection change can re-apply
        # it without the window having to re-push it
        self._can_undo = False
        self._can_redo = False
        # the state dropdown's rows (the window's edit history owns them):
        # kept so _update_edit_controls can re-disable after any refill and
        # retranslate can relabel in place without a re-push
        self._edit_states: list[tuple] = []
        self._guide_stage = "listen"
        self._guide_target: list[int] = []
        self._guide_targets: list[list[int]] = []
        self._guide_index = 0
        self._guide_has_audio = False
        self._guide_listening = False
        self._guide_adjusting = False
        self._build()

    # ------------------------------------------------------------------
    # widgets: the reading surface on the left, the task groups on the right
    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_back = QPushButton("")
        self.btn_back.clicked.connect(self._back_home)
        top.addWidget(self.btn_back)
        # the frozen live roll is a separate reading of the SAME take (the
        # notes as they were captured): the way back to it from the final
        # analysis, shown only for a live take
        self.btn_live_roll = QPushButton("")
        self.btn_live_roll.clicked.connect(self._show_live_roll)
        self.btn_live_roll.setVisible(False)
        top.addWidget(self.btn_live_roll)
        self.btn_compare_readings = QPushButton("")
        self.btn_compare_readings.clicked.connect(self._open_archived_comparison)
        self.btn_compare_readings.setVisible(False)
        top.addWidget(self.btn_compare_readings)
        self.chk_live_comparison = QCheckBox("")
        self.chk_live_comparison.setChecked(True)
        self.chk_live_comparison.toggled.connect(self._toggle_live_comparison_cb)
        self.chk_live_comparison.setVisible(False)
        top.addWidget(self.chk_live_comparison)
        self.label_file = QLabel("")
        self.label_file.setStyleSheet("font-weight: bold;")
        top.addWidget(self.label_file, stretch=1)
        root.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        self.progress.setMaximumHeight(10)
        root.addWidget(self.progress)

        left = QVBoxLayout()
        # ONE reading at a time, chosen by name: stacking the three surfaces
        # cost the roll its height (49 px on a short window) and gave the
        # waveform a slice nobody asked for.
        views = QHBoxLayout()
        views.setSpacing(6)
        self.btn_view_notes = QPushButton("")
        self.btn_view_staff = QPushButton("")
        self.btn_view_wave = QPushButton("")
        self._view_buttons = QButtonGroup(self)
        self._view_buttons.setExclusive(True)
        for button, name in (
            (self.btn_view_notes, VIEW_NOTES),
            (self.btn_view_staff, VIEW_STAFF),
            (self.btn_view_wave, VIEW_WAVE),
        ):
            button.setCheckable(True)
            button.setStyleSheet(_view_chip_style())
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self._view_buttons.addButton(button)
            button.clicked.connect(lambda _checked=False, view=name: self.set_view(view))
            views.addWidget(button)
        views.addStretch(1)
        left.addLayout(views)
        # the roll's own display option: coloured figures belong to the notes
        # view, so the checkbox lives with it and hides with it
        self.notation_row = QWidget()
        notation_row = QHBoxLayout(self.notation_row)
        notation_row.setContentsMargins(0, 0, 0, 0)
        notation_row.addStretch(1)
        self.chk_notation = QCheckBox("")
        self.chk_notation.toggled.connect(self._on_notation_toggled)
        notation_row.addWidget(self.chk_notation)
        left.addWidget(self.notation_row)
        # the diff legend ABOVE the roll: names "before" vs "after" so the
        # ghost wash reads without a second look (hidden until a diff exists)
        self.label_diff_legend = QLabel("")
        self.label_diff_legend.setVisible(False)
        left.addWidget(self.label_diff_legend)
        self.view_stack = QStackedWidget()
        self.waveform = WaveformView()
        self.staff = StaffView()
        self.piano = PianoRoll(selectable=True)
        self.piano.setMinimumHeight(125)
        # the results roll's bars DRAG (onset/duration/pitch as one edit);
        # the live review roll stays pick-only
        self.piano.set_editable(True)
        self.piano.seek_requested.connect(self._on_seek)
        self.piano.note_selection.connect(self._on_roll_selection)
        self.piano.note_dragged.connect(self._note_moved_cb)
        self.piano.audition_requested.connect(self._audition_note_cb)
        for surface in (self.piano, self.staff, self.waveform):
            self.view_stack.addWidget(surface)
        left.addWidget(self.view_stack, stretch=1)

        self.piano_keys = PianoView()
        self.piano_keys.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.piano_keys.setToolTip(tr("ui piano keyboard help"))
        self.piano_keys.key_clicked.connect(self._on_key_clicked)
        self.piano_keys.installEventFilter(self)
        left.addWidget(self.piano_keys)
        self.left_host = QWidget()
        self.left_host.setLayout(left)
        self.left_host.setMinimumWidth(STACKED_LEFT_COLUMN_MIN_W)

        right = QVBoxLayout()
        right.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)
        self.results = ResultsPanel()
        self.results._notes_table.setSelectionBehavior(
            self.results._notes_table.SelectionBehavior.SelectRows
        )
        self.results._notes_table.setSelectionMode(
            self.results._notes_table.SelectionMode.ExtendedSelection
        )
        self.results._notes_table.itemSelectionChanged.connect(self._on_table_selection)
        self.results._notes_table.audition_requested.connect(self._audition_note_cb)
        self.summary_card = QGroupBox("")
        summary = QVBoxLayout(self.summary_card)
        self.label_summary = QLabel("")
        self.label_summary.setWordWrap(True)
        self.label_summary.setStyleSheet("font-size: 11pt; font-weight: 600;")
        summary.addWidget(self.label_summary)
        # the summary is one line of facts, never a panel: without a Maximum
        # policy the column hands it the leftover height and the card opens
        # an empty trough under the count
        self.summary_card.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        self.results.setMinimumHeight(0)
        self.results.setVisible(False)
        right.addWidget(self.summary_card)
        self.guide_card = QGroupBox("")
        guide = QVBoxLayout(self.guide_card)
        self.label_guide = QLabel("")
        self.label_guide.setWordWrap(True)
        guide.addWidget(self.label_guide)
        self.btn_guide = QPushButton("")
        self.btn_guide.setStyleSheet(action_button_style(primary=True))
        self.btn_guide.setMinimumHeight(46)
        self.btn_guide.clicked.connect(self._activate_guidance)
        guide.addWidget(self.btn_guide)
        self.btn_guide_skip = QPushButton("")
        self.btn_guide_skip.clicked.connect(self._skip_guidance)
        self.btn_guide_skip.setStyleSheet(link_button_style())
        guide.addWidget(self.btn_guide_skip, 0, Qt.AlignmentFlag.AlignLeft)
        right.addWidget(self.guide_card)
        self.tools_toggle = QPushButton("")
        self.tools_toggle.setCheckable(True)
        self.tools_toggle.toggled.connect(self._render_guidance)
        self.tools_toggle.setStyleSheet(link_button_style())
        right.addWidget(self.tools_toggle, 0, Qt.AlignmentFlag.AlignLeft)

        # ---- Listen: one NAMED source per action ----
        # each button still starts or stops its own mode; what changes is
        # that it says WHAT it plays (three symbols made the user learn
        # three meanings before pressing anything).
        self.group_listen = QGroupBox("")
        listen = QVBoxLayout(self.group_listen)
        sources = QHBoxLayout()
        self.btn_play = QPushButton("")
        # the piano reading is what a singer checks first: it is the group's
        # primary action, Voice and Together stay named alternatives beside it
        self.btn_play.setStyleSheet(action_button_style(primary=True))
        self.btn_play.setMinimumWidth(64)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_voice = QPushButton("")
        self.btn_voice.setMinimumWidth(64)
        self.btn_voice.clicked.connect(self._toggle_voice)
        self.btn_both = QPushButton("")
        self.btn_both.setMinimumWidth(72)
        self.btn_both.clicked.connect(self._toggle_both)
        for button in (self.btn_play, self.btn_voice, self.btn_both):
            sources.addWidget(button)
        sources.addStretch(1)
        listen.addLayout(sources)
        self.label_play_pos = QLabel("")
        self.label_note_audition = QLabel("")
        self.label_note_audition.setVisible(False)
        self.btn_stop_audition = QPushButton("")
        self.btn_stop_audition.clicked.connect(self._stop_audition_cb)
        self.btn_stop_audition.setVisible(False)
        listen_row = QHBoxLayout()
        listen_row.addWidget(self.label_play_pos, stretch=1)
        listen_row.addWidget(self.label_note_audition)
        listen_row.addWidget(self.btn_stop_audition)
        listen.addLayout(listen_row)
        # the rendered piano's volume, for comparing it with the singer's voice
        self.label_volume = QLabel("")
        self.slider_volume = QSlider(self._qt_orientation())
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(70)
        self.slider_volume.setMaximumWidth(150)
        self.slider_volume.valueChanged.connect(self._piano_volume_changed)
        volume_row = QHBoxLayout()
        volume_row.addWidget(self.label_volume)
        volume_row.addWidget(self.slider_volume)
        volume_row.addStretch(1)
        listen.addLayout(volume_row)
        right.addWidget(self.group_listen)

        # ---- Adjust: what acts on the SELECTED notes ----
        self.group_adjust = QGroupBox("")
        adjust = QVBoxLayout(self.group_adjust)
        self.label_edit_hint = QLabel("")
        self.label_edit_hint.setWordWrap(True)
        adjust.addWidget(self.label_edit_hint)
        # undo/redo share ONE row (short labels); the actions on the
        # selected range keep a full-width button each — five labeled
        # buttons in a row need ~1234 px of column width
        history_row = QHBoxLayout()
        self.btn_undo = QPushButton("")
        self.btn_undo.clicked.connect(self._undo_edit)
        self.btn_redo = QPushButton("")
        self.btn_redo.clicked.connect(self._redo_edit)
        history_row.addWidget(self.btn_undo)
        history_row.addWidget(self.btn_redo)
        adjust.addLayout(history_row)
        self.btn_replace_phrase = QPushButton("")
        self.btn_replace_phrase.clicked.connect(self._replace_phrase)
        adjust.addWidget(self.btn_replace_phrase)
        self.btn_audition_original = QPushButton("")
        self.btn_audition_original.clicked.connect(self._audition_original)
        adjust.addWidget(self.btn_audition_original)
        self.btn_finish_replacement = QPushButton("")
        self.btn_finish_replacement.clicked.connect(self._finish_replacement)
        self.btn_finish_replacement.setVisible(False)
        adjust.addWidget(self.btn_finish_replacement)
        # the toolbar's live readout: how many notes over which seconds —
        # the concrete "what am I about to delete/replace" under the actions
        self.label_selection = QLabel("")
        self.label_selection.setStyleSheet(f"color: {widgets.TEXT_DIM}; font-size: 9pt;")
        self.label_selection.setVisible(False)
        adjust.addWidget(self.label_selection)
        # the take's edit history as one dropdown: every state (what it was
        # before, in the edit's words), the current one last. Choosing a row
        # jumps back to it (a stepped undo/redo — the history keeps ONE
        # owner of the branch bookkeeping); picking the current row does
        # nothing.
        self.combo_edit_states = QComboBox()
        self.combo_edit_states.currentIndexChanged.connect(self._on_state_chosen)
        adjust.addWidget(self.combo_edit_states)
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
        adjust.addLayout(pending_buttons)
        right.addWidget(self.group_adjust)

        # ---- Export: ONE primary action, and it says what it writes ----
        self.group_export = QGroupBox("")
        export = QVBoxLayout(self.group_export)
        self.btn_export = QPushButton("")
        self.btn_export.setStyleSheet(action_button_style(primary=True))
        self.btn_export.setMinimumHeight(46)
        self.btn_export.clicked.connect(self._export_again)
        export.addWidget(self.btn_export)
        self.chk_pdf = QCheckBox("")
        from ..core.pdf_export import find_notation_tool

        self.chk_pdf.setEnabled(find_notation_tool() is not None)
        export.addWidget(self.chk_pdf)
        self.lbl_pdf_note = QLabel("")
        self.lbl_pdf_note.setStyleSheet(
            f"color: {widgets.TEXT_DIM}; font-size: 8pt; font-style: italic;"
        )
        self.lbl_pdf_note.setWordWrap(True)
        export.addWidget(self.lbl_pdf_note)
        # the advanced settings start collapsed, and their split names the
        # STAGE a setting belongs to: "minimum note" drove the re-read, yet
        # sat among the export options as if it only changed the file
        self.options_toggle = QPushButton("")
        self.options_toggle.setCheckable(True)
        self.options_toggle.setChecked(False)
        self.options_toggle.toggled.connect(self._toggle_options)
        export.addWidget(self.options_toggle)
        self.options_host = QWidget()
        options = QVBoxLayout(self.options_host)
        options.setContentsMargins(0, 0, 0, 0)
        self.grp_export_opts = QGroupBox("")
        export_form = QGridLayout(self.grp_export_opts)
        self.chk_quantize = QCheckBox("")
        export_form.addWidget(self.chk_quantize, 0, 0, 1, 2)
        self._lbl_transpose = QLabel("")
        export_form.addWidget(self._lbl_transpose, 1, 0)
        self.spin_transpose = QSpinBox()
        self.spin_transpose.setRange(-48, 48)
        self.spin_transpose.valueChanged.connect(self._transpose_changed)
        export_form.addWidget(self.spin_transpose, 1, 1)
        options.addWidget(self.grp_export_opts)
        # the interpretation thresholds the "re-read" button uses: they change
        # HOW the take's evidence is read, never the evidence itself (pYIN does
        # not rerun — the cached observation is re-interpreted in milliseconds)
        self.grp_transcription = QGroupBox("")
        transcribe = QGridLayout(self.grp_transcription)
        self._lbl_min_note = QLabel("")
        transcribe.addWidget(self._lbl_min_note, 0, 0)
        self.spin_min_note = QSpinBox()
        self.spin_min_note.setRange(40, 300)
        self.spin_min_note.setValue(60)
        self.spin_min_note.setSuffix(" ms")
        transcribe.addWidget(self.spin_min_note, 0, 1)
        # A fast cached-evidence read is deliberately distinct from a fresh
        # pYIN pass over the retained audio.
        self.btn_reinterpret = QPushButton("")
        self.btn_reinterpret.clicked.connect(self._reinterpret)
        transcribe.addWidget(self.btn_reinterpret, 1, 0, 1, 2)
        self.btn_full_reanalyze = QPushButton("")
        self.btn_full_reanalyze.clicked.connect(self._full_reanalyze)
        transcribe.addWidget(self.btn_full_reanalyze, 2, 0, 1, 2)
        options.addWidget(self.grp_transcription)
        export.addWidget(self.options_host)
        self.options_host.setVisible(False)
        right.addWidget(self.group_export)

        self.details_toggle = QPushButton("")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setChecked(False)
        self.details_toggle.toggled.connect(self._toggle_details)
        self.details_toggle.setText(tr("ui show details"))
        self.details_toggle.setStyleSheet(link_button_style())
        right.addWidget(self.details_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.details_scroll.setMinimumHeight(0)
        self.details_scroll.setWidget(self.results)
        self.details_scroll.setVisible(False)
        right.addWidget(self.details_scroll)
        self.btn_open = QPushButton("")
        self.btn_open.clicked.connect(self._open_file)
        self.btn_open.setStyleSheet(link_button_style())
        right.addWidget(self.btn_open, 0, Qt.AlignmentFlag.AlignLeft)
        # the leftover height belongs AFTER the column, not inside whichever
        # card happens to be visible: without it the sections grew apart and
        # the summary, guide and listen blocks floated in empty space
        right.addStretch(1)

        # The complete control column scrolls as one unit. The compact
        # summary and export remain at the top of this column's scroll range.
        right_content = QWidget()
        right_content.setLayout(right)
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.right_scroll.setWidget(right_content)
        self.right_scroll.setMinimumWidth(0)

        # The two columns side by side while the window has the width for
        # both, stacked when it does not: measured, their minimums sum past
        # 1787 px, so in any narrower window the controls column was
        # squeezed to ~300 px and hid its own Export button behind a
        # horizontal scrollbar — at the DEFAULT 1280x780 window too.
        #
        # The roll side's horizontal hint (1085 px) is deliberately IGNORED:
        # a plot widget has no natural width, and letting its hint compete
        # with the controls is exactly what starved them. It keeps a floor
        # (the roll, staff and keyboard must stay readable) and takes the
        # rest; `_arrange_body` flips the orientation on resize.
        self.left_host.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.body = QSplitter(Qt.Orientation.Horizontal)
        self.body.setHandleWidth(12)
        self.body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._refresh_theme_styles()
        self.body.setChildrenCollapsible(False)
        self.body.splitterMoved.connect(self._remember_body_sizes)
        self.body.addWidget(self.left_host)
        self.body.addWidget(self.right_scroll)
        root.addWidget(self.body, stretch=1)
        # Keyboard traversal in task order — the SAME order the screen
        # reads: choose a surface, open the details, pick a note, listen,
        # adjust, then export or re-read.
        focus_order = (
            self.btn_back,
            self.btn_live_roll,
            self.btn_compare_readings,
            self.chk_live_comparison,
            self.btn_view_notes,
            self.btn_view_staff,
            self.btn_view_wave,
            self.chk_notation,
            self.btn_guide,
            self.btn_guide_skip,
            self.tools_toggle,
            self.details_toggle,
            self.results._notes_table,
            self.piano,
            self.piano_keys,
            self.btn_play,
            self.btn_voice,
            self.btn_both,
            self.btn_stop_audition,
            self.slider_volume,
            self.btn_undo,
            self.btn_redo,
            self.btn_replace_phrase,
            self.btn_audition_original,
            self.btn_finish_replacement,
            self.combo_edit_states,
            self.btn_audition_replacement,
            self.btn_accept_replacement,
            self.btn_cancel_replacement,
            self.chk_pdf,
            self.options_toggle,
            self.chk_quantize,
            self.spin_transpose,
            self.spin_min_note,
            self.btn_reinterpret,
            self.btn_full_reanalyze,
            self.btn_export,
            self.btn_open,
        )
        self._focus_order = focus_order
        for first, second in zip(focus_order, focus_order[1:], strict=False):
            QWidget.setTabOrder(first, second)
        self.set_view(VIEW_NOTES)
        self._render_play_buttons()
        self._render_guidance()
        self._refresh_theme_styles()

    def _render_guidance(self) -> None:
        expanded = self.tools_toggle.isChecked()
        self.tools_toggle.setText(tr("ui guide hide tools" if expanded else "ui guide tools"))
        self.guide_card.setTitle(tr("ui guide title"))
        self.btn_guide_skip.setText(tr("ui guide skip"))
        # Listening is not a tool to unfold: the piano reading is available as
        # soon as a result is shown, and the guide only points at WHERE to
        # listen. Adjust and Export keep their disclosure.
        self.group_listen.setVisible(True)
        self.group_adjust.setVisible(expanded or self._guide_adjusting)
        self.group_export.setVisible(expanded)
        busy = (
            self._full_reanalysis_active
            or self._replacement_analyzing
            or self._replacement_pending_ui
            or not self.btn_finish_replacement.isHidden()
        )
        self.guide_card.setVisible(not busy)
        if busy:
            self.group_adjust.setVisible(True)
        self.btn_guide_skip.setVisible(self._guide_stage != "export")
        if self._guide_stage == "listen":
            key = "ui guide listen passage" if self._guide_target else "ui guide listen take"
            self.label_guide.setText(
                tr("ui guide doubt" if self._guide_target else "ui guide listen")
            )
            self.btn_guide.setText(tr("ui guide stop" if self._guide_listening else key))
            self.btn_guide.setEnabled(self._guide_has_audio)
        elif self._guide_stage == "next":
            self.label_guide.setText(tr("ui guide next hint"))
            self.btn_guide.setText(tr("ui guide next"))
            self.btn_guide.setEnabled(True)
        elif self._guide_stage == "review":
            self.label_guide.setText(tr("ui guide review hint"))
            self.btn_guide.setText(tr("ui guide review"))
            self.btn_guide.setEnabled(bool(self._notes))
        else:
            self.label_guide.setText(tr("ui guide export hint"))
            self.btn_guide.setText(tr("ui export"))
            self.btn_guide.setEnabled(self.btn_export.isEnabled() and bool(self._notes))
        if self._guide_targets and self._guide_stage != "export":
            counter = tr("ui guide progress").format(
                current=self._guide_index + 1, total=len(self._guide_targets)
            )
            self.label_guide.setText(f"{counter} — {self.label_guide.text()}")

    def _activate_guidance(self) -> None:
        if self._guide_stage == "listen":
            if self._guide_listening:
                self._stop_audition_cb()
                return
            self._on_roll_selection(self._guide_target or list(range(len(self._notes))))
            if self._audition_original():
                self._guide_listening = True
                self._render_guidance()
        elif self._guide_stage == "review":
            self.set_view(VIEW_NOTES)
            self._on_roll_selection(self._guide_target or list(range(len(self._notes))))
            self.details_toggle.setChecked(True)
            self.piano_keys.setFocus()
            self._guide_stage = (
                "next" if self._guide_index + 1 < len(self._guide_targets) else "export"
            )
            self._guide_adjusting = True
            self._render_guidance()
        elif self._guide_stage == "next":
            self._guide_index += 1
            self._guide_target = self._guide_targets[self._guide_index]
            self._guide_adjusting = False
            self._guide_stage = "listen" if self._guide_has_audio else "review"
            self._on_roll_selection(self._guide_target)
            self._render_guidance()
            self.btn_guide.setFocus()
        else:
            self.btn_export.click()

    def _skip_guidance(self) -> None:
        if self._guide_listening:
            self._stop_audition_cb()
        self._guide_stage = "export"
        self._guide_listening = False
        self._render_guidance()
        self.btn_guide.setFocus()

    def _remember_body_sizes(self, _position: int, _index: int) -> None:
        """Keep the compact split proportional when the window is resized."""
        if self._body_mode != "compact":
            return
        sizes = self.body.sizes()
        total = sum(sizes)
        if total:
            self._compact_roll_share = sizes[0] / total
            self._pending_body_share = self._compact_roll_share

    def _arrange_body(self) -> None:
        """Choose a useful split for the available width and height.

        At compact heights the roll and controls share a vertical splitter;
        the default gives the roll about 65 percent, while the handle lets
        the user trade that space for settings. At normal heights, wide
        windows stay side by side and narrow ones stack. Floors are changed
        on every pass, but sizes are reset only when the mode changes so a
        resize does not undo the user's drag.
        """
        width = self.body.width() or self.width()
        height = self.body.height()
        compact = 0 < height <= COMPACT_BODY_MAX_H
        side_by_side = width >= BODY_SIDE_BY_SIDE_MIN_W and not compact
        mode = "compact" if compact else "side-by-side" if side_by_side else "stacked"

        if mode == "compact":
            self.left_host.setMinimumWidth(STACKED_LEFT_COLUMN_MIN_W)
            self.right_scroll.setMinimumWidth(0)
            self.left_host.setMinimumHeight(COMPACT_ROLL_MIN_H)
            self.right_scroll.setMinimumHeight(COMPACT_SETTINGS_MIN_H)
        else:
            self.left_host.setMinimumWidth(
                LEFT_COLUMN_MIN_W if side_by_side else STACKED_LEFT_COLUMN_MIN_W
            )
            self.right_scroll.setMinimumWidth(RIGHT_COLUMN_MIN_W if side_by_side else 0)
            self.left_host.setMinimumHeight(0)
            self.right_scroll.setMinimumHeight(0)
        # WHICH surface fills the reading area is the view switcher's
        # business, never the body arrangement's: the chosen view keeps its
        # height at every window size (a fold here used to override the
        # user's choice).

        if mode == self._body_mode:
            return
        self._body_mode = mode
        self.body.setOrientation(
            Qt.Orientation.Horizontal if side_by_side else Qt.Orientation.Vertical
        )
        if compact:
            first_share = self._compact_roll_share
        elif side_by_side:
            first_share = 0.75
        else:
            first_share = 0.4
        self._pending_body_share = first_share
        self._apply_body_share()

    def _apply_body_share(self) -> None:
        """Apply the selected ratio after the splitter has its final geometry."""
        extent = (
            self.body.width()
            if self.body.orientation() == Qt.Orientation.Horizontal
            else self.body.height()
        )
        available = max(0, extent - self.body.handleWidth())
        if available:
            first = round(available * self._pending_body_share)
            self.body.setSizes([first, available - first])

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Re-arrange the body for the width and height the page was given."""
        super().resizeEvent(event)
        self._arrange_body()
        self._apply_body_share()

    @staticmethod
    def _qt_orientation():
        """The volume slider's horizontal orientation (import kept local)."""
        from PyQt6.QtCore import Qt

        return Qt.Orientation.Horizontal

    def _render_summary(self) -> None:
        """Render the compact facts in the active language."""
        if self._summary_facts is None:
            self.label_summary.setText("")
            return
        notes, duration, bpm, key_label = self._summary_facts
        tempo = f"{bpm:.0f} BPM" if bpm is not None else "—"
        key = tr_key_label(key_label) if key_label is not None else "—"
        self.label_summary.setText(
            tr("ui summary facts").format(notes=notes, duration=duration, tempo=tempo, key=key)
        )

    def _toggle_details(self, expanded: bool) -> None:
        """Show or hide the diagnostic table without hiding the summary."""
        self.details_scroll.setVisible(expanded)
        self.results.setVisible(True)
        self.details_toggle.setText(tr("ui hide details") if expanded else tr("ui show details"))

    def _on_notation_toggled(self) -> None:
        """The checkbox drives THIS page's roll only — no window fact."""
        self.piano.set_notation_mode(self.chk_notation.isChecked())

    def _toggle_options(self, expanded: bool) -> None:
        """Reveal the advanced settings (export format vs transcription)."""
        self.options_host.setVisible(expanded)
        self.options_toggle.setText(
            tr("ui hide advanced settings") if expanded else tr("ui advanced settings")
        )

    # ------------------------------------------------------------------
    # the reading surface: ONE view at a time (the named surface)
    # ------------------------------------------------------------------
    def set_view(self, name: str) -> None:
        """Show one reading surface and keep its own options in step.

        One writer for three facts that must agree: the stacked surface,
        the chip that reads as current, and what the notes view alone
        offers (the coloured figures checkbox, the diff legend).
        """
        if name not in VIEWS:
            raise ValueError(f"unknown view: {name}")
        self._view = name
        self.view_stack.setCurrentIndex(VIEWS.index(name))
        button = {
            VIEW_NOTES: self.btn_view_notes,
            VIEW_STAFF: self.btn_view_staff,
            VIEW_WAVE: self.btn_view_wave,
        }[name]
        if not button.isChecked():
            button.setChecked(True)
        self.notation_row.setVisible(name == VIEW_NOTES)
        # the ghost bars live on the roll: their legend follows that view
        self.label_diff_legend.setVisible(self._reading_diff is not None and name == VIEW_NOTES)

    def current_view(self) -> str:
        """Which reading surface is shown (one of VIEWS)."""
        return self._view

    # ------------------------------------------------------------------
    # strings
    # ------------------------------------------------------------------
    def refresh_theme(self) -> None:
        """Refresh locally assembled colors without changing reading facts."""
        self._refresh_theme_styles()

    def _refresh_theme_styles(self) -> None:
        """Rebuild the page-local styles from the current shared palette."""
        for button in self._view_buttons.buttons():
            button.setStyleSheet(_view_chip_style())
        self.label_selection.setStyleSheet(f"color: {widgets.TEXT_DIM}; font-size: 9pt;")
        self.lbl_pdf_note.setStyleSheet(
            f"color: {widgets.TEXT_DIM}; font-size: 8pt; font-style: italic;"
        )
        self.btn_guide.setStyleSheet(action_button_style(primary=True))
        self.btn_export.setStyleSheet(action_button_style(primary=True))
        self.btn_play.setStyleSheet(action_button_style(primary=True))
        for link in (
            self.btn_guide_skip,
            self.tools_toggle,
            self.details_toggle,
            self.btn_open,
        ):
            link.setStyleSheet(link_button_style())
        self.body.setStyleSheet(
            f"QSplitter::handle:vertical {{ background: {widgets.BORDER_SOFT}; }}"
            f"QSplitter::handle:vertical:hover {{ background: {widgets.ACCENT}; }}"
        )
        self.piano.setBackground(widgets.BG_APP)
        self.waveform.setBackground(widgets.BG_APP)
        self.piano.set_notes(self.piano._last_notes, force=True)
        self.piano._render_selection()
        self.staff.update()
        self.piano_keys.update()
        self.waveform.update()

    def retranslate(self) -> None:
        """The page's user-facing strings (the window's retranslate calls it)."""
        self.btn_back.setText(tr("ui back home"))
        self.summary_card.setTitle(tr("ui take summary"))
        self._render_summary()
        self.details_toggle.setText(
            tr("ui hide details") if self.details_toggle.isChecked() else tr("ui show details")
        )
        self._toggle_details(self.details_toggle.isChecked())
        self.btn_live_roll.setText(tr("ui live roll"))
        self.btn_live_roll.setToolTip(tr("ui live roll tooltip"))
        self.btn_live_roll.setVisible(self._has_live_result)
        self.btn_compare_readings.setText(tr("ui compare archived readings"))
        self.btn_compare_readings.setToolTip(tr("ui compare archived readings tooltip"))
        self.btn_compare_readings.setVisible(
            self._has_live_result and self._archive_comparison_available
        )
        self.chk_live_comparison.setText(tr("ui live comparison"))
        self.chk_live_comparison.setToolTip(tr("ui live comparison tooltip"))
        self.chk_live_comparison.setVisible(self._has_live_result)
        if self._file_name:
            self.label_file.setText(self._file_name)
        else:
            self.label_file.setText(tr("ui no file"))
        self._render_guidance()
        self.group_listen.setTitle(tr("ui group listen"))
        self.group_adjust.setTitle(tr("ui group adjust"))
        self.group_export.setTitle(tr("ui group export"))
        self.btn_view_notes.setText(tr("ui view notes"))
        self.btn_view_notes.setToolTip(tr("ui view notes tooltip"))
        self.btn_view_staff.setText(tr("ui view staff"))
        self.btn_view_staff.setToolTip(tr("ui staff tooltip"))
        self.btn_view_wave.setText(tr("ui view wave"))
        self.btn_view_wave.setToolTip(tr("ui view wave tooltip"))
        self.grp_export_opts.setTitle(tr("ui export options"))
        self.grp_transcription.setTitle(tr("ui opts transcription"))
        self.options_toggle.setText(
            tr("ui hide advanced settings")
            if self.options_toggle.isChecked()
            else tr("ui advanced settings")
        )
        self.chk_quantize.setText(tr("ui quantize"))
        self._lbl_min_note.setText(tr("ui min note"))
        self.btn_reinterpret.setText(tr("ui reinterpret"))
        self.btn_reinterpret.setToolTip(tr("ui reinterpret tooltip"))
        self.btn_full_reanalyze.setText(tr("ui full reanalyze"))
        self.btn_full_reanalyze.setToolTip(tr("ui full reanalyze tooltip"))
        self._lbl_transpose.setText(tr("ui transpose"))
        self.chk_notation.setText(tr("ui notation mode"))
        self.chk_pdf.setText(tr("ui pdf option"))
        self.chk_pdf.setToolTip(tr("ui pdf tooltip"))
        self.lbl_pdf_note.setVisible(not self.chk_pdf.isEnabled())
        self.lbl_pdf_note.setText(tr("ui pdf badge"))
        self.btn_export.setText(tr("ui export"))
        self.btn_export.setToolTip(tr("ui export tooltip"))
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
        self.piano_keys.setToolTip(tr("ui piano keyboard help"))
        for row_index, (_state, action) in enumerate(self._edit_states):
            self.combo_edit_states.setItemText(row_index, tr(action))
        self.btn_open.setText(tr("ui open another"))
        self._render_play_buttons()  # state prefix + translated name, together
        self.btn_stop_audition.setText(tr("ui piano stop audition"))
        self.btn_stop_audition.setToolTip(tr("ui piano stop audition tooltip"))
        self._render_note_audition()
        self.label_volume.setText(tr("ui volume"))
        self.slider_volume.setToolTip(tr("ui volume tooltip"))
        self.label_play_pos.setText(tr("ui play pos"))
        if self._reading_diff is not None:
            # the kept facts re-render (the core object is untouched)
            self.label_diff_legend.setText(self._diff_legend_text())
            self.set_view(self._view)
            self.results.set_reading_diff(self._reading_diff)

    # ------------------------------------------------------------------
    # the named surface (ADR 0039): the window reads/writes the page's
    # options through verbs, never through the page's spin boxes
    # ------------------------------------------------------------------
    def transpose_st(self) -> int:
        """The transpose box's value (semitones; the export and voice
        player both read it)."""
        return self.spin_transpose.value()

    def min_note_ms(self) -> int:
        """The minimum-note threshold in milliseconds (the reinterpret
        and analysis options read it)."""
        return self.spin_min_note.value()

    def quantize_checked(self) -> bool:
        """Whether the MIDI export should quantize to the grid."""
        return self.chk_quantize.isChecked()

    def pdf_checked(self) -> bool:
        """Whether the export engraves a PDF."""
        return self.chk_pdf.isChecked()

    def set_source_name(self, name: str) -> None:
        """Remember the take's display name (survives a retranslate)."""
        self._file_name = name
        self.label_file.setText(name)

    def set_file_label(self, text: str) -> None:
        """Transient label write (an analysis progress line passes through)."""
        self.label_file.setText(text)

    def begin_analysis(self) -> None:
        """Reset the views for a new analysis: buttons up, progress at 0."""
        self.sync_play_buttons(None)
        self.label_play_pos.setText(tr("ui analyzing file"))
        self.progress.setValue(0)

    def set_progress(self, percent: int) -> None:
        self.progress.setValue(percent)

    def set_reading_diff(self, comparison) -> None:
        """Show one before/after comparison after a successful re-analysis.

        The comparison is RENDER facts only: ghost bars on the roll (the
        old reading), the accented changed bars, the compact count line and
        the suspect detail — all computed by the core (`reading_diff`),
        held here for retranslation. Clearing is `clear_reading_diff`.
        """
        self._reading_diff = comparison
        changed_afters = tuple(p.after.index for p in comparison.pairs if p.status == "changed")
        near_afters = tuple(p.after.index for p in comparison.pairs if p.status == "near")
        self.piano.set_ghost_notes(
            [(n.start, n.end, n.midi) for n in comparison.before_notes],
            changed_afters=changed_afters,
            near_afters=near_afters,
        )
        self.results.set_reading_diff(comparison)
        self.label_diff_legend.setText(self._diff_legend_text())
        self.set_view(self._view)  # the legend belongs to the notes view

    def clear_reading_diff(self) -> None:
        """Drop the before/after overlay and detail (a new reading/edit boundary).
        Idempotent, and harmless before any comparison was shown."""
        self._reading_diff = None
        self.piano.clear_ghost_notes()
        self.results.clear_reading_diff()
        self.set_view(self._view)  # one writer hides the legend again

    def _diff_legend_text(self) -> str:
        return tr("ui diff legend")

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

        The gesture can name SEPARATED notes (Ctrl/Cmd-click toggles one bar
        at a time), so the rows are selected one by one rather than as one
        contiguous range; the table's own `itemSelectionChanged` then runs
        `_on_table_selection` — the highlight and the edit controls refresh
        through the same path a table click takes, with no second
        bookkeeping.
        """
        table = self.results._notes_table
        model = table.model()
        selection = table.selectionModel()
        selection.clearSelection()
        if not indices:
            return
        for row in indices:
            selection.select(
                model.index(row, 0),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )
        # the LAST picked note is the cursor (arrow keys continue from it)
        # without disturbing the set just built
        selection.setCurrentIndex(
            model.index(indices[-1], 0), QItemSelectionModel.SelectionFlag.NoUpdate
        )

    def _update_edit_controls(self) -> None:
        self._apply_edit_controls()
        self._render_guidance()

    def _apply_edit_controls(self) -> None:
        if self._full_reanalysis_active:
            for button in (
                self.btn_export,
                self.btn_reinterpret,
                self.btn_full_reanalyze,
                self.btn_open,
                self.btn_undo,
                self.btn_redo,
                self.btn_replace_phrase,
                self.btn_audition_original,
                self.btn_finish_replacement,
            ):
                button.setEnabled(False)
            self.combo_edit_states.setEnabled(False)
            self.label_edit_hint.setText(tr("ui full reanalysis running"))
            return
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
        self.btn_full_reanalyze.setEnabled(not self.btn_finish_replacement.isVisible())
        self.btn_undo.setEnabled(self._can_undo)
        self.btn_redo.setEnabled(self._can_redo)
        self.combo_edit_states.setEnabled(True)
        self.label_edit_hint.setText(
            tr("ui edit selection").format(first=indices[0] + 1, last=indices[-1] + 1)
            if contiguous
            else tr("ui edit hint")
        )

    def _current_result_is_live(self) -> bool:
        return self._has_live_result

    def _on_key_clicked(self, midi: int) -> None:
        indices = self._selected_note_indices()
        if len(indices) == 1:
            self._correct_note(indices[0], midi)

    def _apply_staged_key(self) -> None:
        """Apply the pitch staged on the focused keyboard (Enter).

        The staged pitch IS the widget's keyboard cursor — one fact, drawn
        where the user looks; it exists only while the keyboard has focus.
        """
        midi = self.piano_keys.keyboard_cursor
        if midi is None:
            return
        self.piano_keys.set_keyboard_cursor(None)
        self._on_key_clicked(midi)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Give the focused results page a keyboard route into its note list."""
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Home, Qt.Key.Key_End):
            self.results._notes_table.setFocus()
            self.results._notes_table.keyPressEvent(event)
            return
        if event.key() == Qt.Key.Key_Space:
            indices = self._selected_note_indices()
            if len(indices) == 1:
                self._audition_note_cb(indices[0])
                event.accept()
                return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event):
        """Stage, step and apply a piano-key correction from the keyboard.

        Enter/Escape/arrows are consumed here so the correction is STAGED
        (arrows choose, Enter applies, Escape cancels) instead of firing
        on a key press. The staged pitch lives only while the keyboard is
        focused: staged on focus from the selected note, cleared on blur.
        """
        if watched is not self.piano_keys:
            return super().eventFilter(watched, event)
        kind = event.type()
        if kind == QEvent.Type.FocusIn:
            indices = self._selected_note_indices()
            midi = (
                self._notes[indices[0]].midi
                if len(indices) == 1 and indices[0] < len(self._notes)
                else None
            )
            # stage only a pitch this keyboard renders: the mouse cannot
            # point at an invisible key either
            if midi is not None and not (
                self.piano_keys.min_midi <= midi <= self.piano_keys.max_midi
            ):
                midi = None
            self.piano_keys.set_keyboard_cursor(midi)
            return super().eventFilter(watched, event)
        if kind == QEvent.Type.FocusOut:
            self.piano_keys.set_keyboard_cursor(None)
            return super().eventFilter(watched, event)
        if kind == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._apply_staged_key()
                event.accept()
                return True
            if event.key() == Qt.Key.Key_Escape:
                self.piano_keys.set_keyboard_cursor(None)
                event.accept()
                return True
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                current = self.piano_keys.keyboard_cursor
                if current is None:
                    current = 60  # middle C when nothing is staged yet
                step = -1 if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Down) else 1
                lo, hi = self.piano_keys.min_midi, self.piano_keys.max_midi
                staged = min(hi, max(lo, current + step))
                self.piano_keys.set_keyboard_cursor(staged)
                # hear the candidate NOW: the pitch is validated by Enter,
                # never guessed from its name alone
                self._audition_pitch(staged)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def set_full_reanalysis(self, active: bool) -> None:
        """Lock edits and exports until a fresh evidence pass resolves."""
        self._full_reanalysis_active = active
        if active:
            self.progress.setValue(0)
            self.label_play_pos.setText(tr("ui full reanalysis running"))
        self._update_edit_controls()

    def finish_full_reanalysis(self) -> None:
        """Restore controls and show complete progress after worker completion."""
        self._full_reanalysis_active = False
        self.progress.setValue(100)
        self.label_play_pos.setText(tr("ui play pos"))
        self._update_edit_controls()

    def set_full_reanalysis_progress(self, fraction: float, message: str) -> None:
        """Show the pipeline's own stage progress during a pYIN rerun."""
        self.progress.setValue(max(0, min(100, int(fraction * 100))))
        self.label_play_pos.setText(message)

    def set_replacement_active(self, active: bool) -> None:
        if active:
            self.tools_toggle.setChecked(True)
        self._replacement_analyzing = False
        self.btn_replace_phrase.setVisible(not active)
        self.btn_finish_replacement.setVisible(active)
        self._update_edit_controls()

    def set_replacement_analyzing(self, analyzing: bool) -> None:
        self._replacement_analyzing = analyzing
        self._update_edit_controls()

    def set_replacement_pending(self, pending: bool) -> None:
        """The spliced take awaits the singer's verdict (keep / cancel)."""
        if pending:
            self.tools_toggle.setChecked(True)
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

        The current state's row (the None reference) jumps to nothing; a row
        whose action is disabled comes through only when the
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

    def set_archive_comparison_available(self, available: bool) -> None:
        """Show the archive comparison only when this live take has two readings."""
        self._archive_comparison_available = bool(available)
        self.btn_compare_readings.setVisible(
            self._has_live_result and self._archive_comparison_available
        )

    def show_archived_comparison(
        self,
        labels: tuple[str, ...],
        compare: Callable[[int, int], "ReadingComparison"],
    ) -> None:
        """Open a read-only comparison dialog for this take's saved readings."""
        if len(labels) < 2:
            return
        dialog = ArchiveComparisonDialog(labels, compare, self)
        dialog.exec()

    def set_result(self, res: "AnalysisResult", signal: np.ndarray | None = None) -> None:
        """Render a completed result and its page-owned presentation state.

        The window resolves `signal` from the file cache or active live take;
        this page owns its canonical display name and completed progress.
        Missing or empty audio leaves the waveform untouched.
        """
        self.set_source_name(tr("ui live file") if res.source == "live" else Path(res.source).name)
        self._summary_facts = (
            len(res.notes),
            res.duration_s,
            res.tempo.bpm if res.tempo is not None else None,
            res.key.best.label if res.key is not None else None,
        )
        self._render_summary()
        self.set_progress(100)
        self.details_toggle.setChecked(False)
        self._toggle_details(False)
        self._has_live_result = res.source == "live"
        if not self._has_live_result:
            self._archive_comparison_available = False
        self.btn_live_roll.setVisible(self._has_live_result)
        self.btn_compare_readings.setVisible(
            self._has_live_result and self._archive_comparison_available
        )
        self.chk_live_comparison.setVisible(self._has_live_result)
        self.chk_live_comparison.setChecked(self._has_live_result)
        was_replacing = self.btn_finish_replacement.isVisible()
        self.results.set_results(res)
        self.results.set_chord_progression(res)
        self._notes = list(res.notes)
        self._guide_stage = "listen"
        self._guide_listening = False
        self._guide_adjusting = False
        self._guide_has_audio = signal is not None and bool(len(signal))
        if not self._guide_has_audio or not self._notes:
            self._guide_stage = "export"
        self._guide_targets = []
        self._guide_index = 0
        if res.observation is not None:
            suspects = coherence_report(res.observation, res.notes)["suspects"]
            for suspect in sorted(suspects, key=lambda item: item["start_s"]):
                start = suspect["start_s"]
                target = [
                    i
                    for i, note in enumerate(res.notes)
                    if note.start < start + suspect["duration_s"] and note.end > start
                ]
                if target and target not in self._guide_targets:
                    self._guide_targets.append(target)
        self._guide_target = self._guide_targets[0] if self._guide_targets else []
        self.tools_toggle.setChecked(False)
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
            grid=(
                res.grid
                if res.grid is not None
                else BeatGrid(res.tempo.bpm)
                if res.tempo is not None
                else None
            ),
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

    def show_note_audition(self, index: int, midi: int, count: int) -> None:
        """Identify the note sounding from Space and expose its stop action."""
        self._note_audition = (index, midi, count)
        self._render_note_audition()
        self._render_guidance()

    def stop_note_audition(self) -> None:
        """Hide the note-specific readout when its audition ends."""
        self._note_audition = None
        self._render_note_audition()
        self._render_guidance()

    def _render_note_audition(self) -> None:
        """Render the current note-audition fact in the active language.

        The stop action sits at its permanent place in the page's ONE
        focus order; Qt skips it while hidden, so showing or hiding the
        readout never relinks the chain.
        """
        current = self._note_audition
        self.label_note_audition.setVisible(current is not None)
        self.btn_stop_audition.setVisible(current is not None)
        if current is not None:
            index, midi, count = current
            self.label_note_audition.setText(
                tr("ui piano audition label").format(
                    note=note_display_pair(midi), i=index + 1, n=count
                )
            )

    def sync_play_buttons(self, mode: str | None) -> None:
        """The three buttons reflect the sounding mode (None = stopped)."""
        self._play_mode = mode
        if self._guide_listening and mode != "voice":
            self._guide_listening = False
            self._guide_stage = "review"
            self._render_guidance()
        self._render_play_buttons()

    def _render_play_buttons(self) -> None:
        """ONE writer: the square marks the sounding mode, the words name it.

        The prefix is state, the label is language: rendering both here
        means a retranslate during playback cannot put an arrow back on the
        button that is currently playing.
        """
        for button, mode, key in (
            (self.btn_play, "piano", "ui play piano"),
            (self.btn_voice, "voice", "ui play voice"),
            (self.btn_both, "both", "ui play both"),
        ):
            prefix = "■" if self._play_mode == mode else "▶"
            button.setText(f"{prefix} {tr(key)}")
        self.btn_play.setToolTip(tr("ui play tooltip"))
        self.btn_voice.setToolTip(tr("ui play voice tooltip"))
        self.btn_both.setToolTip(tr("ui play both tooltip"))

    def reset_play_buttons(self) -> None:
        """Stopped state: every button up, every key released.

        The note-audition readout is NOT this method's to clear — the
        controller's transitions are the ONE path that resolves it (the
        window applies them).
        """
        self.sync_play_buttons(None)
        self.piano_keys.set_playhead(-1.0)  # all keys up
