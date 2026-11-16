"""The note finder page: "which note is this?" and nothing else.

A Page module, ADR-0010-shaped: the page owns BOTH its widgets and its
behavior, dependencies are injected at construction, and the window
only composes it (navigation + the callables the page cannot own —
where home is, where errors go). Python state that outlives a click
lives HERE, not on the window: `FinderSession` (the readings/history
deep module), the capture thread and the level meter. The window keeps
no finder mirror.

The two contracts this page serves (measured, see FinderSession):
- the BIG name answers from the PENDING plateau too — hiding it read
  as "the mic is ignored";
- the key lights on a CONFIRMED note only — a rumble holds no 90 ms.

Product fixes, all measured:
- the mic level is VISIBLE (`LevelMeter`, the silence gate drawn on the
  bar) — a mic too far away or muted is indistinguishable from a
  silent page without it;
- the RAW instant pitch is shown (`seg.heard()`) when the segmenter
  holds nothing — a glide, a wavering beginner tone or a quiet voice
  still read as "chantez une note…" while YIN saw the pitch fine;
- an honest idle line distinguishes "silence" from "sound but no note
  yet" (the mic is working, the confirmation window has not filled).

Layout decisions, each answering a measured reading of the page:
- the answer is ONE card: the name, the tuning needle and the mic's
  reception belong to the same question, so they share one surface
  instead of reading as three unrelated rows;
- the tuning scale NAMES its direction ("plus grave" / "plus aigu")
  under the needle: a dot left of centre means nothing to a singer who
  does not read cents, while the measured cents stay in the detail line;
- the history is a short row of PLAYABLE chips, not an empty box: each
  confirmed note is one click (or one Tab + Enter) away from being heard;
- the keyboard shows a SPAN CHOSEN EXPLICITLY (a singing range rather
  than the whole piano, where a 88-key row cannot name its keys) and is
  never recentred under the pointer.

The mic is faked in tests at the `sounddevice.InputStream` level (the
`LiveThread` module global), so the thread is injected as a zero-arg
factory and the page stays Qt-and-audio agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.i18n import note_display, note_display_pair, tr
from ..core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI, on_piano
from . import widgets
from .display_stabilizer import DisplayStabilizer
from .finder_session import FinderSession
from .widgets import (
    PianoView,
    TuningMeter,
    action_button_style,
    cents_deviation,
    page_scroll,
    tuning_advice,
)

if TYPE_CHECKING:
    from .main_window import LiveThread

# The invite and the answer do not share a style: a 34 pt accent name is
# an answer, and a first-run page must not shout one it does not have.
LEVEL_GATE_COLOR = "#D8877C"  # the silence gate, drawn where it lives


def finder_note_style() -> str:
    """Style the answer note with the current theme's prominent text color."""
    return f"font-size: 34pt; font-weight: bold; color: {widgets.NOTE_TEXT_COLOR};"


def finder_idle_style() -> str:
    """Keep an idle prompt quiet while respecting the active text palette."""
    return f"font-size: 15pt; font-style: italic; color: {widgets.TEXT_FAINT};"


def level_bar_style() -> str:
    """Frame the mic level meter with the current surface and border colors."""
    return (
        f"background: {widgets.BG_CARD}; border: 1px solid {widgets.BORDER}; "
        "border-radius: 4px;"
    )
# The level bar's full scale (block RMS): a sung voice close to the mic
# lands 0.05..0.3, so 0.20 keeps the useful range in the left two thirds
# and clipping a shout is honest.
LEVEL_FULL_SCALE = 0.20
# The answer card is a reading column, not a page-wide table: bounded so
# the needle keeps a readable scale on a wide window.
ANSWER_CARD_MAX_W = 620
def caption_style() -> str:
    """Apply the active secondary-text color to compact page labels."""
    return f"color: {widgets.TEXT_DIM}; font-size: 9pt;"


def chip_style() -> str:
    """Build playable note chips using the application's current palette."""
    return (
        "QPushButton {"
        f" background: {widgets.BG_CARD}; border: 1px solid {widgets.BORDER}; "
        f"border-radius: 11px; padding: 3px 12px; font-weight: bold; color: {widgets.TEXT};"
        "}"
        f"QPushButton:hover, QPushButton:focus {{ border-color: {widgets.ACCENT}; }}"
    )
# The keyboard's spans, in the order the selector lists them: a singer
# reads a voice range (three octaves around the sung register), a low
# voice one octave down, and the whole piano only when asked for.
KEYBOARD_RANGES: tuple[tuple[str, int, int], ...] = (
    ("ui finder range voice", 48, 84),  # C3–C6
    ("ui finder range low", 36, 72),  # C2–C5
    ("ui finder range all", PIANO_MIN_MIDI, PIANO_MAX_MIDI),
)
_UNSET = object()


class LevelMeter(QWidget):
    """The mic level as one thin bar, the silence gate drawn on it.

    The bar makes the LEVEL visible and the gate mark makes the SILENCE
    DECISION visible: a fill that cannot reach the red mark is the mic
    saying "trop faible", not "rien entendu". No decay, no smoothing:
    the raw block RMS, redrawn per event (a width change on a QLabel is
    cheap at 20 Hz).

    The other end is just as decision-bearing: a block whose RMS reaches
    the rail (CLIP_RMS_RMS >= 0.95) turns the fill to the gate's red —
    "the input is clipping, turn the input DOWN". The 0.95 threshold is
    an ABSOLUTE: the WAV is PCM_16, the rail is physical, not a
    relative heuristic.
    """

    CLIP_RMS_RMS = 0.95

    def __init__(self) -> None:
        super().__init__()
        self._fill = QLabel(self)
        self._fill.setStyleSheet(f"background: {widgets.NOTE_TEXT_COLOR}; border-radius: 3px;")
        self._fill.hide()
        self._gate = QLabel(self)
        self._gate.setFixedWidth(2)
        self._gate.setStyleSheet(f"background: {LEVEL_GATE_COLOR};")
        self._gate_share = 0.04
        self._clipping = False
        self.setFixedHeight(8)
        self.setStyleSheet(level_bar_style())

    def set_gate_rms(self, gate_rms: float, full_scale: float = LEVEL_FULL_SCALE) -> None:
        """Where the silence gate sits on the bar (fraction of full scale)."""
        self._gate_share = max(0.02, min(1.0, gate_rms / full_scale))
        self._place()

    def set_level(self, rms: float) -> None:
        share = max(0.0, min(1.0, rms / LEVEL_FULL_SCALE))
        if share <= 0.001:
            self._fill.hide()
            self._clipping = False
            return
        clipping = rms >= self.CLIP_RMS_RMS
        if clipping != self._clipping:
            # the gate's red: one red per bar, already a CSS string — the
            # clipping fill is the same "turn it down" family of signal
            self._fill.setStyleSheet(
                f"background: {LEVEL_GATE_COLOR}; border-radius: 3px;"
                if clipping
                else f"background: {widgets.NOTE_TEXT_COLOR}; border-radius: 3px;"
            )
        self._clipping = clipping
        self._fill.show()
        self._fill.setFixedWidth(max(2, int(round(share * self.width()))))
        self._fill.setFixedHeight(self.height() - 2)
        self._fill.move(1, 1)
        self._fill.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._place()

    def _place(self) -> None:
        self._gate.setFixedHeight(self.height())
        self._gate.move(int(round(self._gate_share * self.width())), 0)
        self._gate.raise_()


class FinderPage(QWidget):
    """Build, run and render the finder — the page IS the feature."""

    def __init__(
        self,
        thread_factory: Callable[[], "LiveThread"],
        go_home: Callable[[], None],
        show_error: Callable[[str], None],
        clear_error: Callable[[], None],
        audition_note: Callable[[int], None],
    ) -> None:
        super().__init__()
        self._thread_factory = thread_factory
        self._go_home = go_home
        self._show_error = show_error
        self._clear_error = clear_error
        self._audition_note = audition_note
        self.session = FinderSession()
        # the name channel is vibration-tolerant (a ±0.5-st vibrato must
        # not flicker the big name): DisplayStabilizer decides the name
        self._stabilizer = DisplayStabilizer()
        self._thread: "LiveThread | None" = None
        self._last_rms = 0.0
        self._note_heard = False
        self._gate_rms: float = 0.008  # refreshed from the capture at open
        self._tuning_cents: float | None = None
        self._input_device_name = ""
        self._input_device_fell_back = False
        # what the chips currently show: rebuilding them on every audio
        # block at 20 Hz would churn the widgets and steal their focus
        self._history_shown: tuple[int, ...] = ()
        self._chips: list[QPushButton] = []
        self._build()

    # ------------------------------------------------------------------
    # widgets
    # ------------------------------------------------------------------
    def _build(self) -> None:
        # the page's whole content sits in ONE vertical scroll surface: the
        # big name, the history and the piano keys all stay reachable in a
        # SHORT window instead of being clipped
        content = QWidget()
        root = QVBoxLayout(content)
        top = QHBoxLayout()
        self.btn_back = QPushButton("")
        self.btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_back.clicked.connect(self.leave)
        top.addWidget(self.btn_back)
        self.btn_toggle = QPushButton("")
        self.btn_toggle.setMinimumHeight(38)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.clicked.connect(self.toggle)
        top.addWidget(self.btn_toggle)
        self.label_state = QLabel("")
        self.label_state.setStyleSheet(caption_style())
        top.addWidget(self.label_state)
        top.addStretch(1)
        self.label_input_device = QLabel("")
        self.label_input_device.setStyleSheet(caption_style())
        top.addWidget(self.label_input_device)
        # the endpoint's driver chain is named for THIS capture, never for the
        # next one: a sentence about processing belongs to a stream that opened
        self._driver_effects = False
        self.label_driver_effects = QLabel("")
        self.label_driver_effects.setStyleSheet(caption_style())
        self.label_driver_effects.setWordWrap(True)
        self.label_driver_effects.setVisible(False)
        top.addWidget(self.label_driver_effects)
        root.addLayout(top)

        # The answer: ONE card holding the name, the tuning needle with its
        # named directions, and the mic's reception. They answer the same
        # question ("what does the mic hear right now?"), so sharing a
        # surface keeps them read as one composition.
        self.box_answer = QGroupBox("")
        answer = QVBoxLayout(self.box_answer)
        self.label_note = QLabel("")
        self.label_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_note.setMinimumHeight(66)
        answer.addWidget(self.label_note)

        self.tuning = TuningMeter()
        self.tuning.setMaximumWidth(360)
        answer.addWidget(self.tuning, alignment=Qt.AlignmentFlag.AlignHCenter)
        # The needle's SCALE is named: a dot left of centre is "plus grave",
        # right is "plus aigu" — the cents stay in the detail line below.
        # The captions share the needle's width so an end name sits under its
        # own end of the track.
        scale_host = QWidget()
        scale_host.setMaximumWidth(360)
        scale = QHBoxLayout(scale_host)
        scale.setContentsMargins(0, 0, 0, 0)
        self.label_tuning_low = QLabel("")
        self.label_tuning_low.setStyleSheet(caption_style())
        scale.addWidget(self.label_tuning_low)
        scale.addStretch(1)
        self.label_tuning_centre = QLabel("")
        self.label_tuning_centre.setStyleSheet(caption_style())
        scale.addWidget(self.label_tuning_centre)
        scale.addStretch(1)
        self.label_tuning_high = QLabel("")
        self.label_tuning_high.setStyleSheet(caption_style())
        scale.addWidget(self.label_tuning_high)
        answer.addWidget(scale_host, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.label_tuning = QLabel("")
        self.label_tuning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        answer.addWidget(self.label_tuning)

        # the mic level, gate drawn on it: what the mic hears, at a glance
        self.level = LevelMeter()
        answer.addWidget(self.level)
        # the diagnostic line: silence vs sound-without-note vs … (idle)
        self.label_heard = QLabel("")
        self.label_heard.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_heard.setStyleSheet(caption_style())
        answer.addWidget(self.label_heard)
        self.box_answer.setMaximumWidth(ANSWER_CARD_MAX_W)
        card_row = QHBoxLayout()
        card_row.addStretch(1)
        card_row.addWidget(self.box_answer, stretch=4)
        card_row.addStretch(1)
        root.addLayout(card_row)

        self.label_hint = QLabel("")
        self.label_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_hint.setWordWrap(True)
        self.label_hint.setMaximumWidth(560)
        # a bounded reading width, but wide enough to read in two lines
        self.label_hint.setMinimumWidth(440)
        hint_lay = QHBoxLayout()
        hint_lay.addStretch(1)
        hint_lay.addWidget(self.label_hint)
        hint_lay.addStretch(1)
        root.addLayout(hint_lay)

        # The history: a short row of PLAYABLE chips rather than an empty
        # box — each entry is the note itself, and one click hears it again.
        self.label_history = QLabel("")
        self.label_history.setStyleSheet(f"color: {widgets.TEXT_DIM}; font-weight: bold;")
        root.addWidget(self.label_history)
        self.history_row = QWidget()
        self._history_layout = QHBoxLayout(self.history_row)
        self._history_layout.setContentsMargins(0, 0, 0, 0)
        self._history_layout.setSpacing(6)
        self._history_layout.addStretch(1)
        self._history_layout.addStretch(1)
        root.addWidget(self.history_row)
        self.label_history_empty = QLabel("")
        self.label_history_empty.setStyleSheet(caption_style())
        self.label_history_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.label_history_empty)
        root.addStretch(1)

        # The keyboard's span is CHOSEN, never guessed: a singing range names
        # its keys, an explicit choice widens it to the whole piano.
        keys_row = QHBoxLayout()
        self.label_range = QLabel("")
        self.label_range.setStyleSheet(caption_style())
        keys_row.addWidget(self.label_range)
        self.combo_range = QComboBox()
        self.combo_range.currentIndexChanged.connect(self._on_range_changed)
        keys_row.addWidget(self.combo_range)
        keys_row.addStretch(1)
        root.addLayout(keys_row)

        start = KEYBOARD_RANGES[0]
        self.keys = PianoView(min_midi=start[1], max_midi=start[2])
        self.keys.setMinimumHeight(150)
        # a keyboard, not a floor: stretched to the page it drew key faces a
        # tenth of the window tall
        self.keys.setMaximumHeight(190)
        # the piano is this page's audition action FROM THE KEYBOARD: Tab
        # reaches it, an arrow stages a key, Enter plays it (a plain
        # QWidget is no Tab stop, so the policy is set here)
        self.keys.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.keys.key_clicked.connect(self._audition_note)
        root.addWidget(self.keys, stretch=2)

        # Keyboard traversal in task order: leave, run the listening, read
        # the history chips, then audition a key on the piano. The chips are
        # rebuilt with the readings, so `_link_history_tab_order` keeps this
        # chain true as they appear and disappear.
        self._focus_order = (self.btn_back, self.btn_toggle, self.keys)
        QWidget.setTabOrder(self.btn_back, self.btn_toggle)
        QWidget.setTabOrder(self.btn_toggle, self.keys)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = page_scroll(content)
        outer.addWidget(self.scroll)
        self._refresh_history(force=True)

    # ------------------------------------------------------------------
    # page surface (what the window composes)
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.isRunning()

    @property
    def thread(self) -> "LiveThread | None":
        """The capture thread, for callers that must stop and wait (tests)."""
        return self._thread

    @property
    def history_chips(self) -> tuple[QPushButton, ...]:
        """The history's chips, oldest first — what a Tab walk reaches."""
        return tuple(self._chips)

    def open(self) -> None:
        """Enter the finder; listening starts right away."""
        self._clear_error()
        if not self.running:
            self.toggle()

    def leave(self) -> None:
        """Leave the finder: the capture is dropped, never analysed nor exported."""
        self._stop()
        self._audition_note(0)  # stop the currently auditioned key
        self._go_home()

    def refresh_theme(self) -> None:
        """Reapply local theme-dependent colors and redraw the finder views."""
        self.label_state.setStyleSheet(caption_style())
        self.label_input_device.setStyleSheet(caption_style())
        self.label_tuning_low.setStyleSheet(caption_style())
        self.label_tuning_centre.setStyleSheet(caption_style())
        self.label_tuning_high.setStyleSheet(caption_style())
        self.label_heard.setStyleSheet(caption_style())
        self.label_history.setStyleSheet(f"color: {widgets.TEXT_DIM}; font-weight: bold;")
        self.label_history_empty.setStyleSheet(caption_style())
        self.label_range.setStyleSheet(caption_style())
        self.label_note.setStyleSheet(
            finder_note_style() if self.session.display is not None else finder_idle_style()
        )
        self.label_tuning.setStyleSheet(f"color: {widgets.TEXT}; font-size: 11pt;")
        self.tuning.update()
        self.level.setStyleSheet(level_bar_style())
        self.level._fill.setStyleSheet(
            f"background: {widgets.NOTE_TEXT_COLOR}; border-radius: 3px;"
        )
        self._refresh_history(force=True)
        self.update()

    def retranslate(self) -> None:
        self.btn_back.setText(tr("ui back home"))
        self.set_input_device_status(
            self._input_device_name or tr("ui input device default name"),
            self._input_device_fell_back,
        )
        self.box_answer.setTitle(tr("ui finder answer"))
        self.label_tuning_low.setText(tr("ui tuning lower"))
        self.label_tuning_centre.setText(tr("ui tuning centre"))
        self.label_tuning_high.setText(tr("ui tuning higher"))
        self.label_hint.setText(tr("ui finder hint"))
        self.label_history.setText(tr("ui finder history"))
        self.label_history_empty.setText(tr("ui finder history empty"))
        self.label_range.setText(tr("ui finder range"))
        self.combo_range.setToolTip(tr("ui finder range tooltip"))
        self._render_range_items()
        self._refresh_state()
        self._refresh_input_state()
        self._render_driver_effects()  # the sentence follows the language
        self._render_tuning()  # the verdict follows the language
        self._refresh_history(force=True)  # the chips follow too

    # ------------------------------------------------------------------
    # behavior
    # ------------------------------------------------------------------
    def toggle(self) -> None:
        if self.running:
            self._stop()
            return
        self._clear_error()
        self.set_note(None)
        self.set_tuning(None)
        self.session.reset()
        self._stabilizer.reset()
        self._last_rms = 0.0
        self._note_heard = False
        self.level.set_level(0.0)
        self._refresh_input_state()
        self._refresh_history()
        self._driver_effects = False  # the previous capture's fact
        self._render_driver_effects()
        self._thread = self._thread_factory()
        self._thread.finderFrame.connect(self._on_note)
        self._thread.blockStats.connect(self._on_block)
        self._thread.failed.connect(self._on_failed)
        # the gate mark on the level bar is the REAL silence gate: read
        # from the capture the thread opens (the same number the DSP uses)
        capture = self._thread.capture
        if capture is not None:
            self._gate_rms = capture.gate_rms
            self.level.set_gate_rms(capture.gate_rms)
        self._thread.start()
        self._refresh_state()
        self._refresh_input_state()

    def _stop(self) -> None:
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.stop()
            thread.wait(6000)
        self._thread = None
        self._driver_effects = False
        self._render_driver_effects()
        self._refresh_state()
        self._refresh_input_state()

    def set_input_device_status(self, name: str, fell_back: bool = False) -> None:
        """Name the endpoint selected next or opened by the running capture."""
        self._input_device_name = name
        self._input_device_fell_back = fell_back
        self._render_input_device_status()

    def set_driver_effects(self, flagged: bool) -> None:
        """Say whether the driver may have processed the capture it opened."""
        self._driver_effects = flagged
        self._render_driver_effects()

    def _render_driver_effects(self) -> None:
        self.label_driver_effects.setText(
            tr("ui input device effects") if self._driver_effects else ""
        )
        self.label_driver_effects.setVisible(self._driver_effects)

    def _render_input_device_status(self) -> None:
        if self._input_device_fell_back:
            key = "ui input device fallback" if self.running else "ui input device fallback next"
        else:
            key = "ui input device active" if self.running else "ui input device next"
        self.label_input_device.setText(tr(key).format(device=self._input_device_name))

    def _refresh_state(self) -> None:
        """One writer for the things that must agree: the button (its label,
        and its weight — the inviting action is the filled one), the state
        line, the level meter (gate position), and the note area, which
        falls back on the invite when nothing is being sung."""
        running = self.running
        self.btn_toggle.setText(tr("ui finder stop") if running else tr("ui finder start"))
        self.btn_toggle.setStyleSheet(action_button_style(primary=not running))
        self.label_state.setText(tr("ui finder live") if running else tr("ui finder off"))
        self._render_input_device_status()
        if not running:
            self._last_rms = 0.0
            self.level.set_level(0.0)
            self.label_heard.setText("")
            self.set_note(None)
            self.set_tuning(None)

    def _on_block(self, _t: float, rms: float) -> None:
        """The mic's ear, per block: level bar + the honest idle line."""
        self._last_rms = rms
        if rms < self._gate_rms:
            self._note_heard = False
        self.level.set_level(rms)
        self._refresh_input_state()

    def _refresh_input_state(self) -> None:
        if not self.running:
            self.label_heard.setText("")
        elif self._note_heard:
            self.label_heard.setText(tr("ui finder note heard"))
        elif self._last_rms <= 0.0:
            self.label_heard.setText(tr("ui finder waiting"))
        elif self._last_rms < self._gate_rms:
            self.label_heard.setText(tr("ui finder silence"))
        else:
            self.label_heard.setText(tr("ui finder no note"))

    def _on_note(self, cur, confirmed=_UNSET, heard=_UNSET) -> None:
        """Render one block's readings and retain its confirmed pitch.

        The readings arrive together from the audio thread. Qt signals are
        queued while the GUI is busy, so querying the mutable segmenter here
        can observe a later note (or silence) than the frame being rendered.
        """
        if confirmed is _UNSET or heard is _UNSET:
            # Legacy test doubles and callers that publish only the immediate
            # reading retain the old behavior; production events always carry
            # the block's full snapshot, including explicit None readings.
            seg = self._thread.segmenter if self._thread is not None else None
            if confirmed is _UNSET:
                confirmed = seg.confirmed() if seg is not None else None
            if heard is _UNSET:
                heard = seg.heard() if seg is not None else None
        confirmed_reading = cast(tuple[float, float] | None, confirmed)
        heard_reading = cast(tuple[float, float] | None, heard)
        midi = int(round(confirmed_reading[0])) if confirmed_reading else None
        # the immediate reading for the big name (PENDING answers too)
        forming = cur[0] if cur else None
        forming_midi = int(round(forming)) if forming is not None else None
        display = forming_midi if forming_midi is not None else midi
        heard_midi = int(round(heard_reading[0])) if heard_reading else None
        # heard() only fills the name when the segmenter holds nothing:
        # current() (median, confirmed window) stays the honest answer
        if display is None and heard_midi is not None:
            display = heard_midi
        if display is None and heard_midi is None:
            self.session.note_idle()  # the mic heard nothing on this tick
        self.session.feed(forming_midi, midi)
        self._note_heard = display is not None or heard_midi is not None
        self._refresh_input_state()
        shown = self._stabilizer.feed(self.session.last_reading or 0.0, forming)
        # confirmed presence lights the key even while the name channel
        # holds the previous note (a vibrato excursion near the seam)
        self.set_note(shown if shown is not None else display, light=midi is not None)
        # the tuning follows the same reading: the note's own median (stable)
        # when it has one, the raw instant otherwise, so the verdict never
        # lags the name it sits under
        probe = (
            cur[0] if cur is not None else (heard_reading[0] if heard_reading is not None else None)
        )
        self.set_tuning(cents_deviation(probe) if probe is not None else None)
        self._refresh_history()

    def _on_failed(self, msg: str, _signal: bytes = b"") -> None:
        """The capture failed: stop, announce. The live thread's `failed`
        signal also carries the take's signal (the failure trace reads
        it) — the finder ignores it, it keeps no take."""
        self._stop()
        self._show_error(msg)

    def set_note(self, midi_float, light: bool = True) -> None:
        """Show the note as a name and, when `light`, a lit key.

        `light=False` shows the name alone: the key records only CONFIRMED
        notes — a pending plateau must answer "the mic hears you" without
        convincing the keyboard, and a rumble's fleeting periodic frames
        light nothing at all (the anti-rumble contract).
        """
        midi = None if midi_float is None else int(round(float(midi_float)))
        if midi is None or not on_piano(midi):
            self.label_note.setText(tr("ui finder idle"))
            self.label_note.setStyleSheet(finder_idle_style())
            self.keys.set_pressed()
            return
        self.label_note.setText(note_display_pair(midi))
        self.label_note.setStyleSheet(finder_note_style())
        self.keys.set_pressed({midi} if light else None)

    def set_tuning(self, cents: float | None) -> None:
        """The tuning verdict of the note being sung: needle + named advice.

        `cents` None hides both — nothing is being sung, and the page keeps
        the same right to say nothing the rest of it keeps on a quiet mic.
        """
        self._tuning_cents = cents
        self.tuning.set_cents(cents)
        self._render_tuning()

    def _render_tuning(self) -> None:
        """Render the current tuning fact in the active language."""
        cents = self._tuning_cents
        if cents is None:
            self.label_tuning.setText("")
            return
        self.label_tuning.setStyleSheet(f"color: {widgets.TEXT}; font-size: 11pt;")
        self.label_tuning.setText(f"{cents:+.0f} ¢ — {tuning_advice(cents)}")

    # ------------------------------------------------------------------
    # the keyboard's span
    # ------------------------------------------------------------------
    def _render_range_items(self) -> None:
        """Name the three spans in the active language, keeping the choice.

        The span label carries both key names so the choice reads as a
        range ("Voix (Do3 – Do6)"), not as a size.
        """
        index = max(0, self.combo_range.currentIndex())
        self.combo_range.blockSignals(True)
        self.combo_range.clear()
        for key, low, high in KEYBOARD_RANGES:
            self.combo_range.addItem(tr(key).format(lo=note_display(low), hi=note_display(high)))
        self.combo_range.setCurrentIndex(index)
        self.combo_range.blockSignals(False)

    def _on_range_changed(self, index: int) -> None:
        """Show the chosen span in place: keys keep their light and place."""
        if not 0 <= index < len(KEYBOARD_RANGES):
            return
        _key, low, high = KEYBOARD_RANGES[index]
        self.keys.set_range(low, high)

    # ------------------------------------------------------------------
    # the history chips
    # ------------------------------------------------------------------
    def _refresh_history(self, force: bool = False) -> None:
        """Re-render the history as chips, only when the readings changed.

        The page refreshes on every audio block: rebuilding identical chips
        at 20 Hz would churn widgets and drop the focus of the chip being
        played. `force` is for a language switch, where the labels change
        while the notes do not.
        """
        shown = tuple(self.session.history)
        self.label_history_empty.setVisible(not shown)
        self.history_row.setVisible(bool(shown))
        if shown == self._history_shown and not force:
            return
        self._history_shown = shown
        while self._history_layout.count() > 2:  # the two end stretches stay
            item = self._history_layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                # taking it out of the layout is not enough: an unparented
                # layout item keeps painting at its last geometry (measured:
                # a stale chip drew a 640x480 band over the page until the
                # next event-loop cleanup), so it leaves the widget tree now
                widget.setParent(None)
                widget.deleteLater()
        self._chips = []
        for position, midi in enumerate(shown):
            chip = QPushButton(note_display(midi))
            chip.setMinimumHeight(26)
            chip.setStyleSheet(chip_style())
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setToolTip(tr("ui finder history tooltip").format(note=note_display_pair(midi)))
            chip.clicked.connect(lambda _checked=False, pitch=midi: self._audition_note(pitch))
            self._history_layout.insertWidget(position + 1, chip)
            self._chips.append(chip)
        self._link_history_tab_order()

    def _link_history_tab_order(self) -> None:
        """Keep the Tab chain in task order as chips come and go.

        Qt's default chain follows widget CREATION order, so chips built
        after the piano would be reached after it — the history belongs
        between listening and the keyboard.
        """
        previous: QWidget = self.btn_toggle
        for chip in self._chips:
            QWidget.setTabOrder(previous, chip)
            previous = chip
        QWidget.setTabOrder(previous, self.keys)
