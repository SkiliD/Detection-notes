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

"Il entend mal la voix" product fixes (2026-09-28), all measured:
- the mic level is VISIBLE (`LevelMeter`, the silence gate drawn on the
  bar) — a mic too far away or muted was indistinguishable from a
  silent page;
- the RAW instant pitch is shown (`seg.heard()`) when the segmenter
  holds nothing — a glide, a wavering beginner tone or a quiet voice
  still read as "chantez une note…" while YIN saw the pitch fine;
- an honest idle line distinguishes "silence" from "sound but no note
  yet" (the mic is working, the confirmation window has not filled).

The mic is faked in tests at the `sounddevice.InputStream` level (the
`LiveThread` module global), so the thread is injected as a zero-arg
factory and the page stays Qt-and-audio agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from ..core.i18n import note_display_pair, tr
from ..core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI, on_piano
from .finder_session import FinderSession
from .widgets import NOTE_TEXT_COLOR, PianoView, action_button_style

if TYPE_CHECKING:
    from .main_window import LiveThread

# The invite and the answer do not share a style: a 34 pt accent name is
# an answer, and a first-run page must not shout one it does not have.
FINDER_NOTE_STYLE = f"font-size: 34pt; font-weight: bold; color: {NOTE_TEXT_COLOR};"
FINDER_IDLE_STYLE = "font-size: 15pt; font-style: italic; color: #6a7480;"
# A whisper bar has no business being a full QSS object: two inline styles.
LEVEL_BAR_BG = "background: #20262e; border: 1px solid #2e3742; border-radius: 4px;"
LEVEL_GATE_COLOR = "#c25b4e"  # the silence gate, drawn where it lives
# The level bar's full scale (block RMS): a sung voice close to the mic
# lands 0.05..0.3, so 0.20 keeps the useful range in the left two thirds
# and clipping a shout is honest.
LEVEL_FULL_SCALE = 0.20


class LevelMeter(QWidget):
    """The mic level as one thin bar, the silence gate drawn on it.

    The bar makes the LEVEL visible and the gate mark makes the SILENCE
    DECISION visible: a fill that cannot reach the red mark is the mic
    saying "trop faible", not "rien entendu". No decay, no smoothing:
    the raw block RMS, redrawn per event (a width change on a QLabel is
    cheap at 20 Hz).
    """

    def __init__(self) -> None:
        super().__init__()
        self._fill = QLabel(self)
        self._fill.setStyleSheet(f"background: {NOTE_TEXT_COLOR}; border-radius: 3px;")
        self._fill.hide()
        self._gate = QLabel(self)
        self._gate.setFixedWidth(2)
        self._gate.setStyleSheet(f"background: {LEVEL_GATE_COLOR};")
        self._gate_share = 0.04
        self.setFixedHeight(8)
        self.setStyleSheet(LEVEL_BAR_BG)

    def set_gate_rms(self, gate_rms: float, full_scale: float = LEVEL_FULL_SCALE) -> None:
        """Where the silence gate sits on the bar (fraction of full scale)."""
        self._gate_share = max(0.02, min(1.0, gate_rms / full_scale))
        self._place()

    def set_level(self, rms: float) -> None:
        share = max(0.0, min(1.0, rms / LEVEL_FULL_SCALE))
        if share <= 0.001:
            self._fill.hide()
            return
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
    ) -> None:
        super().__init__()
        self._thread_factory = thread_factory
        self._go_home = go_home
        self._show_error = show_error
        self._clear_error = clear_error
        self.session = FinderSession()
        self._thread: "LiveThread | None" = None
        self._gate_rms: float = 0.008  # refreshed from the capture at open
        self._build()

    # ------------------------------------------------------------------
    # widgets
    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
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
        self.label_state.setStyleSheet("color: #8a97a5; font-size: 9pt;")
        top.addWidget(self.label_state)
        top.addStretch(1)
        root.addLayout(top)

        self.label_note = QLabel("")
        self.label_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_note.setMinimumHeight(66)
        root.addWidget(self.label_note)

        # the mic level, gate drawn on it: what the mic hears, at a glance
        self.level = LevelMeter()
        root.addWidget(self.level)

        # the diagnostic line: silence vs sound-without-note vs … (idle)
        self.label_heard = QLabel("")
        self.label_heard.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_heard.setStyleSheet("color: #8a97a5; font-size: 9pt;")
        root.addWidget(self.label_heard)

        self.label_hint = QLabel("")
        self.label_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_hint.setWordWrap(True)
        self.label_hint.setMaximumWidth(560)
        root.addWidget(self.label_hint)
        hint_lay = QHBoxLayout()
        hint_lay.addStretch(1)
        hint_lay.addWidget(self.label_hint)
        hint_lay.addStretch(1)
        root.addLayout(hint_lay)

        self.label_history = QLabel("")
        self.label_history.setStyleSheet("color: #9fb3c8; font-weight: bold;")
        root.addWidget(self.label_history)
        self.history = QListWidget()
        self.history.setMaximumHeight(112)
        self.history.setStyleSheet(
            "background: #20262e; border: 1px solid #2e3742; border-radius: 6px;"
        )
        root.addWidget(self.history)
        root.addStretch(1)

        self.keys = PianoView(min_midi=PIANO_MIN_MIDI, max_midi=PIANO_MAX_MIDI)
        self.keys.setMinimumHeight(150)
        root.addWidget(self.keys, stretch=2)

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

    def open(self) -> None:
        """Enter the finder; listening starts right away."""
        self._clear_error()
        if not self.running:
            self.toggle()

    def leave(self) -> None:
        """Leave the finder: the capture is dropped, never analysed nor exported."""
        self._stop()
        self._go_home()

    def retranslate(self) -> None:
        self.btn_back.setText(tr("ui back home"))
        self.label_hint.setText(tr("ui finder hint"))
        self.label_history.setText(tr("ui finder history"))
        self._refresh_state()
        self._refresh_history()

    # ------------------------------------------------------------------
    # behavior
    # ------------------------------------------------------------------
    def toggle(self) -> None:
        if self.running:
            self._stop()
            return
        self._clear_error()
        self.set_note(None)
        self.session.reset()
        self._refresh_history()
        self._thread = self._thread_factory()
        self._thread.noteFrame.connect(self._on_note)
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

    def _stop(self) -> None:
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.stop()
            thread.wait(6000)
        self._thread = None
        self._refresh_state()

    def _refresh_state(self) -> None:
        """One writer for the things that must agree: the button (its label,
        and its weight — the inviting action is the filled one), the state
        line, the level meter (gate position), and the note area, which
        falls back on the invite when nothing is being sung."""
        running = self.running
        self.btn_toggle.setText(tr("ui finder stop") if running else tr("ui finder start"))
        self.btn_toggle.setStyleSheet(action_button_style(primary=not running))
        self.label_state.setText(tr("ui finder live") if running else tr("ui finder off"))
        if not running:
            self.level.set_level(0.0)
            self.label_heard.setText("")
            self.set_note(None)

    def _on_block(self, _t: float, rms: float) -> None:
        """The mic's ear, per block: level bar + the honest idle line."""
        self.level.set_level(rms)
        if self.session.last_reading is not None:
            return  # the note area already answers; the line stays out of the way
        if rms < self._gate_rms:
            self.label_heard.setText(tr("ui finder silence"))
        else:
            self.label_heard.setText(tr("ui finder no note"))

    def _on_note(self, cur) -> None:
        """Render the note being sung, and retain only distinct recent pitches.

        It reads `current()` for the immediate display (the PENDING window
        answers — "the mic hears you") and `confirmed()` for the key and
        the history (a rumble holds no 90 ms: it lights nothing, enters
        nothing). When the segmenter holds NOTHING but the mic hears a
        pitch (a glide, a wavering beginner tone), the RAW instant
        reading (`heard()`) still names a note — the page must never look
        deaf while YIN hears fine.
        """
        seg = self._thread.segmenter if self._thread is not None else None
        confirmed = seg.confirmed() if seg is not None else None
        midi = int(round(confirmed[0])) if confirmed else None
        # the immediate reading for the big name (PENDING answers too)
        forming = cur[0] if cur else None
        forming_midi = int(round(forming)) if forming is not None else None
        display = forming_midi if forming_midi is not None else midi
        heard = seg.heard() if seg is not None else None
        heard_midi = int(round(heard[0])) if heard is not None else None
        # heard() only fills the name when the segmenter holds nothing:
        # current() (median, confirmed window) stays the honest answer
        if display is None and heard_midi is not None:
            display = heard_midi
        if display is None and heard is None:
            self.session.note_idle()  # the mic heard nothing on this tick
        self.session.feed(forming_midi, midi)
        self.set_note(display, light=midi is not None)
        self._refresh_history()

    def _on_failed(self, msg: str) -> None:
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
            self.label_note.setStyleSheet(FINDER_IDLE_STYLE)
            self.keys.set_pressed()
            return
        self.label_note.setText(note_display_pair(midi))
        self.label_note.setStyleSheet(FINDER_NOTE_STYLE)
        self.keys.set_pressed({midi} if light else None)

    def _refresh_history(self) -> None:
        self.history.clear()
        if not self.session.history:
            self.history.addItem(QListWidgetItem(tr("ui finder history empty")))
            return
        for midi in reversed(self.session.history):
            self.history.addItem(QListWidgetItem(f"🎹  {note_display_pair(midi)}"))
