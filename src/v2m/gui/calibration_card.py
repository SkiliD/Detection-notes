"""The home page's calibration card: five guided rounds, one verdict.

The window feeds each round's heard pitch (from the capture the CARD
asks it to open), the card names the target then renders the verdict.
The offset never feeds the recognition path (ADR 0032) — file + display
only.
"""

from __future__ import annotations

import math
from typing import Callable

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import mic_cal
from ..core.i18n import note_display, tr
from ..core.music import hz_to_midi
from .widgets import (
    BG_SUNKEN,
    BORDER,
    TEXT,
    TEXT_DIM,
    action_button_style,
)


class CalibrationCard(QWidget):
    """The flow's presentation; the window owns the capture."""

    def __init__(self, page, on_close: Callable[[], None] | None = None) -> None:
        super().__init__()
        self._page = page
        # the card's own close action must reach the window: the capture it
        # asked for is the window's to stop (a hidden card with a live mic
        # is a leak, not a state)
        self._on_close = on_close or (lambda: None)
        self._heard: list[float] = []
        self._targets: list[float] = []
        self._round = 0
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        self.setStyleSheet(
            f"background: {BG_SUNKEN}; border: 1px solid {BORDER}; border-radius: 8px;"
        )
        self.prompt = QLabel("")
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet(f"color: {TEXT}; font-size: 10pt;")
        root.addWidget(self.prompt)
        self.progress = QLabel("")
        self.progress.setStyleSheet(f"color: {TEXT_DIM};")
        root.addWidget(self.progress)
        # what the mic reads RIGHT NOW: the one proof the flow is hearing
        # something. Without it a dead mic and a wrong note look identical
        # (a frozen prompt) and the user concludes the mic is not detected.
        self.heard_line = QLabel("")
        self.heard_line.setStyleSheet(f"color: {TEXT_DIM};")
        self.heard_line.setVisible(False)
        root.addWidget(self.heard_line)
        # a driver effects chain filters a SHARED capture before the app hears
        # it, so the offset measured here would carry someone else's processing
        self._driver_effects = False
        self.label_driver_effects = QLabel("")
        self.label_driver_effects.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        self.label_driver_effects.setWordWrap(True)
        self.label_driver_effects.setVisible(False)
        root.addWidget(self.label_driver_effects)
        self.error_line = QLabel("")
        self.error_line.setWordWrap(True)
        self.error_line.setStyleSheet(f"color: {TEXT};")
        self.error_line.setVisible(False)
        root.addWidget(self.error_line)
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet(f"color: {TEXT}; font-size: 10pt;")
        self.verdict.setVisible(False)
        root.addWidget(self.verdict)
        actions = QHBoxLayout()
        self.btn_stop = QPushButton("")
        self.btn_stop.setStyleSheet(action_button_style(primary=False))
        self.btn_stop.clicked.connect(self.reset)
        actions.addWidget(self.btn_stop)
        self.btn_save = QPushButton("")
        self.btn_save.setStyleSheet(action_button_style(primary=True))
        self.btn_save.clicked.connect(self._save)
        self.btn_save.setVisible(False)
        actions.addWidget(self.btn_save)
        self.btn_forget = QPushButton("")
        self.btn_forget.setStyleSheet(action_button_style(primary=False))
        self.btn_forget.clicked.connect(self._forget)
        actions.addWidget(self.btn_forget)
        actions.addStretch(1)
        root.addLayout(actions)

    # the flow's states
    def start(self) -> None:
        """A fresh run: rounds cleared, first target named."""
        self._heard = []
        self._targets = []
        self._round = 0
        self.verdict.setVisible(False)
        self.btn_save.setVisible(False)
        self.error_line.setVisible(False)
        self.heard_line.setVisible(False)
        self._driver_effects = False  # a new flow opens its own capture
        self._render_driver_effects()
        self._ask_round()

    def _ask_round(self) -> None:
        """Arm the round: its target is appended, then NAMED to the singer."""
        self._targets.append(self._target_of(self._round))
        self._render_round()

    def _render_round(self) -> None:
        """The current round's prompt (re-rendered on a language switch).

        The target is READ OUT: the rounds rise by semitones (C, C#, D,
        D#, E) and a singer left guessing them holds the same note for
        five rounds — refused by the ±TARGET_ST gate every time, with no
        difference on screen from a microphone that hears nothing.
        """
        target = self._targets[-1] if self._targets else self._target_of(self._round)
        self.prompt.setText(
            tr("ui calibrate target").format(
                n=self._round + 1,
                total=mic_cal.ROUNDS,
                note=note_display(int(round(hz_to_midi(target)))),
            )
        )
        self.progress.setText(self._age_line())

    def set_heard(self, hz: float | None) -> None:
        """What the capture reads NOW (None = nothing above the gate)."""
        if hz is None or not math.isfinite(hz) or hz <= 0:
            self.heard_line.setText(tr("ui calibrate silent"))
        else:
            self.heard_line.setText(
                tr("ui calibrate heard").format(note=note_display(int(round(hz_to_midi(hz)))))
            )
        self.heard_line.setVisible(True)

    def set_driver_effects(self, flagged: bool) -> None:
        """Say whether the driver may have processed this flow's capture."""
        self._driver_effects = flagged
        self._render_driver_effects()

    def _render_driver_effects(self) -> None:
        self.label_driver_effects.setText(
            tr("ui input device effects") if self._driver_effects else ""
        )
        self.label_driver_effects.setVisible(self._driver_effects)

    def set_error(self, message: str) -> None:
        """A capture that never opened must be SAID (never a frozen prompt)."""
        self.heard_line.setVisible(False)
        self.error_line.setText(message)
        self.error_line.setVisible(True)

    def _age_line(self) -> str:
        saved = mic_cal.read()
        if saved is None:
            return ""
        import time

        age_s = time.time() - float(saved["date_s"])
        return tr("ui calibrate saved").format(cents=saved["offset_cents"], age_s=int(age_s))

    def feed(self, hz: float | None) -> bool:
        """One heard round; False while the held pitch is not the target.

        Without this gate, the previous round's held note was graded
        against the NEXT round's target (read -122 cents, measured): the
        round waits until the singer reaches the asked note.
        """
        if self._round >= mic_cal.ROUNDS:
            return True
        if hz is None or not math.isfinite(hz) or hz <= 0:
            return False
        target = self._targets[self._round]
        if abs(1200.0 * math.log2(hz / target)) > mic_cal.TARGET_ST * 100.0:
            return False  # not the asked note: the round waits
        self._heard.append(hz)
        self._round += 1
        if self._round >= mic_cal.ROUNDS:
            self._finish()
        else:
            self._ask_round()
        return True

    def _finish(self) -> None:
        from ..core.mic_cal import collect

        # five consecutive semitones from middle C
        verdict = collect(self._heard, self._targets)
        self._verdict = verdict
        if verdict.recordable:
            self.prompt.setText(tr("ui calibrate found").format(cents=verdict.offset_cents))
            self.btn_save.setVisible(True)
        else:
            self.prompt.setText(tr("ui calibrate none"))
            self.btn_save.setVisible(False)
        self.progress.setText(
            tr("ui calibrate counts").format(used=verdict.n_used, skipped=verdict.n_skipped)
        )
        self.verdict.setVisible(True)
        self.verdict.setText(tr("ui calibrate context"))

    def _save(self) -> None:
        verdict = getattr(self, "_verdict", None)
        if verdict is not None and mic_cal.write(verdict, self._targets):
            self.progress.setText(self._age_line())
            self.btn_save.setVisible(False)
            self.prompt.setText(tr("ui calibrate saved now"))

    def _forget(self) -> None:
        mic_cal.clear()
        self.progress.setText("")
        self.heard_line.setVisible(False)
        self.error_line.setVisible(False)
        self.prompt.setText(tr("ui calibrate forgotten"))
        self.verdict.setVisible(False)

    def reset(self) -> None:
        """Close the card; the next open starts fresh."""
        self.setVisible(False)
        self.heard_line.setVisible(False)
        self.error_line.setVisible(False)
        self._on_close()

    def retranslate(self) -> None:
        self.btn_stop.setText(tr("ui calibrate stop"))
        self.btn_save.setText(tr("ui calibrate save"))
        self.btn_forget.setText(tr("ui calibrate forget"))
        self._render_driver_effects()  # the sentence follows the language
        if self._targets:
            self._render_round()  # the asked note follows the language

    @staticmethod
    def _target_of(round_index: int) -> float:
        """The round's target: middle C rising by semitones."""
        return 440.0 * (2.0 ** ((round_index - 9) / 12.0)) if round_index >= 0 else 440.0
