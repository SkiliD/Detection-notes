"""Segmentation temps réel d'une courbe de hauteur (hystérésis on/off)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .music import hz_to_midi, midi_to_name


@dataclass
class LiveNote:
    start: float
    end: float
    midi: int
    confidence: float = 0.9
    figure: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class LiveNoteSegmenter:
    """Machine à états : IDLE -> PENDING -> ACTIVE.

    - Une note n'est validée ("on") qu'après `confirm_s` de hauteur stable.
    - Elle se termine ("off") après `release_s` sans voix ou changement de
      hauteur. Cela évite le clignotement de notes parasites.
    """

    def __init__(
        self,
        confirm_s: float = 0.08,
        release_s: float = 0.12,
        min_note_s: float = 0.10,
        semitones: float = 0.6,
    ):
        self.confirm_s = confirm_s
        self.release_s = release_s
        self.min_note_s = min_note_s
        self.semitones = semitones

        self.state = "IDLE"
        self.pending_start: float | None = None
        self.pending_midis: list[float] = []
        self.note_start: float | None = None
        self.note_midis: list[float] = []
        self.last_voiced_t: float | None = None
        self.last_hz: float = float("nan")
        self.notes: list[LiveNote] = []

    # ------------------------------------------------------------------
    def push(self, t: float, hz: float) -> str:
        """Avance d'une frame. Renvoie 'on', 'off' ou ''."""
        ev = ""
        voiced = hz == hz and hz > 0  # pas NaN
        if voiced:
            self.last_hz = hz
            m = float(hz_to_midi(hz))

            if self.state == "IDLE":
                self._start_pending(t, m)
            elif self.state == "PENDING":
                ref = float(np.median(self.pending_midis))
                if abs(m - ref) <= self.semitones:
                    self.pending_midis.append(m)
                else:
                    self._start_pending(t, m)
                if t - self.pending_start >= self.confirm_s:
                    self.state = "ACTIVE"
                    self.note_start = self.pending_start
                    self.note_midis = list(self.pending_midis)
                    ev = "on"
            elif self.state == "ACTIVE":
                ref = float(np.median(self.note_midis))
                if abs(m - ref) <= self.semitones:
                    self.note_midis.append(m)
                else:
                    self._close_note(t)
                    ev = "off"
                    self._start_pending(t, m)
            self.last_voiced_t = t
        else:
            if self.state == "ACTIVE" and self.last_voiced_t is not None:
                if t - self.last_voiced_t >= self.release_s:
                    self._close_note(self.last_voiced_t)
                    ev = "off"
            elif self.state == "PENDING" and self.last_voiced_t is not None:
                if t - self.last_voiced_t >= self.release_s:
                    self.state = "IDLE"
                    self.pending_start = None
                    self.pending_midis = []
        return ev

    # ------------------------------------------------------------------
    def _start_pending(self, t: float, m: float) -> None:
        self.state = "PENDING"
        self.pending_start = t
        self.pending_midis = [m]

    def _close_note(self, end_t: float) -> None:
        if self.note_start is None or not self.note_midis:
            self.state = "IDLE"
            return
        start = self.note_start
        midi = int(round(float(np.median(self.note_midis))))
        if end_t - start >= self.min_note_s:
            self.notes.append(LiveNote(start=start, end=end_t, midi=midi))
        self.note_start = None
        self.note_midis = []
        self.state = "IDLE"

    # ------------------------------------------------------------------
    def current(self) -> tuple[float, float] | None:
        """(midi flottant, hz) de la note en cours, ou None."""
        if self.state == "ACTIVE" and self.note_midis:
            return float(np.median(self.note_midis)), self.last_hz
        if self.state == "PENDING" and self.pending_midis:
            return float(np.median(self.pending_midis)), self.last_hz
        return None

    def display_text(self) -> str:
        cur = self.current()
        if cur is None or self.last_hz != self.last_hz:
            return "—"
        name = midi_to_name(int(round(cur[0])))
        return f"{name}  {self.last_hz:.1f} Hz"

    def flush(self, t: float) -> None:
        """Force la fermeture de la note en cours + fusion des fragments.

        Fusionne les notes consécutives de même hauteur séparées d'un trou
        court (<= release_s) : un parasite ou une reprise rapide du même son
        ne doit pas scinder la note musicale.
        """
        if self.state == "ACTIVE":
            self._close_note(self.last_voiced_t if self.last_voiced_t is not None else t)
        merged: list[LiveNote] = []
        for n in self.notes:
            if (
                merged
                and n.midi == merged[-1].midi
                and n.start - merged[-1].end <= self.release_s
            ):
                merged[-1].end = n.end
                merged[-1].confidence = max(merged[-1].confidence, n.confidence)
            else:
                merged.append(n)
        self.notes = merged
