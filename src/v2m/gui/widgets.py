"""Widgets graphiques : piano roll, waveform, panneau de résultats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg

from ..core.figures import figures_summary, short_label
from ..core.music import midi_to_name

# Palette de couleurs par confiance (rouge -> vert)
LOW_COLOR = (220, 80, 80)
HIGH_COLOR = (80, 220, 120)
SILENCE_COLOR = (120, 120, 130)


FIGURE_COLORS = {
    "ronde": (240, 240, 160),
    "blanche": (120, 220, 220),
    "noire": (120, 180, 255),
    "croche": (170, 140, 255),
    "double-croche": (255, 140, 200),
    "triolet": (255, 190, 120),
    "pointée": (200, 200, 200),
}
UNCLASSIFIED_COLOR = (120, 120, 130)


def figure_color(figure: str | None) -> tuple:
    """Couleur par figure (fall-back confiance-générique si inconnue)."""
    if not figure:
        return UNCLASSIFIED_COLOR + (200,)
    if figure.startswith("triolet"):
        key = "triolet"
    elif figure.endswith("pointée"):
        key = "pointée"
    else:
        key = figure
    return FIGURE_COLORS.get(key, UNCLASSIFIED_COLOR) + (210,)


class PianoRoll(pg.PlotWidget):
    """Piano roll : rectangles = notes, courbe = hauteur fine.

    Mode « notation » : chaque note est colorée selon sa figure et porte
    une étiquette courte (R, B, N, C, DC, 3N, point = pointée).
    """

    MIDI_MIN, MIDI_MAX = 21, 108

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("k")
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.25)
        self.getPlotItem().setLabel("left", "Note MIDI")
        self.getPlotItem().setLabel("bottom", "Temps", units="s")
        self._bar_items: list = []
        self._labels: list = []
        self._curve = None
        self._notation_mode = False
        self._last_notes = []
        self.setLimits(xMin=0)

    def set_notation_mode(self, enabled: bool) -> None:
        self._notation_mode = enabled
        if self._last_notes:
            self.set_notes(self._last_notes)

    def set_notes(self, notes) -> None:
        self._last_notes = list(notes)
        for it in self._bar_items:
            self.removeItem(it)
        self._bar_items = []
        for it in self._labels:
            self.removeItem(it)
        self._labels = []
        for n in notes:
            color = (
                figure_color(n.figure)
                if self._notation_mode
                else self._color_for_conf(n.confidence)
            )
            item = pg.BarGraphItem(
                x0=[n.start],
                x1=[n.end],
                y0=[n.midi - 0.4],
                y1=[n.midi + 0.4],
                brush=color,
                pen=pg.mkPen(None),
            )
            self.addItem(item)
            self._bar_items.append(item)
            if self._notation_mode and n.figure:
                w = max(n.end - n.start, 1e-3)
                if w >= (max(n.end for n in notes) - min(n.start for n in notes)) * 0.012:
                    lbl = pg.TextItem(short_label(n.figure), color=(240, 240, 240))
                    lbl.setPos((n.start + n.end) / 2, n.midi + 0.55)
                    self.addItem(lbl)
                    self._labels.append(lbl)
        if notes:
            self.setXRange(0, max(n.end for n in notes) * 1.05, padding=0)
            midis = [n.midi for n in notes]
            lo, hi = min(midis) - 2, max(midis) + 2
            self.setYRange(lo, hi, padding=0.05)

    def set_pitch_curve(self, track) -> None:
        if self._curve is not None:
            self.removeItem(self._curve)
            self._curve = None
        if track is None:
            return
        hz = track.hz
        mask = ~np.isnan(hz)
        if not mask.any():
            return
        midi = 69 + 12 * np.log2(hz[mask] / 440.0)
        self._curve = pg.PlotDataItem(
            track.times[mask], midi, pen=pg.mkPen((180, 180, 200, 160), width=1)
        )
        self.addItem(self._curve)

    def _color_for_conf(self, conf: float):
        t = max(0.0, min(1.0, (conf - 0.5) * 2))
        r = int(LOW_COLOR[0] + (HIGH_COLOR[0] - LOW_COLOR[0]) * t)
        g = int(LOW_COLOR[1] + (HIGH_COLOR[1] - LOW_COLOR[1]) * t)
        b = int(LOW_COLOR[2] + (HIGH_COLOR[2] - LOW_COLOR[2]) * t)
        return (r, g, b, 200)


class WaveformView(pg.PlotWidget):
    """Waveform + position courante (lecture/live)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("k")
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.getPlotItem().setLabel("bottom", "Temps", units="s")
        self._curve = None
        self._line = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen((80, 160, 255, 180), width=1)
        )
        self.addItem(self._line)

    def set_signal(self, y: np.ndarray, sr: int) -> None:
        t = np.arange(len(y)) / sr
        # décimation pour l'affichage
        step = max(1, len(y) // 50_000)
        if self._curve is not None:
            self.removeItem(self._curve)
        self._curve = self.plot(t[::step], y[::step], pen=pg.mkPen((140, 200, 255, 200), width=0.8))
        self.setXRange(0, len(y) / sr, padding=0)

    def set_position(self, t: float) -> None:
        self._line.setPos(t)


class ResultsPanel(pg.LayoutWidget):
    """Panneau latéral : tempo, tonalité, stats (labels mis à jour en place)."""

    ROWS = [
        ("source", "Source"),
        ("duree", "Durée"),
        ("tempo", "Tempo"),
        ("key", "Tonalité"),
        ("score", "Score tonalité"),
        ("alts", "Alternatives"),
        ("nn", "Nb de notes"),
        ("figs", "Figures"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        title = pg.QtWidgets.QLabel("<b>Résultats de l'analyse</b>")
        self.addWidget(title, row=0, col=0, colspan=2)
        self._values: dict[str, pg.QtWidgets.QLabel] = {}
        for i, (k, label) in enumerate(self.ROWS, start=1):
            kl = pg.QtWidgets.QLabel(f"<b>{label}</b>")
            vl = pg.QtWidgets.QLabel("—")
            kl.setTextInteractionFlags(pg.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            vl.setTextInteractionFlags(pg.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            vl.setWordWrap(True)
            self.addWidget(kl, row=i, col=0)
            self.addWidget(vl, row=i, col=1)
            self._values[k] = vl

    def set_results(self, res) -> None:
        if res is None:
            for vl in self._values.values():
                vl.setText("—")
            return
        self._values["source"].setText(Path(res.source).name)
        self._values["duree"].setText(f"{res.duration_s:.2f} s")
        self._values["tempo"].setText(f"{res.tempo.bpm:.1f} BPM" if res.tempo else "—")
        self._values["key"].setText(res.key.best.label if res.key else "—")
        self._values["score"].setText(f"{res.key.best.score:.2f}" if res.key else "—")
        alts = (
            ", ".join(a.label for a in res.key.alternatives) if res.key and res.key.alternatives else "—"
        )
        self._values["alts"].setText(alts)
        self._values["nn"].setText(str(len(res.notes)))
        figs = figures_summary(res.notes)
        self._values["figs"].setText(
            ", ".join(f"{k} ×{v}" for k, v in figs.items()) if figs else "—"
        )


class LiveIndicator(pg.QtWidgets.QWidget):
    """Affichage temps réel : note courante, fréquence, niveau RMS."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = pg.QtWidgets.QVBoxLayout(self)
        self.label_note = pg.QtWidgets.QLabel("—")
        self.label_note.setStyleSheet("font-size: 28px; font-weight: bold; color: #6cf;")
        self.label_hz = pg.QtWidgets.QLabel("")
        self.label_rms = pg.QtWidgets.QLabel("niveau : —")
        lay.addWidget(self.label_note)
        lay.addWidget(self.label_hz)
        lay.addWidget(self.label_rms)

    def update_note(self, midi_float: float | None, hz: float) -> None:
        if midi_float is None or hz != hz:
            self.label_note.setText("—")
            self.label_hz.setText("")
            return
        self.label_note.setText(midi_to_name(int(round(midi_float))))
        self.label_hz.setText(f"{hz:.1f} Hz")

    def update_rms(self, rms: float) -> None:
        bars = int(min(1.0, rms * 20) * 20)
        self.label_rms.setText("niveau : " + "█" * bars + "░" * (20 - bars))
