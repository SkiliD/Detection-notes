"""The words the desktop PAINTS, which no stylesheet reaches.

Every text pair the Qt stylesheet carries is held at 4.5:1 by
`test_gui_stylesheet`; the ones drawn with `QPainter` are invisible to it — a
piano key's name, the intonation chart's note names under each point, the
staff's header, measure numbers, rests and breath marks. pyqtgraph is the
sharpest case: its tick strings and axis titles come from the `foreground`
config option, whose default ('d') is a FIXED grey (#969696) whatever the
palette — measured 2.62:1 on the light page and 2.41:1 on its sunken surface,
the same blind spot the web roll's canvas words had.

So each painted site is listed here with the ink and the surface it sits on,
both palettes are measured, and the plot widgets are asked for the pen they
actually paint with — a view that forgets is a blind spot coming back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from v2m.gui import staff, widgets

GUI = Path(widgets.__file__).parent


def qapp():
    """One QApplication for the painted-widget checks (offscreen in the suite)."""
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _luminance(value: str) -> float:
    channels = [int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    first, second = _luminance(foreground), _luminance(background)
    high, low = max(first, second), min(first, second)
    return (high + 0.05) / (low + 0.05)


def _ink(color) -> str:
    """`#rrggbb` for a token string or a QColor (the staff keeps QColors)."""
    return color.name() if hasattr(color, "name") else color


def _painted_pairs() -> dict[str, tuple[str, str]]:
    """Every hand-painted word, with the ink and the surface it is drawn on."""
    return {
        "a piano key's name": (widgets.KEY_LABEL, widgets.KEY_WHITE),
        "the intonation chart's note names": (widgets.TEXT_DIM, widgets.BG_CARD),
        "the pitch rows and seconds of a roll": (widgets.TEXT_DIM, widgets.BG_APP),
        "the waveform's seconds": (widgets.TEXT_DIM, widgets.BG_APP),
        "the coherence chart's axis numbers": (widgets.TEXT_DIM, widgets.BG_SUNKEN),
        "the staff's header, measure numbers and key": (staff.SOFT_INK, staff.BG),
        "the staff's rests": (staff.REST_INK, staff.BG),
        'the staff\'s breath "B"': (staff.BREATH_INK, staff.BG),
        "the staff's note being sung": (staff.PREVIEW_INK, staff.BG),
    }


@pytest.mark.parametrize("theme", widgets.THEMES)
def test_every_painted_word_clears_the_page_s_ratio(theme):
    widgets.set_theme(theme)
    app = qapp()
    app.setStyleSheet(widgets.theme_stylesheet())
    weak = [
        f"{site}: {_ink(ink)} on {_ink(surface)} = {_contrast(_ink(ink), _ink(surface)):.2f}:1"
        for site, (ink, surface) in _painted_pairs().items()
        if _contrast(_ink(ink), _ink(surface)) < 4.5
    ]
    assert not weak, f"{theme} paints words under 4.5:1:\n  " + "\n  ".join(weak)


@pytest.mark.parametrize("theme", widgets.THEMES)
def test_the_plot_widgets_paint_with_the_theme_ink(theme):
    # The pen is the fact, not the token: pyqtgraph's config default is what
    # actually reaches the painter, and it is a fixed grey.
    widgets.set_theme(theme)
    app = qapp()
    app.setStyleSheet(widgets.theme_stylesheet())
    roll = widgets.PianoRoll()
    waveform = widgets.WaveformView()
    try:
        for widget in (roll, waveform):
            for axis_name in ("left", "bottom"):
                axis = widget.getPlotItem().getAxis(axis_name)
                painted = axis.textPen().color().name().lower()
                assert painted == widgets.TEXT_DIM.lower(), (
                    f"{type(widget).__name__}.{axis_name} paints its numbers "
                    f"{painted}, not the theme's {widgets.TEXT_DIM}"
                )
    finally:
        roll.deleteLater()
        waveform.deleteLater()


def test_no_plot_widget_falls_back_to_pyqtgraphs_grey():
    # The next plot widget must not re-introduce the blind spot: every
    # `PlotWidget` subclass in the shared module inks its own words, and the
    # page that builds a plot inline sets the pens itself.
    source = (GUI / "widgets.py").read_text(encoding="utf-8")
    plots = re.findall(r"class (\w+)\(pg\.PlotWidget\):(.*?)(?=\nclass |\Z)", source, re.S)
    assert plots, "the plot widgets moved: this guard has nothing to read"
    for name, body in plots:
        assert "ink_plot(" in body or "setTextPen" in body, (
            f"{name} paints pyqtgraph's default grey whatever the palette"
        )
    home = (GUI / "home_page.py").read_text(encoding="utf-8")
    assert "setTextPen" in home, "the coherence chart lost its themed axis numbers"
