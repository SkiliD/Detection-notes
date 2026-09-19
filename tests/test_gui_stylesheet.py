"""The application stylesheet must be valid QSS and Qt-warning-free.

Two regressions pinned here (both silent on screen, noisy on the console):

1. The accent color was injected with a naive `str.replace("ACCENT", ...)`
   that also rewrote the `#ACCENT` *placeholder*, producing `##4da3d8` —
   invalid QSS. Qt then dropped the WHOLE application stylesheet ("Could
   not parse application stylesheet") and the UI silently fell back to
   system defaults.
2. A `font-size` in **px** makes the resolved font report pointSize() == -1;
   rich-text labels (any `<b>`/`<h1>` QLabel) then hit
   `QFont::setPointSize: Point size <= 0 (-1)`. Font sizes must be in pt.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.gui.widgets import ACCENT, STYLESHEET


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def qt_messages():
    """Capture Qt's native messages while the fixture is active."""
    from PyQt6.QtCore import qInstallMessageHandler

    msgs: list[str] = []
    qInstallMessageHandler(lambda mtype, ctx, message: msgs.append(message))
    yield msgs
    qInstallMessageHandler(None)


def test_stylesheet_parses_without_qt_warnings(qapp, qt_messages):
    """The full stylesheet must parse; no parse error, no font warning."""
    from PyQt6.QtWidgets import QLabel, QMainWindow

    app = qapp
    previous = app.styleSheet()
    app.setStyleSheet(STYLESHEET)
    try:
        win = QMainWindow()
        # a rich-text label: the exact path that hit setPointSize(-1) on px fonts
        rich = QLabel("<h1>🎤 Voix → MIDI</h1><b>gras</b>")
        win.setCentralWidget(rich)
        win.show()
        app.processEvents()
    finally:
        app.setStyleSheet(previous)

    bad = [m for m in qt_messages if "Could not parse" in m]
    font = [m for m in qt_messages if "setPointSize" in m]
    assert not bad, f"invalid QSS: {bad}"
    assert not font, f"font warnings: {font}"


def test_no_double_hash_and_no_placeholder_left(qapp):
    """`##` can only come from the accent substitution going wrong."""
    assert "##" not in STYLESHEET
    assert "@ACCENT@" not in STYLESHEET  # placeholder fully substituted
    assert ACCENT in STYLESHEET          # accent color actually injected


def test_font_sizes_are_in_points_not_pixels():
    """A px-sized QSS font reports pointSize() == -1 → setPointSize warning.

    Scans every GUI source file (app stylesheet + inline setStyleSheet
    strings): one px font-size is enough to bring the warning back.
    """
    import re
    from pathlib import Path

    gui_dir = Path(__file__).resolve().parents[1] / "src" / "v2m" / "gui"
    offenders: list[str] = []
    for py_file in sorted(gui_dir.glob("*.py")):
        for i, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"font-size:\s*\d+px", line):
                offenders.append(f"{py_file.name}:{i}: {line.strip()}")
    assert not offenders, "use pt, not px, for QSS font sizes:\n" + "\n".join(offenders)
