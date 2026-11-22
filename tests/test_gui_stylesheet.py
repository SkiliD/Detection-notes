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
3. A `#` line comment inside the QSS (Qt reads `#` as an id selector) makes
   Qt discard the WHOLE stylesheet silently — no warning, the app renders in
   the system default theme. The sheet must be rendered once and checked
   against the palette (`test_stylesheet_actually_applies`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import pytest

from v2m.gui import widgets
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


def _luminance(value: str) -> float:
    """WCAG relative luminance of one `#rrggbb` colour."""
    from PyQt6.QtGui import QColor

    rgb = QColor(value).getRgbF()[:3]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))


def _contrast(foreground: str, background: str) -> float:
    first, second = _luminance(foreground), _luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


# (foreground token, background token) pairs the shell ACTUALLY paints: the
# stylesheet's labels, buttons and their hover/checked/pressed fills, list and
# table rows (including alternates and selection), headers, tooltip, status bar
# and progress bar — plus the action/link/quiet button styles and the hint
# card. The accent appears both as a foreground (titles, link hover, checked
# link rows) and as a background (the primary plate, which carries the page
# ink).
TEXT_PAIRS = (
    ("text", "bg_app"),
    ("text", "bg_card"),
    ("text", "bg_sunken"),
    ("text", "accent_soft"),
    ("text_dim", "bg_app"),
    ("text_dim", "bg_card"),
    ("text_dim", "bg_sunken"),
    ("text_dim", "accent_soft"),
    ("text_faint", "bg_app"),
    ("text_faint", "bg_card"),
    ("text_faint", "bg_sunken"),
    ("accent", "bg_app"),
    ("accent", "bg_card"),
    ("accent", "accent_soft"),
    ("bg_app", "accent"),
)


def test_both_palettes_keep_every_text_pair_readable():
    """Every text/background pair of BOTH palettes clears 4.5:1.

    Only the night palette was measured, so the light theme's hovered copper
    plate carried its own label at 3.58:1 — the one button the singer is asked
    to press went dim exactly while the pointer was on it. A colour picked by
    eye passes in one palette and fails in the other.
    """
    from v2m.gui import widgets

    dark = widgets.theme_colors("dark")
    assert (dark["bg_app"], dark["bg_card"], dark["text"], dark["accent"]) == (
        "#111827",
        "#202B3A",
        "#F4EFE6",
        "#D9A76C",
    )
    for theme in widgets.THEMES:
        colors = widgets.theme_colors(theme)
        for foreground, background in TEXT_PAIRS:
            ratio = _contrast(colors[foreground], colors[background])
            assert ratio >= 4.5, f"{theme}: {foreground} on {background} reads {ratio:.2f}:1"
        for state, plate in widgets.PLATE_STATES[theme].items():
            ratio = _contrast(colors["bg_app"], plate)
            assert ratio >= 4.5, f"{theme}: the primary plate's {state} reads {ratio:.2f}:1"
    # the error banner keeps its own two colours in both palettes
    assert _contrast(widgets.ERROR_TEXT, widgets.ERROR_BG) >= 4.5


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_theme_palettes_rebuild_valid_stylesheets_and_staff_colors(theme, qapp):
    from v2m.gui import staff

    original = widgets.current_theme()
    try:
        widgets.set_theme(theme)
        colors = widgets.theme_colors(theme)
        assert widgets.current_theme() == theme
        assert widgets.BG_APP == colors["bg_app"]
        assert widgets.TEXT == colors["text"]
        assert widgets.ACCENT == colors["accent"]
        assert colors["bg_app"] in widgets.theme_stylesheet()
        assert widgets.theme_stylesheet() != ""
        assert widgets.theme_stylesheet() == widgets._stylesheet()
        assert staff.BG.name().lower() == colors["bg_app"].lower()
        assert staff.INK.name().lower() == colors["text"].lower()
        app = qapp
        previous = app.styleSheet()
        app.setStyleSheet(widgets.theme_stylesheet())
        app.processEvents()
        assert app.styleSheet() == widgets.theme_stylesheet()
        app.setStyleSheet(previous)
    finally:
        widgets.set_theme(original)


def test_view_tabs_follow_the_active_theme():
    """A palette switch must reach styles built from shared tokens.

    Copying a token with `from .widgets import ACCENT` freezes the value
    the module was imported with, so the chosen surface kept the previous
    theme's underline until a restart.
    """
    from v2m.gui import results_page

    original = widgets.current_theme()
    try:
        widgets.set_theme("light")
        light = results_page._view_chip_style()
        assert widgets.ACCENT in light
        assert widgets.theme_colors("dark")["accent"] not in light
        widgets.set_theme("dark")
        assert widgets.theme_colors("dark")["accent"] in results_page._view_chip_style()
    finally:
        widgets.set_theme(original)


def test_the_shell_is_flat_and_titled_in_the_serif_face():
    """The look is editorial: no gradients, no card around every section.

    A gradient on every button plus a rounded bordered box around each block
    is what made the window read as a dashboard; the identity now comes from
    one serif face on hairlines, and one flat copper plate for the action the
    singer must press.
    """
    sheet = widgets.theme_stylesheet()
    assert "qlineargradient" not in sheet
    assert f'font-family: "{widgets.display_font_family()}"' in sheet
    assert f'font-family: "{widgets.ui_font_family()}"' in sheet
    assert "QGroupBox::title" in sheet
    assert "border: none" in sheet  # sections carry no rounded card
    # the primary action is one solid plate, the disclosure rows are links
    primary = widgets.action_button_style(primary=True)
    assert "qlineargradient" not in primary
    assert f"background: {widgets.ACCENT}" in primary  # one solid plate
    assert "border: none" in widgets.link_button_style()


def test_controls_are_tactile_without_gradients():
    """A press must be FELT: a lit top edge, a grounded bottom edge.

    The shell refuses gradients, so the depth of a control comes from that
    two-tone bevel — and it must INVERT when pressed, or the button only
    changes colour and the click reads as a flicker instead of a movement.
    """
    import re

    sheet = widgets.theme_stylesheet()
    assert "qlineargradient" not in sheet

    def block(pattern: str) -> str:
        match = re.search(pattern, sheet, re.S)
        assert match, f"missing rule: {pattern}"
        return match.group(1)

    def edge(rule: str, side: str) -> str:
        return rule.split(f"border-{side}-color: ", 1)[1].split(";", 1)[0].strip()

    base = block(r"\nQPushButton \{\n(.*?)\n\}")
    pressed = block(r"\nQPushButton:pressed \{\n(.*?)\n\}")
    assert edge(base, "top") == edge(pressed, "bottom")
    assert edge(base, "bottom") == edge(pressed, "top")
    assert edge(base, "top") != edge(base, "bottom")
    # the hover is the warm copper wash, never a second accent colour
    assert f"QPushButton:hover {{ background: {widgets.ACCENT_SOFT};" in sheet
    assert f"border-color: {widgets.ACCENT};" in sheet

    primary = widgets.action_button_style(primary=True)
    assert "qlineargradient" not in primary
    assert "border-top-color" in primary and "border-bottom-color" in primary
    assert "border-top-color" in widgets.action_button_style(primary=False)


def test_piano_keys_are_ivory_and_ebony_in_both_themes(qapp):
    """A keyboard keeps ivory whites and ebony sharps in BOTH palettes.

    Painting the keys from the theme's text and sunken tokens inverted the
    light theme (dark whites, pale sharps) and let the ebony keys dissolve
    into the dark page, so the keyboard read as a band of stripes.
    """
    from PyQt6.QtGui import QColor

    original = widgets.current_theme()
    try:
        for theme in ("dark", "light"):
            widgets.set_theme(theme)
            white = QColor(widgets.KEY_WHITE).lightness()
            black = QColor(widgets.KEY_BLACK).lightness()
            # ivory against ebony, in BOTH palettes (the inverted light theme
            # read 30 against 220)
            assert white - black > 120, theme
            view = widgets.PianoView()
            view.resize(300, 90)
            assert not view.grab().isNull()
    finally:
        widgets.set_theme(original)


def test_invalid_theme_does_not_replace_active_palette():
    original = widgets.current_theme()
    original_stylesheet = widgets.theme_stylesheet()
    with pytest.raises(ValueError, match="unknown theme"):
        widgets.set_theme("sepia")
    assert widgets.current_theme() == original
    assert widgets.theme_stylesheet() == original_stylesheet


def test_selection_has_an_outline_independent_of_its_fill(qapp):
    from v2m.core.notes import Note
    from v2m.gui.widgets import TEXT, PianoRoll

    roll = PianoRoll(selectable=True)
    roll.set_notes([Note(0, 1, 60, 0.9), Note(1, 2, 62, 0.9)])
    roll.set_selection([0])
    selected, other = roll._bar_items
    assert selected.opts["pen"].widthF() == 3
    assert selected.opts["pen"].color().name().lower() == TEXT.lower()
    assert other.opts["pen"].style() == pg.QtCore.Qt.PenStyle.NoPen
    roll.close()


def test_live_tuning_has_words_and_clears_on_silence(qapp):
    from v2m.core.i18n import tr
    from v2m.gui.widgets import LiveIndicator, tuning_advice

    indicator = LiveIndicator()
    indicator.update_note(60, 261.6)
    assert indicator.label_tuning.text() == tr("ui tuning ok")
    indicator.update_note(60.4, 267.7)
    assert indicator.label_tuning.text() == tuning_advice(40)
    indicator.retranslate()
    assert indicator.label_tuning.text() == tuning_advice(40)
    indicator.update_note(None, 0)
    assert indicator.label_tuning.text() == ""
    indicator.close()


def test_stylesheet_parses_without_qt_warnings(qapp, qt_messages):
    """The full stylesheet must parse; no parse error, no font warning."""
    from PyQt6.QtWidgets import QLabel, QMainWindow

    app = qapp
    previous = app.styleSheet()
    app.setStyleSheet(widgets.theme_stylesheet())
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


def test_every_built_button_style_is_brace_balanced():
    """A stray `}` silently kills a per-widget sheet on Qt 6.12.

    Qt 6.11 tolerated the extra brace this module emitted for the plain action
    buttons; 6.12 refuses the whole sheet ("Could not parse stylesheet of
    object QPushButton") and paints the system default, so the Linux gate ran
    unstyled secondary buttons while every test stayed green. The brace came
    from a NON-f string sitting beside an f-string — the balance below is what
    a machine without both Qt versions can still see.
    """
    for name, style in (
        ("primary action", widgets.action_button_style(primary=True)),
        ("plain action", widgets.action_button_style(primary=False)),
        ("link", widgets.link_button_style()),
        ("quiet", widgets.quiet_button_style()),
        ("theme", widgets.theme_stylesheet()),
    ):
        assert style.count("{") == style.count("}"), f"{name} style: {style[-40:]!r}"


def test_stylesheet_actually_applies(qapp):
    """The sheet must reach the widgets, not only parse.

    A `#` line comment inside the QSS (Qt's parser reads `#` as the start of
    an id selector) makes Qt discard the WHOLE stylesheet SILENTLY — no Qt
    warning, no exception, the app just renders in the system default theme.
    The parse-warning test above cannot see that; this one renders a widget
    and checks the palette actually reached its pixels.
    """
    from PyQt6.QtWidgets import QWidget

    from v2m.gui.widgets import BG_APP

    app = qapp
    previous = app.styleSheet()
    app.setStyleSheet(widgets.theme_stylesheet())
    try:
        widget = QWidget()
        widget.resize(40, 40)
        widget.show()
        app.processEvents()
        image = widget.grab().toImage()
        pixel = image.pixelColor(5, 5)
    finally:
        app.setStyleSheet(previous)
        widget.close()
    expected = widgets.BG_APP.lstrip("#").lower()
    actual = f"{pixel.red():02x}{pixel.green():02x}{pixel.blue():02x}"
    assert actual == expected, f"stylesheet not applied: background {actual} != palette {BG_APP}"


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_text_areas_paint_the_sunken_surface(theme, qapp):
    """The evidence and details views are this shell's surface, not the platform's.

    An unstyled `QPlainTextEdit` paints the style's own base colour, which is
    not the page's sunken surface in either palette; the rule is measured on
    the pixels a laid-out widget produces, never on the stylesheet text.
    """
    from PyQt6.QtWidgets import QPlainTextEdit

    previous = qapp.styleSheet()
    original = widgets.current_theme()
    widget = QPlainTextEdit("mesure")
    widget.resize(160, 90)
    try:
        widgets.set_theme(theme)
        qapp.setStyleSheet(widgets.theme_stylesheet())
        widget.show()
        qapp.processEvents()
        pixel = widget.grab().toImage().pixelColor(140, 80)
    finally:
        widget.close()
        qapp.setStyleSheet(previous)
        widgets.set_theme(original)
    actual = f"{pixel.red():02x}{pixel.green():02x}{pixel.blue():02x}"
    expected = widgets.theme_colors(theme)["bg_sunken"].lstrip("#").lower()
    assert actual == expected, (
        f"{theme}: the text area paints {actual}, not {expected}"
    )


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_splitter_handle_is_the_shells_grab_target(theme, qapp):
    """The results divider is a drag target, so it is painted as one.

    An unstyled `QSplitter` draws the platform's 2 px grip (`656a73` on the
    page's own background here) across a 12 px handle, which reads as a seam
    rather than as something to grab. The rule paints the whole handle in
    `border_soft`. Both states are measured on the pixels a laid-out splitter
    produces, at the width `setHandleWidth` owns.

    The hovered state is measured by rendering its own rule as the base rule:
    the offscreen platform keeps a handle hovered once another test has hovered
    one at the same coordinates, and no pointer move inside the splitter clears
    it, so a `QTest.mouseMove` measurement would report the pointer, not the
    sheet. What the pixels prove here is which colour the hover rule declares.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel, QSplitter

    previous = qapp.styleSheet()
    original = widgets.current_theme()
    split: QSplitter | None = None

    def painted(colour: str) -> int:
        """How many pixels of the middle row carry `colour`."""
        assert split is not None
        image = split.grab().toImage()
        row = image.height() // 2
        target = colour.lstrip("#").lower()
        return sum(
            1
            for x in range(image.width())
            if (
                f"{image.pixelColor(x, row).red():02x}"
                f"{image.pixelColor(x, row).green():02x}"
                f"{image.pixelColor(x, row).blue():02x}"
            )
            == target
        )

    try:
        widgets.set_theme(theme)
        qapp.setStyleSheet(widgets.theme_stylesheet())
        built = QSplitter(Qt.Orientation.Horizontal)
        split = built
        built.setHandleWidth(widgets.SPLITTER_HANDLE_WIDTH)
        built.addWidget(QLabel("gauche"))
        built.addWidget(QLabel("droite"))
        built.resize(240, 48)
        built.show()
        qapp.processEvents()
        built.setSizes([100, 100])
        qapp.processEvents()
        colours = widgets.theme_colors(theme)
        rest = colours["border_soft"].lstrip("#").lower()
        hot = colours["accent"].lstrip("#").lower()
        at_rest = painted(rest)
        qapp.setStyleSheet(
            widgets.theme_stylesheet().replace("QSplitter::handle:hover", "QSplitter::handle")
        )
        qapp.processEvents()
        hovered = painted(hot)
    finally:
        if split is not None:
            split.close()
        qapp.setStyleSheet(previous)
        widgets.set_theme(original)
    width = widgets.SPLITTER_HANDLE_WIDTH
    assert at_rest == width, (
        f"{theme}: the handle paints {at_rest} px of {rest}, not the {width} px it owns"
    )
    assert hovered == width, (
        f"{theme}: the hover rule paints {hovered} px of {hot}, not {width} px"
    )


# A QSS type selector names a class, and a class no page ever builds is a rule
# that styles nothing. QListWidget sat in the sheet after the last page stopped
# using it and no test noticed, because a dead rule never warns: the widget simply
# came back with the platform's look. These families ARE built, just not by our
# own `Name(...)` call, so each carries the reason its rule still has an owner; a
# family that becomes instantiated must leave this table.
QSS_BUILT_ELSEWHERE = {
    "QMainWindow": "the shell subclasses it (`class MainWindow(QMainWindow)`)",
    "QGraphicsView": "pyqtgraph's `PlotWidget` subclasses it; every plot is inked through it",
    "QHeaderView": "`QTableWidget.horizontalHeader()` creates it; the rule styles its sections",
    "QScrollBar": "scroll areas create it; the rule is the shell's 6 px bar everywhere",
    "QStatusBar": "`QMainWindow.statusBar()` creates it on first use",
    "QToolTip": "Qt shows it for any widget's tooltip; no page builds one",
    "QLineEdit": "Qt builds it inside an editable `QComboBox` or a spin box",
}


def _styled_classes(sheet: str) -> set[str]:
    """The Q-classes a sheet SELECTS, never the ones a property value mentions."""
    selectors = re.findall(r"(?m)^([^{}\n]+)\{", sheet)
    return {name for selector in selectors for name in re.findall(r"\bQ[A-Z]\w+\b", selector)}


def _built_classes(sources: str) -> set[str]:
    """The Q-classes the sources INSTANTIATE themselves (`QLabel(...)`)."""
    return set(re.findall(r"\b(Q[A-Z]\w+)\(", sources))


def _gui_sources() -> str:
    root = Path(__file__).resolve().parents[1] / "src" / "v2m" / "gui"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.py")))


def _dead_families(sheet: str, sources: str) -> list[str]:
    """Selectors whose class is neither built here nor justified in the table."""
    return sorted(_styled_classes(sheet) - _built_classes(sources) - set(QSS_BUILT_ELSEWHERE))


def test_every_family_the_sheet_styles_is_built_somewhere():
    """A family nobody builds must not keep a rule.

    The purge that removed QListWidget was done by hand, and nothing would have
    said so had it been forgotten: an unused type selector parses, applies to no
    widget, and leaves the page in the platform's look. This net reads the sheet
    the app really renders, collects the classes it selects, and refuses any that
    no source under `src/v2m/gui` instantiates unless `QSS_BUILT_ELSEWHERE` gives
    its reason. A family that becomes instantiated must leave that table, so the
    exceptions cannot rot into a blanket whitelist.
    """
    sources = _gui_sources()
    dead = _dead_families(STYLESHEET, sources)
    assert not dead, (
        "the sheet styles families no page builds (delete the rules, or add the "
        f"reason to QSS_BUILT_ELSEWHERE): {dead}"
    )
    stale = sorted(name for name in QSS_BUILT_ELSEWHERE if name in _built_classes(sources))
    assert not stale, (
        f"these families are instantiated now, drop them from QSS_BUILT_ELSEWHERE: {stale}"
    )


def test_the_dead_family_net_refuses_a_rule_nobody_builds():
    """The net's own bite: an unused class in a sheet is named, a used one is not."""
    sheet = "QListWidget { color: red; }\nQSlider { min-height: 20px; }\n"
    sources = "slider = QSlider(Qt.Orientation.Horizontal)\n"
    assert _dead_families(sheet, sources) == ["QListWidget"]


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_text_area_keeps_its_inner_rect_under_the_ring(theme, qapp):
    """The text area's inner rect is its own rule's arithmetic, in both states.

    A ring that lands on a widget Qt re-lays-out would shave the evidence text
    every time the Tab chain arrives. This widget does not: the edge and the
    padding the base rule declares are the whole story, so the first glyph and
    the inner rect are read BEFORE and AFTER focus and must not move. The rule's
    numbers come from the sheet, so a changed padding is caught by the measured
    origin disagreeing with the declaration.
    """
    import re

    from PyQt6.QtWidgets import QPlainTextEdit

    sheet = widgets.theme_stylesheet()
    rule = re.search(r"QPlainTextEdit \{[^}]*\}", sheet)
    assert rule is not None, "the text area's base rule is gone from the sheet"
    declared = rule.group(0)
    border = re.search(r"border: (\d+)px solid", declared)
    padding = re.search(r"padding: (\d+)px (\d+)px", declared)
    assert border is not None and padding is not None, (
        f"the base rule no longer declares an edge and a padding: {declared!r}"
    )
    edge = int(border.group(1))
    pad_x, pad_y = int(padding.group(2)), int(padding.group(1))

    previous = qapp.styleSheet()
    original = widgets.current_theme()
    area: QPlainTextEdit | None = None
    try:
        widgets.set_theme(theme)
        qapp.setStyleSheet(sheet)
        text_colour = widgets.theme_colors(theme)["text"].lstrip("#").lower()
        ink = tuple(int(text_colour[i : i + 2], 16) for i in (0, 2, 4))
        built = QPlainTextEdit("MMMM")
        area = built
        built.resize(160, 90)
        built.show()
        qapp.processEvents()

        def first_ink() -> tuple[int, int] | None:
            image = built.grab().toImage()
            for y in range(image.height()):
                for x in range(image.width()):
                    pixel = image.pixelColor(x, y)
                    if (pixel.red(), pixel.green(), pixel.blue()) == ink:
                        return x, y
            return None

        resting_rect = built.contentsRect()
        resting_ink = first_ink()
        built.setFocus()
        qapp.processEvents()
        focused_rect = built.contentsRect()
        focused_ink = first_ink()
    finally:
        if area is not None:
            area.close()
        qapp.setStyleSheet(previous)
        widgets.set_theme(original)

    origin_x, origin_y = edge + pad_x, edge + pad_y
    assert (resting_rect.left(), resting_rect.top()) == (origin_x, origin_y), (
        f"{theme}: the rule declares {edge} px of edge and {pad_x}/{pad_y} px of "
        f"padding, so the text starts at ({origin_x}, {origin_y}); measured "
        f"({resting_rect.left()}, {resting_rect.top()})"
    )
    assert focused_rect == resting_rect, (
        f"{theme}: focus moved the inner rect from {resting_rect.getRect()} to "
        f"{focused_rect.getRect()}"
    )
    assert focused_ink == resting_ink, (
        f"{theme}: focus moved the first glyph from {resting_ink} to {focused_ink}"
    )


def test_no_hash_line_comment_in_the_qss():
    """`#` opens an id selector in QSS — a '#' comment drops the whole sheet."""
    offenders = [line for line in STYLESHEET.splitlines() if line.lstrip().startswith("# ")]
    assert not offenders, "invalid '#' comment inside the QSS:\n" + "\n".join(offenders)


def test_no_double_hash_and_no_placeholder_left(qapp):
    """`##` can only come from the accent substitution going wrong."""
    assert "##" not in STYLESHEET
    assert "@ACCENT@" not in STYLESHEET  # placeholder fully substituted
    assert ACCENT in STYLESHEET  # accent color actually injected


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
