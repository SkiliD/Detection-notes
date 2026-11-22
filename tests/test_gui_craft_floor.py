"""The desktop's UI craft floors, the same four the page pins.

AGENTS.md's "UI craft gates" hold on BOTH surfaces. The page measures them in
Chrome (`web/test/craft-floor.test.mjs`); nothing measured the Qt shell, so the
desktop could drift from a floor the page enforces and nobody would see it.
The gaps this file closes, each one measured rather than assumed:

1. **Hit area.** The page's floor is 24x24 CSS px (WCAG 2.2 AA 2.5.8). The
   window's input-device selector stood 22 px tall — the control that chooses
   which microphone a take records from was the one below the floor.
2. **Focus ring.** The page draws 2 px of copper at 3:1 against the surface
   behind it (WCAG 1.4.11). The shell's ring was 1 px and several focusable
   families (combo box, check box, slider, line edit, the tables) drew no
   ring at all: the Tab chain could reach them blind.
3. **Readable disabled states.** A disabled control keeps its word at 4.5:1 and
   is never faded with `opacity` — the page's rule, and QSS can express the same
   failure.
4. **Locale numbers.** The page has ONE `Intl.NumberFormat`; the desktop wrote
   every decimal with a dot, so a French window read "1.0 s" beside a French
   page reading "1,0 s".    `i18n.format_number` is that one formatter, and the
   census below refuses a call site — and an i18n template — that formats a
   decimal by hand.

The sizes and the ring are read from the WIDGETS (measured after a real layout
pass), not from the sheet: a declaration can be overridden by a later rule.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.gui import widgets

GUI = Path(widgets.__file__).parent

FLOOR = 24


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def app_sheet(qapp):
    """The one application sheet every floor below is measured through.

    A widget-level or page-level sheet would answer for itself; the ring's size
    neutrality is a property of the APP's sheet, so it is installed once and
    both scenes read it: the synthetic rig and the real results page.
    """
    original = widgets.current_theme()
    widgets.set_theme("dark")
    previous = qapp.styleSheet()
    qapp.setStyleSheet(widgets.theme_stylesheet())
    try:
        yield qapp
    finally:
        qapp.setStyleSheet(previous)
        widgets.set_theme(original)


@pytest.fixture(scope="module")
def window(app_sheet):
    """One real window, so the floors are measured on the widgets on screen."""
    from v2m.gui import main_window as mw

    win = mw.MainWindow()
    win.resize(1280, 900)
    win.show()
    app_sheet.processEvents()
    try:
        yield win
    finally:
        win.close()


def _luminance(value: str) -> float:
    channels = [int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    high, low = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


# The families the Tab chain can land on. `PianoView` and `PianoRoll` are
# custom-painted views whose focus the keyboard tests already walk; their ring
# is the QGraphicsView frame, and they are not sized like a control.
CONTROL_TYPES = (
    "QPushButton",
    "QCheckBox",
    "QComboBox",
    "QSpinBox",
    "QSlider",
    "QLineEdit",
    "QPlainTextEdit",
)


def _visible_controls(page) -> list[tuple[str, int, int]]:
    """Every visible, focusable control of the current page, by class and box."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QWidget

    found: list[tuple[str, int, int]] = []
    for child in page.findChildren(QWidget):
        if type(child).__name__ not in CONTROL_TYPES:
            continue
        if child.isHidden() or not child.isVisible():
            continue
        if child.focusPolicy() == Qt.FocusPolicy.NoFocus:
            continue
        found.append((type(child).__name__, child.width(), child.height()))
    return found


def test_every_visible_control_clears_the_twenty_four_pixel_floor(window, qapp):
    measured: list[tuple[str, str, int, int]] = []
    for page_name in ("page_home", "page_live", "page_finder", "page_results"):
        page = getattr(window, page_name)
        window.stack.setCurrentWidget(page)
        qapp.processEvents()
        for kind, width, height in _visible_controls(page):
            measured.append((page_name, kind, width, height))
    assert len(measured) >= 12, f"only {len(measured)} controls were measured"
    short = [
        f"{page} {kind} {width}x{height}"
        for page, kind, width, height in measured
        if width < FLOOR or height < FLOOR
    ]
    assert short == [], f"controls under the {FLOOR} px floor:\n  " + "\n  ".join(short)


# Every family whose focus must be VISIBLE, and the rule the sheet draws it with.
FOCUS_FAMILIES = (
    "QPushButton",
    "QCheckBox",
    "QComboBox",
    "QSpinBox",
    "QSlider",
    "QTableWidget",
    "QLineEdit",
)


def test_every_focusable_family_draws_the_same_ring():
    sheet = widgets.theme_stylesheet()
    # Selector lists are read whole (`QCheckBox:focus, QSlider:focus, …` share
    # one rule, exactly like the page's grouped `:focus-visible` rule).
    rules = [
        (selectors, body)
        for selectors, body in re.findall(r"([^{}]+)\{([^}]*)\}", sheet)
        if ":focus" in selectors
    ]
    # The loop below is the contract: EVERY family must name a rule. A count
    # constant on top of it only encodes how many families share the grouped
    # selector list, so it says nothing the loop does not already refuse.
    assert rules, "no focus rule was found in the sheet"
    for family in FOCUS_FAMILIES:
        ring = [body for selectors, body in rules if f"{family}:focus" in selectors]
        assert ring, f"{family} can take focus but no rule draws its ring"
        # The ring is 2 px (the page's floor), not a 1 px hairline, and it is
        # the SAME copper: a family that picks its own colour drifts at the
        # first theme switch.
        for body in ring:
            assert "2px dashed" in body, f"{family}:focus is not a 2 px dashed ring"
            assert widgets.ACCENT in body, f"{family}:focus lost the copper ring"


def test_the_focus_ring_clears_three_to_one_in_both_palettes(window):
    # Non-text contrast (WCAG 1.4.11): a ring nobody can see is not a focus
    # state. The ring sits on the control's own surface, so every surface a
    # control can stand on is measured.
    original = widgets.current_theme()
    try:
        for theme in widgets.THEMES:
            widgets.set_theme(theme)
            colors = widgets.theme_colors(theme)
            for surface in ("bg_app", "bg_card", "bg_sunken"):
                ratio = _contrast(colors["accent"], colors[surface])
                assert ratio >= 3.0, f"{theme}: the focus ring reads {ratio:.2f}:1 on {surface}"
    finally:
        widgets.set_theme(original)


FOCUS_CONTROL_FAMILIES = (
    "QPushButton",
    "QCheckBox",
    "QComboBox",
    "QSpinBox",
    "QSlider",
    "QLineEdit",
    "QPlainTextEdit",
    "QTableWidget",
)


def test_focus_never_resizes_a_control(app_sheet):
    """A ring is a STATE, never a size: taking focus must not move the box.

    A border drawn only on `:focus` costs the control 2 px on each side, so
    every family is measured on a real, laid-out widget with the application
    sheet installed — hint AND geometry, before and after the focus change. The
    families share one synthetic host, because a page's own layout answers for
    its neighbours and that is the results page's subject below.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QLineEdit,
        QPlainTextEdit,
        QPushButton,
        QSlider,
        QSpinBox,
        QTableWidget,
        QVBoxLayout,
        QWidget,
    )

    builders = {
        "QPushButton": lambda: QPushButton("Bouton"),
        "QCheckBox": lambda: QCheckBox("Case"),
        "QComboBox": lambda: QComboBox(),
        "QSpinBox": lambda: QSpinBox(),
        # the only slider the app builds is horizontal (`ResultsPage._qt_orientation`),
        # and the sheet's groove/handle rules are `:horizontal`: a default
        # `QSlider()` is VERTICAL and would leave them unexercised.
        "QSlider": lambda: QSlider(Qt.Orientation.Horizontal),
        "QLineEdit": lambda: QLineEdit("champ"),
        "QPlainTextEdit": lambda: QPlainTextEdit("texte"),
        "QTableWidget": lambda: QTableWidget(4, 3),
    }
    assert tuple(builders) == FOCUS_CONTROL_FAMILIES, (
        "the rig and the families the page walk filters on drifted"
    )

    def table_facts(table: QTableWidget) -> tuple:
        return (
            table.viewport().size().width(),
            table.viewport().size().height(),
            tuple(table.columnWidth(i) for i in range(table.columnCount())),
            table.verticalScrollBar().isVisible(),
            table.horizontalScrollBar().isVisible(),
        )

    host = QWidget()
    layout = QVBoxLayout(host)
    controls: dict[str, QWidget] = {}
    for name, builder in builders.items():
        control = builder()
        if name == "QComboBox":
            control.addItem("apparence sombre")
            control.addItem("apparence claire")
        if name == "QTableWidget":
            control.setColumnWidth(0, 111)
            control.setColumnWidth(1, 74)
        controls[name] = control
        layout.addWidget(control)
    host.resize(640, 900)
    host.show()
    app_sheet.processEvents()
    moved: list[str] = []
    try:
        for name, control in controls.items():
            app_sheet.processEvents()
            hint_before = control.sizeHint()
            box_before = control.geometry()
            facts_before = table_facts(control) if isinstance(control, QTableWidget) else None
            control.setFocus()
            app_sheet.processEvents()
            hint_after = control.sizeHint()
            box_after = control.geometry()
            if (hint_before.width(), hint_before.height()) != (
                hint_after.width(),
                hint_after.height(),
            ):
                moved.append(
                    f"{name} hint {hint_before.width()}x{hint_before.height()}"
                    f" -> {hint_after.width()}x{hint_after.height()}"
                )
            if (box_before.width(), box_before.height()) != (
                box_after.width(),
                box_after.height(),
            ):
                moved.append(
                    f"{name} box {box_before.width()}x{box_before.height()}"
                    f" -> {box_after.width()}x{box_after.height()}"
                )
            if facts_before is not None and facts_before != table_facts(control):
                moved.append(
                    f"{name} viewport/columns/bars {facts_before} -> {table_facts(control)}"
                )
    finally:
        host.close()
    assert moved == [], "a focus state resized a control:\n  " + "\n  ".join(moved)


def test_the_slider_base_rule_pays_for_the_reserved_ring(app_sheet):
    """The slider's hint follows the rule's ARITHMETIC, not the style's pixels.

    The comment in `widgets._stylesheet` says 24 px = 20 px of body + 2 x 2 px of
    edge; the two numbers are read from the rule before anything is measured, so
    a change to the rule is checked against itself. The body Qt's PLATFORM STYLE
    gives a bare slider is deliberately NOT pinned -- 15 px with offscreen Fusion
    here, another value under 6.12 or another style. What the rule promises is
    that one edge of `border` px costs `2 * border`, and that `min-height` lifts
    a shorter body, so the reserved ring asks `max(body, min-height) + 2 * border`
    and clears the 24 px floor whatever body the platform style gives. The edges
    ALONE would clear it only while that body stays under `min-height` (15 and 16
    px here), which is a property of the style, not of the rule: no assertion
    below reads it.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QSlider

    def hint_height(scene: str) -> int:
        app_sheet.setStyleSheet(scene)
        slider = QSlider(Qt.Orientation.Horizontal)
        try:
            slider.ensurePolished()
            return slider.sizeHint().height()
        finally:
            slider.deleteLater()

    sheet = widgets.theme_stylesheet()
    rule = re.search(r"QSlider \{[^}]*\}", sheet)
    assert rule is not None, "the slider's base rule is gone from the sheet"
    declared = rule.group(0)

    def declared_px(pattern: str) -> int:
        match = re.search(pattern, declared)
        assert match is not None, f"the base rule no longer declares {pattern!r}"
        return int(match.group(1))

    edge_px = declared_px(r"border: (\d+)px solid transparent")
    min_height_px = declared_px(r"min-height: (\d+)px")
    assert min_height_px + 2 * edge_px == FLOOR, (
        f"the rule must reserve the {FLOOR} px floor: min-height {min_height_px} "
        f"+ {2 * edge_px} px of edge"
    )

    without_rule = sheet.replace(declared, "")
    edge_only_sheet = sheet.replace(
        declared, declared.replace(f"min-height: {min_height_px}px; ", "")
    )
    previous = app_sheet.styleSheet()
    try:
        reserved = hint_height(sheet)
        edge_only = hint_height(edge_only_sheet)
        bare = hint_height(without_rule)
    finally:
        app_sheet.setStyleSheet(previous)
    assert edge_only - bare == 2 * edge_px, (
        f"the two edges must add {2 * edge_px} px to the style's own body, "
        f"not {edge_only - bare} px"
    )
    assert reserved == max(bare, min_height_px) + 2 * edge_px, (
        f"the reserved ring must ask max(body {bare}, min-height {min_height_px}) "
        f"+ {2 * edge_px} px, not {reserved} px"
    )


def test_focus_never_reflows_the_results_page(app_sheet):
    """Focus must not move the page AROUND the control that takes it.

    A hint alone never shows the symptom: a control that grows makes the
    details panel taller than its box, the panel's scrollbar appears, and the
    controls beside it lose that scrollbar's width. The real results page is
    walked with its details open — every child compared POSITIONALLY, and the
    scrollbar's state re-read after each stop — because that is where a focus
    state stops being local.
    """
    from PyQt6.QtWidgets import QWidget

    from v2m.core.notes import Note
    from v2m.core.pipeline import AnalysisResult
    from v2m.gui import main_window as mw

    local = mw.MainWindow()
    try:
        result = AnalysisResult(
            source="song.wav",
            duration_s=6.0,
            notes=[Note(i * 0.35, i * 0.35 + 0.25, 60 + i % 5, 0.9) for i in range(16)],
            tempo=None,
            key=None,
            pitch=None,
        )
        local.result = result
        local._show_result(result)
        local.stack.setCurrentWidget(local.page_results)
        local.resize(1280, 620)
        local.page_results.details_toggle.setChecked(True)
        local.show()
        for _ in range(4):
            app_sheet.processEvents()
        page = local.page_results
        bar = page.details_scroll.verticalScrollBar()
        assert page.details_scroll.isVisible() and bar.maximum() > 0, (
            "the details panel must scroll for this check to mean anything"
        )
        bar_state = (bar.isVisible(), bar.minimum(), bar.maximum())

        # `findChildren` rebuilds its Python wrappers on every call, so the
        # snapshot is compared POSITIONALLY: the child order is stable, the
        # objects' ids are not.
        def surroundings() -> tuple[tuple[str, int, bool], ...]:
            return tuple(
                (type(child).__name__, child.width(), child.isVisible())
                for child in page.findChildren(QWidget)
            )

        baseline = surroundings()
        walked = 0
        drifted: list[str] = []
        for control in page.findChildren(QWidget):
            if (
                type(control).__name__ not in FOCUS_CONTROL_FAMILIES
                or not control.isVisible()
            ):
                continue
            control.setFocus()
            for _ in range(2):
                app_sheet.processEvents()
            walked += 1
            now = surroundings()
            changes = [
                f"{baseline[index][0]} {baseline[index][1]} -> {now[index][1]}"
                for index in range(min(len(baseline), len(now)))
                if baseline[index] != now[index]
            ]
            now_bar = (bar.isVisible(), bar.minimum(), bar.maximum())
            if now_bar != bar_state:
                changes.append(f"the details scrollbar state {bar_state} -> {now_bar}")
            if changes:
                drifted.append(f"{type(control).__name__}: " + ", ".join(changes))
        assert walked >= 6, f"only {walked} controls were traversed"
        assert drifted == [], (
            "focus moved the page around a control:\n  " + "\n  ".join(drifted)
        )
    finally:
        local.close()


def test_a_widget_level_sheet_still_draws_the_ring(window):
    """A widget that paints ITSELF repeats the ring, because its sheet wins.

    The appearance selector names its own colours and border in a widget-level
    sheet, which shadows the application's `QComboBox:focus` rule for those
    properties: its Tab stop drew no focus state at all. The selector carries
    the ring itself, and the ring stays size-neutral because the padding pays
    for the extra pixel of border on each side.
    """
    selector = window.page_home.theme_selector
    sheet = selector.styleSheet()
    focus_rules = re.findall(r"QComboBox:focus\s*\{([^}]*)\}", sheet)
    assert len(focus_rules) == 1, f"the self-styled selector draws no ring: {sheet}"
    base_rules = re.findall(r"QComboBox\s*\{([^}]*)\}", sheet)
    assert len(base_rules) == 1, sheet
    assert "2px dashed" in focus_rules[0], sheet
    assert widgets.ACCENT in focus_rules[0], "the ring lost the copper token"

    def _border_plus_padding(body: str) -> int:
        border = re.search(r"border:\s*(\d+)px", body)
        padding = re.search(r"padding:\s*(\d+)px", body)
        assert border and padding, f"a declared rule lost its border or padding: {body}"
        return int(border.group(1)) + int(padding.group(1))

    assert _border_plus_padding(base_rules[0]) == _border_plus_padding(focus_rules[0]), (
        "the ring changes the selector's height when it takes focus"
    )


def test_a_disabled_control_keeps_its_word_in_both_palettes():
    # The page's rule: the inactive state is PAINTED (a quiet field, a readable
    # ink), never faded — an opacity fade is what took the hovered plate's own
    # label to 3.58:1 on the page.
    sheet = widgets.theme_stylesheet()
    assert "opacity" not in sheet, "the shell fades a control's label again"
    for helper in (
        widgets.action_button_style(primary=True),
        widgets.action_button_style(primary=False),
        widgets.link_button_style(),
        widgets.quiet_button_style(),
    ):
        assert "opacity" not in helper
        assert ":disabled" in helper, "a control style ships with no inactive state"
    original = widgets.current_theme()
    try:
        for theme in widgets.THEMES:
            widgets.set_theme(theme)
            colors = widgets.theme_colors(theme)
            for surface in ("bg_app", "bg_card", "bg_sunken"):
                ratio = _contrast(colors["text_faint"], colors[surface])
                assert ratio >= 4.5, f"{theme}: a disabled label reads {ratio:.2f}:1 on {surface}"
    finally:
        widgets.set_theme(original)


def test_french_writes_a_comma_and_english_a_dot():
    from v2m.core import i18n

    original = i18n.get_language()
    try:
        i18n.set_language("fr")
        assert i18n.format_number(3.2, 1) == "3,2"
        assert i18n.format_number(12, 2) == "12,00"
        assert i18n.format_number(-4.5, 1, sign=True) == "-4,5"
        assert i18n.format_number(4.5, 1, sign=True) == "+4,5"
        assert i18n.format_number(float("nan"), 1) == "0,0"
        i18n.set_language("en")
        assert i18n.format_number(3.2, 1) == "3.2"
        assert i18n.format_number(4.5, 1, sign=True) == "+4.5"
    finally:
        i18n.set_language(original)


def test_the_roll_and_the_waveform_write_their_seconds_in_the_window_language(qapp):
    """`SecondAxis`: the timeline's ticks follow the ONE number formatter.

    pyqtgraph writes its own tick strings with `%.*f`, which puts a dot in
    every language: a French roll read "0.5 / 1.0" beside a French table
    reading "0,5 s / 1,0 s". Both plot widgets build this axis, so the labels
    are read off the axis each one really uses.
    """
    from v2m.core import i18n

    original = i18n.get_language()
    views: list = []
    try:
        views = [widgets.PianoRoll(), widgets.WaveformView()]
        for view in views:
            axis = view.getPlotItem().getAxis("bottom")
            assert isinstance(axis, widgets.SecondAxis), (
                f"{type(view).__name__} lost the localized second axis"
            )
            i18n.set_language("fr")
            french = axis.tickStrings([0.0, 1.5, 12.25], 1.0, 0.1)
            i18n.set_language("en")
            english = axis.tickStrings([0.0, 1.5, 12.25], 1.0, 0.1)
            assert french == ["0,0", "1,5", "12,2"], french
            assert english == ["0.0", "1.5", "12.2"], english
            # the precision is pyqtgraph's own spacing rule: whole seconds on a
            # wide view, hundredths once the view zooms in.
            i18n.set_language("fr")
            assert axis.tickStrings([2.0], 1.0, 1.0) == ["2"]
            assert axis.tickStrings([1.25], 1.0, 0.01) == ["1,25"]
    finally:
        for view in views:
            view.close()
        i18n.set_language(original)


def test_the_results_summary_speaks_the_window_s_language(window):
    from v2m.core import i18n

    page = window.page_results
    original = i18n.get_language()
    try:
        page._summary_facts = (4, 1.2, 94.5, None)
        i18n.set_language("fr")
        page._render_summary()
        french = page.label_summary.text()
        i18n.set_language("en")
        page._render_summary()
        english = page.label_summary.text()
    finally:
        i18n.set_language(original)
        page._summary_facts = None
        page._render_summary()
    assert "1,2 s" in french and "94 BPM" in french, french
    assert "1.2 s" in english and "94 BPM" in english, english
    assert french.count(",") == 1, f"a second hand-built number slipped in: {french}"


# A decimal format spec written at a call site is the defect the page's census
# refuses: `f"{x:.1f}"` writes a dot in every language. The exceptions are named
# with WHY, so a new site fails until someone names it too.
HAND_FORMATTED = {
    "capture too short to analyze": (
        "the take's engine failure text, carried into its trace — never painted"
    ),
}

DECIMAL_SPEC = re.compile(r"\.\d+f")


# A format spec inside a TEMPLATE is the same defect one layer lower: the
# template decides the separator, so `"{duration:.1f} s"` writes a dot in
# French whatever the call site passes. Integer specs (`{bpm:.0f}`,
# `{version:03d}`) carry no separator and stay legal; the console verifier's
# lines are named with their reason.
TEMPLATE_EXCEPTIONS = {
    "verify wav": "the console verifier prints its durations with a dot",
    "verify json": "the console verifier prints its durations with a dot",
    "verify gap": "the console verifier prints its durations with a dot",
    "cmp tempo": "printed by scripts/compare_takes.py, never painted by the window",
    "cmp shifted": "printed by scripts/compare_takes.py, never painted by the window",
    "cmp moved": "printed by scripts/compare_takes.py, never painted by the window",
    "cmp missing row": "printed by scripts/compare_takes.py, never painted by the window",
    "cmp added row": "printed by scripts/compare_takes.py, never painted by the window",
}

DECIMAL_TEMPLATE_SPEC = re.compile(r"\.(?:[1-9]\d*)f")


def _decimal_template_specs(table: dict[str, str]) -> list[str]:
    offenders: list[str] = []
    for key, text in table.items():
        if key in TEMPLATE_EXCEPTIONS:
            continue
        for placeholder in re.findall(r"\{[^{}]*\}", text):
            if DECIMAL_TEMPLATE_SPEC.search(placeholder):
                offenders.append(f"{key}: {placeholder}")
    return offenders


def test_no_i18n_template_decides_a_decimal_separator():
    from v2m.core import i18n

    offenders = sorted(
        f"{language} {row}"
        for language, table in i18n.STRINGS.items()
        for row in _decimal_template_specs(table)
    )
    assert not offenders, (
        "templates writing their own decimal separator (pass a string formatted by "
        "i18n.format_number):\n  " + "\n  ".join(offenders)
    )


def test_the_template_census_refuses_a_reintroduced_decimal_spec():
    # The shape that shipped: a tenth inside the template.
    assert _decimal_template_specs({"ui summary facts": "{n} notes · {duration:.1f} s"}) == [
        "ui summary facts: {duration:.1f}"
    ]
    # ...and the shapes that must stay legal: no separator to translate.
    assert _decimal_template_specs({"ui status live ready": "{bpm:.0f} BPM · {n} notes"}) == []
    assert _decimal_template_specs({"ui compare corrected": "Correction v{version:03d}"}) == []
    assert _decimal_template_specs({"verify wav": "wav : {name}  {dur:.2f} s"}) == []


def test_no_gui_module_assembles_a_displayed_decimal_by_hand():
    offenders: list[str] = []
    for py_file in sorted(GUI.glob("*.py")):
        for number, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
            if not DECIMAL_SPEC.search(line) or "format_number" in line:
                continue
            if any(reason for marker, reason in HAND_FORMATTED.items() if marker in line):
                continue
            offenders.append(f"{py_file.name}:{number}: {line.strip()}")
    assert not offenders, (
        "displayed numbers assembled by hand (use i18n.format_number):\n  " + "\n  ".join(offenders)
    )
