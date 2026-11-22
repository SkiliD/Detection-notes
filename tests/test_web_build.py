"""The web port is a COPY, so it needs the pins a copy needs.

`web/` reimplements the live half of the desktop (YIN, the attack tracker,
the segmenter) in JavaScript, because none of the Python stack runs in a
browser: librosa, numba and sounddevice are desktop-only, and the page is a
static folder GitHub Pages serves as-is.

A copy drifts. These tests make the drift loud instead of slow:

- every ported constant must still equal its Python source, from ONE
  generated file (`web/constants.js`) that the modules import: a threshold
  widened on one side would segment differently and nobody would notice;
- `web/test/yin.json` must still be what the REAL Python YIN reads on two
  excerpts of the demo take — the ascent (every scale degree) and the tail (the
  low pedal at the lag search's limit); cutting the take is only allowed while
  `_yin_spans_the_take` sees both ends (the browser's own estimator is compared
  to it frame by frame in `web/test/yin.test.mjs`, so a drift of one lag sample
  cannot hide), and
  `web/test/yin-rates.json` the same at the two capture rates, where the
  window, the hop and the transform size are not the analysis rate's — its
  excerpt is the take's tail so the lag search's low limit (`tau_max`) is
  reached, not just frozen;
- `web/test/cases.json` must still be what the REAL Python segmenter decides
  over the trap families, regenerated in-process by
  `scripts/export_web_fixtures.py` — change liveseg.py and this fails until
  the fixtures are regenerated AND the JavaScript still agrees
  (`node --test web/test/segmenter.test.mjs`, run by the Pages workflow);
- the page stays dependency-free and self-contained (no CDN, no build);
- the archive the page exports, and the parity fixtures its reading is
  compared against, are pinned in `tests/test_web_export.py`;
- the workflow actually publishes `web/` and runs that test first;
- both languages of the page carry the same keys, and every key the page
  asks for exists;
- the page's craft floors that need no browser — a visible focus ring on every
  interactive selector, and one reduced-motion block that silences every
  transition — are read from the sheet itself, so a deleted rule fails the
  ordinary suite before Chrome is ever started.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
sys.path.insert(0, str(REPO))

CONSTANTS = WEB / "constants.js"


def _js_value(text: str, name: str):
    """One generated declaration, read back as the Python value it mirrors.

    JSON covers both shapes the generator emits (a number, a flat list of
    strings), so the read-back cannot accept a value the browser could not
    parse either.
    """
    match = re.search(rf"^export const {name} = (.+);$", text, re.M)
    assert match, f"`export const {name}` not found"
    return json.loads(match.group(1))


def test_the_pages_ported_values_have_one_generated_source():
    # A threshold restated in a JavaScript module is a threshold that can be
    # widened on one side only, and a note-name table restated there can print
    # a spelling the desktop never produces: the page would then decide or
    # say something different and nothing it prints would reveal it. The
    # values are therefore GENERATED from the Python attributes they mirror,
    # and every module imports them instead of declaring its own copy.
    from importlib import import_module

    from scripts.export_web_fixtures import CONSTANT_SOURCES, constants_source

    regenerate = "run `python -m scripts.export_web_fixtures`"
    checked = CONSTANTS.read_text(encoding="utf-8")
    assert constants_source() == checked, f"web/constants.js is stale: {regenerate}"

    declared = {name for names in CONSTANT_SOURCES.values() for name in names}
    for module, names in CONSTANT_SOURCES.items():
        for name in names:
            expected = getattr(import_module(module), name)
            actual = _js_value(checked, name)
            if isinstance(expected, (int, float)):
                assert actual == pytest.approx(float(expected)), (
                    f"constants.js:{name} no longer matches {module}.{name}"
                )
            else:
                # a ported TABLE (the note-name list): element for element, so
                # a renamed spelling cannot hide behind equal lengths
                assert list(actual) == list(expected), (
                    f"constants.js:{name} no longer matches {module}.{name}"
                )

    # No second declaration anywhere else: an `import` is the only way in.
    restated = []
    for source in [*sorted(WEB.glob("*.js")), *sorted((WEB / "test").glob("*.mjs"))]:
        if source.name == CONSTANTS.name:
            continue
        body = source.read_text(encoding="utf-8")
        restated += [
            f"{source.name}:{name}"
            for name in sorted(declared)
            if re.search(rf"^(?:export )?const {name} = ", body, re.M)
        ]
    assert not restated, f"these restate a ported value instead of importing it: {restated}"


def test_the_generator_refuses_a_value_it_cannot_emit(monkeypatch):
    """Only two JavaScript shapes exist; anything else fails AT THE GENERATOR.

    The alternative is Python syntax leaking into `web/constants.js` — a tuple
    written `('C', 'D#')`, a dict written `{'bpm': 120}` — a file the browser
    refuses to parse, discovered in the page instead of here. Both halves are
    pinned: the literal emitter refuses the value, and the generator refuses it
    from the module attribute it would have read.
    """
    from scripts.export_web_fixtures import _js_literal, constants_source
    from v2m.core import music

    # the two shapes a ported value may have
    assert _js_literal(25.0) == "25.0"
    assert _js_literal(12) == "12"
    assert _js_literal(("C", "C#")) == '["C", "C#"]'
    assert _js_literal(["Do", "Ré"]) == '["Do", "Ré"]'

    for refused in (True, {"bpm": 120}, [1, 2], None, ("C", 1), 1 + 2j):
        with pytest.raises(TypeError):
            _js_literal(refused)

    # the generator path: a source attribute that is not emittable fails while
    # naming the value, instead of writing something the page cannot parse
    monkeypatch.setattr(music, "A4_HZ", {"hertz": 440})
    with pytest.raises(TypeError, match="no JavaScript literal"):
        constants_source()


def test_the_generated_values_guard_bites(tmp_path, monkeypatch):
    """Both ways a ported value drifts must FAIL, not pass quietly.

    The guard above earns its lines only if it catches the two shapes: a module
    keeping its own copy of a generated value, and the generated file no longer
    matching the Python table it was written from.
    """
    from scripts.export_web_fixtures import constants_source

    module = sys.modules[__name__]
    web = tmp_path / "web"
    (web / "test").mkdir(parents=True)
    (web / "constants.js").write_text(constants_source(), encoding="utf-8")
    monkeypatch.setattr(module, "WEB", web)
    monkeypatch.setattr(module, "CONSTANTS", web / "constants.js")

    # 1. a module that keeps its own note-name table
    (web / "segmenter.js").write_text('export const NOTE_NAMES = ["C"];\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="restate"):
        test_the_pages_ported_values_have_one_generated_source()

    # 2. a generated file that drifted from music.NOTE_NAMES
    (web / "segmenter.js").write_text("", encoding="utf-8")
    (web / "constants.js").write_text(
        constants_source().replace('"C#",', '"Db",'), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="stale"):
        test_the_pages_ported_values_have_one_generated_source()


def test_the_page_is_a_self_contained_folder():
    for name in ("index.html", "style.css", "app.js", "strings.json", "theme-init.js"):
        assert (WEB / name).is_file(), f"web/{name} is missing"
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'src="app.js"' in html
    # No CDN, no font URL, no analytics: the folder must work from anywhere,
    # including a local static server, and stay deployable without a build.
    for pattern in ('src="http', 'href="http', 'src="//', '<script src="//'):
        assert pattern not in html
    for path in WEB.glob("*.js"):
        assert "http://" not in path.read_text(encoding="utf-8")
    # ... and the driver BORROWS the one dependency it needs instead of adding
    # it to the folder: Playwright is installed by the workflow, and the module
    # the page owns must stay free of it.
    assert "playwright" not in (WEB / "test" / "browser.mjs").read_text(encoding="utf-8"), (
        "the page's own driver grew a dependency"
    )
    assert "/node_modules/" in (REPO / ".gitignore").read_text(encoding="utf-8")


def test_the_web_theme_speaks_the_desktop_s_preference_language():
    # A static page has no filesystem, so it cannot read `~/.v2m_theme`: what
    # can be shared is the VOCABULARY. Same name, same two words, so a value
    # carried from one to the other means the same thing without translating.
    from v2m.gui import widgets

    prefs = (WEB / "prefs.js").read_text(encoding="utf-8")
    key = re.search(r'THEME_KEY = "([^"]+)"', prefs)
    assert key, "web/prefs.js no longer declares THEME_KEY"
    # The DECLARATION, not the imported constant: the suite's sandbox renames
    # the file, and a guard that reads the sandbox would pin the fixture.
    source = (REPO / "src" / "v2m" / "gui" / "main_window.py").read_text(encoding="utf-8")
    declared_file = re.search(r'THEME_FILE = Path\.home\(\) / "([^"]+)"', source)
    assert declared_file, "main_window.THEME_FILE is no longer a home file"
    # the leading dot is the hidden-file convention, not part of the name
    assert key.group(1) == declared_file.group(1).lstrip(".") == "v2m_theme"
    declared = re.search(r"THEMES = \[(.*?)\]", prefs)
    assert declared, "web/prefs.js no longer declares THEMES"
    assert set(re.findall(r'"([a-z]+)"', declared.group(1))) == set(widgets.THEMES)


def _web_tokens(selector: str) -> dict[str, str]:
    """The `--token: value` pairs one stylesheet block declares."""
    css = (WEB / "style.css").read_text(encoding="utf-8")
    block = re.search(rf"{re.escape(selector)} \{{(.*?)\n\}}", css, re.S)
    assert block, f"web/style.css no longer declares {selector}"
    pairs = re.findall(r"(--[a-z-]+):\s*([^;]+);", block.group(1))
    return {name.lstrip("-"): value.strip() for name, value in pairs}


def _web_luminance(value: str) -> float:
    """WCAG relative luminance of an opaque `#rrggbb` colour."""
    channels = [int(value.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))


def _web_contrast(foreground: str, background: str) -> float:
    first, second = _web_luminance(foreground), _web_luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


# Every text/background pair the page actually paints words on: the readout,
# the table, the hints and the colophon on the paper; a control's label on its
# own fill; the plate's label on both of its states; and the two masthead
# colours on the night band. Alpha washes (the zebra rows, the playing tint)
# are not here: they sit UNDER the ink, whose pair with the paper is measured
# instead.
WEB_TEXT_PAIRS = (
    ("ink", "paper"),
    ("ink", "button"),
    ("muted", "paper"),
    ("copper-deep", "paper"),
    ("tune-ok", "paper"),
    ("tune-near", "paper"),
    ("tune-off", "paper"),
    ("plate-ink", "plate"),
    ("plate-ink", "plate-hover"),
    ("ivory", "night"),
    ("deck-ink", "night"),
)

# WCAG 1.4.11 (AA): what identifies a CONTROL, not a decorative hairline. The
# roll's frame is the shape of a seek surface, the plate is the one filled
# action, and a secondary button is now a filled FIELD rather than an outlined
# box (`test_secondary_controls_are_fills_not_strokes`). 3.0 is the non-text
# floor; the rules BETWEEN sections (#e3d6bd / #3a332b) are decoration and are
# deliberately absent.
WEB_CONTROL_EDGES = (
    ("control-edge", "paper"),
    ("control-edge", "night"),
    ("plate", "paper"),
    ("copper", "paper"),
    ("copper", "night"),
)


def test_both_web_palettes_keep_their_words_readable():
    # The page is read on a music stand in a dim room as often as at a desk,
    # and a colour picked by eye passes in one palette and fails in the other:
    # the amber verdict sat at 4.15:1 on paper while its two neighbours passed.
    # 4.5 is the small-text floor — every pair below is 10-16 px type.
    for theme, selector in (("light", ":root"), ("dark", '[data-theme="dark"]')):
        tokens = _web_tokens(selector)
        for foreground, background in WEB_TEXT_PAIRS:
            ratio = _web_contrast(tokens[foreground], tokens[background])
            assert ratio >= 4.5, f"{theme}: {foreground} on {background} reads {ratio:.2f}:1"


def _rgb(value: str) -> tuple[int, int, int]:
    """The channels of an opaque `#rrggbb` value."""
    return tuple(int(value.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _composited(value: str, backdrop: tuple[int, int, int]) -> str:
    """`#rrggbb` or `rgba(...)` laid over an opaque backdrop, as `#rrggbb`.

    The roll's colours are washes, and a canvas paints them over whatever is
    already there: comparing the declared wash with the paper would measure a
    colour no eye ever sees.
    """
    if value.startswith("#"):
        channels, alpha = _rgb(value), 1.0
    else:
        numbers = [float(number) for number in re.findall(r"[\d.]+", value)]
        channels, alpha = numbers[:3], numbers[3] if len(numbers) > 3 else 1.0  # type: ignore[assignment]
    return "#%02x%02x%02x" % tuple(
        round(channel * alpha + back * (1 - alpha))
        for channel, back in zip(channels, backdrop, strict=True)
    )


def test_the_rolls_painted_words_are_readable_too():
    # axe reads the DOM, and the roll's octave labels and second marks are
    # PAINTED on its canvas: nothing would ever report them — and measured, the
    # light palette's second marks sat at 2.36:1, fainter than every word the
    # page's own contrast check covers. Composited the way the canvas paints
    # them (text over a black key's lane over the roll's background), the words
    # must clear the 4.5:1 small-text floor AND the bars, which carry the whole
    # reading, the 3:1 non-text floor of WCAG 1.4.11.
    for theme, selector in (("light", ":root"), ("dark", '[data-theme="dark"]')):
        tokens = _web_tokens(selector)
        lane = _composited(tokens["roll-lane"], _rgb(tokens["roll-bg"]))
        lane_rgb = _rgb(lane)
        for name in ("roll-label", "roll-time"):
            ratio = _web_contrast(_composited(tokens[name], lane_rgb), lane)
            assert ratio >= 4.5, f"{theme}: {name} on the roll reads {ratio:.2f}:1"
        bars = _web_contrast(_composited(tokens["roll-bar"], lane_rgb), lane)
        assert bars >= 3.0, f"{theme}: the roll's note bars read {bars:.2f}:1"


def test_the_notes_table_stays_hidden_until_it_has_rows():
    # An empty table with its header row is a table OF nothing: no automated
    # check can pass it (`th-has-data-cells` only ever answers "incomplete"),
    # and a keyboard visitor lands in a scrollable region with nothing to
    # scroll. The empty sentence speaks for it until the first note.
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="notes-table"' in html and 'class="table-scroll"' in html
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert '$("notes-table").hidden = notes.length === 0;' in app, (
        "the empty table is shown again: headers with no data cells under them"
    )


def test_the_filled_table_stays_readable_on_its_own_tints():
    # A row of the notes table is not bare paper: a zebra strip and the playing
    # row's wash sit UNDER the cents verdicts, the doubt words and the times.
    # axe measured those rows once (zero violations); this keeps them measured
    # after any tint or palette edit, since a wash that only looks quiet is
    # exactly how a verdict slips below 4.5:1 without an eye noticing.
    for theme, selector in (("light", ":root"), ("dark", '[data-theme="dark"]')):
        tokens = _web_tokens(selector)
        for tint in ("zebra", "playing-tint"):
            background = _composited(tokens[tint], _rgb(tokens["paper"]))
            for name in ("ink", "muted", "tune-ok", "tune-near", "tune-off"):
                ratio = _web_contrast(_composited(tokens[name], _rgb(background)), background)
                assert ratio >= 4.5, f"{theme}: {name} on the {tint} row reads {ratio:.2f}:1"


def _web_rule(selector: str) -> str:
    """The declaration block of one stylesheet selector."""
    css = (WEB / "style.css").read_text(encoding="utf-8")
    rule = re.search(rf"{re.escape(selector)} \{{(.*?)\}}", css, re.S)
    assert rule, f"web/style.css no longer declares {selector}"
    return rule.group(1)


def _web_sheet() -> str:
    """`web/style.css` without its comments: prose cannot style a control."""
    return re.sub(r"/\*.*?\*/", "", (WEB / "style.css").read_text(encoding="utf-8"), flags=re.S)


def _web_blocks(css: str) -> list[tuple[str, str]]:
    """Every `selectors { declarations }` pair, in sheet order.

    A media query's prelude is skipped by the pattern itself: it opens a brace
    before its first rule, so `[^{}]*` never reaches a closing one.
    """
    return [
        (selectors.strip(), body) for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
    ]


def _web_media(query: str) -> str:
    """The body of one `@media` block, its braces matched rather than guessed."""
    css = _web_sheet()
    at = css.find(query)
    assert at != -1, f"web/style.css no longer declares {query}"
    start = css.index("{", at)
    depth = 0
    for index in range(start, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[start + 1 : index]
    raise AssertionError(f"{query} never closes")


# Each interactive control on the page, the selector that draws its ring, and
# the markup that proves the control still exists. `button` carries the masthead
# pills, the tutorial's two text links and the reset: the page styles them as
# buttons, so the grouped rule is their ring. The file input's own box is
# invisible, so its ring is painted on the label beside it.
WEB_FOCUS_CONTROLS = (
    ("the buttons", "button", "<button"),
    ("the roll", "#roll", 'id="roll"'),
    ("the tutorial's summary", ".guide > summary", "<summary"),
    ("the file input's label", ".file input", 'type="file"'),
    ("the balance slider", 'input[type="range"]', 'type="range"'),
    ("the notes scroller", ".table-scroll", 'class="table-scroll"'),
)


def test_every_interactive_control_has_a_visible_focus_ring():
    # The browser walk (`web/test/craft-floor.test.mjs`) proves this on a
    # rendered page, but it needs Chrome; this pins the same floor in the
    # ordinary suite, so deleting a focus rule fails here first. A renamed
    # control or selector is caught from both sides: the markup must still
    # carry the control, and the sheet must still draw a ring on it.
    css = _web_sheet()
    html = (WEB / "index.html").read_text(encoding="utf-8")
    focus_rules = [
        (selectors, body) for selectors, body in _web_blocks(css) if ":focus-visible" in selectors
    ]
    assert len(focus_rules) >= 4, f"only {len(focus_rules)} focus rules are left"
    for label, selector, evidence in WEB_FOCUS_CONTROLS:
        assert evidence in html, f"{label}: the control left the page"
        entry = f"{selector}:focus-visible"
        body = next(
            (
                body
                for selectors, body in focus_rules
                if any(
                    piece.strip() == entry or piece.strip().startswith(entry)
                    for piece in selectors.split(",")
                )
            ),
            None,
        )
        assert body is not None, f"{label}: no `{entry}` rule draws its ring"
        ring = re.search(r"outline:\s*(\d+)px\s+solid\s+([^;]+);", body)
        assert ring and int(ring.group(1)) >= 2, f"{label}: the focus rule draws no visible ring"
        assert ring.group(2).strip() != "transparent", f"{label}: the focus ring is transparent"
    # A later `outline: none` on a focus state would erase the ring the grouped
    # rule draws, so the sheet may not carry one.
    for selectors, body in focus_rules:
        assert not re.search(r"outline:\s*(0px|none)\b", body), (
            f"{selectors}: this focus rule erases the ring instead of drawing one"
        )


def test_the_reduced_motion_block_silences_every_transition():
    # Every family the sheet animates must be switched off inside the ONE
    # `prefers-reduced-motion` block: a transition added outside it keeps moving
    # for a reader who asked it not to. The browser test reads the same block;
    # this is the copy that runs without Chrome.
    block = _web_media("@media (prefers-reduced-motion: reduce)")
    assert re.search(r"transition:\s*none", block), (
        "the reduced-motion block no longer removes the transitions"
    )
    silenced = {
        piece.strip() for selectors, _ in _web_blocks(block) for piece in selectors.split(",")
    }

    def animates(body: str) -> bool:
        """True when a block declares a transition that actually runs."""
        values = re.findall(r"transition:\s*([^;]+);", body)
        return any(value.strip() not in {"none", "0s"} for value in values)

    animated = {
        piece.strip()
        for selectors, body in _web_blocks(_web_sheet())
        if animates(body)
        for piece in selectors.split(",")
    }
    assert {"button", ".link", ".meter-fill"} <= animated, (
        "the control transitions are gone: the block would silence nothing"
    )
    missing = animated - silenced
    assert not missing, f"these transitions keep moving under reduced motion: {sorted(missing)}"


def test_every_control_boundary_is_visible_in_both_palettes():
    # axe decides text contrast and never non-text contrast (1.4.11), so the
    # page's control boundaries are measured here instead: the analysis-rate
    # check that only exists in the browser would miss a border that vanishes
    # into the paper. The light palette's warm rule read 1.72:1 and the
    # masthead pills 2.21:1 — under the 3:1 floor — which is why the controls
    # have their own token instead of borrowing the decorative one.
    for theme, selector in (("light", ":root"), ("dark", '[data-theme="dark"]')):
        tokens = _web_tokens(selector)
        for foreground, background in WEB_CONTROL_EDGES:
            ratio = _web_contrast(tokens[foreground], tokens[background])
            assert ratio >= 3.0, f"{theme}: {foreground} on {background} reads {ratio:.2f}:1"


def test_the_copper_focus_ring_reads_on_the_paper_in_both_palettes():
    # The ring is 2 px of `--copper` drawn OUTSIDE the control, so on the page
    # it sits on `--paper` rather than on the control's fill. WCAG 1.4.11 asks
    # non-text indicators for 3:1, and the existing pair clears it with room:
    # 3.74:1 in the light palette, 8.19:1 in the dark one. The colours are
    # pinned, never adjusted — a future palette edit must keep the ring visible.
    for theme, selector in (("light", ":root"), ("dark", '[data-theme="dark"]')):
        tokens = _web_tokens(selector)
        ratio = _web_contrast(tokens["copper"], tokens["paper"])
        assert ratio >= 3.0, f"{theme}: the focus ring reads {ratio:.2f}:1 on the paper"


def test_secondary_controls_are_fills_not_strokes():
    # The 1 px tan line was a boundary drawn ON TOP of a control that already had
    # a shape; a shaded field is what a printed form does, and it costs no line.
    # The floors then argue with each other: 3:1 against the paper pins the fill
    # to a 4 % luminance band, and the label needs 4.5:1 ON that fill, so the
    # hover can move exactly one step. Each pair is measured below, in both
    # palettes — pasting the old fill back is what fails first.
    root = _web_tokens(":root")
    for theme, selector in (("light", ":root"), ("dark", '[data-theme="dark"]')):
        # the masthead pills sit on the night band in BOTH palettes, so their
        # fills live once, in `:root`
        tokens = {**root, **_web_tokens(selector)}
        for fill, surface in (
            ("button", "paper"),
            ("button-hover", "paper"),
            ("link-fill", "night"),
            ("link-fill-hover", "night"),
        ):
            ratio = _web_contrast(tokens[fill], tokens[surface])
            assert ratio >= 3.0, f"{theme}: {fill} on {surface} reads {ratio:.2f}:1"
        for label, fill in (
            ("ink", "button"),
            ("ink", "button-hover"),
            ("ivory", "link-fill"),
            ("ivory", "link-fill-hover"),
        ):
            ratio = _web_contrast(tokens[label], tokens[fill])
            assert ratio >= 4.5, f"{theme}: {label} on {fill} reads {ratio:.2f}:1"

    # and the stroke is GONE, not merely quiet: a border would draw a second
    # boundary over the fill.
    for selector in ("button", ".link"):
        rule = _web_rule(selector)
        assert re.search(r"border:\s*0", rule), f"{selector} draws its boundary with a stroke again"


def test_a_disabled_field_keeps_its_label_readable():
    # Fading the WHOLE control to 45 % took the label with it — measured 1.71:1
    # over the washed field — while the field itself stopped reading as a field,
    # so an unavailable action looked like a smudge instead of saying "not now". The
    # inactive state is therefore its own pair of tokens: a quiet surface
    # (1.17:1 on the paper, and legitimately so — WCAG 1.4.11 exempts inactive
    # components, so the boundary floor is NOT the one to hold here) carrying a
    # label held at the same 4.5:1 as every other word on the page.
    rule = _web_rule("button:disabled")
    assert "var(--button-off)" in rule, "the disabled field is a fade again, not a surface"
    assert "opacity" not in rule, "the disabled control fades its own label again"
    for theme, selector in (("light", ":root"), ("dark", '[data-theme="dark"]')):
        tokens = _web_tokens(selector)
        ratio = _web_contrast(tokens["button-off-ink"], tokens["button-off"])
        assert ratio >= 4.5, f"{theme}: the disabled label reads {ratio:.2f}:1 on its field"
    # and the two states must remain tellable apart: an inactive field that kept
    # the enabled fill would look pressable.
    root = _web_tokens(":root")
    dark = _web_tokens('[data-theme="dark"]')
    assert root["button-off"] != root["button"], "the light disabled field is the enabled one"
    assert dark["button-off"] != dark["button"], "the dark disabled field is the enabled one"


def test_the_two_text_links_disable_through_the_button_rule():
    # `#balance-reset` and the tutorial's `#guide-demo` are buttons painted as
    # text links. `button:disabled` already outranks either class's own ink and
    # background, so a per-class copy of the pair would be a second declaration
    # to keep in sync; what must hold is that they stay buttons and their base
    # rules do not fade the word themselves.
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for control_id in ("balance-reset", "guide-demo"):
        assert re.search(rf'<button[^>]*id="{control_id}"[^>]*>', html), (
            f"#{control_id} is no longer a button, so button:disabled no longer paints it"
        )
    rule = _web_rule("button:disabled")
    assert "var(--button-off)" in rule, "the disabled field is a fade again, not a surface"
    assert "var(--button-off-ink)" in rule, "the disabled label lost its readable ink"
    assert "opacity" not in rule, "the disabled control fades its own label again"
    assert "transition: none" in rule, "a disabled control can be caught mid-change"
    for selector in (".balance-reset", ".guide-link"):
        assert "opacity" not in _web_rule(selector), f"{selector} fades its word again"
    # The underline is the promise of a click, and an inactive control makes none.
    # ONE shared rule drops it for both links, and it carries the line ALONE: the
    # field and the word stay `button:disabled`'s pair, so no contrast moves.
    dropping = [
        (selectors, body)
        for selectors, body in _web_blocks(_web_sheet())
        if ".balance-reset:disabled" in selectors and ".guide-link:disabled" in selectors
    ]
    assert len(dropping) == 1, "the two text links no longer drop their underline in one rule"
    declarations = {
        part.split(":", 1)[0].strip() for part in dropping[0][1].split(";") if ":" in part
    }
    assert declarations == {"text-decoration"}, (
        f"the disabled text link repaints more than the line: {declarations}"
    )
    for selector in (".balance-reset", ".guide-link"):
        assert "text-decoration: underline" in _web_rule(selector), (
            f"{selector} lost the underline its ENABLED state acts through"
        )
    # The pair's own 4.5:1 is measured ONCE, on the shared rule, by
    # `test_a_disabled_field_keeps_its_label_readable`: the links add nothing
    # to that pair, and a second copy of one measurement is a second place to
    # keep in step.


def test_the_playing_row_never_takes_the_page_away_from_the_roll():
    # `scrollIntoView` on the playing row scrolls every ancestor, the DOCUMENT
    # included, so the page would leave the roll and its playhead to show a row
    # the reader was not looking at. The page asks `playingRowScroll` first
    # (pure, in `web/playhead.js`, measured by `web/test/table-follow.test.mjs`)
    # and scrolls by the gap it answers, so the table's own sideways scroll
    # survives — and the direction is not rebuilt at the call site.
    code = re.sub(
        r"/\*.*?\*/|//[^\n]*", "", (WEB / "app.js").read_text(encoding="utf-8"), flags=re.S
    )
    assert "scrollIntoView" not in code, "the playing row scrolls the page without asking"
    assert "playingRowScroll" in code, "the playing row no longer asks before moving the page"
    assert re.search(
        r'import \{[^}]*playingRowScroll[^}]*\} from "\./playhead\.js"', code, re.S
    ), "the follow policy does not come from the page's own view module"
    assert "window.scrollBy" in code, "the follow moves more than the vertical gap"
    # The pure function answers the GAP, sign included; rebuilding the direction
    # here was the second owner of the rule (`box.bottom <= 0 ? … : …`).
    assert "box.bottom - window.innerHeight" not in code, (
        "the page recomputes the gap instead of scrolling by the one it was given"
    )
    assert "box.bottom <= 0" not in code, "the page still decides the direction itself"
    assert "export function playingRowScroll" in (WEB / "playhead.js").read_text(
        encoding="utf-8"
    ), "the follow policy left the module that owns the view's decisions"


def test_the_tutorial_opens_the_demo_from_its_first_step():
    # Step one says "sing" — which a newcomer cannot do before hearing anything,
    # and the demo button lives a whole section below. So the step carries its
    # own entrance to the SAME gesture, and closes the sheet on the way: the
    # reading the demo produces would otherwise appear under a folded tutorial.
    # What it must NOT do is sound on its own (no gesture, no audio — browsers
    # refuse it and the singer never asked) or silence a take that is already
    # playing, since a control labelled "hear an example" must not stop one.
    html = (WEB / "index.html").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    guide = html[html.index('class="guide"') : html.index("</details>")]
    assert 'id="guide-demo"' in guide, "the tutorial lost its way into the demo"
    assert guide.index('data-i18n="guide_1"') < guide.index('id="guide-demo"') < guide.index(
        'data-i18n="guide_2"'
    ), "the entrance left the first step"
    assert '<details class="guide">' in html, (
        "the sheet starts open: a first visit would meet the steps before the page"
    )

    assert '$("guide-demo").addEventListener("click", hearExample)' in app, (
        "the entrance is not wired to its handler"
    )
    # ONE call site and one definition: a demo that can start without a click is
    # exactly the autoplay this control must not be.
    assert app.count("hearExample") == 2, "the demo can start without a click"
    handler = app[app.index("async function hearExample(") :]
    handler = handler[: handler.index("\n}\n")]
    assert "details.guide\").open = false" in handler, "the sheet stays open over the example"
    assert "toggleDemo()" in handler, "the entrance does not start the same demo"
    assert "if (looping) return" in handler, "a second click would stop a take that is sounding"

    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    for language in ("fr", "en"):
        label = strings[language].get("guide_demo")
        assert label, f"{language}: guide_demo is missing"
        assert label != strings[language]["demo"], (
            f"{language}: the step repeats the transport button's own label"
        )
    rule = _web_rule(".guide-link")
    assert re.search(r"min-height:\s*24px", rule), "the step's link is not a 24 px pointer target"


def test_the_numbers_in_the_notes_table_line_up_on_the_right_on_a_wide_page():
    # At the wide measure the six columns are ~170 px each: a figure pinned to
    # the left edge of such a column reads as belonging to its NEIGHBOUR's
    # header, so the numeric columns (row, start, duration, cents) and their
    # headers are set flush right — two start times are then compared by their
    # last digit. The note's name and the doubt word keep the reading edge.
    css = (WEB / "style.css").read_text(encoding="utf-8")
    start = css.index("th:nth-child(1),")
    wide_at = css.rfind("@media (min-width: 1000px)", 0, start)
    assert wide_at != -1, "the alignment left the wide page"
    assert "}" not in css[wide_at:start], "the alignment is outside the wide query"
    selectors, declarations = css[start : css.index("}", start)].split("{", 1)
    assert "text-align: right" in declarations, "the numbers are not set flush right"
    for column in (1, 2, 3, 5):
        assert f"td:nth-child({column})" in selectors, f"column {column} still floats in its cell"
        assert f"th:nth-child({column})" in selectors, (
            f"column {column}'s header does not follow it"
        )
    for text_column in (4, 6):
        assert f"nth-child({text_column})" not in selectors, (
            "a text column was set flush right: a note name reads from its left"
        )


def test_the_page_prints_as_a_sheet_with_its_tutorial():
    # The paper version is taken beside the microphone, where nothing can be
    # clicked: so it carries the tutorial (which the screen keeps folded, hence
    # the printer opens it) and the take's notes, drops every control, and forces
    # a paper palette — a night page printed as it stands lays down a dark sheet.
    css = (WEB / "style.css").read_text(encoding="utf-8")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    printed = css[css.index("@media print {") :]
    printed = printed[: printed.index("\n}\n")]

    assert "--paper: #fff" in printed and "--ink: #000" in printed, (
        "the sheet inherits the screen palette"
    )
    assert '@page' in printed, "the sheet keeps the window's margins"
    controls = (".take", ".notes-actions", ".roll-section", ".readout", ".hints", ".guide-link")
    for gone in controls:
        assert re.search(rf"^\s+{re.escape(gone)},?$", printed, re.M), (
            f"{gone} would print as a dead control"
        )
    assert "display: table-header-group" in printed, (
        "the notes header is not repeated on every page the table spans"
    )
    assert re.search(r"\.guide-steps li,\n\s+tr \{\n\s+break-inside: avoid", printed), (
        "a step or a note row can be split by a page break"
    )

    assert 'id="print"' in html and 'data-i18n="guide_print"' in html
    assert '$("print").addEventListener("click", () => window.print())' in app
    # The printer has no pointer: the sheet is opened around the print and put
    # back as the reader left it (a cancelled dialog fires `afterprint` too).
    handler = app[app.index("function listenForPrint(") :]
    handler = handler[: handler.index("\n}\n")]
    assert 'addEventListener("beforeprint"' in handler and "guide.open = true" in handler
    assert 'addEventListener("afterprint"' in handler, "the sheet stays open after printing"
    assert "guideWasOpen = guide.open" in handler, "the reader's folded sheet is not restored"
    assert "guide.open = guideWasOpen" in handler

    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    for language in ("fr", "en"):
        assert strings[language].get("guide_print"), f"{language}: guide_print is missing"


def test_the_page_paints_its_words_before_any_script_runs():
    # Firefox's first paint — and any browser whose strings.json is still in
    # flight, or blocked by a proxy — showed a page of empty headings: every
    # word came from a fetch that starts after ten modules load, and a
    # screenshot taken at that moment is a page with no text at all. The French
    # copy now sits in the markup as the default and `applyLanguage` overwrites
    # it, so the page reads with no script and before the fetch resolves. The
    # two copies are compared here, because a default that drifts from the table
    # is worse than no default.
    html = (WEB / "index.html").read_text(encoding="utf-8")
    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    elements = re.findall(r'<(\w+)([^>]*\sdata-i18n="([a-z0-9_]+)"[^>]*)>([^<]*)</\1>', html)
    assert len(elements) >= 30, (
        f"only {len(elements)} of the page's words survive without the text fetch"
    )
    for _tag, _attrs, key, text in elements:
        assert text.strip(), f"{key}: the element paints empty until strings.json arrives"
        assert text.strip() == strings["fr"][key].strip(), (
            f"{key}: the markup and the French table disagree"
        )


def test_the_page_adapts_to_every_resolution_class():
    # Four queries carry the whole layout: a phone (<= 420: labels stack), a
    # small screen (< 620: the pills leave the title's corner), the wide page
    # (>= 1000: a wider measure), and a phone held SIDEWAYS — a HEIGHT query,
    # because the widest phone is the landscape one: at 812 x 375 the masthead
    # took 190 px of a 375 px window, half the screen before the first control.
    # A later edit that drops one of these changes nothing on a desk and
    # everything on the device the page is actually used from.
    css = (WEB / "style.css").read_text(encoding="utf-8")
    for query in (
        "@media (max-width: 420px)",
        "@media (max-width: 620px)",
        "@media (min-width: 1000px)",
        "@media (max-height: 480px)",
    ):
        assert query in css, f"the page no longer adapts to {query}"

    short = re.search(r"@media \(max-height: 480px\) \{(.*?)\n\}", css, re.S)
    assert short, "a short window has no layout of its own"
    masthead = re.search(r"\.masthead \{(.*?)\}", short.group(1), re.S)
    assert masthead, "the short window no longer trims the masthead"
    padding = re.search(r"padding:\s*(\d+)px 0", masthead.group(1))
    assert padding and int(padding.group(1)) <= 30, (
        "the masthead still eats half a landscape phone before the first control"
    )



def test_a_wide_window_gets_a_wider_measure_without_longer_lines():
    # Held at one 720 px column, the page is a ribbon on a 1600 px screen: the
    # masthead's rule and the notes row float in empty paper. So the measure
    # grows above 1000 px — and ONLY the measure: the prose keeps its `ch`
    # limits, or a widened page would turn a 14 px paragraph into a 1040 px
    # line nobody can track back to its start.
    css = (WEB / "style.css").read_text(encoding="utf-8")
    narrow = re.search(r"--measure:\s*(\d+)px", css)
    wide = re.search(r"--measure-wide:\s*(\d+)px", css)
    assert narrow and wide, "the page no longer declares its two measures"
    assert int(wide.group(1)) > int(narrow.group(1)), "the wide measure is not wider"

    query = re.search(r"@media \(min-width: 1000px\) \{(.+?)\n\}", css, re.S)
    assert query, "the wide window has no layout of its own"
    assert ".wrap" in query.group(1) and "var(--measure-wide)" in query.group(1), (
        "the wide breakpoint no longer widens the column"
    )

    for block, limit in (
        (".deck", "46ch"),
        (".fineprint", "66ch"),
        (".guide-steps", "64ch"),
    ):
        rule = re.search(rf"{re.escape(block)} \{{(.*?)\n\}}", css, re.S)
        assert rule and f"max-width: {limit}" in rule.group(1), (
            f"{block} lost its {limit} measure: prose would stretch with the window"
        )

    # Two pieces of the page were only ever drawn at 720 px. The level meter
    # became a 900 px hairline under an empty readout (it measures level, it is
    # not a rule), and the tutorial hugged a quarter of the column it shares.
    wide_blocks = re.findall(r"@media \(min-width: 1000px\) \{(.*?)\n\}", css, re.S)
    assert any(".guide-steps" in block and "columns: 2" in block for block in wide_blocks), (
        "the tutorial no longer uses the width of the wide page"
    )
    assert any(
        re.search(r"\.meter \{[^}]*max-width: \d+px", block) for block in wide_blocks
    ), "the meter is uncapped on the wide page: a gauge stretched to the column reads as a rule"


def test_the_page_opens_on_a_complete_tutorial():
    # "Simple but complete" is the whole requirement: a reader who has never
    # seen the page must know what to press first, how to read the numbers, how
    # to hear the result and how to take it away — without opening anything
    # else. So the steps are pinned in ORDER and in BOTH languages, and the
    # block must stay ABOVE the controls it describes.
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert "<details class=\"guide\">" in html, "the page lost its tutorial"
    assert "<summary data-i18n=\"guide_title\">" in html
    assert html.index('class="guide"') < html.index('id="record"'), (
        "the tutorial slipped below the buttons it explains"
    )
    # Each sentence is held by its OWN element: `applyLanguage` replaces the text
    # of the tagged node, so the label of the demo entrance beside step one must
    # not live inside it — the step's text is a span, not the whole `li`.
    steps = [f'data-i18n="guide_{step}"' for step in range(1, 6)]
    positions = [html.index(step) for step in steps]
    assert positions == sorted(positions), "the tutorial's steps are out of order"
    assert '<p class="fineprint" data-i18n="guide_note">' in html

    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    for language in ("fr", "en"):
        for key in ("guide_title", "guide_note", *(f"guide_{i}" for i in range(1, 6))):
            assert strings[language].get(key), f"{language}: {key} is missing or empty"
    # A tutorial nobody can find is a tutorial nobody reads: the summary names
    # the steps, so a reader knows what opening it costs.
    assert "cinq" in strings["fr"]["guide_title"]
    assert "five" in strings["en"]["guide_title"]


def test_the_notes_scroll_region_is_reachable_by_keyboard():
    # axe's `scrollable-region-focusable`: a box that scrolls sideways must be in
    # the TAB CHAIN, or a keyboard user simply cannot see the columns past the
    # edge — the notes list is the page's only scrollable region, and at 282 CSS
    # px it overflows by 5 px, which is enough to hide "cents" and "voice".
    html = (WEB / "index.html").read_text(encoding="utf-8")
    region = re.search(r'<div id="notes-table" class="table-scroll"([^>]*)>', html)
    assert region and 'tabindex="0"' in region.group(1), (
        "the notes list scrolls out of keyboard reach again"
    )
    rule = _web_rule(".table-scroll:focus-visible")
    assert "outline" in rule and "var(--copper)" in rule, (
        "the scroll region has no focus ring: the arrows would move an unnamed box"
    )


def test_every_web_control_offers_a_24_px_target():
    # WCAG 2.2 AA (2.5.8): a pointer target is at least 24 x 24 CSS px. The
    # balance slider drew its 6 px track AS its own box and the reset link was
    # 18 px of underlined text — mouse-friendly, impossibly thin for a thumb.
    slider = _web_rule('.balance-row input[type="range"]')
    assert re.search(r"height:\s*24px", slider), "the balance slider is not a 24 px target"
    assert "background: transparent" in slider, "the track must move inside the target, not fill it"
    for track in ('::-webkit-slider-runnable-track', '::-moz-range-track'):
        assert re.search(r"height:\s*6px", _web_rule(f'.balance-row input[type="range"]{track}'))
    reset = _web_rule(".balance-reset")
    assert re.search(r"min-height:\s*24px", reset), "the reset link is not a 24 px target"


def test_the_dark_theme_repaints_every_colour_token():
    # Every colour the light sheet declares must be overridden by the dark one:
    # a token left behind paints one warm patch on a night page, and nothing
    # fails — it simply looks broken in the theme nobody re-checks.
    css = (WEB / "style.css").read_text(encoding="utf-8")
    light = re.search(r":root \{(.*?)\n\}", css, re.S)
    dark = re.search(r'\[data-theme="dark"\] \{(.*?)\n\}', css, re.S)
    assert light and dark, "both palettes must be declared explicitly"
    fonts = {"--serif", "--sans"}  # a theme repaints colours, not typefaces
    # The masthead pills answer to the night BAND, which both palettes keep
    # dark: one fill clears 3:1 on each (3.04 on #131a27, 3.50 on #0b111c), so
    # a dark copy would be a second value to drift, not a repaint.
    band_only = {"--link-fill", "--link-fill-hover"}

    def declared(block: str) -> set[str]:
        kept = fonts | band_only
        return {token for token in re.findall(r"(--[a-z-]+):", block) if token not in kept}

    missing = declared(light.group(1)) - declared(dark.group(1))
    assert not missing, f"the dark theme leaves these to the light one: {sorted(missing)}"


def test_the_theme_is_painted_before_the_first_frame_and_remembered():
    # The page module is deferred, so a stored dark page would flash its light
    # paper while the strings are fetched: the head script must set the
    # attribute itself, from the SAME key and the same two names. It is a file
    # of its own, which is what lets the page's policy carry no inline
    # exception — the head must LOAD it, and it must still say all three things.
    html = (WEB / "index.html").read_text(encoding="utf-8")
    head = html.split("<body", 1)[0]
    assert '<script src="theme-init.js"></script>' in head
    init = (WEB / "theme-init.js").read_text(encoding="utf-8")
    assert "dataset.theme" in init
    assert 'localStorage.getItem("v2m_theme")' in init
    assert "prefers-color-scheme: dark" in init
    app = (WEB / "app.js").read_text(encoding="utf-8")
    prefs = (WEB / "prefs.js").read_text(encoding="utf-8")
    assert 'THEME_KEY = "v2m_theme"' in prefs  # the one key, one spelling
    assert "themeFor({" in app  # the module re-derives the SAME decision
    assert "dataset.theme = theme" in app
    assert "localStorage.setItem(THEME_KEY" in app  # the pill remembers the choice
    # ... and the roll reads the palette instead of carrying its own: a canvas
    # with hardcoded copper stays light in a dark page.
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert "--roll-bg" in css
    assert "getPropertyValue(name)" in app
    for token in ("--roll-bar", "--roll-playhead", "--roll-lane"):
        assert f'read("{token}")' in app, f"the roll no longer reads {token}"


def test_the_summary_names_the_same_habits_as_the_desktop():
    # A group the desktop computes but the page drops is a habit the singer
    # never sees; a group the page invents is a claim the desktop does not
    # make. Both splits, and the register names, must be the SAME ones.
    from v2m.core import intonation

    source = (WEB / "intonation.js").read_text(encoding="utf-8")
    for field in ("byRegister", "byPosition"):
        assert field in source, f"web/intonation.js no longer carries {field}"
    for band in intonation.BANDS:
        assert f'"{band}"' in source, f"web/intonation.js lost the {band} band"
    for half in ("start", "end"):
        assert f"{half}: tendencyOf" in source, f"web/intonation.js lost the {half} half"
    # The page's app.js must RENDER both lines, or the summary is computed and
    # thrown away.
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'renderLeanLine(\n    "registers"' in app
    assert 'renderLeanLine(\n    "position"' in app


def test_the_conformance_cases_are_what_the_python_segmenter_decides():
    from scripts.export_web_fixtures import build_cases, piano_reference, render

    regenerate = (
        "web/test/cases.json is stale: run "
        "`./.venv/Scripts/python.exe -m scripts.export_web_fixtures` and then "
        "`node --test web/test/segmenter.test.mjs`"
    )
    checked_in = json.loads((WEB / "test" / "cases.json").read_text(encoding="utf-8"))
    assert render(build_cases())["cases"] == checked_in["cases"], regenerate

    # The piano digest is compared with a tolerance: it is the SAME numpy
    # render, but a different summation order in a rebuild must not make the
    # pin cry wolf.
    fresh = piano_reference()
    checked = checked_in["piano"]
    for field in ("sr", "stride", "length", "notes"):
        assert fresh[field] == checked[field], regenerate
    for field in ("peak", "rms"):
        assert fresh[field] == pytest.approx(checked[field]), regenerate
    assert fresh["samples"] == pytest.approx(checked["samples"], abs=1e-9), regenerate


def test_the_intonation_fixture_is_what_the_python_summarizer_decides():
    # `web/intonation.js` is the one port with no conformity proof of its own:
    # it decides a VISIBLE line out of thresholds, and no other check compares
    # it to anything. The fixture is the desktop's own summary over the traps
    # (fewer than MIN_NOTES notes, a mean under MIN_CENTS, identically-pitched
    # notes, an odd count, both inclusive bars, an empty take), replayed by
    # `node --test web/test/intonation.test.mjs`.
    from scripts.export_web_fixtures import INTONATION_CASES, intonation_reference

    regenerate = (
        "web/test/intonation.json is stale: run "
        "`./.venv/Scripts/python.exe -m scripts.export_web_fixtures`"
    )
    checked_in = json.loads((WEB / "test" / "intonation.json").read_text(encoding="utf-8"))
    assert intonation_reference() == checked_in, regenerate
    for name in (
        "fewer_than_four_notes_never_leans",
        "a_mean_under_twelve_cents_is_not_a_habit",
        "identical_pitches_divide_no_register",
        "an_odd_count_gives_its_extra_note_to_the_end",
    ):
        assert name in INTONATION_CASES, f"{regenerate}: the {name} trap left the fixture"


def _yin_spans_the_take(excerpts) -> bool:
    """Whether the frozen frames still reach both ends of the take's range.

    Cutting the take down to excerpts is only worth it if the RANGE survives: a
    head-only crop loses the low pedal (the lag search's limit), a tail-only crop
    loses the octave above it — and both would pass every per-frame bound.
    """
    from v2m.core.pitch import FMIN

    voiced = [hz for excerpt in excerpts for _t, hz, _q in excerpt["frames"] if hz is not None]
    return bool(voiced) and min(voiced) <= FMIN * 1.05 and max(voiced) >= 500.0


def test_the_browser_yin_reads_the_desktop_frames_on_the_demo_take():
    # The two estimators share a formula and nothing else: numpy's rfft against
    # a radix-2 FFT. `web/test/yin.json` holds what `core/pitch.py` reads over
    # two excerpts of the demo take (the ascent, every scale degree, and the tail
    # with its low pedal), and the Node test recomputes every frame from the same
    # samples. Measured: 0.000013 cents and 5e-7 in d' over 470 frames, so the
    # bounds it pins (0.5 cent, 0.01) have margin for float noise but none for
    # drift: one lag sample at 440 Hz is about 40 cents.
    from scripts.export_web_fixtures import yin_reference

    if not (REPO / "demo.wav").exists():
        pytest.skip("demo.wav is absent: run `python scripts/make_demo.py`")
    fresh = yin_reference()
    assert fresh is not None, "demo.wav exists but the reference came back empty"
    checked = json.loads((WEB / "test" / "yin.json").read_text(encoding="utf-8"))
    assert fresh["source"] == checked["source"], "web/test/yin.json was built from another take"
    assert fresh["geometry"] == checked["geometry"]
    assert fresh["excerpts"] == checked["excerpts"], (
        "web/test/yin.json is stale: an excerpt or one of its frames moved"
    )

    # The fixture is a CUT of the take now, so the coverage it keeps is
    # asserted rather than assumed: both ends of the range, and a second excerpt
    # (a head-only fixture puts the pedal — the low-lag limit — out of reach).
    assert _yin_spans_the_take(checked["excerpts"]), "the excerpts lost an end of the take's range"
    assert any(excerpt["start"] > 0 for excerpt in checked["excerpts"]), (
        "every excerpt reads from sample 0: the tail and its pedal are gone"
    )
    names = [excerpt["name"] for excerpt in checked["excerpts"]]
    assert len(set(names)) == len(names), f"two excerpts share a name: {names}"

    node_test = (WEB / "test" / "yin.test.mjs").read_text(encoding="utf-8")
    assert "const MAX_CENTS = 0.5;" in node_test, "the browser's pitch bound was widened"
    assert "const MAX_QUALITY = 0.01;" in node_test, "the browser's d' bound was widened"


def test_the_excerpt_coverage_guard_bites():
    """A head-only or tail-only crop must fail the range check."""

    def crop(*pitches):
        frames = [[0.0, hz, 0.1] for hz in pitches]
        return [{"name": "crop", "start": 0, "samples": 0, "frames": frames}]

    assert _yin_spans_the_take(crop(65.0, 259.0, 523.0))
    assert not _yin_spans_the_take(crop(259.0, 330.0, 523.0)), "a head-only crop lost the pedal"
    assert not _yin_spans_the_take(crop(65.0, 259.0, 330.0)), "a tail-only crop lost the top"


def _reaches_the_low_lag_edge(rate: dict, sr: int) -> bool:
    """Whether some frozen reading sits at the lag search's own maximum.

    Measured, not assumed: a fixture built from a mid-register take answers
    False — the state this excerpt replaced, where `tau_max` was frozen and
    never exercised.
    """
    periods = [sr / hz for _t, hz, _q in rate["frames"] if hz is not None]
    return bool(periods) and max(periods) >= 0.99 * rate["tau_max"]


def _keeps_the_tonal_head(rate: dict) -> bool:
    """Whether the excerpt still carries the lag JUMP, not just a short reading.

    The window is [tonic][pedal], so what it must pin is the CROSSING: a short
    lag (the tonic, > 100 Hz) and then, later in the same window, a reading at
    the pedal's long one. That order is where a port that cached its lag span,
    or that carried a pitch across a note boundary, shows itself — a window with
    a short reading somewhere and no long one after it proves neither. Unvoiced
    frames between the two are allowed: an attack is silent.
    """
    from v2m.core.pitch import FMIN

    seen_short = False
    for _t, hz, _q in rate["frames"]:
        if hz is None:
            continue
        if hz <= FMIN * 1.05:
            if seen_short:
                return True
        elif hz > 100.0:
            seen_short = True
    return False


def test_the_browser_yin_keeps_the_capture_rates_geometry():
    # The analysis rate's fixture pins the arithmetic at a 512/256 frame. A
    # device never runs there: the live settings are DURATIONS, so 44.1 kHz
    # opens 1024/512 and 48 kHz 1115/557 — and 1115 is NOT a power of two, so
    # the difference function's autocorrelation is zero-padded to 4096. A port
    # that kept the analysis-rate window, or that assumed a power-of-two frame,
    # passes the 22 050 Hz comparison and misreads every real capture, so
    # `web/test/yin-rates.json` freezes both geometries with their own frames
    # (read by `web/test/yin-rates.test.mjs`). Measured: 0.000013 cents and 5e-7
    # in d' at both rates, while the analysis rate's window at 48 kHz misses by
    # 8.7 cents and the frame count moves at 44.1 kHz.
    from scripts.export_web_fixtures import CAPTURE_RATES, yin_rates_reference
    from v2m.core.liveseg import LivePitchStream

    if not (REPO / "demo.wav").exists():
        pytest.skip("demo.wav is absent: run `python scripts/make_demo.py`")
    fresh = yin_rates_reference()
    assert fresh is not None, "demo.wav exists but the capture-rate reference came back empty"
    checked = json.loads((WEB / "test" / "yin-rates.json").read_text(encoding="utf-8"))
    assert fresh["source"] == checked["source"], (
        "web/test/yin-rates.json was built from another take"
    )
    assert [rate["sr"] for rate in checked["rates"]] == list(CAPTURE_RATES)

    by_rate = {rate["sr"]: rate for rate in checked["rates"]}
    for sr, frozen in by_rate.items():
        mine = next(rate for rate in fresh["rates"] if rate["sr"] == sr)
        assert {k: v for k, v in frozen.items() if k != "frames"} == {
            k: v for k, v in mine.items() if k != "frames"
        }, f"{sr} Hz: the frozen geometry moved"
        assert len(mine["frames"]) == len(frozen["frames"]), f"{sr} Hz: the frame count moved"
        pairs = zip(mine["frames"], frozen["frames"], strict=True)
        for index, (latest, pinned) in enumerate(pairs):
            assert abs(latest[0] - pinned[0]) < 1e-6, f"{sr} Hz frame {index}: the clock moved"
            assert (latest[1] is None) == (pinned[1] is None), (
                f"{sr} Hz frame {index}: the voicing moved"
            )
            if latest[1] is not None and pinned[1] is not None:
                assert abs(latest[1] - pinned[1]) < 1e-4, f"{sr} Hz frame {index}: the pitch moved"
            gap = abs((latest[2] or 0.0) - (pinned[2] or 0.0))
            assert gap < 1e-6, f"{sr} Hz frame {index}: d' moved"

    # The geometry is not re-derived here: the live stream owns that conversion,
    # so a changed rounding rule fails the frozen numbers instead of agreeing
    # with a second copy of the formula.
    for sr in CAPTURE_RATES:
        stream = LivePitchStream(sr, gate_rms=0.0)
        assert (by_rate[sr]["window"], by_rate[sr]["hop"]) == (stream.window, stream.hop)
    # 44.1 kHz lands on a power of two by coincidence, which is exactly why the
    # padding path needs 48 kHz behind it.
    assert by_rate[44_100]["window"] == 1024
    assert by_rate[48_000]["window"] == 1115
    assert by_rate[48_000]["window"] & (by_rate[48_000]["window"] - 1), (
        "the 48 kHz window became a power of two: no rate exercises the FFT padding"
    )
    assert by_rate[48_000]["fft_size"] == 4096
    assert by_rate[44_100]["fft_size"] == 2048

    # The lag search's LOW edge is what this excerpt exists for. A C4..C5
    # melody never reaches it; the take's final C2 pedal does, and its vibrato
    # dips under FMIN, so some frames read exactly `sr / tau_max`. An excerpt
    # back at the take's head (or a shorter one, or a take without the pedal)
    # would leave `tau_max` frozen but unexercised while every check above
    # still passed.
    from v2m.core.pitch import FMIN

    assert checked["source"]["excerpt_start"] > 0, "the excerpt no longer reads the take's tail"
    for sr, frozen in by_rate.items():
        assert _reaches_the_low_lag_edge(frozen, sr), (
            f"{sr} Hz: the excerpt never reaches the low-lag edge (tau_max)"
        )
        assert _keeps_the_tonal_head(frozen), (
            f"{sr} Hz: no short-then-long crossing — the lag jump is outside the window"
        )
        assert abs(sr / frozen["tau_max"] - FMIN) / FMIN < 0.01, (
            f"{sr} Hz: tau_max is no longer the FMIN limit"
        )

    node_test = (WEB / "test" / "yin-rates.test.mjs").read_text(encoding="utf-8")
    assert "const MAX_CENTS = 0.5;" in node_test, "the browser's pitch bound was widened"
    assert "const MAX_QUALITY = 0.01;" in node_test, "the browser's d' bound was widened"


def test_the_low_lag_guard_bites():
    """A mid-register excerpt must fail the guard the C2 pedal passes."""
    mid_register = {"tau_max": 678, "frames": [[0.0, 261.63, 0.1], [0.0116, 293.66, 0.1]]}
    assert not _reaches_the_low_lag_edge(mid_register, 44_100)
    with_pedal = {"tau_max": 678, "frames": [[0.0, 65.04, 0.1]]}
    assert _reaches_the_low_lag_edge(with_pedal, 44_100)


def test_the_tonal_head_guard_bites():
    """Only the short-then-long CROSSING passes the head guard.

    Each of the three wrong windows is a crop this fixture could plausibly grow:
    the pedal alone, the tonic alone, and the two in the wrong order (a take that
    climbs instead of landing) — all three carry a reading, none of them pins the
    jump the capture-rate geometry needs.
    """
    pedal_only = {"frames": [[0.0, 65.04, 0.1], [0.0126, 65.04, 0.1], [0.025, None, None]]}
    assert not _keeps_the_tonal_head(pedal_only)
    tonic_only = {"frames": [[0.0, 264.6, 0.1], [0.0126, 263.9, 0.1], [0.025, None, None]]}
    assert not _keeps_the_tonal_head(tonic_only), "a short reading alone is not the jump"
    climbing = {"frames": [[0.0, 65.04, 0.1], [0.0126, 264.6, 0.1]]}
    assert not _keeps_the_tonal_head(climbing), "the pedal before the tonic is not the jump"
    # the real order, with the attack's unvoiced frame in between
    with_head = {"frames": [[0.0, 264.6, 0.1], [0.0126, None, None], [0.025, 65.04, 0.1]]}
    assert _keeps_the_tonal_head(with_head)


def test_the_demo_test_reads_the_chain_at_the_pages_own_gate():
    # The Node test feeds the demo signal through the real chain, so its gate
    # must be the page's. Both take it from the generated constants file; the
    # VALUE is pinned against audio_io.GATE_RMS by the one-source test, and what
    # is checked here is the import itself.
    imports = {
        "app.js": 'import { GATE_RMS } from "./constants.js";',
        "test/segmenter.test.mjs": 'import { GATE_RMS } from "../constants.js";',
    }
    for name, line in imports.items():
        body = (WEB / name).read_text(encoding="utf-8")
        assert line in body, f"web/{name} no longer takes GATE_RMS from constants.js"


def test_the_balance_is_remembered_between_visits():
    # The setting outlives the tab on purpose; a session store would silently
    # drop it every time the page is reopened.
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "localStorage" in app
    assert "sessionStorage" not in app


def test_the_language_and_the_listen_mode_are_remembered_between_visits():
    # Both are read before the first paint and written on every choice: a
    # preference stored only when something else happened is a preference the
    # singer loses.
    prefs = (WEB / "prefs.js").read_text(encoding="utf-8")
    assert '"v2m-web-lang"' in prefs
    assert '"v2m-web-mode"' in prefs
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "readStoredChoice(localStorage.getItem(LANG_KEY)" in app
    assert "readStoredChoice(localStorage.getItem(MODE_KEY)" in app
    assert "remember(LANG_KEY, lang)" in app
    assert "remember(MODE_KEY, mode)" in app
    # Both resume gestures (Space and a click in the roll) must ask the SAME
    # decision, or the remembered mode would resume on one and not the other.
    assert app.count("resumeMode({ stored: lastMode, hasVoice:") == 2


def test_the_balance_has_a_one_gesture_reset():
    # The reset must land on the SAME reference the page ships with, and be
    # remembered like a drag: a reset that only moved the pixels would come
    # back to the old value at the next visit.
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="balance-reset"' in html
    assert 'data-i18n="balance_reset"' in html
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert '$("balance-reset").addEventListener("click"' in app
    assert "balance = BALANCE_DEFAULT" in app
    assert "storeBalance()" in app


def test_a_playback_failure_does_not_blame_the_file():
    # The listen path's catch used the file reader's sentence, so a browser that
    # refused to start the audio told the singer "that file could not be read" —
    # a wrong diagnosis, and one that invites deleting a perfectly good take.
    # Each failure now has its own key, and each key has both languages: the
    # parity test below only catches a MISSING key, not a key reused for the
    # wrong failure.
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'setStatus("error_play");' in app, "a playback failure still has no message of its own"
    assert app.count('setStatus("error_file");') == 1, (
        "the file reader's sentence is used somewhere else than the file reader"
    )
    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    for language in ("fr", "en"):
        assert strings[language].get("error_play"), f"{language}: error_play is missing"
        assert strings[language]["error_play"] != strings[language]["error_file"], (
            f"{language}: the two failures share one sentence"
        )


def test_each_microphone_failure_names_the_reason_the_browser_gave():
    # One sentence for every capture failure sent the reader hunting for a
    # device when the permission had been refused, or for a permission when
    # another application held the microphone. Each of the browser's three
    # real reasons gets its own sentence; anything else came from the page's
    # own audio graph and keeps a neutral one, since it is not the mic's fault.
    app = (WEB / "app.js").read_text(encoding="utf-8")
    mapping = app[app.index("function micErrorKey(") :]
    mapping = mapping[: mapping.index("\n}\n")]
    for name, sentence in (
        ("NotAllowedError", "error_mic_blocked"),
        ("NotFoundError", "error_mic_missing"),
        ("NotReadableError", "error_mic_busy"),
    ):
        assert f'case "{name}"' in mapping, f"{name} is not mapped"
        assert f'return "{sentence}";' in mapping, f"{name} does not reach {sentence}"
    assert 'return "error_mic";' in mapping, "an unknown failure has no fallback sentence"
    # The capture path ASKS for the mapping: a status set to the bare key would
    # make the three sentences unreachable code.
    assert "setStatus(micErrorKey(error));" in app

    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    for language in ("fr", "en"):
        table = strings[language]
        capture = ("error_mic", "error_mic_blocked", "error_mic_missing", "error_mic_busy")
        for name in capture:
            assert table.get(name), f"{language}: {name} is missing"
        assert len({table[name] for name in capture}) == 4, (
            f"{language}: two capture failures share one sentence"
        )


def test_a_failed_capture_names_its_stage_and_the_error_in_the_console():
    # The sentence tells the singer WHAT happened; the console has to tell a bug
    # report WHERE. One name means two things here — a refused getUserMedia and
    # a worklet whose module never loaded both throw NotAllowedError — so the
    # stage is the fact the sentence cannot carry.
    app = (WEB / "app.js").read_text(encoding="utf-8")
    report = app[app.index("function logCaptureFailure(") :]
    report = report[: report.index("\n}\n")]
    assert "console.error(" in report, "the capture failure never reaches the console"
    assert "error?.name" in report, "the line does not carry the browser's own name"

    body = app[app.index("async function startCapture(") :]
    body = body[: body.index("\nasync function ")]
    # Each label is set BEFORE the call it names, so the line reports the step
    # that was running: a label set afterwards would name the next one.
    engine = body.index('stage = "audio engine"')
    worklet = body.index('stage = "worklet"')
    microphone = body.index('stage = "microphone"')
    assert engine < worklet < body.index("addModule"), "the worklet runs unlabelled"
    assert worklet < microphone < body.index("await navigator.mediaDevices.getUserMedia"), (
        "the microphone request runs unlabelled"
    )
    assert body.index("await navigator.mediaDevices.getUserMedia") < body.index(
        'stage = "audio engine"', engine + 1
    ), "wiring the open stream is the graph again, not the microphone"

    catch = body[body.index("} catch (error) {") :]
    assert "logCaptureFailure(stage, error);" in catch, "the catch reports no stage"
    assert catch.index("logCaptureFailure") < catch.index("setStatus(micErrorKey"), (
        "the console line must be written before the sentence is shown"
    )
    # The two guards that never reach that catch write their own line, naming
    # the cause instead of a stage that never ran.
    assert 'logCaptureFailure("microphone", {' in body


def test_an_insecure_page_says_so_and_playback_is_not_capture():
    # A browser hides navigator.mediaDevices outside a secure context, so a page
    # served over plain http never prompts: sending the reader to check the
    # permission would point at a setting that cannot help. The test is
    # `isSecureContext` ALONE — a page of ES modules never loads from a local
    # file, so a `file:` arm could not run, and dead code is what this suite is
    # for — while the sentence still names the local file, which is where such a
    # reader usually started.
    app = (WEB / "app.js").read_text(encoding="utf-8")
    reason = app[app.index("function insecurePageReason(") :]
    reason = reason[: reason.index("\n}\n")]
    assert "isSecureContext" in reason, "plain http is not named"
    assert "location.protocol" not in reason, "a file:// arm cannot run from a module page"

    body = app[app.index("async function startCapture(") :]
    body = body[: body.index("\nasync function ")]
    # The insecure check runs BEFORE the API guard: an insecure page usually has
    # no mediaDevices either, and the neutral sentence must not win.
    guard = body.index("navigator.mediaDevices?.getUserMedia")
    assert body.index("insecurePageReason()") < guard, (
        "the neutral sentence wins on an insecure page"
    )
    assert 'setStatus("error_insecure");' in body
    assert 'name: "insecure context"' in body, "the console line does not name the context"

    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    for language in ("fr", "en"):
        table = strings[language]
        assert table.get("error_insecure"), f"{language}: error_insecure is missing"
        assert table["error_insecure"] != table["error_mic"], (
            f"{language}: an insecure page reads like a broken audio graph"
        )
        # The local file keeps its place in the words even though no branch can
        # detect it: it is the reader's first guess, and the fix is the same.
        assert ("fichier local" if language == "fr" else "local file") in table["error_insecure"]
        # Playback and capture are different failures: one word must not serve
        # both, or the reader cannot tell which half of the page broke.
        assert table["error_play"] != table["error_mic"]
        if language == "fr":
            assert "lecture" in table["error_play"].lower()
            assert "écoute" not in table["error_play"].lower()
            assert "écoute" in table["error_mic"].lower()
        else:
            assert "playback" in table["error_play"].lower()
            assert "listen" not in table["error_play"].lower()
            assert "listening" in table["error_mic"].lower()


def test_a_graph_that_never_starts_is_reported_instead_of_awaited_forever():
    # `AudioContext.resume()` settles only once an output device can start: a
    # headless browser or a machine whose audio service is off leaves the
    # promise pending for good, and the page waited on it BEFORE asking for the
    # microphone — so the reader sat on "microphone open" while nothing was
    # ever recorded, no console line said why, and no take could be exported.
    app = (WEB / "app.js").read_text(encoding="utf-8")
    helper = app[app.index("async function startedContext(") :]
    helper = helper[: helper.index("\n}\n")]
    assert "Promise.race(" in helper, "the resume is awaited with no bound"
    assert "setTimeout(" in helper, "nothing gives up on a context that never starts"
    assert 'context.state === "running"' in helper, "the outcome is not the graph's own state"

    body = app[app.index("async function startCapture(") :]
    body = body[: body.index("\nasync function ")]
    assert "await startedContext(context)" in body, "the capture still awaits resume bare"
    crashed = body[body.index("await startedContext(context)") :]
    assert 'name: "AudioContext did not start"' in crashed, (
        "a dead output is not named in the console"
    )
    # The microphone's sentence, not a fresh one: the microphone was never the
    # problem, and the neutral sentence already owns this page's own graph.
    assert crashed.index("logCaptureFailure") < crashed.index('setStatus("error_mic")')
    # And the branch returns BEFORE asking for the stream: a dead output is not
    # a permission problem, and a prompt would ask the reader for nothing.
    assert "getUserMedia" not in crashed[: crashed.index("await stopCapture(true)")]

    # PLAYBACK had the same dead end and now has the same bound: `startListening`
    # awaited `resume()` bare, so a machine with no output sat on "Préparation de
    # l'écoute…" for good. It must wait through `startedContext` and let its own
    # catch name the playback failure.
    play = app[app.index("async function startListening(") :]
    play = play[: play.index("\nasync function ")]
    assert "await playbackContext.resume()" not in play, "an unbounded playback resume is back"
    assert "await startedContext(playbackContext)" in play, (
        "playback awaits resume bare: a dead output is never reported"
    )
    assert 'setStatus("error_play");' in play, "the playback branch lost its own sentence"
    # ... and a console line, like the microphone's: the step that was running
    # and the graph's own state, so a bug report can say why without a
    # screenshot ("suspended" past the bound is a dead output, "running" with a
    # throw is this page's bug).
    assert "logPlaybackFailure(stage, error, state)" in play, (
        "a failed playback leaves no console line to report"
    )
    log = app[app.index("function logPlaybackFailure(") :]
    log = log[: log.index("\n}\n")]
    assert "playback failed [${stage}]" in log, "the console line does not name the step"
    assert "AudioContext ${state}" in log, "the console line does not name the graph's state"


def test_the_pages_workflow_tests_before_it_publishes():
    workflow = (REPO / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "node --test web/test/segmenter.test.mjs" in workflow
    assert "node --test web/test/yin.test.mjs" in workflow
    assert "node --test web/test/yin-rates.test.mjs" in workflow
    assert "node --test web/test/parity.test.mjs" in workflow, (
        "the page is published without comparing it to the desktop on audio"
    )
    assert "path: web" in workflow  # the artifact IS the page folder
    assert "actions/deploy-pages@v4" in workflow
    assert "branches: [main]" in workflow
    # BOTH jobs gate the deploy. The microphone capture used to be an OPT-IN
    # gate (a repository variable nothing in the tree could set), which meant a
    # red capture published anyway; the default is now refusal, and the only
    # escape is an explicit waiver.
    assert "needs: [verify, capture]" in workflow
    assert "needs.verify.result == 'success'" in workflow
    assert "needs.capture.result == 'success'" in workflow
    assert "vars.MIC_CAPTURE_WAIVED == 'true'" in workflow, (
        "the microphone gate is not on by default"
    )
    assert "MIC_CAPTURE_REQUIRED" not in workflow, (
        "the old opt-in is back: a gate that must be switched on is off by default"
    )
    assert workflow.count("vars.MIC_CAPTURE_WAIVED") == 1, "the waiver must be the one door"
    assert "!cancelled()" in workflow
    # And using the waiver is LOUD: the deploy job annotates the run with what
    # was not proven. A quiet bypass is how the coverage stops being honest.
    deploy_job = workflow[workflow.index("\n  deploy:") :]
    waiver_step = deploy_job[deploy_job.index("needs.capture.result != 'success'") :]
    assert "::warning" in waiver_step, "a waived capture passes in silence"
    assert "waived" in waiver_step.lower()

    jobs = workflow[workflow.index("\njobs:") :]
    verify_job = jobs[jobs.index("\n  verify:") : jobs.index("\n  capture:")]
    capture_job = jobs[jobs.index("\n  capture:") : jobs.index("\n  deploy:")]
    assert "node --test web/test/capture.test.mjs" in capture_job
    assert "node --test web/test/capture.test.mjs" not in verify_job, (
        "the microphone path still holds the publication hostage"
    )
    # The chosen file needs no device, so it GATES the publication: it is the
    # one reading a reader picks, and the deploy must stop on it.
    assert "node --test web/test/file.test.mjs" in verify_job, (
        "the chosen-file path does not stop a deploy"
    )
    # The MIDI a reader downloads is written by hand, with no library to refuse
    # a bad tick: the two rules a reader cannot see — an OFF before an ON at a
    # shared tick, a same-pitch repeat released at its attack — are pinned by a
    # test that needs no browser, so the deploy must stop on them.
    assert "node --test web/test/midi.test.mjs" in verify_job, (
        "the MIDI a reader downloads is never checked before a deploy"
    )
    # The keyboard walk gates the deploy too: a focus chain that skips a
    # control, or a Space that does two things, is a page half the readers
    # cannot use. Its playback half needs an output device, so it runs a second
    # time in `capture`, where the null sink exists.
    assert "node --test web/test/keyboard.test.mjs" in verify_job, (
        "the keyboard journey does not stop a deploy"
    )
    assert "node --test web/test/keyboard.test.mjs" in capture_job, (
        "the Space-on-playback half never runs with an output device"
    )
    # The page's own Content-Security-Policy is a gate in its own right: a
    # policy that refuses the worklet's blob: breaks the only recording path.
    assert "node --test web/test/csp.test.mjs" in verify_job, (
        "the page is published without proving its own policy refuses nothing"
    )
    assert "node --test web/test/csp.test.mjs" in capture_job, (
        "the worklet is never proven under the policy"
    )
    # A playback graph that never starts must say so: the page used to wait on
    # `resume()` forever on a machine with no output.
    assert "node --test web/test/playback-failure.test.mjs" in verify_job, (
        "a dead output device would leave the page on \"preparing\" again"
    )
    # The table's follow travels with the playhead, which only ticks once the
    # page's audio graph runs: `verify` cannot prove it, so the null-sink job
    # refuses the skip there.
    assert "node --test web/test/follow.test.mjs" in capture_job, (
        "the table's follow is never measured with a playhead that ticks"
    )
    assert "module-null-sink" in capture_job
    assert 'V2M_REQUIRE_AUDIO: "1"' in capture_job
    assert "timeout-minutes:" in capture_job, "a stuck capture job would hold the deploy"
    # The refusal sentences need no audio device at all, so they stay in the
    # job that gates publication.
    assert "node --test web/test/mic-failures.test.mjs" in verify_job
    # Publication is serialised and a superseded push cannot land: without the
    # cancel, two pushes race and the OLDER one can finish last.
    assert "concurrency:" in workflow
    assert "group: pages" in workflow
    assert "cancel-in-progress: true" in workflow
    # The deploy needs to read the checkout, write the deployment and take the
    # OIDC token — nothing else, and never `write-all`.
    block = workflow[workflow.index("\npermissions:") :]
    block = block[: block.index("\n\n")]
    keys = [line for line in block.splitlines() if line.startswith("  ") and ":" in line]
    granted = {line.split(":")[0].strip() for line in keys}
    assert granted == {"contents", "pages", "id-token"}, granted
    assert "write-all" not in workflow
    # One configuration step, not two: `enablement: true` already does the work
    # a bare call would repeat.
    deploy = workflow[workflow.index("\n  deploy:") :]
    assert deploy.count("actions/configure-pages@v5") == 1, "the deploy configures Pages twice"
    # The PUBLISHED page is read back from its own url, after the deployment:
    # every other check serves `web/` from this machine, so a project site that
    # lives under /<repo>/, a module served as text/plain or a `strings.json`
    # missing from the artifact would only show in a visitor's console.
    assert "node web/test/published.mjs" in deploy, "the deployment is never read back"
    assert deploy.index("actions/deploy-pages@v4") < deploy.index("web/test/published.mjs"), (
        "the published page must be read AFTER it is deployed"
    )
    assert "steps.deployment.outputs.page_url" in deploy, (
        "the check reads some other url than the one that was published"
    )
    # A published page that is not a SECURE CONTEXT has no microphone at all:
    # the browser hides `navigator.mediaDevices`, so the page renders and can
    # never be sung into. Both halves of the browser's rule are read from the
    # published origin, and BEFORE the demo is driven — a check that runs after
    # the work it guards reports a usable page that no singer can use.
    published = (WEB / "test" / "published.mjs").read_text(encoding="utf-8")
    assert "isSecureContext" in published, "the published page's secure context is never read"
    assert "location.protocol" in published and '"https:"' in published, (
        "the published page's scheme is never read"
    )
    assert published.index("isSecureContext") < published.index("await pressDemo("), (
        "the secure-context check runs after the demo instead of gating it"
    )
    # The wide layout has no unit test: a 1600 px shot on every push, kept in
    # the run's artifacts, and the step REFUSES the wide facts itself.
    assert "node web/test/wide-shot.mjs wide-1600.png" in workflow
    assert "name: wide-screenshot" in workflow
    assert "if: always()" in workflow, "a shot taken before a refusal is lost with the refusal"
    # The resolution matrix is measured in a browser on every push: the four
    # CSS queries can be grepped, their RESULT cannot.
    assert "node web/test/resolutions.mjs" in workflow
    # The audit is pinned and downloaded, never vendored: the page stays
    # dependency-free, and a new axe rule cannot arrive on its own.
    assert "axe-core@4.10.2" in workflow
    assert "node web/test/axe-audit.mjs --axe" in workflow
    # The microphone path is the one user path no other check presses: a fake
    # device feeds it, and V2M_REQUIRE_AUDIO refuses the skip a runner without
    # a sound card would otherwise take.



def test_the_page_is_measured_in_a_browser_at_every_resolution_class():
    # A declared media query is not a layout: a rule can lose to a later one, a
    # wrap can overflow by two pixels, a control can end under the tap floor.
    # So the classes are MEASURED, and each one states its own facts.
    script = (WEB / "test" / "resolutions.mjs").read_text(encoding="utf-8")
    for size in (
        'width: 320, height: 640',
        'width: 390, height: 844',
        'width: 600, height: 900',
        "width: 812, height: 375",
        "width: 1000, height: 800",
        "width: 1280, height: 900",
        "width: 1600, height: 1000",
        "width: 2560, height: 1440",
    ):
        assert size in script, f"the matrix no longer measures {size}"
    # The facts a reader would feel, never a screenshot nobody opens.
    for fact in (
        "the page scrolls sideways",
        "the measure is",
        "the pills are",
        "the masthead pads",
        "controls under",
        "content past the right edge",
        "the prose stretches with the window",
        "a landscape phone spends",
    ):
        assert fact in script, f"the matrix cannot fail on: {fact}"
    assert "process.exitCode = 1" in script
    # One browser driver for the page's checks, and no dependency: the script
    # drives the same served checkout the screenshot and the audit use.
    assert 'from "./browser.mjs"' in script
    assert "createServer" not in script, "a second static server, a second set of rules"
    assert "pathToFileURL(process.argv[1]).href === import.meta.url" in script, (
        "importing the matrix would launch a browser of its own"
    )


def test_the_accessibility_audit_replays_the_whole_resolution_matrix():
    # A rule broken ONLY inside a media query is invisible at 1600 px: the audit
    # reads the matrix, so a width cannot be measured for the layout and
    # forgotten by the palette checks (or the other way round).
    script = (WEB / "test" / "axe-audit.mjs").read_text(encoding="utf-8")
    assert 'from "./resolutions.mjs"' in script, (
        "the audit owns a second list of widths instead of the measured one"
    )
    assert "for (const size of RESOLUTIONS)" in script
    assert "setDeviceMetricsOverride" in script, (
        "the audit is at one width, not at the class it names"
    )
    # The page must be TRANSLATED before it is judged: `#demo` exists in the
    # markup and answers on a white document, whose controls have no names yet.
    assert '"#theme"' in script and '"#status"' in script, (
        "the audit can run before the page painted its words"
    )


def test_the_wide_screenshot_presses_the_pages_own_demo_and_checks_its_facts():
    # A screenshot nobody judges catches nothing by itself, so the script also
    # FAILS on the wide facts — and it loads the take by pressing the page's own
    # button, because an empty table would photograph the one thing the wide
    # check exists for as missing.
    script = (WEB / "test" / "wide-shot.mjs").read_text(encoding="utf-8")
    assert "const WIDTH = 1600;" in script
    driver = (WEB / "test" / "browser.mjs").read_text(encoding="utf-8")
    assert "Emulation.setDeviceMetricsOverride" in driver, (
        "the width is the window's guess, not the one the page is measured at"
    )
    assert "querySelector('#demo').click()" in driver, "the shot would show an empty table"
    assert "pressDemo" in script, "the shot does not load the page's own demo"
    assert "scrollHeight" in script, (
        "the shot stops above the notes table: the columns it exists for are not in it"
    )
    for fact in ("the wide measure is", "the numeric columns read", "the text columns read"):
        assert fact in script, f"the step cannot fail on: {fact}"
    # No dependency and no foreign host: the page is served from this checkout.
    assert "createServer" in driver
    assert "http://127.0.0.1" in driver and "https://" not in driver + script
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "/wide-*.png" in ignored, "the shot would be committed"


def test_the_accessibility_audit_covers_both_palettes_and_every_state():
    # axe reads the RENDERED DOM, so the palette matters: six of the page's
    # colours are chosen per theme and a verdict sits on a row wash. The audit
    # visits all six, and refuses what axe could not decide too — an incomplete
    # rule is a check nobody made, which is exactly what a green job would hide.
    script = (WEB / "test" / "axe-audit.mjs").read_text(encoding="utf-8")
    assert 'const STATES = ["empty", "take", "playing"];' in script
    assert 'for (const theme of ["light", "dark"])' in script
    assert 'resultTypes: ["violations", "incomplete"]' in script
    assert "problems.push" in script and "process.exitCode = 1" in script
    # The palette is the reader's own: stored, then RELOADED, so the page paints
    # it before its first frame instead of being recoloured by the test.
    assert 'localStorage.setItem("v2m_theme"' in script
    assert "Page.navigate" in script
    # The heard row is the page's own class, the only difference of that state.
    assert 'classList.add("playing")' in script
    assert "axe.min.js" in script, "the usage does not name the pinned file"


def test_both_languages_carry_every_key_the_page_asks_for():
    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    fr, en = set(strings["fr"]), set(strings["en"])
    assert fr == en, f"one language is missing: {fr ^ en}"
    html = (WEB / "index.html").read_text(encoding="utf-8")
    asked = set(re.findall(r'data-i18n="([^"]+)"', html))
    # The page also names keys from JavaScript (status messages, table cells).
    app = (WEB / "app.js").read_text(encoding="utf-8")
    # `(?<![A-Za-z_])` keeps `createElement("tr")` from reading as a `t(...)` key.
    asked |= set(re.findall(r'(?<![A-Za-z_])(?:setStatus|t)\("([a-z_]+)"\)', app))
    missing = asked - fr
    assert not missing, f"the page asks for keys no language defines: {sorted(missing)}"
