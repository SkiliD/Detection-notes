"""Every number the page shows goes through the ONE formatter.

`web/readout.js` owns the page's only `Intl.NumberFormat`, and every module
that prints a number calls it. AGENTS.md states the rule — "a
locale-dependent number is never assembled by hand" — and the page broke it
twice: the table's seconds kept a dot beside a French clock that said `0:03,2`,
and the readout did the same after the first repair. A unit test of the
formatter cannot see a call site that never uses it, so this file reads the
SOURCE for the shapes those defects took:

- `toFixed`, which writes a dot in every language;
- a number assembled by hand — `Math.`, `String(`, a `start`/`duration`/`hz`
  field, an arithmetic expression — where the page writes text (`textContent`,
  a painted `fillText`, a tooltip, or the `aria-valuetext` a reader HEARS).

MACHINE values keep their dot and are named ONE BY ONE with the reason: an
`aria-valuenow` is parsed by assistive tech and never shown, a `data-index`
selector is not a word, the archive's JSON is a payload for another program.
Anything the census does not name fails with its file and line, so a new
displayed number is a review instead of a silent recurrence — and an entry
whose site is gone fails too, so the census cannot rot into a list of ghosts.

The RENDERED page is read by `web/test/numbers.test.mjs`, in both languages on
one take: French must show no decimal dot, and English must show the same
numbers with dots. This file is the copy that fails before Chrome is started.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"

# The only functions allowed to turn a number into text: the one formatter and
# the writers that call it (a translated sentence, a note's NAME, a filled
# template). `midiToName` is here because "C4" is a name, not a measure.
FORMATTERS = re.compile(
    r"\b(?:formatNumber|formatClock|formatHz|formatCents|intonationNumbers|midiToName|t|fill)\("
)

# What a hand-built number looks like once the formatters are removed, and only
# as narrow as the defects were: an explicit conversion, a machine field, a
# computed value. A bare identifier is deliberately absent — `${hz}` in a
# sentence is caught by the browser census, which sees the character that
# reaches the page, not by a guess about the variable's type.
RAW_NUMBER = re.compile(
    r"\.toFixed\(|\bMath\.|\bString\(|\.(?:start|end|duration|hz|dev|midi)\b"
    r"|[\w)\]]\s*[+\-*/]\s*[\w.(]"
)

# Statements that put text where a reader (or a screen reader) meets it.
SINKS = (".textContent =", ".textContent=", ".title = ", ".innerHTML =", "insertAdjacentHTML(")

# The machine values, and why each may keep the dot `toFixed` writes.
MACHINE_NUMBERS = (
    (
        "app.js",
        "`${Math.min(100, (level / 0.08) * 100)}%`",
        "the level bar's CSS width: a layout length, never a word",
    ),
    (
        "app.js",
        "notes[notes.length - 1].end.toFixed(3)",
        "the table's rebuild signature: a dedupe key, never displayed",
    ),
    (
        "app.js",
        "Math.min(position, playheadDuration).toFixed(1)",
        "aria-valuenow: assistive tech parses a number, not prose",
    ),
    (
        "app.js",
        '"aria-valuemax", end.toFixed(1)',
        "aria-valuemax of the seek surface: a machine bound",
    ),
    (
        "app.js",
        '"aria-valuenow", (playhead ?? 0).toFixed(1)',
        "aria-valuenow of the seek surface: a machine position",
    ),
    (
        "app.js",
        "String(Math.round(balance * 100))",
        "the balance input's own value (0-100), not a sentence",
    ),
    (
        "bundle.js",
        "Number(seconds.toFixed(3))",
        "the archive's JSON payload: a number another program reads",
    ),
    ("bundle.js", "Number(note.start.toFixed(4))", "the archive's JSON payload"),
    ("bundle.js", "Number(note.end.toFixed(4))", "the archive's JSON payload"),
    ("bundle.js", "Number((note.end - note.start).toFixed(4))", "the archive's JSON payload"),
    ("bundle.js", "Number(cents.toFixed(1))", "the archive's JSON payload"),
    (
        "cents.js",
        "${Math.abs(cents)}",
        "the formatter itself: whole signed cents, no separator to get wrong",
    ),
    (
        "segmenter.js",
        "${Math.floor(value / 12) - 1}",
        "a MIDI number's octave: part of a note's NAME, not a measure",
    ),
)


def _blank_formatters(text: str) -> str:
    """`text` with every formatter call and its arguments replaced by `FMT`."""
    while True:
        match = FORMATTERS.search(text)
        if not match:
            return text
        depth, index = 1, match.end()
        while index < len(text) and depth:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
            index += 1
        text = f"{text[: match.start()]}FMT{text[index:]}"


def _arguments(text: str, callee: str) -> list[str]:
    """The top-level arguments of the first `callee(...)` call, parens balanced."""
    at = text.find(callee)
    if at == -1:
        return []
    depth, index, start, parts = 1, at + len(callee), at + len(callee), []
    while index < len(text) and depth:
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                parts.append(text[start:index])
                break
        elif char == "," and depth == 1:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    return [part.strip() for part in parts]


def _interpolations(line: str) -> list[str]:
    """Every `${...}` body of one line, braces counted rather than guessed."""
    found, at = [], 0
    while True:
        start = line.find("${", at)
        if start == -1:
            return found
        depth, index = 1, start + 2
        while index < len(line) and depth:
            if line[index] == "{":
                depth += 1
            elif line[index] == "}":
                depth -= 1
            index += 1
        found.append(line[start + 2 : index - 1])
        at = index


def _display_expressions(line: str) -> list[str]:
    """The text-carrying part of one statement.

    `fillText(text, x, y)` is cut at its first argument and an `aria-valuetext`
    at its second: a coordinate and an attribute name are not words, and
    checking them would flag every painting statement on the page.
    """
    found = []
    for marker in SINKS:
        if marker in line:
            found.append(line.split(marker, 1)[1].rstrip(";").strip())
    if "fillText(" in line:
        found.extend(_arguments(line, "fillText(")[:1])
    if 'setAttribute("aria-valuetext"' in line:
        found.extend(_arguments(line, 'setAttribute("aria-valuetext"')[-1:])
    return [expr for expr in found if expr]


def _scan(web: Path) -> tuple[list[str], int]:
    """Unlisted number sites, and how many sites went through a formatter."""
    unlisted: dict[str, None] = {}
    allowed = 0
    for source in sorted(web.glob("*.js")):
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            candidates = _interpolations(line) + _display_expressions(line)
            if "toFixed(" in line:
                candidates.append(line)
            for expression in dict.fromkeys(candidates):
                blanked = _blank_formatters(expression)
                if RAW_NUMBER.search(blanked):
                    # one statement can be reached twice (its interpolation AND
                    # its `toFixed`): the census names the LINE once
                    unlisted[f"{source.name}:{number}: {line.strip()}"] = None
                elif blanked != expression:
                    allowed += 1
    return list(unlisted), allowed


def _census_trouble(web: Path) -> list[str]:
    """Everything wrong with the census: unlisted sites, and ghosts in it."""
    unlisted, allowed = _scan(web)
    trouble = []
    for site in unlisted:
        name = site.split(":", 1)[0]
        listed = any(
            source == name and fragment in site for source, fragment, _reason in MACHINE_NUMBERS
        )
        if not listed:
            trouble.append(f"a display number is not in the census: {site}")
    for source, fragment, _reason in MACHINE_NUMBERS:
        if fragment not in (web / source).read_text(encoding="utf-8"):
            trouble.append(f"stale census entry: {source} no longer contains {fragment}")
    if allowed < 6:
        trouble.append(f"only {allowed} number sites go through the page's formatter")
    return trouble


def _assert_census(web: Path) -> None:
    trouble = _census_trouble(web)
    assert not trouble, "\n".join(trouble)


def test_every_number_the_page_writes_is_either_formatted_or_a_named_machine_value():
    _assert_census(WEB)


def test_the_page_has_exactly_one_number_formatter():
    # A second `Intl.NumberFormat` is a second separator policy: it would pass
    # every language the first one passed while disagreeing with it in the one
    # line nobody re-checked. `cents.js` is not exempted by any condition: it
    # builds no formatter because it prints whole signed cents
    # (`${Math.abs(cents)}`), an integer where no thousands separator can
    # appear — its census entry records the same fact.
    owners = [
        source.name
        for source in sorted(WEB.glob("*.js"))
        if "Intl.NumberFormat" in source.read_text(encoding="utf-8")
    ]
    assert owners == ["readout.js"], f"these files build their own number format: {owners}"


def test_the_number_census_bites(tmp_path):
    """Both shapes of the defect must FAIL, and a ghost entry must too.

    A census that only ever passes is prose: these three mutations are the ones
    the page actually shipped — a raw `toFixed` in a display statement, a
    number assembled into a template literal, and an allowlisted machine value
    that quietly disappears.
    """
    web = tmp_path / "web"
    web.mkdir()
    for source in WEB.glob("*.js"):
        shutil.copy2(source, web / source.name)
    app = web / "app.js"
    original = app.read_text(encoding="utf-8")

    app.write_text(original + '\n$("hz").textContent = hz.toFixed(1) + " Hz";\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="not in the census"):
        _assert_census(web)

    app.write_text(
        original + '\n$("clock-pos").textContent = `${Math.round(position)} s`;\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="not in the census"):
        _assert_census(web)

    # A named machine value that is gone: the census would keep blessing a line
    # nobody writes any more.
    app.write_text(original.replace(".toFixed(3)", ""), encoding="utf-8")
    assert any("stale census entry" in line for line in _census_trouble(web))
