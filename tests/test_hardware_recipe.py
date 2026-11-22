"""The hardware/browser recipe is CITED, and still covers what it promises.

`AGENTS.md` sends a reviewer to `docs/recette-materielle-navigateurs.md` for the
pass a containerised matrix cannot make: the permission prompt, the sound card,
the Files/Downloads shelf and a phone's WebKit. A citation to a document that
was renamed, emptied or narrowed is worse than no citation — the reader is sent
to a promise nobody keeps, and nothing in the suite notices.

So the recipe's contract is pinned here: it exists, `AGENTS.md` names it, it is
whitelisted in `.gitignore` (a clone must be able to have it), and it still
carries SIX gestures, each with a table for the FOUR targets. What the criteria
themselves say is the recipe's business; that they exist for every target is
not.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECIPE = REPO / "docs" / "recette-materielle-navigateurs.md"
AGENTS = REPO / "AGENTS.md"
GITIGNORE = REPO / ".gitignore"

TARGETS = ("iOS Safari", "Android Chrome", "Safari macOS", "Firefox")

# The six gestures, in the order a singer meets them. A renamed gesture is a
# new promise: it must be re-read here, not drift under the same heading.
GESTURES = (
    "Ouvrir la page",
    "Chanter une prise au micro",
    "Ouvrir un fichier",
    "Lire la transcription",
    "Écouter, doser, se déplacer",
    "Exporter",
)


def _sections(text: str) -> dict[str, str]:
    """The `## Geste N — name` sections, keyed by their numbered title."""
    marks = list(re.finditer(r"^## Geste (\d+) — (.+)$", text, re.MULTILINE))
    sections: dict[str, str] = {}
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        sections[f"{mark.group(1)}. {mark.group(2).strip()}"] = text[mark.start() : end]
    return sections


def _missing_targets(section: str) -> list[str]:
    """The targets a section's table does not give a row to."""
    cells = [
        line.split("|")[1].strip()
        for line in section.splitlines()
        if line.startswith("|") and line.count("|") >= 3
    ]
    return [target for target in TARGETS if cells.count(target) != 1]


def test_the_recipe_exists_and_agents_names_it():
    assert RECIPE.is_file(), f"{RECIPE.relative_to(REPO)} is missing"
    cited = f"`{RECIPE.relative_to(REPO).as_posix()}`"
    agents = AGENTS.read_text(encoding="utf-8")
    assert cited in agents, f"AGENTS.md no longer cites {cited}"
    # The doc must be able to ship: `docs/*` is ignored, and each real document
    # is whitelisted one by one (root-anchored, like its neighbours).
    whitelist = f"!/{RECIPE.relative_to(REPO).as_posix()}"
    assert whitelist in GITIGNORE.read_text(encoding="utf-8"), (
        "the recipe is gitignored: a clone would cite a file it does not have"
    )


def test_the_recipe_keeps_its_six_gestures():
    sections = _sections(RECIPE.read_text(encoding="utf-8"))
    expected = [f"{number}. {name}" for number, name in enumerate(GESTURES, start=1)]
    assert list(sections) == expected, f"the gestures moved: {list(sections)} instead of {expected}"
    assert len(sections) == 6, f"the recipe carries {len(sections)} gestures, not six"


def test_every_gesture_gives_a_row_to_all_four_targets():
    sections = _sections(RECIPE.read_text(encoding="utf-8"))
    assert sections, "no gesture sections to read"
    for title, section in sections.items():
        missing = _missing_targets(section)
        assert not missing, f"{title} has no criteria for {', '.join(missing)}"
    # The four targets are also named once up front, with what each one is
    # alone to cover.
    text = RECIPE.read_text(encoding="utf-8")
    for target in TARGETS:
        assert f"**{target}**" in text, f"the targets table lost {target}"


def test_the_net_bites():
    """A gesture that lost a target, or a doc with one gesture, is refused."""
    text = "## Geste 1 — Ouvrir la page\n\n| Cible | Geste |\n|---|---|\n| iOS Safari | x |\n"
    sections = _sections(text)
    assert list(sections) == ["1. Ouvrir la page"]
    assert _missing_targets(sections["1. Ouvrir la page"]) == [
        "Android Chrome",
        "Safari macOS",
        "Firefox",
    ]
    assert not _missing_targets(
        "| iOS Safari | a |\n| Android Chrome | b |\n| Safari macOS | c |\n| Firefox | d |\n"
    )
