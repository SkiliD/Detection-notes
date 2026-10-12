"""The deletion test for `core/music.py` as the contract seam.

The music facts every reader shares — the bilingual pitch-class name
tables, the seconds<->beats guard, the piano range — are READ from
`music.py` (the architecture review's named seam). A consumer that keeps
its own copy fails here: the copy is what a later change forgets (an
English table engraved into a French label, a grid dividing by a zero
tempo, a key detected from a second pitch-class vocabulary).

Both detectors are bitten by a planted copy: a scan that cannot flag a
copy would pass on any codebase and pin nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import v2m

SRC = Path(v2m.__file__).parent              # src/v2m
REPO = SRC.parents[1]
MUSIC = SRC / "core" / "music.py"
SCRIPTS = REPO / "scripts"

# the contract names the seam owns (the review's list)
CONTRACT = ("NOTE_NAMES", "NOTE_NAMES_FR", "sane_bpm", "on_piano")

# the two pitch-class name tables, as music.py spells them
EN_TABLE = '("C", "C#", "D"'
FR_TABLE = '("Do", "Do#"'

# the consumers the review names: file -> the seam fact it must import
NAMED_CONSUMERS = {
    "core/chords.py": "NOTE_NAMES",
    "core/i18n.py": "NOTE_NAMES_FR",
    "core/keys.py": "NOTE_NAMES",
    "core/musicxml.py": "NOTE_NAMES",
    "core/grid.py": "sane_bpm",
    "gui/staff.py": "sane_bpm",
    "core/notes.py": "on_piano",
    "core/liveseg.py": "on_piano",
    "gui/finder_page.py": "on_piano",
}


def _sources() -> list[tuple[Path, str]]:
    """Every production module that may read the contract (seam excluded)."""
    paths = [p for p in sorted(SRC.rglob("*.py")) if p != MUSIC]
    paths += sorted(SCRIPTS.rglob("*.py")) if SCRIPTS.is_dir() else []
    return [(p, p.read_text(encoding="utf-8")) for p in paths]


def _name_tables(text: str) -> list[str]:
    """The pitch-class name tables the text carries, by language."""
    return [
        label
        for label, needle in (("English", EN_TABLE), ("French", FR_TABLE))
        if needle in text
    ]


def _imported_from_music(tree: ast.AST) -> set[str]:
    """Names the module imports from the music seam itself (not a re-export)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[-1] == "music":
                names |= {alias.asname or alias.name for alias in node.names}
    return names


def _defined(tree: ast.AST) -> set[str]:
    """Contract names the module defines itself — a local copy of a fact."""
    wanted = set(CONTRACT)
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined |= {node.name} & wanted
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    defined.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in wanted:
                defined.add(node.target.id)
    return defined


def _used(tree: ast.AST) -> set[str]:
    """Contract names the module reads in code (docstrings are not reads)."""
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in CONTRACT
    }


def test_no_module_defines_a_contract_name_itself():
    """A fact has ONE owner: a local `sane_bpm = ...` is a fork of the seam."""
    offenders = []
    for path, text in _sources():
        tree = ast.parse(text, filename=str(path))
        for name in sorted(_defined(tree)):
            offenders.append(f"{path.relative_to(REPO)}: {name}")
    assert not offenders, f"contract facts redefined outside music.py: {offenders}"
    # bite: the detector reports a planted copy
    assert _defined(ast.parse("sane_bpm = 120.0")) == {"sane_bpm"}
    assert _defined(ast.parse("def on_piano(m): return True")) == {"on_piano"}


def test_every_use_of_a_contract_name_comes_from_music():
    """A consumer reads the seam, never a re-export or an inline copy."""
    offenders = []
    for path, text in _sources():
        tree = ast.parse(text, filename=str(path))
        imported = _imported_from_music(tree)
        for name in sorted(_used(tree) - imported):
            offenders.append(f"{path.relative_to(REPO)}: {name}")
    assert not offenders, f"contract names not imported from music.py: {offenders}"
    # bite: a use without the seam's import is exactly what fails
    tree = ast.parse("from .grid import sane_bpm\nsane_bpm(0)\n")
    assert _used(tree) - _imported_from_music(tree) == {"sane_bpm"}


def test_there_is_no_second_pitch_class_name_table():
    """The bilingual pair lives once: a second tuple is a fork waiting to
    drift from its translation (the labels would disagree by name)."""
    copies = [
        f"{path.relative_to(REPO)}: {lang}"
        for path, text in _sources()
        for lang in _name_tables(text)
    ]
    assert not copies, f"pitch-class name tables copied outside music.py: {copies}"
    # bite: the planted table is reported
    assert _name_tables('PC = ("C", "C#", "D", "D#", "E")') == ["English"]
    assert _name_tables('PC_FR = ("Do", "Do#", "R\u00e9")') == ["French"]


def test_the_named_consumers_read_the_seam():
    """The review's consumer list: each fact is imported where it is used."""
    for rel, name in NAMED_CONSUMERS.items():
        path = SRC / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert name in _imported_from_music(tree), f"{rel} must import {name} from music.py"
