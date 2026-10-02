"""The guard of the facts/presentation structure: adapters consume.

The decisive split of the architecture passes: core owns the musical
FACTS (bpm, meter, figures, grid), the GUI and the CLI own only the
PRESENTATION and consume published facts (`note_closed` carries the
labeled note and the grid). The regression to refuse is an adapter that
RE-DERIVES a fact — it already happened three times: the CLI re-labeled
the list (`estimate_bpm_live` + `label_notes` in `_on_event`), the GUI
tick rebuilt a third `BeatGrid`, and the GUI metronome called
`detect_meter` itself — which CRASHED silently on `LiveNote`s without a
`strength` field.

The check is an AST, not a grep: aliases (`import x as y`) and module
path calls (`x.detect_meter(...)`) are seen like direct imports. The
proof the guard bites lives in the test: the historical regressions,
rewritten as snippets, are rejected by the same checker — should anyone
reintroduce them in a real adapter, the first test fails by the same
logic.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DETECTION_ENTRIES = {"estimate_bpm_live", "detect_meter", "label_notes"}
ROOT = Path(__file__).parents[1]
# The GUI package, plus the CLI module — the CLI live adapter
# (`LiveSession`) lives in cli.py (v2m/live.py deleted): the guard
# follows the adapters where they live.
ADAPTER_FILES = [
    *sorted((ROOT / "src" / "v2m" / "gui").glob("*.py")),
    ROOT / "src" / "v2m" / "cli.py",
]


def _violations(source: str) -> list[str]:
    """Imports and calls of a detection entry, aliases included.

    The import alone already counts: an adapter keeping the entry up its
    sleeve is the smell to refuse, even without a call today.
    """
    tree = ast.parse(source)
    bound = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name in DETECTION_ENTRIES
    }
    found = list(bound)  # the import itself is already the violation
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in DETECTION_ENTRIES:
            found.append(func.attr)  # module-path call: x.detect_meter(...)
        elif isinstance(func, ast.Name) and (func.id in bound or func.id in DETECTION_ENTRIES):
            found.append(func.id)  # call of an imported alias (or the bare name)
    return found


def test_adapters_import_no_detection_entry():
    """GUI and CLI contain NEITHER an import NOR a call of the detection entries."""
    missing = [p for p in ADAPTER_FILES if not p.exists()]
    assert not missing, f"the guard scans a file that no longer exists: {missing}"
    offenders = []
    for path in ADAPTER_FILES:
        found = _violations(path.read_text(encoding="utf-8"))
        if found:
            offenders.append(f"{path.name}: {sorted(set(found))}")
    assert not offenders, (
        "an adapter re-derives a fact (core owns the policy) : "
        + "; ".join(offenders)
    )


@pytest.mark.parametrize(
    "snippet",
    [
        "from v2m.core.tempo import estimate_bpm_live\nestimate_bpm_live(notes)\n",
        "from v2m.core.tempo import estimate_bpm_live as ebl\nebl(notes)\n",
        "from ..core.meter import detect_meter\nmeter = detect_meter(notes, bpm)\n",
        "import v2m.core.meter as m\nm.detect_meter(notes, bpm)\n",
        "from v2m.core.figures import label_notes\nlabel_notes(notes, grid)\n",
        # the deleted adapter must not come back to life: an import of the
        # detection entries FROM the old v2m.live path is the same smell
        "from v2m.live import estimate_bpm_live\nestimate_bpm_live(notes)\n",
    ],
    ids=["cli-direct", "cli-alias", "gui-direct", "gui-module-path", "gui-labels",
         "live-module-must-not-come-back"],
)
def test_the_guard_bites_on_the_historical_regressions(snippet):
    """The historical regressions (CLI re-labeling, GUI tick, GUI metronome
    on detect_meter) are exactly what the checker refuses."""
    assert _violations(snippet), "the checker must reject this snippet"
