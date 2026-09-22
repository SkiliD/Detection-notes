"""La garde de la structure faits/présentation : les adaptateurs consomment.

Le partage décisif des passes d'architecture : core possède les FAITS
musicaux (bpm, mesure, figures, grille), la GUI et le CLI ne possèdent que
la PRÉSENTATION et consomment des faits publiés (`note_closed` porte la
note étiquetée et la grille). La régression à refuser est un adaptateur qui
REDÉRIVE un fait — c'est déjà arrivé trois fois : le CLI ré-étiquetait la
liste (`estimate_bpm_live` + `label_notes` dans `_on_event`), le tick GUI
reconstruisait une troisième `BeatGrid`, et le métronome GUI appelait
`detect_meter` lui-même — ce qui CRASHAIT en silence sur les `LiveNote`
sans champ `strength`.

Le contrôle est un AST, pas un grep : les alias
(`import x as y`) et les appels par chemin de module (`x.detect_meter(...)`)
sont vus comme les imports directs. La preuve que la garde mord est dans le
test : les régressions historiques, réécrites en extraits, sont rejetées par
le même vérificateur — si quelqu'un les réintroduit dans un vrai adaptateur,
le premier test échoue par la même logique.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DETECTION_ENTRIES = {"estimate_bpm_live", "detect_meter", "label_notes"}
ROOT = Path(__file__).parents[1]
ADAPTER_FILES = [
    *sorted((ROOT / "src" / "v2m" / "gui").glob("*.py")),
    ROOT / "src" / "v2m" / "live.py",
]


def _violations(source: str) -> list[str]:
    """Imports et appels d'une entrée de détection, alias compris.

    L'import seul compte déjà : un adaptateur qui garde l'entrée sous le
    coude est l'odeur à refuser, même sans appel aujourd'hui.
    """
    tree = ast.parse(source)
    bound = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name in DETECTION_ENTRIES
    }
    found = list(bound)  # l'import lui-même est déjà la violation
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in DETECTION_ENTRIES:
            found.append(func.attr)  # appel par chemin de module : x.detect_meter(...)
        elif isinstance(func, ast.Name) and (func.id in bound or func.id in DETECTION_ENTRIES):
            found.append(func.id)  # appel d'un alias importé (ou du nom brut)
    return found


def test_adapters_import_no_detection_entry():
    """GUI et CLI ne contiennent NI import NI appel des entrées de détection."""
    assert ADAPTER_FILES, "les fichiers adaptateurs existent"
    offenders = []
    for path in ADAPTER_FILES:
        found = _violations(path.read_text(encoding="utf-8"))
        if found:
            offenders.append(f"{path.name}: {sorted(set(found))}")
    assert not offenders, (
        "un adaptateur re-dérive un fait (core possède la politique) : "
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
    ],
    ids=["cli-direct", "cli-alias", "gui-direct", "gui-module-path", "gui-labels"],
)
def test_the_guard_bites_on_the_historical_regressions(snippet):
    """Les régressions historiques (CLI ré-étiquetant, tick GUI, métronome
    GUI sur detect_meter) sont exactement ce que le vérificateur refuse."""
    assert _violations(snippet), "le vérificateur doit rejeter cet extrait"
