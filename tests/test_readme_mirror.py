"""La paire bilingue README : le miroir est un CONTRAT, pas une intention.

`README.md` (anglais, principal) et `README.fr.md` (français) doivent
rester le même document : même squelette de sections (même séquence de
niveaux de titres — les textes, eux, sont traduits), mêmes drapeaux CLI
documentés, mêmes scripts référencés, liens croisés en tête. La dérive
silencieuse — un drapeau documenté dans une langue seulement, une section
ajoutée d'un seul côté — est ce que cette épingle refuse ; c'est la même
philosophie que la parité des clés fr/en de i18n (tests/test_gui_i18n.py),
et elle donne sa dent à la règle de session « README obligatoire »
(AGENTS.md) : la règle sans test n'était qu'une intention.

Le miroir est volontairement structurel, pas littéral : la prose est
traduite et le compte de tests cité peut évoluer librement. Les NOMS DE
FICHIERS des exemples (`chant.wav`, `rapport.json`…), eux, ne sont pas de
la prose : ils doivent être identiques dans les deux langues — un exemple
qui copie-colle différemment selon la langue est un bug.

La doc ne documente aussi que le RÉEL : chaque drapeau des READMEs doit
exister dans `build_parser`, et réciproquement — sinon la doc annonce une
option qui n'existe pas, ou le CLI gagne une option que personne ne voit.
Les drapeaux que les SCRIPTS de diagnostic définissent eux-mêmes
(`--golden`, `--no-warmup`…) doivent de plus être expliqués dans la prose
des deux langues : ils n'ont pas de help visible ailleurs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

# drapeaux CLI (--min-conf) : précédés d'un caractère non-mot pour ne pas
# avaler les traits d'union d'un mot ou la ponctuation d'une phrase.
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]+)\b")
SCRIPT = re.compile(r"scripts/[a-z_]+\.py")
HEADING = re.compile(r"^(#{1,3}) ", re.M)
COMMAND = re.compile(r"^v2m\b.*$|^\./\.venv/Scripts/python\.exe .*$", re.M)
COMMENT = re.compile(r"#.*$")
# les valeurs d'exemple d'une commande : fichiers audio, sorties JSON…
SAMPLE = re.compile(r"\b(\w+\.(?:wav|json|mid|musicxml|csv))\b")

# drapeaux internes au programme lui-même, jamais à documenter :
# `--version` (action version) et `--help` (l'aide d'argparse).
# `-o` est la forme courte de `--output` (CLI analyze) : documentée une
# fois suffit — les deux formes désignent la même option.
INTERNAL_FLAGS = {"--version", "--help"}
SHORT_ALIASES = {"-o"}


def _skeleton(text: str) -> tuple[list[int], set[str], set[str]]:
    """Le squelette d'un README, indépendant de la langue.

    Les titres sont lus hors blocs de code (les exemples markdown imbriqués
    ne sont pas des sections) ; les drapeaux et scripts sur le texte ENTIER,
    blocs compris — les exemples de commandes sont la documentation réelle.
    """
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    levels = [len(h) for h in HEADING.findall(prose)]
    return levels, set(FLAG.findall(text)), set(SCRIPT.findall(text))


def test_readme_pair_is_mirror_structured():
    en = (REPO / "README.md").read_text(encoding="utf-8")
    fr = (REPO / "README.fr.md").read_text(encoding="utf-8")

    levels_en, flags_en, scripts_en = _skeleton(en)
    levels_fr, flags_fr, scripts_fr = _skeleton(fr)

    # même séquence de niveaux de titres (les textes sont traduits)
    assert levels_en == levels_fr, (
        "les sections des deux README ont divergé — mets à jour les DEUX "
        f"dans le même commit :\nEN : {levels_en}\nFR : {levels_fr}"
    )
    # mêmes drapeaux CLI documentés
    assert flags_en == flags_fr, (
        "drapeaux CLI asymétriques : "
        f"EN seul : {sorted(flags_en - flags_fr)}, "
        f"FR seule : {sorted(flags_fr - flags_en)}"
    )
    # mêmes scripts référencés
    assert scripts_en == scripts_fr, (
        "scripts asymétriques : "
        f"EN seul : {sorted(scripts_en - scripts_fr)}, "
        f"FR seul : {sorted(scripts_fr - scripts_en)}"
    )


def test_language_switchers_cross_link():
    en = (REPO / "README.md").read_text(encoding="utf-8")
    fr = (REPO / "README.fr.md").read_text(encoding="utf-8")
    # l'anglais est le principal : sa première ligne pointe le français
    assert "[Français](README.fr.md)" in en.splitlines()[0]
    assert "[English](README.md)" in fr.splitlines()[0]


def test_example_commands_are_mirrored():
    """Les exemples de commandes portent les mêmes valeurs dans les deux langues.

    Un nom de fichier n'est pas de la prose : `chant.wav` copie-colle
    pareil en français et en anglais. On compare l'ENSEMBLE des commandes
    v2m / scripts de chaque README (ordre libre : la prose décide où ils
    vivent) et, dans chaque commande, les valeurs d'exemple qu'elle porte.
    """
    en = (REPO / "README.md").read_text(encoding="utf-8")
    fr = (REPO / "README.fr.md").read_text(encoding="utf-8")
    # les commentaires inline sont de la prose traduite : on les retire
    commands_en = sorted(COMMENT.sub("", c).strip() for c in COMMAND.findall(en))
    commands_fr = sorted(COMMENT.sub("", c).strip() for c in COMMAND.findall(fr))
    assert commands_en == commands_fr, (
        "les exemples de commandes ont divergé : "
        f"EN seul : {sorted(set(commands_en) - set(commands_fr))}, "
        f"FR seul : {sorted(set(commands_fr) - set(commands_en))}"
    )


def _parser_flags() -> set[str]:
    """Les drapeaux RÉELS : les sous-commandes du CLI + les scripts de diagnostic.

    Les drapeaux des subparsers vivent dans les subparsers (pas dans
    parser._actions de tête) ; `--golden`, `--no-warmup`… sont définis par
    les scripts — ils sont tout aussi réels. `-h`/`--version`/`--lang`
    n'ont pas à être documentés.
    """
    from v2m.cli import build_parser

    parser = build_parser()
    flags: set[str] = set()
    for action in parser._actions:
        if action.option_strings:
            flags.update(action.option_strings)
    subparsers = parser._subparsers
    if subparsers is not None:
        for group_action in subparsers._group_actions:
            choices = getattr(group_action, "choices", None) or {}
            for sub in choices.values():
                for a in sub._actions:
                    if a.option_strings:
                        flags.update(a.option_strings)
    for script in ("score_accuracy.py", "profile_analysis.py"):
        text = (REPO / "scripts" / script).read_text(encoding="utf-8")
        flags.update(FLAG.findall(text))
    return flags - INTERNAL_FLAGS - SHORT_ALIASES - {"-h"}


def test_documented_flags_exist_in_cli_and_vice_versa():
    """La doc ne documente que le réel : drapeaux des READMEs == CLI + scripts."""
    documented: set[str] = set()
    for name in ("README.md", "README.fr.md"):
        documented |= set(FLAG.findall((REPO / name).read_text(encoding="utf-8")))
    actual = _parser_flags()
    assert documented == actual, (
        f"doc ≠ réel — documenté seul : {sorted(documented - actual)}, "
        f"non documenté : {sorted(actual - documented)}"
    )


def _script_flags() -> set[str]:
    """Les drapeaux que les deux scripts de diagnostic définissent eux-mêmes."""
    flags: set[str] = set()
    for script in ("score_accuracy.py", "profile_analysis.py"):
        text = (REPO / "scripts" / script).read_text(encoding="utf-8")
        flags.update(FLAG.findall(text))
    return flags


def test_script_flags_documented_with_semantics():
    """Chaque drapeau de script est expliqué dans la PROSE des deux READMEs.

    La présence suffit pour le CLI (le help de build_parser est lisible via
    --help), mais un drapeau de script n'a pas d'aide visible ailleurs : sa
    sémantique doit vivre dans le texte, pas seulement dans un exemple à
    recopier. Contracte testable : le token du drapeau apparaît HORS des
    blocs de code, dans chaque langue — une phrase par drapeau, comme le
    fait déjà `--no-warmup`.
    """
    missing: dict[str, list[str]] = {}
    for name in ("README.md", "README.fr.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        missing[name] = sorted(f for f in _script_flags() if f not in prose)
    assert not any(missing.values()), (
        f"drapeaux de scripts sans sémantique dans la prose (hors exemples) : {missing} — "
        "une phrase par drapeau, dans les DEUX READMEs, comme --no-warmup."
    )
