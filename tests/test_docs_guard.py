"""Les règles de session ont des DENTS : elles sont épinglées, pas espérées.

Deux contrats qui jusqu'ici ne vivaient que dans AGENTS.md :

1. **Le repo ne versionne AUCUN artefact généré** (l'en-tête du .gitignore :
   « ignorer tout ce qui n'est pas une source »). Les seuls JSON versionnés
   sont les références figées de tests. Un artefact commité par accident
   (out.mid, live_20260923_….json) est exactement la régression que ce test
   refuse — la revue de .gitignore à chaque nouveau chemin de sortie ne
   suffit pas quand elle dépend de la mémoire.

2. **La docstring de module de `meter.py` est le contrat de detect_meter** :
   elle nomme les étages dans leur ordre. Un étage ajouté au code sans
   être documenté (ou documenté sous un nom qui n'existe plus) fait
   échouer ce test — la doc ne peut plus pourrir derrière le code.

3. **« Aucune chaîne FR en dur dans le code »** : tout littéral français
   (lettre accentuée) dans `src/v2m` doit vivre dans la table i18n ou être
   du vocabulaire canonique (figures, solfège). Un `RuntimeError("...")`
   français hors `tr()` est exactement la dérive que la garde refuse.

4. **La même frontière pour les mots français SANS accent** ('echec',
   'trouve', 'fichier'…) : un message FR sans lettre accentuée passerait
   la garde 3 invisible. Les exemptions sont explicites, par
   (fichier, littéral) — un identifiant interne déclare sa raison ici.

5. **La suite coûte au plus 90 s** : le suivi est imprimé à chaque fin de
   run (`suite cost: …`) et un run VERT au-delà du budget échoue. Un test
   lent ajouté devient ainsi une décision explicite, pas une dérive
   silencieuse de la boucle de dev.

6. **Toute clé littérale passée à `tr()` existe dans les DEUX tables**
   fr/en : le fallback runtime (`.get(key, key)`) rend la clé brute à
   l'utilisateur au lieu de lever, donc une faute de frappe dans une clé
   ne peut être vue que statiquement — ici, sur src/ et scripts/.
"""

from __future__ import annotations

import ast
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent

# Lettres accentuées françaises — un littéral de CODE (hors docstrings) qui
# en porte une hors i18n.py est un message utilisateur qui n'a pas suivi
# `tr()`. Docstrings et vocabulaire canonique (figures « pointée », solfège
# « Ré »...) sont exemptés par design : ils sont intenables en anglais.
FRENCH_LETTERS = set("àâäéèêëîïôöùûüçœÀÂÄÉÈÊËÎÏÔÖÙÛÜÇŒ")
CANONICAL = ("pointée", "trente-deuxième", "seizième de soupir", "Ré", "Ré#", "Do", "La")
CANONICAL_RE = tuple(re.escape(w) + r"\b" for w in CANONICAL)


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def test_repo_tracks_no_generated_artifacts():
    """Aucun artefact généré versionné : sources + configs + référence figée."""
    tracked = _tracked()
    offenders = [f for f in tracked if f.endswith((".mid", ".musicxml", ".csv", ".pdf", ".wav"))]
    # JSON : seules les références figées des tests ont le droit de cité
    offenders += [
        f for f in tracked if f.endswith(".json") and not f.startswith("tests/reference/")
    ]
    assert not offenders, f"artefacts générés versionnés : {offenders}"


def test_meter_docstring_names_its_stages():
    """Chaque étage que detect_meter appelle est nommé dans la docstring de module."""
    source = (REPO / "src" / "v2m" / "core" / "meter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_doc = ast.get_docstring(tree) or ""
    detect = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "detect_meter")

    # les étages réels : les fonctions _detect_* que detect_meter appelle
    called = {
        n.func.id
        for n in ast.walk(detect)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.startswith("_detect")
    }
    assert called, "detect_meter n'appelle aucun étage ? (structure changée ?)"
    undocumented = {stage for stage in called if stage not in module_doc}
    assert not undocumented, (
        f"étages non documentés dans la docstring de module de meter.py : {sorted(undocumented)} — "
        "le contrat d'ordre vit dans cette docstring : documente l'étage ou ne l'ajoute pas."
    )


def _iter_code_literals(source: str) -> list[tuple[int, str]]:
    """Tous les littéraux de CODE d'une source (hors docstrings)."""
    tree = ast.parse(source)
    # les docstrings sont le PREMIER statement (Constant) de module/classe/fonction
    docs: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docs.add(id(body[0].value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
    ]


def _code_literals_with_french_letters(path: Path) -> list[tuple[int, str]]:
    """Littéraux de CODE (hors docstrings) portant une lettre accentuée."""
    return [
        (lineno, text)
        for lineno, text in _iter_code_literals(path.read_text(encoding="utf-8"))
        if FRENCH_LETTERS & set(text)
    ]


def test_no_french_literals_outside_i18n():
    """Aucune chaîne FR en dur hors i18n.py : la règle de session a sa dent AST.

    La docstring de module de i18n.py porte déjà la frontière (c'est la
    table elle-même) ; le garde exclut les docstrings (en anglais) et le
    vocabulaire canonique intenable (figures, solfège) — la règle visée
    est le MESSAGE utilisateur qui aurait dû passer par `tr()`.
    """
    offenders: list[str] = []
    for path in sorted((REPO / "src" / "v2m").rglob("*.py")):
        if path.name == "i18n.py":
            continue
        for lineno, text in _code_literals_with_french_letters(path):
            if any(re.search(pat, text) for pat in CANONICAL_RE):
                continue
            offenders.append(f"{path}:{lineno}: {text!r}")
    assert not offenders, (
        "chaînes FR en dur hors i18n.py — passe par tr() (clé dans la table fr/en) :\n  "
        + "\n  ".join(offenders)
    )


# --- garde 4 : mots français courants SANS accent ---------------------------
# 'echec de l'analyse', 'fichier non trouve' : aucun ü, ç, é ici, la garde 3
# ne voit RIEN — pourtant ce sont des messages français qui n'ont pas suivi
# tr(). On replie les accents des DEUX côtés pour que 'trouvé' et 'trouve'
# tombent sous la même règle, et on borne chaque mot (\b) pour ne pas
# avaler 'west', 'manifest', 'latest' (l'anglais a aussi des 'est').
FRENCH_WORDS = (
    "est", "trouve", "echec", "fichier", "dossier", "introuvable",
    "indisponible", "delai", "duree", "analyse",
)
FRENCH_WORD_RE = re.compile(r"\b(" + "|".join(FRENCH_WORDS) + r")\b", re.I)
ACCENT_FOLD = str.maketrans({
    "à": "a", "â": "a", "ä": "a", "é": "e", "è": "e", "ê": "e", "ë": "e",
    "î": "i", "ï": "i", "ô": "o", "ö": "o", "ù": "u", "û": "u", "ü": "u",
    "ç": "c", "œ": "e",
})

# Exemptions EXPLICITES, par (fichier relatif à src/v2m, littéral) : un mot
# Un mot FR sans accent légitime dans le code se déclare ICI avec sa raison.
# Portée volontairement étroite : le même littéral ailleurs reste un offender.
# (Vide aujourd'hui : le dernier habitant — le slot 'duree' de widgets.py —
# a disparu quand SLOT_KEYS est devenu la liste des clés tr() elles-mêmes.)
EXEMPT_LITERALS = {}


def _unaccented_french_hits(rel: str, source: str) -> list[tuple[int, str]]:
    """Littéraux de `source` portant un mot FR sans accent, hors exemptions."""
    hits: list[tuple[int, str]] = []
    for lineno, text in _iter_code_literals(source):
        if (rel, text) in EXEMPT_LITERALS:
            continue
        if FRENCH_WORD_RE.search(text.translate(ACCENT_FOLD)):
            hits.append((lineno, text))
    return hits


def test_no_unaccented_french_literals_outside_i18n():
    """Aucun mot français sans accent hors i18n.py — la garde 3 a une dent 2.

    Les exemptions vivent dans EXEMPT_LITERALS avec leur raison ; les
    docstrings (anglais) et le vocabulaire canonique restent hors garde
    comme pour la garde 3. La cible est le MESSAGE utilisateur qui aurait
    dû passer par `tr()` sans avoir besoin d'un accent pour se faire
    remarquer.
    """
    offenders: list[str] = []
    for path in sorted((REPO / "src" / "v2m").rglob("*.py")):
        if path.name == "i18n.py":
            continue
        rel = path.relative_to(REPO / "src" / "v2m").as_posix()
        for lineno, text in _unaccented_french_hits(
            rel, path.read_text(encoding="utf-8")
        ):
            offenders.append(f"{path}:{lineno}: {text!r}")
    assert not offenders, (
        "mots FR sans accent hors i18n.py — passe par tr(), ou déclare "
        "l'exception dans EXEMPT_LITERALS avec sa raison :\n  "
        + "\n  ".join(offenders)
    )


def test_unaccented_french_guard_bites():
    """La garde 4 a des DENTS : elle mord un vrai offender, et seulement lui."""
    bad = 'raise RuntimeError("echec de l\'analyse: fichier non trouve")\n'
    assert _unaccented_french_hits("cli.py", bad), "le garde ne mord pas ?"
    # l'exemption est SCOPÉE au fichier : déclarée chez widgets.py, elle ne
    # blanchit pas le même littéral ailleurs (exemption synthétique — le slot
    # 'duree' réel n'existe plus)
    EXEMPT_LITERALS[("gui/widgets.py", "duree")] = "raison de test"
    try:
        assert not _unaccented_french_hits("gui/widgets.py", 'k = "duree"\n')
        assert _unaccented_french_hits("cli.py", 'k = "duree"\n')
    finally:
        del EXEMPT_LITERALS[("gui/widgets.py", "duree")]
    # un mot anglais qui CONTIENT un mot français ne compte pas
    assert not _unaccented_french_hits("cli.py", 'msg = "the latest manifest"\n')


class _StubReporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def _stub_session(exitstatus: int, reporter):
    """La seule surface que le hook de fin de session lit."""
    pm = SimpleNamespace(get_plugin=lambda name: reporter)
    return SimpleNamespace(exitstatus=exitstatus, config=SimpleNamespace(pluginmanager=pm))


def test_suite_cost_over_budget_fails_a_green_run(monkeypatch):
    """Le budget de la suite a une dent : run vert trop lent -> exit 1."""
    import conftest

    reporter = _StubReporter()
    session = _stub_session(0, reporter)
    monkeypatch.setattr(
        conftest, "_SUITE_STARTED_AT",
        time.perf_counter() - conftest.SUITE_BUDGET_S - 1,
    )
    conftest.pytest_sessionfinish(session, exitstatus=0)
    assert session.exitstatus == 1, "un run vert au-delà du budget doit échouer"
    assert any("suite cost" in line and "OVER BUDGET" in line for line in reporter.lines)


def test_suite_cost_keeps_failure_codes_and_reports_green_runs(monkeypatch):
    """Le suivi ne masque rien : un exit 1/3 existant reste tel quel."""
    import conftest

    # sous le budget : le run vert reste vert, le coût est quand même imprimé
    reporter = _StubReporter()
    session = _stub_session(0, reporter)
    monkeypatch.setattr(conftest, "_SUITE_STARTED_AT", time.perf_counter())
    conftest.pytest_sessionfinish(session, exitstatus=0)
    assert session.exitstatus == 0
    assert any("suite cost" in line and "OVER BUDGET" not in line for line in reporter.lines)

    # au-delà du budget mais déjà en échec / interrompu : les codes survivent
    monkeypatch.setattr(
        conftest, "_SUITE_STARTED_AT",
        time.perf_counter() - conftest.SUITE_BUDGET_S - 1,
    )
    for existing in (1, 3):
        session = _stub_session(existing, reporter)
        conftest.pytest_sessionfinish(session, exitstatus=existing)
        assert session.exitstatus == existing


def test_the_suite_is_independent_of_the_users_language_file():
    """La suite ne lit ni n'écrit le ~/.v2m_lang réel de l'utilisateur.

    Régression observée : un `--lang en` manuel avait basculé le fichier
    persisté ; `cli.main` sans --lang et MainWindow rappellent
    `get_language()` au milieu du run, tout le process passait en anglais
    et les tests français (« décoder », « Terminé ») échouaient selon l'état
    du poste. La fixture session de conftest sandbox le fichier ET force le
    français : point de départ identique à chaque run.
    """
    from v2m.core import i18n

    assert "pytest-" in str(i18n.LANG_FILE), (
        f"LANG_FILE non sandboxé ({i18n.LANG_FILE}) : la suite lirait/écrirait "
        "le ~/.v2m_lang réel — fixture _sandbox_language_file dans tests/conftest.py"
    )
    assert i18n.LANG_FILE.read_text(encoding="utf-8") == "fr", (
        "la suite doit DÉMARRER en français, quel que soit l'état ambiant"
    )


# --- garde 6 : toute clé littérale de tr() existe dans les deux tables ------
# tr() retombe sur la clé elle-même (contrat runtime épinglé par
# test_tr_falls_back_to_key : les clés dynamiques doivent rester possibles),
# donc une faute de frappe dans une clé ne plante JAMAIS à l'exécution —
# l'utilisateur voit la clé brute. Seul un contrôle statique peut la voir.
def _tr_names(tree: ast.AST) -> set[str]:
    """Noms LOCAUX de i18n.tr importé ('from v2m.core.i18n import tr as _tr').

    Les appels attributaires (QObject.tr de Qt, i18n.tr) ne sont PAS des
    ast.Name : ils sortent naturellement du champ.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("i18n"):
            for a in node.names:
                if a.name == "tr":
                    names.add(a.asname or "tr")
    return names


def _tr_key_offenders(rel: str, source: str, tables: dict) -> list[str]:
    """Littéraux passés à tr() absents d'UNE des deux tables de `tables`."""
    tree = ast.parse(source)
    names = _tr_names(tree)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in names
        ):
            continue
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if first.value not in tables["fr"] or first.value not in tables["en"]:
                offenders.append(f"{rel}:{node.lineno}: {first.value!r}")
    return offenders


def test_every_tr_literal_key_exists_in_both_language_tables():
    """Une clé tr() mal orthographiée est vue ICI, pas par l'utilisateur.

    Le garde couvre src/ ET scripts/ (profile_analysis, score_accuracy
    traduisent aussi) ; un argument non littéral (variable, f-string) est
    volontairement hors garde — c'est le cas dynamique que le fallback
    runtime continue de servir.
    """
    from v2m.core import i18n

    offenders: list[str] = []
    for base in ("src/v2m", "scripts"):
        for path in sorted((REPO / base).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            offenders += _tr_key_offenders(
                rel, path.read_text(encoding="utf-8"), i18n.STRINGS
            )
    assert not offenders, (
        "clés tr() absentes des tables fr/en — le fallback rendrait la clé "
        "brute à l'utilisateur ; ajoute la clé aux DEUX tables :\n  "
        + "\n  ".join(offenders)
    )


def test_tr_key_guard_bites():
    """Le garde 6 a des DENTS : clé absente mordue, clé réelle et variable non."""
    from v2m.core import i18n

    bad = 'from v2m.core.i18n import tr\nmsg = tr("err missing key")\n'
    assert _tr_key_offenders("cli.py", bad, i18n.STRINGS), "le garde ne mord pas ?"
    # une clé réelle des DEUX tables passe
    ok = 'from v2m.core.i18n import tr\nmsg = tr("err file not found")\n'
    assert not _tr_key_offenders("cli.py", ok, i18n.STRINGS)
    # une clé présente d'un SEUL côté mord aussi (parité fr/en)
    half = 'from v2m.core.i18n import tr\nmsg = tr("key tempo")\n'
    one_sided = {"fr": {"key tempo": "Tempo"}, "en": {}}
    assert _tr_key_offenders("cli.py", half, one_sided)
    # un argument dynamique n'est pas vérifié (contrat du fallback runtime)
    dyn = 'from v2m.core.i18n import tr\nmsg = tr(key)\n'
    assert not _tr_key_offenders("cli.py", dyn, i18n.STRINGS)
    # l'alias d'import est suivi ('import tr as _tr' du smoke test)
    aliased = 'from v2m.core.i18n import tr as _tr\nmsg = _tr("err missing key")\n'
    assert _tr_key_offenders("smoke.py", aliased, i18n.STRINGS)
    # morsure sur la FORME du vrai code : une faute injectée en mémoire dans
    # cli.py est vue — le scan réel du test au-dessus n'a rien raté
    cli_src = (REPO / "src" / "v2m" / "cli.py").read_text(encoding="utf-8")
    poisoned = cli_src.replace('tr("err file not found")', 'tr("err file not foundd")', 1)
    assert poisoned != cli_src, "la sonde n'a plus rien à injecter : cli.py a changé ?"
    assert _tr_key_offenders("cli.py", poisoned, i18n.STRINGS)
