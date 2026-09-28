"""The session rules have TEETH: they are pinned, not hoped for.

Two contracts that so far lived only in AGENTS.md:

1. **The repo tracks NO generated artifact** (the .gitignore header:
   "ignore everything that is not a source"). The only versioned JSONs
   are the tests' frozen references. An artifact committed by accident
   (out.mid, live_20260923_….json) is exactly the regression this test
   refuses — reviewing .gitignore at every new output path is not enough
   when it depends on memory.

2. **`meter.py`'s module docstring IS the detect_meter contract**: it
   names the stages in their order. A stage added to the code without
   being documented (or documented under a name that no longer exists)
   fails this test — the doc can no longer rot behind the code.

3. **"No hard-coded FR string in the code"**: every French literal (an
   accented letter) in `src/v2m` must live in the i18n table or be    canonical vocabulary (figures, sol-fa). A French `RuntimeError("...")`
   outside `tr()` is exactly the drift the guard refuses.

4. **The same border for French words WITHOUT accents** ('echec',
   'trouve', 'fichier'…): an FR message without an accented letter would
   pass guard 3 invisibly. Exemptions are explicit, per (file, literal)
   — an internal identifier declares its reason here.

5. **The suite costs at most 90 s**: the tracking is printed at every run
   end (`suite cost: …`) and a GREEN run over budget fails. A slow test
   added becomes an explicit decision, not a silent drift of the dev
   loop.

6. **Every literal key passed to `tr()` exists in BOTH tables** fr/en:
   the runtime fallback (`.get(key, key)`) renders the raw key to the
   user instead of raising, so a typo in a key can only be seen
   statically — here, on src/ and scripts/.
"""

from __future__ import annotations

import ast
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent

# French accented letters — a CODE literal (docstrings excluded) carrying
# one outside i18n.py is a user message that did not go through `tr()`.
# Docstrings and canonical vocabulary (figure "pointée", sol-fa "Ré"...)
# are exempted by design: they are untenable in English.
FRENCH_LETTERS = set("àâäéèêëîïôöùûüçœÀÂÄÉÈÊËÎÏÔÖÙÛÜÇŒ")
CANONICAL = ("pointée", "trente-deuxième", "seizième de soupir", "Ré", "Ré#", "Do", "La")
CANONICAL_RE = tuple(re.escape(w) + r"\b" for w in CANONICAL)


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def test_repo_tracks_no_generated_artifacts():
    """No generated artifact tracked: sources + configs + frozen reference."""
    tracked = _tracked()
    offenders = [f for f in tracked if f.endswith((".mid", ".musicxml", ".csv", ".pdf", ".wav"))]
    # JSON: only the tests' frozen references have standing
    offenders += [
        f for f in tracked if f.endswith(".json") and not f.startswith("tests/reference/")
    ]
    assert not offenders, f"generated artifacts tracked : {offenders}"


def test_meter_docstring_names_its_stages():
    """Every stage detect_meter calls is named in the module docstring."""
    source = (REPO / "src" / "v2m" / "core" / "meter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_doc = ast.get_docstring(tree) or ""
    detect = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "detect_meter")

    # the real stages: the _detect_* functions detect_meter calls
    called = {
        n.func.id
        for n in ast.walk(detect)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.startswith("_detect")
    }
    assert called, "detect_meter calls no stage? (structure changed?)"
    undocumented = {stage for stage in called if stage not in module_doc}
    assert not undocumented, (
        f"stages not documented in meter.py's module docstring : {sorted(undocumented)} — "
        "the order contract lives in that docstring: document the stage or do not add it."
    )


def _iter_code_literals(source: str) -> list[tuple[int, str]]:
    """All CODE literals of a source (docstrings excluded)."""
    tree = ast.parse(source)
    # docstrings are the FIRST statement (Constant) of a module/class/function
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
    """CODE literals (docstrings excluded) carrying an accented letter."""
    return [
        (lineno, text)
        for lineno, text in _iter_code_literals(path.read_text(encoding="utf-8"))
        if FRENCH_LETTERS & set(text)
    ]


def test_no_french_literals_outside_i18n():
    """No hard-coded FR string outside i18n.py: the session rule has its AST tooth.

    i18n.py's module docstring already carries the border (it IS the
    table); the guard excludes docstrings (English) and the untenable
    canonical vocabulary (figures, sol-fa) — the target is the user
    MESSAGE that should have gone through `tr()`.
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
        "hard-coded FR strings outside i18n.py — go through tr() (key in the fr/en table):\n  "
        + "\n  ".join(offenders)
    )


# --- guard 4: common French words WITHOUT accents ---------------------------
# 'echec de l'analyse', 'fichier non trouve': no accented letters here, guard 3 sees
# NOTHING — yet those are French messages that did not go through tr().
# Accents are folded on BOTH sides so 'trouvé' and 'trouve' fall under the
# same rule, and each word is bounded (\b) so 'west', 'manifest', 'latest'
# are not swallowed (English has 'est' too).
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

# EXPLICIT exemptions, per (file relative to src/v2m, literal): a legit
# unaccented FR word in the code declares ITSELF here with its reason.
# Narrow scope on purpose: the same literal elsewhere stays an offender.
# (Empty today: the last tenant — widgets.py's 'duree' slot — disappeared
# when SLOT_KEYS became the tr() keys themselves.)
EXEMPT_LITERALS = {}


def _unaccented_french_hits(rel: str, source: str) -> list[tuple[int, str]]:
    """Literals of `source` carrying an unaccented FR word, exemptions excluded."""
    hits: list[tuple[int, str]] = []
    for lineno, text in _iter_code_literals(source):
        if (rel, text) in EXEMPT_LITERALS:
            continue
        if FRENCH_WORD_RE.search(text.translate(ACCENT_FOLD)):
            hits.append((lineno, text))
    return hits


def test_no_unaccented_french_literals_outside_i18n():
    """No unaccented French word outside i18n.py — guard 3 has a tooth 2.

    The exemptions live in EXEMPT_LITERALS with their reason; docstrings
    (English) and canonical vocabulary stay outside the guard as for
    guard 3. The target is the user MESSAGE that should have gone through
    `tr()` without needing an accent to be noticed.
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
        "unaccented FR words outside i18n.py — go through tr(), or declare "
        "the exception in EXEMPT_LITERALS with its reason:\n  "
        + "\n  ".join(offenders)
    )


def test_unaccented_french_guard_bites():
    """Guard 4 has TEETH: it bites a real offender, and only it."""
    bad = 'raise RuntimeError("echec de l\'analyse: fichier non trouve")\n'
    assert _unaccented_french_hits("cli.py", bad), "the guard does not bite?"
    # the exemption is SCOPED to the file: declared on widgets.py, it does
    # not whiten the same literal elsewhere (synthetic exemption — the real
    # 'duree' slot no longer exists)
    EXEMPT_LITERALS[("gui/widgets.py", "duree")] = "test reason"
    try:
        assert not _unaccented_french_hits("gui/widgets.py", 'k = "duree"\n')
        assert _unaccented_french_hits("cli.py", 'k = "duree"\n')
    finally:
        del EXEMPT_LITERALS[("gui/widgets.py", "duree")]
    # an English word CONTAINING a French word does not count
    assert not _unaccented_french_hits("cli.py", 'msg = "the latest manifest"\n')


class _StubReporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def _stub_session(exitstatus: int, reporter):
    """The only surface the session-finish hook reads."""
    pm = SimpleNamespace(get_plugin=lambda name: reporter)
    return SimpleNamespace(exitstatus=exitstatus, config=SimpleNamespace(pluginmanager=pm))


def test_suite_cost_over_budget_fails_a_green_run(monkeypatch):
    """The suite budget has a tooth: a too-slow green run -> exit 1."""
    import conftest

    reporter = _StubReporter()
    session = _stub_session(0, reporter)
    monkeypatch.setattr(
        conftest, "_SUITE_STARTED_AT",
        time.perf_counter() - conftest.SUITE_BUDGET_S - 1,
    )
    conftest.pytest_sessionfinish(session, exitstatus=0)
    assert session.exitstatus == 1, "a green run over budget must fail"
    assert any("suite cost" in line and "OVER BUDGET" in line for line in reporter.lines)


def test_suite_cost_keeps_failure_codes_and_reports_green_runs(monkeypatch):
    """The tracking masks nothing: an existing exit 1/3 stays as is."""
    import conftest

    # under budget: the green run stays green, the cost is still printed
    reporter = _StubReporter()
    session = _stub_session(0, reporter)
    monkeypatch.setattr(conftest, "_SUITE_STARTED_AT", time.perf_counter())
    conftest.pytest_sessionfinish(session, exitstatus=0)
    assert session.exitstatus == 0
    assert any("suite cost" in line and "OVER BUDGET" not in line for line in reporter.lines)

    # over budget but already failing / interrupted: the codes survive
    monkeypatch.setattr(
        conftest, "_SUITE_STARTED_AT",
        time.perf_counter() - conftest.SUITE_BUDGET_S - 1,
    )
    for existing in (1, 3):
        session = _stub_session(existing, reporter)
        conftest.pytest_sessionfinish(session, exitstatus=existing)
        assert session.exitstatus == existing


def test_the_suite_is_independent_of_the_users_language_file():
    """The suite neither reads nor writes the user's real ~/.v2m_lang.

    Observed regression: a manual `--lang en` had flipped the persisted
    file; `cli.main` without --lang and MainWindow re-call `get_language()`
    mid-run, the whole process went English and the French tests
    ("décoder", "Terminé") literals failed depending on the machine's state. The
    conftest session fixture sandboxes the file AND forces French: the
    same starting point at every run.
    """
    from v2m.core import i18n

    assert "pytest-" in str(i18n.LANG_FILE), (
        f"LANG_FILE not sandboxed ({i18n.LANG_FILE}) : the suite would read/write "
        "the real ~/.v2m_lang — _sandbox_language_file fixture in tests/conftest.py"
    )
    assert i18n.LANG_FILE.read_text(encoding="utf-8") == "fr", (
        "the suite must START in French, whatever the ambient state"
    )


# --- guard 6: every literal key of tr() exists in both tables ---------------
# tr() falls back to the key itself (the runtime contract pinned by
# test_tr_falls_back_to_key: dynamic keys must stay possible), so a typo in
# a key NEVER crashes at runtime — the user sees the raw key. Only a static
# check can see it.
def _tr_names(tree: ast.AST) -> set[str]:
    """LOCAL names of an imported i18n.tr ('from v2m.core.i18n import tr as _tr').

    Attribute calls (Qt's QObject.tr, i18n.tr) are NOT ast.Name: they fall
    out of scope naturally.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("i18n"):
            for a in node.names:
                if a.name == "tr":
                    names.add(a.asname or "tr")
    return names


def _tr_key_offenders(rel: str, source: str, tables: dict) -> list[str]:
    """Literals passed to tr() absent from ONE of `tables`' two tables."""
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
    """A mistyped tr() key is seen HERE, not by the user.

    The guard covers src/ AND scripts/ (profile_analysis, score_accuracy
    translate too); a non-literal argument (a variable, an f-string) is
    deliberately outside the guard — that is the dynamic case the runtime
    fallback keeps serving.
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
        "tr() keys missing from the fr/en tables — the fallback would render "
        "the raw key to the user; add the key to BOTH tables:\n  "
        + "\n  ".join(offenders)
    )


def test_tr_key_guard_bites():
    """Guard 6 has TEETH: a missing key is bitten, a real key and a variable are not."""
    from v2m.core import i18n

    bad = 'from v2m.core.i18n import tr\nmsg = tr("err missing key")\n'
    assert _tr_key_offenders("cli.py", bad, i18n.STRINGS), "the guard does not bite?"
    # a real key of BOTH tables passes
    ok = 'from v2m.core.i18n import tr\nmsg = tr("err file not found")\n'
    assert not _tr_key_offenders("cli.py", ok, i18n.STRINGS)
    # a key present on ONE side bites too (fr/en parity)
    half = 'from v2m.core.i18n import tr\nmsg = tr("key tempo")\n'
    one_sided = {"fr": {"key tempo": "Tempo"}, "en": {}}
    assert _tr_key_offenders("cli.py", half, one_sided)
    # a dynamic argument is not checked (the runtime fallback's contract)
    dyn = 'from v2m.core.i18n import tr\nmsg = tr(key)\n'
    assert not _tr_key_offenders("cli.py", dyn, i18n.STRINGS)
    # the import alias is followed ('import tr as _tr' of the smoke test)
    aliased = 'from v2m.core.i18n import tr as _tr\nmsg = _tr("err missing key")\n'
    assert _tr_key_offenders("smoke.py", aliased, i18n.STRINGS)
    # biting on the SHAPE of the real code: a typo injected in memory into
    # cli.py is seen — the real scan of the test above missed nothing
    cli_src = (REPO / "src" / "v2m" / "cli.py").read_text(encoding="utf-8")
    poisoned = cli_src.replace('tr("err file not found")', 'tr("err file not foundd")', 1)
    assert poisoned != cli_src, "the probe has nothing left to inject: did cli.py change?"
    assert _tr_key_offenders("cli.py", poisoned, i18n.STRINGS)
