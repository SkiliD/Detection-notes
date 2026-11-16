"""The session rules have TEETH: they are pinned, not hoped for.

Contracts that require automated enforcement:

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
   accented letter) in `src/v2m` must live in the i18n table or be canonical
   vocabulary (figures, sol-fa). A French `RuntimeError("...")` outside `tr()`
   is exactly the drift the guard refuses.

4. **The same border for French words WITHOUT accents** ('echec',
   'trouve', 'fichier'…): an FR message without an accented letter would
   pass guard 3 invisibly. Exemptions are explicit, per (file, literal)
   — an internal identifier declares its reason here.

5. **The suite has a cost budget**: the tracking is printed at every run
   end (`suite cost: …`) and a GREEN run over budget fails. A slow test
   added becomes an explicit decision, not a silent drift of the dev
   loop.

6. **Every literal key passed to `tr()` exists in BOTH tables** fr/en:
   the runtime fallback (`.get(key, key)`) renders the raw key to the
   user instead of raising, so a typo in a key can only be seen
   statically — here, on src/ and scripts/.

7. **Comments and docstrings are in ENGLISH** (2026-09-28 whole-repo
   sweep): a COMMENT or docstring in src/ or scripts/ (i18n.py exempted —
   its tables ARE the French) carrying an accented-French letter, or one
   of the UNACCENTED French words the accent test structurally cannot see
   (`verite`, `dans`, `aucune`… — the guide_capture sweep, 2026-10-02),
   fails. Canonical vocabulary inside the text is exempted, as for
   guard 3. The word list is bounded on both sides and corpus-audited:
   `quiet`/`describe` are English words, not matches. tests/ is outside
   the border on purpose: test comments QUOTE the French UI by design.

8. **A subprocess booted by a test is AUDITED against the user's
   `~/.v2m_lang`**: the conftest language sandbox is process-local, so
   anything a test boots in a fresh interpreter reads/writes the REAL
   file (one hash-diff proved the suite clean, 2026-09-28 — the guard
   keeps it so). Every booting test file is whitelisted with its
   justification, the booted argv must match the declared kind, and the
   machine re-verifies the claims (the qt-free probe's module still
   imports no i18n; a whitelisted v2m script still never calls
   `get_language`/`set_language`). The same audit covers PRODUCTION
   code: every `subprocess.*` launch under `src/v2m` (and `scripts/`)
   needs a whitelisted justification per module — today the single site
   (pdf_export) runs the user's external engraver, never an
   interpreter, never i18n.

9. **A Qt widget is constructed only where its screen is owned** (ADR
   0010's pattern, complete): the shared library (gui/widgets.py,
   gui/staff.py), a Page module (results/live/finder), the home builder,
   or the composition window. A widget born in a helper, a dialog or a
   fourth place re-opens "who owns this screen?" and fails until it
   joins a Page (or earns its allowlist line with a reason).

10. **Shell scripts keep LF endings**: `.gitattributes` pins
    `*.sh text eol=lf` and `docker-entrypoint.sh` carries no CR — the
    entry point runs through bash inside the Linux image, where a CRLF
    file fails with `env: '\r': No such file or directory`.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import tempfile
import time
import tokenize
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent

# French accented letters — a CODE literal (docstrings excluded) carrying
# one outside i18n.py is a user message that did not go through `tr()`.
# Docstrings and canonical vocabulary (figure "pointée", sol-fa "Ré"...)
# are exempted by design: they are untenable in English.
FRENCH_LETTERS = set("àâäéèêëîïôöùûüçœÀÂÄÉÈÊËÎÏÔÖÙÛÜÇŒ")
CANONICAL = (
    "pointée",
    "trente-deuxième",
    "seizième de soupir",
    "Ré",
    "Ré#",
    "Do",
    "La",
    "détaché",  # musical vocabulary quoted as a term
)
CANONICAL_RE = tuple(re.escape(w) + r"\b" for w in CANONICAL)


# Guard 7's text test: an accented letter (canonical exempted) OR one of
# guard 4's unaccented French words — FRENCH_WORD_RE is defined with guard
# 4 below (one shared list, bounded on both sides, accent-folded, so
# `quiet`/`describe` are English words and never matches). Deliberately
# absent from the list: words that appear inside QUOTED French in
# src/scripts ("chantez une note…").
def _has_french(text: str) -> bool:
    """The text carries French: an accented letter or a guard-4 word."""
    if FRENCH_LETTERS & set(text) and not any(re.search(p, text) for p in CANONICAL_RE):
        return True
    return bool(FRENCH_WORD_RE.search(text.translate(ACCENT_FOLD)))


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


# The web port's JSON has standing, and it is the FROZEN kind: `strings.json` is
# the page's own language table (a source, like `constants.js`), and the three
# under `web/test/` are fixtures the browser checks replay in CI, where the
# Python toolchain that generates them does not run. They are not free to drift:
# `tests/test_web_build.py` regenerates `cases.json` in-process and compares it,
# and the YIN fixtures carry the digest of the take they were cut from.
WEB_JSON = (
    "web/strings.json",
    "web/test/cases.json",
    "web/test/yin.json",
    "web/test/yin-rates.json",
)


def test_repo_tracks_no_generated_artifacts():
    """No generated artifact tracked: sources + configs + frozen reference."""
    tracked = _tracked()
    offenders = [f for f in tracked if f.endswith((".mid", ".musicxml", ".csv", ".pdf", ".wav"))]
    # JSON: only the tests' frozen references and the web port's own table and
    # fixtures have standing.
    offenders += [
        f
        for f in tracked
        if f.endswith(".json")
        and not f.startswith("tests/reference/")
        and f not in WEB_JSON
    ]
    assert not offenders, f"generated artifacts tracked : {offenders}"


def test_meter_docstring_names_its_stages():
    """Every stage detect_meter calls is named in the module docstring."""
    source = (REPO / "src" / "v2m" / "core" / "meter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_doc = ast.get_docstring(tree) or ""
    detect = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "detect_meter"
    )

    # the real stages: the _detect_* functions detect_meter calls
    called = {
        n.func.id
        for n in ast.walk(detect)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id.startswith("_detect")
    }
    assert called, "detect_meter calls no stage? (structure changed?)"
    undocumented = {stage for stage in called if stage not in module_doc}
    assert not undocumented, (
        f"stages not documented in meter.py's module docstring : {sorted(undocumented)} — "
        "the order contract lives in that docstring: document the stage or do not add it."
    )


def _docstring_consts(tree: ast.AST) -> list[ast.Constant]:
    """The first-statement Constants of every module/class/function (docstrings)."""
    consts: list[ast.Constant] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                consts.append(body[0].value)
    return consts


def _iter_code_literals(source: str) -> list[tuple[int, str]]:
    """All CODE literals of a source (docstrings excluded)."""
    tree = ast.parse(source)
    # docstrings are the FIRST statement (Constant) of a module/class/function
    docs = {id(const) for const in _docstring_consts(tree)}
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
# ONE shared list: guard 4 applies it to code literals, guard 7 to
# comments and docstrings (the 2026-10-02 guide_capture sweep — unaccented
# French hid in comments no accent test could see). Every added word is
# audited against this corpus: English neighbours ('quiet', 'describe')
# are not members, and words quoted in French stay out ('chantez', 'une').
FRENCH_WORDS = (
    "est",
    "trouve",
    "echec",
    "fichier",
    "dossier",
    "introuvable",
    "indisponible",
    "delai",
    "duree",
    "analyse",
    # guard 7's extension: comments/docstrings vocabulary
    "aucun",
    "aucune",
    "aussi",
    "avait",
    "borne",
    "bornes",
    "chaque",
    "courbe",
    "dans",
    "deja",
    "depuis",
    "donc",
    "ecrit",
    "entree",
    "etre",
    "etait",
    "faut",
    "fenetre",
    "hauteur",
    "lorsque",
    "mais",
    "meme",
    "parce",
    "peut",
    "peuvent",
    "plusieurs",
    "pour",
    "puis",
    "rafine",
    "raffine",
    "repere",
    "trame",
    "trames",
    "tres",
    "toujours",
    "toutes",
    "tous",
    "verite",
    "avec",
    "cette",
    "ainsi",
    "alors",
    "cela",
    "enfin",
    "manquant",
    "doivent",
    "pendant",
    "reelle",
    "reelles",
    "verifiee",
    "verifiees",
    "arret",
    "separe",
    "valeur",
    "vide",
)
FRENCH_WORD_RE = re.compile(r"\b(" + "|".join(FRENCH_WORDS) + r")\b", re.I)
ACCENT_FOLD = str.maketrans(
    {
        "à": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "œ": "e",
    }
)

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

    Exemptions live in EXEMPT_LITERALS with their reason; docstrings
    (English) and canonical vocabulary stay outside the guard as for
    guard 3. The target is the user MESSAGE that should have gone through
    `tr()` without needing an accent to be noticed.
    """
    offenders: list[str] = []
    for path in sorted((REPO / "src" / "v2m").rglob("*.py")):
        if path.name == "i18n.py":
            continue
        rel = path.relative_to(REPO / "src" / "v2m").as_posix()
        for lineno, text in _unaccented_french_hits(rel, path.read_text(encoding="utf-8")):
            offenders.append(f"{path}:{lineno}: {text!r}")
    assert not offenders, (
        "unaccented FR words outside i18n.py — go through tr(), or declare "
        "the exception in EXEMPT_LITERALS with its reason:\n  " + "\n  ".join(offenders)
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
        conftest,
        "_SUITE_STARTED_AT",
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
        conftest,
        "_SUITE_STARTED_AT",
        time.perf_counter() - conftest.SUITE_BUDGET_S - 1,
    )
    for existing in (1, 3):
        session = _stub_session(existing, reporter)
        conftest.pytest_sessionfinish(session, exitstatus=existing)
        assert session.exitstatus == existing


def test_the_suite_is_independent_of_the_users_language_file():
    """The suite neither reads nor writes the user's real ~/.v2m_lang.

    The conftest session fixture sandboxes the file and forces French,
    giving every run the same starting language regardless of user settings.
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
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
        ):
            continue
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if first.value not in tables["fr"] or first.value not in tables["en"]:
                offenders.append(f"{rel}:{node.lineno}: {first.value!r}")
    return offenders


def _input_status_offenders(rel: str, source: str, tables: dict) -> list[str]:
    """Literal keys passed to `_set_input_state()` missing from one table.

    The live page names its microphone-status message by KEY and shows it
    through the indicator's own `set_input_state`, so the tr()-literal guard
    never sees these four: a missing entry printed the raw key under the
    device name.
    """
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "_set_input_state" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if first.value not in tables["fr"] or first.value not in tables["en"]:
                offenders.append(f"{rel}:{node.lineno}: {first.value!r}")
    return offenders


def _docstring_offenders(path: Path) -> list[tuple[int, str]]:
    """French docstrings of `path` (accents or the word list), canonical exempted."""
    hits: list[tuple[int, str]] = []
    for const in _docstring_consts(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(const.value, str):
            continue
        text = const.value
        if _has_french(text):
            hits.append((const.lineno, text))
    return hits


def _comment_offenders(path: Path) -> list[tuple[int, str]]:
    """French comments of `path` (accents or the word list), canonical exempted."""
    hits: list[tuple[int, str]] = []
    source = path.read_text(encoding="utf-8")
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and _has_french(tok.string):
                hits.append((tok.start[0], tok.string))
    except tokenize.TokenError:
        pass  # an unparseable tail hides no comment the parser could read
    return hits


def test_no_french_comments_or_docstrings_outside_i18n():
    """No French comment or docstring in src/ or scripts/ — guard 7's border."""
    offenders: list[str] = []
    for base in ("src/v2m", "scripts"):
        for path in sorted((REPO / base).rglob("*.py")):
            if path.name == "i18n.py":
                continue
            for lineno, text in _docstring_offenders(path) + _comment_offenders(path):
                offenders.append(f"{path}:{lineno}: {text[:80]!r}")
    assert not offenders, (
        "French comments/docstrings in src/scripts — both are in ENGLISH "
        "(session rule, 2026-09-28 sweep; unaccented words caught 2026-10-02):\n  "
        + "\n  ".join(offenders)
    )


def test_french_docstring_guard_bites():
    """Guard 7 has TEETH: a French docstring AND a French comment bite."""
    tmp = Path(tempfile.mkdtemp())
    bad = tmp / "bad.py"
    bad.write_text(
        "def f():\n"
        '    """Renvoie le tempo détecté.\n\n'
        "    Retourne None sinon.\n"
        '    """\n'
        "    return None\n",
        encoding="utf-8",
    )
    assert _docstring_offenders(bad), "the guard does not bite?"
    # an unaccented French DOCSTRING (the guide_capture shape) bites too
    plain = tmp / "plain.py"
    plain.write_text(
        'def f():\n    """Raffine le repere humain sur la courbe."""\n    return None\n',
        encoding="utf-8",
    )
    assert _docstring_offenders(plain), "unaccented French slipped through"
    # a French COMMENT is on the same border (comments were swept too)
    cmt = tmp / "cmt.py"
    cmt.write_text("x = 1  # aucune plage ne correspond\n", encoding="utf-8")
    assert _comment_offenders(cmt), "a French comment does not bite?"
    # canonical vocabulary inside a docstring is exempted (as guard 3)
    ok = tmp / "ok.py"
    ok.write_text(
        "def f():\n"
        '    """Label this duration as a noire pointée.\n\n'
        "    Sol-fa names stay canonical: Do, Ré, Mi.\n"
        '    """\n'
        "    return None\n",
        encoding="utf-8",
    )
    assert not _docstring_offenders(ok), "canonical vocabulary must pass"


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
            offenders += _tr_key_offenders(rel, path.read_text(encoding="utf-8"), i18n.STRINGS)
    assert not offenders, (
        "tr() keys missing from the fr/en tables — the fallback would render "
        "the raw key to the user; add the key to BOTH tables:\n  " + "\n  ".join(offenders)
    )


def test_every_input_status_key_exists_in_both_language_tables():
    """The microphone status shows a TRANSLATED sentence, never a raw key."""
    from v2m.core import i18n

    offenders: list[str] = []
    for path in sorted((REPO / "src/v2m/gui").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        offenders += _input_status_offenders(
            rel, path.read_text(encoding="utf-8"), i18n.STRINGS
        )
    assert not offenders, (
        "_set_input_state() keys missing from the fr/en tables — the live page "
        "would print the raw key:\n  " + "\n  ".join(offenders)
    )


def test_input_status_key_guard_bites():
    """The status guard has teeth on a missing key and spares a real one."""
    from v2m.core import i18n

    bad = 'self._set_input_state("ui mic absent")\n'
    assert _input_status_offenders("live_page.py", bad, i18n.STRINGS)
    ok = 'self._set_input_state("ui mic waiting")\n'
    assert not _input_status_offenders("live_page.py", ok, i18n.STRINGS)


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
    dyn = "from v2m.core.i18n import tr\nmsg = tr(key)\n"
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


# --------------------------------------------------------------------------- #
# Guard 8: every subprocess a test boots is audited against ~/.v2m_lang
# --------------------------------------------------------------------------- #

SUBPROCESS_LAUNCHERS = frozenset(
    {"run", "Popen", "check_output", "check_call", "getoutput", "getstatusoutput"}
)

# The audited whitelist: test file -> (declared kind, why ~/.v2m_lang is
# safe). Kinds: "git" (git plumbing only), "code" ([sys.executable, "-c",
# payload]), "script:<name>.py" (a v2m script — its language-safety is
# machine-checked below). A NEW subprocess in any test fails the guard
# until it is audited and listed here; this guard file's own `git
# ls-files` is audited like any other.
SUBPROCESS_WHITELIST: dict[str, tuple[str, str]] = {
    "test_docs_guard.py": (
        "git",
        "git ls-files plumbing only (cwd=REPO): no v2m code runs, "
        "no language state is read or written",
    ),
    "test_metronome.py": (
        "code",
        "the qt-free proof probe: a fresh interpreter importing "
        "v2m.core.metronome ONLY (machine-checked below) — metronome "
        "imports no i18n, so nothing touches the persisted language",
    ),
}


def _is_sys_executable(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "executable"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _subprocess_launches(source: str) -> list[tuple[int, str]]:
    """(lineno, classified argv) of every subprocess launch in a source.

    Classification is deliberately narrow: a launch the whitelist can
    review must carry its argv as a LITERAL list whose head is a constant
    ('git') or sys.executable followed by a constant ('-c' or a .py path).
    Anything else comes back "unreviewed" and fails the guard — the
    author rewrites it into a reviewable shape instead of exempting it.
    """
    tree = ast.parse(source)
    launches: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in SUBPROCESS_LAUNCHERS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ):
            continue
        argv = node.args[0] if node.args else None
        first = argv.elts[0] if isinstance(argv, ast.List) and argv.elts else None
        second = argv.elts[1] if isinstance(argv, ast.List) and len(argv.elts) > 1 else None
        if isinstance(first, ast.Constant) and first.value == "git":
            launches.append((node.lineno, "git"))
        elif (
            _is_sys_executable(first)
            and isinstance(second, ast.Constant)
            and isinstance(second.value, str)
        ):
            if second.value == "-c":
                launches.append((node.lineno, "code"))
            elif second.value.endswith(".py"):
                launches.append((node.lineno, f"script:{Path(second.value).name}"))
            else:
                launches.append((node.lineno, "unreviewed (boot target is not .py/-c)"))
        else:
            launches.append((node.lineno, "unreviewed (argv is not a literal list)"))
    return launches


def _i18n_touches(path: Path) -> bool:
    """Does this module CALL get/set/toggle_language (the persisted state)?"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("i18n")
        for alias in node.names
        if alias.name in {"get_language", "set_language", "toggle_language"}
    }
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
        for node in ast.walk(tree)
    )


def _imports_i18n(path: Path) -> bool:
    """Does this module import i18n AT ALL (the qt-free probe's claim)?"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        (isinstance(node, ast.ImportFrom) and (node.module or "").endswith("i18n"))
        or (
            isinstance(node, ast.Import)
            and any(alias.name.endswith("i18n") for alias in node.names)
        )
        for node in ast.walk(tree)
    )


def _subprocess_offenders_for(
    rel: str,
    launches: list[tuple[int, str]],
    scripts_dir: Path,
    whitelist: dict[str, tuple[str, str]] | None = None,
) -> list[str]:
    """One test file's launch audit against the whitelist (injectable for bites)."""
    whitelist = SUBPROCESS_WHITELIST if whitelist is None else whitelist
    if not launches:
        return []
    if rel not in whitelist:
        kinds = ", ".join(sorted({kind for _, kind in launches}))
        return [
            f"{rel}: boots a subprocess ({kinds}) — a fresh interpreter reads/writes "
            "the user's real ~/.v2m_lang (the conftest sandbox is process-local); "
            "audit it, then whitelist it in SUBPROCESS_WHITELIST with justification"
        ]
    kind, _why = whitelist[rel]
    offenders: list[str] = []
    for lineno, actual in launches:
        # for "script:" the whitelist fixes the SHAPE only — the argv's
        # actual script is machine-checked below, whichever it names
        shape_ok = (
            actual == kind
            if kind in ("git", "code")
            else (actual.startswith("script:") and kind.startswith("script:"))
        )
        if not shape_ok:
            offenders.append(
                f"{rel}:{lineno}: boots {actual!r} but is whitelisted as {kind!r} "
                "— update the whitelist after re-auditing"
            )
        elif actual.startswith("script:"):
            script = scripts_dir / actual.split(":", 1)[1]
            if _i18n_touches(script):
                offenders.append(
                    f"{rel}:{lineno}: {actual} CALLS the language state — run it "
                    "in-process (like test_profile_analysis) or sandbox LANG_FILE"
                )
    return offenders


def _whitelist_claim_offenders() -> list[str]:
    """The justifications are re-verified, not hoped for."""
    out: list[str] = []
    if "test_metronome.py" in SUBPROCESS_WHITELIST and _imports_i18n(
        REPO / "src" / "v2m" / "core" / "metronome.py"
    ):
        out.append(
            "test_metronome.py's whitelist claim is void: "
            "src/v2m/core/metronome.py now imports i18n — re-audit the probe"
        )
    return out


# --------------------------------------------------------------------------- #
# Guard 8 (production): every subprocess the PRODUCT boots is audited too
# --------------------------------------------------------------------------- #

# module path relative to src/v2m (or scripts/) -> why the launch cannot
# touch the persisted language. The launch itself is machine-checked:
# its argv must be a literal list whose head is a non-interpreter — a
# bare command (an external tool), never sys.executable/-c/a .py path.
PRODUCTION_WHITELIST: dict[str, tuple[str, str]] = {
    "core/pdf_export.py": (
        "external-tool",
        "runs the user's engraver (MuseScore/LilyPond) on a .musicxml — "
        "an external binary, never an interpreter: it cannot read or "
        "write ~/.v2m_lang",
    ),
}


def _production_launches(path: Path) -> list[tuple[int, str]]:
    """Classify one PRODUCTION file's subprocess launches against the audit.

    A production argv is auditable when its head is a CONSTANT string:
    a bare external command (pdf_export's engraver cmd, resolved from
    PATH/install dirs). A sys.executable/-c or a .py target is an
    interpreter boot BY the product — refused on principle (re-derive in
    process, or document a real interpreter need). A non-literal argv
    (a variable, a comprehension) is refused as unreviewable: the author
    reshapes it rather than exempting it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in SUBPROCESS_LAUNCHERS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ):
            continue
        argv = node.args[0] if node.args else None
        # the list may be built by name before the call (pdf_export's
        # engraver argv): audit the LIST NODE, wherever it was assigned —
        # its head still decides external-tool vs interpreter boot
        if isinstance(argv, ast.Name):
            argv = _literal_argv(tree, argv.id)
        first = argv.elts[0] if isinstance(argv, ast.List) and argv.elts else None
        # external tool = a constant command OR a command VARIABLE (the
        # resolved engraver path); interpreter boot = sys.executable, a
        # '-c' payload or a .py target — refused on principle. Anything
        # else (a call, a bare string argv) is unreviewable.
        head_is_command = isinstance(first, (ast.Constant, ast.Name)) and not (
            isinstance(first, ast.Constant)
            and (
                not isinstance(first.value, str)
                or first.value == "-c"
                or first.value.endswith(".py")
            )
        )
        if head_is_command and not _is_sys_executable(first):
            out.append((node.lineno, "external-tool"))
        else:
            out.append((node.lineno, "unreviewed (not a constant external command)"))
    return out


def _literal_argv(tree: ast.AST, name: str) -> ast.AST | None:
    """The literal list assigned to `name` anywhere in the module (module
    level or any function scope) — None when the argv is not a literal."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.List)
        ):
            return node.value
    return None


def test_every_production_subprocess_site_is_audited():
    """The product boots a process only for an audited, non-interpreter reason.

    Mirror of the test-side guard: pdf_export's engraver is today's only
    launch; a NEW production subprocess (a helper interpreter, a shelled
    converter, os.system alias) fails until it earns its whitelist line.
    """
    offenders: list[str] = []
    for base in (REPO / "src" / "v2m", REPO / "scripts"):
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            # strip the src/v2m prefix (scripts/ paths stay whole)
            key = rel[len("src/v2m/") :] if rel.startswith("src/v2m/") else rel
            launches = _production_launches(path)
            if not launches:
                continue
            if key not in PRODUCTION_WHITELIST:
                kinds = ", ".join(sorted({kind for _, kind in launches}))
                offenders.append(
                    f"{rel}: production subprocess ({kinds}) — audit it against "
                    "~/.v2m_lang and whitelist it in PRODUCTION_WHITELIST"
                )
                continue
            kind, _why = PRODUCTION_WHITELIST[key]
            for lineno, actual in launches:
                if actual != kind:
                    offenders.append(
                        f"{rel}:{lineno}: boots {actual!r} but is whitelisted "
                        f"as {kind!r} — re-audit and update PRODUCTION_WHITELIST"
                    )
    assert not offenders, "an unaudited PRODUCTION subprocess:\n  " + "\n  ".join(offenders)


def test_production_subprocess_guard_bites():
    """The production half of guard 8 has TEETH: stranger module, drift."""
    # a whitelisted module that boots an INTERPRETER is refused: the
    # product must not need a second of itself (build in memory, no CRLF
    # coupling to the real file's line endings)
    interpreter_probe = (
        "import subprocess, sys\n"
        "cmd = 'mscore'\n"
        "argv = [sys.executable, '-c', 'print(1)']\n"
        "subprocess.run(argv)\n"
    )
    tmp = Path(tempfile.mkdtemp())
    probe = tmp / "probe.py"
    probe.write_text(interpreter_probe, encoding="utf-8")
    assert _production_launches(probe) == [
        (
            4,
            "unreviewed (not a constant external command)",
        )
    ]
    # a stranger module with an external-tool launch is refused until whitelisted
    stranger = tmp / "stranger.py"
    stranger.write_text(
        "import subprocess\nsubprocess.run(['magick', 'a.png', 'b.png'])\n",
        encoding="utf-8",
    )
    stranger_launches = _production_launches(stranger)
    assert stranger_launches and stranger_launches[0][1] == "external-tool"
    # a NAME-built argv is followed to its literal list (pdf_export's shape)
    named = tmp / "named.py"
    named.write_text(
        "import subprocess\n"
        "cmd = 'mscore'\n"
        "args = [cmd, '-o', 'out.pdf', 'in.musicxml']\n"
        "subprocess.run(args)\n",
        encoding="utf-8",
    )
    assert _production_launches(named) == [(4, "external-tool")]
    # a NON-list argv is refused: unreviewable, reshape it (a command
    # VARIABLE as list head is fine — pdf_export's shape — a shell string is not)
    dynamic = tmp / "dynamic.py"
    dynamic.write_text(
        "import subprocess\ndef go(args):\n    subprocess.run(args)\n",
        encoding="utf-8",
    )
    assert _production_launches(dynamic) == [
        (
            3,
            "unreviewed (not a constant external command)",
        )
    ]
    # the REAL files stay clean: pdf_export classifies as whitelisted kind
    real = _production_launches(REPO / "src" / "v2m" / "core" / "pdf_export.py")
    assert real and all(kind == "external-tool" for _, kind in real)
    """Guard 8: a test boots an interpreter only for an audited reason.

    The conftest language sandbox is process-local: anything booted in a
    fresh interpreter sees the user's REAL ~/.v2m_lang (read AND write —
    set_language persists). One md5 of that file before/after a full
    suite proved the whole net clean (2026-09-28); this guard keeps the
    property structural: no new boot without an audit entry.
    """
    offenders: list[str] = []
    scripts_dir = REPO / "scripts"
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        offenders += _subprocess_offenders_for(
            path.name, _subprocess_launches(path.read_text(encoding="utf-8")), scripts_dir
        )
    offenders += _whitelist_claim_offenders()
    assert not offenders, "an unaudited subprocess (the ~/.v2m_lang hazard):\n  " + "\n  ".join(
        offenders
    )


def test_subprocess_guard_bites():
    """Guard 8 has TEETH: stranger files, drifting kinds, void claims."""
    tmp = Path(tempfile.mkdtemp())
    # 1. an unlisted test file that boots a subprocess is refused
    assert _subprocess_offenders_for("test_stranger.py", [(12, "code")], tmp)
    # a file with no launch is never flagged
    assert not _subprocess_offenders_for("test_stranger.py", [], tmp)
    # 2. a whitelisted file drifting from its declared kind is refused
    assert _subprocess_offenders_for("test_metronome.py", [(7, "git")], tmp)
    assert not _subprocess_offenders_for("test_metronome.py", [(7, "code")], tmp)
    # 3. a whitelisted SCRIPT that calls the language state is refused;
    #    a script that only translates is safe (the whitelist is injected:
    #    the guard file itself is whitelisted "git" and must stay so)
    toucher = tmp / "toucher.py"
    toucher.write_text(
        "from v2m.core.i18n import set_language\nset_language('en')\n",
        encoding="utf-8",
    )
    assert _i18n_touches(toucher)
    fake_script_whitelist = {"test_x.py": ("script:probe.py", "audited probe")}
    assert _subprocess_offenders_for(
        "test_x.py", [(9, "script:toucher.py")], tmp, fake_script_whitelist
    )
    clean = tmp / "clean.py"
    clean.write_text("from v2m.core.i18n import tr\nprint(tr('key tempo'))\n", encoding="utf-8")
    assert not _i18n_touches(clean)
    assert not _subprocess_offenders_for(
        "test_x.py", [(9, "script:clean.py")], tmp, fake_script_whitelist
    )
    # 4. the qt-free claim is verified against the REAL metronome today
    assert not _imports_i18n(REPO / "src" / "v2m" / "core" / "metronome.py")


# --- guard 9: a widget is constructed where its screen is owned -------------
# ADR 0010's pattern, complete (results, live, finder are Page modules):
# the construction sites stay auditable by MODULE. A widget born in a
# helper, a dialog or any fourth place re-opens "who owns this screen?"
# — the drift the three-slice extraction just closed.
WIDGET_OWNERS = {
    # the shared library: every reusable view, its helpers and its styles
    "gui/widgets.py": "shared widget library",
    "gui/staff.py": "shared staff view (vector, no music font)",
    # the Page modules: widgets + behavior in one place
    "gui/results_page.py": "results Page module",
    "gui/live_page.py": "live Page module",
    "gui/finder_page.py": "finder Page module",
    # page modules own their screen widgets and behavior
    "gui/home_page.py": "home Page module",
    "gui/calibration_card.py": "home page's calibration card (in-page guided flow)",
    # the composition window: the stack, the central widget, the banner
    "gui/main_window.py": "window composition (stack, central, banner)",
}
# Q-names that are NOT screens: event sources and handles a non-GUI owner
# may create. QTimer is load-bearing in core/metronome.qt_scheduler — the
# lazily-imported, injected GUI binding (a GC'd timer dies, the caller
# keeps the handle); it owns no screen, so it passes.
NON_WIDGET_QNAMES = {"QTimer"}


def _widget_constructions(path: Path) -> list[tuple[int, str]]:
    """Widget CONSTRUCTION calls of a source, with a provenance verdict.

    A call whose callee (dotted path or bare name) starts with Q, or is a
    pyqtgraph item (`pg.…`), is a widget under construction. The Q-prefix
    is sound HERE: PyQt6 names every Qt class with Q, and this codebase's
    own classes are never Q-prefixed. Non-widget callees (NotePlayer,
    BeatGrid, Path, Verdict…) never match. Layouts, timers and other
    non-screen Q-names pass through (NON_WIDGET_QNAMES) — only Page
    modules and the window may build screens.
    """
    verdicts: list[tuple[int, str]] = []
    try:
        key = path.relative_to(REPO / "src" / "v2m").as_posix().replace("\\", "/")
    except ValueError:  # a probe outside the repo: nothing owns it
        key = ""
    kind = WIDGET_OWNERS.get(key, "unowned")
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name: str | None = None
        if isinstance(callee, ast.Name):
            name = callee.id
        elif isinstance(callee, ast.Attribute):
            parts = []
            while isinstance(callee, ast.Attribute):
                parts.append(callee.attr)
                callee = callee.value
            if isinstance(callee, ast.Name):
                parts.append(callee.id)
            name = ".".join(reversed(parts))
        if name is None or name in NON_WIDGET_QNAMES:
            continue
        if name.startswith("Q") or name.startswith("pg."):
            verdicts.append((node.lineno, kind))
    return verdicts


def test_widgets_are_built_where_their_screen_is_owned():
    """Guard 9: widget construction lives in the library, a Page, or the window.

    ADR 0010's pattern is COMPLETE (three Page modules); this keeps it
    complete — a widget born in a helper re-opens the ownership question
    the extraction answered, so the fourth place fails until it joins a
    Page module or earns its WIDGET_OWNERS line with a reason.
    """
    offenders: list[str] = []
    for path in sorted((REPO / "src" / "v2m").rglob("*.py")):
        for lineno, kind in _widget_constructions(path):
            if kind != "unowned":
                continue
            rel = path.relative_to(REPO / "src" / "v2m").as_posix().replace("\\", "/")
            offenders.append(
                f"src/v2m/{rel}:{lineno}: widget built where no screen is owned — "
                "move it into a Page module (gui/*_page.py) or whitelist "
                "the module in WIDGET_OWNERS with its reason"
            )
    assert not offenders, "a widget constructed outside its owner:\n  " + "\n  ".join(offenders)


def test_widget_guard_bites():
    """Guard 9 has TEETH: a stranger module building a widget is refused."""
    # a bare Q-construction in an unowned module is flagged, with its line
    helper = Path(tempfile.mkdtemp()) / "helper.py"
    helper.write_text(
        "from PyQt6.QtWidgets import QLabel\ndef badge():\n    return QLabel('boom')\n",
        encoding="utf-8",
    )
    assert _widget_constructions(helper) == [(3, "unowned")]
    # pyqtgraph items count too (dotted callee)
    plotter = Path(tempfile.mkdtemp()) / "plotter.py"
    plotter.write_text(
        "import pyqtgraph as pg\nw = pg.PlotWidget()\n",
        encoding="utf-8",
    )
    assert _widget_constructions(plotter) == [(2, "unowned")]
    # non-widget classes are invisible to the guard (no Q-prefix, no pg. path)
    plain = Path(tempfile.mkdtemp()) / "plain.py"
    plain.write_text(
        "from pathlib import Path\n"
        "from ..core.grid import BeatGrid\n"
        "p = Path('.')\n"
        "g = BeatGrid(120.0)\n",
        encoding="utf-8",
    )
    assert _widget_constructions(plain) == []
    # and the real tree stays clean: every construction site is an owner,
    # and the timer pass-through keeps core/metronome's qt_scheduler legal
    assert all(
        kind != "unowned"
        for path in sorted((REPO / "src" / "v2m").rglob("*.py"))
        for _, kind in _widget_constructions(path)
    )
    assert not _widget_constructions(REPO / "src" / "v2m" / "core" / "metronome.py")


def test_shell_scripts_keep_lf_for_the_linux_image():
    """Guard 10: the `.gitattributes` pin exists and the entry point has no CR."""
    attrs = (REPO / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attrs, (
        "without the pin, a Windows checkout (autocrlf) hands bash a CRLF "
        "docker-entrypoint.sh and the Linux image refuses to start"
    )
    entrypoint = (REPO / "docker-entrypoint.sh").read_bytes()
    assert b"\r" not in entrypoint, "docker-entrypoint.sh carries CR line endings"
