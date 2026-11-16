"""The bilingual README pair: the mirror is a CONTRACT, not an intention.

`README.md` (English, primary) and `README.fr.md` (French) must stay the
same document: same section skeleton (same heading-level sequence — the
texts themselves are translated), same documented CLI flags, same
referenced scripts, cross-links at the top. The silent drift — a flag
documented in one language only, a section added on one side — is what
this pin refuses; it is the same philosophy as the fr/en key parity of
i18n (tests/test_gui_i18n.py), and it gives teeth to the "README
obligatoire" rule: without this test, the mirrored sections could drift.

The mirror is deliberately structural, not literal: the prose is
translated and the quoted test count may move freely. The example FILE
NAMES (`chant.wav`, `rapport.json`…) are not prose: they must be
identical in both languages — an example that copy-pastes differently
by language is a bug.

The docs also document only the REAL: every README flag must exist in
`build_parser`, and conversely — otherwise the docs announce an option
that does not exist, or the CLI gains an option nobody sees. The flags
the diagnostic SCRIPTS define themselves (`--golden`, `--no-warmup`…)
must in addition be explained in the prose of both languages: they have
no visible help anywhere else.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

# CLI flags (--min-conf): preceded by a non-word character so the match
# cannot swallow a word's hyphens or a sentence's punctuation.
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]+)\b")
SCRIPT = re.compile(r"scripts/[a-z_]+\.py")
HEADING = re.compile(r"^(#{1,3}) ", re.M)
# Every line that RUNS something, anchored at the start of a line: `v2m`
# itself, the interpreter the Install section names per platform
# (`py`/`python3.12`/`python`), the venv activation steps and
# `pytest`. Both READMEs must carry the same ones.
COMMAND = re.compile(
    r"^(?:v2m\b|py\b|python[\d.]*\b|pytest\b|pip\b|source\b"
    r"|QT_QPA_PLATFORM=\S+\s+python\b|\.venv\\Scripts\\activate).*$",
    re.M,
)
COMMENT = re.compile(r"#.*$")

# flags internal to the program itself, never to be documented:
# `--version` (version action) and `--help` (argparse's help).
# `-o` is the short form of `--output` (CLI analyze): documenting it
# once is enough — both forms name the same option.
INTERNAL_FLAGS = {"--version", "--help"}
SHORT_ALIASES = {"-o"}


def _skeleton(text: str) -> tuple[list[int], set[str], set[str]]:
    """A README's skeleton, language-independent.

    Headings are read outside code blocks (nested markdown examples are
    not sections); flags and scripts over the WHOLE text, blocks
    included — the command examples are the real documentation.
    """
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    levels = [len(h) for h in HEADING.findall(prose)]
    return levels, set(FLAG.findall(text)), set(SCRIPT.findall(text))


def test_readme_pair_is_mirror_structured():
    en = (REPO / "README.md").read_text(encoding="utf-8")
    fr = (REPO / "README.fr.md").read_text(encoding="utf-8")

    levels_en, flags_en, scripts_en = _skeleton(en)
    levels_fr, flags_fr, scripts_fr = _skeleton(fr)

    # same heading-level sequence (the texts are translated)
    assert levels_en == levels_fr, (
        "the two READMEs' sections diverged — update BOTH in the same "
        f"commit:\nEN : {levels_en}\nFR : {levels_fr}"
    )
    # same documented CLI flags
    assert flags_en == flags_fr, (
        "asymmetric CLI flags: "
        f"EN only : {sorted(flags_en - flags_fr)}, "
        f"FR only : {sorted(flags_fr - flags_en)}"
    )
    # same referenced scripts
    assert scripts_en == scripts_fr, (
        "asymmetric scripts: "
        f"EN only : {sorted(scripts_en - scripts_fr)}, "
        f"FR only : {sorted(scripts_fr - scripts_en)}"
    )


def test_language_switchers_cross_link():
    en = (REPO / "README.md").read_text(encoding="utf-8")
    fr = (REPO / "README.fr.md").read_text(encoding="utf-8")
    # English is primary: its first line points at the French
    assert "[Français](README.fr.md)" in en.splitlines()[0]
    assert "[English](README.md)" in fr.splitlines()[0]


def test_example_commands_are_mirrored():
    """The command examples carry the same values in both languages.

    A file name is not prose: `chant.wav` copy-pastes the same in French
    and English. We compare the SET of v2m / script commands of each
    README (order free: the prose decides where they live) and, inside
    each command, the sample values it carries.
    """
    en = (REPO / "README.md").read_text(encoding="utf-8")
    fr = (REPO / "README.fr.md").read_text(encoding="utf-8")
    # inline comments are translated prose: strip them
    commands_en = sorted(COMMENT.sub("", c).strip() for c in COMMAND.findall(en))
    commands_fr = sorted(COMMENT.sub("", c).strip() for c in COMMAND.findall(fr))
    assert commands_en == commands_fr, (
        "the command examples diverged: "
        f"EN only : {sorted(set(commands_en) - set(commands_fr))}, "
        f"FR only : {sorted(set(commands_fr) - set(commands_en))}"
    )


def _parser_flags() -> set[str]:
    """The REAL flags: the CLI's subcommands + the diagnostic scripts.

    Subparser flags live in the subparsers (not in the top-level
    parser._actions); `--golden`, `--no-warmup`… are defined by the
    scripts — they are just as real. `-h`/`--version`/`--lang` need not
    be documented.
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
    for script in ("score_accuracy.py", "profile_analysis.py", "confirm_s_sweep.py",
                   "liveseg_sweep.py",
                   "guide_capture.py", "reinterpret.py", "replay_takes.py",
                   "engine_drift_watch.py", "compare_takes.py"):
        text = (REPO / "scripts" / script).read_text(encoding="utf-8")
        flags.update(FLAG.findall(text))
    return flags - INTERNAL_FLAGS - SHORT_ALIASES - {"-h"}


def test_documented_flags_exist_in_cli_and_vice_versa():
    """The docs document only the real: README flags == CLI + scripts."""
    documented: set[str] = set()
    for name in ("README.md", "README.fr.md"):
        documented |= set(FLAG.findall((REPO / name).read_text(encoding="utf-8")))
    actual = _parser_flags()
    assert documented == actual, (
        f"docs ≠ real — documented only : {sorted(documented - actual)}, "
        f"undocumented : {sorted(actual - documented)}"
    )


def _script_flags() -> set[str]:
    """The flags the diagnostic scripts define themselves."""
    flags: set[str] = set()
    for script in ("score_accuracy.py", "profile_analysis.py", "confirm_s_sweep.py",
                   "liveseg_sweep.py",
                   "guide_capture.py", "reinterpret.py", "replay_takes.py",
                   "compare_takes.py"):
        text = (REPO / "scripts" / script).read_text(encoding="utf-8")
        flags.update(FLAG.findall(text))
    return flags


def test_script_flags_documented_with_semantics():
    """Every script flag is explained in the PROSE of both READMEs.

    Presence suffices for the CLI (build_parser's help is readable via
    --help), but a script flag has no visible help anywhere else: its
    semantics must live in the text, not only in an example to copy.
    Testable contract: the flag's token appears OUTSIDE code blocks, in
    each language — one sentence per flag, the way `--no-warmup`
    already does.
    """
    missing: dict[str, list[str]] = {}
    for name in ("README.md", "README.fr.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        missing[name] = sorted(f for f in _script_flags() if f not in prose)
    assert not any(missing.values()), (
        f"script flags without semantics in the prose (outside examples) : {missing} — "
        "one sentence per flag, in BOTH READMEs, like --no-warmup."
    )


# --------------------------------------------------------------------------- #
# Every path a README puts in backticks exists
# --------------------------------------------------------------------------- #

# What a README may name without it being on disk. Each entry is a REASON, not
# a convenience: a clone has none of these, and a typo must not be able to hide
# behind the list.
RUNTIME_ONLY = {
    # the artifacts of a run (gitignored, produced by the app or a script)
    "prises_live/",
    "prises_verite/",
    "demo.wav",  # `scripts/make_demo.py` regenerates it
    "chant.wav",  # the reference take, documented as gitignored
    # the interpreter of a local virtualenv, named per platform
    ".venv/Scripts/python.exe",
    ".venv/bin/python",
}

PATH_SUFFIXES = (
    ".md",
    ".py",
    ".json",
    ".wav",
    ".npz",
    ".js",
    ".mjs",
    ".yml",
    ".yaml",
    ".toml",
    ".cfg",
    ".txt",
    ".csv",
    ".xml",
    ".mid",
    ".musicxml",
    ".sh",
    ".bat",
    ".ps1",
    ".html",
    ".css",
)


def _archive_members() -> set[str]:
    """The files the page's ZIP carries, read from its OWN writer.

    `prise.wav` and `notes.json` exist only inside the archive the page builds
    in the browser, so a README may name them; taking the names from
    `web/bundle.js` rather than restating them here means a renamed member
    makes the README stale instead of silently tolerant.
    """
    source = (REPO / "web" / "bundle.js").read_text(encoding="utf-8")
    return set(re.findall(r'export const [A-Z_]+_NAME = "([^"]+)"', source))


def _cited_paths(text: str) -> list[str]:
    """The path-looking backticked spans of one README, in reading order."""
    cited: list[str] = []
    for span in re.findall(r"`([^`\n]+)`", text):
        token = span.strip()
        # `GET /health` is a route; a home path is not in the repository; a bare
        # suffix (`.json`) is prose about a format; globs and templates name no
        # one file.
        if not token or any(ch.isspace() for ch in token):
            continue
        if token.startswith("~"):
            continue
        if token.startswith(".") and "/" not in token:
            continue  # a bare suffix: prose about a format, not a file
        if any(ch in token for ch in "*{}<>|$"):
            continue
        if "/" not in token and not token.endswith(PATH_SUFFIXES):
            continue
        cited.append(token)
    return cited


def test_every_path_a_readme_names_exists():
    """A README must not send its reader to a file that is not there.

    The design notes (`AGENTS.md`, `CONTEXT.md`, `docs/adr/`) are named in the
    text AND explained as working-tree-only, so a reader knows where they
    stand; a path that exists in neither place is a dead reference, and nothing
    in the suite saw one until this test.
    """
    archive = _archive_members()
    assert archive, "web/bundle.js no longer names its archive members"

    cited: set[str] = set()
    missing: list[str] = []
    for name in ("README.md", "README.fr.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        for token in _cited_paths(text):
            cited.add(token)
            if token in RUNTIME_ONLY or token in archive:
                continue
            if not (REPO / token.rstrip("/")).exists():
                missing.append(f"{name}: `{token}`")

    # A net that matched nothing would pass forever: these READMEs do cite the
    # repository, so a low count means the detector stopped seeing it.
    assert len(cited) >= 15, f"the path net only saw {len(cited)} names: {sorted(cited)}"
    assert not missing, "a README names a path that does not exist:" + "".join(
        f"\n  {one}" for one in sorted(missing)
    )


def test_the_path_net_refuses_a_name_that_is_not_there():
    """The detector's own bite: it sees a missing path and ignores what it should."""
    assert "scripts/nowhere.py" in _cited_paths("run `scripts/nowhere.py` now")
    assert not (REPO / "scripts" / "nowhere.py").exists()
    assert _cited_paths("`GET /health`, `~/.v2m_theme`, `.json`, `v2m analyze`") == []
    # A directory the README names with its trailing slash is still a directory.
    assert "scripts/" in _cited_paths("the `scripts/` folder")
