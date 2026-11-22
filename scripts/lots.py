"""Rebuild one lot's index from a manifest, and prove its tests on a frozen export.

A lot is the set of files ONE commit carries, plus the tests that decide
whether that commit is good. Two questions come up every time such a set is
moved — which files belong together, and whether that set is green WITHOUT the
rest of a dirty working tree — and both were answered by hand, in commands
whose answers no longer exist once the terminal closes.

`scripts/lots.toml` is the manifest. Each lot names its `files` (what the commit
carries), its `tests` (what decides) and optionally its `takes` (gitignored
audio a test needs: copied from the checkout when present, named on the console
when absent, so a skip cannot pass for a green run).

Verification never trusts the working tree: it exports HEAD with `git archive`
into a temporary folder, copies the lot's files over it, and runs the tests
THERE. A lot that only passes because an uncommitted neighbour happens to sit
in the tree fails, which is the whole point of the exercise.

`--stage` rebuilds the lot's index (`git add` exactly the manifest's files)
before verifying it, and prints the index it produced. One lot at a time: an
index holds one commit's worth of changes, so staging two lots at once is the
incoherence this script exists to prevent.

Usage:
    python -X utf8 scripts/lots.py --list
    python -X utf8 scripts/lots.py
    python -X utf8 scripts/lots.py --lot gui-rings --stage
    python -X utf8 scripts/lots.py --dry-run

Exit 0 when every selected lot is green, 1 when one is red, 2 on a usage or
manifest error. Nothing is written outside the system temp folder, which is
removed unless `--keep` is given.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "scripts" / "lots.toml"

# The venv's console script, not `python -m pytest`: the same runner a reader
# invokes by hand, resolved once from PATH and then from the interpreter's own
# bin directory, so a machine that only added .venv/Scripts to nothing still
# finds it.
PYTEST = shutil.which("pytest") or str(Path(sys.executable).with_name("pytest"))
PYTEST_ARGS = ("-q", "-n", "0", "-rs")


class LotError(ValueError):
    """The manifest or the command line asks for something no lot can be."""


def load_lots(path: Path = MANIFEST) -> dict[str, dict[str, list[str]]]:
    """The manifest as {lot: {"files": [...], "tests": [...], "takes": [...]}}."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    lots = document.get("lots")
    if not isinstance(lots, dict) or not lots:
        raise LotError(f"{path.name} names no lot")
    loaded: dict[str, dict[str, list[str]]] = {}
    for name, lot in lots.items():
        if not isinstance(lot, dict):
            raise LotError(f"lot {name!r} is not a table")
        files = list(lot.get("files") or [])
        tests = list(lot.get("tests") or [])
        if not files or not tests:
            raise LotError(f"lot {name!r} needs both `files` and `tests`")
        loaded[name] = {
            "files": files,
            "tests": tests,
            "takes": list(lot.get("takes") or []),
        }
    return loaded


def absent(lot: dict[str, list[str]]) -> list[str]:
    """The lot's files and tests that are not on disk: a stale manifest."""
    return [name for name in lot["files"] + lot["tests"] if not (REPO / name).exists()]


def plan(test: str) -> str:
    """The command a lot's test runs with, printed by `--dry-run`.

    The lot names paths, not commands: a `.py` test runs through pytest and a
    `.mjs` one through `node --test`, so the page's tests and the suite's tests
    live in the same manifest without a second table to keep in step.
    """
    if test.endswith(".py"):
        return " ".join([PYTEST, test, *PYTEST_ARGS])
    if test.endswith(".mjs"):
        return f"node --test {test}"
    raise LotError(f"no runner for {test}: a lot's test is .py or .mjs")


def environment(export: Path) -> dict[str, str]:
    """The export's own package, and no display where the machine has none."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(export / "src")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def export_head(destination: Path) -> None:
    """The tracked tree at HEAD, without a single working-tree change."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")


def copy_lot(lot: dict[str, list[str]], export: Path) -> list[str]:
    """Put the lot's files over the export; report the takes it could not find."""
    notes = []
    for name in lot["files"]:
        target = export / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    for name in lot["takes"]:
        source = REPO / name
        if not source.exists():
            notes.append(f"{name} is absent: its test can only skip here")
            continue
        shutil.copy2(source, export / name)
    return notes


def last_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


def node_tally(output: str) -> str:
    """`pass 4, fail 0`: node's summary lines sit above its duration line."""
    tally = [
        line.strip().removeprefix("\u2139 ")
        for line in output.splitlines()
        if line.strip().startswith(("\u2139 pass", "\u2139 fail"))
    ]
    return ", ".join(tally) if tally else last_line(output)


def run_test(test: str, export: Path) -> tuple[bool, float, str]:
    """Run one test file in the export; report (green, seconds, its own tally)."""
    started = time.monotonic()
    if test.endswith(".mjs"):
        done = subprocess.run(
            ["node", "--test", test],
            cwd=export,
            env=environment(export),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        done = subprocess.run(
            [PYTEST, test, *PYTEST_ARGS],
            cwd=export,
            env=environment(export),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    elapsed = time.monotonic() - started
    output = f"{done.stdout}\n{done.stderr}"
    tally = node_tally(output) if test.endswith(".mjs") else last_line(output)
    return done.returncode == 0, elapsed, tally


def verify(name: str, lot: dict[str, list[str]], keep: bool) -> bool:
    """Export HEAD, lay the lot over it, run its tests; report each one."""
    work = Path(tempfile.mkdtemp(prefix=f"v2m-lot-{name}-"))
    try:
        export_head(work)
        notes = copy_lot(lot, work)
        print(f"lot {name}: {len(lot['files'])} fichiers sur un export de HEAD -> {work}")
        for note in notes:
            print(f"  note   {note}")
        green = True
        for test in lot["tests"]:
            ok, elapsed, tally = run_test(test, work)
            print(f"  {'ok   ' if ok else 'RED  '}  {test:<36} {elapsed:6.2f} s  {tally}")
            green = green and ok
        return green
    finally:
        if keep:
            print(f"  garde  {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


def stage(name: str, lot: dict[str, list[str]]) -> None:
    """Index exactly the lot's files, then show the index that produced."""
    subprocess.run(["git", "add", "--", *lot["files"]], cwd=REPO, check=True)
    shown = subprocess.run(
        ["git", "diff", "--cached", "--name-status"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"index = lot {name}")
    print(shown.stdout.strip() or "(vide)")


def describe(lots: dict[str, dict[str, list[str]]]) -> None:
    for name, lot in lots.items():
        print(f"{name}")
        for path in lot["files"]:
            print(f"  file   {path}")
        for path in lot["tests"]:
            print(f"  test   {path}")
        for path in lot["takes"]:
            print(f"  take   {path}  (ignore par git, copie quand il est la)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "lots").splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="le manifeste des lots")
    parser.add_argument("--lot", action="append", default=[], metavar="NAME", help="un lot")
    parser.add_argument("--stage", action="store_true", help="reindexer le lot d'abord")
    parser.add_argument("--dry-run", action="store_true", help="imprimer le plan, rien lancer")
    parser.add_argument("--list", action="store_true", help="imprimer les lots")
    parser.add_argument("--keep", action="store_true", help="garder l'export gele")
    args = parser.parse_args(argv)

    try:
        lots = load_lots(args.manifest)
    except (OSError, ValueError) as error:
        print(f"lots: {error}", file=sys.stderr)
        return 2

    names = args.lot or list(lots)
    unknown = [name for name in names if name not in lots]
    if unknown:
        known = ", ".join(lots)
        print(f"lots: aucun lot {', '.join(unknown)} (connus : {known})", file=sys.stderr)
        return 2

    if args.list:
        describe(lots)
        return 0

    stale = {name: absent(lots[name]) for name in names}
    stale = {name: missing for name, missing in stale.items() if missing}
    if stale:
        for name, missing in stale.items():
            print(f"lots: {name} nomme des chemins absents : {', '.join(missing)}", file=sys.stderr)
        return 2

    if args.stage and len(names) != 1:
        print(
            "lots: --stage prend UN lot ; un index porte un seul commit à la fois",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        for name in names:
            lot = lots[name]
            print(f"lot {name}: exporter HEAD, y poser {len(lot['files'])} fichiers, puis")
            for test in lot["tests"]:
                print(f"  {plan(test)}")
        return 0

    if args.stage:
        stage(names[0], lots[names[0]])

    every_green = True
    for name in names:
        green = verify(name, lots[name], args.keep)
        print(f"lot {name}: {'vert' if green else 'ROUGE'}")
        every_green = every_green and green
    return 0 if every_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
