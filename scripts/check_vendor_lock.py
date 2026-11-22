"""Check the vendored third-party folder against the hash `vendor-lock.json` pins.

The ONE vendored third-party folder lives in
`vendor/make-interfaces-feel-better/`; `vendor-lock.json` records the upstream
commit it was copied from and the folder digest (`computedHash`) computed at
vendor time. `tests/test_docs_guard.py` recomputes the same digest, so this
check answers the same question on a machine without pytest or the network.

The digest framing is fixed: in path order, each file contributes its
path, NUL, the decimal byte length, NUL, then its bytes. Text files are
normalised to LF first, because a checkout may hand back CRLF for the same
committed content.

Usage:
    ./.venv/Scripts/python.exe scripts/check_vendor_lock.py

Exit code 0 means the folder matches the lock; 1 means it does not, or the
folder, the lock or its `computedHash` cannot be read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENDORED_NAME = "make-interfaces-feel-better"
VENDORED_DIR = REPO / "vendor" / VENDORED_NAME
LOCK = REPO / "vendor-lock.json"


def folder_hash(files: list[tuple[str, bytes]]) -> str:
    """The vendored-folder digest: path, NUL, byte length, NUL, content."""
    digest = hashlib.sha256()
    for relative, content in sorted(files, key=lambda item: item[0].encode("utf-8")):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(content)
    return digest.hexdigest()


def folder_files(folder: Path) -> list[tuple[str, bytes]]:
    """The files a vendored folder ships, text normalised to LF.

    `.git` and `node_modules` are not part of the vendored copy, so a checkout
    that materialised them must not move the digest.
    """
    files: list[tuple[str, bytes]] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(folder)
        if any(part in {".git", "node_modules"} for part in relative.parts):
            continue
        content = path.read_bytes()
        if b"\x00" not in content:
            content = content.replace(b"\r\n", b"\n")
        files.append((relative.as_posix(), content))
    return files


def locked_hash(lock: Path = LOCK, name: str = VENDORED_NAME) -> str:
    """The `computedHash` the lock records for the vendored folder."""
    data = json.loads(lock.read_text(encoding="utf-8"))
    return data["vendored"][name]["computedHash"]


def _label(path: Path) -> str:
    """Name a path as the report should: repository-relative when it is inside."""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def main(
    vendored_dir: Path = VENDORED_DIR,
    lock: Path = LOCK,
    name: str = VENDORED_NAME,
) -> int:
    """Compare the folder with the lock; 0 matches, 1 does not.

    The paths are PARAMETERS, defaulting to the vendored pair, so the check can
    be run against a temporary folder and lock: a comparison that can only be
    wrong about one folder is a comparison nobody can test.
    """
    if not vendored_dir.is_dir():
        print(
            f"vendor lock: the vendored folder is gone: {_label(vendored_dir)}",
            file=sys.stderr,
        )
        return 1
    try:
        expected = locked_hash(lock, name)
        actual = folder_hash(folder_files(vendored_dir))
    except (OSError, KeyError, ValueError) as exc:
        print(f"vendor lock: cannot read the lock or the folder: {exc}", file=sys.stderr)
        return 1
    if actual != expected:
        print(
            f"vendor lock: {_label(vendored_dir)} no longer matches "
            f"vendor-lock.json ({actual[:12]} != {expected[:12]}); re-vendor from "
            "the upstream source instead of editing the copy",
            file=sys.stderr,
        )
        return 1
    print(f"vendor lock: {name} matches vendor-lock.json ({expected})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
