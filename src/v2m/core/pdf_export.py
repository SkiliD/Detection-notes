"""Score PDF export via MuseScore or LilyPond (automatic detection).

One adapter is enough: pick whichever tool is installed on the machine and
hand it the MusicXML already produced by the pipeline. Without any tool, a
clear error states what was searched and where to install it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .i18n import tr

TOOLS = [
    ("MuseScore", "musescore", ["musescore4", "musescore3", "musescore", "MuseScore4.exe", "MuseScore3.exe"]),
    ("LilyPond", "lilypond", ["lilypond"]),
]

# The Windows installers do NOT add themselves to the PATH: probe the
# standard install locations too (machine-wide and per-user — MuseScore 4
# defaults to the latter). Keyed by the PATH command they complete.
_STANDARD_DIRS = {
    "musescore4": ["MuseScore 4", "bin", "MuseScore4.exe"],
    "musescore3": ["MuseScore 3", "bin", "MuseScore3.exe"],
}


def _installed_bases() -> list[Path]:
    bases = []
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        v = os.environ.get(var)
        if v:
            bases.append(Path(v))
    local = os.environ.get("LocalAppData")
    if local:
        bases.append(Path(local) / "Programs")
    return bases


class PdfExportError(RuntimeError):
    """A PDF was requested but could not be produced."""


class PdfToolMissing(PdfExportError):
    """No engraving tool installed: the PDF is skipped, the rest is fine.

    Distinct from a conversion failure: the caller decides whether a missing
    tool is an error (it is not one for the CLI, which exits 0 there) while
    an install hint reaches the user either way.
    """


def find_notation_tool() -> tuple[str, str] | None:
    """Return (name, command) of the first available tool, else None.

    Beyond the PATH, the standard Windows install locations are probed:
    MuseScore installed but invisible to `shutil.which` is the common case
    on a real machine (the tool WAS there, the PDF was skipped anyway).
    """
    for name, pkg, cmds in TOOLS:
        for cmd in cmds:
            if shutil.which(cmd):
                return name, cmd
            parts = _STANDARD_DIRS.get(cmd)
            if parts:
                for base in _installed_bases():
                    exe = base.joinpath(*parts)
                    if exe.exists():
                        return name, str(exe)
    return None


def install_hint() -> str:
    return tr("err pdf install hint")


def export_pdf(musicxml_path: str | Path, pdf_path: str | Path) -> str:
    """Convert a .musicxml to PDF. Raise PdfExportError when impossible."""
    tool = find_notation_tool()
    if tool is None:
        raise PdfExportError(tr("err pdf no tool").format(hint=install_hint()))
    name, cmd = tool
    musicxml_path = Path(musicxml_path)
    pdf_path = Path(pdf_path)
    if not musicxml_path.exists():
        raise PdfExportError(tr("err file not found").format(path=musicxml_path))

    # the argv stays a LITERAL list at the run() call site: the docs
    # guard audits every production subprocess launch, and a reviewable
    # launch carries its argv in line (the head is the external tool's
    # resolved command — never sys.executable, never a .py payload)
    out_arg = str(pdf_path.with_suffix("")) if name == "LilyPond" else str(pdf_path)
    try:
        proc = subprocess.run(
            [cmd, "-o", out_arg, str(musicxml_path)],
            capture_output=True, text=True,
            # the tool's stderr is decoded in the LOCAL encoding, strict by
            # default: a UTF-8-emitting MuseScore under a cp1252 console
            # once raised UnicodeDecodeError OUTSIDE the PdfExportError
            # contract (an unhandled crash for a cosmetic detail line).
            errors="replace",
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        raise PdfExportError(tr("err pdf not runnable").format(cmd=cmd, hint=install_hint())) from None
    except subprocess.TimeoutExpired:
        raise PdfExportError(tr("err pdf timeout").format(name=name)) from None

    if proc.returncode != 0 or not pdf_path.exists():
        # the tool's own stderr is in the tool's language; only OUR fallback
        # line and the message frame must follow the UI language
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = detail[-1] if detail else tr("err pdf exit code").format(code=proc.returncode)
        raise PdfExportError(tr("err pdf conversion").format(name=name, msg=msg))
    return str(pdf_path)
