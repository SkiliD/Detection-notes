"""Score PDF export via MuseScore or LilyPond (automatic detection).

One adapter is enough: pick whichever tool is installed on the machine and
hand it the MusicXML already produced by the pipeline. Without any tool, a
clear error states what was searched and where to install it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

TOOLS = [
    ("MuseScore", "musescore", ["musescore4", "musescore3", "musescore", "MuseScore4.exe", "MuseScore3.exe"]),
    ("LilyPond", "lilypond", ["lilypond"]),
]


class PdfExportError(RuntimeError):
    """A PDF was requested but could not be produced."""


class PdfToolMissing(PdfExportError):
    """No engraving tool installed: the PDF is skipped, the rest is fine.

    Distinct from a conversion failure: the caller decides whether a missing
    tool is an error (it is not one for the CLI, which exits 0 there) while
    an install hint reaches the user either way.
    """


def find_notation_tool() -> tuple[str, str] | None:
    """Return (name, command) of the first available tool, else None."""
    for name, pkg, cmds in TOOLS:
        for cmd in cmds:
            if shutil.which(cmd):
                return name, cmd
    return None


def install_hint() -> str:
    return (
        "Installez MuseScore (https://musescore.org) ou LilyPond "
        "(https://lilypond.org) puis vérifiez qu'il est dans le PATH."
    )


def export_pdf(musicxml_path: str | Path, pdf_path: str | Path) -> str:
    """Convert a .musicxml to PDF. Raise PdfExportError when impossible."""
    tool = find_notation_tool()
    if tool is None:
        raise PdfExportError(f"No engraving tool found (MuseScore, LilyPond). {install_hint()}")
    name, cmd = tool
    musicxml_path = Path(musicxml_path)
    pdf_path = Path(pdf_path)
    if not musicxml_path.exists():
        raise PdfExportError(f"Fichier introuvable : {musicxml_path}")

    if name == "LilyPond":
        cmd_args = [cmd, "-o", str(pdf_path.with_suffix("")), str(musicxml_path)]
    else:  # MuseScore
        cmd_args = [cmd, "-o", str(pdf_path), str(musicxml_path)]

    try:
        proc = subprocess.run(
            cmd_args, capture_output=True, text=True, timeout=120,
            check=False,
        )
    except FileNotFoundError:
        raise PdfExportError(f"{cmd} introuvable à l'exécution. {install_hint()}") from None
    except subprocess.TimeoutExpired:
        raise PdfExportError(f"{name} did not respond (timeout 120 s).") from None

    if proc.returncode != 0 or not pdf_path.exists():
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = detail[-1] if detail else f"exit code {proc.returncode}"
        raise PdfExportError(f"Conversion via {name} failed: {msg}")
    return str(pdf_path)
