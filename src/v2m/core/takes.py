"""Live take identity and ARTIFACTS: where a take lives, what it owns on disk.

A live take is one folder of artifacts. Its name IS its timestamp, minted by
`live_take_name` and read back by `parse_take_name` — ONE grammar, so anything
that groups takes by date (the coherence summary) derives the format instead
of re-encoding it. The directory layout lives here too: one folder for every
live export, one sub-folder per take.

This module also owns the artifact CONTRACT: which files a take owns
(`ARTIFACTS`), how each path is spelled (`stem_path`, `sidecar_path`,
`evidence_path`), and how the live/final audit sidecar is read
(`read_sidecar`). Every consumer that needs a take artifact path — the GUI announcer, replay
script, drift watch, coherence reader and reinterpret tool — asks here. A
single path contract keeps the sidecar and its take stem consistent.

This module is a LEAF on purpose (stdlib only). The readers of a take's
name — the GUI, the CLI live session, `report` and the offline tools —
share this vocabulary without sharing each other's weight: reading a name
must not import the engraving stack.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

# Where live takes land: one folder for every live export, one sub-folder per
# take (timestamped, so no two takes can collide). ONE owner: the export
# writers, the GUI, the CLI and the coherence tools all read this.
LIVE_TAKES_DIR = Path("prises_live")

# The live take stem format, minted and parsed from ONE literal: deriving the
# pattern from the format keeps both ends in sync if the format ever changes.
TAKE_NAME_PREFIX = "live_"
TAKE_NAME_STAMP = "%Y%m%d_%H%M%S"

# What each strftime directive reads back as when parsing a take name.
_STAMP_DIGITS = {
    "%Y": r"\d{4}",
    "%m": r"\d{2}",
    "%d": r"\d{2}",
    "%H": r"\d{2}",
    "%M": r"\d{2}",
    "%S": r"\d{2}",
}


def _stamp_pattern(stamp: str) -> str:
    """Regex source reading back a strftime `stamp`: every directive -> digits."""
    pattern = stamp
    for directive, digits in _STAMP_DIGITS.items():
        pattern = pattern.replace(directive, digits)
    return pattern


_TAKE_NAME_RE = re.compile(
    re.escape(TAKE_NAME_PREFIX)
    + f"(?P<date>{_stamp_pattern('%Y%m%d')})"
    + "_"
    + f"(?P<clock>{_stamp_pattern('%H%M%S')})$"
)

# --------------------------------------------------------------------------- #
# The artifact contract: what a take's folder OWNS, and how it is spelled.
# --------------------------------------------------------------------------- #

# The data artifacts `report.export_result` writes (the export announcer and
# the verify gate read this, never their own tuple).
ARTIFACTS = (".mid", ".musicxml", ".json", ".csv", ".coherence.csv")

# The take's LIVE/FINAL audit sidecar (the gap breakdown: `core.compare`).
SIDECAR_SUFFIX = ".live_vs_final.json"
# The evidence cache (the decoded-once pYIN+curve payload, `VoiceObservation`).
EVIDENCE_SUFFIX = ".observation.npz"

# A live take's preserved readings are kept outside the root artifact bundle:
# the root remains the current export consumed by existing audit tools.
READINGS_DIRNAME = "readings"
AUTOMATIC_READING_DIRNAME = "automatic"
CORRECTED_READINGS_DIRNAME = "corrected"


def live_take_name() -> str:
    """Unique stem for one live take: live_YYYYmmdd_HHMMSS."""
    return TAKE_NAME_PREFIX + time.strftime(TAKE_NAME_STAMP)


def parse_take_name(name: str | Path) -> tuple[str, str] | None:
    """(date, time) read from a live take stem — the twin of `live_take_name`.

    Returns ("YYYYmmdd", "HHMMSS") for a stem the minting could have written,
    else None. The SHAPE is the contract, not the calendar: a hand-made
    `live_20261340_000000` is a take name here, exactly as before. Callers
    that need the moment turn it into a datetime themselves.
    """
    match = _TAKE_NAME_RE.match(Path(name).name)
    return (match["date"], match["clock"]) if match else None


def live_export_dir(stem: str | Path) -> Path:
    """Folder holding one live take's artifacts: `prises_live/<stem>/`."""
    return LIVE_TAKES_DIR / Path(stem).name


def stem_path(stem: str | Path, suffix: str) -> Path:
    """A take's `<stem><suffix>` artifact path (one spelling for every writer)."""
    return Path(f"{stem}{suffix}")


def take_dir_stem(take_dir: Path) -> str | None:
    """The stem a take folder holds, or None when the folder is not a take.

    The folder name IS the stem (one sub-folder per take) — but only for a
    name the grammar mints: the coherence tools must not read a stray
    directory as a take.
    """
    name = Path(take_dir).name
    return name if parse_take_name(name) else None


def automatic_reading_dir(take_dir: str | Path) -> Path:
    """The immutable first interpretation's archive directory for one take."""
    return Path(take_dir) / READINGS_DIRNAME / AUTOMATIC_READING_DIRNAME


def corrected_readings_dir(take_dir: str | Path) -> Path:
    """The parent directory holding a take's numbered corrected readings."""
    return Path(take_dir) / READINGS_DIRNAME / CORRECTED_READINGS_DIRNAME


def corrected_reading_dir(take_dir: str | Path, version: int) -> Path:
    """One corrected-reading archive (`v001`, `v002`, …); versions start at 1."""
    if version < 1:
        raise ValueError("corrected reading versions start at 1")
    return corrected_readings_dir(take_dir) / f"v{version:03d}"


def sidecar_path(take_dir: str | Path, stem: str | None = None) -> Path:
    """The take's live/final audit sidecar, under its folder.

    `stem` defaults to the folder's own name (the layout's rule: one
    sub-folder per take, the folder IS the stem).
    """
    d = Path(take_dir)
    return d / f"{stem or d.name}{SIDECAR_SUFFIX}"


def read_sidecar(take_dir: str | Path, stem: str | None = None) -> dict | None:
    """The take's archived live/final audit, or None (never a guess).

    The ONE reader of the sidecar (the coherence aggregator, the drift
    watch, the weekly report): a missing, corrupt or non-dict file means
    "no audit", not an exception.
    """
    try:
        data = json.loads(sidecar_path(take_dir, stem).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def evidence_path(audio_path: str | Path) -> Path:
    """The evidence cache next to its audio: `<audio-stem>.observation.npz`."""
    return Path(audio_path).with_suffix(EVIDENCE_SUFFIX)


def has_artifacts(stem_path_prefix: str | Path, extras: tuple[str, ...] = ()) -> list[str]:
    """The suffixes of the take's artifacts that EXIST on disk.

    The export announcer's rule (a half-written export must never be
    announced complete), in one place: `extras` (the conditional ones,
    `.wav`/`.pdf`/the sidecar) are checked the same way.
    """
    base = str(stem_path_prefix)
    return [s for s in (*ARTIFACTS, *extras) if Path(base + s).exists()]
