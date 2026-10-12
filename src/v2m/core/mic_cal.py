"""Mic calibration: five held notes, one device offset.

A run records an offset only when the intonation report's gates hold it
— scattered runs record nothing. The offset is a DISPLAY fact: the
recognition path never reads it (ADR 0032).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .intonation import MIN_CENTS, MIN_NOTES

CAL_FILE = Path.home() / ".v2m_mic_cal"
# Five rounds is the smallest set the report's gates back (MIN_NOTES 4),
# with one wasted round to spare.
ROUNDS = 5
# A round counts only on its target (±0.5 st): an octave off measures the
# singer, not the mic. Tight enough that an alternating voice is refused
# at the MEAN (MIN_CENTS), not graded as five consistent misses.
TARGET_ST = 0.5


@dataclass(frozen=True)
class CalibrationVerdict:
    """One run's end state: a recordable offset, or the honest refusal."""

    offset_cents: float | None  # signed: heard minus target
    n_used: int
    n_skipped: int
    tendency: dict | None  # the intonation gate's own shape (or None)

    @property
    def recordable(self) -> bool:
        """Gates passed; the run may write the file."""
        return self.offset_cents is not None


def collect(
    heard: list[float],
    targets: list[float],
) -> CalibrationVerdict:
    """The device offset from the run's rounds, via the report's gates.

    A round counts only on its target; the lean is named only when the
    gates back it (enough rounds AND a big enough mean), else None.

    The jump is read in CENTS (1200·log2), not Hz: a fixed Hz window
    would calibrate an octave at the top and refuse it at the bottom.
    """
    import math

    used = [
        (1200.0 * math.log2(heard_i / target))
        for target, heard_i in zip(targets, heard, strict=True)
        if target > 0
        and heard_i > 0
        and abs(1200.0 * math.log2(heard_i / target)) <= TARGET_ST * 100.0
    ]
    skipped = len(heard) - len(used)
    if len(used) < MIN_NOTES or abs(sum(used) / len(used)) < MIN_CENTS:
        return CalibrationVerdict(
            offset_cents=None, n_used=len(used), n_skipped=skipped, tendency=None
        )
    mean = sum(used) / len(used)
    return CalibrationVerdict(
        offset_cents=round(mean, 1),
        n_used=len(used),
        n_skipped=skipped,
        tendency={"n": len(used), "mean_cents": round(mean, 1)},
    )


def read() -> dict | None:
    """The persisted calibration, or None."""
    try:
        import json

        data = json.loads(CAL_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("offset_cents"), (int, float)):
        return None
    return data


def write(verdict: CalibrationVerdict, targets: list[float]) -> bool:
    """Persist a recordable verdict (best-effort, never blocks)."""
    if not verdict.recordable:
        return False
    import json

    try:
        CAL_FILE.write_text(
            json.dumps(
                {
                    "offset_cents": verdict.offset_cents,
                    "n_used": verdict.n_used,
                    "targets_hz": [round(t, 1) for t in targets],
                    "date_s": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        return False
    return True


def clear() -> bool:
    """Drop the calibration (the opt-out); True when gone."""
    try:
        CAL_FILE.unlink()
    except OSError:
        return not CAL_FILE.exists()
    return True
