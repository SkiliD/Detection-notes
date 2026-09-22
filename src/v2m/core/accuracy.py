"""Persistent per-take tuning history (~/.v2m_accuracy.json).

A practice aid across sessions: every finished live take appends one entry
(one mean |cents| per note), so a singer can watch the drift shrink from
week to week. Plain data in, plain data out — no Qt, no analysis; the GUI
owns when to call. A logging failure must never cost the take that was
already exported, so every failure path returns quietly.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .music import midi_to_name, midi_to_name_fr


class AccuracyHistory:
    """Running per-note tuning history for one take.

    This is domain state, not a widget: the GUI plot and persistence both
    consume the same model, so they cannot silently disagree about means or
    note order.
    """

    def __init__(self) -> None:
        self.means: dict[int, float] = {}
        self._count: dict[int, int] = {}
        self.points: list[tuple[int, float]] = []
        self._point_count: list[int] = []
        self.onsets: list[float] = []

    def add(self, midi: int | None, cents: float | None, onset: float | None = None) -> None:
        if midi is None or cents is None or cents != cents:
            return
        absolute = abs(cents)
        if onset is not None and (
            not self.onsets
            or onset != self.onsets[-1]
            or not self.points
            or self.points[-1][0] != midi
        ):
            self.points.append((midi, absolute))
            self._point_count.append(1)
            self.onsets.append(onset)
        elif self.points and self.points[-1][0] == midi:
            count = self._point_count[-1]
            self.points[-1] = (
                midi,
                (self.points[-1][1] * count + absolute) / (count + 1),
            )
            self._point_count[-1] = count + 1
        count = self._count.get(midi, 0)
        self.means[midi] = (self.means.get(midi, 0.0) * count + absolute) / (count + 1)
        self._count[midi] = count + 1

    def reset(self) -> None:
        self.means.clear()
        self._count.clear()
        self.points.clear()
        self._point_count.clear()
        self.onsets.clear()


ACCURACY_FILE = Path.home() / ".v2m_accuracy.json"


def load_log(path: Path = ACCURACY_FILE) -> list[dict]:
    """The recorded takes, oldest first; [] when absent or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def record_take(
    history,
    duration_s: float = 0.0,
    bpm: float | None = None,
    path: Path = ACCURACY_FILE,
) -> dict | None:
    """Append one take's per-note mean |cents| to the log; return the entry.

    `history` is an AccuracyHistory (means: {midi: mean_cents}). An empty history
    (nothing sung) records nothing.
    """
    if not history.means:
        return None
    notes = [
        {
            "midi": midi,
            "name": midi_to_name(midi),
            "name_fr": midi_to_name_fr(midi),
            "cents": round(cents, 1),
        }
        for midi, cents in history.means.items()
    ]
    entry = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(duration_s, 2),
        "bpm": bpm,
        "n_notes": len(notes),
        "mean_cents": round(sum(history.means.values()) / len(notes), 1),
        "notes": notes,
    }
    log = load_log(path)
    log.append(entry)
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash never truncates the log
    except OSError:
        return None  # unwritable home, full disk: the take stays exported
    return entry
