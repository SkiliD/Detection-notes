"""Comparing two readings of the same take: ONE net, shared by every consumer.

A live take is read twice — once in real time by the streaming segmenter
(`LiveNote`), once offline by the interpretation (`Note`). Four consumers
compare the two readings and each used to own its own matching rule: the
GUI's diff mark, `score_accuracy`'s live diagnosis, the AGENTS.md replay
measurements and (now) the JSON breakdown. Same take, four nets, four ways
to drift.

The policy lives here, in two halves:

- `match(live_notes, final_notes, grid)` pairs every live note with the
  NEAREST final note (by onset) and returns the moves: a `Move` carries the
  live note, its final counterpart, the onset shift and the REASON
  (`"off1"`: the ±1 semitone rounding family — the same plateau read by two
  roundings, resolved by the key snap offline; `"moved"`: a real pitch
  change or an onset beyond the tolerance — a merge, a split or an absorbed
  glide);
- `classify(moves, live_count, final_count)` counts them and publishes the
  health indicator: the CONTOUR RATE (moves / confirmed live notes) is the
  number the GUI outlines and the JSON breakdown reports. Its measured
  values (47–57 % on the real takes) live in AGENTS.md; read it NEXT to the
  final count — the ±1 family dominates and is not a segmentation error.

The match window is the take's grid sixteenth OR the live segmenter's
dating lead, whichever is larger: a live note is DATED at its first stable
frame, tens of ms before the audio attack the offline pass reads (the
measured 37 ms lead that set ATTACK_TOL_S) — a pure one-sixteenth window
alone outlined most notes of a take the engine read identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2m.core.liveseg import LiveNote
    from v2m.core.notes import Note

# The live segmenter's dating lead, in seconds (see module docstring).
LIVE_DATING_LEAD_S = 0.10

# Fallback tolerance when no grid exists yet (a hand-built result in a
# test): one sixteenth at ~120 BPM.
DEFAULT_WINDOW_S = 0.12


@dataclass(frozen=True)
class Move:
    """One live note the final analysis wrote differently, and why.

    `final` is the nearest final note by onset; `reason` is `"off1"` for a
    ±1 semitone neighbour (the rounding family) and `"moved"` for anything
    else — a real pitch change or an onset beyond the match window.
    """

    live: "LiveNote"
    final: "Note"
    onset_shift_s: float
    reason: str


def sixteenth_seconds(grid) -> float:
    """One sixteenth at the grid's tempo, or the fallback without a grid.

    The grid is the take's own `BeatGrid`: the tolerance is metric, decided
    by the same arithmetic every other consumer reads — never a literal.
    """
    if grid is not None and getattr(grid, "bpm", None):
        return grid.step_seconds(4.0)
    return DEFAULT_WINDOW_S


def match_window(grid=None, window: float | None = None) -> float:
    """The matching tolerance: the grid sixteenth OR the dating lead."""
    if window is not None:
        return float(window)
    return max(sixteenth_seconds(grid), LIVE_DATING_LEAD_S)


def match(live_notes, final_notes, grid=None, window: float | None = None) -> list[Move]:
    """Pair every live note with its nearest final note; return the moves.

    A merge or a split shows as its moved parts — judged against the
    NEAREST final note, which is the honest picture of what the offline
    pass corrected. Notes are read as-is (duck-typed: `start`/`end`/`midi`),
    so the same function serves `LiveNote` vs `Note` and any audit replay.
    """
    live_notes = list(live_notes or [])
    final_notes = list(final_notes or [])
    if not live_notes or not final_notes:
        return []
    tol = match_window(grid, window)
    moves: list[Move] = []
    for live in live_notes:
        best = min(final_notes, key=lambda fn: abs(fn.start - live.start))
        shift = float(best.start) - float(live.start)
        if best.midi == live.midi and abs(shift) <= tol:
            continue  # the same reading: not a move
        reason = "off1" if abs(int(best.midi) - int(live.midi)) == 1 else "moved"
        moves.append(Move(live=live, final=best, onset_shift_s=round(shift, 4), reason=reason))
    return moves


def classify(moves, live_count: int, final_count: int) -> dict:
    """Count the moves by reason; the JSON/GUI breakdown of the live gap.

    - `contours` / `rate`: the health indicator of real-time recognition
      (AGENTS.md: 47–57 % on the measured takes, read next to the final
      count — the ±1 family dominates and is NOT a segmentation error);
    - `off1`: the ±1 semitone rounding family (the offline pass absorbs the
      global tuning offset BEFORE rounding, the live rounds raw — resolved
      toward the key's degree in every measured case);
    - `moved`: pitch changes and onset shifts beyond the tolerance — the
      merges, splits and glides the offline look-ahead read differently;
    - `absorbed`: the final notes with no live counterpart (a glide the
      offline path can read as one note thanks to its look-ahead while the
      live one honestly confirms the plateau it watches).
    """
    moves = list(moves)
    off1 = sum(1 for m in moves if m.reason == "off1")
    return {
        "live": int(live_count),
        "final": int(final_count),
        "contours": len(moves),
        "off1": off1,
        "moved": len(moves) - off1,
        "absorbed": max(0, int(final_count) - int(live_count)),
        "rate": round(len(moves) / live_count, 3) if live_count else 0.0,
    }


def breakdown(live_notes, final_notes, grid=None, window: float | None = None) -> dict:
    """`match` + `classify` in one call — the JSON block's shape."""
    live_notes = list(live_notes or [])
    final_notes = list(final_notes or [])
    moves = match(live_notes, final_notes, grid, window)
    return classify(moves, len(live_notes), len(final_notes))
