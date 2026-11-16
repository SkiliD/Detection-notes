"""Comparing two readings of the same take: ONE net, shared by every consumer.

A live take is read twice — once in real time by the streaming segmenter
(`LiveNote`), once offline by the interpretation (`Note`). The GUI diff,
`score_accuracy`, replay measurements and JSON breakdown all use this
shared matching policy, its counts and the same breakdown block:

- `match(live_notes, final_notes, grid)` pairs every live note with the
  NEAREST final note (by onset) and returns the moves: a `Move` carries
  the live note, its final counterpart, the onset shift and the REASON
  (`"off1"`: the ±1 semitone rounding family — the same plateau read by two
  roundings, resolved by the key snap offline; `"moved"`: a real pitch
  change or an onset beyond the tolerance — a merge, a split or an absorbed
  glide);
- `classify(moves, live_count, final_count)` counts them and publishes the
  health indicator: the CONTOUR RATE (moves / confirmed live notes) is the
  number the GUI outlines and the JSON breakdown reports. On measured takes
  it ranges from 47–57 %; read it alongside the final count because the ±1
  family dominates and is not a segmentation error.
- the BLOCK is owned end-to-end: `breakdown` builds the wire dict,
  `read_block` parses a persisted one into a `Breakdown` (the ONE tolerant
  parse), `invalidate` drops a stale block, `write_sidecar` archives it —
  construction, reading, invalidation and the file name have one place.
  `explain` stays the i18n phrasing on top of `read_block`, and the wire
  shape is FROZEN: sidecars and reports of past takes keep reading.

The match window is the take's grid sixteenth OR the live segmenter's
dating lead, whichever is larger: a live note is DATED at its first stable
frame, tens of ms before the audio attack the offline pass reads (the
measured 37 ms lead that set ATTACK_TOL_S) — a pure one-sixteenth window
alone outlined most notes of a take the engine read identically.

`chronological_pairs` is the pairing used when BOTH sides are readings of
one performance in order (the reading diff before/after, the phrase
take-to-take): a chronological dynamic program with explicit costs for an
association, a missing note and an added note, so a nearest-first choice
cannot fabricate an absence (see its docstring for the annotated case).
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
    ±1 semitone neighbour within the onset window and `"moved"` for anything
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


def chronological_pairs(
    a_starts: list[float],
    b_starts: list[float],
    bound: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Pair two chronological note lists one-to-one, globally optimal.

    Dynamic programming over the two sequences with EXPLICIT costs — the
    pairing policy of two readings of one performance (the reading diff)
    and of two takes of one phrase (the phrase comparison):

    - an ASSOCIATION costs its onset shift and only forms inside `bound`
      (one beat of the take: past it the two notes are not the same sung
      event);
    - an unpaired note of either list costs `bound` — so an association is
      always cheaper than a missing/added couple (2 × bound), and the
      program only decides WHICH associations to keep, never trades one
      away cheaply.

    A nearest-first greedy cannot do that: on the report's annotated case
    (before 0.4/0.7 s, after 0.0/0.5 s, bound 0.5 s) it takes 0.4 -> 0.5
    first and leaves 0.7 with no partner, fabricating a missing line and
    an added one although both notes pair (0.4 -> 0, 0.7 -> 0.5). Here the
    whole alignment is minimised instead: 0.4 -> 0 + 0.7 -> 0.5 (0.6) beats
    0.4 -> 0.5 + two unpaired (1.1).

    Returns `(matches, missing_a, added_b)`: matches as index pairs in
    chronological order, then the unpaired indexes of each list, each in
    its own list's order. Ties are DECIDED, not arbitrary: the candidate
    order is missing, added, association — an equal-cost alignment leaves
    the LATEST note unpaired (so the oldest note keeps the partner, the
    deterministic tie the reading diff pins) and an unpaired A note wins
    one over an unpaired B note. Same inputs always give the same output.
    """
    n, m = len(a_starts), len(b_starts)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    step = [[""] * (m + 1) for _ in range(n + 1)]  # "miss" / "add" / "pair"
    for i in range(1, n + 1):
        cost[i][0] = i * bound
        step[i][0] = "miss"
    for j in range(1, m + 1):
        cost[0][j] = j * bound
        step[0][j] = "add"
    for i in range(1, n + 1):
        a = float(a_starts[i - 1])
        for j in range(1, m + 1):
            best = cost[i - 1][j] + bound  # a_i unpaired (candidate order:
            choice = "miss"  # miss, then add, then associate)
            candidate = cost[i][j - 1] + bound
            if candidate < best:
                best, choice = candidate, "add"
            shift = abs(a - float(b_starts[j - 1]))
            if shift <= bound:
                candidate = cost[i - 1][j - 1] + shift
                if candidate < best:
                    best, choice = candidate, "pair"
            cost[i][j] = best
            step[i][j] = choice
    matches: list[tuple[int, int]] = []
    missing: list[int] = []
    added: list[int] = []
    i, j = n, m
    while i or j:
        choice = step[i][j]
        if choice == "pair":
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif choice == "miss":
            missing.append(i - 1)
            i -= 1
        else:
            added.append(j - 1)
            j -= 1
    matches.reverse()
    missing.reverse()
    added.reverse()
    return matches, missing, added


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
        nearby_rounding = abs(int(best.midi) - int(live.midi)) == 1 and abs(shift) <= tol
        reason = "off1" if nearby_rounding else "moved"
        moves.append(Move(live=live, final=best, onset_shift_s=round(shift, 4), reason=reason))
    return moves


def classify(
    moves,
    live_count: int,
    final_count: int,
    dropped: int | None = None,
    drain_min: float | None = None,
) -> dict:
    """Count the moves by reason; the JSON/GUI breakdown of the live gap.

    - `contours` / `rate`: the health indicator of real-time recognition;
      read the rate next to the final count, since the ±1 family dominates
      and is not a segmentation error (47–57 % on measured takes);
    - `off1`: the ±1 semitone rounding family (the offline pass absorbs the
      global tuning offset BEFORE rounding, the live rounds raw — resolved
      toward the key's degree in every measured case);
    - `moved`: pitch changes and onset shifts beyond the tolerance — the
      merges, splits and glides the offline look-ahead read differently;
    - `absorbed`: the final notes with no live counterpart (a glide the
      offline path can read as one note thanks to its look-ahead while the
      live one honestly confirms the plateau it watches);
    - `dropped`/`drain_min`: the capture's honesty counters, passed only by
      a live reading (a file analysis has none) — an ADDITIVE pair on the
      frozen wire: old readers ignore it, new readers see None on blocks
      that predate it. `dropped` counts the blocks a stop abandoned;
      `drain_min` is the worst mid-take progress the drain itself sampled
      (a take that starved and RECOVERED reads dropped=0 but drain_min<1).
    """
    moves = list(moves)
    off1 = sum(1 for m in moves if m.reason == "off1")
    block = {
        "live": int(live_count),
        "final": int(final_count),
        "contours": len(moves),
        "off1": off1,
        "moved": len(moves) - off1,
        "absorbed": max(0, int(final_count) - int(live_count)),
        "rate": round(len(moves) / live_count, 3) if live_count else 0.0,
    }
    if dropped is not None:
        block["dropped"] = int(dropped)
    if drain_min is not None:
        block["drain_min"] = round(float(drain_min), 3)
    return block


def breakdown(
    live_notes,
    final_notes,
    grid=None,
    window: float | None = None,
    dropped: int | None = None,
    drain_min: float | None = None,
) -> dict:
    """`match` + `classify` in one call — the JSON block's shape."""
    live_notes = list(live_notes or [])
    final_notes = list(final_notes or [])
    moves = match(live_notes, final_notes, grid, window)
    return classify(moves, len(live_notes), len(final_notes), dropped=dropped, drain_min=drain_min)


@dataclass(frozen=True)
class Breakdown:
    """One parsed `live_vs_final` block — fields, never dict keys.

    Absent or unreadable fields are None (the wire may predate a key or
    carry a hand edit); a consumer reads the fields it means and treats
    None as "this take says nothing", never as zero.
    """

    live: int | None
    final: int | None
    contours: int | None
    off1: int | None
    moved: int | None
    absorbed: int | None
    rate: float | None
    dropped: int | None
    drain_min: float | None

    @property
    def comparable_rate(self) -> float | None:
        """No contour rate can establish agreement when either reading is empty.

        Keep the persisted rate intact: old reports carry the same counts and
        can be interpreted without changing their frozen wire fields.
        """
        if self.live == 0 or self.final == 0:
            return None
        return self.rate


def read_block(block) -> Breakdown | None:
    """Parse a persisted block; absent, empty or non-dict reads None.

    The ONE tolerant parse: a count is accepted only as an int, the rate
    only as a number. The persisted wire shape remains stable; a corrupt
    field degrades alone instead of costing the whole block.
    """
    if not isinstance(block, dict) or not block:
        return None

    def count(key: str) -> int | None:
        value = block.get(key)
        return value if isinstance(value, int) else None

    rate = block.get("rate")
    return Breakdown(
        live=count("live"),
        final=count("final"),
        contours=count("contours"),
        off1=count("off1"),
        moved=count("moved"),
        absorbed=count("absorbed"),
        rate=float(rate) if isinstance(rate, (int, float)) else None,
        dropped=count("dropped"),
        drain_min=block.get("drain_min")
        if isinstance(block.get("drain_min"), (int, float))
        else None,
    )


def invalidate(result) -> None:
    """Drop a result's live/final decomposition: the audio it described moved.

    A manual pitch correction, a note deletion or a kept splice makes the
    captured reading stale — the block dies with it. The result's OTHER
    manual-edit facts stay with the edit path that owns them.
    """
    result.live_vs_final = None


def write_sidecar(stem, block: dict) -> str | None:
    """Archive the block as `<stem>.live_vs_final.json` (best-effort).

    The format's writer sits with the format: the export sequence and the
    replay worker both archive through here, and the PATH is the artifact
    contract's (`takes.stem_path` + `SIDECAR_SUFFIX`) — one spelling, the
    two writers never disagree on the name. An empty block writes nothing;
    an unwritable folder returns None (the artifacts are on disk either
    way).
    """
    import json

    from .takes import SIDECAR_SUFFIX, stem_path

    if not block:
        return None
    try:
        path = stem_path(stem, SIDECAR_SUFFIX)
        path.write_text(json.dumps(block, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)
    except OSError:
        return None
