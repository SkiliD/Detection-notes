"""Side-by-side comparison of two takes of the same phrase: ONE engine.

The user works by repetition: sing a phrase, listen, sing it again. The
question "did the second take change anything?" used to mean opening two
JSON reports and diffing them by eye. This module answers it with the
ONE matching policy already shared by every live/final consumer
(`core.compare.match`): each note of take A pairs with the NEAREST note
of take B, and every pairing is CLASSIFIED:

- `same`: identical pitch, onset within the window — the phrase is
  stable there;
- `off1`: a ±1-semitone neighbour, within the window — the same sung
  degree read by two roundings (the same family the live/final gap
  decomposes into; a take-to-tune-of-the-day difference, not a mistake);
- `shifted`: same pitch, onset beyond the window — the note moved in
  TIME (the measured case: hesitation, an extra breath);
- `moved`: a real pitch change — the singer sang something different;
- plus the counts the net cannot pair: notes take A sang that take B
  has no counterpart for (`missing`), and the reverse (`added`).

The window is the SAME rule as the live/final net: the largest of the
take's grid sixteenth and the segmenter's 0.10 s dating lead — a
time-shifted label under one sixteenth is the same reading, not a move.
A pair only forms WITHIN ONE BEAT of take B (`60 / bpm_b`, the same
bound CONTEXT.md publishes): the nearest-by-onset net alone would pair
every A note with SOMETHING and a count imbalance would read as bogus
`moved` pairs against the wrong notes; past a beat the two notes are
not the same sung event (the singer restructured the phrase) and the
note is missing/added instead.
The onsets are compared at a COMMON TEMPO: each take has its own BPM,
so the shift is expressed in the SECOND take's beats (`tempo-scaled`):
the same sung phrase at a slightly different tempo must not flood the
comparison with `shifted`. A ±1-label pairing never counts `shifted`
for its onset: the rounding family absorbs it.

The inputs are the takes' exported report dicts (`<stem>.json`, the
`export_result` schema: `notes` with `start_s`/`midi`, `tempo.bpm`),
read by `read_report` — missing or corrupt means None, never a guess
(the take is listed as unreadable, the comparison goes on without it).
Stdlib only: reading a take must not import the engraving stack (the
`core.takes` leaf rule).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .compare import LIVE_DATING_LEAD_S

# The two reports' tempo disagreement that still counts as "the same
# phrase": beyond this relative gap the timing shift is dominated by the
# tempo change itself and the report says so instead of listing every
# note as shifted (measured: takes of one phrase sit within 2% of each
# other; a rehearsal speeding up 15% is a different reading, and each
# note's shift would say nothing useful).
TEMPO_TOLERANCE = 0.05


@dataclass(frozen=True)
class Pair:
    """One note of take A and its nearest counterpart in take B.

    `reason` is the classification above; `shift_s` is the signed onset
    shift (B minus A, seconds); `shift_beats` is the same shift in
    take B's quarter beats (the tempo-scaled view the singer reads).
    """

    a_index: int
    b_index: int
    midi_a: int
    midi_b: int
    reason: str
    shift_s: float
    shift_beats: float


@dataclass(frozen=True)
class PhraseComparison:
    """The full side-by-side of two takes, ready to print or archive.

    `paired` is every pairing (same + off1 + shifted + moved); the
    counts are derived views. `notes_a`/`notes_b` are the plain dicts
    the caller passed (for the drill-down: names, figures).
    """

    stem_a: str
    stem_b: str
    bpm_a: float
    bpm_b: float
    paired: list[Pair]
    missing: list[int]      # indexes into A: no B note near enough to pair
    added: list[int]        # indexes into B: no A note claimed them
    tempo_ratio: float      # bpm_b / bpm_a
    notes_a: list[dict]     # the plain report notes (the drill-down reads them)
    notes_b: list[dict]

    @property
    def n_same(self) -> int:
        return sum(1 for p in self.paired if p.reason == "same")

    @property
    def n_off1(self) -> int:
        return sum(1 for p in self.paired if p.reason == "off1")

    @property
    def n_shifted(self) -> int:
        return sum(1 for p in self.paired if p.reason == "shifted")

    @property
    def n_moved(self) -> int:
        return sum(1 for p in self.paired if p.reason == "moved")

    @property
    def stable(self) -> bool:
        """True when the two takes read as the same performance.

        The ±1 family is tolerated (a rounding, not a musical fact); a
        real time shift or pitch change is not — the singer re-sang
        something, and the verdict must say so. The counts only have to
        agree: an absorbed glide on one side lands as missing/added and
        fails the verdict honestly.
        """
        return (
            not self.missing
            and not self.added
            and self.n_moved == 0
            and self.n_shifted == 0
        )


def read_report(path: str | Path) -> dict | None:
    """A take's exported report (`<stem>.json`), or None — never a guess.

    The ONE reader of a report for comparison (the twin of
    `takes.read_sidecar`): a missing, corrupt or non-dict file means
    "this take is unreadable", not an exception.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _bpm_of(report: dict) -> float:
    """The report's BPM, or the grid's plausible fallback (the same rule
    `BeatGrid.__post_init__` applies to a tempo-less result)."""
    tempo = report.get("tempo") or {}
    bpm = tempo.get("bpm") if isinstance(tempo, dict) else None
    if isinstance(bpm, (int, float)) and bpm > 0:
        return float(bpm)
    return 120.0


def _note_list(report: dict) -> list[dict]:
    """The report's note dicts with a start and a midi, in singing order."""
    notes = report.get("notes")
    if not isinstance(notes, list):
        return []
    return sorted(
        (n for n in notes
         if isinstance(n, dict) and isinstance(n.get("start_s"), (int, float))
         and isinstance(n.get("midi"), int)),
        key=lambda n: float(n["start_s"]),
    )


def _pairing_window(grid_bpm: float) -> float:
    """The pairing tolerance: the same max(sixteenth, dating lead) rule as
    the live/final net (`core.compare.match_window`) — a time-shifted
    label under one sixteenth of take B is the same reading, not a move;
    the lead is the segmenter's dating (the measured tens of ms)."""
    return max(60.0 / grid_bpm / 4.0, LIVE_DATING_LEAD_S)


def compare_reports(
    a: dict,
    b: dict,
    stem_a: str = "A",
    stem_b: str = "B",
) -> PhraseComparison:
    """Compare two exported reports of the same phrase, side by side.

    The matching is the nearest-by-onset net over both directions at
    once (greedy best-first, one-to-one): every candidate pair (an A
    note, a B note) is a candidate; the closest onsets pair first, each
    note used once. One-to-one is what the COUNTS mean: a missing note
    is one the singer did not re-sing, and a shared note would hide it
    (the first draft paired the nearest B note per A note without
    exclusivity — the extra A notes read as bogus `moved` pairs against
    the wrong B note instead of `missing`). Candidates beyond ONE BEAT
    of take B (`60 / bpm_b`, the bound CONTEXT.md publishes) are never
    formed: past it the two notes are not the same sung event. The
    pairs then classify per the policy above, onsets judged at take B's
    tempo (beats), so a slightly different tempo does not read as a
    time shift.
    """
    notes_a = _note_list(a)
    notes_b = _note_list(b)
    bpm_a, bpm_b = _bpm_of(a), _bpm_of(b)
    tol = _pairing_window(bpm_b)
    beats_s = 60.0 / bpm_b  # one quarter beat of take B, in seconds

    # every candidate pair inside the bound, closest onsets first: the
    # greedy one-to-one matching (a phrase has tens of notes, not
    # thousands — the sort is free and the rule reads itself)
    candidates: list[tuple[float, int, int]] = []
    for ia, na in enumerate(notes_a):
        start_a = float(na["start_s"])
        for ib, nb in enumerate(notes_b):
            d = abs(float(nb["start_s"]) - start_a)
            if d <= beats_s:  # one beat of B: past it, not the same event
                candidates.append((d, ia, ib))
    candidates.sort()
    used_a: set[int] = set()
    chosen_b: set[int] = set()
    pairs: dict[int, tuple[int, float]] = {}  # ia -> (ib, shift)
    for _d, ia, ib in candidates:
        if ia in used_a or ib in chosen_b:
            continue
        used_a.add(ia)
        chosen_b.add(ib)
        shift = float(notes_b[ib]["start_s"]) - float(notes_a[ia]["start_s"])
        pairs[ia] = (ib, shift)

    paired: list[Pair] = []
    missing: list[int] = []
    for ia, na in enumerate(notes_a):
        if ia not in pairs:
            missing.append(ia)  # nothing of B inside one beat: not re-sung
            continue
        ib, shift = pairs[ia]
        nb = notes_b[ib]
        same_pitch = int(nb["midi"]) == int(na["midi"])
        if same_pitch and abs(shift) <= tol:
            reason = "same"
        elif abs(int(nb["midi"]) - int(na["midi"])) == 1:
            reason = "off1"  # the rounding family: the onset is absorbed
        elif same_pitch:
            reason = "shifted"
        else:
            reason = "moved"
        paired.append(
            Pair(
                a_index=ia,
                b_index=ib,
                midi_a=int(na["midi"]),
                midi_b=int(nb["midi"]),
                reason=reason,
                shift_s=round(shift, 4),
                shift_beats=round(shift / beats_s, 3) if beats_s else 0.0,
            )
        )
    # the reverse coverage: a B note the matching did not claim is one
    # take B sang that take A has nothing at that moment for
    added = [ib for ib in range(len(notes_b)) if ib not in chosen_b]
    return PhraseComparison(
        stem_a=stem_a,
        stem_b=stem_b,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        paired=paired,
        missing=missing,
        added=added,
        tempo_ratio=round(bpm_b / bpm_a, 4) if bpm_a else 1.0,
        notes_a=notes_a,
        notes_b=notes_b,
    )


def compare_take_files(
    path_a: str | Path,
    path_b: str | Path,
) -> PhraseComparison | None:
    """Read both reports, then compare; None when either is unreadable.

    The adapter's one call: it names the tempo drift in the result
    (`tempo_ratio` against `TEMPO_TOLERANCE`) but never decides for the
    caller — the printed table is the product.
    """
    ra, rb = read_report(path_a), read_report(path_b)
    if ra is None or rb is None:
        return None
    return compare_reports(
        ra, rb, stem_a=Path(path_a).stem, stem_b=Path(path_b).stem
    )
