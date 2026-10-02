"""Compare two takes of the same phrase, side by side (the repeat mode).

The singer's work loop is REPETITION: sing a phrase, listen, sing it
again. This script answers "did the second take change anything?" with
the ONE matching policy every live/final consumer already shares
(`core.compare`'s nearest-by-onset net), applied take-to-take through
`core.phrase`: each note of take A pairs with its nearest note of take
B, and every pairing is classified — same, ±1 neighbour (the rounding
family), time-shifted, moved — plus the notes without a counterpart
(missing from B, added in B). The timing shift is expressed in take B's
beats, so a slightly different tempo does not read as hesitation.

Inputs are the takes' exported reports (`<stem>.json`), passed as two
paths or two take stem names inside `prises_live/`. Nothing is written:
the table IS the product.

Usage:
    python -X utf8 scripts/compare_takes.py prises_live/live_20260929_161442 prises_live/live_20260929_161537
    python -X utf8 scripts/compare_takes.py live_20260929_161442 live_20260929_161537
    python -X utf8 scripts/compare_takes.py --json rapport.json live_A live_B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from v2m.core.i18n import note_display_pair, tr
from v2m.core.phrase import TEMPO_TOLERANCE, compare_take_files
from v2m.core import takes

REASONS = ("same", "off1", "shifted", "moved")


def _resolve(token: str) -> Path:
    """One argument -> the report path: a path, or a stem in prises_live."""
    p = Path(token)
    if p.suffix != ".json":
        # a bare stem: the take's folder holds its report (the artifact
        # contract's layout: one sub-folder per take). Read at CALL time
        # through the module: the owner stays the one `takes` owns.
        p = takes.LIVE_TAKES_DIR / token / f"{token}.json"
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_takes.py",
        description="Compares two takes of the same phrase side by side: "
        "note counts, pitches (the ±1 rounding family included) and the "
        "timing gaps, in beats of the second take.",
    )
    parser.add_argument("take_a", help="first take: a stem in prises_live or a report path")
    parser.add_argument("take_b", help="second take (the reference tempo: shifts read in ITS beats)")
    parser.add_argument("--json", help="write the comparison to this JSON file")
    args = parser.parse_args(argv)

    path_a, path_b = _resolve(args.take_a), _resolve(args.take_b)
    cmp = compare_take_files(path_a, path_b)
    if cmp is None:
        missing = path_a if not path_a.exists() else path_b
        print(tr("cmp unreadable").format(path=missing))
        return 2

    out = sys.stdout
    reconfigure = getattr(out, "reconfigure", None)  # a real TextIO may lack it
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

    print(tr("cmp head").format(
        a=cmp.stem_a, b=cmp.stem_b,
        n_a=len(cmp.paired) + len(cmp.missing), n_b=len(cmp.paired) + len(cmp.added),
    ))
    print(tr("cmp tempo").format(
        bpm_a=cmp.bpm_a, bpm_b=cmp.bpm_b,
        ratio=(cmp.tempo_ratio - 1.0) * 100,
    ))
    if abs(cmp.tempo_ratio - 1.0) > TEMPO_TOLERANCE:
        print(tr("cmp tempo drift"))
    counts = {
        "same": cmp.n_same, "off1": cmp.n_off1,
        "shifted": cmp.n_shifted, "moved": cmp.n_moved,
        "missing": len(cmp.missing), "added": len(cmp.added),
    }
    print(tr("cmp counts").format(
        same=counts["same"], off1=counts["off1"], shifted=counts["shifted"],
        moved=counts["moved"], missing=counts["missing"], added=counts["added"],
    ))
    # the detail FIRST (one line per non-same pairing, in singing order):
    # a ±1-only difference is a green verdict (the rounding family, not a
    # wrong note) but the singer still sees WHICH note it touched
    detail = [p for p in cmp.paired if p.reason != "same"]
    if detail:
        print(tr("cmp detail title"))
        for p in detail:
            print(tr(f"cmp {p.reason}").format(
                a=note_display_pair(p.midi_a), b=note_display_pair(p.midi_b),
                beats=p.shift_beats,
            ))
    if cmp.missing:
        print(tr("cmp missing title"))
        for i in cmp.missing:
            n = cmp.notes_a[i]
            print(tr("cmp missing row").format(
                i=i + 1, note=note_display_pair(int(n["midi"])),
                at=float(n["start_s"]),
            ))
    if cmp.added:
        print(tr("cmp added title"))
        for i in cmp.added:
            n = cmp.notes_b[i]
            print(tr("cmp added row").format(
                i=i + 1, note=note_display_pair(int(n["midi"])),
                at=float(n["start_s"]),
            ))
    if cmp.stable:
        print(tr("cmp stable"))
    return _maybe_write(args, cmp)


def _maybe_write(args, cmp) -> int:
    if not args.json:
        return 0
    Path(args.json).write_text(
        json.dumps(
            {
                "a": cmp.stem_a,
                "b": cmp.stem_b,
                "bpm_a": cmp.bpm_a,
                "bpm_b": cmp.bpm_b,
                "tempo_ratio": cmp.tempo_ratio,
                "counts": {
                    "same": cmp.n_same, "off1": cmp.n_off1,
                    "shifted": cmp.n_shifted, "moved": cmp.n_moved,
                    "missing": len(cmp.missing), "added": len(cmp.added),
                },
                "stable": cmp.stable,
                "pairs": [
                    {
                        "reason": p.reason, "midi_a": p.midi_a, "midi_b": p.midi_b,
                        "shift_s": p.shift_s, "shift_beats": p.shift_beats,
                    }
                    for p in cmp.paired if p.reason != "same"
                ],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(tr("cmp json written").format(path=args.json))
    return 0


if __name__ == "__main__":
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    raise SystemExit(main())
