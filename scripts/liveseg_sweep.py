"""Sweep of the live segmenter's hold thresholds: HOLD_ST x REARTIC_ST.

Question this script answers (2026-09-28): does a BEGINNER's vibrato
(+/-0.5 st, measured as the gap behind "il entend mal la voix") deserve a
wider hold than HOLD_ST 0.6 — and if yes, at what REARTIC_ST cost? The
hold test reads the RAW pitch spread over `_hold_window` (5 raw frames +
5 median frames); a +/-0.5 st sweep spans 1.0 st and fragments (measured:
one held note read 3-7 notes). Wider holds risk GLUING the same-note
repetition cases the 2026-09-25 repairs fought for.

Replays the 18 live truth cases (`test_voice_accuracy.LIVE_CASES`) PLUS
two dedicated "vibrato debutant" cases through the full real-time path
(LivePitchStream + flush), for each (HOLD_ST, REARTIC_ST) pair and several
synthesizer seeds — the harness's noise makes some cases SPORADIC, so one
seed proves nothing. `HOLD_ST`/`REARTIC_ST` are module globals read at
call time: the sweep re-binds them per cell.

Reading the table: `OK` = the case is exact at every seed; `xN` = N seeds
fail (graine list given when partial). A "vibrato debutant tenu" cell
reads OK only when the single sung note comes back as exactly ONE note.

Usage:
    python scripts/liveseg_sweep.py                                  # full grid
    ... --holds 0.6,0.8 --reartics 0.3 --seeds 1,2                     # targeted
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

import voicetruth  # noqa: E402
from voicetruth import SungNote, measure, synthesize  # noqa: E402
from v2m.core.liveseg import LivePitchStream  # noqa: E402

BLOCK = 1024
GATE = 0.008
HOLDS = (0.4, 0.6, 0.8, 1.0)
REARTICS = (0.2, 0.3, 0.4)
SEEDS = (1, 2, 3)

# The motivating case, built here (the truth corpus stays frozen): a
# beginner's vibrato of +/-0.5 st — one long held note, then a short
# phrase (a held-note verdict hides glue; a phrase verdict hides drift).
VIB_CASES: dict[str, tuple[list[SungNote], dict]] = {
    "vibrato debutant tenu": (
        [SungNote(64, 0.2, 1.6)],
        dict(vibrato_st=0.5, consonant_s=0.05),
    ),
    "vibrato debutant phrase": (
        [SungNote(60, 0.2, 0.7), SungNote(62, 0.9, 0.7), SungNote(64, 1.6, 0.7)],
        dict(glide_s=0.09, consonant_s=0.05, vibrato_st=0.5),
    ),
}


def run_case(
    name: str, truth: list[SungNote], kwargs: dict, seed: int, known: bool
) -> bool:
    """True when the case reads exact at this cell (or within its known limits)."""
    import v2m.core.liveseg as liveseg

    y = synthesize(truth, **{**kwargs, "seed": seed})
    stream = LivePitchStream(voicetruth.SR, gate_rms=GATE)
    for i in range(0, len(y) - BLOCK, BLOCK):
        stream.feed(y[i : i + BLOCK], (i + BLOCK) / voicetruth.SR)
    stream.seg.flush(len(y) / voicetruth.SR)
    if known:
        sung = {n.midi for n in truth}
        return len(stream.seg.notes) <= len(truth) and all(
            n.midi in sung for n in stream.seg.notes
        )
    return measure(stream.seg.notes, truth, tol_s=0.20).exact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holds", default=",".join(str(v) for v in HOLDS))
    parser.add_argument("--reartics", default=",".join(str(v) for v in REARTICS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = parser.parse_args()
    holds = [float(v) for v in args.holds.split(",")]
    reartics = [float(v) for v in args.reartics.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    import v2m.core.liveseg as liveseg
    from test_voice_accuracy import CASES, LIVE_CASES, LIVE_KNOWN_LIMITS

    cases: dict[str, tuple[list, dict, bool]] = {}
    for name in LIVE_CASES:
        truth, kwargs = CASES[name]
        cases[name] = (truth, kwargs, name in LIVE_KNOWN_LIMITS)
    for name, (truth, kwargs) in VIB_CASES.items():
        cases[name] = (truth, kwargs, False)

    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    out.write(
        f"liveseg sweep — {len(cases)} cas x HOLD_ST x REARTIC_ST"
        f" x {len(seeds)} graines {seeds}\n\n"
    )

    default_ok: dict[tuple[float, float], list[str]] = {}
    for h in holds:
        liveseg.HOLD_ST = h
        header = f"HOLD_ST={h:.1f}" + " " * 10 + "".join(
            f"{'R=' + format(r, '.1f'):>14}" for r in reartics
        )
        out.write(header + "\n")
        out.write("-" * len(header) + "\n")
        for name, (truth, kwargs, known) in cases.items():
            row = f"{name:34s}"
            for r in reartics:
                liveseg.REARTIC_ST = r
                fails = [s for s in seeds if not run_case(name, truth, kwargs, s, known)]
                tag = (
                    "OK"
                    if not fails
                    else f"x{len(fails)}{f' ({fails})' if len(fails) < len(seeds) else ''}"
                )
                row += f"{tag:>14}"
                if not fails:
                    default_ok.setdefault((h, r), []).append(name)
            out.write(row + "\n")
        out.write("\n")

    liveseg.HOLD_ST, liveseg.REARTIC_ST = 0.6, 0.3  # restore before reporting
    out.write("-" * 60 + "\n")
    for (h, r), ok in sorted(default_ok.items()):
        out.write(f"HOLD_ST={h:.1f} REARTIC_ST={r:.1f} : {len(ok)}/{len(cases)} cas OK\n")

    full = [k for k, v in default_ok.items() if len(v) == len(cases)]
    held_ok = [
        (h, r)
        for (h, r), ok in default_ok.items()
        if "vibrato debutant tenu" in ok and "vibrato debutant phrase" in ok
    ]
    out.write("\nverdict :\n")
    if full:
        out.write(
            "  cellules 100% OK (tous cas) : "
            + ", ".join(f"H={h} R={r}" for h, r in full)
            + "\n"
        )
    else:
        out.write("  AUCUNE cellule ne tient tous les cas\n")
    widest = max(held_ok) if held_ok else None
    if widest:
        out.write(
            f"  vibrato debutant +/-0.5 st lu en UNE note jusqu'a : "
            f"HOLD_ST={widest[0]} REARTIC_ST={widest[1]}\n"
        )
    else:
        out.write("  aucun (H, R) du balayage ne lit le vibrato debutant en une note\n")
    out.write(
        f"  defaut actuel H=0.6 R=0.3 : "
        f"{len(default_ok.get((0.6, 0.3), []))}/{len(cases)} cas OK\n"
    )
    out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
