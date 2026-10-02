"""Replay the WAV takes of `prises_live/` through the GUI chain, and archive
the live/final gap + the contour rate per take.

The engine lives in the CORE (`core.replay.replay_take`: a real-time 1x
injected-InputStream replayer, `LiveTake.finalize`, the one comparison
policy `core.compare.breakdown`); this script is only the COMMAND-LINE
ADAPTER — the same engine as the GUI's ReplayWorker, which replays each
take when its recording stops and archives the gap in its folder. The
table is printed and archived as JSON (`--out`, default: write nothing).

Usage:
    python -X utf8 scripts/replay_takes.py
    python -X utf8 scripts/replay_takes.py prises_live --out replay_report.json
    python -X utf8 scripts/replay_takes.py --take live_20260926_140425

For each take owning a `<stem>.wav`: the WAV is read back in 1024-sample
blocks and fed to the REAL capture (`LiveCapture`) at the real clock —
delivering faster than real time is MEASURED to lie (7 notes instead of
24 on take 140425: the segmenter confirms on the blocks' time, not on
their content). No take artifact is modified: the script only LOOKS.
Takes without a WAV are listed as not replayable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from v2m.core.replay import REPLAY_SR, engine_fingerprint, replay_take
from v2m.core.takes import read_sidecar

DEFAULT_ROOT = "prises_live"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="replay_takes.py",
        description="Replays the WAV takes through the GUI chain (a fake real-time "
        "mic) and archives the live/final gap + the contour rate per take.",
    )
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT,
                        help="the takes' folder (default: prises_live)")
    parser.add_argument("--out", help="write the JSON table to this file")
    parser.add_argument("--take", help="replay only this take (stem name)")
    parser.add_argument(
        "--changed", action="store_true",
        help="replay only the takes whose audit was measured by an OLDER "
        "engine version (the code hash stored in the live_vs_final "
        "sidecar) — after an engine change",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"Folder not found: {root}")
        print("Name the folder that CONTAINS the takes (one sub-folder per take).")
        return 2

    wavs = sorted(root.glob("*/*.wav")) if not args.take else sorted(root.glob(f"{args.take}/*.wav"))
    if args.take:
        wavs = [w for w in wavs if w.parent.name == args.take]
    if args.changed:
        # keep only the WAVs whose archived audit was measured by an OLDER
        # engine (or has no audit yet): a changed engine invalidates the
        # VERDICT, not the take — and the fresh audits this very run writes
        # would otherwise re-qualify the take on every subsequent run.
        current = engine_fingerprint()
        kept = []
        for wav in wavs:
            archived = read_sidecar(wav.parent)  # the artifact contract's reader
            if archived is None:
                kept.append(wav)  # no audit yet: it is due
                continue
            if archived.get("engine") != current:
                kept.append(wav)
        skipped = len(wavs) - len(kept)
        wavs = kept
        if skipped:
            print(f"[--changed] {skipped} take(s) already audited by this engine — skipped.")
    if not wavs:
        print(f"No take WAV in \"{root}\" — tick \"Record the audio\" at the take.")
        return 0

    rows: list[dict] = []
    for wav in wavs:
        stem = wav.stem
        if stem != wav.parent.name:
            continue  # a stray artifact, not a take
        print(f"[replay] {stem} …", flush=True)
        try:
            row = replay_take(wav)
        except Exception as e:  # one failing take must not cost the others
            rows.append({"stem": stem, "error": str(e)})
            print(f"    failed: {e}")
            continue
        rows.append(row)
        print(
            f"    live {row['live_notes']:>3} | final {row['final_notes']:>3}"
            f" | contours {row['contours']:>3} (±1: {row['off1']},"
            f" moved: {row['moved']}) | rate {row['rate']*100:.0f}%"
            f" | {row['duration_s']} s @ {REPLAY_SR} Hz"
        )

    if rows:
        print("\nTake                               live final contours  ±1 moved rate")
        print("-" * 78)
        for r in rows:
            if "error" in r:
                print(f"{r['stem']:<34}  ERROR: {r['error'][:40]}")
                continue
            print(
                f"{r['stem']:<34}{r['live_notes']:>5}{r['final_notes']:>6}"
                f"{r['contours']:>8}{r['off1']:>5}{r['moved']:>5}{r['rate']*100:>5.0f}%"
            )
    if args.out:
        Path(args.out).write_text(
            json.dumps({"root": str(root), "takes": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nReport written: {args.out}")
    return 0


if __name__ == "__main__":
    reconfigure = getattr(sys.stdout, "reconfigure", None)  # a real TextIO may lack it
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    raise SystemExit(main())
