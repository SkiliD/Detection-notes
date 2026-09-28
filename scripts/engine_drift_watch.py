"""Non-drift watch: re-derive every stale engine audit, report the deltas.

The `--changed` flag of `replay_takes.py` answers "which audits were
measured by an OLDER engine" (the sidecar's `engine` hash vs
`engine_fingerprint()`); this script AUTOMATES the follow-up: replay
everything stale, compare each fresh row against the sidecar it replaces,
and write a Markdown drift report — the answer to "did my last engine
change move any measured verdict?" without hand-running the script and
diffing by eye.

`core.replay` owns the whole policy (real-time delivery, the chain, the
one matching policy); this is a thin CLI adapter over it, like
`replay_takes.py` — a third consumer must not re-implement a replay.

Usage: `python scripts/engine_drift_watch.py [--root prises_live]
[--report docs/engine-drift.md]`. Exit 0 when nothing is stale or every
replay matched its previous audit; exit 1 when at least one metric moved
beyond tolerance (the CI/follow-up hook fails loudly); the report is
written either way (best-effort: a read-only repo keeps the console
output).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v2m.core.replay import engine_fingerprint, replay_take  # noqa: E402
from v2m.core.takes import read_sidecar as read_sidecar  # noqa: E402
from v2m.core.takes import LIVE_TAKES_DIR  # noqa: E402

# A metric "moved" past this delta — the report's DRIFT verdict per take.
# The ±1-semitone rounding family dominates the contour count; a one-note
# wobble (1/20 ≈ 0.05) is the engine's honest session noise, three notes
# is a reading that changed.
DRIFT_TOLERANCE = 0.15


def _stale_takes(root: Path, current: str) -> list[tuple[str, Path, dict]]:
    """Takes whose sidecar was measured by another engine (or lacks one).

    Only audits that CARRY an engine hash participate in the drift verdict:
    the export-time sidecar (a plain live_vs_final block, no replay yet)
    would compare a LIVE reading against a REPLAY — measured families that
    always differ (the ±1 rounding). A hash-less sidecar is not "stale":
    it has never been replayed, there is no previous engine verdict to
    defend.
    """
    out: list[tuple[str, Path, dict]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        row = read_sidecar(d)  # the contract's reader, never a private spelling
        if row is not None and row.get("engine") not in (None, current):
            out.append((d.name, d, row))
    return out


def _deltas(old: dict, row: dict) -> dict:
    """Per-metric movement between the archived audit and the fresh replay."""
    return {
        "live_notes": row["live_notes"] - int(old.get("live", 0)),
        "final_notes": row["final_notes"] - int(old.get("final", 0)),
        "contours": row["contours"] - int(old.get("contours", 0)),
        "rate": row["rate"] - float(old.get("rate", 0.0)),
    }


def _drifted(d: dict) -> bool:
    return (
        abs(d["live_notes"]) > 0
        or abs(d["final_notes"]) > 0
        or abs(d["contours"]) > 0
        or abs(d["rate"]) > DRIFT_TOLERANCE
    )


def _report(
    path: Path, current: str, rows: list[dict], drifted: list[dict]
) -> None:
    lines = [
        "# Non-dérive du moteur de rejeu",
        "",
        f"_Généré le {datetime.now():%Y-%m-%d %H:%M} — moteur `{current}`._",
        "",
        f"- prises re-dérivées : **{len(rows)}**",
        f"- lectures déplacées : **{len(drifted)}**",
        "",
    ]
    if not rows:
        lines += ["Aucun audit périmé : le moteur n'a pas changé depuis les",
                  "derniers rejeux (ou aucune prise n'a encore de sidecar).", ""]
    else:
        lines += [
            "| prise | live | final | contours | taux | verdict |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['stem']} | {r['row']['live_notes']} vs {r['old_live']} "
                f"| {r['row']['final_notes']} vs {r['old_final']} "
                f"| {r['row']['contours']} vs {r['old_contours']} "
                f"| {r['row']['rate']:.0%} vs {r['old_rate']:.0%} "
                f"| {'DÉRIVE' if r['drift'] else 'stable'} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rejoue les audits périmés (--changed) et publie les deltas."
    )
    parser.add_argument("--root", default=str(LIVE_TAKES_DIR))
    parser.add_argument("--report", default="docs/engine-drift.md")
    args = parser.parse_args(argv)

    root = Path(args.root)
    current = engine_fingerprint()
    stale = _stale_takes(root, current)
    print(f"moteur {current} — {len(stale)} audit(s) périmé(s)")
    rows: list[dict] = []
    for stem, take_dir, old in stale:
        wav = take_dir / f"{stem}.wav"
        if not wav.exists():
            print(f"  {stem}: pas de WAV, audit conservé tel quel")
            continue
        try:
            row = replay_take(wav)
        except Exception as e:
            print(f"  {stem}: ERREUR {e}")
            continue
        d = _deltas(old, row)
        drift = _drifted(d)
        rows.append({
            "stem": stem,
            "row": row,
            "old_live": int(old.get("live", 0)),
            "old_final": int(old.get("final", 0)),
            "old_contours": int(old.get("contours", 0)),
            "old_rate": float(old.get("rate", 0.0)),
            "deltas": d,
            "drift": drift,
        })
        moved = ", ".join(
            f"{k} {v:+d}" if k != "rate" else f"{k} {v:+.0%}"
            for k, v in d.items()
        )
        print(f"  {stem}: {'DÉRIVE' if drift else 'stable'} ({moved})")
    drifted = [r for r in rows if r["drift"]]
    try:
        _report(Path(args.report), current, rows, drifted)
        print(f"rapport : {args.report}")
    except OSError as e:
        print(f"rapport non écrit : {e}")
    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
