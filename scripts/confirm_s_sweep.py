"""Sweep of the live confirmation threshold `confirm_s` (0.05 -> 0.10).

Replays the 18 live truth cases (`test_voice_accuracy.LIVE_CASES`) through
the full real-time path (LivePitchStream + flush), for each `confirm_s`
value and several synthesizer seeds — the harness's noise makes some cases
SPORADIC (the same case passes at seed 1 and fails at seed 2: measured on
"noise and vibrato" at confirm_s=0.08), so one seed proves nothing.

Reading the table: `exact` = every truth note found, none extra; `lat` =
the measured confirmation latency (ms, first ACTIVE note minus its
attack). A case may be "known" (LIVE_KNOWN_LIMITS): it is then judged on
its documented limits (no invented note), not on exactness.

Usage:
    ./.venv/Scripts/python.exe scripts/confirm_s_sweep.py            # table
    ... --values 0.08,0.09 --seeds 1,2,3                              # targeted
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
from voicetruth import measure, synthesize  # noqa: E402
from v2m.core.liveseg import LivePitchStream  # noqa: E402

BLOCK = 1024
GATE = 0.008
VALUES = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
SEEDS = (1, 2, 3, 4, 5)


def run_case(name: str, seed: int, confirm_s: float) -> tuple[bool, float]:
    """(exact, confirmation latency in ms) for one case at one seed."""
    from test_voice_accuracy import CASES, LIVE_KNOWN_LIMITS, LIVE_TOL_S

    truth, kwargs = CASES[name]
    kwargs = {**kwargs, "seed": seed}
    y = synthesize(truth, **kwargs)
    stream = LivePitchStream(voicetruth.SR, gate_rms=GATE)
    stream.seg.confirm_s = confirm_s
    first_active_at: float | None = None
    first_attack_at: float | None = None
    was_active = False
    for i in range(0, len(y) - BLOCK, BLOCK):
        _, frames = stream.feed(y[i : i + BLOCK], (i + BLOCK) / voicetruth.SR)
        if first_attack_at is None and len(stream.attacks.recent):
            first_attack_at = min(stream.attacks.recent)
        # the CONFIRMATION instant is the frame time when the state machine
        # first goes ACTIVE (note_start is the ONSET date, often the attack
        # itself — subtracting those would measure nothing)
        if first_active_at is None and not was_active and stream.seg.state == "ACTIVE" and frames:
            first_active_at = frames[-1][0]
        was_active = stream.seg.state == "ACTIVE"
    stream.seg.flush(len(y) / voicetruth.SR)
    if name in LIVE_KNOWN_LIMITS:
        sung = {n.midi for n in truth}
        ok = (
            len(stream.seg.notes) <= len(truth)
            and all(n.midi in sung for n in stream.seg.notes)
        )
    else:
        ok = measure(stream.seg.notes, truth, tol_s=LIVE_TOL_S).exact
    latency = (
        max(0.0, (first_active_at - first_attack_at) * 1000.0)
        if first_active_at is not None and first_attack_at is not None
        else float("nan")
    )
    return ok, latency


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", default=",".join(str(v) for v in VALUES))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = parser.parse_args()
    values = [float(v) for v in args.values.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    from test_voice_accuracy import LIVE_CASES, LIVE_KNOWN_LIMITS

    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    out.write(f"confirm_s sweep — {len(LIVE_CASES)} cas x {len(seeds)} graines\n")
    out.write(f"graines : {seeds}\n\n")

    header = f"{'cas':34s}" + "".join(f"{v:>12.2f}" for v in values)
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    all_ok: dict[float, list[str]] = {v: [] for v in values}
    for name in LIVE_CASES:
        row = f"{name:34s}"
        for v in values:
            fails = []
            lats = []
            for s in seeds:
                ok, lat = run_case(name, s, v)
                if not ok:
                    fails.append(s)
                lats.append(lat)
            tag = "OK" if not fails else f"x{len(fails)}{f' (graines {fails})' if len(fails) < len(seeds) else ''}"
            lat_med = float(np.nanmedian(lats))
            row += f"{tag:>6}/{lat_med:4.0f}ms"
            if not fails:
                all_ok[v].append(name)
        out.write(row + "\n")
    out.write("-" * len(header) + "\n")
    for v in values:
        out.write(f"confirm_s={v:.2f} : {len(all_ok[v])}/{len(LIVE_CASES)} cas OK\n")
    out.write("\nplancher observe : la plus basse valeur 100% OK sur toutes les graines\n")
    out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
