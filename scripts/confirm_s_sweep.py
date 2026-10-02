"""Sweep of the live confirmation threshold `confirm_s` (0.05 -> 0.10).

Replays the live truth cases (`test_voice_accuracy.LIVE_CASES`) through
the full real-time path (LivePitchStream + flush), for each `confirm_s`
value and several synthesizer seeds — the harness's noise makes some cases
SPORADIC (the same case passes at seed 1 and fails at seed 2: measured on
"noise and vibrato" at confirm_s=0.08), so one seed proves nothing.

An `--amps` axis re-synthesizes every case QUIETER (the `synthesize` amp,
0.35 by default): the 2026-09-29 corpus question — the real takes sat at
RMS 0.019..0.062, 3-7x under the reference takes, and the weakest degraded
the PITCH (12/22 labels off), not the note count. The sweep verifies the
segmentation bounds HOLD on quiet signals: at amp 0.10 the cell 0.09 must
stay exact on every seed that is exact at 0.35 (a quiet voice must not
cost a note or invent one); below ~0.05 the harness approaches the
silence gate itself (the DEFAULT amp already saturates near 0.35), so the
floor is where the sweep stops being about `confirm_s`.

Reading the table: `exact` = every truth note found, none extra; `lat` =
the measured confirmation latency (ms, first ACTIVE note minus its
attack). A case may be "known" (LIVE_KNOWN_LIMITS): it is then judged on
its documented limits (no invented note), not on exactness.

Usage:
    python scripts/confirm_s_sweep.py                                # table
    ... --values 0.08,0.09 --seeds 1,2,3                              # targeted
    ... --amps 0.35,0.10,0.05                                         # quiet axis
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
# the synthesis amplitude axis: the corpus default (0.35) and the two
# quiet levels of the 2026-09-29 diagnostic (the real takes sat 3-7x
# under the reference takes; 0.05 grazes the silence gate itself)
AMPS = (0.35, 0.10, 0.05)


def run_case(name: str, seed: int, confirm_s: float, amp: float = 0.35) -> tuple[bool, float]:
    """(exact, confirmation latency in ms) for one case at one seed."""
    from test_voice_accuracy import CASES, LIVE_KNOWN_LIMITS, LIVE_TOL_S

    truth, kwargs = CASES[name]
    kwargs = {**kwargs, "seed": seed, "amp": amp}
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", default=",".join(str(v) for v in VALUES))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--amps", default=",".join(str(a) for a in AMPS),
                        help="synthesis amplitude axis (0.35 = the corpus default)")
    args = parser.parse_args(argv)
    values = [float(v) for v in args.values.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    amps = [float(a) for a in args.amps.split(",")]

    from test_voice_accuracy import LIVE_CASES, LIVE_KNOWN_LIMITS

    # wrapping (not replacing) stdout: the utf-8 encoder over the real
    # buffer, DETACHED at the end so the underlying stream (pytest's
    # capture) stays open — a plain TextIOWrapper close() once closed
    # pytest's own buffer under it. In-process `main()` is the contract.
    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    try:
        return _run(out, values, seeds, amps, LIVE_CASES)
    finally:
        out.detach()


def _run(out, values, seeds, amps, LIVE_CASES) -> int:
    out.write(f"confirm_s sweep — {len(LIVE_CASES)} cas x {len(seeds)} graines\n")
    out.write(f"graines : {seeds}\n")

    all_ok_by_amp: dict[float, dict[float, list[str]]] = {}
    for amp in amps:
        out.write(
            f"\n=== amplitude {amp:.2f} "
            + ("(corpus default)" if amp == AMPS[0] else "(signal faible)")
            + " ===\n"
        )
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
                    ok, lat = run_case(name, s, v, amp=amp)
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
        all_ok_by_amp[amp] = all_ok
    floor = min(
        (
            v
            for ok in all_ok_by_amp.values()
            for v, names in ok.items()
            if len(names) == len(LIVE_CASES)
        ),
        default=None,
    )
    if floor is not None:
        out.write(
            f"\nplancher observe : {floor:.2f} — la plus basse valeur 100% OK "
            "sur toutes les graines, toutes amplitudes\n"
        )
    out.write(
        "\nbornes de segmentation : une amplitude faible ne doit ni couter une "
        "note ni en inventer une — lire les colonnes 0.08..0.10 de chaque "
        "bloc d'amplitude.\n"
    )
    out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
