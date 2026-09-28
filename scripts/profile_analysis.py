"""Diagnose where an analysis's time goes.

Usage:
    python scripts/profile_analysis.py [audio.wav] [--repeat N] [--profile]
                                       [--top N] [--json measures.json] [--output DIR]

The stage table is built from `analyze_signal`'s `progress` callback: the
pipeline already announces every stage boundary, so the script times the
REAL pipeline instead of re-implementing it (a new stage shows up here on
its own). The "real time" column divides the stage duration by the audio's:
0.07x means 14x faster than listening — the metric that matters for a long
take.

Each stage also carries its MEMORY PEAK (tracemalloc, Python allocations):
the profile covers CPU and memory — a fast stage that allocates big (pYIN
on a long take) shows up. The measured reference: 66 MB total peak on a
68 s take; a peak that doubles without the take's duration doubling is a
leak to hunt. tracemalloc traces PYTHON allocations only (numpy buffers
count, librosa's C labels partially): it is a comparative order of
magnitude, not an RSS.

`--profile` adds a cProfile table (own time): it measures a PROFILED run,
to compare between functions only, not with the table above.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import v2m
from v2m.core import audio_io
from v2m.core.i18n import tr
from v2m.core.pipeline import AnalyzeOptions, analyze_signal
from v2m.core.report import export_result

LABEL_WIDTH = 38
TABLE_WIDTH = LABEL_WIDTH + 35  # label + the four numeric columns
DEFAULT_AUDIO = "demo.wav"
PACKAGE_DIR = Path(v2m.__file__).resolve().parent  # our own code, not the libraries


@dataclass
class Stage:
    """One measured span: a pipeline stage, or a step around it."""

    label: str
    fraction: float | None  # progress fraction, None for non-pipeline steps
    seconds: float
    peak_mib: float = 0.0   # tracemalloc peak of Python allocations, in MiB

    @property
    def start_text(self) -> str:
        return "—" if self.fraction is None else f"{self.fraction * 100:.0f}%"


class StageTimer:
    """Turn `progress(fraction, message)` into per-stage timings + memory.

    The pipeline reports a stage BEFORE running it, so the time between two
    callbacks is the duration of the stage announced by the first one. The
    closing callback (1.0, "Done") ends the last real stage and opens one
    that nothing follows: `close()` drops it.

    Each stage also measures its tracemalloc PEAK (Python allocations only,
    see the module docstring): the trace runs for the WHOLE analysis and
    each stage reports the peak reached during its own span. The counter is
    RESET at each stage boundary — a stage's peak is its own allocations,
    not a running total, so the table answers "which stage allocates the
    most" and the TOTAL row answers "how much at once".
    """

    def __init__(self, with_memory: bool = True) -> None:
        self.stages: list[Stage] = []
        self._pending: tuple[str, float, float] | None = None
        self.with_memory = with_memory

    def __call__(self, fraction: float, message: str) -> None:
        now = time.perf_counter()
        if self._pending is not None:
            label, frac, started = self._pending
            peak = (
                tracemalloc.get_traced_memory()[1] / (1024 * 1024)
                if self.with_memory
                else 0.0
            )
            self.stages.append(Stage(label, frac, now - started, peak))
            if self.with_memory:
                tracemalloc.reset_peak()
        self._pending = (message, float(fraction), now)

    def close(self) -> list[Stage]:
        """Stages completed so far (the pending one has no measured end)."""
        self._pending = None
        return self.stages


@dataclass
class FuncCost:
    """One cProfile entry: a function and what it costs."""

    name: str      # "refine_bpm_with_onsets (tempo.py:69)"
    self_s: float  # time spent in the function itself
    cum_s: float   # including what it calls
    own: bool      # belongs to this project (src/v2m), not to a library


@dataclass
class Measurement:
    """One run: the stage rows, what was detected, and the profiling table."""

    path_name: str = ""
    opts: AnalyzeOptions = field(default_factory=AnalyzeOptions)
    warmup: bool = False
    rows: list[Stage] = field(default_factory=list)
    total_s: float = 0.0
    audio_s: float = 0.0
    peak_mib: float = 0.0   # whole-analysis tracemalloc peak (Python allocs)
    sr: int = audio_io.SR
    n_notes: int = 0
    bpm: float | None = None
    key: str = "—"
    meter: str = "—"
    profile: list[FuncCost] = field(default_factory=list)


def measure(
    y, path: Path, load_s: float, opts: AnalyzeOptions, out_dir: Path
) -> Measurement:
    """Analyze then export an already decoded signal, timing every step."""
    m = Measurement(path_name=str(path), opts=opts)
    m.rows.append(Stage("file decoding", None, load_s))
    m.audio_s = len(y) / audio_io.SR

    # tracemalloc wraps the WHOLE measured run (analysis + export): the
    # decode above already paid its allocations. Each stage reports its own
    # span's peak (reset at every stage boundary); the TOTAL row carries the
    # whole run's peak. The trace overhead on timing is a few percent,
    # accepted for the memory column.
    with_memory = True
    timer = StageTimer(with_memory=with_memory)
    if with_memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    res = analyze_signal(y, audio_io.SR, source=str(path), opts=opts, progress=timer)
    analyze_s = time.perf_counter() - t0
    m.rows.extend(timer.close())

    t1 = time.perf_counter()
    export_result(res, out_dir / path.stem, with_pitch_curve=True)
    export_s = time.perf_counter() - t1
    export_peak = (
        tracemalloc.get_traced_memory()[1] / (1024 * 1024) if with_memory else 0.0
    )
    m.rows.append(
        Stage("export (5 artifacts, disk write)", None, export_s, export_peak)
    )
    if with_memory:
        tracemalloc.stop()
    m.peak_mib = max((r.peak_mib for r in m.rows), default=0.0)

    m.total_s = load_s + analyze_s + export_s
    m.n_notes = len(res.notes)
    m.bpm = res.tempo.bpm if res.tempo else None
    m.key = res.key.best.label if res.key else "—"
    m.meter = res.meter.label if res.meter else "4/4 (défaut)"
    return m


def warm_up(y, path: Path, opts: AnalyzeOptions, seconds: float = 2.0) -> None:
    """Analyze a short slice and throw the result away.

    The first analysis of a process pays for what has nothing to do with the
    audio: librosa/numba imports and the JIT compilation of the onset and
    pYIN kernels (over a second on this machine). Measured cold, that cost
    lands on the first stage and hides where the time really goes.
    """
    slice_ = y[: int(seconds * audio_io.SR)]
    analyze_signal(slice_, audio_io.SR, source=str(path), opts=opts)


def measure_run(
    y, path: Path, load_s: float, opts: AnalyzeOptions, out_dir: Path, profile: bool
) -> Measurement:
    """`measure`, optionally under cProfile (the profile is part of the run)."""
    if not profile:
        return measure(y, path, load_s, opts, out_dir)
    profiler = cProfile.Profile()
    profiler.enable()
    m = measure(y, path, load_s, opts, out_dir)
    profiler.disable()
    m.profile = top_functions(profiler)
    return m


def top_functions(profiler: cProfile.Profile) -> list[FuncCost]:
    """Every profiled function, heaviest self time first.

    C and builtin entries are kept (they explain most of a library's cost)
    but flagged: `own` marks the code of this project, which is the only
    part the table's reader can actually change.
    """
    stats = pstats.Stats(profiler).stats  # {(file, line, func): (cc, nc, tt, ct, ...)}
    out: list[FuncCost] = []
    for (filename, lineno, func), values in stats.items():
        if filename.endswith(".py"):
            path = Path(filename).resolve()
            name = f"{func} ({path.name}:{lineno})"
            own = path.is_relative_to(PACKAGE_DIR)
        else:
            name, own = f"~ {func}", False  # C or builtin: no source to edit
        out.append(FuncCost(name, values[2], values[3], own))
    out.sort(key=lambda row: row.self_s, reverse=True)
    return out


def average(runs: list[Measurement]) -> Measurement:
    """Mean of one stage per label, in first-seen order (the runs align)."""
    if len(runs) == 1:
        return runs[0]
    total: dict[str, float] = {}
    counts: dict[str, int] = {}
    peaks: dict[str, float] = {}
    for measurement in runs:
        for row in measurement.rows:
            total[row.label] = total.get(row.label, 0.0) + row.seconds
            counts[row.label] = counts.get(row.label, 0) + 1
            peaks[row.label] = max(peaks.get(row.label, 0.0), row.peak_mib)
    fractions = {row.label: row.fraction for row in runs[0].rows}
    m = runs[-1]  # keeps the detected notes and the profile of the last run
    m.rows = [
        Stage(label, fractions.get(label), seconds / counts[label], peaks.get(label, 0.0))
        for label, seconds in total.items()
    ]
    m.total_s = sum(r.total_s for r in runs) / len(runs)
    m.peak_mib = max(r.peak_mib for r in runs)
    return m


def print_table(m: Measurement, runs: int, profile_top: int) -> None:
    print(
        f"\nFile      : {m.path_name} — {m.audio_s:.2f} s, "
        f"{m.sr} Hz ({m.audio_s * m.sr / 1000:.0f}k samples)"
    )
    print(
        f"Detected  : {m.n_notes} notes · "
        f"{'—' if m.bpm is None else f'{m.bpm:.2f} BPM'} · {m.key} · {m.meter}"
    )
    print(
        f"Options   : min_conf={m.opts.min_conf:.2f} · "
        f"min_note={m.opts.min_note_s * 1000:.0f} ms · semitones={m.opts.semitones:.2f}"
        + (f" · mean over {runs} runs" if runs > 1 else "")
        + (" · 2 s warm-up discarded (imports/JIT)" if m.warmup else "")
    )

    print(
        f"\n{'Stage':<{LABEL_WIDTH}} {'Start':>5} {'Time':>9} {'Share':>7} {'Real time':>10} {'Mem peak':>9}"
    )
    print("-" * (TABLE_WIDTH + 10))
    for row in m.rows:
        print(
            f"{_clip(row.label, LABEL_WIDTH):<{LABEL_WIDTH}} {row.start_text:>5} "
            f"{row.seconds:>8.3f}s {_share(row.seconds, m.total_s):>6.1f}% "
            f"{_realtime(row.seconds, m.audio_s):>9.3f}× {row.peak_mib:>7.1f}M"
        )
    print("-" * (TABLE_WIDTH + 10))
    print(
        f"{'TOTAL':<{LABEL_WIDTH}} {'':>5} {m.total_s:>8.3f}s {100.0:>6.1f}% "
        f"{_realtime(m.total_s, m.audio_s):>9.3f}× {m.peak_mib:>7.1f}M"
    )
    print(
        "\n'Real time' = stage duration / audio duration "
        "(0.07x = 14x faster than listening)."
    )
    print(
        "'Mem peak' = the stage's tracemalloc peak (Python allocations, MiB) — "
        "a comparative order of magnitude, not an RSS."
    )

    heaviest = max(m.rows, key=lambda r: r.seconds)
    print(
        f"Heaviest stage: {heaviest.label} — {heaviest.seconds:.3f} s "
        f"({_share(heaviest.seconds, m.total_s):.1f}%)"
    )

    if m.profile:
        total = sum(row.self_s for row in m.profile) or 1.0
        print(
            "\ncProfile, last run — compare these functions against each other, "
            "not with the table above ('Share' = share of the whole profile)."
        )
        for title, rows in (
            ("Project code", [r for r in m.profile if r.own]),
            ("Libraries", [r for r in m.profile if not r.own]),
        ):
            if not rows:
                continue
            print(f"\n{title} — top {min(profile_top, len(rows))} by own time")
            print(f"{'Own':>9} {'Share':>7} {'Cumul':>9}  Function")
            for row in rows[:profile_top]:
                print(
                    f"{row.self_s:>8.3f}s {_share(row.self_s, total):>6.1f}% "
                    f"{row.cum_s:>8.3f}s  {row.name}"
                )


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _share(seconds: float, total: float) -> float:
    return 100.0 * seconds / total if total > 0 else 0.0


def _realtime(seconds: float, audio_s: float) -> float:
    return seconds / audio_s if audio_s > 0 else 0.0


def as_json(m: Measurement, runs: int) -> dict:
    return {
        "file": m.path_name,
        "audio_s": round(m.audio_s, 3),
        "sr": m.sr,
        "runs": runs,
        "total_s": round(m.total_s, 4),
        "n_notes": m.n_notes,
        "bpm": m.bpm,
        "key": m.key,
        "meter": m.meter,
        "stages": [
            {
                "label": row.label,
                "fraction": row.fraction,
                "seconds": round(row.seconds, 4),
                "share_pct": round(_share(row.seconds, m.total_s), 2),
                "peak_mib": round(row.peak_mib, 2),
            }
            for row in m.rows
        ],
        "peak_mib": round(m.peak_mib, 2),
        "profile": [
            {
                "function": row.name,
                "self_s": round(row.self_s, 4),
                "cumulative_s": round(row.cum_s, 4),
                "project_code": row.own,
            }
            for row in m.profile
        ],
    }


def _force_utf8_stdout() -> None:
    """Survive a Windows cp1252 console (accents, symbols...).

    Without this the script dies with UnicodeEncodeError after doing all
    the measuring work; the test runs it with a UTF-8 pipe, the user's
    console is not always so kind.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # not a reconfigurable stream (redirect, IDE, old Python)


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(
        prog="profile_analysis.py",
        description="Times every analysis stage (tempo, pYIN, segmentation, "
        "tonalité, accords, export) et affiche un tableau.",
    )
    parser.add_argument("audio", nargs="?", default=DEFAULT_AUDIO, help="Audio file (default: demo.wav)")
    parser.add_argument("--repeat", type=int, default=1, help="Runs to average (default: 1)")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not discard a 2 s warm-up (imports + numba compilation)",
    )
    parser.add_argument("--profile", action="store_true", help="Add the cProfile table (own time)")
    parser.add_argument("--top", type=int, default=12, help="cProfile table rows (default: 12)")
    parser.add_argument("--json", help="Write the measurements to this JSON file")
    parser.add_argument("--output", help="Keep the exported artifacts in this folder (else: temporary)")
    args = parser.parse_args(argv)

    path = Path(args.audio)
    if not path.exists():
        print(tr("err file not found").format(path=path), file=sys.stderr)
        return 1
    if path.suffix.lower() not in audio_io.AUDIO_EXTS:
        fmts = ", ".join(e.lstrip(".") for e in audio_io.AUDIO_EXTS)
        print(tr("err bad format").format(ext=path.suffix or "?", fmts=fmts), file=sys.stderr)
        return 1

    opts = AnalyzeOptions()
    runs = max(1, args.repeat)
    tmp: tempfile.TemporaryDirectory | None = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="v2m-profile-")
        out_dir = Path(tmp.name)

    try:
        print(f"[profil] {path.name} …", flush=True)
        t0 = time.perf_counter()
        y = audio_io.load_file(str(path))
        load_s = time.perf_counter() - t0  # measured once, shared by every run
        if not args.no_warmup:
            warm_up(y, path, opts)
        measurements = [
            measure_run(y, path, load_s, opts, out_dir, args.profile) for _ in range(runs)
        ]
        m = average(measurements)
        m.warmup = not args.no_warmup
        print_table(m, runs, args.top)
        if args.output:
            print(f"\nArtefacts écrits dans : {out_dir}")
        if args.json:
            Path(args.json).write_text(
                json.dumps(as_json(m, runs), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Mesures écrites dans  : {args.json}")
    finally:
        if tmp is not None:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
