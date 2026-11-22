"""Measure the WSL bridge's import cost for the suite's package stack.

Run it under WSL with the venv that runs the suite:

    ~/.venvs/v2m/bin/python scripts/import_bench.py

The interpreter's own `site-packages` is the ext4 copy. The script copies that
same tree to `/mnt/c/Users/Public` (overridable with
`V2M_IMPORT_BENCH_MNT`), warms both copies once, then reports the median of
seven fresh imports of numpy, scipy, librosa, numba, soundfile and PyQt6. The
copy is removed afterwards, so a re-run measures the bridge instead of a stale
directory.

The figure cited in README.md, README.fr.md and AGENTS.md comes from this
script. Set `V2M_IMPORT_BENCH_RUNS` to change the sample count and
`V2M_IMPORT_BENCH_SOURCE` to measure another `site-packages` tree.
"""

from __future__ import annotations

import os
import shutil
import statistics
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

IMPORTS = ("numpy", "scipy", "librosa", "numba", "soundfile", "PyQt6")
DEFAULT_RUNS = 7
DEFAULT_MNT_ROOT = Path("/mnt/c/Users/Public")


def _tree_stats(root: Path) -> tuple[int, int]:
    """File count and byte count, without following symlinks."""
    files = size = 0
    for folder, _directories, names in os.walk(root):
        for name in names:
            path = Path(folder) / name
            files += 1
            try:
                size += path.stat().st_size
            except OSError:
                pass  # a dangling entry is not part of the import cost
    return files, size


def _measure(packages: Path, runs: int) -> float:
    """Median seconds to import the stack in a fresh forked process."""
    fork = getattr(os, "fork", None)
    if fork is None:
        raise SystemExit("this measurement needs WSL or Linux (os.fork)")
    samples: list[float] = []
    for _ in range(runs):
        read_fd, write_fd = os.pipe()
        pid = fork()
        if pid == 0:  # child: the parent has imported no package yet
            os.close(read_fd)
            try:
                sys.path.insert(0, str(packages))
                started = time.perf_counter()
                for name in IMPORTS:
                    __import__(name)
                elapsed = time.perf_counter() - started
                os.write(write_fd, f"{elapsed:.6f}".encode())
            except BaseException as error:  # report, then leave through _exit
                os.write(write_fd, f"error: {error!r}".encode())
                os._exit(1)
            finally:
                os.close(write_fd)
            os._exit(0)
        os.close(write_fd)
        with os.fdopen(read_fd, "rb") as pipe:
            answer = pipe.read().decode()
        _pid, status = os.waitpid(pid, 0)
        if status != 0:
            raise RuntimeError(f"importing from {packages} failed: {answer}")
        samples.append(float(answer))
    return statistics.median(samples)


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise SystemExit(f"{name} must be an integer, not {raw!r}") from error
    if value < 1:
        raise SystemExit(f"{name} must be at least 1")
    return value


def main() -> int:
    source = Path(os.environ.get("V2M_IMPORT_BENCH_SOURCE", sysconfig.get_paths()["purelib"]))
    source = source.resolve()
    if not (source / "numpy").is_dir():
        raise SystemExit(f"no numpy under {source}; run this with the venv that runs the suite")
    mnt_root = Path(os.environ.get("V2M_IMPORT_BENCH_MNT", DEFAULT_MNT_ROOT))
    if not mnt_root.is_dir():
        raise SystemExit(f"{mnt_root} is not a directory; set V2M_IMPORT_BENCH_MNT")
    runs = _positive_int("V2M_IMPORT_BENCH_RUNS", DEFAULT_RUNS)

    source_files, source_bytes = _tree_stats(source)
    mnt_copy = Path(tempfile.mkdtemp(dir=mnt_root, prefix="v2m-import-bench-"))
    target = mnt_copy / "site-packages"
    try:
        print(f"copying {source} -> {target}", flush=True)
        shutil.copytree(source, target)
        target_files, target_bytes = _tree_stats(target)
        if (target_files, target_bytes) != (source_files, source_bytes):
            raise SystemExit(
                "the /mnt/c copy differs from the ext4 tree: "
                f"{target_files} files / {target_bytes} bytes vs "
                f"{source_files} / {source_bytes}"
            )

        # The first import of each copy fills the filesystem cache and writes
        # any missing bytecode; neither belongs in the warm-cache figure.
        _measure(source, 1)
        _measure(target, 1)
        ext4 = _measure(source, runs)
        bridge = _measure(target, runs)
    finally:
        shutil.rmtree(mnt_copy, ignore_errors=True)

    mib = source_bytes / (1024 * 1024)
    print(f"tree: {source_files:,} files, {mib:.0f} MiB")
    print(f"ext4: {ext4:.3f} s (median of {runs} warm runs)")
    print(f"/mnt/c: {bridge:.3f} s (median of {runs} warm runs)")
    print(f"ratio: x{bridge / ext4:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
