"""Disk retention for take folders and their derived caches.

The budget concern, separate from the coherence aggregation that carries
sessions, journal and trends: the importers (`write_take` at export time,
`v2m weekly` at recap time) want trimming, not cohort analysis. Two
sweeps, both best-effort — a trim failure never costs the take that
triggered it. The journal keeps every pruned take's audit, so pruning
rewrites nothing the trend depends on.
"""

from __future__ import annotations

from pathlib import Path

from .takes import EVIDENCE_SUFFIX, parse_take_name

# A take folder exceeding this many MB is trimmed: the OLDEST takes are
# deleted first, whole folders (an artifact alone is useless — the WAV is
# what a later replay or verify reads). The evidence cache (.observation.npz)
# is the FIRST thing dropped per take: it is a derived artifact, its cost
# is only time. -1 disables the retention entirely.
RETENTION_MAX_MB = 800


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def prune_observations(root: Path) -> int:
    """Delete the evidence caches whose take folder no longer exists.

    A `.observation.npz` is only readable NEXT to its WAV (`matches_take`
    validates the audio's size+mtime): an orphan cache — the take folder
    deleted by hand — is pure disk waste. Returns the number of files
    removed. Best-effort by design: an unreadable folder costs nothing.
    """
    removed = 0
    for cache in root.rglob(f"*{EVIDENCE_SUFFIX}"):
        stem = cache.name[: -len(EVIDENCE_SUFFIX)]
        # the cache is only usable NEXT to its WAV (`matches_take` validates
        # the audio's size+mtime): a cache whose WAV sibling is gone — the
        # take folder deleted by hand — is pure disk waste
        if not (cache.parent / (stem + ".wav")).exists():
            try:
                cache.unlink()
                removed += 1
            except OSError:
                pass  # read-only media: the waste stays, the tool never fails
    return removed


def prune_takes(root: Path, max_mb: int = RETENTION_MAX_MB) -> list[str]:
    """Trim `root` under `max_mb` by deleting the OLDEST take folders whole.

    The journal keeps every deleted take's audit (the trend survives —
    that is its whole point), so pruning rewrites NOTHING the coherence
    view depends on; the deleted take's WAV is what is gone (a replay of
    it becomes impossible, its week's history stays). Returns the deleted
    stems. -1 disables. Deletion is DELIBERATE here (disk space is the
    request) but narrow: only take folders recognized by the name grammar,
    never the journal, never anything at the root.
    """
    if max_mb is None or max_mb < 0:
        return []
    pruned: list[str] = []
    take_dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and parse_take_name(d.name)),
        key=lambda d: d.name,
    )
    budget = max_mb * 1024 * 1024
    i = 0  # oldest first (the timestamp name sorts chronologically)
    while _dir_size_bytes(root) > budget and i < len(take_dirs):
        victim = take_dirs[i]
        try:
            import shutil

            shutil.rmtree(victim)
            pruned.append(victim.name)
        except OSError:
            pass
        i += 1
    if pruned:
        # whole folders were removed: rmtree already took their caches —
        # the sweep only catches orphans a HAND deletion left behind
        prune_observations(root)
    return pruned
