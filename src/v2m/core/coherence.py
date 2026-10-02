"""Session-level coherence: aggregate the per-take audits into a trend.

`continuity.py` audits ONE take (its `.coherence.csv`); this module answers
the FOLLOW-UP question — **are swallowed phrases decreasing session over
session?** — by aggregating every take of `prises_live/` into sessions.

Decisions owned here, each with its reason:

- **Sessions are split on a TIME GAP (30 min), not on the date alone**: a
  morning session and a late-night session of the same date are two
  practices; a session that crosses midnight is ONE practice. A day
  subtotal keeps the calendar readable.
- **The suspect rate is per 100 audited notes**, never per take: ten
  suspects over 200 notes is better than three over 20 — raw counts cannot
  be compared when the number of notes audited changes. Each session also
  carries its DELTA against the previous one, so the trend reads without
  comparing rows by hand.
- **Every take is verified before being counted** (`verify.verify_take`):
  perfectly readable coherence CSVs can describe the WRONG signal when a
  hand-off mixed the artifacts. A take whose artifacts
  diverge is flagged in the table and its notes are NOT trusted into the
  rate (they still count in `takes`, visible).
- **A persisted journal** (`coherence_journal.json`) remembers each take's
  audit next to its artifacts: folder cleanup (deleting old takes) must
  not rewrite history — the delta against a previous session would
  otherwise silently re-baseline.

Every string the user reads goes through i18n (`tr()`), so the CLI and the
GUI page follow the app's language choice; the keys live in the two tables
like every visible string (the tr() guard sees them statically here).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import takes
from .takes import LIVE_TAKES_DIR, parse_take_name
from .takes import read_sidecar as _read_sidecar
from .verify import verify_take

CSV_NAME = ".coherence.csv"
JOURNAL_NAME = "coherence_journal.json"

# Two takes more than this far apart belong to two sessions (a pause longer
# than half an hour is a new practice, not a breath).
SESSION_GAP_S = 30 * 60

# Per-take CSV columns (superset of what continuity wrote before: the
# plateau now carries its time and its signed interval).
CSV_HEADER = (
    "start_s", "duration_s", "midi", "inner_range_st",
    "secondary_plateau_s", "secondary_plateau_start_s",
    "secondary_plateau_midi", "secondary_plateau_offset_st",
)

# The real-time health REGRESSION threshold, in contour-rate POINTS between
# two consecutive weeks (0.52 -> 0.68 reads +16 points and rings the bell;
# ±5 points is a practice-to-practice wiggle). Owned here: the GUI's
# warning and any future consumer read the SAME number.
REGRESSION_DELTA = 10.0


@dataclass
class TakeAudit:
    """One take, as the aggregator reads it."""

    stem: str
    date: str
    clock: str
    epoch_s: float          # the take's timestamp, for the gap rule
    suspects: list[dict] = field(default_factory=list)
    checked: int | None = None
    verify_ok: bool | None = None  # None: nothing to verify (no artifacts)
    verify_lines: list[str] = field(default_factory=list)
    # the take's live/final contour rate (its JSON's live_vs_final block,
    # written when the take's live notes were handed over at export): the
    # health indicator of real-time recognition. None on takes exported
    # before the block existed — the column shows "-", it never guesses.
    contour_rate: float | None = None
    live_notes: int | None = None

    @property
    def rate_per_100(self) -> float | None:
        """Suspects per 100 audited notes; None when the count is unknown."""
        if self.checked is None:
            return None
        return 100.0 * len(self.suspects) / self.checked if self.checked else 0.0


@dataclass(frozen=True)
class EvidenceSnapshot:
    """One pass of take evidence and every aggregate derived from it."""

    root: Path
    takes: list[TakeAudit]
    sessions: list[dict]
    days: list[dict]
    practice_windows: list[dict]
    health_weeks: list[dict]
    regression: dict | None
    journal: dict



def build_snapshot(root: Path) -> EvidenceSnapshot:
    """Read audits once, refresh the journal, and derive all aggregate views."""
    takes_list = takes_under(root)
    journal = update_journal(root, takes_list)
    sessions = session_rows(takes_list)
    weeks = contour_trend_entries(journal)
    return EvidenceSnapshot(
        root=root,
        takes=takes_list,
        sessions=sessions,
        days=day_rows(sessions, takes_list) if takes_list else [],
        practice_windows=practice_time_report(takes_list),
        health_weeks=weeks,
        regression=regression_report(weeks),
        journal=journal,
    )


def _epoch(date: str, clock: str) -> float:
    """The take's timestamp as seconds (the gap rule compares durations).

    The take-name grammar is shape-only (`takes.parse_take_name`), so a
    hand-made but well-formed name still parses; an unparseable DATE is a
    day of -1 and sorts first, never a crash.
    """
    import calendar
    import datetime as dt

    try:
        moment = dt.datetime.strptime(date + clock, "%Y%m%d%H%M%S")
        return calendar.timegm(moment.timetuple())
    except ValueError:
        return -1.0


def load_take(csv_path: Path) -> TakeAudit:
    """One take summarized from its coherence CSV (+ its JSON, + verify).

    The audit reads the take's OWN artifacts in its folder: the CSV is the
    suspect list, the JSON's `coherence.checked` the audited count, and
    `verify_take` the WAV/JSON consistency gate.
    """
    stem = csv_path.name[: -len(CSV_NAME)]
    stamp = parse_take_name(stem)
    date, clock = stamp if stamp else ("", "")
    suspects: list[dict] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                suspects.append(
                    {
                        "midi": int(row["midi"]),
                        "inner_range_st": float(row["inner_range_st"]),
                        "secondary_plateau_s": float(row["secondary_plateau_s"]),
                        "secondary_plateau_start_s": (
                            float(row["secondary_plateau_start_s"])
                            if row.get("secondary_plateau_start_s") not in (None, "")
                            else None
                        ),
                        "secondary_plateau_midi": (
                            int(row["secondary_plateau_midi"])
                            if row.get("secondary_plateau_midi") not in (None, "")
                            else None
                        ),
                        "secondary_plateau_offset_st": (
                            float(row["secondary_plateau_offset_st"])
                            if row.get("secondary_plateau_offset_st") not in (None, "")
                            else None
                        ),
                    }
                )
    except (OSError, ValueError, KeyError, csv.Error):
        # a truncated or hand-edited CSV is a broken AUDIT, not a broken
        # table: the take shows as suspect-less (its coherence verdict in
        # the JSON stays authoritative) and the aggregation goes on. The
        # read is guarded ONCE here; the per-row parse shares the guard.
        suspects = []
    checked: int | None = None
    contour_rate: float | None = None
    live_notes: int | None = None
    json_path = csv_path.with_name(stem + ".json")
    if json_path.exists():
        try:
            rep = json.loads(json_path.read_text(encoding="utf-8"))
            block = rep.get("coherence") or {}
            if isinstance(block.get("checked"), int):
                checked = int(block["checked"])
            # the live/final decomposition, when the take exported one: the
            # per-session contour trend reads here, take by take
            lvf = rep.get("live_vs_final") or {}
            if isinstance(lvf.get("rate"), (int, float)):
                contour_rate = float(lvf["rate"])
            if isinstance(lvf.get("live"), int):
                live_notes = int(lvf["live"])
        except (OSError, ValueError):
            checked = None  # an unreadable JSON must not cost the table
    if contour_rate is None:
        # the replay's own archive (`<stem>.live_vs_final.json`, written by
        # the GUI's ReplayWorker or the replay script) fills the gap when
        # the report predates the block: the sidecar IS the take's live/final
        # gap, in the SAME vocabulary — read here, never rewritten
        sdata = _read_sidecar(csv_path.parent, stem)
        if sdata is not None:
            if isinstance(sdata.get("rate"), (int, float)):
                contour_rate = float(sdata["rate"])
            if isinstance(sdata.get("live"), int):
                live_notes = int(sdata["live"])
        # no sidecar, no data: the trend simply waits for a replay
    ok, lines = verify_take(csv_path.with_name(stem))
    # the NOTHING-ON-DISK case (`verify_none` is its only line): nothing was
    # verified, not failed — the table shows "-", it never guesses
    verified = None if len(lines) <= 1 else ok
    return TakeAudit(
        stem=stem,
        date=date,
        clock=clock,
        epoch_s=_epoch(date, clock),
        suspects=suspects,
        checked=checked,
        verify_ok=verified,
        verify_lines=lines if not (ok and verified) else [],
        contour_rate=contour_rate,
        live_notes=live_notes,
    )


def _trusted(takes: list[TakeAudit]) -> list[TakeAudit]:
    """The takes whose artifacts were verified (or had nothing to verify).

    A take whose WAV and JSON diverge describes an UNKNOWN signal: its
    suspects and its audited count must not
    feed the rates — it stays in the table, flagged, visible.
    """
    return [t for t in takes if t.verify_ok is not False]


def takes_under(root: Path) -> list[TakeAudit]:
    """Every take to report, chronologically, only real take names.

    An export of a file (`chant.coherence.csv`) is noise here, not a take
    to follow: only a name `takes.parse_take_name` recognizes aggregates.
    """
    takes: list[TakeAudit] = []
    for csv_path in sorted(root.glob(f"*/*{CSV_NAME}")):
        stem = csv_path.name[: -len(CSV_NAME)]
        if parse_take_name(stem) is None:
            continue
        try:
            takes.append(load_take(csv_path))
        except (OSError, ValueError, KeyError):
            continue  # one unreadable take must not cost the others
    takes.sort(key=lambda t: (t.epoch_s, t.stem))
    return takes


def session_rows(takes: list[TakeAudit]) -> list[dict]:
    """One row per SESSION (a 30-min gap splits), with the trend deltas.

    Each row carries: the label (first take's date+clock), the takes, the
    audited-notes total (None when ANY take's count is unknown — a missing
    JSON must not read as zero), the suspects, the rate per 100 notes, the
    mean hidden-plateau length and the DELTA of the rate against the
    previous session (None on the first session: no previous row exists).
    """
    rows: list[dict] = []
    group: list[TakeAudit] = []
    prev: list[TakeAudit] = []
    for t in takes:
        if (
            group
            and t.epoch_s >= 0
            and group[-1].epoch_s >= 0
            and t.epoch_s - group[-1].epoch_s > SESSION_GAP_S
        ):
            rows.append(_session_row(group, prev))
            prev, group = group, [t]
        else:
            group.append(t)
    if group:
        rows.append(_session_row(group, prev))
    return rows


def _session_row(group: list[TakeAudit], prev: list[TakeAudit]) -> dict:
    # the RATES read only the verified takes: a diverging take's counts
    # describe an unknown signal (it stays in `takes`, flagged in the table)
    trusted = _trusted(group)
    checked_list = [t.checked for t in trusted]
    checked = (
        sum(c for c in checked_list if c is not None)
        if trusted and all(c is not None for c in checked_list)
        else None
    )
    suspects = [s for t in trusted for s in t.suspects]
    rate = (
        round(100.0 * len(suspects) / checked, 2)
        if checked is not None and checked
        else (0.0 if checked == 0 else None)
    )
    # the contour rate: the MEAN of the take rates that exist (a take
    # exported without its live reading does not dilute the average)
    rates = [t.contour_rate for t in trusted if t.contour_rate is not None]
    contour = round(sum(rates) / len(rates), 3) if rates else None
    # the delta compares RATES, and the previous session's rate only exists
    # when there IS a previous session with a known audited count — an empty
    # history is not a 0.0 rate (the first session would otherwise read a
    # fake +5.0 "worse than nothing")
    prev_rate: float | None = None
    if prev:
        prev_trusted = _trusted(prev)
        prev_checked = [t.checked for t in prev_trusted]
        prev_total = (
            sum(c for c in prev_checked if c is not None)
            if all(c is not None for c in prev_checked)
            else None
        )
        prev_suspects = sum(len(t.suspects) for t in prev_trusted)
        prev_rate = (
            round(100.0 * prev_suspects / prev_total, 2)
            if prev_total
            else (0.0 if prev_total == 0 else None)
        )
    deltas = (
        None
        if rate is None or prev_rate is None
        else round(rate - prev_rate, 2)
    )
    plateaus = [
        s["secondary_plateau_s"] for s in suspects
        if s.get("secondary_plateau_s") is not None
    ]
    return {
        "label": f"{group[0].date} {group[0].clock}",
        "date": group[0].date,
        "takes": len(group),
        "checked": checked,
        "suspects": len(suspects),
        "rate_per_100": rate,
        "contour_rate": contour,
        "delta_vs_prev": deltas,
        "plateau_mean_s": (
            round(sum(plateaus) / len(plateaus), 2) if plateaus else 0.0
        ),
        "plateau_max_s": max(plateaus) if plateaus else 0.0,
        "verify_bad": sum(1 for t in group if t.verify_ok is False),
        "takes_list": list(group),
    }


def day_rows(session_rows_list: list[dict], takes: list[TakeAudit]) -> list[dict]:
    """Subtotals per calendar day, from the TAKES' own dates.

    A session follows the CLOCK (30-min gaps), the day follows the DATE:
    both are wanted — the day answers "how did I practice this date". A
    session that crosses midnight (the 23:50 -> 00:05 case) is ONE session
    (attributed to the day it started) but its takes count on their OWN
    days — grouping the session's whole content under its first date would
    move a 00:05 take to yesterday.
    """
    by_date: dict[str, list[TakeAudit]] = {}
    for t in takes:
        by_date.setdefault(t.date, []).append(t)
    session_starts: dict[str, int] = {}
    for row in session_rows_list:
        session_starts[row["date"]] = session_starts.get(row["date"], 0) + 1
    out = []
    for date in sorted(by_date):
        group = by_date[date]
        checked = [t.checked for t in group]
        suspects = sum(len(t.suspects) for t in group)
        total = (
            sum(c for c in checked if c is not None)
            if all(c is not None for c in checked)
            else None
        )
        out.append(
            {
                "date": date,
                "sessions": session_starts.get(date, 0),
                "takes": len(group),
                "checked": total,
                "suspects": suspects,
                "rate_per_100": (
                    round(100.0 * suspects / total, 2)
                    if total
                    else (0.0 if total == 0 else None)
                ),
            }
        )
    return out


# --------------------------------------------------------------------- #
# The persisted journal: the trend survives folder cleanup
# --------------------------------------------------------------------- #


def journal_path(root: Path) -> Path:
    return root / JOURNAL_NAME


def read_journal(root: Path) -> dict:
    """The persisted per-take audit, {} when absent or unreadable."""
    try:
        return json.loads(journal_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def update_journal(root: Path, takes: list[TakeAudit]) -> dict:
    """Merge this pass's audits into the journal and write it back.

    A take re-audited REPLACES its entry (the newest reading wins — the
    artifacts were re-exported); a take absent from the folder KEEPS its
    entry (cleanup must not rewrite history, which is the journal's whole
    point). Each entry also carries the take's live/final CONTOUR RATE
    when its JSON published a `live_vs_final` block: the multi-week
    real-time health trend reads HERE, from the journal, even after the
    take's folder has been cleaned up. Best-effort: a read-only folder
    keeps the in-memory merge.
    """
    data = read_journal(root)
    for t in takes:
        data[t.stem] = {
            "date": t.date,
            "clock": t.clock,
            "epoch_s": t.epoch_s,
            "checked": t.checked,
            "suspects": len(t.suspects),
            "rate_per_100": t.rate_per_100,
            "verify_ok": t.verify_ok,
            "contour_rate": t.contour_rate,
            "live_notes": t.live_notes,
            "final_notes": (
                int(round(t.live_notes / t.contour_rate))
                if t.live_notes and t.contour_rate
                else None
            ),
        }
    try:
        journal_path(root).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (OSError, TypeError, ValueError):
        # unwritable folder, or a value json cannot serialize: the merge
        # above still served this pass, only the persistence is lost
        pass
    return data


# --------------------------------------------------------------------- #
# The multi-week real-time health trend, read from the journal
# --------------------------------------------------------------------- #


def _week_key(entry: dict) -> str:
    """ISO year + week of a journal entry (its own date field)."""
    import datetime as dt

    try:
        d = dt.datetime.strptime(entry["date"], "%Y%m%d")
        return f"{d.isocalendar()[0]}-S{d.isocalendar()[1]:02d}"
    except (KeyError, ValueError):
        return "?"


def contour_trend(root: Path) -> list[dict]:
    """The weekly real-time health trend, from the JOURNAL (not the folder)."""
    return contour_trend_entries(read_journal(root))


def contour_trend_entries(data: dict) -> list[dict]:
    """Build the weekly trend from an already-read journal snapshot.

    A take without a live/final block does not dilute a week's mean, and
    entries retained after folder cleanup remain part of the trend.
    """
    weeks: dict[str, list[dict]] = {}
    for stem, entry in data.items():
        if not isinstance(entry, dict) or entry.get("contour_rate") is None:
            continue
        weeks.setdefault(
            _week_key(entry),
            [],
        ).append({**entry, "stem": entry.get("stem", stem)})
    out = []
    for week in sorted(k for k in weeks if k != "?"):
        entries = weeks[week]
        rates = [e["contour_rate"] for e in entries]
        out.append(
            {
                "week": week,
                "takes": len(entries),
                "contour_rate": round(sum(rates) / len(rates), 3),
                "live_notes": sum(int(e.get("live_notes") or 0) for e in entries),
                "final_notes": sum(int(e.get("final_notes") or 0) for e in entries),
                # the week's OWN takes (chronological): the health chart's
                # click-through hands them to the session table — the drill-
                # down reads the SAME journal, it never re-reads the folder
                "stems": sorted(
                    e.get("stem", "") for e in entries if e.get("stem")
                ),
            }
        )
    return out


def regression_report(rows: list[dict]) -> dict | None:
    """The real-time health REGRESSION to announce, or None when healthy.

    A regression is a MEASURED deterioration of the live/final contour rate:
    the LAST row degrades by more than `REGRESSION_DELTA` (10 points) against
    the previous one, AND it is not a lone take facing a multi-take week —
    a single fresh take carries far more variance than a week's average, so
    it must not ring the bell alone (the singer would learn to ignore it).
    Old rows without contour data do not participate: no data is not a
    regression.
    """
    if len(rows) < 2:
        return None
    prev, last = rows[-2], rows[-1]
    if prev.get("contour_rate") is None or last.get("contour_rate") is None:
        return None
    delta = (last["contour_rate"] - prev["contour_rate"]) * 100.0
    if delta <= REGRESSION_DELTA:
        return None
    if last["takes"] == 1 and prev["takes"] > 1:
        return None
    return {
        "delta_points": round(delta, 1),
        "prev_rate": prev["contour_rate"],
        "last_rate": last["contour_rate"],
        "prev_week": prev["week"],
        "last_week": last["week"],
    }


# --------------------------------------------------------------------- #
# The best practice TIME OF DAY: when does the singer sound best?
# --------------------------------------------------------------------- #

# The day's practice windows: a take reads the window its CLOCK starts in.
# Boundaries chosen for a singer's day (a 9-to-5 world): morning ends at
# noon, afternoon at 18:00, the evening carries everything after. A take
# whose clock cannot be parsed has no window — it is counted nowhere here.
DAY_WINDOWS = (
    ("morning", 6, 12),
    ("afternoon", 12, 18),
    ("evening", 18, 6),
)


def _hour(clock: str) -> int | None:
    try:
        return int(clock[:2])
    except (ValueError, TypeError):
        return None


def _window_of(clock: str) -> str | None:
    hour = _hour(clock)
    if hour is None:
        return None
    for name, lo, hi in DAY_WINDOWS:
        if lo <= hi:
            if lo <= hour < hi:
                return name
        elif hour >= lo or hour < hi:  # the evening wraps midnight
            return name
    return None


def practice_time_report(takes: list[TakeAudit]) -> list[dict]:
    """The singer's best window: mean suspect rate and contour rate per one.

    Read from the take audits the session view already uses (no new
    evidence): a window's rate is the mean of the take rates that exist,
    so a take without contour data does not dilute its window. Windows
    with no take are absent rows — the GUI and the weekly PDF show only
    what was sung. The verdict lives in the numbers: the caller renders.
    """
    groups: dict[str, list[TakeAudit]] = {}
    for t in takes:
        w = _window_of(t.clock)
        if w is not None:
            groups.setdefault(w, []).append(t)
    out = []
    for name, _lo, _hi in DAY_WINDOWS:
        group = groups.get(name)
        if not group:
            continue
        rates = [t.rate_per_100 for t in group if t.rate_per_100 is not None]
        contours = [t.contour_rate for t in group if t.contour_rate is not None]
        out.append(
            {
                "window": name,
                "takes": len(group),
                "suspect_rate": round(sum(rates) / len(rates), 2) if rates else None,
                "contour_rate": (
                    round(sum(contours) / len(contours), 3) if contours else None
                ),
            }
        )
    return out


# --------------------------------------------------------------------- #
# The retention: `prises_live/` is the singer's disk, not an archive
# --------------------------------------------------------------------- #

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
    for cache in root.rglob(f"*{takes.EVIDENCE_SUFFIX}"):
        stem = cache.name[: -len(takes.EVIDENCE_SUFFIX)]
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


def sparkline(values, width: int = 10) -> str:
    """The one-line trend of a numeric series: ▁▂▃▅▆▇, left = oldest.

    Monospace-safe (each block is one column wide), rendered from the
    values' own min..max — a flat series is a row of ▄, a falling one
    starts tall. `width` caps the history shown (the LAST values: the
    trend is read right-to-left in time). None values are gaps (␣). Fewer
    than two values read as a single block — no trend to draw.
    """
    if not values:
        return ""
    vals = list(values)[-width:]
    if all(v is None for v in vals):
        return " " * len(vals)
    known = [v for v in vals if v is not None]
    lo, hi = min(known), max(known)
    blocks = "▁▂▃▄▅▆▇█"
    out = []
    for v in vals:
        if v is None:
            out.append(" ")
        elif hi - lo < 1e-9:
            out.append("▄")  # flat: a mid block, not an empty one
        else:
            out.append(blocks[min(7, int((v - lo) / (hi - lo) * 7.999))])
    return "".join(out)


def render(
    takes: list[TakeAudit],
    root: str = str(LIVE_TAKES_DIR),
    rows: list[dict] | None = None,
    days: list[dict] | None = None,
) -> str:
    """The full report, in the current language: sessions, days, takes."""
    import io

    from .i18n import tr

    out = io.StringIO()
    if not takes:
        out.write(tr("coh empty title").format(root=root) + "\n")
        out.write(tr("coh empty hint") + "\n")
        return out.getvalue()
    rows = rows if rows is not None else session_rows(takes)
    days = days if days is not None else day_rows(rows, takes)
    out.write(tr("coh sessions title") + "\n")
    out.write("-" * 96 + "\n")
    out.write(
        f"{tr('coh col session'):<16}{tr('coh col takes'):>7}{tr('coh col checked'):>12}"
        f"{tr('coh col suspects'):>10}{tr('coh col rate'):>8}{tr('coh col delta'):>8}"
        f"{tr('coh col contours'):>9}{tr('coh col trend'):>11}"
        f"{tr('coh col pmean'):>13}{tr('coh col pmax'):>13}"
        f"{tr('coh col verify'):>8}\n"
    )
    # the suspect-rate TREND across sessions: each row's sparkline carries
    # the history UP TO that session (left = oldest, tall = more suspects
    # per 100 notes) — one column per row, no global line to read sideways
    trend_by_label = {
        r["label"]: sparkline(
            [x["rate_per_100"] for x in rows[: i + 1] if x["rate_per_100"] is not None],
            width=12,
        )
        for i, r in enumerate(rows)
    }
    for r in rows:
        checked = f"{r['checked']}" if r["checked"] is not None else "-"
        rate = f"{r['rate_per_100']:.2f}" if r["rate_per_100"] is not None else "-"
        delta = (
            f"{r['delta_vs_prev']:+.2f}"
            if r["delta_vs_prev"] is not None
            else "-"
        )
        contours = (
            f"{r['contour_rate']*100:.0f}%" if r["contour_rate"] is not None else "-"
        )
        bad = f"{r['verify_bad']}" if r["verify_bad"] else "ok"
        out.write(
            f"{r['label']:<16}{r['takes']:>7}{checked:>12}{r['suspects']:>10}"
            f"{rate:>8}{delta:>8}{contours:>9}  {trend_by_label.get(r['label'], ''):<10}"
            f"{r['plateau_mean_s']:>13.2f}"
            f"{r['plateau_max_s']:>13.2f}{bad:>8}\n"
        )
    out.write("-" * 96 + "\n")
    out.write(tr("coh days title") + "\n")
    for d in days:
        checked = f"{d['checked']}" if d["checked"] is not None else "-"
        rate = f"{d['rate_per_100']:.2f}" if d["rate_per_100"] is not None else "-"
        out.write(
            tr("coh day row").format(
                date=d["date"], sessions=d["sessions"], takes=d["takes"],
                checked=checked, suspects=d["suspects"], rate=rate,
            )
            + "\n"
        )
    out.write(tr("coh takes title") + "\n")
    for t in takes:
        checked = f"{t.checked}" if t.checked is not None else "-"
        rate = f"{t.rate_per_100:.2f}" if t.rate_per_100 is not None else "-"
        if t.verify_ok is False:
            out.write(
                tr("coh take divergent").format(
                    stem=t.stem, checked=checked, rate=rate
                )
                + "\n"
            )
            for line in t.verify_lines:
                out.write(f"      {line}\n")
            continue
        if not t.suspects:
            out.write(
                tr("coh take clean").format(stem=t.stem, checked=checked)
                + "\n"
            )
            continue
        out.write(
            tr("coh take suspects").format(
                stem=t.stem, checked=checked, rate=rate, n=len(t.suspects)
            )
            + "\n"
        )
        for s in t.suspects:
            at = (
                tr("coh plateau at").format(t=s["secondary_plateau_start_s"])
                if s.get("secondary_plateau_start_s") is not None
                else ""
            )
            out.write(
                tr("coh suspect row").format(
                    midi=s["midi"],
                    span=s["inner_range_st"],
                    dur=s["secondary_plateau_s"],
                    at=at,
                    offset=(
                        f"{s['secondary_plateau_offset_st']:+.2f}st"
                        if s.get("secondary_plateau_offset_st") is not None
                        else "?"
                    ),
                    target=(
                        s["secondary_plateau_midi"]
                        if s.get("secondary_plateau_midi") is not None
                        else "?"
                    ),
                )
                + "\n"
            )
    total = sum(len(t.suspects) for t in takes)
    known = [t.checked for t in takes if t.checked is not None]
    checked_total = sum(known)
    out.write("-" * 96 + "\n")
    if total == 0 and checked_total:
        out.write(tr("coh verdict clean").format(takes=len(takes), notes=checked_total) + "\n")
    elif total == 0:
        out.write(tr("coh verdict clean nonotes").format(takes=len(takes)) + "\n")
    else:
        out.write(
            tr("coh verdict suspects").format(
                suspects=total, takes=len(takes), notes=checked_total if known else "-"
            )
            + "\n"
        )
    return out.getvalue()


# --------------------------------------------------------------------- #
# One line per session, for the GUI page (same numbers, no argparse)
# --------------------------------------------------------------------- #


def session_table_text(
    takes: list[TakeAudit],
    root: Path | None = None,
    rows: list[dict] | None = None,
    weeks: list[dict] | None = None,
) -> str:
    """Render the session table, optionally using a prebuilt evidence snapshot."""
    import io

    from .i18n import tr

    out = io.StringIO()
    rows = rows if rows is not None else session_rows(takes)
    out.write(
        f"{tr('coh col session'):<16}{tr('coh col takes'):>7}{tr('coh col checked'):>12}"
        f"{tr('coh col suspects'):>10}{tr('coh col rate'):>8}{tr('coh col delta'):>8}"
        f"{tr('coh col contours'):>9}  {tr('coh col trend')}\n"
    )
    for i, r in enumerate(rows):
        checked = f"{r['checked']}" if r["checked"] is not None else "-"
        rate = f"{r['rate_per_100']:.2f}" if r["rate_per_100"] is not None else "-"
        delta = (
            f"{r['delta_vs_prev']:+.2f}"
            if r["delta_vs_prev"] is not None
            else "-"
        )
        contours = (
            f"{r['contour_rate']*100:.0f}%" if r["contour_rate"] is not None else "-"
        )
        trend = sparkline(
            [x["rate_per_100"] for x in rows[: i + 1] if x["rate_per_100"] is not None],
            width=12,
        )
        out.write(
            f"{r['label']:<16}{r['takes']:>7}{checked:>12}{r['suspects']:>10}"
            f"{rate:>8}{delta:>8}{contours:>9}  {trend}\n"
        )
    out.write(tr("coh gui footer").format(takes=len(takes)) + "\n")
    if weeks is not None:
        selected_weeks = weeks
    elif root is not None:
        selected_weeks = contour_trend(root)
    else:
        selected_weeks = []
    if selected_weeks:
        out.write("\n" + tr("coh weeks title") + "\n")
        out.write(
            sparkline([w["contour_rate"] for w in selected_weeks], width=16)
            + "\n"
        )
        for w in selected_weeks:
            out.write(
                tr("coh week row").format(
                    week=w["week"], takes=w["takes"],
                    rate=f"{w['contour_rate']*100:.0f}%",
                    live=w["live_notes"], final=w["final_notes"],
                )
                + "\n"
            )
    return out.getvalue()
