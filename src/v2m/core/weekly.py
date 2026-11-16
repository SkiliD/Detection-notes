"""The WEEKLY progress aggregate: what the recap PDF prints.

One module owns the shape (the dict `build_weekly` returns) so the PDF
engraver and any future consumer (mail, JSON export) read the same
numbers the coherence page shows. Everything is derived from `core`
evidence that already exists: the journal's weekly contour trend, the
practice-time report and the day subtotals from the take audits. No new
measurements, no new thresholds — an aggregate, not an opinion.
"""

from __future__ import annotations

from pathlib import Path

from . import coherence as coh
from .i18n import tr

# i18n keys (user-facing output): the table lives in i18n.py like every
# visible string — the AST guard (test_docs_guard) refuses them elsewhere.
WEEKLY_TITLE = "weekly title"
WEEKLY_HEALTH = "weekly health"
WEEKLY_WINDOWS = "weekly windows"
WEEKLY_DAYS = "weekly days"
WEEKLY_WINDOW_ROW = "weekly window row"
WEEKLY_DAY_ROW = "weekly day row"
WEEKLY_NO_DATA = "weekly no data"


def build_weekly(root: Path | None = None) -> dict:
    """Assemble the weekly recap's sections from the coherence evidence.

    Sections: the multi-week health trend (the chart's own rows), the
    practice-window comparison (morning/afternoon/evening), and the day
    subtotals. The journal is refreshed first so a deleted take keeps its
    week (the trend's rule) and the newest audits are in.
    """
    root = Path(root) if root is not None else coh.LIVE_TAKES_DIR
    snapshot = coh.build_snapshot(root)
    return {
        "takes": len(snapshot.takes),
        "health": snapshot.health_weeks,
        "windows": snapshot.practice_windows,
        "days": snapshot.days,
        "regression": snapshot.regression,
    }


def weekly_text(data: dict) -> str:
    """The recap as text (the PDF renders this same content; tests read it)."""
    import io

    out = io.StringIO()
    out.write(tr(WEEKLY_TITLE) + "\n")
    if not data["health"] and not data["windows"] and not data["days"]:
        out.write(tr(WEEKLY_NO_DATA) + "\n")
        return out.getvalue()
    out.write("\n" + tr(WEEKLY_HEALTH) + "\n")
    for w in data["health"]:
        live = w["live_notes"] if w["live_notes"] is not None else "-"
        final = w["final_notes"] if w["final_notes"] is not None else "-"
        out.write(
            tr("coh week row").format(
                week=w["week"],
                takes=w["takes"],
                rate=f"{w['contour_rate'] * 100:.0f}%",
                live=live,
                final=final,
            )
            + "\n"
        )
    if data["windows"]:
        out.write("\n" + tr(WEEKLY_WINDOWS) + "\n")
        for w in data["windows"]:
            rate = f"{w['suspect_rate']:.1f}" if w["suspect_rate"] is not None else "-"
            contours = f"{w['contour_rate'] * 100:.0f}%" if w["contour_rate"] is not None else "-"
            out.write(
                tr(WEEKLY_WINDOW_ROW).format(
                    window=tr(f"time {w['window']}"),
                    takes=w["takes"],
                    rate=rate,
                    contours=contours,
                )
                + "\n"
            )
    if data["days"]:
        out.write("\n" + tr(WEEKLY_DAYS) + "\n")
        for d in data["days"]:
            checked = f"{d['checked']}" if d["checked"] is not None else "-"
            rate = f"{d['rate_per_100']:.2f}" if d["rate_per_100"] is not None else "-"
            out.write(
                tr(WEEKLY_DAY_ROW).format(
                    date=d["date"],
                    sessions=d["sessions"],
                    takes=d["takes"],
                    checked=checked,
                    suspects=d["suspects"],
                    rate=rate,
                )
                + "\n"
            )
    return out.getvalue()


# The three windows' labels (the keys `time morning/afternoon/evening`).
WINDOW_LABEL_KEYS = {
    "morning": "time morning",
    "afternoon": "time afternoon",
    "evening": "time evening",
}
