"""Graphic trends of the coherence follow-up: sparkline, weeks, panel.

Three pins, each for a measured reason:

1. **sparkline**: a flat series is a row of ▄ (not empty blocks — a
   constant series is information), None is a GAP (␣), the width keeps
   the LAST values (a trend reads to the right);
2. **multi-week trend (`contour_trend`)**: read from the JOURNAL, not
   the folder — a deleted take keeps its week's history (that is what
   the journal is for), a take without `live_vs_final` does not dilute
   its week's mean;
3. **the screen**: the trend column appears in the sessions table (GUI
   footer) and the weeks section only exists when a root is provided;
   the replay worker does not interfere with the aggregation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core import coherence as coh
from v2m.core.i18n import set_language


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _restore_language():
    """Tests that force a language give it back at the end: the suite's
    later modules (test_continuity checks "ECART" in fr) must not inherit
    a dissipating fr/en flip."""
    from v2m.core.i18n import get_language, set_language

    lang = get_language()
    yield
    set_language(lang)


# --------------------------------------------------------------------------- #
# 1. The sparkline: a numeric series as one monospace line
# --------------------------------------------------------------------------- #


def test_a_flat_series_is_a_row_of_mid_blocks_not_empty():
    assert coh.sparkline([5.0, 5.0, 5.0]) == "▄▄▄"


def test_none_values_are_gaps_and_width_keeps_the_last_values():
    # None = no data (a session with no audited notes): a gap
    out = coh.sparkline([1.0, None, 3.0])
    assert out[1] == " "
    assert out[0] in "▁▂" and out[2] in "▆▇█"
    # width: only the LAST values survive (the trend reads right = recent)
    out = coh.sparkline([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], width=3)
    assert len(out) == 3 and out == "▁▄█"


def test_empty_and_single_value_series():
    assert coh.sparkline([]) == ""
    assert len(coh.sparkline([2.0])) == 1  # no trend to draw, one block


def test_the_render_table_carries_a_trend_column_per_session_row(tmp_path):
    set_language("en")
    # three SESSIONS (45-min gaps): one trend column per session row
    for h, m in [(9, 0), (9, 45), (10, 30)]:
        stem = f"live_20260927_{h:02d}{m:02d}00"
        d = tmp_path / stem
        d.mkdir()
        (d / f"{stem}.json").write_text(
            json.dumps({"coherence": {"checked": 10, "suspects": []}}),
            encoding="utf-8",
        )
        (d / f"{stem}.coherence.csv").write_text("h\n", encoding="utf-8")
    takes = coh.takes_under(tmp_path)
    text = coh.render(takes, root=str(tmp_path))
    header = text.splitlines()[2]
    assert "trend" in header
    # each session row ends with its own sparkline (history up to itself)
    body = [ln for ln in text.splitlines() if ln.startswith("20260927")]
    assert len(body) == 3
    assert all(set("▁▂▃▄▅▆▇█") & set(ln) for ln in body)


# --------------------------------------------------------------------------- #
# 2. The multi-week trend, read from the JOURNAL
# --------------------------------------------------------------------------- #


def _journal_with(root, entries):
    (root / coh.JOURNAL_NAME).write_text(
        json.dumps(entries), encoding="utf-8"
    )


def test_contour_trend_groups_by_iso_week_and_means_the_rates(tmp_path):
    _journal_with(tmp_path, {
        "live_20260921_090000": {
            "date": "20260921", "clock": "090000", "epoch_s": 1000.0,
            "checked": 10, "suspects": 0, "rate_per_100": 0.0,
            "verify_ok": True, "contour_rate": 0.50, "live_notes": 20,
            "final_notes": 40,
        },
        "live_20260922_090000": {
            "date": "20260922", "clock": "090000", "epoch_s": 2000.0,
            "checked": 10, "suspects": 0, "rate_per_100": 0.0,
            "verify_ok": True, "contour_rate": 0.30, "live_notes": 10,
            "final_notes": 20,
        },
        "live_20260928_090000": {  # next ISO week
            "date": "20260928", "clock": "090000", "epoch_s": 8000.0,
            "checked": 10, "suspects": 0, "rate_per_100": 0.0,
            "verify_ok": True, "contour_rate": 0.10, "live_notes": 15,
            "final_notes": 30,
        },
    })
    trend = coh.contour_trend(tmp_path)
    assert [w["week"] for w in trend] == ["2026-S39", "2026-S40"]
    assert trend[0]["contour_rate"] == pytest.approx(0.40)
    assert trend[0]["takes"] == 2
    assert trend[0]["live_notes"] == 30
    assert trend[1]["contour_rate"] == pytest.approx(0.10)


def test_a_take_without_contour_data_never_dilutes_its_week(tmp_path):
    _journal_with(tmp_path, {
        "live_20260921_090000": {
            "date": "20260921", "clock": "090000", "epoch_s": 1000.0,
            "checked": 5, "suspects": 0, "rate_per_100": 0.0,
            "verify_ok": True, "contour_rate": 0.50, "live_notes": 10,
            "final_notes": 20,
        },
        "live_20260922_090000": {
            "date": "20260922", "clock": "090000", "epoch_s": 2000.0,
            "checked": 5, "suspects": 0, "rate_per_100": 0.0,
            "verify_ok": True,
            "contour_rate": None, "live_notes": None, "final_notes": None,
        },
    })
    trend = coh.contour_trend(tmp_path)
    assert len(trend) == 1 and trend[0]["contour_rate"] == pytest.approx(0.50)


def test_the_trend_survives_a_deleted_take_via_the_journal(tmp_path):
    """The journal's point: tidying the folder does not rewrite history."""
    _journal_with(tmp_path, {
        "live_20260921_090000": {
            "date": "20260921", "clock": "090000", "epoch_s": 1000.0,
            "checked": 5, "suspects": 0, "rate_per_100": 0.0,
            "verify_ok": True, "contour_rate": 0.50, "live_notes": 10,
            "final_notes": 20,
        },
    })
    # NO take folder on disk at all: the journal alone answers
    trend = coh.contour_trend(tmp_path)
    assert len(trend) == 1
    assert trend[0]["week"] == "2026-S39"


def test_update_journal_persists_the_contour_rate(tmp_path):
    """The journal stores what the weekly trend will re-read later."""
    stem = "live_20260921_090000"
    d = tmp_path / stem
    d.mkdir()
    (d / f"{stem}.json").write_text(
        json.dumps({
            "coherence": {"checked": 10, "suspects": []},
            "live_vs_final": {"live": 20, "final": 31, "contours": 18,
                              "off1": 11, "moved": 6, "absorbed": 1,
                              "rate": 0.9},
        }),
        encoding="utf-8",
    )
    (d / f"{stem}.coherence.csv").write_text("h\n", encoding="utf-8")
    takes = coh.takes_under(tmp_path)
    assert takes[0].contour_rate == 0.9  # read from the JSON's own block
    data = coh.update_journal(tmp_path, takes)
    assert data[stem]["contour_rate"] == 0.9
    assert data[stem]["live_notes"] == 20
    # final_notes is DERIVED (live / rate), never guessed from nowhere
    assert data[stem]["final_notes"] == 22


def test_week_keys_hit_the_iso_year_boundaries():
    """The ISO mapping verified by hand: Dec 29-31 may fall in S1/S53."""
    # 2025-12-29 is a Monday whose Thursday is 2026-01-01: ISO week 1 of 2026
    assert coh._week_key({"date": "20251229"}) == "2026-S01"
    assert coh._week_key({"date": "20260101"}) == "2026-S01"
    # 2020 has 53 ISO weeks: 2020-12-28 (Monday, Thursday = Dec 31) is S53
    assert coh._week_key({"date": "20201228"}) == "2020-S53"
    # a malformed date never crashes the trend: it sorts out ("?")
    assert coh._week_key({"date": "garbage"}) == "?"
    assert coh._week_key({}) == "?"


# --------------------------------------------------------------------------- #
# 3. The screen: the GUI table carries the trend, the weeks with a root
# --------------------------------------------------------------------------- #


def _take(tmp_path: Path, stem: str, checked: int, rate: float | None) -> None:
    d = tmp_path / stem
    d.mkdir(exist_ok=True)
    (d / f"{stem}.json").write_text(
        json.dumps({
            "coherence": {"checked": checked, "suspects": []},
            **({"live_vs_final": {"rate": rate, "live": 10}}
               if rate is not None else {}),
        }),
        encoding="utf-8",
    )
    (d / f"{stem}.coherence.csv").write_text("h\n", encoding="utf-8")


def test_session_table_text_shows_trends_and_the_weeks_section(tmp_path):
    set_language("en")
    for i, rate in enumerate([0.5, 0.3, None]):
        stem = f"live_20260927_090{i}00"
        _take(tmp_path, stem, checked=10 + i, rate=rate)
    takes = coh.takes_under(tmp_path)
    coh.update_journal(tmp_path, takes)  # the weeks section reads the journal
    # with a root: the session table + the weekly health section
    text = coh.session_table_text(takes, root=tmp_path)
    assert "trend" in text
    assert "Real-time health" in text
    assert "2026-S" in text
    # without a root: no weeks section (the GUI always passes its root)
    text_bare = coh.session_table_text(takes)
    assert "Real-time health" not in text_bare


def test_the_home_page_button_renders_the_weekly_trend(qapp, monkeypatch, tmp_path):
    from v2m.gui import main_window as mw

    monkeypatch.setattr(coh, "LIVE_TAKES_DIR", tmp_path)
    for i, rate in enumerate([0.5, 0.4]):
        stem = f"live_2026092{i + 1}_090000"
        _take(tmp_path, stem, checked=10, rate=rate)
    win = mw.MainWindow()
    try:
        win.stack.setCurrentWidget(win.page_home)
        win.on_show_coherence()
        text = win.page_home.coherence_text.toPlainText()
        # the weeks section is on screen: the journal was fed by the takes
        assert "Real-time health" in text or "Santé du temps réel" in text
        assert "2026-S" in text
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 4. The health chart (pyqtgraph) and its drill-down
# --------------------------------------------------------------------------- #


def _seed_weeks(tmp_path, weeks):
    """A journal whose WEEK GROUPS carry the given rates.

    Each group lands in its OWN ISO week (the dates advance 7 days per
    group, starting Monday 2026-09-21 = S39): a journal of same-day takes
    would aggregate into ONE week and could never show a trend.
    """
    import datetime as dt

    entries = {}
    n = 0
    for gi, rates in enumerate(weeks.values()):
        day = dt.date(2026, 9, 21) + dt.timedelta(days=7 * gi)
        date = day.strftime("%Y%m%d")
        for rate in rates:
            n += 1
            stem = f"live_202609{n:02d}_090000"
            entries[stem] = {
                "date": date, "clock": "090000", "epoch_s": float(n),
                "checked": 10, "suspects": 0, "rate_per_100": 0.0,
                "verify_ok": True, "contour_rate": rate,
                "live_notes": 10, "final_notes": 20,
            }
    _journal_with(tmp_path, entries)


def test_the_health_chart_draws_the_weeks_and_hides_when_empty(qapp, monkeypatch, tmp_path):
    from v2m.gui import main_window as mw

    monkeypatch.setattr(coh, "LIVE_TAKES_DIR", tmp_path)
    _seed_weeks(tmp_path, {"a": [0.5, 0.4], "b": [0.3]})
    win = mw.MainWindow()
    try:
        win.page_home.show_coherence()
        assert not win.page_home.health_plot.isHidden(), "the chart must show with weeks"
        assert win.page_home._snapshot.health_weeks[0]["contour_rate"] == pytest.approx(0.45)
        # no data at all: the plot hides, the invite is the honest answer
        (tmp_path / coh.JOURNAL_NAME).unlink()
        win.page_home.show_coherence()
        assert win.page_home.health_plot.isHidden()
    finally:
        win.close()


def test_clicking_a_week_shows_that_weeks_takes(qapp, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from PyQt6.QtCore import QPointF, Qt

    from v2m.gui import main_window as mw

    monkeypatch.setattr(coh, "LIVE_TAKES_DIR", tmp_path)
    # Two take folders land in separate ISO weeks so the drill-down
    # contains only the clicked week's stem.
    first = "live_20260921_090000"
    second = "live_20260928_090000"
    _take(tmp_path, first, checked=5, rate=0.5)
    _take(tmp_path, second, checked=5, rate=0.4)
    journal = {
        first: {
            "date": "20260921", "clock": "090000", "epoch_s": 1.0,
            "checked": 5, "suspects": 0, "rate_per_100": 0.0,
            "verify_ok": True, "contour_rate": 0.5,
            "live_notes": 10, "final_notes": 20,
        },
        second: {
            "date": "20260928", "clock": "090000", "epoch_s": 2.0,
            "checked": 5, "suspects": 0, "rate_per_100": 0.0,
            "verify_ok": True, "contour_rate": 0.4,
            "live_notes": 10, "final_notes": 20,
        },
    }
    _journal_with(tmp_path, journal)
    win = mw.MainWindow()
    try:
        win.page_home.show_coherence()
        vb = win.page_home.health_plot.getPlotItem().vb
        vb.mapSceneToView = lambda p: p  # the click maps to its own point
        ev = SimpleNamespace(
            button=lambda: Qt.MouseButton.LeftButton,
            scenePos=lambda: QPointF(1, 50),  # week index 1 (the second point)
        )
        win.page_home._on_health_clicked(ev)
        text = win.page_home.coherence_text.toPlainText()
        # the clicked week's take is listed BY STEM (the per-take lines under
        # the session table), the other week's take is absent
        assert second in text, "the clicked week's take is missing"
        assert first not in text, "another week bled into it"
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 5. The real-time health regression
# --------------------------------------------------------------------------- #


def _rows(*rates):
    return [
        {"week": f"2026-S{40 + i}", "takes": 2, "contour_rate": r,
         "live_notes": 10, "final_notes": 20, "stems": ["a", "b"]}
        for i, r in enumerate(rates)
    ]


def test_a_rising_contour_rate_is_a_regression_over_the_threshold():
    reg = coh.regression_report(_rows(0.30, 0.60))
    assert reg is not None
    assert reg["delta_points"] == pytest.approx(30.0)
    assert reg["last_rate"] == pytest.approx(0.60)


def test_small_moves_and_missing_data_are_never_a_regression():
    assert coh.regression_report(_rows(0.30, 0.36)) is None  # within ±10 pts
    assert coh.regression_report(_rows(0.60, 0.30)) is None  # improving
    # a missing rate is NOT a zero: no data, no bell
    rows = [{"week": "2026-S40", "takes": 2, "contour_rate": None},
            {"week": "2026-S41", "takes": 2, "contour_rate": 0.7}]
    assert coh.regression_report(rows) is None
    assert coh.regression_report(_rows(0.3)) is None  # no previous week


def test_a_lone_take_cannot_ring_the_bell_again_a_week():
    # one fresh take varies far more than a week's mean: it must not warn
    rows = _rows(0.30, 0.60)
    rows[-1]["takes"] = 1
    assert coh.regression_report(rows) is None
    # two takes this week back the comparison: it warns
    rows[-1]["takes"] = 2
    assert coh.regression_report(rows) is not None


def test_the_gui_announces_a_measured_regression(qapp, monkeypatch, tmp_path):
    from v2m.gui import main_window as mw

    monkeypatch.setattr(coh, "LIVE_TAKES_DIR", tmp_path)
    _seed_weeks(tmp_path, {"a": [0.3, 0.3], "b": [0.6, 0.6]})
    win = mw.MainWindow()
    try:
        win._announce_regression()
        assert not win.error_banner.isHidden(), "the regression must be announced"
        assert "30" in win.error_banner.text()  # the measured points
        # healthy data: silent
        _seed_weeks(tmp_path, {"a": [0.3, 0.3], "b": [0.3, 0.3]})
        win.error_banner.clear()
        win._announce_regression()
        assert win.error_banner.isHidden()
    finally:
        win.close()


def test_two_consecutive_snapshots_agree_and_the_second_sees_the_worse_week(tmp_path):
    """The weekly pipeline read TWICE through `build_snapshot`.

    Building is a READ over one pass of evidence: a second build over the
    same folder must produce the same aggregates (no re-baselining, no
    ordering jitter), and a snapshot already built stays the value it was
    when a worse next week arrives — that next build is the one that
    reports the measured regression.
    """
    # week 1 (2026-S39): two takes that moved few notes
    _take(tmp_path, "live_20260921_090000", checked=10, rate=0.30)
    _take(tmp_path, "live_20260922_090000", checked=10, rate=0.30)
    first = coh.build_snapshot(tmp_path)
    assert [w["week"] for w in first.health_weeks] == ["2026-S39"]
    assert first.regression is None  # one week has nothing to compare to

    # the next week (2026-S40) moved twice as many notes
    _take(tmp_path, "live_20260928_090000", checked=10, rate=0.60)
    _take(tmp_path, "live_20260929_090000", checked=10, rate=0.60)
    second = coh.build_snapshot(tmp_path)
    assert [w["week"] for w in second.health_weeks] == ["2026-S39", "2026-S40"]
    assert second.regression is not None
    assert second.regression["delta_points"] == pytest.approx(30.0)

    # reading the same evidence again agrees with the reading before it
    again = coh.build_snapshot(tmp_path)
    assert again.health_weeks == second.health_weeks
    assert again.regression == second.regression
    assert again.journal == second.journal

    # the first snapshot is a VALUE: the worse week never bleeds back into it
    assert len(first.health_weeks) == 1
    assert first.regression is None


# --------------------------------------------------------------------------- #
# 6. The sidecar link and corrupted CSVs
# --------------------------------------------------------------------------- #


def test_the_replay_sidecar_fills_the_trend_when_the_report_has_no_block(tmp_path):
    """An old take (JSON without the block) is re-read via its replay sidecar."""
    stem = "live_20260921_090000"
    d = tmp_path / stem
    d.mkdir()
    (d / f"{stem}.json").write_text(
        json.dumps({"coherence": {"checked": 10, "suspects": []}}),
        encoding="utf-8",
    )
    (d / f"{stem}.coherence.csv").write_text("h\n", encoding="utf-8")
    # no sidecar yet: no contour data
    assert coh.takes_under(tmp_path)[0].contour_rate is None
    # the replay archives its audit: the trend reads it (same vocabulary)
    (d / f"{stem}.live_vs_final.json").write_text(
        json.dumps({"live": 12, "final": 20, "contours": 5, "off1": 4,
                    "moved": 1, "absorbed": 0, "rate": 0.417}),
        encoding="utf-8",
    )
    audit = coh.takes_under(tmp_path)[0]
    assert audit.contour_rate == pytest.approx(0.417)
    assert audit.live_notes == 12


def test_a_corrupted_coherence_csv_never_breaks_the_aggregation(tmp_path):
    """A truncated CSV is a lost audit, not a broken table."""
    stem = "live_20260921_090000"
    d = tmp_path / stem
    d.mkdir()
    (d / f"{stem}.json").write_text(
        json.dumps({"coherence": {"checked": 10, "suspects": []}}),
        encoding="utf-8",
    )
    (d / f"{stem}.coherence.csv").write_text(
        "start_s,duration_s,midi,inner\n1.0,0.6,57", encoding="utf-8"
    )
    audit = coh.takes_under(tmp_path)[0]
    assert audit.suspects == []  # the broken rows are dropped, not raised
    assert audit.checked == 10  # the JSON's count still reads
