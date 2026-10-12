"""The aggregation core (`core.coherence`): sessions, rates, delta, verify.

The per-session follow-up lives in the CORE (the CLI script and the GUI
page are adapters); this test pins its decisions, each with its measured
reason:

1. **sessions by time gap**: a hole > 30 min opens a session, even on
   the same day; two takes 10 min apart stay together even across
   midnight;
2. **rate per 100 notes + delta**: the drop reads without comparing raw
   counts — 10 suspects over 200 notes beats 3 over 20;
3. **verify_take in the loop**: a take whose WAV and JSON describe two
   signals is flagged in the table and its notes are not counted in the
   rate;
4. **persisted journal**: tidying folders does not rewrite history —
   the delta against the previous session stays reliable;
5. **published plateau**: every suspect carries its time, pitch and the
   SIGNED interval to the label.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from v2m.core import coherence as coh
from v2m.core.i18n import set_language


def _take_dir(tmp_path: Path, stem: str) -> Path:
    d = tmp_path / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_csv(take_dir: Path, stem: str, suspects: list[tuple]) -> None:
    with open(take_dir / f"{stem}{coh.CSV_NAME}", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(coh.CSV_HEADER)
        for s in suspects:
            w.writerow(s)


def _write_json(take_dir: Path, stem: str, checked: int, duration: float = 5.0) -> None:
    (take_dir / f"{stem}.json").write_text(
        json.dumps({"coherence": {"checked": checked, "suspects": []},
                    "duration_s": duration, "n_notes": checked}),
        encoding="utf-8",
    )


def _write_wav(take_dir: Path, stem: str, seconds: float) -> None:
    import numpy as np
    import soundfile as sf

    sr = 8000
    y = (0.2 * np.sin(2 * np.pi * 220.0 * np.arange(int(seconds * sr)) / sr)).astype(np.float32)
    sf.write(str(take_dir / f"{stem}.wav"), y, sr)


# --------------------------------------------------------------------------- #
# 1. Sessions follow the CLOCK, not the date alone
# --------------------------------------------------------------------------- #


def test_a_30_min_gap_opens_a_session_on_the_same_day(tmp_path):
    _write_csv(_take_dir(tmp_path, "live_20260927_090000"), "live_20260927_090000", [])
    _write_json(_take_dir(tmp_path, "live_20260927_090000"), "live_20260927_090000", 5)
    _write_csv(_take_dir(tmp_path, "live_20260927_100000"), "live_20260927_100000", [])
    _write_json(_take_dir(tmp_path, "live_20260927_100000"), "live_20260927_100000", 5)

    rows = coh.session_rows(coh.takes_under(tmp_path))
    assert len(rows) == 2, "a one-hour gap: two sessions, same date"
    assert all(r["takes"] == 1 for r in rows)


def test_takes_across_midnight_stay_one_session(tmp_path):
    # 23:50 then 00:05: 15 minutes apart, one single practice
    _write_csv(_take_dir(tmp_path, "live_20260926_235000"), "live_20260926_235000", [])
    _write_json(_take_dir(tmp_path, "live_20260926_235000"), "live_20260926_235000", 4)
    _write_csv(_take_dir(tmp_path, "live_20260927_000500"), "live_20260927_000500", [])
    _write_json(_take_dir(tmp_path, "live_20260927_000500"), "live_20260927_000500", 4)

    rows = coh.session_rows(coh.takes_under(tmp_path))
    assert len(rows) == 1
    assert rows[0]["takes"] == 2


def test_day_subtotals_come_after_the_session_rows(tmp_path):
    _write_csv(_take_dir(tmp_path, "live_20260926_235000"), "live_20260926_235000", [])
    _write_json(_take_dir(tmp_path, "live_20260926_235000"), "live_20260926_235000", 4)
    _write_csv(_take_dir(tmp_path, "live_20260927_000500"), "live_20260927_000500", [])
    _write_json(_take_dir(tmp_path, "live_20260927_000500"), "live_20260927_000500", 4)
    _write_csv(_take_dir(tmp_path, "live_20260927_150000"), "live_20260927_150000", [])
    _write_json(_take_dir(tmp_path, "live_20260927_150000"), "live_20260927_150000", 6)

    takes = coh.takes_under(tmp_path)
    rows = coh.session_rows(takes)
    assert len(rows) == 2
    days = coh.day_rows(rows, takes)
    assert [d["date"] for d in days] == ["20260926", "20260927"]
    # the 23:50/00:05 session crosses midnight: ONE session, but its takes
    # count on their OWN days (the day follows the take's date)
    assert days[0]["takes"] == 1 and days[1]["takes"] == 2


# --------------------------------------------------------------------------- #
# 2. The rate per 100 notes and its delta
# --------------------------------------------------------------------------- #


def test_rate_per_100_and_delta_against_the_previous_session(tmp_path):
    # session 1: 1 suspect over 20 notes -> 5.0/100
    _write_csv(_take_dir(tmp_path, "live_20260926_090000"), "live_20260926_090000",
               [(1.0, 0.6, 57, 2.0, 0.20, 0.2, 55, 2.0)])
    _write_json(_take_dir(tmp_path, "live_20260926_090000"), "live_20260926_090000", 20)
    # session 2 (45 min later): 2 suspects over 200 -> 1.0/100
    _write_csv(_take_dir(tmp_path, "live_20260926_094500"), "live_20260926_094500",
               [(1.0, 0.6, 57, 2.0, 0.20, 0.2, 55, 2.0)])
    _write_json(_take_dir(tmp_path, "live_20260926_094500"), "live_20260926_094500", 100)
    _write_csv(_take_dir(tmp_path, "live_20260926_095000"), "live_20260926_095000",
               [(2.0, 0.5, 60, 1.5, 0.13, 0.1, 62, 2.0)])
    _write_json(_take_dir(tmp_path, "live_20260926_095000"), "live_20260926_095000", 100)

    rows = coh.session_rows(coh.takes_under(tmp_path))
    assert len(rows) == 2
    first, second = rows
    assert first["rate_per_100"] == 5.0
    assert second["rate_per_100"] == 1.0
    # the delta says the DROP (-4.0) with no manual line comparison
    assert first["delta_vs_prev"] is None
    assert second["delta_vs_prev"] == -4.0


def test_a_missing_json_keeps_the_rate_indisponible_not_zero(tmp_path):
    _write_csv(_take_dir(tmp_path, "live_20260926_090000"), "live_20260926_090000", [])
    # no JSON: "checked" unavailable -> rate unavailable
    rows = coh.session_rows(coh.takes_under(tmp_path))
    assert rows[0]["checked"] is None
    assert rows[0]["rate_per_100"] is None


# --------------------------------------------------------------------------- #
# 3. verify_take in the loop
# --------------------------------------------------------------------------- #


def test_a_diverging_take_is_flagged_and_excluded_from_the_rate(tmp_path):
    # the verified take: 1 suspect over 20 notes -> 5.0/100
    d1 = _take_dir(tmp_path, "live_20260926_100000")
    _write_csv(d1, "live_20260926_100000",
               [(1.0, 0.6, 57, 2.0, 0.20, 0.2, 55, 2.0)])
    _write_json(d1, "live_20260926_100000", 20)
    # the DIVERGING take: 3 suspects over 30 notes -> would drag to 8.0/100
    d2 = _take_dir(tmp_path, "live_20260926_110000")
    _write_csv(d2, "live_20260926_110000",
               [(1.0, 0.6, 57, 2.0, 0.20, 0.2, 55, 2.0),
                (2.0, 0.6, 57, 2.0, 0.20, 0.2, 55, 2.0),
                (3.0, 0.6, 57, 2.0, 0.20, 0.2, 55, 2.0)])
    _write_json(d2, "live_20260926_110000", 30, duration=5.0)
    _write_wav(d2, "live_20260926_110000", seconds=9.0)  # the OTHER take's audio

    takes = coh.takes_under(tmp_path)
    assert takes[1].verify_ok is False
    rows = coh.session_rows(takes)
    # TWO sessions (a 1-hour gap): the clean one carries the rate; the
    # diverging one is flagged and its counts stay OUT of every rate
    assert rows[0]["suspects"] == 1
    assert rows[0]["checked"] == 20
    assert rows[0]["rate_per_100"] == 5.0
    assert rows[0]["verify_bad"] == 0
    assert rows[1]["suspects"] == 0      # excluded, not counted
    assert rows[1]["checked"] is None    # nothing trustworthy to sum
    assert rows[1]["rate_per_100"] is None
    assert rows[1]["verify_bad"] == 1
    text = coh.render(takes)
    assert "DIVERG" in text or "diverg" in text


def test_the_session_row_carries_the_mean_contour_rate(tmp_path):
    """The per-session contour rate: the mean of the takes that have a
    live_vs_final block — a blockless take does not dilute the mean."""
    _write_csv(_take_dir(tmp_path, "live_20260926_090000"), "live_20260926_090000", [])
    (tmp_path / "live_20260926_090000" / "live_20260926_090000.json").write_text(
        json.dumps({"coherence": {"checked": 5, "suspects": []},
                    "live_vs_final": {"live": 20, "final": 18,
                                       "contours": 10, "rate": 0.5}}),
        encoding="utf-8",
    )
    _write_csv(_take_dir(tmp_path, "live_20260926_091000"), "live_20260926_091000", [])
    (tmp_path / "live_20260926_091000" / "live_20260926_091000.json").write_text(
        json.dumps({"coherence": {"checked": 5, "suspects": []},
                    "live_vs_final": {"live": 10, "final": 10,
                                       "contours": 2, "rate": 0.2}}),
        encoding="utf-8",
    )
    _write_csv(_take_dir(tmp_path, "live_20260926_092000"), "live_20260926_092000", [])
    (tmp_path / "live_20260926_092000" / "live_20260926_092000.json").write_text(
        json.dumps({"coherence": {"checked": 5, "suspects": []}}),
        encoding="utf-8",
    )
    rows = coh.session_rows(coh.takes_under(tmp_path))
    # (0.5 + 0.2) / 2 = 0.35 — the blockless take does not dilute
    assert rows[0]["contour_rate"] == 0.35
    text = coh.render(coh.takes_under(tmp_path))
    assert "35%" in text


def test_a_matching_take_passes_verification(tmp_path):
    d = _take_dir(tmp_path, "live_20260926_120000")
    _write_csv(d, "live_20260926_120000", [])
    _write_json(d, "live_20260926_120000", 6, duration=3.0)
    _write_wav(d, "live_20260926_120000", seconds=3.0)
    takes = coh.takes_under(tmp_path)
    assert takes[0].verify_ok is True
    assert takes[0].verify_lines == []


# --------------------------------------------------------------------------- #
# 4. The journal survives the tidy-up
# --------------------------------------------------------------------------- #


def test_the_journal_keeps_a_deleted_take(tmp_path):
    d = _take_dir(tmp_path, "live_20260926_130000")
    _write_csv(d, "live_20260926_130000", [])
    _write_json(d, "live_20260926_130000", 7)
    takes = coh.takes_under(tmp_path)
    data = coh.update_journal(tmp_path, takes)
    assert "live_20260926_130000" in data

    # the folder is cleaned: the take's artifacts are gone...
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    # ...but the journal keeps the entry: the trend is not re-baselined
    data2 = coh.update_journal(tmp_path, [])
    assert "live_20260926_130000" in data2
    assert data2["live_20260926_130000"]["checked"] == 7


def test_a_re_exported_take_replaces_its_journal_entry(tmp_path):
    d = _take_dir(tmp_path, "live_20260926_140000")
    _write_csv(d, "live_20260926_140000",
               [(1.0, 0.6, 57, 2.0, 0.20, 0.2, 55, 2.0)])
    _write_json(d, "live_20260926_140000", 10)
    data = coh.update_journal(tmp_path, coh.takes_under(tmp_path))
    assert data["live_20260926_140000"]["suspects"] == 1

    _write_csv(d, "live_20260926_140000", [])  # re-export: clean now
    data2 = coh.update_journal(tmp_path, coh.takes_under(tmp_path))
    assert data2["live_20260926_140000"]["suspects"] == 0


# --------------------------------------------------------------------------- #
# 5. The published plateau: time, pitch, signed interval
# --------------------------------------------------------------------------- #


def test_a_suspect_carries_time_pitch_and_signed_offset(tmp_path):
    d = _take_dir(tmp_path, "live_20260926_150000")
    _write_csv(d, "live_20260926_150000",
               [(4.2, 0.8, 60, 2.5, 0.30, 4.5, 62, 2.0)])
    _write_json(d, "live_20260926_150000", 8)
    takes = coh.takes_under(tmp_path)
    s = takes[0].suspects[0]
    assert s["secondary_plateau_start_s"] == 4.5   # WHERE inside the note
    assert s["secondary_plateau_midi"] == 62       # at WHICH pitch
    assert s["secondary_plateau_offset_st"] == 2.0 # SIGNED interval to the label
    text = coh.render(takes)
    assert "4.50" in text and "+2.00st" in text and "62" in text


def test_render_follows_the_language_tables():
    set_language("fr")
    fr = coh.render([])
    assert "Aucun rapport" in fr or "rien à agréger" in fr
    set_language("en")
    en = coh.render([])
    assert "No coherence report" in en
    set_language("fr")


def test_only_real_take_names_are_aggregated(tmp_path):
    _write_csv(_take_dir(tmp_path, "chant"), "chant", [])
    _write_csv(_take_dir(tmp_path, "live_bad"), "notes", [])
    assert coh.takes_under(tmp_path) == []


def test_the_replayed_sidecar_wins_the_trend_reading(tmp_path):
    """A trend point reads ONE source, never a mix: the REPLAYED sidecar
    (fresh, engine-tagged, replaced by the cross-check) beats the
    export-time block in the report; the report fills the gap when no
    sidecar exists, and no reading at all leaves the trend waiting."""
    from v2m.core.compare import write_sidecar

    stem = "live_20260101_120000"
    d = _take_dir(tmp_path, stem)
    _write_csv(d, stem, [])
    # the export-time reading, in the take's own report
    (d / f"{stem}.json").write_text(
        json.dumps({
            "coherence": {"checked": 4, "suspects": []},
            "live_vs_final": {"live": 37, "final": 31, "contours": 37,
                              "off1": 20, "moved": 17, "absorbed": 0,
                              "rate": 1.0},
        }),
        encoding="utf-8",
    )
    # the replayed cross-check, archived beside the artifacts
    sidecar = write_sidecar(d / stem, {"live": 17, "final": 31, "contours": 8,
                                       "off1": 5, "moved": 3, "absorbed": 14,
                                       "rate": 0.48})
    assert sidecar is not None  # the cross-check must be archived here
    audit = coh.load_take(d / f"{stem}{coh.CSV_NAME}")
    assert audit.contour_rate == 0.48   # the sidecar wins
    assert audit.live_notes == 17       # both fields from the SAME source
    assert audit.checked == 4           # the report's own fact is untouched

    # no sidecar: the export-time block fills the gap
    Path(sidecar).unlink()
    audit = coh.load_take(d / f"{stem}{coh.CSV_NAME}")
    assert audit.contour_rate == 1.0
    assert audit.live_notes == 37

    # neither: the trend waits — it never guesses a number
    (d / f"{stem}.json").write_text(
        json.dumps({"coherence": {"checked": 4, "suspects": []}}),
        encoding="utf-8",
    )
    audit = coh.load_take(d / f"{stem}{coh.CSV_NAME}")
    assert audit.contour_rate is None and audit.live_notes is None
