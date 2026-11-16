"""The weekly recap, the per-moment-of-day view, the retention, and the
`--changed` replay.

Every decision has its measured reason:

1. **practice windows**: a window's rates are the mean of the takes
   THAT HAVE the data (a take without a contour does not dilute its
   window), the evening WRAPS midnight (18h-6h), a take without a clock
   belongs nowhere — never guessed;
2. **retention**: the folder is trimmed by WHOLE take folders (oldest
   first, only the recognized name grammar), the journal keeps the
   history (its reason for being), a cache orphaned by a hand-deleted
   take is removed; `max_mb < 0` disables;
3. **--changed**: the engine hash lives in the sidecar — a replay with
   the SAME engine is skipped, a sidecar without a hash (or from
   another engine) is replayed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core import coherence as coh
from v2m.core import replay
from v2m.core import retention as ret
from v2m.core import weekly as wk
from v2m.core.i18n import get_language, set_language, tr
from v2m.core.replay import engine_fingerprint


@pytest.fixture(autouse=True)
def _restore_language():
    """Language-specific recap tests must not affect later tests."""
    original = get_language()
    yield
    set_language(original)


# --------------------------------------------------------------------------- #
# 1. Practice windows
# --------------------------------------------------------------------------- #


def _audit(stem: str, clock: str, checked=10, suspects=0, contour=None):
    t = coh.TakeAudit(
        stem=stem,
        date=stem.replace("live_", "").split("_")[0],
        clock=clock,
        epoch_s=coh._epoch(stem.replace("live_", "").split("_")[0], clock),
        suspects=[],
        checked=checked,
        verify_ok=True,
        contour_rate=contour,
        live_notes=None,
    )
    return t


def test_windows_group_by_clock_and_evening_wraps_midnight():
    takes = [
        _audit("live_20260927_080000", "080000", contour=0.5),  # morning
        _audit("live_20260927_140000", "140000", contour=0.3),  # afternoon
        _audit("live_20260927_210000", "210000", contour=0.7),  # evening
        _audit("live_20260927_003000", "003000", contour=0.9),  # 00:30 -> evening
    ]
    rows = coh.practice_time_report(takes)
    by_name = {r["window"]: r for r in rows}
    assert set(by_name) == {"morning", "afternoon", "evening"}
    assert by_name["morning"]["takes"] == 1
    assert by_name["evening"]["takes"] == 2  # 21:00 and 00:30
    assert by_name["evening"]["contour_rate"] == pytest.approx(0.8)


def test_a_take_without_data_never_dilutes_its_window():
    takes = [
        _audit("live_20260927_080000", "080000", checked=None, contour=0.5),
        _audit("live_20260927_090000", "090000", checked=None, contour=None),
    ]
    rows = coh.practice_time_report(takes)
    assert rows == [
        {"window": "morning", "takes": 2, "suspect_rate": None, "contour_rate": 0.5},
    ]


def test_an_unparseable_clock_belongs_to_no_window():
    rows = coh.practice_time_report([_audit("live_20260927_080000", "", contour=0.5)])
    assert rows == []


# --------------------------------------------------------------------------- #
# 2. Retention
# --------------------------------------------------------------------------- #


def _take_dir(tmp_path: Path, stem: str, with_cache: bool = False, size_kb: int = 1):
    d = tmp_path / stem
    d.mkdir()
    (d / f"{stem}.json").write_text("x" * size_kb * 1024, encoding="utf-8")
    (d / f"{stem}.coherence.csv").write_text("h\n", encoding="utf-8")
    if with_cache:
        (d / f"{stem}.observation.npz").write_bytes(b"0" * 512)


def test_orphan_observation_caches_are_removed(tmp_path):
    import numpy as np
    import soundfile as sf

    # the LIVE cache: its take folder carries the WAV the cache validates
    d = tmp_path / "live_20260927_090000"
    d.mkdir()
    sf.write(str(d / "live_20260927_090000.wav"), np.zeros(1000, dtype=np.float32), 8000)
    (d / "live_20260927_090000.observation.npz").write_bytes(b"0" * 512)
    # the ORPHAN: no WAV sibling, the cache can never be matched again
    (tmp_path / "live_20260920_090000.observation.npz").write_bytes(b"0" * 512)
    removed = ret.prune_observations(tmp_path)
    assert removed == 1
    assert not (tmp_path / "live_20260920_090000.observation.npz").exists()
    assert (d / "live_20260927_090000.observation.npz").exists()


def test_retention_trims_oldest_takes_whole_and_keeps_the_journal(tmp_path):
    journal = {"live_20260920_090000": {"date": "20260920", "suspects": 0}}
    (tmp_path / coh.JOURNAL_NAME).write_text(json.dumps(journal), encoding="utf-8")
    # ~1.1 MB: two 600 KB takes, one 4 KB — the budget is 1 MB
    _take_dir(tmp_path, "live_20260920_090000", size_kb=600)
    _take_dir(tmp_path, "live_20260921_090000", size_kb=600, with_cache=True)
    _take_dir(tmp_path, "live_20260927_090000", size_kb=4)
    pruned = ret.prune_takes(tmp_path, max_mb=1)
    # the OLDEST was trimmed whole; the journal and the recent take survive
    assert pruned == ["live_20260920_090000"]
    assert not (tmp_path / "live_20260920_090000").exists()
    assert (tmp_path / "live_20260921_090000").exists()
    assert (tmp_path / coh.JOURNAL_NAME).exists()
    # history survives in the journal: the trend still reads the pruned take
    assert (
        json.loads((tmp_path / coh.JOURNAL_NAME).read_text(encoding="utf-8"))[
            "live_20260920_090000"
        ]["date"]
        == "20260920"
    )


def test_retention_disabled_with_negative_budget(tmp_path):
    _take_dir(tmp_path, "live_20260920_090000", size_kb=600)
    assert ret.prune_takes(tmp_path, max_mb=-1) == []
    assert (tmp_path / "live_20260920_090000").exists()


def test_v2m_weekly_runs_both_retention_sweeps(tmp_path, monkeypatch, capsys):
    """The recap folder is the moment to trim: `v2m weekly` runs the
    orphan-cache sweep AND the take-folder budget, and announces the
    stems it deleted."""
    from argparse import Namespace

    from v2m.cli import _cmd_weekly

    calls: dict = {}

    def fake_observations(root):
        calls["obs"] = root
        return 0

    def fake_takes(root):
        calls["takes"] = root
        return ["live_old"]

    monkeypatch.setattr(ret, "prune_observations", fake_observations)
    monkeypatch.setattr(ret, "prune_takes", fake_takes)
    assert _cmd_weekly(Namespace(root=str(tmp_path), pdf=False)) == 0
    assert calls["obs"] == tmp_path and calls["takes"] == tmp_path
    assert "live_old" in capsys.readouterr().out  # the announcement names it


# --------------------------------------------------------------------------- #
# 3. The weekly recap
# --------------------------------------------------------------------------- #


def test_build_weekly_aggregates_the_existing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(coh, "LIVE_TAKES_DIR", tmp_path)
    _take_dir(tmp_path, "live_20260921_090000")
    (tmp_path / "live_20260921_090000" / "live_20260921_090000.json").write_text(
        json.dumps(
            {
                "coherence": {"checked": 10, "suspects": []},
                "live_vs_final": {"rate": 0.5, "live": 10},
            }
        ),
        encoding="utf-8",
    )
    data = wk.build_weekly(tmp_path)
    assert data["takes"] == 1
    assert data["health"] and data["health"][0]["contour_rate"] == pytest.approx(0.5)
    assert data["windows"] and data["windows"][0]["window"] == "morning"
    assert data["days"] and data["days"][0]["date"] == "20260921"


def test_weekly_text_renders_every_section(tmp_path):
    set_language("en")
    data = {
        "takes": 2,
        "health": [
            {
                "week": "2026-S39",
                "takes": 2,
                "contour_rate": 0.4,
                "live_notes": 20,
                "final_notes": 40,
                "stems": [],
            }
        ],
        "windows": [{"window": "morning", "takes": 2, "suspect_rate": 0.0, "contour_rate": 0.4}],
        "days": [
            {
                "date": "20260921",
                "sessions": 1,
                "takes": 2,
                "checked": 20,
                "suspects": 0,
                "rate_per_100": 0.0,
            }
        ],
        "regression": None,
    }
    text = wk.weekly_text(data)
    assert "Weekly recap" in text
    assert "2026-S39" in text
    assert "morning" in text
    assert "20260921" in text


def test_weekly_text_empty_when_no_data():
    set_language("en")
    assert "No data yet" in wk.weekly_text(
        {"takes": 0, "health": [], "windows": [], "days": [], "regression": None}
    )


def test_the_recap_musicxml_carries_the_text_as_lyrics(tmp_path):
    from v2m.core.weekly_pdf import build_recap_musicxml

    data = {
        "takes": 1,
        "health": [
            {
                "week": "2026-S39",
                "takes": 1,
                "contour_rate": 0.5,
                "live_notes": 10,
                "final_notes": 20,
                "stems": [],
            }
        ],
        "windows": [],
        "days": [],
        "regression": None,
    }
    xml = build_recap_musicxml(data)
    assert "<lyric" in xml and "2026-S39" in xml
    assert 'measure="yes"' in xml  # the carrier is a whole-measure rest


def test_engrave_weekly_without_a_tool_raises_the_missing_kind(tmp_path, monkeypatch):
    from v2m.core import pdf_export
    from v2m.core.weekly_pdf import WeeklyPdfToolMissing, engrave_weekly_pdf

    monkeypatch.setattr(pdf_export, "find_notation_tool", lambda: None)
    data = {"takes": 0, "health": [], "windows": [], "days": [], "regression": None}
    with pytest.raises(WeeklyPdfToolMissing):
        engrave_weekly_pdf(data, tmp_path / "recap")
    # the intermediate never survived the failure
    assert not (tmp_path / "recap.weekly.musicxml").exists()


def test_engrave_weekly_runs_the_same_tool_as_the_score(tmp_path, monkeypatch):
    from v2m.core import pdf_export
    from v2m.core.weekly_pdf import engrave_weekly_pdf

    called = {}

    def fake_export(xml_path, pdf_path):
        called["xml"] = Path(xml_path)
        Path(pdf_path).write_bytes(b"%PDF-1.4 fake")
        return str(pdf_path)

    monkeypatch.setattr(pdf_export, "export_pdf", fake_export)
    data = {"takes": 0, "health": [], "windows": [], "days": [], "regression": None}
    path = engrave_weekly_pdf(data, tmp_path / "recap")
    assert path.endswith("recap.pdf")
    assert Path(path).exists()
    # the intermediate MusicXML was cleaned up, the PDF is the product
    assert not called["xml"].exists()


# --------------------------------------------------------------------------- #
# 4. --changed: the engine hash in the sidecar
# --------------------------------------------------------------------------- #


def test_engine_fingerprint_tracks_source_content_and_requires_every_source(tmp_path, monkeypatch):
    """A readable source change changes the fingerprint; no source is a valid hash input."""
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    for name in replay._ENGINE_SOURCE_NAMES:
        (core_dir / name).write_text(f"source: {name}\n", encoding="utf-8")
    monkeypatch.setattr(replay, "__file__", str(core_dir / "replay.py"))

    first = engine_fingerprint()
    assert first == engine_fingerprint()
    (core_dir / "liveseg.py").write_text("changed engine source\n", encoding="utf-8")
    assert engine_fingerprint() != first

    (core_dir / "notes.py").unlink()
    with pytest.raises(FileNotFoundError, match="notes.py"):
        engine_fingerprint()


def test_an_old_journal_final_count_without_a_source_is_unknown(tmp_path):
    """Pre-fix rate-derived counts are not presented as exact history."""
    coh.read_journal(tmp_path)
    stem = "live_20260921_090000"
    (tmp_path / coh.JOURNAL_NAME).write_text(
        json.dumps(
            {
                stem: {
                    "date": "20260921",
                    "contour_rate": 0.6,
                    "live_notes": 20,
                    "final_notes": 33,
                }
            }
        ),
        encoding="utf-8",
    )

    trend = coh.contour_trend(tmp_path)
    assert trend[0]["live_notes"] == 20
    assert trend[0]["final_notes"] is None
    text = wk.weekly_text(
        {
            "takes": 1,
            "health": trend,
            "windows": [],
            "days": [],
            "regression": None,
        }
    )
    expected_row = tr("coh week row").format(
        week=trend[0]["week"],
        takes=trend[0]["takes"],
        rate=f"{trend[0]['contour_rate'] * 100:.0f}%",
        live=20,
        final="-",
    )
    assert expected_row in text
    assert "None" not in text


def test_changed_keeps_only_takes_an_older_engine_measured(tmp_path, capsys, monkeypatch):
    import importlib.util
    import sys

    ROOT = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "replay_takes_mod", ROOT / "scripts" / "replay_takes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("replay_takes_mod", mod)
    spec.loader.exec_module(mod)

    for stem, engine in [
        ("live_20260926_090000", engine_fingerprint()),  # current engine
        ("live_20260926_091500", "older-engine-hash"),  # older engine
    ]:
        d = tmp_path / stem
        d.mkdir()
        (d / f"{stem}.wav").write_bytes(b"RIFF....")
        (d / f"{stem}.live_vs_final.json").write_text(
            json.dumps(
                {
                    "live": 3,
                    "final": 3,
                    "contours": 0,
                    "off1": 0,
                    "moved": 0,
                    "absorbed": 0,
                    "rate": 0.0,
                    "engine": engine,
                }
            ),
            encoding="utf-8",
        )
    replays: list[str] = []

    def fake_replay(wav, **kw):
        replays.append(Path(wav).stem)
        return {
            "stem": Path(wav).stem,
            "duration_s": 1.0,
            "live_notes": 1,
            "final_notes": 1,
            "contours": 0,
            "off1": 0,
            "moved": 0,
            "absorbed": 0,
            "rate": 0.0,
            "tempo_bpm": 120.0,
        }

    monkeypatch.setattr(mod, "replay_take", fake_replay)
    assert mod.main([str(tmp_path), "--changed"]) == 0
    assert replays == ["live_20260926_091500"], (
        "the --changed replay must only take audits from an older engine"
    )


def test_a_take_without_sidecar_is_always_due_for_replay(tmp_path, monkeypatch):
    import importlib.util
    import sys

    ROOT = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "replay_takes_mod2", ROOT / "scripts" / "replay_takes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("replay_takes_mod2", mod)
    spec.loader.exec_module(mod)

    d = tmp_path / "live_20260926_090000"
    d.mkdir()
    (d / "live_20260926_090000.wav").write_bytes(b"RIFF....")
    monkeypatch.setattr(
        mod,
        "replay_take",
        lambda wav, **kw: {
            "stem": Path(wav).stem,
            "duration_s": 1.0,
            "live_notes": 0,
            "final_notes": 0,
            "contours": 0,
            "off1": 0,
            "moved": 0,
            "absorbed": 0,
            "rate": 0.0,
            "tempo_bpm": None,
        },
    )
    assert mod.main([str(tmp_path), "--changed"]) == 0  # no crash, replayed
