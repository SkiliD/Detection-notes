"""The CLI adapter (`scripts/replay_takes.py`): the engine owns the contract.

The replay ENGINE lives in `core.replay` (pinned by `test_replay.py`);
this module tests only the adapter: exit codes, take filtering (a stray
artifact is not a take), the `--out` report and the printed table. The
replay is MOCKED: no test here pays the real clock, that is the engine
tests' job.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPEC = importlib.util.spec_from_file_location(
    "replay_takes", ROOT / "scripts" / "replay_takes.py"
)
replay_takes = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("replay_takes", replay_takes)
SPEC.loader.exec_module(replay_takes)


def _take_with_wav(tmp_path: Path, stem: str) -> None:
    d = tmp_path / stem
    d.mkdir()
    (d / f"{stem}.wav").write_bytes(b"RIFF....")  # the replay is mocked


def test_a_missing_folder_explains_and_exits_2(capsys):
    assert replay_takes.main(["n_existe_pas"]) == 2
    out = capsys.readouterr().out
    assert "n_existe_pas" in out


def test_a_folder_without_wav_is_not_an_error(capsys, tmp_path):
    assert replay_takes.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "WAV" in out  # the hint names what is missing (the audio box)


def test_the_table_and_the_out_report_classify_each_take(capsys, tmp_path, monkeypatch):
    _take_with_wav(tmp_path, "live_20260927_090000")
    _take_with_wav(tmp_path, "live_20260927_091500")
    # an artifact lost at the root of a wrongly-named folder: NOT a take
    stray = tmp_path / "autre"
    stray.mkdir()
    (stray / "egare.wav").write_bytes(b"RIFF....")

    def fake_replay(wav_path, **kw):
        return {
            "stem": Path(wav_path).stem, "duration_s": 1.9, "live_notes": 5,
            "final_notes": 4, "contours": 1, "off1": 1, "moved": 0,
            "absorbed": 0, "rate": 0.2, "tempo_bpm": 120.0,
        }

    monkeypatch.setattr(replay_takes, "replay_take", fake_replay)
    out_path = tmp_path / "rapport.json"
    assert replay_takes.main([str(tmp_path), "--out", str(out_path)]) == 0
    out = capsys.readouterr().out
    # both takes in the table, the stray never replayed
    assert "live_20260927_090000" in out and "live_20260927_091500" in out
    assert "egare" not in out
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["root"] == str(tmp_path)
    assert [r["stem"] for r in report["takes"]] == [
        "live_20260927_090000", "live_20260927_091500",
    ]
    assert report["takes"][0]["rate"] == 0.2


def test_a_failing_take_never_costs_the_others(capsys, tmp_path, monkeypatch):
    _take_with_wav(tmp_path, "live_20260927_090000")
    _take_with_wav(tmp_path, "live_20260927_091500")

    def flaky(wav_path, **kw):
        if Path(wav_path).stem.endswith("090000"):
            raise RuntimeError("prise cassée")
        return {
            "stem": Path(wav_path).stem, "duration_s": 1.0, "live_notes": 1,
            "final_notes": 1, "contours": 0, "off1": 0, "moved": 0,
            "absorbed": 0, "rate": 0.0, "tempo_bpm": 120.0,
        }

    monkeypatch.setattr(replay_takes, "replay_take", flaky)
    assert replay_takes.main([str(tmp_path)]) == 0  # one failure is not fatal
    out = capsys.readouterr().out
    assert "ERROR" in out and "prise cassée" in out
    assert "live_20260927_091500" in out  # the healthy take is still tabled
