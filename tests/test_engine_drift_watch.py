"""The drift watch WITHOUT real replays: the policy is in the deltas.

`core.replay` owns the replay (its tests pin it); these pin the WATCH:
which sidecars are stale (the engine hash, and ONLY replayed audits —
the export-time block has no engine and compares a LIVE reading against
a REPLAY, a measured always-differs family), how the deltas read, when
the verdict says DÉRIVE (drift), and that the report is written either way.
`replay_take` is monkeypatched: no real-time cost here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import engine_drift_watch as watch


def _take(root: Path, stem: str, sidecar: dict | None) -> Path:
    d = root / stem
    d.mkdir(parents=True, exist_ok=True)
    if sidecar is not None:
        (d / f"{stem}.live_vs_final.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    return d


def _sidecar(engine: str | None = "old-engine") -> dict:
    row = {"live": 20, "final": 19, "contours": 12, "rate": 0.6}
    if engine is not None:
        row["engine"] = engine
    return row


def test_only_hashed_sidecars_are_stale(tmp_path):
    _take(tmp_path, "live_a", _sidecar("old-engine"))
    _take(tmp_path, "live_b", _sidecar("current"))  # THIS engine measured it
    _take(tmp_path, "live_c", _sidecar(None))       # export-time block
    _take(tmp_path, "live_d", None)                 # no sidecar at all
    stale = watch._stale_takes(tmp_path, "current")
    assert [s for s, _, _ in stale] == ["live_a"], (
        "un audit sans rejeu n'est pas 'périmé': il n'y a rien à défendre"
    )


def test_deltas_and_drift_verdict():
    old = _sidecar()
    row = {"live_notes": 21, "final_notes": 19, "contours": 11, "rate": 0.55}
    d = watch._deltas(old, row)
    assert d["live_notes"] == 1 and d["final_notes"] == 0
    assert d["contours"] == -1
    assert abs(d["rate"] + 0.05) < 1e-9  # 0.55 - 0.60 (float path)
    # any note-count movement is a drift; a sub-tolerance rate wobble alone
    # is the engine's session noise
    assert watch._drifted(d) is True
    quiet = watch._deltas(old, {"live_notes": 20, "final_notes": 19,
                                "contours": 12, "rate": 0.62})
    assert watch._drifted(quiet) is False


def test_main_replays_stale_and_writes_the_report(tmp_path, monkeypatch):
    for stem in ("live_a", "live_b"):
        _take(tmp_path, stem, _sidecar("old-engine"))
        (tmp_path / stem / f"{stem}.wav").write_bytes(b"RIFF")  # the gate

    def fake_replay(wav):
        # live_b moved one contour: the report must SAY so, exit 1
        stem = Path(wav).stem
        if stem == "live_b":
            return {"stem": stem, "live_notes": 21, "final_notes": 19,
                    "contours": 13, "off1": 0, "moved": 0, "absorbed": 0,
                    "rate": 0.62, "tempo_bpm": 120.0}
        return {"stem": stem, "live_notes": 20, "final_notes": 19,
                "contours": 12, "off1": 0, "moved": 0, "absorbed": 0,
                "rate": 0.6, "tempo_bpm": 120.0}

    monkeypatch.setattr(watch, "replay_take", fake_replay)
    report = tmp_path / "drift.md"
    rc = watch.main(["--root", str(tmp_path), "--report", str(report)])
    assert rc == 1, "une lecture déplacée doit faire échouer le suivi"
    text = report.read_text(encoding="utf-8")
    assert "DÉRIVE" in text and "stable" in text
    assert "live_b" in text


def test_main_all_quiet_exits_zero(tmp_path, monkeypatch):
    _take(tmp_path, "live_a", _sidecar("old-engine"))
    (tmp_path / "live_a" / "live_a.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(
        watch, "replay_take",
        lambda wav: {"stem": Path(wav).stem, "live_notes": 20,
                     "final_notes": 19, "contours": 12, "off1": 0,
                     "moved": 0, "absorbed": 0, "rate": 0.6,
                     "tempo_bpm": 120.0},
    )
    report = tmp_path / "drift.md"
    assert watch.main(["--root", str(tmp_path), "--report", str(report)]) == 0


def test_a_take_without_wav_is_kept_as_is(tmp_path, monkeypatch, capsys):
    _take(tmp_path, "live_a", _sidecar("old-engine"))  # no WAV written
    called = []
    monkeypatch.setattr(watch, "replay_take", lambda wav: called.append(wav))
    report = tmp_path / "drift.md"
    rc = watch.main(["--root", str(tmp_path), "--report", str(report)])
    assert called == [] and rc == 0
    assert "live_a" not in report.read_text(encoding="utf-8"), (
        "sans audio il n'y a pas de re-dérivation, l'audit reste tel quel"
    )
