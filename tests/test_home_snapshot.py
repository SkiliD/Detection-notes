"""The shared coherence snapshot is built once per consumer."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v2m.core import coherence as coh


def _empty_snapshot(root: Path) -> coh.EvidenceSnapshot:
    return coh.EvidenceSnapshot(
        root=root,
        takes=[],
        sessions=[],
        days=[],
        practice_windows=[],
        health_weeks=[],
        regression=None,
        journal={},
    )


def test_snapshot_builds_take_audits_and_journal_once(tmp_path):
    with (
        patch.object(coh, "takes_under", return_value=[]) as takes_under,
        patch.object(coh, "update_journal", return_value={}) as update_journal,
    ):
        snapshot = coh.build_snapshot(tmp_path)

    takes_under.assert_called_once_with(tmp_path)
    update_journal.assert_called_once_with(tmp_path, [])
    assert snapshot.takes == []
    assert snapshot.health_weeks == []


def test_cli_and_weekly_reuse_one_snapshot(tmp_path, monkeypatch):
    import scripts.coherence_summary as summary
    from v2m.core import weekly

    snapshot = _empty_snapshot(tmp_path)
    stream = SimpleNamespace(buffer=io.BytesIO(), flush=lambda: None)
    monkeypatch.setattr(summary, "build_snapshot", lambda _root: snapshot)
    monkeypatch.setattr(summary, "sys", SimpleNamespace(stdout=stream))
    assert summary.main([str(tmp_path)]) == 0

    monkeypatch.setattr(coh, "build_snapshot", lambda _root: snapshot)
    data = weekly.build_weekly(tmp_path)
    assert data["takes"] == 0
    assert data["health"] == []
