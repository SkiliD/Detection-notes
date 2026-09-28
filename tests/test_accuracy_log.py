"""accuracy.record_take: the accuracy history persists from take to take."""

from __future__ import annotations

import json
from pathlib import Path

from v2m.core.accuracy import load_log, record_take


class _Hist:
    """Duck-typed stand-in for gui AccuracyHistory."""

    def __init__(self, means):
        self.means = means


def test_record_take_appends_and_rounds(tmp_path: Path):
    p = tmp_path / "acc.json"
    h = _Hist({60: 12.34, 64: 30.0})
    entry = record_take(h, duration_s=4.321, bpm=171.4, path=p)
    assert entry is not None
    assert entry["n_notes"] == 2
    assert entry["mean_cents"] == 21.2
    assert entry["duration_s"] == 4.32
    assert entry["bpm"] == 171.4
    assert entry["notes"][0] == {
        "midi": 60, "name": "C4", "name_fr": "Do4", "cents": 12.3,
    }
    # the second take lands AFTER the first (progress across sessions)
    entry2 = record_take(_Hist({62: 5.0}), duration_s=1.0, path=p)
    log = load_log(p)
    assert len(log) == 2
    assert log[1]["mean_cents"] == 5.0


def test_record_take_skips_empty_and_survives_bad_file(tmp_path: Path):
    p = tmp_path / "acc.json"
    assert record_take(_Hist({}), path=p) is None  # nothing sung -> nothing kept
    assert load_log(p) == []                       # absent file -> []
    p.write_text("not json", encoding="utf-8")
    assert load_log(p) == []                       # corrupt file -> []
    # a corrupt file is REPLACED, not fatal: the next take records fine
    entry = record_take(_Hist({60: 9.0}), path=p)
    assert entry is not None
    assert load_log(p)[0]["mean_cents"] == 9.0


def test_record_take_json_shape_is_stable(tmp_path: Path):
    p = tmp_path / "acc.json"
    record_take(_Hist({69: 8.0, 72: 14.0}), duration_s=2.0, bpm=120.0, path=p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 1
    e = data[0]
    assert set(e) == {"date", "duration_s", "bpm", "n_notes", "mean_cents", "notes"}
    assert set(e["notes"][0]) == {"midi", "name", "name_fr", "cents"}
