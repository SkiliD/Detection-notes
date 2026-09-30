"""The CLI adapter (`scripts/compare_takes.py`): the engine owns the contract.

Like `test_replay_script.py`: the ENGINE (`core.phrase`) is pinned by
`test_phrase.py`; this module tests only the adapter — the stem
resolution inside `prises_live/`, the printed table (its i18n goes
through `tr()`, both languages), the `--json` archive and the exit
codes. The comparison is mocked: no test here pays the engine.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPEC = importlib.util.spec_from_file_location(
    "compare_takes", ROOT / "scripts" / "compare_takes.py"
)
assert SPEC is not None and SPEC.loader is not None  # narrows for the reader
compare_takes = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("compare_takes", compare_takes)
SPEC.loader.exec_module(compare_takes)

from v2m.core.i18n import set_language  # noqa: E402


def _report(midis, starts, bpm=120.0):
    return {
        "tempo": {"bpm": bpm},
        "notes": [
            {"midi": m, "start_s": s, "end_s": s + 0.4}
            for m, s in zip(midis, starts)
        ],
    }


def _takes_in(tmp_path, monkeypatch, reports: dict[str, dict]) -> None:
    """Two takes laid out as the artifact contract holds them."""
    import v2m.core.takes as takes_mod

    monkeypatch.setattr(takes_mod, "LIVE_TAKES_DIR", tmp_path)

    def write(stem, rep):
        d = tmp_path / stem
        d.mkdir(exist_ok=True)
        (d / f"{stem}.json").write_text(json.dumps(rep), encoding="utf-8")

    for stem, rep in reports.items():
        write(stem, rep)


def test_a_missing_report_explains_and_exits_2(capsys, tmp_path, monkeypatch):
    _takes_in(tmp_path, monkeypatch, {"live_20260929_160000": _report([60], [0.2])})
    assert compare_takes.main(
        ["live_20260929_160000", "live_20260929_161442"]
    ) == 2
    out = capsys.readouterr().out
    assert "live_20260929_161442" in out  # the missing one is NAMED


def test_stable_takes_say_so_in_both_languages(capsys, tmp_path, monkeypatch):
    _takes_in(
        tmp_path,
        monkeypatch,
        {
            "live_20260929_161442": _report([60, 62, 64], [0.2, 0.9, 1.6]),
            "live_20260929_161537": _report([60, 62, 64], [0.21, 0.92, 1.62]),
        },
    )
    argv = ["live_20260929_161442", "live_20260929_161537"]
    set_language("fr")
    assert compare_takes.main(argv) == 0
    fr = capsys.readouterr().out
    assert "répéter la phrase" in fr and "Identiques : 3" in fr
    assert "✓" in fr
    set_language("en")
    assert compare_takes.main(argv) == 0
    en = capsys.readouterr().out
    assert "Phrase-repeat comparison" in en and "Identical: 3" in en
    set_language("fr")


def test_differences_are_listed_in_singing_order(capsys, tmp_path, monkeypatch):
    _takes_in(
        tmp_path,
        monkeypatch,
        {
            # A: 60, 64(at 0.9), 67 — B: 60, 65(at 1.2), 67: the middle
            # note is BOTH ±1 (64->65) and shifted (0.9 -> 1.2 beyond the
            # window); the ±1 family absorbs the shift (the net's rule)
            "live_a": _report([60, 64, 67], [0.2, 0.9, 1.6]),
            "live_b": _report([60, 65, 67], [0.2, 1.2, 1.6]),
        },
    )
    set_language("fr")
    assert compare_takes.main(["live_a", "live_b"]) == 0
    out = capsys.readouterr().out
    # the ±1 neighbour (64->65, Mi4->Fa4) is NAMED as the rounding family,
    # NOT as a wrong note, and the verdict is NOT stable (a label change)
    assert "voisin d'un demi-ton" in out and "Fa4" in out
    assert not out.startswith("✓")
    # a purely shifted same-pitch note carries its beats
    _takes_in(
        tmp_path,
        monkeypatch,
        {
            "live_a": _report([60, 64], [0.2, 0.9]),
            "live_b": _report([60, 64], [0.2, 1.4]),
        },
    )
    assert compare_takes.main(["live_a", "live_b"]) == 0
    out = capsys.readouterr().out
    assert "décalée de" in out
    set_language("fr")


def test_the_json_archive_reflects_the_comparison(capsys, tmp_path, monkeypatch):
    _takes_in(
        tmp_path,
        monkeypatch,
        {
            "live_a": _report([60, 62], [0.2, 0.9]),
            "live_b": _report([60, 64], [0.2, 0.9]),
        },
    )
    out_path = tmp_path / "rapport.json"
    assert compare_takes.main(["live_a", "live_b", "--json", str(out_path)]) == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["a"] == "live_a" and data["b"] == "live_b"
    assert data["counts"]["same"] == 1 and data["counts"]["moved"] == 1
    assert data["stable"] is False
    assert data["pairs"][0]["reason"] == "moved"


def test_a_direct_report_path_is_accepted(capsys, tmp_path, monkeypatch):
    pa = tmp_path / "prise1.json"
    pb = tmp_path / "prise2.json"
    pa.write_text(json.dumps(_report([60], [0.2])), encoding="utf-8")
    pb.write_text(json.dumps(_report([60], [0.25])), encoding="utf-8")
    assert compare_takes.main([str(pa), str(pb)]) == 0
    assert "✓" in capsys.readouterr().out
