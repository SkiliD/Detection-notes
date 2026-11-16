"""The artifact contract: one spelling for every path a take owns.

`core.takes` owns the suffixes, paths and the single sidecar reader;
these tests walk the artifact contract's round trip
(write through `explain`, read through `read_sidecar`) and the name-grammar
guard that keeps a stray directory out of the
take tables.
"""

from __future__ import annotations

from v2m.core import takes
from v2m.core.compare import write_sidecar


def test_sidecar_round_trip_through_the_contract(tmp_path):
    """write through `compare`, read through `read_sidecar` — one spelling."""
    stem = tmp_path / "live_20260927_204442"
    block = {"live": 23, "final": 17, "contours": 15, "off1": 9,
             "moved": 6, "absorbed": 0, "rate": 0.65,
             "engine": "a037199ccf6aecb0"}
    written = write_sidecar(stem, block)
    assert written and written.endswith(takes.SIDECAR_SUFFIX)
    # the folder-relative view (the coherence/drift readers' angle) agrees
    assert takes.read_sidecar(stem.parent, stem.name) == block
    # the folder-IS-the-stem view (one sub-folder per take) agrees too
    take_dir = tmp_path / "live_20260927_204442x"
    take_dir.mkdir()
    write_sidecar(take_dir / take_dir.name, block)
    assert takes.read_sidecar(take_dir) == block


def test_read_sidecar_never_guesses(tmp_path):
    """Missing, corrupt or non-dict: None, not an exception."""
    assert takes.read_sidecar(tmp_path) is None  # nothing there
    d = tmp_path / "live_20260921_191031"
    d.mkdir()
    (d / f"{d.name}{takes.SIDECAR_SUFFIX}").write_text("{oops", encoding="utf-8")
    assert takes.read_sidecar(d) is None  # corrupt: no audit, not a crash
    (d / f"{d.name}{takes.SIDECAR_SUFFIX}").write_text("[1, 2]", encoding="utf-8")
    assert takes.read_sidecar(d) is None  # a list is not an audit dict


def test_take_dir_stem_refuses_a_stray_directory(tmp_path):
    """The grammar guards the take tables: only minted names are takes."""
    good = tmp_path / "live_20260927_204442"
    good.mkdir()
    bad = tmp_path / "not_a_take"
    bad.mkdir()
    assert takes.take_dir_stem(good) == good.name
    assert takes.take_dir_stem(bad) is None


def test_evidence_path_sits_next_to_its_audio(tmp_path):
    audio = tmp_path / "live_20260927_204442.wav"
    assert takes.evidence_path(audio) == tmp_path / "live_20260927_204442.observation.npz"


def test_has_artifacts_lists_only_what_exists(tmp_path):
    stem = tmp_path / "live_20260927_204442"
    (tmp_path / "live_20260927_204442.json").write_text("{}", encoding="utf-8")
    (tmp_path / "live_20260927_204442.csv").write_text("t,start", encoding="utf-8")
    listed = takes.has_artifacts(stem, (".wav",))
    assert listed == [".json", ".csv"]  # the ARTIFACTS order, missing ones gone
    assert ".wav" not in listed  # an extra is checked the same way: on disk only


def test_the_report_tuple_and_the_contract_agree():
    """report.ARTIFACT_EXTENSIONS is takes.ARTIFACTS — one owner, no drift."""
    from v2m.core.report import ARTIFACT_EXTENSIONS

    assert ARTIFACT_EXTENSIONS == takes.ARTIFACTS
