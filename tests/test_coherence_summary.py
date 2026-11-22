"""The CLI adapter (`scripts/coherence_summary.py`): the core owns the contract.

Since `core.coherence`, the script is only an ADAPTER: the decisions
(sessions by a 30-min gap, rate/100, delta, verify, journal) live in the
core and are pinned there by `tests/test_core_coherence.py`. This test
checks the ADAPTER — what a front-end can break on its own:

1. the REAL round-trip: an export (a real `export_result`, a real
   observation hiding a secondary plateau) is re-read by the aggregator
   — what the exporter writes (CSV extended with the published plateau),
   the adapter reads;
2. empty/missing folders: empty table, return code, orientation
   messages — day one, before even the first take.

The core's decisions around that round-trip (real take names only, fr/en
labels, sessions, rates, verify) are pinned in
`tests/test_core_coherence.py`.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

SPEC = importlib.util.spec_from_file_location(
    "coherence_summary", ROOT / "scripts" / "coherence_summary.py"
)
assert SPEC is not None and SPEC.loader is not None  # the script exists
coherence_summary = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("coherence_summary", coherence_summary)
SPEC.loader.exec_module(coherence_summary)


# --------------------------------------------------------------------------- #
# Real round-trip: export -> aggregation
# --------------------------------------------------------------------------- #


def _take_with_hidden_plateau():
    """A test take: a quarter note hiding a +2 st plateau (a suspect)."""
    from v2m.core.continuity import coherence_report
    from v2m.core.notes import Note
    from v2m.core.observation import VoiceObservation
    from v2m.core.pitch import PitchTrack

    midi = 57
    hz = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    drift = 440.0 * 2.0 ** ((midi - 2 - 69) / 12.0)
    times, curve = [], []
    t = 0.0
    for i in range(30):  # 0.60 s at 20 ms, plateau >= 0.12 s on both sides
        times.append(t)
        curve.append(drift if i >= 20 else hz)  # the top 0.2 s: the suspect
        t += 0.02
    observation = VoiceObservation(
        pitch=PitchTrack(
            times=np.asarray(times),
            hz=np.asarray(curve),
            confidence=np.ones(len(times)),
            transition=np.zeros(len(times), dtype=bool),
        ),
        attacks=None,
        duration_s=t,
        sample_rate=22_050,
        frame_dt=0.02,
        accent_envelope=None,
    )
    notes = [Note(0.0, 0.6, midi, 0.9)]
    return observation, notes, coherence_report(observation, notes)


def test_the_exported_coherence_csv_round_trips_through_the_aggregator(
    tmp_path, monkeypatch
):
    """What `export_result` writes, `takes_under` reads without loss."""
    from v2m.core.grid import BeatGrid
    from v2m.core.keys import KeyCandidate, KeyResult
    from v2m.core.pipeline import AnalysisResult
    from v2m.core.report import ARTIFACT_EXTENSIONS, export_result
    from v2m.core.tempo import TempoResult

    observation, notes, block = _take_with_hidden_plateau()
    assert block["suspects"], "the hidden plateau must be seen (else the test proves nothing)"
    result = AnalysisResult(
        source="live", duration_s=0.6, notes=notes,
        tempo=TempoResult(bpm=120.0, beats=np.arange(0.0, 0.6, 0.5)),
        key=KeyResult(KeyCandidate("A", "minor", 0.9), []),
        grid=BeatGrid(120.0),
        observation=observation,
    )
    take_dir = tmp_path / "live_20260926_120000"
    stem = take_dir / "live_20260926_120000"
    export_result(result, stem)

    # the suffix the aggregator sweeps is the one the exporter writes: the
    # round-trip catches an extension renamed on one side only
    from v2m.core.coherence import CSV_NAME

    assert CSV_NAME in ARTIFACT_EXTENSIONS
    assert (take_dir / f"{stem.name}{CSV_NAME}").exists()

    from v2m.core import coherence as coh

    takes = coh.takes_under(tmp_path)
    assert [t.stem for t in takes] == ["live_20260926_120000"]
    assert takes[0].checked == block["checked"]
    read = takes[0].suspects
    assert len(read) == len(block["suspects"])
    for got, want in zip(read, block["suspects"], strict=False):
        assert got["midi"] == want["midi"]
        assert got["inner_range_st"] == pytest.approx(want["inner_range_st"], abs=0.01)
        assert got["secondary_plateau_s"] == pytest.approx(want["secondary_plateau_s"], abs=0.01)
        # the PUBLISHED plateau (time, pitch, signed interval) crosses too
        assert got["secondary_plateau_start_s"] == pytest.approx(
            want["secondary_plateau_start_s"], abs=0.02
        )
        assert got["secondary_plateau_offset_st"] == pytest.approx(
            want["secondary_plateau_offset_st"], abs=0.02
        )
    text = coh.render(takes)
    assert "20260926" in text and "suspects 1" in text


# --------------------------------------------------------------------------- #
# The adapter: return codes and first-run orientation
# --------------------------------------------------------------------------- #


def test_empty_and_missing_folders_do_not_raise(tmp_path, capsys):
    from v2m.core import coherence as coh

    assert coh.takes_under(tmp_path) == []
    assert "Aucun" in coh.render([]) or "rien" in coh.render([])
    assert coherence_summary.main([str(tmp_path)]) == 0
    assert coherence_summary.main([str(tmp_path / "n_existe_pas")]) == 2


def test_the_summary_writes_its_numbers_in_the_current_language():
    """A French run reads "1,25", an English one "1.25" — one row, one language.

    The adapter is bilingual (`main` follows ~/.v2m_lang and every label comes
    from the fr/en tables), so the numbers go through `i18n.format_number` too:
    a localized column must never sit beside a hand-dotted one.
    """
    from v2m.core import coherence as coh
    from v2m.core.i18n import get_language, set_language

    take = coh.TakeAudit(
        stem="live_20260926_120000",
        date="2026-09-26",
        clock="12:00",
        epoch_s=0.0,
        suspects=[],
        checked=80,
    )
    rows = [
        {
            "label": "2026-09-26 12:00",
            "takes": 1,
            "checked": 80,
            "suspects": 0,
            "rate_per_100": 1.25,
            "delta_vs_prev": 0.5,
            "contour_rate": 0.9425,
            "verify_bad": 0,
            "plateau_mean_s": 3.5,
            "plateau_max_s": 4.5,
        }
    ]
    original = get_language()
    try:
        set_language("fr")
        french = coh.render([take], rows=rows, days=[])
        set_language("en")
        english = coh.render([take], rows=rows, days=[])
    finally:
        set_language(original)

    # rate, delta and both plateau columns: one separator per run
    assert "1,25" in french and "+0,50" in french and "3,50" in french
    assert "1.25" in english and "+0.50" in english and "3.50" in english
    assert "1.25" not in french, "a French row mixed a dot into a comma table"
    assert "1,25" not in english
    # a whole-point share has no separator to translate
    assert "94%" in french and "94%" in english


def test_first_run_guides_instead_of_confusing(capsys):
    """The first run is a starting state, not an error."""
    assert coherence_summary.main([str(ROOT / "prises_live_inexistant")]) == 2
    out = capsys.readouterr().out
    assert "prises_live" in out  # where we searched
    assert ".coherence.csv" in out or "prise" in out  # what creates a report
