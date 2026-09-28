"""The recording guide (`scripts/guide_capture.py`) and its format.

The `prises_verite/` folder collects ear-verified takes: the truth
format (`<name>.truth.json`) is a contract documented in its README —
these tests keep it alive: an invalid JSON is REFUSED (not corrected),
and the bounds refinement must follow the real sung note, never the
neighbouring one.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "guide_capture", ROOT / "scripts" / "guide_capture.py"
)
gc = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("guide_capture", gc)
SPEC.loader.exec_module(gc)

sys.path.insert(0, str(ROOT / "tests"))
from voicetruth import SungNote, synthesize  # noqa: E402


# --------------------------------------------------------------------------- #
# The truth format is a contract
# --------------------------------------------------------------------------- #


def test_a_valid_truth_passes_validation():
    data = {
        "name": "a",
        "audio": "a.wav",
        "checked_by": "humain",
        "notes": [{"midi": 64, "start": 0.5, "end": 1.0, "lyric": "la"}],
    }
    assert gc.validate_truth(data, "a.wav") == []


def test_an_invalid_truth_is_refused_not_corrected():
    data = {
        "audio": "a.wav",
        "checked_by": "machine",
        "notes": [
            {"midi": 64, "start": 1.0, "end": 0.9},     # inverted
            {"midi": 200, "start": 0.5, "end": 0.8},    # off piano
            {"midi": 64, "start": 0.3, "end": 0.6},     # overlaps the 1st
            {"start": 1.5, "end": 2.0},                 # missing field
        ],
    }
    errs = gc.validate_truth(data, "a.wav")
    assert any("checked_by" in e for e in errs)
    assert any("end <= start" in e for e in errs)
    assert any("hors piano" in e for e in errs)
    assert any("chevauche" in e for e in errs)
    assert any("manquant" in e for e in errs)


def test_the_truth_loader_refuses_an_invalid_file(tmp_path: Path):
    p = tmp_path / "x.truth.json"
    p.write_text(json.dumps({"audio": "x.wav", "checked_by": "humain", "notes": []}),
                 encoding="utf-8")
    with pytest.raises(SystemExit):
        gc.load_truth(p)


# --------------------------------------------------------------------------- #
# The refinement follows the real note, never the neighbour
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def two_notes() -> tuple[np.ndarray, int]:
    truth = [SungNote(64, 0.50, 0.60), SungNote(67, 1.30, 0.55)]
    return synthesize(truth, glide_s=0.09, consonant_s=0.05), 22_050


def test_a_late_human_mark_refines_to_the_sung_note(two_notes):
    """The human sang 0.50-1.10 but placed their mark 0.75-1.40: the
    refinement must fall back on the sung note, not slide into the next
    one (the old algorithm — first/last stable instant of the
    margin — extended right INTO note 2)."""
    y, sr = two_notes
    s, e = gc.refine_bounds(y, sr, 0.75, 1.40)
    assert 0.40 <= s <= 0.70, s
    assert 0.90 <= e <= 1.25, e  # note 2 starts at 1.30: forbidden to go there


def test_a_precise_mark_stays_on_its_note(two_notes):
    y, sr = two_notes
    s, e = gc.refine_bounds(y, sr, 0.50, 1.10)
    assert 0.40 <= s <= 0.70 and 0.90 <= e <= 1.25


def test_a_mark_outside_any_note_is_kept_asis(two_notes):
    """A mark placed in the silence: no stable span within half the
    mark's length — the human bounds are kept as is (the helper must
    never degrade a mark)."""
    y, sr = two_notes
    s, e = gc.refine_bounds(y, sr, 2.50, 2.60)
    assert (s, e) == (2.50, 2.60)


def test_the_refined_note_reads_the_right_midi(two_notes):
    """End to end: a late mark -> refined bounds -> the dominant midi read."""
    y, sr = two_notes
    s, e = gc.refine_bounds(y, sr, 0.75, 1.40)
    assert gc._dominant_midi(y, sr, s, e) == 64
