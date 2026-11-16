"""The recording guide (`scripts/guide_capture.py`) and its format.

The `prises_verite/` folder collects ear-verified takes: the truth
format (`<name>.truth.json`) is a contract documented in its README —
these tests keep it alive: an invalid JSON is REFUSED (not corrected),
and the bounds refinement must follow the real sung note, never the
neighbouring one.
"""

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "guide_capture", ROOT / "scripts" / "guide_capture.py"
)
# the spec is built from a path in this repo: asserting it keeps pyright
# honest about the loader instead of sprinkling ignores
assert SPEC is not None and SPEC.loader is not None
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
            {"midi": 64, "start": 1.0, "end": 0.9},  # inverted
            {"midi": 200, "start": 0.5, "end": 0.8},  # off piano
            {"midi": 64, "start": 0.3, "end": 0.6},  # overlaps the 1st
            {"start": 1.5, "end": 2.0},  # missing field
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
    p.write_text(
        json.dumps({"audio": "x.wav", "checked_by": "humain", "notes": []}), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as excinfo:
        gc.load_truth(p)
    # refused for ITS reason (no notes), not by accident on another field
    assert "notes" in str(excinfo.value)


def test_a_truth_the_guide_wrote_loads_back(tmp_path: Path):
    """The round trip the audit lives on: `<stem>.truth.json` carrying
    `audio: "<stem>.wav"` (exactly what capture_and_annotate writes) must
    PASS validation. Deriving the compared name from the bare stem
    refused every take the guide produced, so `--audit` could only run on
    a hand-typed truth."""
    p = tmp_path / "ma_prise.truth.json"
    p.write_text(
        json.dumps(
            {
                "name": "ma_prise",
                "audio": "ma_prise.wav",
                "recorded": "2026-10-13",
                "checked_by": "humain",
                "notes": [{"midi": 64, "start": 0.5, "end": 1.0, "lyric": "la"}],
            }
        ),
        encoding="utf-8",
    )
    data = gc.load_truth(p)
    assert data["notes"][0]["midi"] == 64
    assert gc.truth_path_for(tmp_path / "ma_prise.wav") == p


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


# --------------------------------------------------------------------------- #
# The documented workflow runs on a real console (cp1252 on Windows)          #
# --------------------------------------------------------------------------- #


def _checked_take(tmp_path: Path) -> Path:
    """A take plus the truth an ear check would write next to it."""
    sung = [
        SungNote(64, 0.40, 0.70),
        SungNote(67, 1.30, 0.70),
        SungNote(65, 2.20, 0.70),
        SungNote(64, 3.10, 0.70),
    ]
    wav = tmp_path / "ma_prise.wav"
    gc._write_wav(wav, synthesize(sung, glide_s=0.09, consonant_s=0.05, seed=1), 22_050)
    gc.truth_path_for(wav).write_text(
        json.dumps(
            {
                "name": "ma_prise",
                "audio": wav.name,
                "recorded": "2026-10-13",
                "checked_by": "humain",
                "notes": [
                    {
                        "midi": n.midi,
                        "start": round(n.start_s, 2),
                        "end": round(n.start_s + n.dur_s, 2),
                    }
                    for n in sung
                ],
            }
        ),
        encoding="utf-8",
    )
    return wav


def test_annotate_writes_a_machine_draft(tmp_path):
    """`--annotate` gives a take recorded elsewhere a machine SKELETON.

    Writing a truth by hand over a waveform is what keeps a real take out
    of the corpus; the draft is the reading, and BOTH gates (the audit and
    the corpus net) refuse it until an ear check flips `checked_by`.
    """
    wav = _checked_take(tmp_path)
    gc.truth_path_for(wav).unlink()  # as if the take came from a phone
    assert gc.main([str(wav), "--annotate"]) == 0
    draft = json.loads(gc.truth_path_for(wav).read_text(encoding="utf-8"))
    assert draft["checked_by"] == "machine"
    assert draft["audio"] == wav.name
    assert len(draft["notes"]) == 4, "a draft carries the engine's reading entire"
    assert any("checked_by" in e for e in gc.validate_truth(draft, wav.name))
    with pytest.raises(SystemExit):
        gc.load_truth(gc.truth_path_for(wav))


def test_annotate_never_overwrites_a_truth(tmp_path):
    """A verified truth is not raw material for a draft."""
    wav = _checked_take(tmp_path)
    before = gc.truth_path_for(wav).read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        gc.main([str(wav), "--annotate"])
    assert gc.truth_path_for(wav).read_text(encoding="utf-8") == before


def test_note_markers_are_seconds_from_the_capture_start(tmp_path, monkeypatch):
    """The markers are pressed AFTER the take; the refinement reads them as
    seconds INTO the signal. An epoch timestamp put every note at the end of
    the recording, so a guided take came out as one note at the tail."""
    import types

    class FakeCapture:
        def __init__(self, on_block=None) -> None:  # noqa: ANN001 - the real contract
            self.on_block = on_block
            self.duration = 8.0

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def signal(self) -> np.ndarray:
            return np.zeros(int(8.0 * 22_050), dtype=np.float32)

    monkeypatch.setattr("v2m.core.audio_io.LiveCapture", FakeCapture)

    clock = iter([1000.0, 1005.0, 1007.0])  # capture start, then the two presses

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        gc,
        "time",
        types.SimpleNamespace(
            time=lambda: next(clock), sleep=interrupt, strftime=__import__("time").strftime
        ),
    )
    answers = iter(["", "", "", "s"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    gc.capture_and_annotate(tmp_path / "essai")
    notes = json.loads((tmp_path / "essai.truth.json").read_text(encoding="utf-8"))["notes"]
    assert notes[0]["start"] == 5.0
    assert notes[0]["end"] == 7.0


def test_the_audit_verdict_survives_a_cp1252_console(tmp_path, monkeypatch):
    """`--audit` is the door into the corpus and it runs on a Windows
    console: the verdict line's arrow is not cp1252-encodable, so the run
    died with UnicodeEncodeError right after printing its whole note-by-
    note diagnosis — the verdict the human needs never appeared. The
    script now forces UTF-8 on its own streams."""
    wav = _checked_take(tmp_path)
    sink = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", sink)
    assert gc.main([str(wav), "--audit"]) == 0
    sink.flush()
    # the forced stream emits UTF-8 whatever the console was opened as
    out = sink.buffer.getvalue().decode("utf-8")
    assert "corpus de" in out  # the ACCEPTING verdict, not the divergence one
