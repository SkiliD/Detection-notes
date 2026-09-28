"""The `prises_verite/` corpus replayed on every suite run.

Every ear-verified take (`<name>.wav` + `<name>.truth.json`, format
documented in the folder's README) is ANALYZED and compared to its
truth: it is the non-drift net on real voices — the equivalent of
`test_real_take_segmentation_does_not_drift` (golden fingerprint), but
judged against a human rather than against the machine's own past.

The test does NOT demand accuracy (a real voice may reveal a
segmentation defect; the score is printed), it demands STABILITY: a
first run freezes the per-take fingerprint (in
`prises_verite/<name>.fingerprint.json`, gitignored), and any later
divergence fails the test — conscious re-freeze via
`rm <name>.fingerprint.json` after reading the diff, like the golden
fingerprints of `chant.wav`/`demo.wav`.

Missing takes (empty folder, cloned repo): the test skips.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from v2m.core import audio_io  # noqa: E402
from v2m.core.pipeline import analyze_signal  # noqa: E402

TRUTH_DIR = ROOT / "prises_verite"


def _fingerprint(res) -> dict:
    """A take's compared landmark: note count + their fingerprint."""
    return {
        "n_notes": len(res.notes),
        "notes": [
            [round(n.start, 2), round(n.end, 2), n.midi] for n in res.notes
        ],
    }


def _corpus() -> list[Path]:
    if not TRUTH_DIR.exists():
        return []
    return sorted(
        p for p in TRUTH_DIR.glob("*.wav") if p.with_name(p.stem + ".truth.json").exists()
    )


def test_the_corpus_folder_documents_its_format():
    """The folder exists in the repo with its README committing to the format."""
    assert (TRUTH_DIR / "README.md").exists()


@pytest.mark.parametrize("audio", _corpus(), ids=lambda p: p.stem)
def test_each_verified_take_does_not_drift(audio: Path):
    """Replays the take's analysis and compares to the frozen fingerprint.

    The first run WRITES the fingerprint (and fails on purpose to force a
    human read of the first score — a take nobody looked at is not a
    net).
    """
    truth_path = audio.with_name(audio.stem + ".truth.json")
    fp_path = audio.with_name(audio.stem + ".fingerprint.json")

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    res = analyze_signal(audio_io.load_file(str(audio)), audio_io.SR,
                         source=str(audio))
    fp = _fingerprint(res)

    # accuracy is REPORTED (the test harness, like score_accuracy),
    # never demanded here: the fingerprint judges drift
    from voicetruth import SungNote, measure

    sung = [SungNote(n["midi"], n["start"], n["end"] - n["start"])
            for n in truth["notes"]]
    acc = measure(res.notes, sung, tol_s=0.15)
    print(f"\n{audio.name}: {acc.line()}")

    if not fp_path.exists():
        fp_path.write_text(json.dumps(fp, indent=2), encoding="utf-8")
        pytest.fail(
            f"first run for {audio.name} : fingerprint frozen "
            f"({fp['n_notes']} notes) — re-read the score above, "
            f"run again to arm the net"
        )
    saved = json.loads(fp_path.read_text(encoding="utf-8"))
    assert fp == saved, (
        f"segmentation drift on {audio.name} :\n"
        f"  expected {saved['n_notes']} notes, got {fp['n_notes']}\n"
        f"  (conscious re-freeze: rm {fp_path.name} after reading the diff)"
    )
