"""The segmentation audit engine: fingerprints, golden files, comparison.

The FROZEN FORMAT of the non-drift net lives here — not in the script
that prints it. `segmentation_fingerprint` defines what a pass must
remember (tempo, meter, key, every note with its rounded numbers, every
rest); `compare_fingerprints` is the whole comparison policy; the
golden I/O owns where a take's reference lives (`<audio>.golden.json`)
and how it is read and written. A fingerprint must survive the
json.dumps/json.loads round-trip (tuples become lists on purpose).

Why a core module and not script-local helpers: three consumers read
this format — the `score_accuracy.py` adapter (print it), the suite's
drift nets (`test_voice_accuracy`, `test_score_accuracy`, the frozen
references) and any future audit. Two adapters of one format are a real
seam; the module is the seam's owner. The script keeps argparse and
printing only.
"""

from __future__ import annotations

import json
from pathlib import Path

from v2m.core.pipeline import AnalyzeOptions, analyze_signal
from v2m.core import audio_io


def segmentation_fingerprint(result) -> dict:
    """The segmentation numbers that must stay stable from one pass to the next.

    Tuples become lists: the fingerprint also goes to JSON (--golden) and
    must survive the json.dumps/json.loads round-trip.
    """
    return {
        "tempo_bpm": result.tempo.bpm if result.tempo else None,
        "meter": result.meter.label if result.meter else None,
        "key": result.key.best.label if result.key else None,
        "notes": [
            [round(n.start, 4), round(n.duration, 4), n.midi, n.figure, round(n.strength, 6)]
            for n in result.notes
        ],
        "rests": [
            [round(r.start, 4), round(r.duration, 4), r.figure]
            for r in (getattr(result, "rests", None) or [])
        ],
    }


def golden_path(audio: str) -> Path:
    """A take's golden fingerprint lives beside it: <audio>.golden.json."""
    return Path(audio).with_suffix(".golden.json")


def load_golden(audio: str) -> dict | None:
    """The recorded fingerprint, None when it does not exist yet."""
    path = golden_path(audio)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_golden(audio: str, fingerprint: dict) -> Path:
    """Freeze `fingerprint` as `audio`'s golden reference (first visit)."""
    path = golden_path(audio)
    path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return path


def replay_fingerprint(path: str, use_attacks: bool) -> dict:
    """A full second pass (re-decoding included, no artifact written)."""
    y = audio_io.load_file(path)
    result = analyze_signal(y, audio_io.SR, source=path, opts=AnalyzeOptions(use_attacks=use_attacks))
    return segmentation_fingerprint(result)


def compare_fingerprints(first: dict, second: dict) -> list[str]:
    """The differences between two passes, as readable lines (empty = stable)."""
    lines = []
    for field in ("tempo_bpm", "meter", "key"):
        if first[field] != second[field]:
            lines.append(f"  {field} : {first[field]} -> {second[field]}")
    for label, key in (("note", "notes"), ("respiration", "rests")):
        if len(first[key]) != len(second[key]):
            lines.append(f"  {key} : {len(first[key])} -> {len(second[key])}")
            continue
        for i, (a, b) in enumerate(zip(first[key], second[key])):
            if a != b:
                lines.append(f"  {label} {i + 1} : {a} -> {b}")
    return lines
