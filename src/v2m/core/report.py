"""Report and export: one interface to all artifacts.

`export_result(res, stem, ...)` writes .mid / .musicxml / .json / .csv (and
optionally .pdf) from an `AnalysisResult` and returns the report dict (one
schema shared by the three contexts: CLI, GUI, live).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import cast

from .figures import figures_summary
from .grid import BeatGrid, Meter
from .midi_export import build_midi
from .music import midi_to_name, midi_to_name_fr
from .musicxml import save_musicxml
from .i18n import tr
from .pdf_export import PdfToolMissing, export_pdf, find_notation_tool, install_hint


# Where live takes land: one folder for every live export, one sub-folder
# per take (timestamped, so no two takes can collide).
LIVE_TAKES_DIR = Path("prises_live")


def live_take_name() -> str:
    """Unique stem for one live take: live_YYYYmmdd_HHMMSS."""
    return f"live_{time.strftime('%Y%m%d_%H%M%S')}"


def live_export_dir(stem: str | Path) -> Path:
    """Folder holding one live take's artifacts: `prises_live/<stem>/`."""
    return LIVE_TAKES_DIR / Path(stem).name


def export_result(
    res,
    stem: str | Path,
    quantize_s: float | None = None,
    transpose: int = 0,
    with_pitch_curve: bool = False,
    meter=None,
) -> dict:
    """Write the artifacts under the `stem` prefix and return the JSON dict.

    `meter`: detected Meter for the score (default: res.meter).
    `with_pitch_curve`: adds the pitch curve to the JSON (CLI usage).

    The PDF is a separate step (`engrave_pdf`): it needs an external tool
    and must never cost the data artifacts, nor the report returned here.
    """
    if res.tempo is None or res.key is None:
        # a French user reads this through `err analyze failed`/`err ui export
        # failed`: the message itself must follow the UI language too
        raise ValueError(tr("err export incomplete"))
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)  # a nested stem (live take) needs its folder
    grid = getattr(res, "grid", None)
    if grid is None or meter is not None:
        # An explicit meter is an intentional export override; otherwise the
        # pipeline's immutable grid remains the single metric decision.
        selected_meter = meter or getattr(res, "meter", None)
        if selected_meter is None:
            raise ValueError(tr("err export needs meter"))
        grid = BeatGrid(res.tempo.bpm, cast(Meter, selected_meter))
    meter = grid.meter
    positions = res.note_positions if meter == getattr(res, "meter", None) else None
    if positions is None:
        positions = grid.positions(res.notes)
    pos_by_idx = {i: p for i, p in enumerate(positions)} if positions else {}

    mid = build_midi(
        res.notes,
        grid=grid,
        key=res.key,
        quantize_s=quantize_s,
        transpose=transpose,
    )
    mid.save(str(stem) + ".mid")
    save_musicxml(
        res.notes,
        grid,
        res.key,
        str(stem) + ".musicxml",
        chord_progression=getattr(res, "chord_progression", None),
        rests=getattr(res, "rests", None),
    )

    rep = {
        "source": res.source,
        "duration_s": round(res.duration_s, 3),
        "tempo": {"bpm": res.tempo.bpm},
        "meter": meter.label if meter else None,
        "num_measures": grid.measures(res.notes),
        "key": {
            "best": res.key.best.label,
            "score": round(res.key.best.score, 4),
            "alternatives": [
                {"label": a.label, "score": round(a.score, 4)} for a in res.key.alternatives
            ],
        },
        "n_notes": len(res.notes),
        "tuning_offset_cents": round(
            100.0 * getattr(res, "tuning_offset_st", 0.0), 1
        ),
        "rests": [
            {
                "start_s": round(r.start, 4),
                "end_s": round(r.end, 4),
                "duration_s": round(r.duration, 4),
                "figure": r.figure,
            }
            for r in (getattr(res, "rests", None) or [])
        ],
        "figures_summary": figures_summary(res.notes),
        "chord": (
            {"label": res.chord.best.label, "score": res.chord.best.score,
             "alternatives": [c.label for c in res.chord.alternatives]}
            if res.chord else None
        ),
        "chord_segments": [
            {"start_s": round(t0, 3), "end_s": round(t1, 3), "label": c.label}
            for t0, t1, c in (res.chord_segments or [])
        ],
        "chord_progression": [
            {"measures": [m0, m1], "label": c.label}
            for m0, m1, c in (getattr(res, "chord_progression", None) or [])
        ],
        "notes": [
            {
                "start_s": round(n.start, 4),
                "end_s": round(n.end, 4),
                "duration_s": round(n.duration, 4),
                "midi": n.midi,
                "name": midi_to_name(n.midi),
                "name_fr": midi_to_name_fr(n.midi),
                "confidence": round(n.confidence, 3),
                "strength": round(getattr(n, "strength", 0.0), 2),
                "figure": n.figure,
                "position": list(pos_by_idx[i]) if i in pos_by_idx else None,
            }
            for i, n in enumerate(res.notes)
        ],
        "quantized": bool(quantize_s),
        "transpose_semitones": transpose,
    }
    if with_pitch_curve and res.pitch is not None:
        rep["pitch_curve_hz"] = [
            round(float(v), 2) if v == v else None  # NaN -> null
            for v in res.pitch.hz[::2]
        ]
        rep["pitch_times_s"] = [round(float(t), 4) for t in res.pitch.times[::2]]
    with open(str(stem) + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    with open(str(stem) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "measure", "beat", "start_s", "end_s", "duration_s",
                    "midi", "name", "name_fr", "confidence", "figure"])
        for i, n in enumerate(res.notes):
            pos = pos_by_idx.get(i)
            w.writerow([i, pos[0] if pos else "", pos[1] if pos else "",
                        round(n.start, 4), round(n.end, 4), round(n.duration, 4),
                        n.midi, midi_to_name(n.midi), midi_to_name_fr(n.midi),
                        round(n.confidence, 3), n.figure or ""])
    return rep


def engrave_pdf(stem: str | Path) -> str:
    """Engrave `<stem>.musicxml` into `<stem>.pdf` (MuseScore or LilyPond).

    Raises `PdfToolMissing` when no tool is installed and `PdfExportError`
    when the conversion itself fails, so the caller can decide how loud that
    is; the artifacts written by `export_result` are already on disk either
    way.
    """
    stem = Path(stem)
    if find_notation_tool() is None:
        raise PdfToolMissing(tr("err pdf skipped").format(hint=install_hint()))
    return export_pdf(str(stem) + ".musicxml", str(stem) + ".pdf")
