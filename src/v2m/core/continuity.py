"""Per-note coherence: is the pitch INSIDE each written note the note itself?

A long note whose curve holds a DIFFERENT plateau for 120 ms or more is a
swallowed phrase (re-sung notes glued into one continuity — the 2026-09-26
user report), not a held note. The detectors already split most of them; this
module is the SAFETY NET that makes what slipped through VISIBLE instead of
silent: `report.export_result` publishes one coherence block per take, each
suspect note carrying the plateau it hides (where it sits, how long, at which
pitch). A report that shows the doubt lets the singer judge — the machine
does not silently rewrite what it is unsure of.
"""

from __future__ import annotations

import numpy as np

from .music import hz_to_midi

# The contract, in one place: a note shorter than twice the plateau minimum
# cannot hide one; a plateau shorter than this is a transition, not a note
# (the live segmenter's own SPLIT_FRAMES ceiling lives at the same scale);
# a plateau within this band of the label is the note's own vibrato/drift.
MIN_NOTE_S = 0.30
PLATEAU_MIN_S = 0.12
PLATEAU_ST = 1.0


def suspect_spans(observation, notes: list) -> list[dict]:
    """The notes that hide a SECONDARY plateau (>= PLATEAU_MIN_S away).

    One dict per suspect: the note's identity, the pitch range really sung
    inside it, and the hidden plateau (length in frames + its own pitch).
    Everything is read on the observation's cleaned curve — the same evidence
    the notes were interpreted from, so a suspect is always a decision the
    interpreter made (or a residual glide), never a reading artifact.
    """
    track = getattr(observation, "pitch", None)
    if track is None or not notes:
        return []
    midi_curve = hz_to_midi(track.hz)
    times = track.times
    frame_dt = getattr(observation, "frame_dt", 0.0) or 0.01
    min_frames = max(3, int(round(PLATEAU_MIN_S / frame_dt)))
    suspects: list[dict] = []
    for n in notes:
        if n.duration < max(MIN_NOTE_S, 2 * PLATEAU_MIN_S):
            continue
        inside = (times >= n.start) & (times <= n.end)
        seg = midi_curve[inside]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 2 * min_frames:
            continue
        label = float(n.midi)
        best_len = 0
        best_start = 0
        run = 0
        run_start = 0
        for k, v in enumerate(seg):
            if abs(v - label) > PLATEAU_ST:
                if run == 0:
                    run_start = k
                run += 1
                if run > best_len:
                    best_len, best_start = run, run_start
            else:
                run = 0
        if best_len < min_frames:
            continue
        plateau = seg[best_start : best_start + best_len]
        # the plateau is PUBLISHED, not just counted: when it sits, at which
        # pitch, and the SIGNED interval to the label — a +2 st plateau is a
        # note sung too high and glued in, a -2 st one a low approach; the
        # sign is what the singer can act on
        seg_times = times[inside]
        plateau_midi = float(np.median(plateau))
        suspects.append(
            {
                "start_s": round(float(n.start), 3),
                "duration_s": round(float(n.duration), 3),
                "midi": int(n.midi),
                "inner_range_st": round(float(seg.max() - seg.min()), 2),
                "secondary_plateau_frames": int(best_len),
                "secondary_plateau_s": round(best_len * frame_dt, 2),
                "secondary_plateau_start_s": round(float(seg_times[best_start]), 3),
                "secondary_plateau_midi": int(round(plateau_midi)),
                "secondary_plateau_offset_st": round(plateau_midi - label, 2),
            }
        )
    return suspects


def coherence_report(observation, notes: list) -> dict:
    """The JSON block: how many notes were audited + the suspects, if any."""
    checked = sum(1 for n in notes if n.duration >= max(MIN_NOTE_S, 2 * PLATEAU_MIN_S))
    return {
        "checked": checked,
        "suspects": suspect_spans(observation, notes),
    }
