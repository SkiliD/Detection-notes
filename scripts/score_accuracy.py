"""Diagnose what the transcription does to a REAL take, and where it strains.

Usage:
    python scripts/score_accuracy.py take.wav [--truth truth.json] [--json report.json]
                                              [--no-attacks] [--compare] [--golden]
                                              [--diagnose] [--output DIR]

A real take has no ground truth, so the script first prints what the
pipeline found (tempo, meter, key, chords, every note with its figure and
position), then the SIGNS of doubtful segmentation — what you look at when
the result "sounds wrong":

- orphan attacks        : a consonant detected with no note start — a note
                          may be missing;
- unarticulated starts  : the note exists but no attack dates it, so its
                          start is the pitch's stabilization point;
- fragments             : notes shorter than confirmation (120 ms),
                          typical of a glissando cut in two;
- attacks-vs-geometry   : when both give the same note count, the attack
                          evidence adds nothing here (or the recording has
                          no exploitable consonants).

`--truth truth.json` compares against known notes, in the format
`{"notes": [{"midi": 60, "start_s": 0.3, "dur_s": 0.6}, ...]}` (the same
harness as the tests, `tests/voicetruth.py`).

`--compare` replays the analysis a second time (same file, same engine,
decoding included, no artifacts exported) and compares the segmentation
numbers — tempo, meter, key, every note, every rest: any difference is a
determinism defect, and the exit code becomes 1.

`--diagnose` (a THRESHOLDS audit, orthogonal to --compare) prints which
ranges the take actually exercised, so a threshold is never recalibrated
on a take that does not test it: curve pitch range (cleaned raw f0) vs the
reading bounds (65 Hz..2100 Hz); normalized attack strength (0..1,
deciles) vs MIN_STRENGTH; the absorbed tuning offset (cents); residual
pitch deviations by deviation — the bands that decide the key snap
(AMBIGUOUS_DEV_ST). And the LIVE path replayed in stream: real display
latency (confirm_s), the note in progress, and where real time is spent.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import v2m
from v2m.core import audio_io
from v2m.core.audit import (
    compare_fingerprints,
    golden_path,
    load_golden,
    replay_fingerprint,
    save_golden,
    segmentation_fingerprint,
)
from v2m.core.i18n import tr_figure
from v2m.core.notes import ATTACK_TOL_S, MIN_UNCONFIRMED_S
from v2m.core.onsets import detect_attacks
from v2m.core.pipeline import AnalyzeOptions, analyze_signal
from v2m.core.report import export_result

DEFAULT_AUDIO = "demo.wav"
SKIP_PIPELINE_STAGES = ("decoding", "Done")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcription report for a real take")
    p.add_argument("audio", nargs="?", default=DEFAULT_AUDIO, help="audio file to analyze")
    p.add_argument("--truth", help="JSON of known notes (ground truth)")
    p.add_argument("--json", dest="json_out", help="write the report as JSON")
    p.add_argument("--no-attacks", action="store_true", help="without attack evidence (geometry only)")
    p.add_argument("--compare", action="store_true",
                   help="replay the analysis and check the segmentation numbers' stability")
    p.add_argument("--golden", action="store_true",
                   help="save the segmentation fingerprint to <audio>.golden.json")
    p.add_argument("--diagnose", action="store_true",
                   help="thresholds audit: pitch ranges, strengths, deviations, the live path")
    p.add_argument("--output", help="folder for the exported artifacts (default: temporary)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def analyze(path: str, use_attacks: bool, out_dir: Path):
    """Analyze the file and return (result, detected attacks)."""
    y = audio_io.load_file(path)
    result = analyze_signal(y, audio_io.SR, source=path, opts=AnalyzeOptions(use_attacks=use_attacks))
    found = detect_attacks(y, audio_io.SR)
    export_result(result, str(out_dir / Path(path).stem))
    return result, found


def rests_line(result) -> str:
    """The rests written between the phrases (breaths, pauses)."""
    from collections import Counter

    rests = getattr(result, "rests", None) or []
    if not rests:
        return "  breaths: none (glued notes, or gaps < 70 ms)"
    counts = Counter(r.figure or "off grid" for r in rests)
    brief = ", ".join(f"{tr_figure(f)} ×{c}" for f, c in counts.items())
    times = ", ".join(f"{r.start:.2f}s ({r.duration:.2f}s)" for r in rests[:8])
    return f"  breaths: {len(rests)} rest(s) written — {brief}\n            {times}"


def diagnostics(result, attacks, attacks_used: bool) -> dict:
    """What in the segmentation deserves a look."""
    onsets = [n.start for n in result.notes]
    articulated = sum(1 for t in onsets if attacks.near(t, ATTACK_TOL_S) is not None)
    orphans = [
        float(t)
        for t in attacks.times
        if not any(abs(t - o) <= ATTACK_TOL_S for o in onsets)
    ]
    fragments = [
        {"midi": n.midi, "start": round(n.start, 3), "duration": round(n.duration, 3)}
        for n in result.notes
        if n.duration < MIN_UNCONFIRMED_S
    ]
    rests = getattr(result, "rests", None) or []
    return {
        "rests": [
            {
                "start": round(r.start, 3),
                "duration": round(r.duration, 3),
                "figure": r.figure,
            }
            for r in rests
        ],
        "notes": len(result.notes),
        "attacks": len(attacks),
        "articulated_onsets": articulated,
        "unarticulated_onsets": len(onsets) - articulated,
        "orphan_attacks": [round(t, 3) for t in orphans],
        "fragments": fragments,
        "attacks_used": attacks_used,
    }


# --------------------------------------------------------------------------- #
# --diagnose: which threshold ranges did the take actually exercise?          #
# --------------------------------------------------------------------------- #


def _deciles(values) -> list[float]:
    """A series' deciles, rounded (empty -> [])."""
    return [round(float(v), 3) for v in np.percentile(values, np.arange(0, 101, 10))] if len(values) else []


def diagnose_thresholds(result) -> dict:
    """The ranges this take actually exercised, threshold by threshold.

    Reason to be: recalibrating a threshold on a take that does NOT test it
    (too-high a pitch range, no weak attacks, no ambiguous deviation) is
    the most common misstep — the diagnosis makes it visible.
    `result.pitch.transition` carries the derivative mask.
    """
    from v2m.core.notes import AMBIGUOUS_DEV_ST, _measure_tuning_offset
    from v2m.core.onsets import MIN_STRENGTH as ONSET_MIN_STRENGTH
    from v2m.core.pitch import FMAX, FMIN

    track = result.pitch
    curve_hz = track.hz if track is not None else np.zeros(0)
    voiced_hz = curve_hz[~np.isnan(curve_hz)]
    transitions = getattr(track, "transition", None)
    if transitions is not None and track is not None:
        voiced_ok = ~np.isnan(track.hz)
        trans_share = float(np.mean(transitions[voiced_ok])) if voiced_ok.any() else 0.0
    else:
        trans_share = 0.0
    deviations = [float(getattr(n, "deviation", 0.0)) for n in result.notes]
    strengths = [float(getattr(n, "strength", 0.0)) for n in result.notes]
    tuning = _measure_tuning_offset(
        [float(n.midi) + float(getattr(n, "deviation", 0.0)) for n in result.notes]
    )
    return {
        "pitch_range_hz": [round(float(voiced_hz.min()), 1), round(float(voiced_hz.max()), 1)] if len(voiced_hz) else None,
        "reading_bounds_hz": [FMIN, FMAX],
        "transition_share": round(trans_share, 3),
        "tuning_offset_cents": round(100.0 * tuning, 1),
        "strengths_deciles": _deciles(strengths),
        "strength_floor": ONSET_MIN_STRENGTH,
        "weak_attacks_under_floor": sum(1 for s in strengths if s < ONSET_MIN_STRENGTH),
        "deviations_deciles": _deciles(deviations),
        "ambiguous_dev_threshold": AMBIGUOUS_DEV_ST,
        "notes_in_ambiguous_band": sum(1 for d in deviations if abs(d) >= AMBIGUOUS_DEV_ST),
    }


def live_diagnosis(path: str) -> dict:
    """The real-time path, replayed in stream: real display latency.

    The measured latency is (first moment the held note is DISPLAYED -
    take start) — confirm_s plus the hold window's fill, on the real
    block stream. Also returns the live note count and the live/final
    DIAGNOSIS: the contour count and the ONE policy's breakdown
    (`core.compare` — the same net as the GUI marking and the JSON
    block), not a local rule.
    """
    from v2m.core.compare import breakdown
    from v2m.core.liveseg import LivePitchStream
    from v2m.core.pipeline import analyze_signal as _analyze

    sr = audio_io.SR
    y = audio_io.load_file(path)
    stream = LivePitchStream(sr, gate_rms=audio_io.GATE_RMS)
    first_on = None
    for i in range(0, len(y) - 1024, 1024):
        t_end = (i + 1024) / sr
        stream.feed(y[i : i + 1024], t_end)
        if first_on is None and stream.seg.state == "ACTIVE":
            first_on = t_end
    stream.seg.flush(len(y) / sr)
    result = _analyze(y, sr, source=path, opts=AnalyzeOptions())
    block = breakdown(stream.seg.notes, result.notes, result.grid)
    return {
        "notes": len(stream.seg.notes),
        "first_active_s": round(first_on, 3) if first_on is not None else None,
        "confirm_s": stream.seg.confirm_s,
        "contours": block["contours"],
        "off1": block["off1"],
        "moved": block["moved"],
        "rate": block["rate"],
    }


def notes_table(result) -> list[str]:
    """One line per note: pitch, time, figure, position, strength."""
    lines = []
    positions = result.note_positions or []
    for i, n in enumerate(result.notes):
        pos = positions[i] if i < len(positions) else None
        place = f"{pos[0]}/{pos[1]:.1f}" if pos else "-"
        figure = tr_figure(n.figure) if n.figure else "hors grille"
        name = f"{n.midi}"
        lines.append(
            f"  {i + 1:3d}  {name:>4}  {n.start:7.3f}  {n.duration:6.3f}  "
            f"{figure:<16}  {place:>8}  {n.strength:.2f}"
        )
    return lines


def summary(result) -> list[str]:
    tempo = f"{result.tempo.bpm:.1f} BPM" if result.tempo else "non détecté"
    meter = result.meter.label if result.meter else "4/4 (défaut)"
    key = result.key.best.label if result.key else "non détectée"
    chord = result.chord.best.label if result.chord else "non détecté"
    return [
        f"  audio      : {result.source}  ({result.duration_s:.1f} s)",
        f"  tempo      : {tempo}",
        f"  mesure     : {meter}  ({result.num_measures} mesures)",
        f"  tonalité   : {key}",
        f"  accord     : {chord}",
        f"  notes      : {len(result.notes)}",
    ]


def report(result, diag: dict, attacks_used: bool, table: bool = True) -> str:
    out = ["", "Analyse", "-------", *summary(result), "", "Découpage", "---------"]
    out.append(
        f"  attaques   : {diag['attacks']} "
        f"(utilisées : {'oui' if attacks_used else 'non'})"
    )
    out.append(
        f"  débuts     : {diag['articulated_onsets']} articulés, "
        f"{diag['unarticulated_onsets']} non (datés par la stabilisation)"
    )
    out.append(f"  orphelines : {len(diag['orphan_attacks'])} {diag['orphan_attacks'][:12]}")
    out.append(f"  fragments  : {len(diag['fragments'])} {diag['fragments'][:12]}")
    out.append(rests_line(result))
    if table and result.notes:
        out += [
            "",
            "  #   midi    début   durée   figure            mesure  force",
            "  --- ---- -------- ------- ----------------- -------- -----",
        ]
        out += notes_table(result)
    return "\n".join(out)


def truth_accuracy(path: str, truth_path: str) -> str:
    """Compare against known notes, with the tests' harness."""
    root = Path(v2m.__file__).resolve().parents[2]
    if str(root / "tests") not in sys.path:
        sys.path.insert(0, str(root / "tests"))
    from voicetruth import SungNote, detect_notes, describe, measure  # test helper

    payload = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    truth = [
        SungNote(int(n["midi"]), float(n["start_s"]), float(n["dur_s"]))
        for n in payload["notes"]
    ]
    notes = detect_notes(audio_io.load_file(path), audio_io.SR)
    return "  " + describe(measure(notes, truth))


# --------------------------------------------------------------------------- #
# The audit engine (fingerprints, golden I/O, comparison) lives in
# `v2m.core.audit` — the FROZEN format is core, the script only prints
# it. Imported at the top like every other v2m piece.
# --------------------------------------------------------------------------- #



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(args.audio).exists():
        print(f"File not found: {args.audio}")
        return 2
    out_dir = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="v2m_score_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    use_attacks = not args.no_attacks
    result, attacks = analyze(args.audio, use_attacks=use_attacks, out_dir=out_dir)
    diag = diagnostics(result, attacks, attacks_used=use_attacks)
    print(report(result, diag, attacks_used=use_attacks))

    if args.diagnose:
        d = diagnose_thresholds(result)
        d["live"] = live_diagnosis(args.audio)
        print("")
        print("Thresholds diagnosis")
        print("--------------------")
        pr = d["pitch_range_hz"]
        bounds = d["reading_bounds_hz"]
        print(
            f"  hauteur courbe   : {pr[0]}-{pr[1]} Hz (bornes de lecture {bounds[0]}-{bounds[1]} Hz)"
            if pr else "  hauteur courbe   : aucune trame voisée"
        )
        print(f"  trames transition: {d['transition_share'] * 100:.1f} % (dérivée f0)")
        print(f"  accordage absorbé: {d['tuning_offset_cents']:+.1f} cents (offset global)")
        st = d["strengths_deciles"]
        print(f"  forces d'attaque : déciles {st} (plancher {d['strength_floor']})")
        print(f"  attaques faibles : {d['weak_attacks_under_floor']} sous le plancher")
        dv = d["deviations_deciles"]
        print(f"  déviations pitch : déciles {dv} (seuil ambigu ±{d['ambiguous_dev_threshold']})")
        print(f"  notes ambiguës   : {d['notes_in_ambiguous_band']} dans la bande de collage")
        lv = d["live"]
        print(
            f"  chemin live      : {lv['notes']} notes, confirm_s={lv['confirm_s']}, "
            f"première note active à {lv['first_active_s']} s"
        )
        print(
            f"  live/final       : contours {lv['contours']} "
            f"(±1: {lv['off1']}, déplacées: {lv['moved']}), taux {lv['rate']*100:.0f}%"
        )

    stable = True
    if args.compare:
        replay = replay_fingerprint(args.audio, use_attacks)
        diffs = compare_fingerprints(segmentation_fingerprint(result), replay)
        stable = not diffs
        print("")
        print("Stability (replayed analysis)")
        print("-----------------------------")
        print("  segmentation identical on both passes" if stable else "\n".join(diffs))

    if args.golden:
        golden = load_golden(args.audio)
        current = segmentation_fingerprint(result)
        if golden is None:
            # first visit: TODAY's fingerprint becomes the reference
            save_golden(args.audio, current)
            print("")
            print(f"Golden fingerprint created: {golden_path(args.audio)}")
        else:
            diffs = compare_fingerprints(golden, current)
            stable &= not diffs
            print("")
            print("Fidelity (against the golden fingerprint)")
            print("-----------------------------------------")
            print("  segmentation identical to the reference" if not diffs else "\n".join(diffs))

    if args.truth:
        print("")
        print("Ground truth")
        print("------------")
        print(truth_accuracy(args.audio, args.truth))

    payload = {
        "source": result.source,
        "duration_s": round(result.duration_s, 3),
        "tempo_bpm": result.tempo.bpm if result.tempo else None,
        "meter": result.meter.label if result.meter else None,
        "key": result.key.best.label if result.key else None,
        "diagnostics": diag,
        "notes": [
            {
                "midi": n.midi,
                "start": round(n.start, 4),
                "duration": round(n.duration, 4),
                "figure": n.figure,
                "strength": n.strength,
            }
            for n in result.notes
        ],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON report: {args.json_out}")
    print(f"Artifacts    : {out_dir}")
    # the exit code makes drift visible to CI: an unstable golden fingerprint
    # or double pass => 1, else 0.
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
