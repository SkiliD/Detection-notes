"""Reinterpret a take without re-decoding the audio or re-running pYIN.

Usage:
    python scripts/reinterpret.py take.wav [--save] [--observation obs.npz]
                                            [--min-conf F] [--min-note MS]
                                            [--semitones F] [--no-attacks]
                                            [--sweep V1,V2,...] [--sweep-note N1,N2,...]
                                            [--against obs2.npz]
                                            [-o DIR]

The pipeline now separates TWO things: the vocal EVIDENCE (cleaned pitch
curve, attacks, clock, accent envelope, tuning offset) and its musical
INTERPRETATION (notes, key, chords, grid, figures). The observation saves
to `.npz` and reloads identically: this script therefore re-reads the SAME
take with other thresholds in milliseconds and shows what moves.

- `--save` writes the computed observation to `<take>.observation.npz`; a
  later run with `--observation <file>` reads it back instead of re-running
  pYIN.
- `--min-conf`, `--min-note` (in milliseconds) and `--semitones` change the
  interpretation threshold, not the evidence: that is exactly what is
  measured here (the gap between two readings of one take).
- `--sweep V1,V2,...` sweeps several `--semitones` values (and
  `--sweep-note N1,N2,...` several `--min-note` values, in milliseconds)
  over the SAME observation: one line per combination (note count, tempo),
  the default starred — the table to choose a threshold knowingly, without
  re-running pYIN at each value.
- `--against obs2.npz` diffs TWO observations: the evidence itself first
  (curve sample by sample, attacks, clock — where the ENGINE changed, not
  only the reading), then the segmentation gap.
- `--no-attacks` ignores the attack evidence (geometry only).
- `-o` / `--output` re-exports the artifacts (score + report) to a folder;
  without it the script writes nothing but `--save`'s observation.

Without `-o`, nothing is written: the script is for LOOKING at the diff,
not overwriting an existing analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v2m.core import audio_io
from v2m.core.notes import DEFAULT_MIN_NOTE_S, DEFAULT_SEMITONES
from v2m.core.observation import DEFAULT_MIN_CONF, VoiceObservation
from v2m.core.pipeline import AnalyzeOptions, interpret_observation
from v2m.core.report import export_result
from v2m.core.tempo import BPM_RANGE, TempoResult, detect_tempo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reinterpret a vocal observation (no pYIN)"
    )
    p.add_argument("audio", nargs="?", help="audio take to reinterpret")
    p.add_argument(
        "--observation",
        help="pre-computed .npz observation (instead of decoding the audio)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="write the computed observation to <take>.observation.npz",
    )
    p.add_argument(
        "--min-conf",
        type=float,
        default=DEFAULT_MIN_CONF,
        help="minimum voicing confidence (cleaning mask)",
    )
    p.add_argument(
        "--min-note",
        type=float,
        default=DEFAULT_MIN_NOTE_S * 1000.0,
        help="minimum note duration, in milliseconds",
    )
    p.add_argument(
        "--semitones",
        type=float,
        default=DEFAULT_SEMITONES,
        help="a note's hysteresis band, in semitones",
    )
    p.add_argument(
        "--sweep",
        help="sweep several --semitones values (comma-separated)",
    )
    p.add_argument(
        "--sweep-note",
        help="sweep several --min-note values in ms (comma-separated)",
    )
    p.add_argument(
        "--against",
        help=" a second .npz observation: evidence diff then segmentation diff",
    )
    p.add_argument(
        "--no-attacks",
        action="store_true",
        help="ignore the attack evidence (geometry only)",
    )
    p.add_argument(
        "-o",
        "--output",
        help="folder for the re-exported artifacts (default: none)",
    )
    return p.parse_args(argv)


def observation_path(audio: str) -> Path:
    """A take's observation lives beside it: <take>.observation.npz."""
    return Path(audio).with_suffix(".observation.npz")


def source_stem(args: argparse.Namespace) -> str:
    """The common stem of the audio, the observation and the report."""
    if args.audio:
        return Path(args.audio).stem
    name = Path(args.observation).name
    suffix = ".observation.npz"
    return name[: -len(suffix)] if name.endswith(suffix) else Path(name).stem


def load_observation(args: argparse.Namespace) -> VoiceObservation:
    """The requested observation: read from a .npz, or rebuilt from the audio."""
    if args.observation:
        path = Path(args.observation)
        if not path.exists():
            raise FileNotFoundError(f"Observation not found: {path}")
        return VoiceObservation.load(path)
    y = audio_io.load_file(args.audio)
    return VoiceObservation.from_signal(
        y, audio_io.SR, min_conf=args.min_conf, use_attacks=not args.no_attacks
    )


def tempo_of(args: argparse.Namespace, stem: str, observation: VoiceObservation):
    """The input tempo: detected on the audio, else read from the existing report.

    Without audio or report, `None`: the interpretation can only produce
    notes (no grid, no figures) — the diff stays readable.
    """
    if args.audio:
        y = audio_io.load_file(args.audio)
        return detect_tempo(y, audio_io.SR, bpm_range=BPM_RANGE)
    report_path = Path(f"{stem}.json")
    if report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        bpm = (rep.get("tempo") or {}).get("bpm")
        if bpm:
            beats = np.arange(0.0, observation.duration_s, 60.0 / float(bpm))
            return TempoResult(bpm=float(bpm), beats=beats)
    return None


def notes_fingerprint(result) -> list[list]:
    """Every note as [start, duration, midi] — what decides if a reading moved."""
    return [
        [round(float(n.start), 4), round(float(n.duration), 4), int(n.midi)]
        for n in result.notes
    ]


def diff_notes(old: list[list], new: list[list]) -> list[str]:
    """The differences between two note readings (empty = identical)."""
    if len(old) != len(new):
        lines = [f"  note count: {len(old)} -> {len(new)}"]
    else:
        lines = []
    for i, (a, b) in enumerate(zip(old, new, strict=False)):
        if a != b:
            lines.append(f"  note {i + 1}: {a} -> {b}")
    return lines


# --------------------------------------------------------------------------- #
# --sweep: a threshold is chosen from the table, not by instinct
# --------------------------------------------------------------------------- #


def _floats(spec: str) -> list[float]:
    """Les valeurs d'un balayage « 0.4,0.5,0.6 » (une valeur mal lue = erreur)."""
    try:
        return [float(v.strip()) for v in spec.split(",") if v.strip()]
    except ValueError as exc:
        raise SystemExit(f"Valeur de balayage invalide : {spec} ({exc})") from exc


def interpret_at(
    observation: VoiceObservation,
    tempo: TempoResult | None,
    opts: AnalyzeOptions,
    source: str,
):
    """The interpretation at these thresholds — a sweep's only repeated computation."""
    return interpret_observation(
        observation, tempo=tempo, source=source, opts=opts, report=lambda f, m: None
    )


def sweep_table(
    rows: list[tuple[float, float, Any]], default_st: float, default_ms: float
) -> list[str]:
    """The sweep's table: one line per (semitones, min_note).

    `rows` carries (semitones, min_note_ms, result); the current
    calibration's line is starred. The tempo is an INDICATOR: thresholds
    that move it move the whole grid (the known 0.35 -> ~155 trap).
    """
    lines = [
        "  st     min_note  notes  tempo      stable boundary",
        "  ------ --------- ------ ---------- ----------------",
    ]
    for st, ms, result in rows:
        tempo = f"{result.tempo.bpm:.1f}" if result.tempo else "-"
        stable = "oui" if result.notes else "non"
        mark = "*" if (st == default_st and ms == default_ms) else " "
        lines.append(
            f"  {mark}{st:<5} {ms:>8.0f} {len(result.notes):>6} "
            f"{tempo:>10} {stable:>17}"
        )
    return lines


def run_sweep(args: argparse.Namespace, observation, tempo, source) -> int:
    """The sweep: same evidence, several readings, one table."""
    sts = _floats(args.sweep) if args.sweep else [args.semitones]
    mss = _floats(args.sweep_note) if args.sweep_note else [args.min_note]
    base = AnalyzeOptions(
        min_conf=args.min_conf,
        min_note_s=args.min_note / 1000.0,
        semitones=args.semitones,
        use_attacks=not args.no_attacks,
    )
    rows = []
    print("")
    print(
        f"Sweep over {source} — {len(sts)} x {len(mss)} readings, "
        "evidence unchanged"
    )
    print("-" * 60)
    for st in sts:
        for ms in mss:
            opts = AnalyzeOptions(
                min_conf=args.min_conf,
                min_note_s=ms / 1000.0,
                semitones=st,
                use_attacks=not args.no_attacks,
            )
            result = interpret_at(observation, tempo, opts, source)
            rows.append((st, ms, result))
    print("\n".join(sweep_table(rows, args.semitones, args.min_note)))
    print(
        "  (* = calibrage courant ; st = bande d'hystérésis en demi-tons, "
        "min_note en ms)"
    )
    del base
    return 0


# --------------------------------------------------------------------------- #
# --against: the EVIDENCE diff, not only the interpretation
# --------------------------------------------------------------------------- #


def evidence_diffs(a: VoiceObservation, b: VoiceObservation) -> list[str]:
    """Where TWO observations diverge, field by field.

    This is the diff `diff_notes` cannot make: two readings of the same
    evidence only move the interpretation where the thresholds decide, two
    different observations move the evidence itself (another engine,
    another version of the cleaning). The arrays are compared element by
    element when lengths agree, else by length only.
    """
    lines: list[str] = []
    if a.sample_rate != b.sample_rate:
        lines.append(f"  sample_rate : {a.sample_rate} -> {b.sample_rate}")
    if abs(a.duration_s - b.duration_s) > 1e-6:
        lines.append(f"  duration    : {a.duration_s:.3f} -> {b.duration_s:.3f} s")
    if abs(a.frame_dt - b.frame_dt) > 1e-9:
        lines.append(f"  frame_dt    : {a.frame_dt:.6f} -> {b.frame_dt:.6f} s")
    if abs(a.tuning_offset_st - b.tuning_offset_st) > 1e-9:
        lines.append(
            f"  tuning      : {100.0 * a.tuning_offset_st:+.1f} -> "
            f"{100.0 * b.tuning_offset_st:+.1f} cents"
        )
    ta, tb = a.pitch.times, b.pitch.times
    if len(ta) != len(tb):
        lines.append(f"  frames      : {len(ta)} -> {len(tb)}")
    elif ta.size:
        # the two clocks agree: a sample-by-sample comparison means
        # something (else it would mean nothing)
        ha, hb = a.pitch.hz, b.pitch.hz
        both = ~np.isnan(ha) & ~np.isnan(hb)
        moved = np.zeros(ha.shape, dtype=bool)
        voiced_a = ~np.isnan(ha)
        voiced_b = ~np.isnan(hb)
        moved |= voiced_a != voiced_b
        moved[both] = np.abs(ha[both] - hb[both]) / hb[both] > 1e-4  # > 0.0017 st
        share = float(np.mean(moved)) if moved.size else 0.0
        if share > 0.0:
            lines.append(
                f"  courbe      : {share * 100:.1f} % des trames divergent"
                f" ({int(moved.sum())}/{moved.size})"
            )
    else:
        lines.append("  courbe      : sans trame d'un côté, rien à comparer")
    na = len(a.attacks) if a.attacks is not None else None
    nb = len(b.attacks) if b.attacks is not None else None
    if (na is None) != (nb is None):
        lines.append(
            f"  attaques    : {'présentes' if na is not None else 'absentes'} -> "
            f"{'présentes' if nb is not None else 'absentes'}"
        )
    elif na is not None and nb is not None and na != nb:
        lines.append(f"  attaques    : {na} -> {nb}")
    if a.accent_envelope is None or b.accent_envelope is None:
        if (a.accent_envelope is None) != (b.accent_envelope is None):
            lines.append("  enveloppe   : présente -> absente")
    elif len(a.accent_envelope) != len(b.accent_envelope):
        lines.append(
            f"  enveloppe   : {len(a.accent_envelope)} -> "
            f"{len(b.accent_envelope)} valeurs"
        )
    else:
        delta = np.abs(a.accent_envelope - b.accent_envelope)
        if delta.size and float(np.max(delta)) > 1e-6:
            lines.append(
                f"  enveloppe   : écart max {float(np.max(delta)):.4f} "
                f"(moyenne {float(np.mean(delta)):.4f})"
            )
    return lines


def run_against(
    args: argparse.Namespace, observation: VoiceObservation, tempo, source: str
) -> int:
    """The diff of two observations: the evidence first, the reading next."""
    other_path = Path(args.against)
    if not other_path.exists():
        print(f"Observation not found: {other_path}")
        return 2
    other = VoiceObservation.load(other_path)
    print("")
    print(f"Evidence: {source}  against  {args.against}")
    print("-" * 60)
    diffs = evidence_diffs(observation, other)
    print("  identical evidence (same proof)" if not diffs else "\n".join(diffs))
    if diffs:
        print("")
        print("  (the interpretations differ at least as much —")
        print("   a segmentation-only diff would have masked the cause)")
    print("")
    print("Segmentation")
    print("------------")
    opts = AnalyzeOptions(
        min_conf=args.min_conf,
        min_note_s=args.min_note / 1000.0,
        semitones=args.semitones,
        use_attacks=not args.no_attacks,
    )
    first = interpret_at(observation, tempo, opts, source)
    # the tempo is read back from the CURRENT STEM's report for both: one
    # grid for both readings, the comparison is about the segmentation
    other_tempo = tempo_of(args, source_stem(args), other)
    second = interpret_at(other, other_tempo, opts, str(args.against))
    diffs = diff_notes(notes_fingerprint(first), notes_fingerprint(second))
    print("  identical segmentation at equal thresholds" if not diffs else "\n".join(diffs))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.audio and not args.observation:
        print("Il faut une prise audio ou une observation (--observation).")
        return 2
    if args.audio and not Path(args.audio).exists():
        print(f"Fichier introuvable : {args.audio}")
        return 2

    stem = source_stem(args)
    try:
        observation = load_observation(args)
    except FileNotFoundError as exc:
        print(exc)
        return 2

    if args.save:
        target = Path(args.observation) if args.observation else observation_path(args.audio)
        observation.save(target)
        print(f"Observation écrite : {target}")

    tempo = tempo_of(args, stem, observation)
    source = args.audio or args.observation

    if args.against:
        return run_against(args, observation, tempo, source)
    if args.sweep or args.sweep_note:
        return run_sweep(args, observation, tempo, source)

    opts = AnalyzeOptions(
        min_conf=args.min_conf,
        min_note_s=args.min_note / 1000.0,
        semitones=args.semitones,
        use_attacks=not args.no_attacks,
    )
    result = interpret_at(observation, tempo, opts, source)

    print("")
    print("Réinterprétation")
    print("----------------")
    print(f"  source      : {source}  ({observation.duration_s:.1f} s)")
    print(
        f"  seuils      : min_conf={args.min_conf}, min_note={args.min_note:.0f} ms, "
        f"semitones={args.semitones}, attaques={'non' if args.no_attacks else 'oui'}"
    )
    tempo_line = f"{result.tempo.bpm:.1f} BPM" if result.tempo else "non détecté"
    print(f"  tempo       : {tempo_line}")
    print(f"  mesure      : {result.meter.label if result.meter else '4/4 (défaut)'}")
    print(f"  tonalité    : {result.key.best.label if result.key else 'non détectée'}")
    print(f"  notes       : {len(result.notes)}")

    report_path = Path(f"{stem}.json")
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding="utf-8"))
        old_notes = [
            [round(n["start_s"], 4), round(n["duration_s"], 4), int(n["midi"])]
            for n in old.get("notes", [])
        ]
        diffs = diff_notes(old_notes, notes_fingerprint(result))
        print("")
        print(f"Diff contre {report_path}")
        print("-" * (13 + len(str(report_path))))
        print("  lecture identique au rapport existant" if not diffs else "\n".join(diffs))
        old_tempo = (old.get("tempo") or {}).get("bpm")
        if old_tempo and result.tempo and abs(float(old_tempo) - result.tempo.bpm) >= 0.005:
            print(f"  tempo : {old_tempo} -> {result.tempo.bpm}")
    else:
        print(f"\nAucun rapport {report_path} : diff impossible (première lecture).")

    if args.output:
        if result.tempo is None or result.key is None:
            print("\nRéexport impossible : tempo ou tonalité manquants.")
            return 1
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        export_result(result, str(out_dir / stem))
        print(f"\nArtefacts : {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
