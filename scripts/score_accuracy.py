"""Diagnostic : que fait la transcription d'une VRAIE prise, et où ça coince ?

Usage :
    python scripts/score_accuracy.py prise.wav [--truth verite.json] [--json rapport.json]
                                              [--no-attacks] [--output DIR]

Sur une prise réelle il n'y a pas de vérité terrain : le script imprime donc
d'abord ce que le pipeline a trouvé (tempo, mesure, tonalité, accords, chaque
note avec sa figure et sa position), puis les SIGNES de découpage douteux, qui
sont ce qu'on regarde quand le résultat « sonne faux » :

- attaques orphelines     : une consonne détectée sans début de note — une
                           note est peut-être manquée ;
- débuts non articulés    : la note existe mais aucune attaque ne la date, le
                           début est donc celui de la stabilisation de hauteur ;
- fragments               : notes plus courtes que la confirmation (120 ms),
                           typiques d'un glissando coupé en deux ;
- comparaison attaques/géométrie : si les deux donnent le même nombre de
                           notes, la preuve d'attaque ne sert à rien ici (ou
                           l'enregistrement n'a pas de consonnes exploitables).

`--truth verite.json` compare à des notes connues, au format
`{"notes": [{"midi": 60, "start_s": 0.3, "dur_s": 0.6}, ...]}` (le même
harnais que les tests, `tests/voicetruth.py`).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import v2m
from v2m.core import audio_io
from v2m.core.i18n import tr_figure
from v2m.core.notes import ATTACK_TOL_S, MIN_UNCONFIRMED_S
from v2m.core.onsets import detect_attacks
from v2m.core.pipeline import AnalyzeOptions, analyze_signal
from v2m.core.report import export_result

DEFAULT_AUDIO = "demo.wav"
SKIP_PIPELINE_STAGES = ("décodage", "Terminé")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rapport de transcription d'une prise réelle")
    p.add_argument("audio", nargs="?", default=DEFAULT_AUDIO, help="fichier audio à analyser")
    p.add_argument("--truth", help="JSON de notes connues (vérité terrain)")
    p.add_argument("--json", dest="json_out", help="écrire le rapport en JSON")
    p.add_argument("--no-attacks", action="store_true", help="sans preuve d'attaque (géométrie seule)")
    p.add_argument("--output", help="dossier des artefacts exportés (défaut : temporaire)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Analyse
# --------------------------------------------------------------------------- #


def analyze(path: str, use_attacks: bool, out_dir: Path):
    """Analyse le fichier et renvoie (résultat, attaques détectées)."""
    y = audio_io.load_file(path)
    opts = AnalyzeOptions(use_attacks=use_attacks)
    result = analyze_signal(y, audio_io.SR, source=path, opts=opts)
    found = detect_attacks(y, audio_io.SR)
    export_result(result, str(out_dir / Path(path).stem))
    return result, found


def rests_line(result) -> str:
    """Les silences écrits entre les phrases (respirations, pauses)."""
    from collections import Counter

    rests = getattr(result, "rests", None) or []
    if not rests:
        return "  respirations : aucune (notes collées, ou écarts < 70 ms)"
    counts = Counter(r.figure or "hors grille" for r in rests)
    brief = ", ".join(f"{tr_figure(f)} ×{c}" for f, c in counts.items())
    times = ", ".join(f"{r.start:.2f}s ({r.duration:.2f}s)" for r in rests[:8])
    return f"  respirations : {len(rests)} silence(s) écrit(s) — {brief}\n               {times}"


def diagnostics(result, attacks, attacks_used: bool) -> dict:
    """Ce qui, dans le découpage, mérite un coup d'œil."""
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


def notes_table(result) -> list[str]:
    """Une ligne par note : hauteur, temps, figure, position, force."""
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
    """Compare à des notes connues, avec le harnais des tests."""
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
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(args.audio).exists():
        print(f"Fichier introuvable : {args.audio}")
        return 2
    out_dir = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="v2m_score_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    use_attacks = not args.no_attacks
    result, attacks = analyze(args.audio, use_attacks=use_attacks, out_dir=out_dir)
    diag = diagnostics(result, attacks, attacks_used=use_attacks)
    print(report(result, diag, attacks_used=use_attacks))

    if args.truth:
        print("")
        print("Vérité terrain")
        print("--------------")
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
        print(f"\nRapport JSON : {args.json_out}")
    print(f"Artefacts    : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
