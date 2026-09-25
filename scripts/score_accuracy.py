"""Diagnostic : que fait la transcription d'une VRAIE prise, et où ça coince ?

Usage :
    python scripts/score_accuracy.py prise.wav [--truth verite.json] [--json rapport.json]
                                              [--no-attacks] [--compare] [--golden]
                                              [--diagnose] [--output DIR]

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

`--compare` rejoue l'analyse une seconde fois (même fichier, même moteur,
décodage inclus, sans exporter les artefacts) et compare les chiffres de
segmentation — tempo, mesure, tonalité, chaque note, chaque respiration :
toute différence est un défaut de déterminisme, et le code de sortie
passe à 1.

`--diagnose` (audit des SEUILS, orthogonal à --compare) imprime quelles
plages la prise a réellement exercées, pour ne jamais recalibrer un seuil
sur une prise qui ne le teste pas : plage de hauteur courbe (f0 brut
nettoyé) vs les bornes de lecture (65 Hz..2100 Hz) ; force des attaques
normalisée (0..1, déciles) vs MIN_STRENGTH ; décalage systématique de
l'accordage absorbé (cents) ; déviations de hauteurs résiduelles par
deviation — les tranches qui décident du collage à la gamme (AMBIGUOUS_DEV_ST).
Et le chemin LIVE rejoué en flux : latence d'affichage réelle (confirm_s),
note en cours, et où est-ce qui se passe réellement en temps réel.
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
    p.add_argument("--compare", action="store_true",
                   help="rejoue l'analyse et vérifie la stabilité des chiffres de segmentation")
    p.add_argument("--golden", action="store_true",
                   help="sauve l'empreinte de segmentation dans <audio>.golden.json")
    p.add_argument("--diagnose", action="store_true",
                   help="audit des seuils : plages de hauteur, forces, déviations, chemin live")
    p.add_argument("--output", help="dossier des artefacts exportés (défaut : temporaire)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Analyse
# --------------------------------------------------------------------------- #


def analyze(path: str, use_attacks: bool, out_dir: Path):
    """Analyse le fichier et renvoie (résultat, attaques détectées)."""
    y = audio_io.load_file(path)
    result = analyze_signal(y, audio_io.SR, source=path, opts=AnalyzeOptions(use_attacks=use_attacks))
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


# --------------------------------------------------------------------------- #
# --diagnose : quelles plages de seuils la prise a-t-elle exercées ?          #
# --------------------------------------------------------------------------- #


def _deciles(values) -> list[float]:
    """Les déciles d'une série, arrondis (vide -> [])."""
    return [round(float(v), 3) for v in np.percentile(values, np.arange(0, 101, 10))] if len(values) else []


def diagnose_thresholds(result) -> dict:
    """Les plages réellement exercées par cette prise, seuil par seuil.

    Raison d'être : recalibrer un seuil sur une prise qui ne le teste PAS
    (plage de hauteur trop haute, aucune attaque faible, aucune déviation
    ambiguë) est le faux pas le plus courant — le diagnostic le rend
    visible. `result.pitch.transition` porte le masque de dérivée.
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
    """Le chemin temps réel, rejoué en flux : latence d'affichage réelle.

    La latence mesurée est (premier moment où la note tenue est AFFICHÉE -
    début de la prise) — confirm_s + le remplissage de la fenêtre de hold,
    sur le flux de blocs réel. Renvoie aussi le nombre de notes live.
    """
    from v2m.core.liveseg import LivePitchStream

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
    return {
        "notes": len(stream.seg.notes),
        "first_active_s": round(first_on, 3) if first_on is not None else None,
        "confirm_s": stream.seg.confirm_s,
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
def segmentation_fingerprint(result) -> dict:
    """Les chiffres de segmentation qui doivent être stables d'une passe à l'autre.

    Les tuples deviennent des listes : l'empreinte part aussi en JSON
    (--golden) et doit survivre au round-trip json.dumps/json.loads.
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
    """L'empreinte dorée d'une prise vit à côté d'elle : <audio>.golden.json."""
    return Path(audio).with_suffix(".golden.json")


def load_golden(audio: str) -> dict | None:
    """L'empreinte enregistrée, None quand elle n'existe pas encore."""
    path = golden_path(audio)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def replay_fingerprint(path: str, use_attacks: bool) -> dict:
    """Seconde passe complète (re-décodage compris, aucun artefact écrit)."""
    y = audio_io.load_file(path)
    result = analyze_signal(y, audio_io.SR, source=path, opts=AnalyzeOptions(use_attacks=use_attacks))
    return segmentation_fingerprint(result)


def compare_fingerprints(first: dict, second: dict) -> list[str]:
    """Les différences entre deux passes, en lignes lisibles (vide = stable)."""
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

    if args.diagnose:
        d = diagnose_thresholds(result)
        d["live"] = live_diagnosis(args.audio)
        print("")
        print("Diagnostic des seuils")
        print("----------------------")
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

    stable = True
    if args.compare:
        replay = replay_fingerprint(args.audio, use_attacks)
        diffs = compare_fingerprints(segmentation_fingerprint(result), replay)
        stable = not diffs
        print("")
        print("Stabilité (analyse rejouée)")
        print("---------------------------")
        print("  segmentation identique sur les deux passes" if stable else "\n".join(diffs))

    if args.golden:
        golden = load_golden(args.audio)
        current = segmentation_fingerprint(result)
        if golden is None:
            # première visite : l'empreinte D'AUJOURD'HUI devient la référence
            golden_path(args.audio).write_text(json.dumps(current, indent=2), encoding="utf-8")
            print("")
            print(f"Empreinte dorée créée : {golden_path(args.audio)}")
        else:
            diffs = compare_fingerprints(golden, current)
            stable &= not diffs
            print("")
            print("Fidélité (comparée à l'empreinte dorée)")
            print("---------------------------------------")
            print("  segmentation identique à la référence" if not diffs else "\n".join(diffs))

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
    # le code de sortie rend la dérive visible pour CI : empreinte dorée ou
    # double passe instable => 1, sinon 0.
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
