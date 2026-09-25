"""Artefacts de référence : la sortie complète de deux prises, figée.

Avant d'unifier les conversions secondes ↔ temps (une Grille, voir CONTEXT.md),
il faut un filet qui voie le moindre décalage du résultat d'Analyse. Ce fichier
rejoue les deux prises présentes dans la copie de travail — `chant.wav`, une
vraie prise chantée, et `demo.wav` — et compare la partition, le rapport et le
MIDI, octet par octet, à `tests/reference/artifacts.json`.

C'est un filet sur les SEPT consommateurs à la fois : la partition prouve les
figures, les mesures, les silences, l'armure et les positions ; le rapport
prouve les compteurs et les positions ; le MIDI prouve la quantification.

Un second test interroge la partition pour elle-même, sans référence : chaque
mesure doit totaliser la durée de sa signature et le `<type>` de chaque note
vaut exactement sa `<duration>` — deux chiffres qui viennent de chemins
différents (l'analyse, puis le catalogue des valeurs écrites).

Deux précautions assumées :

- la référence est liée aux fichiers audio, qui sont gitignorés : le test est
  ignoré quand ils manquent (autre poste, conteneur) plutôt que d'échouer ;
- re-geler la référence est une décision, pas un réflexe : lancer
  `V2M_FREEZE_REFERENCE=1 ./.venv/Scripts/python.exe -m pytest
  tests/test_reference_artifacts.py`, après avoir VU ce qui a bougé (le test
  écrit les artefacts frais dans son dossier temporaire et en donne le chemin).

Si `demo.wav` est régénéré avec une autre recette, ou `chant.wav` remplacé,
c'est un nouvel étalon : re-geler est alors correct.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from scorexml import assert_measures_close, score_root

from v2m.core import audio_io
from v2m.core.pipeline import AnalyzeOptions, analyze_signal
from v2m.core.report import export_result

REFERENCE = Path(__file__).parent / "reference" / "artifacts.json"
ROOT = Path(__file__).parents[1]
TAKES = ("chant.wav", "demo.wav")
ARTIFACTS = ("score.musicxml", "score.json", "score.mid")
FREEZE = os.environ.get("V2M_FREEZE_REFERENCE") == "1"


def _analyse(take: Path, out_dir: Path) -> dict:
    """Analyse une prise, écrit ses artefacts, renvoie compteurs + empreintes.

    `source` est le NOM du fichier (pas son chemin) : le rapport JSON le
    recopie, donc la référence ne doit pas dépendre de l'endroit d'où le test
    est lancé.
    """
    res = analyze_signal(
        audio_io.load_file(str(take)), audio_io.SR, source=take.name, opts=AnalyzeOptions()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    export_result(res, str(out_dir / "score"))
    return {
        "n_notes": len(res.notes),
        "n_rests": len(res.rests or []),
        "tempo_bpm": res.tempo.bpm if res.tempo else 0.0,
        "sha256": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in ARTIFACTS
        },
    }


def _moved(want: dict, got: dict) -> str:
    """Ce qui a bougé dans les compteurs, pour un message de panne lisible."""
    return ", ".join(
        f"{key} {want[key]} → {got[key]}"
        for key in ("n_notes", "n_rests", "tempo_bpm")
        if want[key] != got[key]
    )


def test_the_reference_scores_fill_their_measures_and_agree_with_their_types(tmp_path):
    """Une partition doit se refermer : chaque mesure totalise la durée annoncée
    par sa signature, et le `<type>` de chaque note vaut exactement sa
    `<duration>` — sur les vraies prises, où les deux chiffres viennent de
    chemins différents (l'analyse pour la durée, le catalogue pour le type).
    Le chemin arrimé est le seul concerné ; `snap=False` n'est qu'un crochet
    de test dans `test_rests.py`."""
    present = [(name, ROOT / name) for name in TAKES if (ROOT / name).exists()]
    if not present:
        pytest.skip(f"aucune prise de référence dans {ROOT}")

    for name, take in present:
        _analyse(take, tmp_path / name)
        xml = (tmp_path / name / "score.musicxml").read_text(encoding="utf-8")
        assert_measures_close(score_root(xml), name)


def test_the_reference_takes_still_export_the_same_bytes(tmp_path):
    present = [(name, ROOT / name) for name in TAKES if (ROOT / name).exists()]
    if not present:
        pytest.skip(f"aucune prise de référence dans {ROOT}")

    fresh = {name: _analyse(take, tmp_path / name) for name, take in present}

    if FREEZE:
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(
            json.dumps(fresh, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pytest.skip(f"référence re-gelée : {REFERENCE}")

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for name, got in fresh.items():
        want = reference.get(name)
        assert want is not None, f"{name} absent de {REFERENCE}"
        assert got["sha256"] == want["sha256"], (
            f"{name} : la sortie a bougé ({_moved(want, got) or 'mêmes compteurs, octets différents'})"
            f" — artefacts frais à comparer : {tmp_path / name}"
        )
