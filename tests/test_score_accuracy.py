"""Le mode --compare/--golden du diagnostic rejoue la MÊME analyse.

Le script rejoue la segmentation entière (décodage compris) et compare
l'empreinte des chiffres : ce test vérifie le cas stable sur un audio
synthétique (déterminisme = zéro différence, sortie 0), puis que la
comparaison MORDE — un script qui comparerait mal (toujours « stable »)
ferait moins que rien. Même choix que test_profile_analysis : le vrai
script est exécuté, jamais copié.

Les prises RÉELLES du dépôt (demo.wav, chant.wav — gitignorées) sont
rejouées contre leur empreinte dorée quand elles existent : une dérive
de segmentation sur une vraie prise est alors vue par la suite intégrale,
pas seulement quand on y pense. Ces rejeux coûtent ~16 s sur ~47 s
(~34 % de la suite — mesure de 2026-09-23) : ils sont marqués `slow`
(désélection avec `-m "not slow"` pour une boucle rapide), mais restent
DANS la suite par défaut — un filet qui ne tourne plus n'est pas un filet.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, silence, sine_note

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_accuracy.py"


def _wav(tmp_path) -> str:
    parts: list[np.ndarray] = []
    for hz in (220.0, 261.63, 329.63):
        parts.append(sine_note(hz, 0.4))
        parts.append(silence(0.1))
    path = tmp_path / "melody.wav"
    sf.write(str(path), np.concatenate(parts), SR)
    return str(path)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )


# --------------------------------------------------------------------------- #
# --diagnose : l'audit des seuils réellement exercés
# --------------------------------------------------------------------------- #


def test_diagnose_reports_the_thresholds_a_take_exercised(tmp_path):
    """Le mode --diagnose imprime les plages de la prise : hauteur courbe
    vs bornes de lecture, offset d'accordage absorbé, forces d'attaque (et
    combien sous le plancher), déviations ambiguës, et le chemin live.
    Chaque section annoncée doit exister dans la sortie — un audit qui ne
    dit rien ne diagnostique rien."""
    wav = _wav(tmp_path)
    proc = _run(wav, "--diagnose")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    # ASCII-safe substrings only: the subprocess console encoding (cp1252
    # on Windows) mangles accents in the captured bytes, not the content
    assert "Diagnostic" in out
    assert "65.0-2100.0" in out  # the reading bounds, FMIN-FMAX
    assert "confirm_s=0.09" in out  # the live path replayed
    assert "offset" in out or "accordage" in out
    assert "0.2" in out  # the strength floor appears in the report


@pytest.mark.slow
def test_golden_saves_then_detects_the_same_take(tmp_path):
    """Première visite : l'empreinte est créée (fichier à côté de l'audio).
    Seconde visite : identique, sortie 0 — l'audit est vert sur une prise
    stable.
    """
    wav = _wav(tmp_path)
    first = _run(wav, "--golden")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "créée" in first.stdout
    golden = tmp_path / "melody.golden.json"
    assert golden.exists()

    second = _run(wav, "--golden")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "identique à la référence" in second.stdout


@pytest.mark.slow
def test_golden_catches_a_drift(tmp_path):
    """L'audit MORDE : l'empreinte enregistrée ne peut pas être trompée par
    un moteur qui dérive (la segmentation de référence a changé) — sortie 1
    et la différence nommée.
    """
    wav = _wav(tmp_path)
    assert _run(wav, "--golden").returncode == 0
    golden = tmp_path / "melody.golden.json"
    data = json.loads(golden.read_text(encoding="utf-8"))
    data["tempo_bpm"] = 119.0
    data["notes"][0][0] = 0.37
    golden.write_text(json.dumps(data), encoding="utf-8")

    drifted = _run(wav, "--golden")
    assert drifted.returncode == 1, drifted.stdout
    assert "tempo_bpm" in drifted.stdout


def test_compare_reports_stable_segmentation(tmp_path):
    """Deux passes sur le même fichier : chiffres identiques, sortie 0."""
    proc = _run(_wav(tmp_path), "--compare")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "identique" in proc.stdout


def test_compare_fingerprints_bites_on_a_drift():
    """La comparaison rapporte TOUTE dérive — tempo, note, force, silence."""
    spec = importlib.util.spec_from_file_location("score_accuracy", SCRIPT)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    stable = {
        "tempo_bpm": 120.0,
        "meter": "4/4",
        "key": "C",
        "notes": [(0.0, 0.5, 60, "noire", 0.1)],
        "rests": [],
    }
    assert script.compare_fingerprints(stable, dict(stable)) == []

    drifted = dict(stable, tempo_bpm=119.0, notes=[(0.0, 0.5, 60, "noire", 0.9)])
    diffs = script.compare_fingerprints(stable, drifted)
    assert diffs and len(diffs) == 2  # le tempo ET la force de la note 1


@pytest.mark.slow
@pytest.mark.parametrize("take", ["demo.wav", "chant.wav"])
def test_real_takes_match_their_golden_fingerprint(take: str):
    """Les vraies prises du dépôt rejouent leur empreinte dorée (si elles y sont).

    demo.wav est régénérable (scripts/make_demo.py, voir README) et
    chant.wav est la prise de référence locale : les deux sont gitignorés,
    d'où le skip silencieux là où ils sont absents. Une empreinte dorée
    qui existe et dérive = échec dur : c'est le filet --golden branché sur
    la suite intégrale.
    """
    audio = Path(__file__).resolve().parents[1] / take
    if not audio.exists():
        pytest.skip(f"{take} absent (gitignore, régénérable) — empreinte non rejouée")
    proc = _run(str(audio), "--golden")
    if not (audio.parent / audio.with_suffix(".golden.json").name).exists():
        # première visite : le test vient de créer l'empreinte, rien à rejouer
        assert proc.returncode == 0, proc.stdout + proc.stderr
        pytest.skip(f"empreinte dorée de {take} créée à ce passage ; rejouer pour vérifier")
    assert proc.returncode == 0, "dérive de segmentation sur " + take + " :\n" + proc.stdout
