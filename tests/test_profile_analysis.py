"""Le diagnostic de profilage doit rester branché sur le vrai pipeline.

Le script ne réimplémente pas les étapes : il les lit dans le callback
`progress` d'`analyze_signal`. Ce test compare donc les libellés RÉELLEMENT
publiés par le pipeline à ceux que le script affiche — si une étape est
renommée ou ajoutée, le tableau suit (et le test le vérifie), sans qu'aucune
liste ne soit à maintenir ici.

Un seul appel au script couvre le cas riche (moyenne + cProfile + JSON) :
chaque exécution paie l'interpréteur et la compilation numba, la suite reste
à quelques secondes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, silence, sine_note
from v2m.core.audio_io import SR as ANALYSIS_SR
from v2m.core.pipeline import analyze_signal

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_analysis.py"


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
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(SCRIPT.parents[1]),
    )


def _pipeline_stages(wav: str) -> list[str]:
    """Étapes annoncées par le pipeline sur ce fichier, closing callback exclu.

    Le dernier appel (fraction 1.0, « Terminé ») ne fait que fermer la
    dernière étape : le chronomètre le laisse tomber, d'où `[:-1]`.
    """
    labels: list[str] = []
    y, _ = sf.read(wav, dtype="float32")
    analyze_signal(y, ANALYSIS_SR, progress=lambda f, m: labels.append(m))
    assert labels[-1] == "Terminé"
    return labels[:-1]


def test_table_json_and_profiler(tmp_path):
    wav = _wav(tmp_path)
    report = tmp_path / "mesures.json"
    proc = _run(wav, "--repeat", "2", "--profile", "--top", "3", "--json", str(report))
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout

    # tableau des étapes : toutes celles du pipeline, une ligne chacune
    assert "Étape" in out and "TOTAL" in out and "Étape la plus lourde" in out
    assert "décodage du fichier" in out
    assert "export (4 artefacts" in out
    assert "passage à blanc" in out, "le warm-up par défaut doit être annoncé"
    assert "moyenne sur 2 exécutions" in out
    # chaque étape occupe UNE ligne du tableau — la ligne récapitulative
    # « Étape la plus lourde : <label> » répète un libellé à dessein, elle
    # ne compte pas comme une seconde étape (flake de timing sinon : selon
    # la machine, n'importe quelle étape peut être la plus lourde).
    body_lines = [
        line for line in out.splitlines() if not line.startswith("Étape la plus lourde")
    ]
    for label in _pipeline_stages(wav):
        assert sum(label in line for line in body_lines) == 1, (
            f"étape absente ou dupliquée : {label}"
        )
    assert "Terminé" not in out  # closing callback: never a row
    assert "20%" in out  # les fractions viennent du callback, pas d'une copie

    # cProfile : code du projet d'un côté, bibliothèques de l'autre
    assert "Code du projet" in out and "Bibliothèques" in out

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["runs"] == 2
    assert data["audio_s"] == pytest.approx(1.5, abs=0.05)
    assert data["n_notes"] >= 3
    assert data["total_s"] > 0
    stages = data["stages"]
    assert stages[0]["label"] == "décodage du fichier"
    # >= 0 : une étape quasi gratuite (détection des respirations) peut
    # mesurer sous la résolution d'écriture du JSON — 0.0 est légitime.
    assert all(row["seconds"] >= 0 for row in stages)
    assert sum(row["share_pct"] for row in stages) == pytest.approx(100.0, abs=0.5)
    assert any(row["fraction"] is not None for row in stages)
    assert data["profile"] and isinstance(data["profile"][0]["project_code"], bool)


def test_bad_inputs_exit_cleanly(tmp_path):
    missing = _run(str(tmp_path / "absent.wav"))
    assert missing.returncode == 1
    assert "introuvable" in missing.stderr

    text_file = tmp_path / "notes.txt"
    text_file.write_text("pas de l'audio", encoding="utf-8")
    wrong = _run(str(text_file))
    assert wrong.returncode == 1
    assert "Format non pris en charge" in wrong.stderr
