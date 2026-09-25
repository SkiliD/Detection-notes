"""Fixtures : signaux audio synthétiques, langue sandboxée, budget de la suite."""

import time

import numpy as np
import pytest

from v2m.core import i18n

SR = 22_050


@pytest.fixture(scope="session", autouse=True)
def _sandbox_language_file(tmp_path_factory):
    """La suite ne lit ni n'écrit le ~/.v2m_lang de l'utilisateur.

    Régression : un `--lang en` manuel (ou le choix persisté d'un utilisateur
    anglophone) rendait la suite NON reproductible — `cli.main` sans --lang
    et MainWindow appellent `get_language()`, qui relisait ce fichier au
    milieu du run et basculait tout le process en anglais : les tests qui
    attendent du français (`match="décoder"`, les libellés d'étage de
    `Terminé`) échouaient selon l'état ambiant. On force aussi le français
    au démarrage : l'état de départ est le même partout, à chaque run.
    """
    original = i18n.LANG_FILE
    i18n.LANG_FILE = tmp_path_factory.mktemp("lang") / "lang"
    i18n.set_language("fr")
    yield
    i18n.LANG_FILE = original


# Contrat de coût (AGENTS.md) : la suite complète doit rester sous ce
# budget — un test lent ajouté au-delà doit être une décision EXPLICITE
# (mesurer, abaisser le budget avec une raison, ou marquer le test slow).
# Recalibré 2026-09-24 sur la vitesse mesurée du jour : suite complète
# 51.3 s (483 tests), marge d'environnement ×1.85 => 95 s. La procédure
# de calibration vit dans AGENTS.md ("Calibration du budget suite") : un
# budget qui dérape est d'abord une VITESSE de machine à comparer, pas un
# test à supprimer.
SUITE_BUDGET_S = 95.0

# Le début du coût : ce module se charge pendant la COLLECTION, avant le
# premier test (un hook pytest_sessionstart ici ne serait pas encore
# enregistré : les conftests se chargent après sessionstart).
_SUITE_STARTED_AT = time.perf_counter()


def suite_cost_verdict(elapsed_s: float, budget_s: float = SUITE_BUDGET_S) -> bool:
    """La décision seule, testable sans attendre la fin d'une suite réelle."""
    return elapsed_s > budget_s


def pytest_sessionfinish(session, exitstatus):
    """Affiche le coût total ; fait échouer une suite VERTE trop lente.

    `wrap_session` lit `session.exitstatus` APRÈS ce hook : muter l'attribut
    change le code de sortie, sans écraser un échec/une interruption déjà
    code (1/2/3 restent ce qu'ils sont — seul un run vert dérape en 1).
    """
    total = time.perf_counter() - _SUITE_STARTED_AT
    line = f"suite cost: {total:.1f} s (budget {SUITE_BUDGET_S:.0f} s)"
    if suite_cost_verdict(total):
        line += " — OVER BUDGET: mark the slow tests or raise SUITE_BUDGET_S on purpose"
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(line)
    else:  # --no-terminalreporter : the cost stays visible anyway
        print(line)
    if suite_cost_verdict(total) and session.exitstatus == 0:
        session.exitstatus = 1


def sine_note(hz: float, dur_s: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Sinusoïde avec enveloppe ADSR simplifiée (évite les clics)."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    y = amp * np.sin(2 * np.pi * hz * t)
    # enveloppe : 10 ms fondu entrée/sortie
    fade = int(0.01 * sr)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (y * env).astype(np.float32)


def silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(dur_s * sr), dtype=np.float32)


@pytest.fixture
def sr() -> int:
    return SR
