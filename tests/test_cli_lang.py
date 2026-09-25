"""La CLI parle la langue demandée — rendu réel vérifié, pas espéré.

Deux contrats, deux langues :

1. **Le résumé d'`analyze` est affiché, pas recopié tel quel** : l'étiquette
   de tonalité ('C major') et les noms de figures ('noire') sont stockés
   CANONIQUES et traduits au bord du rendu (`tr_key_label` / `tr_figure`).
   Une sortie française ne porte aucun mot anglais de l'analyse, une
   sortie anglaise aucune figure française — et l'étiquette `Mesure`/`Meter`
   s'aligne sur les autres du résumé (colonne deux-points stable).

2. **Chaque message d'erreur migré vers `tr()` sort dans LA langue
   demandée** : le message est comparé mot pour mot à la table i18n de la
   langue courante. Un littéral codé en dur ('PDF ignored:'), un message
   qui lit la mauvaise table ou une clé oubliée font échouer ce test —
   c'est la dent du principe « tout message utilisateur passe par tr() ».
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, silence, sine_note
from v2m import cli
from v2m.core import audio_io, i18n, pdf_export, playback, report
from v2m.core.i18n import set_language
from v2m.core.pdf_export import PdfExportError, PdfToolMissing

LANGS = ("fr", "en")
# figure names in English — used to prove the Figures line was translated
EN_FIGURE_RE = re.compile(
    r"\b(quarter|eighth|sixteenth|whole|half|dotted|triplet|rest)\b"
)
FR_FIGURE_RE = re.compile(r"\b(noire|blanche|ronde|croche|pointée|triolet|soupir|pause)\b")


@pytest.fixture(autouse=True)
def _lang_sandbox(monkeypatch, tmp_path):
    """Never touch the real ~/.v2m_lang; restore French after each test."""
    monkeypatch.setattr(i18n, "LANG_FILE", tmp_path / "lang")
    yield
    set_language("fr")


def _wav(tmp_path: Path) -> str:
    parts: list[np.ndarray] = []
    for hz in (220.0, 261.63, 329.63, 349.23):
        parts.append(sine_note(hz, 0.45))
        parts.append(silence(0.1))
    path = tmp_path / "melody.wav"
    sf.write(str(path), np.concatenate(parts), SR)
    return str(path)


def _expected(lang: str, key: str, **fields) -> str:
    """The message the i18n table mandates for `lang`, fields formatted."""
    return i18n.STRINGS[lang][key].format(**fields)


@pytest.mark.parametrize("lang", LANGS)
def test_summary_speaks_the_selected_language(lang, tmp_path, monkeypatch, capsys):
    """Le résumé complet (progrès + clés + figures) est dans la langue demandée."""
    code = cli.main(["--lang", lang, "analyze", _wav(tmp_path), "-o", str(tmp_path / "out")])
    out, err = capsys.readouterr()
    assert code == 0, f"l'analyse a échoué : {err}"

    lines = out.splitlines()
    tempo_line = next(l for l in lines if l.startswith("Tempo"))
    meter_line = next(l for l in lines if l.startswith(("Mesure", "Meter")))
    key_line = next(l for l in lines if l.startswith(("Tonalité", "Key")))
    fig_line = next(l for l in lines if l.startswith("Figures"))

    # the label columns share one colon position in BOTH languages
    assert tempo_line.index(":") == meter_line.index(":") == key_line.index(":")

    if lang == "fr":
        assert "Analyse de" in out and "Détection du tempo" in out
        assert "Tonalité" in key_line and "candidat" in out
        # canonical French figures stay French in French mode
        assert FR_FIGURE_RE.search(fig_line), f"figures FR absentes : {fig_line!r}"
        assert not EN_FIGURE_RE.search(fig_line), f"figures anglaises en FR : {fig_line!r}"
        assert not re.search(r"\b(major|minor)\b", out), f"tonalité non traduite : {out}"
        assert "Detecting" not in out, "progression anglaise en FR"
        assert "not detected" not in out
    else:
        assert "Analyzing" in out and "Detecting tempo" in out
        assert "Key" in key_line and "candidate" in out
        assert EN_FIGURE_RE.search(fig_line), f"figures not translated: {fig_line!r}"
        assert not FR_FIGURE_RE.search(fig_line), f"French figures in EN: {fig_line!r}"
        assert re.search(r"\b(major|minor)\b", key_line), "key label lost in EN"
        assert not re.search(r"\bTonalité\b", out), "libellé FR en EN"
        # word boundary: "candidat" is a substring of "candidate"
        assert not re.search(r"\bcandidat\b", out), "candidat FR en EN"
        assert "Détection" not in out, "progression française en EN"
        assert "non détectée" not in out


@pytest.mark.parametrize("lang", LANGS)
def test_input_validation_speaks_the_selected_language(lang, tmp_path, capsys):
    """Les erreurs d'entrée du CLI sortent telles quelles, dans la langue."""
    wav = _wav(tmp_path)

    assert cli.main(["--lang", lang, "analyze", "nope.wav"]) == 1
    err = capsys.readouterr().err.strip()
    assert err == _expected(lang, "err file not found", path="nope.wav")

    bad = tmp_path / "notes.txt"
    bad.write_text("not audio", encoding="utf-8")
    assert cli.main(["--lang", lang, "analyze", str(bad)]) == 1
    err = capsys.readouterr().err.strip()
    fmts = ", ".join(sorted(e.lstrip(".") for e in audio_io.AUDIO_EXTS))
    assert err == _expected(lang, "err bad format", ext=".txt", fmts=fmts)

    assert cli.main(["--lang", lang, "analyze", wav, "--min-conf", "1.5"]) == 1
    err = capsys.readouterr().err.strip()
    assert err == _expected(lang, "err minconf", v=1.5)


@pytest.mark.parametrize("lang", LANGS)
def test_every_migrated_error_message_follows_the_language(lang, tmp_path, monkeypatch):
    """Chaque site migré vers tr() rend EXACTEMENT la clé i18n de `lang`.

    Chaque déclenchement est le vrai chemin du code (levée, callback,
    sortie CLI) — seul le contexte externe (outil absent, micro absent) est
    simulé. Un message codé en dur ou lu dans l'autre table diverge ici.
    """
    set_language(lang)
    hint = _expected(lang, "err pdf install hint")

    # --- cli: err ui pdf wrapping err pdf skipped (the real --pdf run) ----
    monkeypatch.setattr(report, "find_notation_tool", lambda: None)
    with pytest.raises(PdfToolMissing) as ei:
        report.engrave_pdf(tmp_path / "stem")
    wrapped = _expected(lang, "err ui pdf", e=_expected(lang, "err pdf skipped", hint=hint))
    assert _expected(lang, "err pdf skipped", hint=hint) in wrapped
    assert str(ei.value) == _expected(lang, "err pdf skipped", hint=hint)

    # --- pdf_export: no tool, missing file, not runnable, timeout, failed -
    monkeypatch.setattr(pdf_export, "find_notation_tool", lambda: None)
    with pytest.raises(PdfExportError) as ei:
        pdf_export.export_pdf(tmp_path / "in.musicxml", tmp_path / "out.pdf")
    assert str(ei.value) == _expected(lang, "err pdf no tool", hint=hint)

    monkeypatch.setattr(pdf_export, "find_notation_tool", lambda: ("MuseScore", "mscore"))
    with pytest.raises(PdfExportError) as ei:
        pdf_export.export_pdf(tmp_path / "absent.musicxml", tmp_path / "out.pdf")
    assert str(ei.value) == _expected(
        lang, "err file not found", path=str(tmp_path / "absent.musicxml")
    )

    src = tmp_path / "in.musicxml"
    src.write_text("<score-partwise/>", encoding="utf-8")
    monkeypatch.setattr(
        pdf_export, "find_notation_tool", lambda: ("MuseScore", "v2m-no-such-cmd-xyz")
    )
    with pytest.raises(PdfExportError) as ei:
        pdf_export.export_pdf(src, tmp_path / "out.pdf")
    assert str(ei.value) == _expected(
        lang, "err pdf not runnable", cmd="v2m-no-such-cmd-xyz", hint=hint
    )

    class _FakeSubprocess:
        @staticmethod
        def run(*a, **k):  # noqa: ANN002, ANN003
            raise subprocess.TimeoutExpired(cmd="mscore", timeout=120)

        TimeoutExpired = subprocess.TimeoutExpired

    monkeypatch.setattr(pdf_export, "subprocess", _FakeSubprocess)
    with pytest.raises(PdfExportError) as ei:
        pdf_export.export_pdf(src, tmp_path / "out.pdf")
    assert str(ei.value) == _expected(lang, "err pdf timeout", name="MuseScore")

    class _FailingSubprocess(_FakeSubprocess):
        @staticmethod
        def run(*a, **k):  # noqa: ANN002, ANN003
            return SimpleNamespace(returncode=3, stderr="", stdout="")

    monkeypatch.setattr(pdf_export, "subprocess", _FailingSubprocess)
    with pytest.raises(PdfExportError) as ei:
        pdf_export.export_pdf(src, tmp_path / "out.pdf")
    assert str(ei.value) == _expected(
        lang, "err pdf conversion", name="MuseScore",
        msg=_expected(lang, "err pdf exit code", code=3),
    )
    monkeypatch.setattr(pdf_export, "subprocess", subprocess)

    # --- live: take stopped, take too short, analysis incomplete ----------
    from v2m.core.live_session import LiveTake

    take = LiveTake()
    take.stop()  # capture never opened: the take is stopped without a mic
    with pytest.raises(RuntimeError) as ei:
        take.start()
    assert str(ei.value) == _expected(lang, "err take stopped")

    with pytest.raises(ValueError) as ei:
        LiveTake().finalize()  # no capture -> empty signal -> too short
    assert str(ei.value) == _expected(lang, "err ui too short")

    from v2m.live import LiveSession

    session = LiveSession(out_stem=str(tmp_path / "take"))
    sig = np.zeros(SR, dtype=np.float32)
    # a take whose analysis came back WITHOUT a tempo: the adapter must
    # refuse it in the current language (cast: a stub stands in for LiveTake)
    session.take = cast(LiveTake, SimpleNamespace(
        state="stopped",
        result=None,
        signal=sig,
        sr=SR,
        finalize=lambda: (SimpleNamespace(tempo=None, key=None), sig),
    ))
    with pytest.raises(ValueError) as ei:
        session.export()
    assert str(ei.value) == _expected(lang, "err live incomplete")

    # --- audio_io + playback: sounddevice missing, two sites --------------
    monkeypatch.setattr(audio_io, "sd", None)
    with pytest.raises(RuntimeError) as ei:
        audio_io.LiveCapture(on_block=None, sr=SR).start()
    assert str(ei.value) == _expected(lang, "err sounddevice missing")

    monkeypatch.setattr(playback, "sd", None)
    errors: list[str] = []
    assert playback.PlaybackSession(on_error=errors.append).play() is False
    assert errors == [_expected(lang, "err sounddevice missing")]
