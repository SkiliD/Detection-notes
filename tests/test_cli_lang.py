"""The CLI speaks the requested language — the rendered output is verified,
not hoped for.

Two contracts, two languages:

1. **The `analyze` summary is displayed, not copied verbatim**: the key
   label ('C major') and the figure names ('noire') are stored
   CANONICAL and translated at the rendering edge (`tr_key_label` /
   `tr_figure`). A French output carries no English analysis word, an
   English output no French figure — and the `Mesure`/`Meter` label
   aligns with the summary's others (stable colon column).

2. **Every error message migrated to `tr()` comes out in THE requested
   language**: the message is compared word for word against the i18n
   table of the current language. A hardcoded literal ('PDF ignored:'),
   a message reading the wrong table or a forgotten key fail this test
   — it is the teeth of the "every user message goes through tr()"
   principle.
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
    """The full summary (progress + keys + figures) is in the requested language."""
    code = cli.main(["--lang", lang, "analyze", _wav(tmp_path), "-o", str(tmp_path / "out")])
    out, err = capsys.readouterr()
    assert code == 0, f"the analysis failed : {err}"

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
        assert FR_FIGURE_RE.search(fig_line), f"FR figures missing : {fig_line!r}"
        assert not EN_FIGURE_RE.search(fig_line), f"English figures in FR : {fig_line!r}"
        assert not re.search(r"\b(major|minor)\b", out), f"key not translated : {out}"
        assert "Detecting" not in out, "English progress in FR"
        assert "not detected" not in out
    else:
        assert "Analyzing" in out and "Detecting tempo" in out
        assert "Key" in key_line and "candidate" in out
        assert EN_FIGURE_RE.search(fig_line), f"figures not translated: {fig_line!r}"
        assert not FR_FIGURE_RE.search(fig_line), f"French figures in EN: {fig_line!r}"
        assert re.search(r"\b(major|minor)\b", key_line), "key label lost in EN"
        assert not re.search(r"\bTonalité\b", out), "FR label in EN"
        # word boundary: "candidat" is a substring of "candidate"
        assert not re.search(r"\bcandidat\b", out), "FR candidat in EN"
        assert "Détection" not in out, "French progress in EN"
        assert "non détectée" not in out


@pytest.mark.parametrize("lang", LANGS)
def test_input_validation_speaks_the_selected_language(lang, tmp_path, capsys):
    """The CLI's input errors come out as they are, in the language."""
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
    """Every site migrated to tr() renders EXACTLY `lang`'s i18n key.

    Each trigger is the code's real path (raise, callback, CLI exit) —
    only the external context (missing tool, missing mic) is simulated.
    A hardcoded message, or one read from the other table, diverges
    here.
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
