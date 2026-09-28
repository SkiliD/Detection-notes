"""Tests i18n.py: language switching, persistence, note formats, translations."""

import contextlib
import re

import pytest

from v2m.core import i18n
from v2m.core.i18n import (
    get_language,
    note_display,
    note_display_pair,
    set_language,
    tr,
    tr_chord_label,
    tr_figure,
    tr_key_label,
)


@pytest.fixture(autouse=True)
def _restore_fr(monkeypatch, tmp_path):
    """Run each test in a sandboxed language state, restore French after."""
    monkeypatch.setattr(i18n, "LANG_FILE", tmp_path / "lang")
    set_language("fr")
    yield
    set_language("fr")


def test_supported_languages():
    assert i18n.available_languages() == ("fr", "en")


def test_default_is_french():
    assert get_language() == "fr"


def test_switch_to_english_and_back():
    set_language("en")
    assert get_language() == "en"
    assert tr("key tempo") == "Tempo"  # same word, but the catalog resolved it
    assert tr("key chord") == "Chord"
    set_language("fr")
    assert tr("key chord") == "Accord"


def test_persistence(tmp_path):
    set_language("en")
    assert i18n.LANG_FILE.read_text(encoding="utf-8") == "en"
    # a fresh reader picks the stored value up from disk
    assert get_language() == "en"


def test_invalid_language_rejected():
    with pytest.raises(ValueError):
        set_language("de")


def test_note_display_follows_language():
    set_language("fr")
    assert note_display(69) == "La4"
    set_language("en")
    assert note_display(69) == "A4"


def test_note_display_pair_orders_current_first():
    set_language("fr")
    assert note_display_pair(69) == "La4 / A4"
    set_language("en")
    assert note_display_pair(69) == "A4 / La4"


def test_tr_figure_translates_for_display():
    set_language("fr")
    assert tr_figure("noire") == "noire"
    assert tr_figure("croche pointée") == "croche pointée"
    assert tr_figure("triolet de croche") == "triolet de croche"
    set_language("en")
    assert tr_figure("noire") == "quarter"
    # English word order: the dot comes BEFORE the value, never after —
    # the old per-word replacement produced « eighth dotted ».
    assert tr_figure("croche pointée") == "dotted eighth"
    assert tr_figure("blanche pointée") == "dotted half"
    assert tr_figure("double-croche pointée") == "dotted sixteenth"
    # 'double-croche' is ONE French word: not « double eighth »
    assert tr_figure("double-croche") == "sixteenth"
    # dotted rests follow the same order as dotted notes
    assert tr_figure("demi-pause pointée") == "dotted half rest"
    assert tr_figure("quart de soupir") == "sixteenth rest"
    assert tr_figure("triolet de croche") == "triplet of eighth"
    assert tr_figure("triolet de blanche") == "triplet of half"
    # no word-level fallback: an unknown name comes back unchanged, like
    # tr()'s key fallback — half-translating it would engrave nonsense
    assert tr_figure("figure inconnue") == "figure inconnue"
    assert tr_figure(None) == ""


def test_tr_figure_covers_the_whole_catalogue_vocabulary():
    """Every CATALOGUE name (figures, rests, dots, triplets) is translated
    with no French residue — a value added to the catalogue without going
    through the display table fails HERE, instead of leaking FR into an
    English UI."""
    from v2m.core.grid import CATALOGUE

    names = {c.name for c in CATALOGUE} | {c.rest for c in CATALOGUE}
    residue = re.compile(
        r"\b(noire|blanche|ronde|croche|pointée|triolet|soupir|pause|de)\b"
    )
    set_language("fr")
    for name in names:
        assert tr_figure(name) == name  # canonical stays canonical in FR
    set_language("en")
    for name in sorted(names):
        en = tr_figure(name)
        assert en != name, f"'{name}' not translated"
        assert not residue.search(en), f"'{name}' -> '{en}' : French residue"


def test_every_catalogue_figure_has_a_non_identity_english_translation():
    """The display table covers the grid: every name the CATALOGUE can emit
    (figure, rest, dotted, triplet) has an EN translation that is NOT the
    name itself, carried by _FIGURE_EN and not by the fallback.

    Identity would go unnoticed by the user (the fallback returns the
    canonical name, same rendering): this test forces the proof through
    the table — _FIGURE_EN[name] must exist and differ, directly or via
    the triplet composition. Any catalogue value added without its line
    in the table fails HERE.
    """
    from v2m.core.grid import CATALOGUE
    from v2m.core.i18n import _FIGURE_EN

    names = {c.name for c in CATALOGUE} | {c.rest for c in CATALOGUE}
    # Triplets are not table entries: they are COMPOSED from the base value's
    # own translation (one rule in tr_figure, zero duplication in the table).
    # Their coverage check is the composition, not a direct lookup.
    plain = sorted(n for n in names if not n.startswith("triolet de "))
    triplets = sorted(n for n in names if n.startswith("triolet de "))

    missing = sorted(n for n in plain if n not in _FIGURE_EN)
    assert not missing, (
        "figures without an EN translation in _FIGURE_EN : " + ", ".join(missing)
    )
    for name in plain:
        en = _FIGURE_EN[name]
        assert en != name, f"'{name}' translated by itself: not a translation"
    set_language("en")
    for name in triplets:
        base = name[len("triolet de "):]
        assert base in _FIGURE_EN, f"triplet '{name}' : base value not translated"
        en = _FIGURE_EN[base]
        assert tr_figure(name) == f"triplet of {en}", (
            f"'{name}' wrongly composed from '{base}' -> '{en}'"
        )


def test_tr_key_label():
    set_language("fr")
    assert tr_key_label("C major") == "C majeur"
    assert tr_key_label("A minor") == "A mineur"
    set_language("en")
    assert tr_key_label("C major") == "C major"


def test_tr_chord_label_translates_the_whole_label():
    """The whole-label translator: sol-fa root + French quality.

    Replaces the removed tr_chord_quality (ADR 0004): the suffix-based
    translation rendered "Am" for a French reader reading "La mineur".
    The root is parsed against the English table (chords.py builds the
    chords with NOTE_NAMES) — two characters first, else 'C#m' would be
    'C' + '#m' — and DISPLAYED in sol-fa.
    """
    set_language("fr")
    assert tr_chord_label("Am") == "La mineur"
    assert tr_chord_label("C") == "Do majeur"
    assert tr_chord_label("Cmaj7") == "Do septième majeure"
    assert tr_chord_label("C#m") == "Do# mineur"    # two-character root
    assert tr_chord_label("Gdim") == "Sol diminué"
    # EN stays canonical (storage is canonical, display translates)
    set_language("en")
    assert tr_chord_label("Am") == "Am"


def test_tr_chord_label_covers_the_detection_vocabulary():
    """Every quality of chords.CHORD_TEMPLATES has its display word.

    Derived from the producer — the dict _classify iterates: a quality
    added there without going through _CHORD_QUALITY_FR would leak as
    code ("Am9") in a French UI, and this test catches it instead of the
    user. An unknown or malformed label comes back unchanged (closed
    vocabulary, same contract as tr_figure) — never half-translated.
    """
    from v2m.core.chords import CHORD_TEMPLATES

    set_language("fr")
    for quality in CHORD_TEMPLATES:
        assert tr_chord_label(f"C{quality}") != f"C{quality}", (
            f"quality '{quality}' not translated"
        )
        # the root word is translated, the label is NOT left as code
        assert not tr_chord_label(f"C{quality}").startswith("C{0}".format(quality))
    # unknown quality / unknown root: unchanged, never half-translated
    assert tr_chord_label("Am9") == "Am9"
    assert tr_chord_label("Xq") == "Xq"


def test_tr_key_label_covers_the_detection_vocabulary():
    """Every mode produced by the detection has its display translation.

    Derived from keys.KEY_PROFILES — the source of truth detect_key
    iterates: a mode added there without going through _MODE_FR would
    leak as English in a French UI, and this test catches it instead of
    the user.
    """
    from v2m.core.keys import KEY_PROFILES

    set_language("fr")
    for mode in KEY_PROFILES:
        assert tr_key_label(f"C {mode}") != f"C {mode}", (
            f"mode '{mode}' not translated"
        )
        # the tonic word is preserved, only the mode translates
        assert tr_key_label(f"C {mode}").startswith("C ")
    # a mode outside the table comes back unchanged (closed vocabulary,
    # same contract as tr_figure) — never half-translated
    assert tr_key_label("C dorien") == "C dorien"


def test_tr_falls_back_to_key():
    set_language("en")
    assert tr("key does not exist") == "key does not exist"


def test_tr_debug_wrapper_logs_missing_keys(capsys):
    """In DEBUG, a missing key is logged on stderr — the net for dynamic
    call sites.

    The fallback NEVER raises (contract pinned above), so a misspelled
    key in a NON-literal call (variable, f-string) escapes the static
    guard and ships silently to production: the user reads the raw key
    with no error anywhere. The wrapper (set_debug(True), or
    V2M_DEBUG_TR=1 at import) logs every miss — what the AST guard
    cannot see.
    """
    import io

    from v2m.core.i18n import set_debug

    set_language("fr")
    try:
        set_debug(True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert tr("key absent x") == "key absent x"   # unchanged fallback
        assert "key absent x" in err.getvalue()
        assert "fr" in err.getvalue()
        # a PRESENT key logs nothing
        err2 = io.StringIO()
        with contextlib.redirect_stderr(err2):
            assert tr("key tempo") == "Tempo"
        assert err2.getvalue() == ""
        set_debug(False)
        err3 = io.StringIO()
        with contextlib.redirect_stderr(err3):
            tr("key absent x")
        assert err3.getvalue() == ""   # silent mode by default
    finally:
        set_debug(False)


def test_toggle_language():
    set_language("fr")
    assert i18n.toggle_language() == "en"
    assert i18n.toggle_language() == "fr"


def test_polish_strings_complete_and_formatted():
    """Polish strings exist in BOTH catalogs and their placeholders format."""
    fr_keys = set(i18n.STRINGS["fr"])
    en_keys = set(i18n.STRINGS["en"])
    assert fr_keys == en_keys  # no untranslated (or over-translated) key
    # first-run guidance and empty state are present in both languages
    for key in ("ui steps", "ui pdf badge", "ui panel waiting", "ui exported ok"):
        assert key in fr_keys and i18n.STRINGS["en"][key]
    # {placeholders} resolve without KeyError in either language
    set_language("fr")
    assert "mid" in tr("ui status exported").format(files="a.mid")
    set_language("en")
    assert tr("ui status live done").format(bpm=120.0, key="C major", n=5)
