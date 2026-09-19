"""Tests i18n.py: language switching, persistence, note formats, translations."""

import pytest

from v2m.core import i18n
from v2m.core.i18n import (
    get_language,
    note_display,
    note_display_pair,
    set_language,
    tr,
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
    assert tr_figure("croche pointée") == "eighth dotted"
    assert tr_figure("triolet de croche") == "triplet of eighth"
    assert tr_figure(None) == ""


def test_tr_key_label():
    set_language("fr")
    assert tr_key_label("C major") == "C majeur"
    assert tr_key_label("A minor") == "A mineur"
    set_language("en")
    assert tr_key_label("C major") == "C major"


def test_tr_falls_back_to_key():
    set_language("en")
    assert tr("key does not exist") == "key does not exist"


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
