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
    """Chaque nom du CATALOGUE (figures, silences, pointés, triolets) est
    traduit sans résidu français — une valeur ajoutée au catalogue sans
    passer par la table d'affichage échoue ici, au lieu de fuir en FR
    dans une UI anglaise."""
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
        assert en != name, f"« {name} » non traduit"
        assert not residue.search(en), f"« {name} » -> « {en} » : résidu français"


def test_every_catalogue_figure_has_a_non_identity_english_translation():
    """La table d'affichage couvre la grille : tout nom que le CATALOGUE peut
    émettre (figure, silence, pointé, triolet) a une traduction EN qui n'est
    PAS le nom lui-même, portée par _FIGURE_EN et non par le fallback.

    L'identité passerait inaperçue à l'utilisateur (le fallback renvoie le
    nom canonique, même rendu) : ce test force la preuve par la table —
    _FIGURE_EN[name] doit exister et différer, directement ou par la
    composition triolet. Toute valeur ajoutée au catalogue sans sa ligne
    dans la table échoue ICI.
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
        "figures sans traduction EN dans _FIGURE_EN : " + ", ".join(missing)
    )
    for name in plain:
        en = _FIGURE_EN[name]
        assert en != name, f"« {name} » traduit par lui-même : pas une traduction"
    set_language("en")
    for name in triplets:
        base = name[len("triolet de "):]
        assert base in _FIGURE_EN, f"triolet « {name} » : valeur de base non traduite"
        en = _FIGURE_EN[base]
        assert tr_figure(name) == f"triplet of {en}", (
            f"« {name} » mal composé depuis « {base} » -> « {en} »"
        )


def test_tr_key_label():
    set_language("fr")
    assert tr_key_label("C major") == "C majeur"
    assert tr_key_label("A minor") == "A mineur"
    set_language("en")
    assert tr_key_label("C major") == "C major"


def test_tr_chord_label_translates_the_whole_label():
    """Le traducteur whole-label : racine solfège + qualité française.

    Remplace le tr_chord_quality supprimé (ADR 0004) : la traduction par
    suffixe rendait « Am » pour un lecteur français qui lit « La mineur ».
    La racine est parsée contre la table anglaise (les accords sont construits
    par chords.py avec NOTE_NAMES) — deux caractères d'abord, sinon 'C#m'
    serait 'C' + '#m' — et AFFICHÉE en solfège.
    """
    set_language("fr")
    assert tr_chord_label("Am") == "La mineur"
    assert tr_chord_label("C") == "Do majeur"
    assert tr_chord_label("Cmaj7") == "Do septième majeure"
    assert tr_chord_label("C#m") == "Do# mineur"    # racine deux caractères
    assert tr_chord_label("Gdim") == "Sol diminué"
    # EN stays canonical (storage is canonical, display translates)
    set_language("en")
    assert tr_chord_label("Am") == "Am"


def test_tr_chord_label_covers_the_detection_vocabulary():
    """Toute qualité de chords.CHORD_TEMPLATES a son mot d'affichage.

    Dérivé du producteur — le dict que _classify itère : une qualité
    ajoutée là sans passer par _CHORD_QUALITY_FR fuirait en code (« Am9 »)
    dans une UI française, et ce test l'attrape au lieu de l'utilisateur.
    Un label inconnu ou mal formé revient inchangé (vocabulaire fermé,
    même contrat que tr_figure) — jamais à moitié traduit.
    """
    from v2m.core.chords import CHORD_TEMPLATES

    set_language("fr")
    for quality in CHORD_TEMPLATES:
        assert tr_chord_label(f"C{quality}") != f"C{quality}", (
            f"qualité « {quality} » non traduite"
        )
        # the root word is translated, the label is NOT left as code
        assert not tr_chord_label(f"C{quality}").startswith("C{0}".format(quality))
    # unknown quality / unknown root: unchanged, never half-translated
    assert tr_chord_label("Am9") == "Am9"
    assert tr_chord_label("Xq") == "Xq"


def test_tr_key_label_covers_the_detection_vocabulary():
    """Tout mode produit par la détection a sa traduction d'affichage.

    Dérivé de keys.KEY_PROFILES — la source de vérité que detect_key
    itère : un mode ajouté là sans passer par _MODE_FR fuirait en anglais
    dans une UI française, et ce test l'attrape au lieu de l'utilisateur.
    """
    from v2m.core.keys import KEY_PROFILES

    set_language("fr")
    for mode in KEY_PROFILES:
        assert tr_key_label(f"C {mode}") != f"C {mode}", (
            f"mode « {mode} » non traduit"
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
    """En DEBUG, une clé absente est loggée sur stderr — le filet des sites
    dynamiques.

    Le fallback ne lève JAMAIS (contrat épinglé ci-dessus), donc une clé
    mal orthographiée dans un appel NON littéral (variable, f-string)
    échappe à la garde statique et part en production muette : l'utilisateur
    lit la clé brute sans erreur nulle part. Le wrapper (set_debug(True),
    ou V2M_DEBUG_TR=1 à l'import) logge chaque manquage — c'est ce que la
    garde AST ne peut pas voir.
    """
    import io

    from v2m.core.i18n import set_debug

    set_language("fr")
    try:
        set_debug(True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert tr("key absent x") == "key absent x"   # fallback inchangé
        assert "key absent x" in err.getvalue()
        assert "fr" in err.getvalue()
        # une clé PRÉSENTE ne logge rien
        err2 = io.StringIO()
        with contextlib.redirect_stderr(err2):
            assert tr("key tempo") == "Tempo"
        assert err2.getvalue() == ""
        set_debug(False)
        err3 = io.StringIO()
        with contextlib.redirect_stderr(err3):
            tr("key absent x")
        assert err3.getvalue() == ""   # mode muet par défaut
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
