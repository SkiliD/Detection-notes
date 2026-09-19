"""La fenêtre suit la langue courante — y compris hors des widgets.

Régression : `QFileDialog.getOpenFileName` recevait un titre et un filtre
codés en dur en français ; `tr()` ne les voyait pas, donc la boîte de
sélection restait française même en anglais.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core import i18n
from v2m.core.i18n import set_language, tr


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _restore_language(monkeypatch, tmp_path):
    """Sandbox the language state: never touch the real ~/.v2m_lang."""
    monkeypatch.setattr(i18n, "LANG_FILE", tmp_path / "lang")
    yield
    set_language("fr")


def test_file_dialog_speaks_the_current_language(qapp, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from v2m.gui import main_window as mw

    seen: dict[str, str] = {}

    def fake_get_open_file_name(parent, title, directory, selected_filter):  # noqa: ANN001
        seen["title"] = title
        seen["filter"] = selected_filter
        return "", ""

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(fake_get_open_file_name)
    )
    win = mw.MainWindow()
    try:
        set_language("fr")
        win.on_open_file()
        assert seen["title"] == tr("ui choose file")
        assert "Tous les fichiers" in seen["filter"]

        set_language("en")
        win.on_open_file()
        assert seen["title"] == "Choose an audio file"
        assert "All files" in seen["filter"]
    finally:
        win.close()


def test_both_languages_define_the_same_keys():
    """Une clé ajoutée d'un seul côté retomberait sur la clé (donc en anglais)."""
    assert set(i18n.STRINGS["fr"]) == set(i18n.STRINGS["en"])
