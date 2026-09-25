"""Contrôles sur une partition MusicXML exportée (support de test, pas de code).

Le même contrat sert aux deux endroits qui jugent une partition : le filet de
référence, qui compare des octets, et les cas synthétiques, qui exercent ce que
les prises réelles ne contiennent pas. Écrire le contrôle une fois évite qu'ils
divergent.

Le juge n'est pas la table de l'export mais l'arithmétique de MusicXML : une
blanche vaut deux noires, un point ajoute la moitié, un triolet prend les deux
tiers. C'est le seul point de vue qui puisse voir un `<type>` mentir sur sa
`<duration>`, puisque les deux chiffres viennent de chemins différents dans
l'export (la durée de l'analyse, le type du catalogue des valeurs écrites).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# Durée de chaque type, dans les divisions de l'export (24 par noire).
BASE_DIVISIONS = {
    "whole": 96.0,
    "half": 48.0,
    "quarter": 24.0,
    "eighth": 12.0,
    "16th": 6.0,
    "32nd": 3.0,
}


def score_root(xml: str) -> ET.Element:
    """L'arbre d'un score exporté, sans sa DOCTYPE (une DTD externe)."""
    return ET.fromstring(xml.split("?>", 1)[1])


def text_of(el: ET.Element | None) -> str:
    """Le texte d'un élément requis : absent ou vide, le fichier est invalide."""
    if el is None or el.text is None:
        raise AssertionError("élément requis absent du score exporté")
    return el.text


def written_divisions(note: ET.Element) -> float:
    """Durée qu'annonce le `<type>` écrit, points et triolet compris."""
    divisions = BASE_DIVISIONS[text_of(note.find("type"))] * (
        2.0 - 2.0 ** -len(note.findall("dot"))
    )
    if note.find("time-modification") is not None:
        return divisions * 2.0 / 3.0
    return divisions


def assert_measures_close(root: ET.Element, label: str) -> None:
    """Chaque mesure totalise sa signature, et aucun `<type>` ne ment.

    La longueur de mesure est calculée depuis le fichier lui-même (`beats` ×
    divisions × 4 / valeur du temps), donc 6/8 vaut 36 divisions comme un
    importateur le lirait — le contrôle ne connaît pas la Grille.
    """
    attrs = root.find("part/measure/attributes")
    assert attrs is not None, "attributs de mesure absents"
    divisions = int(text_of(attrs.find("divisions")))
    beats = int(text_of(attrs.find("time/beats")))
    beat_type = int(text_of(attrs.find("time/beat-type")))
    measure = beats * 4 * divisions // beat_type
    for m in root.findall("part/measure"):
        where = f"{label} mesure {m.get('number')}"
        notes = m.findall("note")
        total = sum(int(text_of(n.find("duration"))) for n in notes)
        assert total == measure, f"{where} : la mesure totalise {total} / {measure}"
        for n in notes:
            duration = int(text_of(n.find("duration")))
            assert abs(written_divisions(n) - duration) < 1e-9, (
                f"{where} : <type> {text_of(n.find('type'))} pour {duration} divisions"
            )
