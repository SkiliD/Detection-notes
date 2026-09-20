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


def written_divisions(note: ET.Element) -> float:
    """Durée qu'annonce le `<type>` écrit, points et triolet compris."""
    divisions = BASE_DIVISIONS[note.find("type").text] * (
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
    divisions = int(attrs.find("divisions").text)
    measure = (
        int(attrs.find("time/beats").text)
        * 4
        * divisions
        // int(attrs.find("time/beat-type").text)
    )
    for m in root.findall("part/measure"):
        where = f"{label} mesure {m.get('number')}"
        notes = m.findall("note")
        total = sum(int(n.find("duration").text) for n in notes)
        assert total == measure, f"{where} : la mesure totalise {total} / {measure}"
        for n in notes:
            duration = int(n.find("duration").text)
            assert abs(written_divisions(n) - duration) < 1e-9, (
                f"{where} : <type> {n.find('type').text} pour {duration} divisions"
            )
