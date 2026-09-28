"""Checks on an exported MusicXML score (test support, not shipped code).

The same contract serves the two places that judge a score: the
reference net, which compares bytes, and the synthetic cases, which
exercise what the real takes never contain. Writing the check once
keeps them from diverging.

The judge is not the export's table but MusicXML's arithmetic: a half
is two quarters, a dot adds half, a triplet takes two thirds. That is
the only vantage that can see a `<type>` lie about its `<duration>`,
since the two numbers come from different paths in the export (the
analysis's duration, the written-values catalogue's type).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# Duration of each type, in the export's divisions (24 per quarter).
BASE_DIVISIONS = {
    "whole": 96.0,
    "half": 48.0,
    "quarter": 24.0,
    "eighth": 12.0,
    "16th": 6.0,
    "32nd": 3.0,
}


def score_root(xml: str) -> ET.Element:
    """The tree of an exported score, without its DOCTYPE (an external DTD)."""
    return ET.fromstring(xml.split("?>", 1)[1])


def text_of(el: ET.Element | None) -> str:
    """The text of a required element: absent or empty, the file is invalid."""
    if el is None or el.text is None:
        raise AssertionError("required element missing from the exported score")
    return el.text


def written_divisions(note: ET.Element) -> float:
    """The duration the written `<type>` announces, dots and triplet included."""
    divisions = BASE_DIVISIONS[text_of(note.find("type"))] * (
        2.0 - 2.0 ** -len(note.findall("dot"))
    )
    if note.find("time-modification") is not None:
        return divisions * 2.0 / 3.0
    return divisions


def assert_measures_close(root: ET.Element, label: str) -> None:
    """Every measure totals its signature, and no `<type>` lies.

    The measure length is computed from the file itself (`beats` x
    divisions x 4 / beat-type), so 6/8 is 36 divisions the way an
    importer would read it — the check knows nothing about the Grid.
    """
    attrs = root.find("part/measure/attributes")
    assert attrs is not None, "measure attributes missing"
    divisions = int(text_of(attrs.find("divisions")))
    beats = int(text_of(attrs.find("time/beats")))
    beat_type = int(text_of(attrs.find("time/beat-type")))
    measure = beats * 4 * divisions // beat_type
    for m in root.findall("part/measure"):
        where = f"{label} measure {m.get('number')}"
        notes = m.findall("note")
        total = sum(int(text_of(n.find("duration"))) for n in notes)
        assert total == measure, f"{where} : the measure totals {total} / {measure}"
        for n in notes:
            duration = int(text_of(n.find("duration")))
            assert abs(written_divisions(n) - duration) < 1e-9, (
                f"{where} : <type> {text_of(n.find('type'))} pour {duration} divisions"
            )
