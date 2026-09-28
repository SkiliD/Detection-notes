"""The weekly recap PDF: the singer's progress, printed and kept.

Engraving is a SEPARATE step from the aggregate (`weekly.build_weekly`)
for the same measured reason the score's PDF is separate from
`export_result` (`report.engrave_pdf`): an external-tool failure must
never cost the data — and here the data is `weekly_text`, which the CLI
can print and the tests can read without a notation tool installed.

The tool contract mirrors `pdf_export`: `find_notation_tool` probes the
standard Windows install dirs beyond the PATH (the installers do not add
themselves to the PATH); `WeeklyPdfToolMissing` (nothing installed =
skipped) is distinct from `WeeklyPdfExportError` (the conversion ran and
failed) so the caller can decide whether to warn or to exit 2. The recap
itself travels as a MusicXML score whose lyrics CARRY the text: MuseScore
and LilyPond both engrave lyrics, so ONE tool — the one the score already
uses — renders the recap without a second install.
"""

from __future__ import annotations

from pathlib import Path

from .i18n import tr
from .weekly import weekly_text


class WeeklyPdfToolMissing(RuntimeError):
    """No notation tool found: the recap stays text-only (not an error)."""


class WeeklyPdfExportError(RuntimeError):
    """The tool ran and the conversion failed (the caller warns, exit 2)."""


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def build_recap_musicxml(data: dict) -> str:
    """The recap as a one-measure MusicXML: each recap line is a lyric.

    A whole-measure rest with `<lyric>` lines engraves plain text under a
    staff — enough for a printed recap, readable by every importer, and
    produced with the same writer discipline the score uses (exact
    divisions, `<type>` matching `<duration>`).
    """
    lines = [ln for ln in weekly_text(data).splitlines() if ln.strip()]
    lyrics = "".join(
        f'<lyric number="{i + 1}"><syllabic>single</syllabic>'
        f"<text>{_escape(ln)}</text></lyric>"
        for i, ln in enumerate(lines)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 '
        'Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="4.0">\n'
        "  <part-list><score-part id=\"P1\"><part-name>"
        + tr("weekly title")
        + "</part-name>"
        "</score-part></part-list>\n"
        "  <part id=\"P1\">\n"
        '    <measure number="1">\n'
        "      <attributes>\n"
        "        <divisions>24</divisions>\n"
        "        <time><beats>4</beats><beat-type>4</beat-type></time>\n"
        "        <clef><sign>G</sign><line>2</line></clef>\n"
        "      </attributes>\n"
        "      <note>\n"
        "        <rest measure=\"yes\"/>\n"
        "        <duration>96</duration>\n"
        "        <voice>1</voice>\n"
        "        <type>whole</type>\n"
        f"        {lyrics}\n"
        "      </note>\n"
        "    </measure>\n"
        "  </part>\n"
        "</score-partwise>\n"
    )


def engrave_weekly_pdf(data: dict, stem: str | Path) -> str:
    """Write `<stem>.pdf` from the weekly recap; return the path written.

    The recap MusicXML is converted by the SAME tool the score uses
    (`pdf_export.export_pdf`: MuseScore or LilyPond, probed beyond the
    PATH). `WeeklyPdfToolMissing` when nothing is installed; the
    intermediate MusicXML is removed after conversion (the recap's real
    content lives in `weekly_text`, the PDF is a print of it).
    """
    from .pdf_export import PdfExportError, PdfToolMissing, export_pdf, find_notation_tool

    stem = Path(stem)
    xml_path = stem.with_suffix(".weekly.musicxml")
    try:
        xml_path.write_text(build_recap_musicxml(data), encoding="utf-8")
        try:
            return export_pdf(xml_path, stem.with_suffix(".pdf"))
        except PdfToolMissing as e:
            raise WeeklyPdfToolMissing(str(e)) from e
        except PdfExportError as e:
            # a tool ABSENT (pdf_export folds it into its base class when
            # probed inside export_pdf) must read as a SKIP for the caller:
            # only a real conversion failure is an export error
            if find_notation_tool() is None:
                raise WeeklyPdfToolMissing(str(e)) from e
            raise WeeklyPdfExportError(str(e)) from e
        except Exception as e:  # a conversion failure, re-framed for the caller
            raise WeeklyPdfExportError(str(e)) from e
    finally:
        try:
            xml_path.unlink()  # the intermediate is never the product
        except OSError:
            pass
