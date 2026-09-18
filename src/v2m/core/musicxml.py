"""Export MusicXML : partition lisible à partir de l'analyse.

Principes :
- divisions = 12 (commun multiple de 3 et 4) : toutes les figures simples,
  pointées ET triolets ont une durée entière de divisions.
- 1 temps (quarter) = 12 divisions, mesure 4/4 = 48 divisions.
- Chaque temps est attribué à la note qui le recouvre le plus (seuil 30 ms).
- Une note couvrant plusieurs temps est rendue note par temps avec des ties
  (stop si elle a commencé avant, start si elle continue après) — y compris
  à travers les frontières de mesure, comme le veut MusicXML.
- Les notes étiquetées « triolet de … » sont groupées en blocs consécutifs ;
  un bloc entièrement contenu dans une mesure est rendu avec son bracket
  tuplet (time-modification 3:2) et fait avancer la position du temps.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

STEP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# divisions=12 : pas -> (type MusicXML, nb de points)
_TYPE_BY_STEPS = {
    48: ("whole", 0),
    36: ("half", 1),
    32: ("whole", 0),      # triolet de ronde
    24: ("half", 0),
    18: ("quarter", 1),
    16: ("half", 0),       # triolet de blanche
    12: ("quarter", 0),
    9: ("eighth", 1),
    8: ("quarter", 0),     # triolet de noire
    6: ("eighth", 0),
    4: ("eighth", 0),      # triolet de croche
    3: ("16th", 0),
    2: ("16th", 0),        # triolet de double-croche
    1: ("32nd", 0),
}

_TUPLET_STEPS = {32, 16, 8, 4, 2}  # pas qui correspondent à des triolets

_FIFTHS_MAJ = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}
_ENH = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def _key_fifths(tonic: str, mode: str) -> int:
    t = tonic
    if mode == "minor":
        t = STEP_NAMES[(STEP_NAMES.index(t) + 3) % 12]  # relative majeure
    # la table couvre dièses ET bémols usuels ; secours enharmonique sinon
    if t in _FIFTHS_MAJ:
        return _FIFTHS_MAJ[t]
    return _FIFTHS_MAJ.get(_ENH.get(t, t), 0)


def _pitch_xml(midi: int) -> str:
    name = STEP_NAMES[midi % 12]
    octave = midi // 12 - 1
    step, alter = (name[:-1], 1) if name.endswith("#") else (name, 0)
    alter_xml = "<alter>1</alter>" if alter else ""
    return f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"


def _note_xml(
    midi: int | None,
    steps: int,
    ties: tuple[str, ...] = (),
    tuplet_side: str | None = None,
) -> str:
    """Une <note> (ou un silence si midi est None)."""
    steps = max(1, int(steps))
    type_, dots = _TYPE_BY_STEPS.get(steps, ("quarter", 0))
    is_tuplet = steps in _TUPLET_STEPS
    parts = ["<note>"]
    parts.append("<rest/>" if midi is None else _pitch_xml(midi))
    parts.append(f"<duration>{steps}</duration>")
    for t in ties:
        parts.append(f'<tie type="{t}"/>')
    parts.append("<voice>1</voice>")
    parts.append(f"<type>{type_}</type>")
    parts.extend(["<dot/>"] * dots)
    if is_tuplet and midi is not None:
        parts.append(
            "<time-modification><actual-notes>3</actual-notes>"
            "<normal-notes>2</normal-notes></time-modification>"
        )
    notations = [f'<tied type="{t}"/>' for t in ties]
    if is_tuplet and midi is not None and tuplet_side:
        notations.append(f'<tuplet type="{tuplet_side}"/>')
    if notations:
        parts.append(f"<notations>{''.join(notations)}</notations>")
    parts.append("</note>")
    return "".join(parts)


@dataclass
class _TripletBlock:
    """Bloc de notes triolet consécutives (rendu groupé)."""

    notes: list
    start: float
    end: float


def _find_triplet_blocks(notes) -> list[_TripletBlock]:
    """Groupes de ≥2 notes triolet consécutives (chevauchement nul autorisé)."""
    trips = sorted(
        (n for n in notes if n.figure and n.figure.startswith("triolet")),
        key=lambda n: n.start,
    )
    blocks: list[_TripletBlock] = []
    cur: list = []
    for n in trips:
        if cur and n.start - cur[-1].end > 0.25:  # trou > 250 ms : nouveau bloc
            if len(cur) >= 2:
                blocks.append(_TripletBlock(cur, cur[0].start, cur[-1].end))
            cur = []
        cur.append(n)
    if len(cur) >= 2:
        blocks.append(_TripletBlock(cur, cur[0].start, cur[-1].end))
    return blocks


def _render_triplet_block(block: _TripletBlock, bpm: float) -> str:
    """Rend le bloc : durées égales + bracket start/stop."""
    beat_s = 60.0 / bpm
    total_steps = max(1, int(round((block.end - block.start) / beat_s * 12)))
    per = max(1, total_steps // len(block.notes))
    out = []
    for i, n in enumerate(block.notes):
        side = "start" if i == 0 else ("stop" if i == len(block.notes) - 1 else None)
        out.append(_note_xml(n.midi, per, tuplet_side=side))
    return "".join(out)


def _snap_notes(notes, bpm: float, divisions_per_beat: int = 4):
    """Calage des frontières sur la grille de doubles-croches.

    Le tempo détecté dérive légèrement du tempo réel humain ; sans calage,
    les frontières finissent hors grille et génèrent des fragments
    (16th/32nd). Chaque frontière rejoint son pas le plus proche.
    """
    from types import SimpleNamespace

    step_s = (60.0 / bpm) / divisions_per_beat
    out = []
    for n in notes:
        s = round(n.start / step_s) * step_s
        e = round(n.end / step_s) * step_s
        if e - s < step_s / 2:
            e = s + step_s
        out.append(
            SimpleNamespace(
                start=s, end=e, midi=n.midi, confidence=n.confidence, figure=n.figure
            )
        )
    return out


def _beat_owner(notes, b_start: float, b_end: float):
    """Note recouvrant le plus ce beat (None si recouvrement < 30 ms)."""
    best, best_ov = None, 0.030
    for n in notes:
        ov = min(n.end, b_end) - max(n.start, b_start)
        if ov > best_ov:
            best, best_ov = n, ov
    return best


def build_score_xml(notes, bpm: float, key=None, title: str = "Transcription", snap: bool = True) -> str:
    """Génère le MusicXML complet (chaîne)."""
    if snap:
        notes = _snap_notes(notes, bpm)
    divisions = 12
    beat_s = 60.0 / bpm
    steps_per_beat = 12
    measure_beats = 4

    max_end = max((n.end for n in notes), default=0.0)
    total_beats = max(1, int(max_end / beat_s + 0.999))
    total_measures = max(1, (total_beats + measure_beats - 1) // measure_beats)

    fifths = _key_fifths(key.best.tonic, key.best.mode) if key else 0

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n',
        '<score-partwise version="4.0">\n',
        f"  <work><work-title>{escape(title)}</work-title></work>\n",
        "  <part-list>\n",
        '    <score-part id="P1">\n',
        "      <part-name>Voix</part-name>\n",
        "    </score-part>\n",
        "  </part-list>\n",
        '  <part id="P1">\n',
        "    <measure number='1'>\n",
        "      <attributes>\n",
        f"        <divisions>{divisions}</divisions>\n",
        f"        <key><fifths>{fifths}</fifths></key>\n",
        "        <time><beats>4</beats><beat-type>4</beat-type></time>\n",
        "        <clef><sign>G</sign><line>2</line></clef>\n",
        "      </attributes>\n",
        "      <direction placement='above'>\n",
        "        <direction-type>\n",
        "          <metronome><beat-unit>quarter</beat-unit>"
        f"<per-minute>{bpm:.1f}</per-minute></metronome>\n",
        "        </direction-type>\n",
        f"        <sound tempo='{bpm:.1f}'/>\n",
        "      </direction>\n",
    ]

    blocks = _find_triplet_blocks(notes)

    for m in range(total_measures):
        if m > 0:
            out.append(f"    <measure number='{m + 1}'>\n")
        m_start = m * measure_beats * beat_s
        m_end = m_start + measure_beats * beat_s
        skip_until = m_start
        for b in range(measure_beats):
            b_start = (m * measure_beats + b) * beat_s
            b_end = b_start + beat_s
            if b_start < skip_until - 1e-6:
                continue  # déjà couvert par un bloc triolet

            # --- bloc triolet commençant dans ce temps, contenu dans la mesure ---
            blk = next(
                (
                    k
                    for k in blocks
                    if abs(k.start - b_start) < beat_s * 0.4
                    and k.end <= m_end + 1e-6
                    and k.start >= m_start - 1e-6
                ),
                None,
            )
            if blk is not None:
                out.append(_render_triplet_block(blk, bpm))
                skip_until = blk.end
                continue

            # --- temps entier tenu par une seule note ---
            owner = _beat_owner(notes, b_start, b_end)
            beat_dur = b_end - b_start
            # une note concurrente significative (> 20 % du temps) force la
            # sous-division ; en dessous, c'est un artefact de détection que
            # l'on absorbe dans la note dominante (une vraie double-croche
            # couvre 25 % d'un temps, elle est donc préservée)
            others = [
                n
                for n in notes
                if n is not owner
                and (min(n.end, b_end) - max(n.start, b_start)) > 0.20 * beat_dur
            ]
            if owner is not None and not others:
                ties = []
                if owner.start < b_start - 1e-6:
                    ties.append("stop")
                if owner.end > b_end + 1e-6:
                    ties.append("start")
                out.append(_note_xml(owner.midi, steps_per_beat, ties=tuple(ties)))
                continue

            # --- sous-division du temps ---
            inside = [
                n
                for n in notes
                if min(n.end, b_end) - max(n.start, b_start) > 0.030
            ]
            inside.sort(key=lambda n: n.start)
            if inside:
                cursor = b_start
                for n in inside:
                    if n.start > cursor + 1e-6:
                        gap = max(1, int(round((n.start - cursor) / beat_s * 12)))
                        out.append(_note_xml(None, gap))
                    seg_start = max(n.start, b_start)
                    seg_end = min(n.end, b_end)
                    steps = max(1, int(round((seg_end - seg_start) / beat_s * 12)))
                    ties = []
                    if n.start < b_start - 1e-6:
                        ties.append("stop")
                    if n.end > b_end + 1e-6:
                        ties.append("start")
                    out.append(_note_xml(n.midi, min(steps, 12), ties=tuple(ties)))
                    cursor = seg_end
                if b_end - cursor > 1e-6:
                    rest = max(1, int(round((b_end - cursor) / beat_s * 12)))
                    out.append(_note_xml(None, rest))
            else:
                out.append(_note_xml(None, steps_per_beat))
        out.append("    </measure>\n")

    out.extend(["  </part>\n", "</score-partwise>\n"])
    return "".join(out)


def save_musicxml(notes, bpm: float, key, path: str, title: str = "Transcription") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_score_xml(notes, bpm, key, title=title))
