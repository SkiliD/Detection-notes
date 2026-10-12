"""MusicXML export: readable score built from the analysis.

Principles:
- divisions = 24 (a multiple of 3, 4 and 8): every plain, dotted AND
  triplet figure — and the 32nd — has an integer duration in divisions.
- 1 quarter beat = 24 divisions, a 4/4 measure = 96 divisions, a 6/8 (three
  quarter beats) 72.
- Each beat is written note by note: a note's own duration, then a rest for
  what is left of the beat (no note is stretched to fill the beat).
- A note spanning several beats is rendered beat by beat with ties
  (stop when it started before, start when it continues after) — including
  across measure boundaries, as MusicXML requires.
- Notes labeled "triolet de ..." are grouped into consecutive blocks; a
  block fully contained in a measure is rendered with its tuplet bracket
  (time-modification 3:2) and advances the beat position.
- Every detection is shown on the score: chord progression as <harmony>
  (one per measure span), sol-fa note names as <lyric>, confidence as the
  notehead color (green = confident -> red = doubtful) and a subtitle
  credit line with tempo / key / meter / global chord.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

from .grid import BeatGrid, CATALOGUE
from .grid import MUSICXML_DIVISIONS as DIVISIONS
from .music import NOTE_NAMES, midi_to_name_fr
from .i18n import tr_chord_label


# Below this overlap a "note" on a beat is a segmentation artifact, not a
# sound: the beat is written without it.
MIN_OVERLAP_S = 0.030

_TYPE_BY_STEPS: dict[int, tuple[str, int]] = {
    round(f.beats * DIVISIONS): (f.type_, f.dots)
    for f in CATALOGUE
}
_TUPLET_STEPS = {
    round(f.beats * DIVISIONS) for f in CATALOGUE if f.tuplet is not None
}

# The rest name is part of the same Figure entry as its MusicXML shape.
_REST_BY_NAME = {f.rest: f for f in CATALOGUE if f.tuplet is None}

# Rest decomposition order for unlabeled silent spans: plain values only
# (no triplet flavours — a half TRIPLET rest is not a notation), largest
# first, so an empty measure engraves ONE whole rest instead of four
# quarter rests.
_REST_FILL_ORDER = [
    (steps, type_dots)
    for steps, type_dots in sorted(_TYPE_BY_STEPS.items(), reverse=True)
    if steps not in _TUPLET_STEPS
]


def _rest_run_xml(steps: int, figure: str | None, breath: bool = False) -> str:
    """One or more rest <note> elements covering exactly `steps` divisions.

    A labeled figure (soupir, demi-pause...) is engraved as many times as
    it fits the span; the remainder — and unlabeled runs — fall back to the
    largest drawable plain value, so a silent run of N beats reads as the
    rest an engraver would write, not a sequence of generic quarters.
    A HEARD breath carries a <words> direction — where the singer took air,
    the annotation a vocalist reads on any printed part.
    """
    parts: list[str] = []
    if breath:
        parts.append(
            '<direction placement="above"><direction-type><words>Respiration</words>'
            "</direction-type></direction>"
        )
    rest_figure = (
        _REST_BY_NAME.get(figure) if figure is not None else None
    )
    fig_steps = (
        round(rest_figure.beats * DIVISIONS)
        if rest_figure is not None
        else 0
    )
    while fig_steps and steps >= fig_steps:
        parts.append(_note_xml(None, fig_steps))
        steps -= fig_steps
    for s, _td in _REST_FILL_ORDER:
        while steps >= s:
            steps -= s
            parts.append(_note_xml(None, s))
    if steps > 0:  # pragma: no cover — degenerate off-grid remnant
        parts.append(_note_xml(None, steps))
    return "".join(parts)


_FIFTHS_MAJ = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}
_ENH = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def _key_fifths(tonic: str, mode: str) -> int:
    t = tonic
    if mode == "minor":
        t = NOTE_NAMES[(NOTE_NAMES.index(t) + 3) % 12]  # relative majeure
    # the table covers sharps AND common flats; enharmonic fallback otherwise
    if t in _FIFTHS_MAJ:
        return _FIFTHS_MAJ[t]
    return _FIFTHS_MAJ.get(_ENH.get(t, t), 0)


def _pitch_xml(midi: int) -> str:
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    step, alter = (name[:-1], 1) if name.endswith("#") else (name, 0)
    alter_xml = "<alter>1</alter>" if alter else ""
    return f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"


def _confidence_color(conf: float) -> str | None:
    """Notehead fill color from confidence: green (sure) -> red (doubtful)."""
    conf = min(1.0, max(0.0, conf))
    if conf >= 0.97:
        return None  # default black for very confident notes
    r = int(220 - 140 * conf)   # 220 (low) -> 80 (high)
    g = int(80 + 140 * conf)    #  80 (low) -> 220 (high)
    return f"#{r:02x}{g:02x}80"


def _note_xml(
    midi: int | None,
    steps: int,
    ties: tuple[str, ...] = (),
    tuplet_side: str | None = None,
    color: str | None = None,
    lyric: str | None = None,
) -> str:
    """One <note> (or a rest when midi is None), typed by its duration.

    A silence is typed from its written duration, never from a detected
    label: the <duration> is what fills the measure, so a label that
    disagreed with it would leave a file importers read differently.
    """
    steps = max(1, int(steps))
    type_, dots = _TYPE_BY_STEPS.get(steps, ("quarter", 0))
    is_tuplet = steps in _TUPLET_STEPS
    parts = [f'<note color="{color}">'] if color else ["<note>"]
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
    if midi is not None and lyric and "stop" not in ties:  # not on tied continuations
        parts.append(
            f'<lyric><syllabic>single</syllabic><text>{escape(lyric)}</text></lyric>'
        )
    parts.append("</note>")
    return "".join(parts)


def _harmony_xml(root: str, kind: str) -> str:
    """A MusicXML <harmony> (chord symbol above the staff).

    `root` is a pitch-class name from chords.py ('C', 'F#'), `kind` the
    quality suffix ('', 'm', '7', 'maj7', 'm7', 'dim', 'aug').
    """
    step = root.rstrip("#b")
    alter = "1" if root.endswith("#") else ("-1" if root.endswith("b") else None)
    alter_xml = f"<alter>{alter}</alter>" if alter else ""
    kind_xml = f"<kind>{escape(kind)}</kind>" if kind else "<kind>major</kind>"
    return (
        "<harmony>"
        f"<root><root-step>{step}</root-step>{alter_xml}</root>"
        f"{kind_xml}"
        "</harmony>"
    )


def _harmony_words_xml(root: str, kind: str) -> str:
    """The chord's READABLE name above the staff, next to its symbol.

    MusicXML carries symbols (<harmony>: 'Am'); a musician reads 'La
    mineur' (the i18n display table's whole-label translation, the same one
    the GUI and the CLI use). It travels as a `<direction><words>` — the
    standard, importable way to print text above the staff. Canonical
    'Am' stays in the JSON/CSV and in every harmonic structure; this is a
    DISPLAY-only addition (like the note lyrics).
    """
    label = tr_chord_label(f"{root}{kind}")
    return (
        "<direction placement='above'>"
        "<direction-type>"
        f"<words font-style='italic'>{escape(label)}</words>"
        "</direction-type>"
        "</direction>"
    )


@dataclass
class _TripletBlock:
    """A block of consecutive triplet notes (rendered as a group)."""

    notes: list
    start: float
    end: float


def _find_triplet_blocks(notes) -> list[_TripletBlock]:
    """Groups of >= 2 consecutive triplet notes (zero overlap allowed)."""
    trips = sorted(
        (n for n in notes if n.figure and n.figure.startswith("triolet")),
        key=lambda n: n.start,
    )
    blocks: list[_TripletBlock] = []
    cur: list = []
    for n in trips:
        if cur and n.start - cur[-1].end > 0.25:  # gap > 250 ms: new block
            if len(cur) >= 2:
                blocks.append(_TripletBlock(cur, cur[0].start, cur[-1].end))
            cur = []
        cur.append(n)
    if len(cur) >= 2:
        blocks.append(_TripletBlock(cur, cur[0].start, cur[-1].end))
    return blocks


def _render_triplet_block(
    block: _TripletBlock,
    grid: BeatGrid,
    colored: bool = False,
    with_lyric: bool = False,
) -> str:
    """Render the block: equal durations + start/stop bracket."""
    total_steps = max(1, grid.steps(block.end - block.start, per_beat=DIVISIONS))
    per = max(1, total_steps // len(block.notes))
    out = []
    for i, n in enumerate(block.notes):
        side = "start" if i == 0 else ("stop" if i == len(block.notes) - 1 else None)
        out.append(
            _note_xml(
                n.midi,
                per,
                tuplet_side=side,
                color=_confidence_color(n.confidence) if colored else None,
                lyric=midi_to_name_fr(n.midi) if with_lyric else None,
            )
        )
    return "".join(out)


def _rest_figure(t0: float, t1: float, rests) -> str | None:
    """Figure of the detected rest covering the most of [t0, t1].

    A labeled rest counts when its center falls inside the written gap; the
    longest overlap wins (a gap can straddle a rest boundary after
    snapping). Unlabeled or missing rests return None (generic fill).

    Used by the silent-run decomposition, so a labeled rest engraves as
    the figure a musician would write instead of generic quarters; the
    duration itself is what types each rest (`_note_xml`).
    """
    best, best_ov = None, 0.0
    for r in rests or []:
        fig = getattr(r, "figure", None)
        if not fig:
            continue
        ov = min(r.end, t1) - max(r.start, t0)
        if ov <= best_ov:
            continue
        mid = (r.start + r.end) / 2
        if t0 - 1e-6 <= mid <= t1 + 1e-6:
            best, best_ov = fig, ov
    return best


def _rest_is_breath(t0: float, t1: float, rests) -> bool:
    """True when a HEARD-breath rest covers most of the written gap [t0, t1].

    Same center rule as `_rest_figure`: after snapping, a silent run can
    straddle a rest's boundary — the overlap decides.
    """
    for r in rests or []:
        if not getattr(r, "breath", False):
            continue
        ov = min(r.end, t1) - max(r.start, t0)
        if ov <= 0:
            continue
        span = t1 - t0
        if ov >= 0.5 * span:
            return True
        mid = (r.start + r.end) / 2
        if t0 - 1e-6 <= mid <= t1 + 1e-6:
            return True
    return False


def build_score_xml(
    notes,
    grid: BeatGrid,
    key=None,
    title: str = "Transcription",
    snap: bool = True,
    chord_progression=None,
    with_note_colors: bool = True,
    with_solfa_lyrics: bool = True,
    rests=None,
) -> str:
    """Generate the full MusicXML (string).

    `grid`: the take's beat grid, carrying the tempo and the meter
    (core/grid.py); it decides the measure length and the fallback 4/4.
    `chord_progression`: [(m0, m1, ChordCandidate)] measure-aligned chords
    (core/chords.py) rendered as <harmony> elements; None -> no chord symbols.
    `with_note_colors`: color each notehead by detection confidence.
    `with_solfa_lyrics`: sol-fa note names (Do, Ré, Mi...) under each note.
    """
    bpm = grid.bpm
    if snap:
        notes = grid.snap_notes(notes)
    divisions = DIVISIONS
    grid_s = grid.slot_seconds           # one displayed beat
    grid_steps = grid.steps(grid_s, per_beat=DIVISIONS)   # per written beat
    measure_s = grid.measure_seconds
    meter = grid.meter
    # the numerator counts the notated beat (six EIGHTHS in 6/8); the export
    # writes DISPLAYED beats (two dotted quarters), so a measure is filled in
    # `n_slots` steps — the same unit `measure_seconds` is built from.
    slots_per_measure = meter.n_slots

    max_end = max((n.end for n in notes), default=0.0)
    total_measures = max(1, int(max_end / measure_s + 0.999))

    fifths = _key_fifths(key.best.tonic, key.best.mode) if key else 0
    key_label = key.best.label if key else None

    # measure-aligned chords -> quick lookup by starting measure (1-based)
    harmony_by_measure: dict[int, str] = {}
    harmony_words_by_measure: dict[int, str] = {}
    if chord_progression:
        for m0, m1, c in chord_progression:
            if m0 < 1:
                continue
            harmony_by_measure.setdefault(m0, _harmony_xml(c.root, c.quality))
            harmony_words_by_measure.setdefault(
                m0, _harmony_words_xml(c.root, c.quality)
            )

    # subtitle credit line: every top-level detection on one line
    credits = " · ".join(
        part
        for part in (
            f"Tempo: {bpm:.1f} BPM",
            f"Key: {key_label}" if key_label else None,
            f"Meter: {meter.label}",
            f"Chord: {chord_progression and ' / '.join(c.label for _, _, c in chord_progression[:8])}",
        )
        if part
    )

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n',
        '<score-partwise version="4.0">\n',
        f"  <work><work-title>{escape(title)}</work-title></work>\n",
        f"  <identification>\n    <creator type=\"lyricist\">{escape(credits)}</creator>\n"
        "  </identification>\n",
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
        f"        <time><beats>{meter.beats_per_measure}</beats><beat-type>{meter.beat_unit}</beat-type></time>\n",
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
        harmony = harmony_by_measure.get(m + 1)
        if harmony:
            out.append(f"      {harmony}\n")
            words = harmony_words_by_measure.get(m + 1)
            if words:
                out.append(f"      {words}\n")
        m_start = m * measure_s
        m_end = m_start + measure_s
        skip_until = m_start
        b = 0
        while b < slots_per_measure:
            b_start = m_start + b * grid_s
            b_end = b_start + grid_s
            if b_start < skip_until - 1e-6:
                b += 1
                continue  # already covered by a triplet block

            # --- triplet block starting in this beat, contained in the measure ---
            blk = next(
                (
                    k
                    for k in blocks
                    if abs(k.start - b_start) < grid.step_seconds() * 0.4
                    and k.end <= m_end + 1e-6
                    and k.start >= m_start - 1e-6
                ),
                None,
            )
            if blk is not None:
                out.append(
                    _render_triplet_block(
                        blk,
                        grid,
                        colored=with_note_colors,
                        with_lyric=with_solfa_lyrics,
                    )
                )
                skip_until = blk.end
                b += 1
                continue

            # --- subdivide the beat ---
            # One path only: every beat a note touches is written note by
            # note. A note keeps its own written duration and what is left of
            # the beat becomes a rest, so a short isolated note is no longer
            # stretched to the whole beat (nor its silence swallowed). A note
            # covering the whole beat still comes out as one beat-long note,
            # tied across when it continues — which is all the old
            # "beat owner" shortcut ever did.
            inside = [
                n
                for n in notes
                if min(n.end, b_end) - max(n.start, b_start) > MIN_OVERLAP_S
            ]
            inside.sort(key=lambda n: n.start)
            if inside:
                cursor = b_start
                for n in inside:
                    if n.start > cursor + 1e-6:
                        gap = max(1, grid.steps(n.start - cursor, per_beat=DIVISIONS))
                        out.append(_note_xml(None, gap))
                    seg_start = max(n.start, b_start)
                    seg_end = min(n.end, b_end)
                    steps = max(1, grid.steps(seg_end - seg_start, per_beat=DIVISIONS))
                    ties = []
                    if n.start < b_start - 1e-6:
                        ties.append("stop")
                    if n.end > b_end + 1e-6:
                        ties.append("start")
                    out.append(
                        _note_xml(
                            n.midi,
                            min(steps, grid_steps),
                            ties=tuple(ties),
                            color=_confidence_color(n.confidence) if with_note_colors else None,
                            lyric=midi_to_name_fr(n.midi) if with_solfa_lyrics else None,
                        )
                    )
                    cursor = seg_end
                if b_end - cursor > 1e-6:
                    tail_steps = max(1, grid.steps(b_end - cursor, per_beat=DIVISIONS))
                    out.append(_note_xml(None, tail_steps))
                b += 1
                continue

            # --- silent beat(s): fuse the run into proper rest figures ---
            # (an empty measure engraves one whole rest; a breath labeled by
            # rests.py shows its real shape — soupir, demi-pause...)
            run = 1
            while b + run < slots_per_measure:
                nb_start = m_start + (b + run) * grid_s
                nb_end = nb_start + grid_s
                if any(
                    min(n.end, nb_end) - max(n.start, nb_start) > MIN_OVERLAP_S
                    for n in notes
                ):
                    break
                run += 1
            run_steps = run * grid_steps
            run_end = m_start + (b + run) * grid_s
            out.append(
                _rest_run_xml(
                    run_steps,
                    _rest_figure(b_start, run_end, rests),
                    breath=_rest_is_breath(b_start, run_end, rests),
                )
            )
            b += run
        out.append("    </measure>\n")

    out.extend(["  </part>\n", "</score-partwise>\n"])
    return "".join(out)


def save_musicxml(
    notes,
    grid: BeatGrid,
    key,
    path: str,
    title: str = "Transcription",
    chord_progression=None,
    with_note_colors: bool = True,
    with_solfa_lyrics: bool = True,
    rests=None,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            build_score_xml(
                notes,
                grid,
                key,
                title=title,
                chord_progression=chord_progression,
                with_note_colors=with_note_colors,
                with_solfa_lyrics=with_solfa_lyrics,
                rests=rests,
            )
        )
