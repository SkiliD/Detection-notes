"""MusicXML export: readable score built from the analysis.

Principles:
- divisions = 12 (common multiple of 3 and 4): every plain, dotted AND
  triplet figure has an integer duration in divisions.
- 1 quarter beat = 12 divisions, a 4/4 measure = 48 divisions.
- Each beat is attributed to the note overlapping it most (30 ms threshold).
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

from .meter import Meter
from .music import midi_to_name_fr

_DEFAULT_METER = Meter(4, 4, 1.0)

STEP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# divisions=12: steps -> (MusicXML type, number of dots)
_TYPE_BY_STEPS = {
    48: ("whole", 0),
    36: ("half", 1),
    32: ("whole", 0),      # whole triplet
    24: ("half", 0),
    18: ("quarter", 1),
    16: ("half", 0),       # half triplet
    12: ("quarter", 0),
    9: ("eighth", 1),
    8: ("quarter", 0),     # quarter triplet
    6: ("eighth", 0),
    4: ("eighth", 0),      # eighth triplet
    3: ("16th", 0),
    2: ("16th", 0),        # sixteenth triplet
    1: ("32nd", 0),
}

_TUPLET_STEPS = {32, 16, 8, 4, 2}  # steps that correspond to triplets

# Canonical French rest figures (core/rests.py) -> (MusicXML type, dots).
# Used to label the written silences with their real shape instead of the
# duration-table fallback (a 0.38 s breath reads as a soupir, not as a
# quarter rest that "happens" to last that long).
_REST_TYPE_BY_FIGURE = {
    "pause": ("whole", 0),
    "demi-pause": ("half", 0),
    "soupir": ("quarter", 0),
    "demi-soupir": ("eighth", 0),
    "quart de soupir": ("16th", 0),
    "pause pointée": ("whole", 1),
    "demi-pause pointée": ("half", 1),
    "soupir pointée": ("quarter", 1),
    "demi-soupir pointée": ("eighth", 1),
}

# Rest decomposition order for unlabeled silent spans: plain values only
# (no triplet flavours — a half TRIPLET rest is not a notation), largest
# first, so an empty measure engraves ONE whole rest instead of four
# quarter rests.
_REST_FILL_ORDER = [
    (steps, type_dots)
    for steps, type_dots in sorted(_TYPE_BY_STEPS.items(), reverse=True)
    if steps not in _TUPLET_STEPS
]


def _rest_run_xml(steps: int, figure: str | None) -> str:
    """One or more rest <note> elements covering exactly `steps` divisions.

    A labeled figure (soupir, demi-pause...) is engraved as many times as
    it fits the span; the remainder — and unlabeled runs — fall back to the
    largest drawable plain value, so a silent run of N beats reads as the
    rest an engraver would write, not a sequence of generic quarters.
    """
    parts: list[str] = []
    if figure in _REST_TYPE_BY_FIGURE:
        want_type, want_dots = _REST_TYPE_BY_FIGURE[figure]
        fig_steps = max(
            (
                s
                for s, (ty, do) in _TYPE_BY_STEPS.items()
                if (ty, do) == (want_type, want_dots)
                and s <= steps
                and s not in _TUPLET_STEPS
            ),
            default=0,
        )
        while fig_steps and steps >= fig_steps:
            parts.append(_note_xml(None, fig_steps, figure=figure))
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
        t = STEP_NAMES[(STEP_NAMES.index(t) + 3) % 12]  # relative majeure
    # the table covers sharps AND common flats; enharmonic fallback otherwise
    if t in _FIFTHS_MAJ:
        return _FIFTHS_MAJ[t]
    return _FIFTHS_MAJ.get(_ENH.get(t, t), 0)


def _pitch_xml(midi: int) -> str:
    name = STEP_NAMES[midi % 12]
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
    figure: str | None = None,
) -> str:
    """One <note> (or a rest when midi is None).

    For rests, `figure` (a canonical French rest name) overrides the
    duration-derived type when it is a known shape, so the engraved silence
    shows its real value (soupir, demi-pause...).
    """
    steps = max(1, int(steps))
    type_, dots = _TYPE_BY_STEPS.get(steps, ("quarter", 0))
    if midi is None and figure in _REST_TYPE_BY_FIGURE:
        type_, dots = _REST_TYPE_BY_FIGURE[figure]
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
    bpm: float,
    colored: bool = False,
    with_lyric: bool = False,
) -> str:
    """Render the block: equal durations + start/stop bracket."""
    beat_s = 60.0 / bpm
    total_steps = max(1, int(round((block.end - block.start) / beat_s * 12)))
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


def _snap_notes(notes, bpm: float, divisions_per_beat: int = 4):
    """Snap boundaries to the sixteenth grid.

    The detected tempo drifts slightly from the real human tempo; without
    snapping, boundaries fall off-grid and generate fragments (16th/32nd).
    Each boundary joins its nearest grid step.
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
    """Note covering this beat most (None if overlap < 30 ms)."""
    best, best_ov = None, 0.030
    for n in notes:
        ov = min(n.end, b_end) - max(n.start, b_start)
        if ov > best_ov:
            best, best_ov = n, ov
    return best


def _rest_figure(t0: float, t1: float, rests) -> str | None:
    """Figure of the detected breath covering the most of [t0, t1].

    A labeled rest counts when its center falls inside the written gap; the
    longest overlap wins (a gap can straddle a rest boundary after
    snapping). Unlabeled or missing rests return None (generic fill).
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


def build_score_xml(
    notes,
    bpm: float,
    key=None,
    title: str = "Transcription",
    snap: bool = True,
    meter=None,
    chord_progression=None,
    with_note_colors: bool = True,
    with_solfa_lyrics: bool = True,
    rests=None,
) -> str:
    """Generate the full MusicXML (string).

    `meter`: detected Meter (core/meter.py); None -> 4/4 default.
    `chord_progression`: [(m0, m1, ChordCandidate)] measure-aligned chords
    (core/chords.py) rendered as <harmony> elements; None -> no chord symbols.
    `with_note_colors`: color each notehead by detection confidence.
    `with_solfa_lyrics`: sol-fa note names (Do, Ré, Mi...) under each note.
    """
    if snap:
        notes = _snap_notes(notes, bpm)
    if meter is None:
        meter = _DEFAULT_METER
    divisions = 12
    beat_s = 60.0 / bpm                                  # quarter-beat seconds
    grid_s = beat_s * (3.0 if meter.is_compound else 1.0)  # one displayed beat
    grid_steps = int(round(grid_s / beat_s * 12))          # divisions per beat
    measure_s = meter.beats_per_measure * grid_s

    max_end = max((n.end for n in notes), default=0.0)
    total_measures = max(1, int(max_end / measure_s + 0.999))

    fifths = _key_fifths(key.best.tonic, key.best.mode) if key else 0
    key_label = key.best.label if key else None

    # measure-aligned chords -> quick lookup by starting measure (1-based)
    harmony_by_measure: dict[int, str] = {}
    if chord_progression:
        for m0, m1, c in chord_progression:
            if m0 < 1:
                continue
            harmony_by_measure.setdefault(m0, _harmony_xml(c.root, c.quality))

    # subtitle credit line: every top-level detection on one line
    credits = " · ".join(
        part
        for part in (
            f"Tempo: {bpm:.1f} BPM",
            f"Key: {key_label}" if key_label else None,
            f"Meter: {meter.label}" if meter else None,
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
        m_start = m * measure_s
        m_end = m_start + measure_s
        skip_until = m_start
        b = 0
        while b < meter.beats_per_measure:
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
                    if abs(k.start - b_start) < beat_s * 0.4
                    and k.end <= m_end + 1e-6
                    and k.start >= m_start - 1e-6
                ),
                None,
            )
            if blk is not None:
                out.append(
                    _render_triplet_block(
                        blk,
                        bpm,
                        colored=with_note_colors,
                        with_lyric=with_solfa_lyrics,
                    )
                )
                skip_until = blk.end
                b += 1
                continue

            # --- whole beat held by a single note ---
            owner = _beat_owner(notes, b_start, b_end)
            beat_dur = b_end - b_start
            # a significant concurrent note (> 20% of the beat) forces
            # subdivision; below that it is a detection artifact absorbed
            # into the dominant note (a real sixteenth covers 25% of a beat,
            # so it is preserved)
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
                out.append(
                    _note_xml(
                        owner.midi,
                        grid_steps,
                        ties=tuple(ties),
                        color=_confidence_color(owner.confidence) if with_note_colors else None,
                        lyric=midi_to_name_fr(owner.midi) if with_solfa_lyrics else None,
                    )
                )
                b += 1
                continue

            # --- subdivide the beat ---
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
                        out.append(_note_xml(None, gap, figure=_rest_figure(cursor, n.start, rests)))
                    seg_start = max(n.start, b_start)
                    seg_end = min(n.end, b_end)
                    steps = max(1, int(round((seg_end - seg_start) / beat_s * 12)))
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
                    tail_steps = max(1, int(round((b_end - cursor) / beat_s * 12)))
                    out.append(_note_xml(None, tail_steps, figure=_rest_figure(cursor, b_end, rests)))
                b += 1
                continue

            # --- silent beat(s): fuse the run into proper rest figures ---
            # (an empty measure engraves one whole rest; a breath labeled by
            # rests.py shows its real shape — soupir, demi-pause...)
            run = 1
            while b + run < meter.beats_per_measure:
                nb_start = m_start + (b + run) * grid_s
                nb_end = nb_start + grid_s
                if _beat_owner(notes, nb_start, nb_end) is not None or any(
                    min(n.end, nb_end) - max(n.start, nb_start) > 0.030 for n in notes
                ):
                    break
                run += 1
            run_steps = run * grid_steps
            run_end = m_start + (b + run) * grid_s
            out.append(
                _rest_run_xml(run_steps, _rest_figure(b_start, run_end, rests))
            )
            b += run
        out.append("    </measure>\n")

    out.extend(["  </part>\n", "</score-partwise>\n"])
    return "".join(out)


def save_musicxml(
    notes,
    bpm: float,
    key,
    path: str,
    title: str = "Transcription",
    meter=None,
    chord_progression=None,
    with_note_colors: bool = True,
    with_solfa_lyrics: bool = True,
    rests=None,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            build_score_xml(
                notes,
                bpm,
                key,
                title=title,
                meter=meter,
                chord_progression=chord_progression,
                with_note_colors=with_note_colors,
                with_solfa_lyrics=with_solfa_lyrics,
                rests=rests,
            )
        )
