"""MIDI export tests: tempo/key-signature meta-events + re-read notes."""

import mido
import pytest

from v2m.core.grid import BeatGrid
from v2m.core.keys import detect_key
from v2m.core.midi_export import build_midi
from v2m.core.notes import Note

GRID = BeatGrid(120.0)


def _sample_notes() -> list[Note]:
    return [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),
        Note(start=0.5, end=1.0, midi=64, confidence=0.8),
        Note(start=1.2, end=2.0, midi=67, confidence=0.7),
    ]


def test_midi_has_tempo_meta(tmp_path):
    mid = build_midi(_sample_notes(), GRID, key=None)
    path = str(tmp_path / "out.mid")
    mid.save(path)
    reloaded = mido.MidiFile(path)
    tempos = [m for t in reloaded.tracks for m in t if m.type == "set_tempo"]
    assert len(tempos) == 1
    assert abs(mido.tempo2bpm(tempos[0].tempo) - 120.0) < 0.1


def test_midi_has_key_signature(tmp_path):
    notes = _sample_notes()
    key = detect_key(notes)
    mid = build_midi(notes, GRID, key=key)
    path = str(tmp_path / "out.mid")
    mid.save(path)
    reloaded = mido.MidiFile(path)
    keys = [m for t in reloaded.tracks for m in t if m.type == "key_signature"]
    assert len(keys) == 1
    # mido decodes the key signature as a string ('C', 'Am', 'F#', ...)
    k = keys[0].key
    assert isinstance(k, str) and k
    valid = {"Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"}
    base = k[:-1] if k.endswith("m") else k
    assert base in valid


def test_midi_note_count_and_pitch(tmp_path):
    mid = build_midi(_sample_notes(), GRID, key=None)
    path = str(tmp_path / "out.mid")
    mid.save(path)
    reloaded = mido.MidiFile(path)
    ons = [m for t in reloaded.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    assert sorted(m.note for m in ons) == [60, 64, 67]


def test_midi_transpose_and_velocity():
    mid = build_midi(_sample_notes(), GRID, key=None, transpose=12)
    ons = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    assert sorted(m.note for m in ons) == [72, 76, 79]
    # velocity tied to confidence (0.9 -> high)
    assert all(40 <= m.velocity <= 127 for m in ons)


@pytest.mark.parametrize("quantize", [None, 0.125])
def test_repeated_key_releases_before_restrike_at_the_same_tick(tmp_path, quantize):
    notes = [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),
        Note(start=0.5, end=1.0, midi=60, confidence=0.9),
    ]
    mid = build_midi(notes, GRID, key=None, quantize_s=quantize)
    path = tmp_path / "repeat.mid"
    mid.save(path)
    events = []
    tick = 0
    for event in mido.MidiFile(path).tracks[1]:
        tick += event.time
        if tick == 480 and event.type in ("note_on", "note_off"):
            events.append(event.type)
    assert events == ["note_off", "note_on"]


def test_a_same_pitch_repeat_that_overlaps_releases_the_note_before_it(tmp_path):
    """The earlier note lets go at the repeat's attack, so the repeat keeps its length.

    One key sounds one note at a time, and the analysis can leave a same-pitch
    repeat starting 20 ms before the note before it ends (measured on
    chant.wav). Written as-is, that earlier note's OFF lands after the repeat's
    ON and closes the repeat: it would last the overlap (20 ticks) instead of
    its own 518.
    """
    notes = [
        Note(start=0.0, end=0.98, midi=60, confidence=0.9),
        Note(start=0.96, end=1.50, midi=60, confidence=0.9),
    ]
    mid = build_midi(notes, GRID, key=None)
    path = tmp_path / "overlap.mid"
    mid.save(path)

    events = []
    tick = 0
    for event in mido.MidiFile(path).tracks[1]:
        tick += event.time
        if event.type == "note_on" and event.velocity > 0:
            events.append(("on", tick))
        elif event.type == "note_off":
            events.append(("off", tick))

    # GRID is 120 BPM: one beat (0.5 s) is 480 ticks, so 0.96 s is tick 922.
    assert events == [("on", 0), ("off", 922), ("on", 922), ("off", 1440)]


def test_notes_of_different_pitches_sharing_a_tick_keep_their_time_order(tmp_path):
    """One written tick can hold events of DIFFERENT times: only the exact same
    instant is ordered by kind (pedal, off, on). A note-off and the next note's
    on 0.4 ms apart land on one tick, and the pedal edge between them keeps its
    place by its own time — a kind sort over the whole tick would move the
    pedal ahead of an earlier note-off and the new note's on ahead of the
    note-off that precedes it.
    """
    notes = [
        Note(start=0.30, end=0.5000, midi=60, confidence=0.9),  # short: no pedal
        Note(start=0.5004, end=1.0, midi=64, confidence=0.9),  # held: pedal down
    ]
    mid = build_midi(notes, GRID, key=None)  # 480 tpq at 120 BPM: one tick = 1.04 ms
    path = tmp_path / "shared-tick.mid"
    mid.save(path)

    written = []
    tick = 0
    for event in mido.MidiFile(path).tracks[1]:
        tick += event.time
        if tick != 480:
            continue
        if event.type == "control_change" and event.control == 64:
            written.append(("pedal", event.value))
        elif event.type == "note_on" and event.velocity > 0:
            written.append(("on", event.note))
        elif event.type == "note_off":
            written.append(("off", event.note))

    assert written == [("off", 60), ("pedal", 127), ("on", 64)]


def test_a_repeat_that_overlaps_the_note_before_it_keeps_its_own_length():
    """A repeat the analysis leaves OVERLAPPING keeps its length in the file.

    The segmenter clamps a note's end to the next start, but an analysis can
    still leave a same-pitch repeat INSIDE the note before it: measured 20 ms
    on chant.wav (pitches 53 and 57) and on demo.wav (pitch 60, twice). Both
    events land on one key, so the earlier note's OFF — written where the
    analysis put it, after the repeat's attack — closes the repeat at the
    overlap, and the repeat lasts 20 ms instead of its own length. The earlier
    note is released AT the repeat's attack, so the two spans touch and each
    keeps the length the analysis gave it.
    """
    notes = [
        Note(start=0.0, end=0.52, midi=60, confidence=0.9),
        Note(start=0.50, end=1.00, midi=60, confidence=0.9),
    ]
    mid = build_midi(notes, GRID, key=None)
    spans: list[tuple[int, int]] = []
    tick = 0
    opened: int | None = None
    for message in mid.tracks[1]:
        tick += message.time
        if getattr(message, "note", None) != 60 or message.type not in ("note_on", "note_off"):
            continue
        if message.type == "note_on" and message.velocity:
            opened = tick
        elif opened is not None:
            spans.append((opened, tick))
            opened = None
    # 120 BPM, 480 ticks per beat: 0.50 s is tick 480, 1.00 s is tick 960.
    assert spans == [(0, 480), (480, 960)], (
        f"the two spans must touch at the repeat's attack, not overlap: {spans}"
    )


@pytest.mark.parametrize(
    "mode,tonic,shift,expected",
    [
        ("major", "C", 2, "D"),
        ("major", "B", 2, "C#"),
        ("major", "C", -2, "Bb"),
        ("minor", "A", 2, "Bm"),
        ("minor", "D#", 1, "Em"),
        ("minor", "C", -1, "Bm"),
    ],
)
def test_transpose_moves_key_signature_with_notes(tmp_path, mode, tonic, shift, expected):
    from v2m.core.keys import KeyCandidate, KeyResult

    key = KeyResult(best=KeyCandidate(tonic, mode, 1.0), alternatives=[])
    mid = build_midi(_sample_notes(), GRID, key=key, transpose=shift)
    path = tmp_path / "transposed.mid"
    mid.save(path)
    loaded = mido.MidiFile(path)
    signatures = [e.key for e in loaded.tracks[0] if e.type == "key_signature"]
    assert signatures == [expected]
    assert [e.note for e in loaded.tracks[1] if e.type == "note_on"] == [
        60 + shift,
        64 + shift,
        67 + shift,
    ]


def test_quantize_aligns_to_grid():
    step = GRID.step_seconds(per_beat=4.0)  # sixteenth at 120 BPM
    assert abs(step - 0.125) < 1e-9
    mid = build_midi(
        [Note(start=0.13, end=0.49, midi=60, confidence=0.9)],
        GRID,
        key=None,
        quantize_s=step,
    )
    # the track also carries the pedal (CC64), so a note's
    # RELATIVE tick is no longer its position — we read the ABSOLUTE time.
    abs_times: list[tuple[str, float]] = []
    for t in mid.tracks:
        acc = 0
        for m in t:
            acc += m.time
            abs_times.append((m.type, acc))
    # 0.13 -> rounds to 0.125 s = 1 sixteenth = 120 ticks (480 tpq, 120 BPM)
    ons = [tick for kind, tick in abs_times if kind == "note_on"]
    assert ons == [120]


def test_a_held_note_writes_sustain_pedal_events():
    """A held note writes sustain over its duration and render-matched tail."""
    from v2m.core import playback as pb
    from v2m.core.grid import BeatGrid

    notes = [
        Note(start=0.0, end=0.3, midi=60, confidence=0.9),  # short
        Note(start=0.4, end=1.2, midi=64, confidence=0.9),  # held
    ]
    mid = build_midi(notes, BeatGrid(bpm=120.0, meter=None), key=None)
    # GRID is at 120 BPM -> MIDI ticks in beats = seconds x 2: we read
    # times in SECONDS to compare them to the notes.
    sec_per_beat = 0.5
    seq: list[tuple[str, float, int]] = []
    for t in mid.tracks:
        acc = 0
        for m in t:
            acc += m.time
            if m.type == "control_change" and m.control == 64:
                seq.append(("cc", acc / 480 * sec_per_beat, m.value))
            elif m.type == "note_on":
                seq.append(("on", acc / 480 * sec_per_beat, m.note))
    # the short note produces nothing; the held one produces down@0.4 and
    # up@1.2 + 0.30x0.8 = 1.44 s (same arithmetic as the render)
    downs = [x for x in seq if x[2] == 127]
    ups = [x for x in seq if x[2] == 0]
    assert len(downs) == 1 and len(ups) == 1
    assert downs[0][1] == pytest.approx(0.4, abs=0.01)
    assert ups[0][1] == pytest.approx(1.2 + pb.PEDAL_TAIL_FACTOR * 0.8, abs=0.01)
    # at the same instant, the pedal comes before the note
    order = [(kind, tick) for kind, tick, _ in seq]
    assert order.index(("cc", 0.4)) < order.index(("on", 0.4))


def test_overlapping_held_notes_keep_sustain_down_until_both_tails_end():
    """One note's tail must not release the global pedal under another note."""
    from v2m.core.playback import PEDAL_TAIL_FACTOR

    notes = [
        Note(start=0.0, end=0.8, midi=60, confidence=0.9),
        Note(start=0.4, end=1.2, midi=64, confidence=0.9),
    ]
    mid = build_midi(notes, GRID, key=None)

    pedal_events: list[tuple[float, int]] = []
    for track in mid.tracks:
        tick = 0
        for event in track:
            tick += event.time
            if event.type == "control_change" and event.control == 64:
                pedal_events.append((tick / 960.0, event.value))

    assert [value for _at, value in pedal_events] == [127, 0]
    assert pedal_events[0][0] == pytest.approx(0.0)
    assert pedal_events[1][0] == pytest.approx(1.44, abs=0.001)
    assert not any(value == 0 and at < 1.2 for at, value in pedal_events)
    assert PEDAL_TAIL_FACTOR == 0.30


def test_adjacent_held_notes_do_not_create_a_pedal_gap():
    """A repeated note at the first note's release keeps the pedal continuous."""
    notes = [
        Note(start=0.0, end=0.6, midi=60, confidence=0.9),
        Note(start=0.6, end=1.0, midi=62, confidence=0.9),
    ]
    mid = build_midi(notes, GRID, key=None)

    pedal_events: list[tuple[float, int]] = []
    for track in mid.tracks:
        tick = 0
        for event in track:
            tick += event.time
            if event.type == "control_change" and event.control == 64:
                pedal_events.append((tick / 960.0, event.value))

    assert [value for _at, value in pedal_events] == [127, 0]
    assert pedal_events[0][0] == pytest.approx(0.0)
    assert pedal_events[1][0] == pytest.approx(1.12, abs=0.001)


def test_non_overlapping_held_notes_release_between_tails():
    """A real gap between sustain windows still produces a pedal release."""
    notes = [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),
        Note(start=1.0, end=1.5, midi=64, confidence=0.9),
    ]
    mid = build_midi(notes, GRID, key=None)

    pedal_events: list[tuple[float, int]] = []
    for track in mid.tracks:
        tick = 0
        for event in track:
            tick += event.time
            if event.type == "control_change" and event.control == 64:
                pedal_events.append((tick / 960.0, event.value))

    assert [value for _at, value in pedal_events] == [127, 0, 127, 0]
    assert pedal_events[1][0] == pytest.approx(0.65)
    assert pedal_events[2][0] == pytest.approx(1.0)
    assert pedal_events[3][0] == pytest.approx(1.65)


def test_quantized_held_note_uses_quantized_sustain_window():
    """Pedal duration follows the written, quantized note event times."""
    note = Note(start=0.13, end=0.49, midi=60, confidence=0.9)
    mid = build_midi([note], GRID, key=None, quantize_s=0.125)

    pedal_events: list[tuple[float, int]] = []
    for track in mid.tracks:
        tick = 0
        for event in track:
            tick += event.time
            if event.type == "control_change" and event.control == 64:
                pedal_events.append((tick / 960.0, event.value))

    # Written span is 0.125..0.5 (>= HELD_MIN_S after quantization).
    assert [value for _at, value in pedal_events] == [127, 0]
    assert pedal_events[0][0] == pytest.approx(0.125)
    assert pedal_events[1][0] == pytest.approx(0.6125, abs=0.001)
