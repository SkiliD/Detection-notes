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


def test_quantize_aligns_to_grid():
    step = GRID.step_seconds(per_beat=4.0)  # sixteenth at 120 BPM
    assert abs(step - 0.125) < 1e-9
    mid = build_midi(
        [Note(start=0.13, end=0.49, midi=60, confidence=0.9)],
        GRID,
        key=None,
        quantize_s=step,
    )
    # 2026-09-25: the track also carries the pedal (CC64), so a note's
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
    """The sustain pedal (user request): a HELD note (>= HELD_MIN_S) writes
    CC64=127 at its attack and CC64=0 after its end + tail (the same
    arithmetic as playback.render_notes), so the file played on any synth
    keeps the resonance. A short note writes no CC64. The order at the
    same instant: pedal BEFORE the note (key 0), so the string rings from
    the attack."""
    from v2m.core import playback as pb
    from v2m.core.grid import BeatGrid

    notes = [
        Note(start=0.0, end=0.3, midi=60, confidence=0.9),   # short
        Note(start=0.4, end=1.2, midi=64, confidence=0.9),   # held
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
