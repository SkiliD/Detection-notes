"""meter.py tests: simple vs compound (6/8) detection, and positions rendered
by the Grid (grid.py) that the detection feeds."""

from v2m.core.grid import DEFAULT_METER, BeatGrid, Meter
from v2m.core.meter import detect_meter
from v2m.core.notes import Note

BPM = 120.0
BEAT = 60.0 / BPM  # 0.5 s


def _notes_from_starts(starts, dur=BEAT * 0.8, midi=60):
    return [Note(start=t, end=t + dur, midi=midi, confidence=0.9) for t in starts]


def test_binary_grid_no_conclusion():
    """Binary grid (attacks on beats and sixteenths): 2/4, 3/4 and 4/4 are
    indistinguishable without accents -> None (4/4 by default)."""
    starts = [i * BEAT for i in range(8)] + [i * BEAT + 0.375 for i in range(8)]
    assert detect_meter(_notes_from_starts(starts), BPM) is None


def test_detect_6_8_by_offbeats():
    """6/8: attacks on dotted beats 1 and 2 (eighths 1 and 4)."""
    eighth = 60.0 / BPM / 2
    starts = [m * 1.5 + e for m in range(3) for e in (0.0, 3 * eighth)]
    meter = detect_meter(_notes_from_starts(starts), BPM)
    assert meter is not None and meter.label == "6/8"
    assert meter.score >= 0.6


def test_6_8_beats_fall_between_binary_slots():
    """The case that breaks a 4/4 rendering: the 6/8's 2nd dotted beat falls
    on the binary "and" (0.75 quarter) — undetectable from a simple grid."""
    eighth = 60.0 / BPM / 2
    second_slot = 3 * eighth  # 0.75 s = 1.5 quarters
    assert abs((second_slot / (60.0 / BPM)) - 1.5) < 1e-9


def test_binary_only_returns_none():
    """Strictly binary grid (beats and eighths): nothing on the ternary
    dotted beats -> no conclusion."""
    eighth = BEAT / 2
    starts = [i * BEAT for i in range(8)] + [i * BEAT + eighth for i in range(8)]
    assert detect_meter(_notes_from_starts(starts), BPM) is None


def test_a_binary_4_4_line_is_not_6_8():
    """A binary 4/4 line never reads 6/8: its attacks often land on the
    ternary grid (0.71 on the reference motif) — that is why alignment
    alone does not decide, the simple grid must be clearly less well
    explained (0.86).
    """
    starts = [i * BEAT + h * BEAT / 2 for i in range(4) for h in (0, 1)]
    meter = detect_meter(_notes_from_starts(starts), BPM)
    assert meter is None or meter.label != "6/8"


def test_too_few_notes_returns_no_meter():
    notes = _notes_from_starts([0.0, BEAT, 2 * BEAT])
    assert detect_meter(notes, BPM) is None


def test_default_meter():
    assert DEFAULT_METER.label == "4/4"
    assert not DEFAULT_METER.is_compound
    assert BeatGrid(BPM, DEFAULT_METER).measure_seconds == 2.0


def test_note_positions():
    meter = Meter(3, 4)
    notes = _notes_from_starts([0.0, BEAT, 2 * BEAT, 3 * BEAT, 4 * BEAT])
    assert BeatGrid(BPM, meter).positions(notes) == [
        (1, 1.0), (1, 2.0), (1, 3.0), (2, 1.0), (2, 2.0),
    ]


def test_note_positions_compound():
    meter = Meter(6, 8)
    eighth = 60.0 / BPM / 2
    notes = _notes_from_starts([0.0, 3 * eighth, 1.5, 1.5 + 3 * eighth])
    assert BeatGrid(BPM, meter).positions(notes) == [
        (1, 1.0), (1, 2.0), (2, 1.0), (2, 2.0),
    ]


def test_compound_meter_length():
    """6/8 = 6 eighths = 3 quarters = 1.5 s at 120 BPM; 2 dotted beats displayed."""
    meter = Meter(6, 8)
    assert meter.measure_quarters == 3.0
    assert BeatGrid(BPM, meter).measure_seconds == 1.5
    assert meter.n_slots == 2
    assert meter.slot_quarters == 1.5
    assert meter.is_compound


def test_reads_dotted_pulse():
    """Compound stress is a strong/weak alternation of period 2.

    A timing-only discriminator is impossible: uniform gaps describe
    "quarters at 2/3 of the tempo" just as well as "dotted beats".
    The alternation decides — hence the attack strength (live and
    offline alike) feeding the dotted-beat reading.
    """
    from v2m.core.meter import reads_dotted_pulse

    # strict strong/weak alternation: the compound-stress signature
    dotted = _notes_from_starts([i * 1.5 for i in range(9)])
    for i, n in enumerate(dotted):
        n.strength = 0.9 if i % 2 == 0 else 0.2
    assert reads_dotted_pulse(dotted)

    # a flat beat (all strengths equal) does not alternate: nothing to conclude
    flat = _notes_from_starts([i * 0.5 for i in range(12)])
    for n in flat:
        n.strength = 0.5
    assert not reads_dotted_pulse(flat)

    # fewer than 4 notes: no observable period
    few = _notes_from_starts([0.0, 0.5, 1.0])
    for i, n in enumerate(few):
        n.strength = 0.9 if i % 2 == 0 else 0.2
    assert not reads_dotted_pulse(few)


def test_two_runs_of_constant_level_never_read_dotted():
    """Two RUNS of constant level (a loud phrase, a soft one) never read
    as a dotted pulse: they validate each other at lag 2 (`111111000000000`
    scores 11/13 on ALTERNATE_SHARE and passes MINORITY_SHARE) but a true
    alternation CHANGES label at almost every step, runs only at their
    boundary (1 change in 14). The change-rate guard refuses the family
    without touching the true alternation.
    """
    from v2m.core.meter import reads_dotted_pulse

    runs = _notes_from_starts([i * 0.5 for i in range(15)])
    for i, n in enumerate(runs):
        n.strength = 0.9 if i < 6 else 0.1  # 111111000000000
    assert not reads_dotted_pulse(runs)

    # the healthy alternation at the SAME note count still reads
    alternating = _notes_from_starts([i * 0.5 for i in range(15)])
    for i, n in enumerate(alternating):
        n.strength = 0.9 if i % 2 == 0 else 0.1
    assert reads_dotted_pulse(alternating)


def test_detect_meter_6_8_from_dotted_pulse():
    """A take on dotted beats IS published as 6/8 when the stress
    alternates (strong/weak) — the compound-stress signature.

    It is the live mirror of `test_detect_6_8_by_offbeats`: at 100 BPM,
    the medians lived as quarters of a 67-BPM (the x2 fold never repairs
    the x1.5 ratio) and the compound meter was never published.
    detect_meter now skips the accents stage (which would read the
    ternary grid's phases 0/2 as a waltz) to go straight to the ternary
    grid.
    """
    starts = [i * 0.9 for i in range(10)]  # dotted beats at 100 BPM
    notes = _notes_from_starts(starts, dur=0.75)
    for i, n in enumerate(notes):
        n.strength = 0.9 if i % 2 == 0 else 0.2
    meter = detect_meter(notes, 100.0)
    assert meter is not None and meter.label == "6/8"
    assert meter.score >= 0.6
