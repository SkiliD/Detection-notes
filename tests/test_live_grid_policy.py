"""The live grid policy has ONE owner: LiveTake (core.live_session).

The policy is "estimate an approximate BPM from the notes so far, keep the
take's BeatGrid, label the notes' figures on it". During the LiveTake
refactor it existed in THREE places: LiveTake._on_block (before emitting
note_closed), the CLI adapter LiveSession._on_event (again, on the same
list) and the GUI tick (a third BeatGrid(estimate_bpm_live(...)) rebuilt
on every note-count change). Identical outcomes today, but three owners
ready to drift — and O(n) relabels per note close.

The design pinned here: LiveTake writes the grid (property `grid`), labels
the figures BEFORE emitting `note_closed`, and publishes the grid ON the
event — the METER included, detected from the closed notes' attack
strengths and carried inside the same grid. Adapters (CLI and GUI) consume
a ready-labeled note and a ready grid, and never call the policy entry
points themselves.

The adapter test fails on the pre-dedup code (`_on_event` called
`estimate_bpm_live` itself); the publication tests fail on the draft where
the grid stayed private to LiveTake.
"""

from __future__ import annotations

import numpy as np

from v2m.core import live_session
from v2m.core.liveseg import LiveNote
from v2m.core.live_session import LiveEvent


class FakeCapture:
    """Stand-in for LiveCapture (same shape as tests/test_live_session.py)."""

    samplerate = 22_050

    def __init__(self, **kwargs):
        self.on_block = kwargs["on_block"]
        self.started = False
        self.stop_calls = 0
        self._signal = np.zeros(22_050, dtype=np.float32)

    def start(self):
        self.started = True

    def stop(self):
        self.stop_calls += 1

    def elapsed(self):
        return 2.0

    def signal(self):
        return self._signal


class StubSeg:
    def __init__(self):
        # three previous notes; the fourth closes during the block
        self.notes = [
            LiveNote(0.0, 0.45, 69),
            LiveNote(0.5, 0.95, 71),
            LiveNote(1.0, 1.45, 72),
        ]

    def current(self):
        return None


class StubStream:
    def __init__(self, sr, gate_rms):
        self.seg = StubSeg()

    def feed(self, block, t):
        self.seg.notes.append(LiveNote(1.5, 2.0, 69))
        return 0.01, []

    def elapsed_notes(self):
        return len(self.seg.notes)


def test_livetake_labels_figures_and_publishes_the_grid(monkeypatch):
    """The take labels the closed note AND carries its grid on the event."""

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", StubStream)

    events = []
    take = live_session.LiveTake(on_event=events.append)
    assert take.grid is not None  # a fallback grid exists from the start
    take.start()
    take._on_block(np.zeros(512, dtype=np.float32), 2.0)

    closed = [e for e in events if e.kind == "note_closed"]
    assert len(closed) == 1
    # starts 0 / .5 / 1 / 1.5 s -> 120 BPM -> 0.5 s = one beat = "noire"
    assert closed[0].note.figure == "noire"
    # the grid travels ON the event and IS the take's grid (one object):
    # adapters never rebuild one from the notes themselves
    assert closed[0].grid is not None
    assert closed[0].grid.bpm == 120.0
    assert closed[0].grid is take.grid


def test_cli_adapter_does_not_rederive_the_grid(monkeypatch):
    """The CLI adapter prints the labeled note; it never re-labels.

    The spies replace the policy entry points INSIDE the adapter module:
    on the pre-dedup code, `_on_event` called `estimate_bpm_live` itself
    and this test failed.
    """
    import v2m.live as live_mod

    calls: list[str] = []
    monkeypatch.setattr(
        live_mod, "estimate_bpm_live", lambda *a, **k: calls.append("bpm"),
        raising=False,
    )
    monkeypatch.setattr(
        live_mod, "label_notes", lambda *a, **k: calls.append("label"),
        raising=False,
    )

    session = live_mod.LiveSession()
    note = LiveNote(0.0, 0.5, 69, figure="noire")
    session._on_event(LiveEvent("note_closed", time_s=0.5, note=note))

    assert calls == [], "l'adaptateur CLI ne doit pas redériver la grille live"


class AccentSeg:
    """A segmenter whose notes close one per block, with accent strengths."""

    def __init__(self):
        self.notes = []

    def current(self):
        return None


class AccentStream:
    """One closed note per block: every 0.4 s, accented every third.

    0.4 s gaps -> 150 BPM (estimate_bpm_live); a strength of 1.0 on every
    third onset and 0.3 elsewhere is the waltz pattern `detect_meter` reads
    (the same shapes tests/test_meter.py pins).
    """

    def __init__(self, sr, gate_rms):
        self.seg = AccentSeg()

    def feed(self, block, t):
        i = len(self.seg.notes)
        self.seg.notes.append(
            LiveNote(
                start=i * 0.4, end=i * 0.4 + 0.3, midi=69,
                strength=(1.0 if i % 3 == 0 else 0.3),
            )
        )
        return 0.01, []

    def elapsed_notes(self):
        return len(self.seg.notes)


class DottedStream(AccentStream):
    """One note per dotted beat (1.5 s), strong/weak alternation: strict
    6/8 material — the compound stress whose signature reads_dotted_pulse
    looks for.
    """

    def feed(self, block, t):
        i = len(self.seg.notes)
        self.seg.notes.append(
            LiveNote(start=i * 1.5, end=i * 1.5 + 1.2, midi=69,
                     strength=(0.9 if i % 2 == 0 else 0.2))
        )
        return 0.01, []


class FlatDottedStream(DottedStream):
    """Same dotted spacing, but a FLAT pulse: no alternation, no reading."""

    def feed(self, block, t):
        i = len(self.seg.notes)
        self.seg.notes.append(LiveNote(start=i * 1.5, end=i * 1.5 + 1.2, midi=69, strength=0.8))
        return 0.01, []


def test_livetake_publishes_the_detected_meter(monkeypatch):
    """The take's grid carries the METER the accents reveal.

    Below meter.MIN_NOTES the detection stays silent and the grid falls
    back on the default 4/4; from the sixth note on, the waltz pattern tips
    the published grid to 3 displayed beats. The GUI metronome reads exactly
    this — it never calls detect_meter itself (the old path CRASHED there:
    LiveNote has no strength, so the accents stage raised AttributeError).
    """
    from v2m.core.grid import DEFAULT_METER

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", AccentStream)

    events = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    block = np.zeros(512, dtype=np.float32)
    for _ in range(5):
        take._on_block(block, 2.0)
    # 5 notes: too few for a meter conclusion — the default 4/4 stands
    assert take.grid.bpm == 150.0
    assert take.grid.meter == DEFAULT_METER
    assert take.grid.meter.n_slots == 4

    take._on_block(block, 2.0)  # the sixth note closes: the accents decide
    closed = [e for e in events if e.kind == "note_closed"]
    assert take.grid.meter.label == "3/4"
    assert take.grid.meter.n_slots == 3
    assert closed[-1].grid is take.grid


def test_livetake_publishes_compound_meter_on_dotted_pulse(monkeypatch):
    """Une prise sur temps pointés (6/8) est publiée avec sa mesure en direct.

    La chaîne complète : estimate_bpm_live corrige la médiane (un temps
    pointé avec alternance forte/faible vaut 1,5 noire — le repli ×2 ne
    répare jamais ce rapport), puis detect_meter saute l'étape accents (qui
    lirait les phases 0/2 de la grille ternaire comme une fausse valse) et
    conclut 6/8 sur la grille ternaire. Le métronome live lit exactement
    `grid.meter.n_slots`.
    """
    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", DottedStream)

    events: list[live_session.LiveEvent] = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    block = np.zeros(512, dtype=np.float32)
    for _ in range(8):  # eight dotted beats of strict 6/8 material
        take._on_block(block, 2.0)
    # corrected: median 1.5 s read as a DOTTED beat -> quarter tempo 60;
    # without the correction the fold would publish 80 (the flat case below)
    assert take.grid.bpm == 60.0
    assert take.grid.meter.label == "6/8"
    assert take.grid.meter.n_slots == 2
    closed = [e for e in events if e.kind == "note_closed"]
    assert closed[-1].grid is take.grid


def test_livetake_flat_dotted_pulse_stays_default(monkeypatch):
    """Limite assumée : sans alternance forte/faible, les écarts pointés
    restent indistinguables de noires lents — le repli d'octave ×2 s'applique
    (80 BPM ici, jamais les 100 du tempo composé) et la mesure reste 4/4.
    Le discriminant timing-seul n'existe pas : des écarts uniformes
    décrivent aussi bien « noires à 2/3 du tempo » que « temps pointés ».
    """
    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", FlatDottedStream)

    take = live_session.LiveTake(on_event=lambda e: None)
    take.start()
    block = np.zeros(512, dtype=np.float32)
    for _ in range(8):
        take._on_block(block, 2.0)
    assert take.grid.bpm == 80.0
    assert take.grid.meter.label == "4/4"  # the default stands, no false 3/4
