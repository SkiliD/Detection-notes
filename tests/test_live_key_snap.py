"""The key snap LIVE: live notes snap as soon as they are confirmed, the way
`keys.snap_to_key` does offline.

Offline absorbs the global tuning offset BEFORE rounding, the live rounds
its raw median: the same plateau read by two roundings is THE dominant
family of ±1 label gaps between the live counter and the JSON (measured
on the real takes: 19/37, 10/24, 13/21). The live snap, applied at each
note's close in `LiveTake._apply_grid_policy` (the sole owner of the live
policy), reads `LiveNote.dev` — the live twin of `Note.deviation`.

What the test pins:
- an ambiguous live note (dev beyond ±0.30) whose pitch class is NOT in the
  key leans toward the neighbour it leans to and changes label at close,
  keeping its measured `dev` (the label is the key's decision, the mistune
  is the singer's — `LiveNote.snap_st` records the move);
- a clear note (dev under the threshold) stays itself;
- the `note_closed` event publishes the ALREADY-snapped note (adapters
  consume, they do not re-snap);
- a take with no discriminating pitch (all identical) snaps nothing: the
  key exists but no deviation is ambiguous.
"""

from __future__ import annotations

import numpy as np

from v2m.core import live_session
from v2m.core.liveseg import LiveNote


class FakeCapture:
    samplerate = 22_050

    def __init__(self, **kwargs):
        self.on_block = kwargs["on_block"]
        self.started = False
        self._signal = np.zeros(22_050, dtype=np.float32)

    def start(self):
        self.started = True

    def stop(self):
        pass

    def elapsed(self):
        return 2.0

    def signal(self):
        return self._signal


class SnapStream:
    """One note per block, as (label, dev): the snap's raw material.

    The LABEL is what the live segmenter would have rounded (a mistuned
    plateau can round up or down a semitone — the ±1 family), the dev is
    how far the plateau sat from that label.
    """

    notes_src: tuple[tuple[int, float], ...] = ((64, 0.0),)  # per test

    def __init__(self, sr, gate_rms):
        self.seg = _SnapSeg()

    def feed(self, block, t):
        i = len(self.seg.notes)
        midi, dev = self.notes_src[i % len(self.notes_src)]
        self.seg.notes.append(
            LiveNote(start=i * 0.4, end=i * 0.4 + 0.3, midi=midi, strength=0.5, dev=dev)
        )
        return 0.01, []

    def elapsed_notes(self):
        return len(self.seg.notes)


class _SnapSeg:
    notes: list

    def __init__(self):
        self.notes = []

    def current(self):
        return None


def _run_take(monkeypatch, stream_cls, n_blocks: int):
    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", stream_cls)
    events = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    block = np.zeros(512, dtype=np.float32)
    for _ in range(n_blocks):
        take._on_block(block, 2.0)
    return take, events


def _triad_then(source, n_blocks: int):
    """A C major triad (the key the take is in), then the note under test."""

    class _Stream(SnapStream):
        notes_src = source

    return _Stream, n_blocks


def test_an_ambiguous_live_note_snaps_toward_the_scale(monkeypatch):
    """An F# sung 35 cents SHARP in a C major take: outside the key, leaning
    up — the live note is re-labelled G at its close, so the ±1 gap
    disappears WHERE the singer watches it (the counter), not only in the
    JSON. Its measured `dev` survives the move (re-measured against the new
    label) and the relabel itself is recorded on the note."""
    stream_cls, blocks = _triad_then(((60, 0.0), (64, 0.0), (67, 0.0)) * 3 + ((66, 0.35),), 10)
    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", stream_cls)
    events = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    block = np.zeros(512, dtype=np.float32)
    for _ in range(blocks):
        take._on_block(block, 2.0)

    closed = [e for e in events if e.kind == "note_closed"]
    assert closed, "the policy must publish the closed notes"
    assert take.snapped_notes == 1  # only the chromatic note moved
    last = closed[-1].note
    assert last.midi == 67 and last.snap_st == 1.0
    assert last.dev == -0.5  # 35 cents sharp of F#, read against G: clipped


def test_closed_note_keeps_its_facts_when_the_detected_key_changes(monkeypatch):
    from types import SimpleNamespace

    from v2m.core.intonation_live import LiveIntonation
    from v2m.core.keys import KeyCandidate

    keys = iter(["C", "D", "C"])
    monkeypatch.setattr(
        live_session,
        "detect_key",
        lambda notes: SimpleNamespace(best=KeyCandidate(next(keys), "major", 1.0)),
    )
    take = live_session.LiveTake()
    summary = LiveIntonation()
    first = LiveNote(start=0.0, end=0.3, midi=66, dev=-0.4)
    notes = [first]
    take._apply_grid_policy(notes, 0.3)
    summary.add(first)
    published = (first.midi, first.dev, first.snap_st)
    expected_summary = summary.summary()
    for index in [1, 2]:
        notes.append(LiveNote(start=index * 0.4, end=index * 0.4 + 0.3, midi=60))
        take._apply_grid_policy(notes, index * 0.4 + 0.3)
        assert (first.midi, first.dev, first.snap_st) == published
        current = LiveIntonation()
        current.add(first)
        assert current.summary() == expected_summary
    assert take.snapped_notes == 1


def test_a_clear_live_note_stays_itself(monkeypatch):
    """An in-tune take (dev under ±0.30) does not move: a shade stays a
    shade, the snap is not a brutal rounding."""

    class CleanStream(SnapStream):
        notes_src = ((64, 0.1),)

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", CleanStream)
    take, events = _run_take(monkeypatch, CleanStream, 6)
    assert take.snapped_notes == 0
    assert take.segmenter is not None
    assert all(n.midi == 64 for n in take.segmenter.notes)


def test_note_closed_publishes_the_snapped_note(monkeypatch):
    """The event carries the note AFTER the snap: an adapter (CLI, GUI)
    displaying it shows the same label as the final JSON."""

    class SnapAtCloseStream(SnapStream):
        notes_src = ((60, 0.0), (64, 0.0), (67, 0.0)) * 3 + ((66, 0.35),)

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", SnapAtCloseStream)
    events = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    block = np.zeros(512, dtype=np.float32)
    for _ in range(10):
        take._on_block(block, 2.0)
    closed = [e for e in events if e.kind == "note_closed"]
    assert len(closed) == 10
    # the ambiguous note was decided BEFORE its event: the published note is
    # the snapped one, carrying both its label and the mistune it measured
    assert closed[-1].note.midi == 67
    assert closed[-1].note.snap_st == 1.0
    assert closed[-1].note.dev == -0.5
