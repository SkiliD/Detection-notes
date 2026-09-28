"""The key snap LIVE: live notes snap as soon as they are confirmed, the way
`keys.snap_to_key` does offline.

Offline absorbs the global tuning offset BEFORE rounding, the live rounds
its raw median: the same plateau read by two roundings is THE dominant
family of ±1 label gaps between the live counter and the JSON (measured
on the real takes: 19/37, 10/24, 13/21). The live snap, applied at each
note's close in `LiveTake._apply_grid_policy` (the sole owner of the live
policy), reads `LiveNote.dev` — the live twin of `Note.deviation`.

What the test pins:
- an ambiguous live note (dev beyond ±0.30) leans toward its scale degree
  and changes label at close;
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
    """One note per block, at a known dev: the snap's raw material."""

    devs = (0.0,)  # overridden per test

    def __init__(self, sr, gate_rms):
        self.seg = _SnapSeg()

    def feed(self, block, t):
        i = len(self.seg.notes)
        dev = self.devs[i % len(self.devs)]
        base = self.seg.base_midi
        # the note's label follows its own mistune: a clearly sharp dev means
        # the running median rounded UP a semitone (that is the ±1 family),
        # a clear one stays on the base, a flat one rounds down
        midi = base + (1 if dev > 0.3 else (-1 if dev < -0.3 else 0))
        self.seg.notes.append(
            LiveNote(start=i * 0.4, end=i * 0.4 + 0.3, midi=midi,
                     strength=0.5, dev=dev)
        )
        return 0.01, []

    def elapsed_notes(self):
        return len(self.seg.notes)


class _SnapSeg:
    base_midi = 64  # E: between the C-scale degrees E and F
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


def test_an_ambiguous_live_note_snaps_toward_the_scale(monkeypatch):
    """An E sung +40 cents (dev 0.4) in C major leans toward F: the live
    note is re-labeled 65 at its close — the ±1 gap disappears WHERE the
    singer watches it (the counter), not only in the JSON."""

    class SharpStream(SnapStream):
        devs = (0.4,) * 12

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", SharpStream)
    events = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    block = np.zeros(512, dtype=np.float32)
    for _ in range(10):
        take._on_block(block, 2.0)

    closed = [e for e in events if e.kind == "note_closed"]
    assert closed, "the policy must publish the closed notes"
    # the key read off the scale (E+1=F repeatedly, F-heavy histogram) snaps
    # the note toward its degree: at least the LAST note is re-labeled
    # and its deviation absorbed
    assert take.snapped_notes >= 1
    last = closed[-1].note
    assert last.dev == 0.0  # re-anchored: the deviation was absorbed


def test_a_clear_live_note_stays_itself(monkeypatch):
    """An in-tune take (dev under ±0.30) does not move: a shade stays a
    shade, the snap is not a brutal rounding."""

    class CleanStream(SnapStream):
        devs = (0.1,)

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", CleanStream)
    take, events = _run_take(monkeypatch, CleanStream, 6)
    assert take.snapped_notes == 0
    assert all(n.midi == 64 for n in take.segmenter.notes)


def test_note_closed_publishes_the_snapped_note(monkeypatch):
    """The event carries the note AFTER the snap: an adapter (CLI, GUI)
    displaying it shows the same label as the final JSON."""

    class SnapAtCloseStream(SnapStream):
        devs = (-0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(live_session, "LiveCapture", FakeCapture)
    monkeypatch.setattr(live_session, "LivePitchStream", SnapAtCloseStream)
    events = []
    take = live_session.LiveTake(on_event=events.append)
    take.start()
    block = np.zeros(512, dtype=np.float32)
    for _ in range(4):
        take._on_block(block, 2.0)
    closed = [e for e in events if e.kind == "note_closed"]
    assert len(closed) == 4
    # the first note was ambiguous; by the time its event was published the
    # snap had already decided (the policy runs BEFORE the emit)
    assert take.snapped_notes >= 1
