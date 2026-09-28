"""The controller WITHOUT Qt: the take lifecycle, one verdict per branch.

`LiveSessionController` is the deep module the 2026-09-27 regressions
pointed at; these tests walk EVERY branch of its interface — begin,
export (once, twice), analysis queued behind a newer capture, replay
guards (capture running / replay running / empty queue), the queue
surviving a new take, and close. No Qt: the interface IS the test
surface (the window applies the `Verdict`s; here the plain data is
asserted in a unit test without an event loop).

Each pin carries the measured regression it exists to prevent:

- **the stem is minted once per take** — re-minting `live_take_name()`
  per writer scattered one take's artifacts over timestamped folders;
- **the signal comes from the event, ALWAYS** — an `or None` guard once
  turned an empty payload into "keep the previous take's signal" (the
  2026-09-26 stale-WAV family);
- **the queue survives `begin_take`** — a queued replay belongs to an
  ALREADY EXPORTED take; losing it on the next start loses the audit;
- **never pump during a capture** — a replay's real-time DSP starved
  the live drain thread (live counter 2-12 notes vs 17-23 on replay of
  the same WAVs, chained takes 20:45-20:46);
- **one replay at a time** — two workers would race for the same DSP
  budget and for the same sidecar writer;
- **the verdict follows the CURRENT capture** — stopping the timer
  unconditionally froze the take the singer had already started.
"""

from __future__ import annotations

from v2m.gui.live_session_controller import LiveSessionController, Verdict


def _ctl() -> LiveSessionController:
    return LiveSessionController()


# -- the stem is minted ONCE per take, at the first export --------------------


def test_the_stem_is_minted_at_the_first_export_and_reused():
    ctl = _ctl()
    assert ctl.take_stem() is None, "rien n'est exporté: pas de stem"
    ctl.handle_export({"live": 1, "final": 1})
    stem = ctl.take_stem()
    assert stem, "le premier export a frappé le stem"
    # a RE-EXPORT (the Export button) reuses the SAME take folder
    ctl.handle_export({"live": 1, "final": 1})
    assert ctl.take_stem() == stem


def test_begin_take_forgets_the_buffers_but_keeps_the_queue():
    ctl = _ctl()
    ctl.handle_export({"live": 1, "final": 1})  # mints AND queues this take
    this_take = ctl.replay_queue()[0]
    ctl.set_signal(b"\x00\x01")
    ctl.enqueue_replay("live_older")  # an already-exported take's audit
    ctl.begin_take()
    assert ctl.take_stem() is None, "une NOUVELLE prise ouvre son dossier"
    assert ctl.signal() is None, "le signal de la prise précédente est mort"
    # the deferred replays belong to ALREADY-EXPORTED takes: they survive
    # (this take's own cross-check stays queued until a quiet moment too)
    assert ctl.replay_queue() == [this_take, "live_older"]


# -- the export: sidecar stem, ONE queue entry --------------------------------


def test_handle_export_queues_once_and_names_the_sidecar_stem():
    ctl = _ctl()
    block = {"live": 3, "final": 3, "contours": 0, "off1": 0, "moved": 0,
             "absorbed": 0, "rate": 0.0}
    stem = ctl.handle_export(block)
    assert stem == ctl.take_stem(), "le sidecar vit sous le stem de la prise"
    assert ctl.replay_queue() == [stem], "le WAV existe: le rejeu est enfilé"
    # a RE-EXPORT reuses the stem and does NOT re-queue (dedup either way)
    assert ctl.handle_export(block) == stem
    assert ctl.replay_queue() == [stem]
    # no block: no sidecar (None, never a guess) — but still queued once
    ctl2 = _ctl()
    assert ctl2.handle_export(None) is None
    assert ctl2.replay_queue() == [ctl2.take_stem()]


# -- the queued analysis: signal source of truth, timer verdict ---------------


def test_handle_analysis_takes_the_event_signal_always():
    ctl = _ctl()
    ctl.set_signal(b"previous")
    # an EMPTY bytes payload CLEARS the buffer: the `or None` shortcut once
    # kept the PREVIOUS take's signal here (the stale-WAV family)
    verdict = ctl.handle_analysis(capture_running=False, signal_bytes=b"")
    assert ctl.signal() == b""
    assert verdict == Verdict(keep_timer=False)
    # a real payload is stored as is; None (no event carried one) keeps it
    ctl.handle_analysis(capture_running=True, signal_bytes=b"this-take")
    assert ctl.signal() == b"this-take"
    ctl.handle_analysis(capture_running=True, signal_bytes=None)
    assert ctl.signal() == b"this-take"


def test_the_timer_verdict_follows_the_current_capture():
    ctl = _ctl()
    # the analysis arrives QUEUED behind the take the singer already
    # started: that take's render timer must KEEP RUNNING
    assert ctl.handle_analysis(capture_running=True).keep_timer is True
    # no capture running: the take that finished IS the live one — stop
    assert ctl.handle_analysis(capture_running=False).keep_timer is False


# -- the replay pump: the 2026-09-27 DSP-starvation guards --------------------


def test_pump_guards_capture_running_and_one_at_a_time():
    ctl = _ctl()
    ctl.handle_export({"live": 1, "final": 1})
    stem = ctl.replay_queue()[0]
    # a capture is running: the replay would starve its DSP — wait
    assert ctl.pump_replay(capture_running=True) is None
    assert ctl.replay_queue() == [stem], "rien n'est consommé sous garde"
    # one replay at a time: the RUNNING one blocks the next pump
    first = ctl.pump_replay()
    assert first == stem
    ctl.enqueue_replay("live_second")
    assert ctl.pump_replay() is None, "un seul rejeu à la fois"
    # it ended: the queue drains
    ctl.replay_finished()
    assert ctl.pump_replay() == "live_second"
    ctl.replay_finished()
    assert ctl.pump_replay() is None and ctl.replay_queue() == []


def test_pump_on_an_empty_queue_is_none_never_a_guess():
    assert _ctl().pump_replay() is None


def test_close_dies_with_the_session():
    ctl = _ctl()
    ctl.enqueue_replay("live_a")
    ctl.enqueue_replay("live_b")
    ctl.close()
    assert ctl.replay_queue() == []
    assert ctl.pump_replay() is None
