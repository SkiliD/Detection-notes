"""The GUI twin of the Session live: the take in progress, one deep module.

The 2026-09-27 user-reported regressions (a sidecar archived twice, a
frozen live roll, a replay starving the next capture) all landed in the
same vacuum: nobody owned "the take in progress" on the GUI side — the
window spread its state across five handlers and mutated it from each.
This module owns that state behind a SMALL interface; the window becomes
a thin adapter that applies what the controller decides.

Plain Python on purpose: the interface is the test surface, and these
tests must not pay an event loop. A `Verdict` is returned, never applied
here — the controller never touches a widget, a QTimer or a thread.

Decisions owned here, each with its measured reason:

- **The stem is minted once per take**: the first export names the
  folder, every later writer (re-export, sidecar, replay) reuses it —
  re-minting `live_take_name()` per call once scattered one take's
  artifacts over several timestamped folders.
- **The replay queue survives a new take**: a queued replay belongs to
  an ALREADY EXPORTED take, so `begin_take()` forgets the take's stem
  and signal but keeps the queue.
- **One replay at a time, never during a capture**: a replay's real-time
  DSP (the same LiveCapture -> YIN cost as a capture) starved the live
  drain thread — the live counter confirmed 2-12 notes while replaying
  the same WAVs confirmed 17-23 (measured on the chained 20:45-20:46
  takes). The guards read the CALLER's facts (its QTimer, its worker),
  never Qt itself.
- **The analysis arrives queued**: when a newer capture is running, its
  render timer must keep running (`keep_timer`) — stopping it froze the
  roll the singer was watching.
- **The sidecar has one writer and one stem**: the export-time block is
  archived once; the replay worker replaces it later with the re-derived
  audit. `handle_export` returns the stem to archive under, or None.

Also the accuracy-harness bridge: the live accuracy counters
(`update_notes`, `elapsed`) are fed here — via `sink()` — so the take
presents ONE object to `record_take` and the GUI hint, instead of the
window re-deriving them per tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core import audio_io
from ..core.live_session import MIN_CAPTURE_S
from ..core.takes import live_export_dir, live_take_name

# A trace with no audio behind it would tell nothing the empty folder did
# not: without a signal there is no take, only a Start press (the mic
# failed to open) — the failure is still announced (the thread's `failed`
# event), there is just no diagnosable folder to write one into.
MIN_TRACEABLE_SIGNAL_BYTES = 1


@dataclass(frozen=True)
class Verdict:
    """What the adapter must DO after one event; the controller decides.

    Plain data, no Qt: the window applies it (keep or stop its timer).
    Every branch of the take lifecycle is assertable in a unit test
    without an event loop. NO replay field: the cross-check can only
    start after the take's WAV exists — the export writes it in the SAME
    handler, after the analysis verdict is applied — so the single pump
    after the export is the only starter; deciding earlier would race a
    missing file (the worker would find no WAV and skip silently).
    """

    keep_timer: bool = False


class LiveSessionController:
    """Own the CURRENT take's resources; decide, never render.

    Interface (everything a caller must know):

    - `begin_take()` — the singer pressed Start: the NEW take owns every
      buffer (stem, signal); the replay queue SURVIVES, and the adapter
      must BREAK a running replay first (the new capture is the DSP
      priority; the worker's `stop()` breaks its wait loop without
      joining).
    - `export_stem()` — the take's folder name: minted at the FIRST
      export, reused afterwards.
    - `handle_export(block)` — the export finished: queues the take's
      cross-check replay (the WAV now exists), marks the take exported
      and returns the stem to archive the live_vs_final sidecar under,
      or None (no block). The pump AFTER this call is the ONLY replay
      starter.
    - `handle_analysis(capture_running, signal_bytes)` — the finished
      analysis arrived (possibly QUEUED, i.e. after the singer started
      the next take): stores the event's signal — ALWAYS, an empty bytes
      payload CLEARS the buffer (the stale-signal family) — and returns
      the `Verdict` (timer) for the adapter to apply.
    - `fail_take()` — the take died before its export (the thread's
      `failed` event, no analysis arrived): writes the diagnostic trace
      into the take's own folder and returns the path, or None. The
      window calls it best-effort from the failure path ONLY — never
      from a normal stop, where the queued analysis may still arrive.
    - `write_trace(reason, detail)` — the trace writer `fail_take` uses;
      callers with a reason of their own (a future failure path) name it
      here.
    - `pump_replay(capture_running, timer_running, worker_running)` —
      the next queued replay to start, or None when a guard says wait.
    - `replay_finished()` — the running replay ended; the next pump may
      start one.
    - `take_stem()` / `take_dir()` / `replay_queue()` — read-only views
      for the adapter and the tests.
    - `close()` — the window closes: the queue dies with the session
      (queued replays are cross-checks, not data — takes are exported).

    `capture_running` / `timer_running` / `worker_running` come from the
    CALLER (the adapter reads its own QTimer and QThread): the controller
    decides from the facts it is given — it never probes Qt.
    """

    def __init__(self) -> None:
        self._stem: str | None = None
        self._signal: bytes | None = None
        self._exported = False
        self._traced = False
        self._queue: list[str] = []
        self._replay_running = False

    # -- take lifecycle ---------------------------------------------------

    def begin_take(self) -> None:
        """A NEW take owns every buffer; the replay queue SURVIVES."""
        self._stem = None
        self._exported = False
        self._traced = False
        self._signal = None

    def set_signal(self, signal_bytes: bytes) -> None:
        """The take's audio, from the event that carries it (source of truth)."""
        self._signal = signal_bytes

    def signal(self) -> bytes | None:
        return self._signal

    # -- the take names its folder once ------------------------------------

    def export_stem(self) -> str:
        """The take's stem: minted at the FIRST export, reused afterwards."""
        if self._stem is None:
            self._stem = live_take_name()
        return self._stem

    def take_stem(self) -> str | None:
        """The stem this take has already exported under (None before)."""
        return self._stem if self._exported else None

    def take_dir(self):
        """The take's artifact folder, once exported (else None)."""
        stem = self.take_stem()
        return live_export_dir(stem) if stem else None

    def handle_export(self, block: dict | None) -> str | None:
        """The export finished: queue the cross-check, name the sidecar stem.

        The FIRST export is the take's completion: it names the folder
        (`export_stem()` mints it) and QUEUES the take's replay — the WAV
        now exists, the pump after the export may start it. A re-export
        (the Export button again) reuses the stem and does NOT re-queue:
        the cross-check already ran or is queued (dedup either way). The
        RETURN is only the stem to archive the live_vs_final sidecar
        under — None when there is no block (no live reading), never a
        guess.
        """
        first = not self._exported
        self._exported = True
        stem = self.export_stem()  # minted HERE if the adapter skipped it
        if first:
            self.enqueue_replay(stem)
        return stem if block else None

    # -- the finished analysis: ONE place that decides -----------------------

    def handle_analysis(
        self,
        capture_running: bool,
        signal_bytes: bytes | None = None,
    ) -> Verdict:
        """The queued final analysis arrived: store its signal, judge the timer.

        The live timer is the render of the CURRENT capture: keep it when
        one is running (the singer had already started the next take —
        stopping it froze that take's roll, counter and cursor, the
        "the mic stopped detecting me" feel of 2026-09-27), stop it
        otherwise. The replay is NOT decided here (see `Verdict`): the
        take's WAV does not exist until the export in the same handler.

        `signal_bytes` is the event's audio, taken ALWAYS (an EMPTY bytes
        payload clears the buffer: keeping the previous value here is the
        2026-09-26 stale-signal bug — a take's WAV describing the
        previous take). None (no event carried one) keeps what a earlier
        event stored.
        """
        if signal_bytes is not None:
            self._signal = signal_bytes
        return Verdict(keep_timer=bool(capture_running))

    # -- the failed take: a diagnosable trace ---------------------------------

    # -- the failed take: a diagnosable trace ---------------------------------

    @staticmethod
    def reason_for_failure(signal_bytes: bytes | None) -> str:
        """Why this take has no final analysis, read from the signal size.

        The window (and any future caller) passes the take's signal bytes;
        the classification is the CONTROLLER's — the ADR's decision puts
        the failure vocabulary with the take's single owner, so every
        trace names a failure the same way. The thresholds: `finalize`
        refuses a capture under half a second (`err ui too short`), and
        nothing captured at all is a dead or silent mic.
        """
        size = len(signal_bytes) if signal_bytes else 0
        samples = size // 4  # float32 mono: the live signal's wire format
        if samples == 0:
            return "no audio captured (mic silent or unavailable)"
        if samples < audio_io.SR * MIN_CAPTURE_S:
            seconds = samples / audio_io.SR
            # FLOOR at the display precision: one sample under the 0.5 s
            # minimum is 0.49995 s — rounding (even to 3 decimals) would
            # read "too short to analyze (0.500 s)", the text contradicting
            # its own complaint. Down is always honest: a too-short capture
            # displays strictly under the bound it violated.
            shown = math.floor(seconds * 1000) / 1000
            return f"capture too short to analyze ({shown:.3f} s)"
        return "no final analysis (capture stopped before finalize)"

    def fail_take(self) -> str | None:
        """The take died before its export: write its trace, return the path.

        The 2026-09-29 corpus hazard, THE case this exists for: a take
        that dies before its first export has NO stem yet (the stem is
        minted at the first export), so the trace must mint it HERE — the
        failure itself names the folder (a pre-export mint is harmless:
        the stem is a timestamp, `begin_take` forgets it, and an exported
        take never reaches this path). The failure is only diagnosable
        when there is a take to diagnose: without captured audio (a Start
        press against a dead mic) there is no folder, the failure stays
        on the status banner where it was announced. One trace per take:
        the first failure names the cause, later ones would contradict it.

        Best-effort, never raises (the take is already lost; the trace
        must not raise a second error): returns the path written, or None
        when there is nothing to write on.
        """
        if self._exported:
            return None  # its artifacts and sidecar explain it — a trace would be noise
        if not self._signal or len(self._signal) < MIN_TRACEABLE_SIGNAL_BYTES:
            return None
        reason = self.reason_for_failure(self._signal)
        return self.write_trace(reason)

    def write_trace(self, reason: str, detail: str = "") -> str | None:
        """Write `<stem>.trace.json` into the take's folder; return the path.

        The take's folder is named HERE (minting the stem if the failure
        came before any export — see `fail_take`, the only caller the
        failure path needs). Refusals: an `exported` take (its artifacts
        and `live_vs_final.json` sidecar are the explanation — a trace
        beside them would be noise) and an already-traced take (ONE trace
        per take: the first failure is the cause, a later one would
        contradict it — the invariant lives with the single writer).
        Best-effort, never raises: returns the path written, or None.
        """
        if self._exported or self._traced:
            return None
        if not self._signal or len(self._signal) < MIN_TRACEABLE_SIGNAL_BYTES:
            return None
        import json
        from datetime import datetime, timezone

        stem = self.export_stem()  # the failure names the folder too
        try:
            folder = live_export_dir(stem)
            folder.mkdir(parents=True, exist_ok=True)
            payload = {
                "take": stem,
                "reason": reason,
                "detail": detail,
                "signal_bytes": len(self._signal),
                "exported": self._exported,
                "written_at": datetime.now(timezone.utc).isoformat(),
            }
            path = folder / f"{stem}.trace.json"
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._traced = True
            return str(path)
        except Exception:
            return None

    # -- the replay queue ----------------------------------------------------

    def enqueue_replay(self, stem_name: str | None) -> None:
        """Queue one take's cross-check replay (no duplicates, real stems)."""
        if not stem_name or stem_name in self._queue:
            return
        self._queue.append(stem_name)

    def replay_queue(self) -> list[str]:
        return list(self._queue)

    def pump_replay(
        self,
        capture_running: bool = False,
        timer_running: bool = False,
        worker_running: bool = False,
    ) -> str | None:
        """The next replay to start, or None when a guard says wait.

        Guards, in order: a capture is running (its timer ticks) — the
        replay would starve the live DSP; a replay is already running —
        one at a time, the worker's `finished` re-pumps; nothing queued.
        The popped stem leaves the queue and marks the replay running.
        """
        if capture_running or timer_running or worker_running:
            return None
        if self._replay_running or not self._queue:
            return None
        self._replay_running = True
        return self._queue.pop(0)

    def replay_finished(self) -> None:
        """The running replay ended: the next pump may start one."""
        self._replay_running = False

    def replay_running(self) -> bool:
        return self._replay_running

    # -- shutdown --------------------------------------------------------------

    def close(self) -> None:
        """The window closes: the queue dies with the session (cross-checks,
        not data — every queued take is already exported)."""
        self._queue.clear()
        self._replay_running = False
