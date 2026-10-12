"""The finished take, replayed through the REAL chain — in the background.

Every live take is read twice (the streaming segmenter, then the offline
pass); the ReplayWorker reads it a THIRD time the way `scripts/replay_takes`
does: its own exported WAV is re-sung to a fake `InputStream` at real-time
speed through the true `LiveCapture` -> `LiveTake` chain, the two readings
are compared with the ONE policy (`core.compare`), and the gap is archived
as `<stem>.live_vs_final.json` beside the take's artifacts — the per-take
archive `scripts/replay_takes.py`, the coherence page and the multi-week
trend read back without re-running anything.

Two hard rules, both learned the hard way elsewhere in this codebase:

- **Never block the UI**: the replay costs the take's DURATION (real-time
  delivery is the only honest clock — AGENTS.md). It lives on its own
  QThread, emits `done`/`failed` through queued connections (thread-safe
  by construction), and `stop()` breaks the wait loop without touching
  any stream: `closeEvent` calls it, so closing the window never joins a
  hung capture on the GUI thread.
- **Never patch a global**: the script swapped `sd.InputStream` on the
  module — safe in a CLI process, a race in a GUI whose real microphone
  capture reads the same global. The fake stream is INJECTED instead
  (`core.replay.replay_take` takes a `stream_factory`), and the audio
  OUTPUT device is optional (`output_factory=None` skips it): headless
  machines and CI replay without one.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from ..core.replay import engine_fingerprint, replay_take


class ReplayWorker(QThread):
    """One take's cross-check: replay its WAV, archive the gap, emit once.

    `done` carries the audit row (the same dict `replay_take` returns);
    `failed` carries the message — the take is already exported either way,
    so both signals are informational. `stop()` sets the worker's stop
    flag: a running replay finishes its current block, skips the rest of
    the wait loop and the archive, and returns promptly.
    """

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, stem: Path, stream_factory=None, parent=None) -> None:
        super().__init__(parent)
        self.stem = Path(stem)
        self.stream_factory = stream_factory
        self._stop_flag = False

    def stop(self) -> None:
        """Break the replay's wait loop; the thread then exits on its own."""
        self._stop_flag = True

    def run(self) -> None:
        # the replay only has to finish EVENTUALLY: below the capture's
        # priority, so a queued cross-check cannot starve a live drain
        # (see gui/priority.py)
        from .priority import lower_background_priority

        lower_background_priority()
        wav = Path(str(self.stem) + ".wav")
        if not wav.exists():
            return  # the box was off for this take: nothing to replay
        try:
            row = replay_take(
                wav,
                stream_factory=self.stream_factory,
                stop=lambda: self._stop_flag,
            )
        except Exception as e:  # the take is exported; a replay failure is noise
            self.failed.emit(str(e))
            return
        if self._stop_flag:
            return
        # the audit REPLACES the export-time sidecar (same stem, one writer:
        # this thread), in the BLOCK shape (`live`/`final`, the vocabulary
        # `core.compare.classify` writes and every reader — the coherence
        # aggregator, `core.explain` — expects), not the script table's row
        # (`live_notes`/`final_notes`)
        from ..core.compare import write_sidecar

        # (the sidecar PATH itself is the artifact contract's: takes.SIDECAR_SUFFIX
        # through write_sidecar — this thread never spells a suffix)

        block = {
            "live": row["live_notes"],
            "final": row["final_notes"],
            "contours": row["contours"],
            "off1": row["off1"],
            "moved": row["moved"],
            "absorbed": row["absorbed"],
            "rate": row["rate"],
            # the replay's own honesty counter: a hollow live reading stays
            # legible in the archive, not only in the moment's status line
            "dropped": row["dropped"],
            "drain_min": row["drain_min"],
            # the engine that measured THIS audit: `scripts/replay_takes.py
            # --changed` re-derives only the takes an OLDER engine read
            "engine": engine_fingerprint(),
        }
        write_sidecar(self.stem, block)
        self.done.emit(row)
