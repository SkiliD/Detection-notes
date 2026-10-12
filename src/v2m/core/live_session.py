"""Shared live take lifecycle for the GUI and CLI adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import audio_io
from .audio_io import LiveCapture
from .figures import label_notes
from .grid import BeatGrid
from .i18n import tr
from .keys import detect_key, snap_to_key
from .liveseg import LiveNote, LiveNoteSegmenter, LivePitchStream
from .meter import detect_meter
from .pipeline import LIVE_CONF, AnalysisResult, AnalyzeOptions, analyze_signal
from .tempo import estimate_bpm_live

# The captured length under which `finalize` refuses to analyze (a shorter
# take has nothing readable: `err ui too short`). The failure-vocabulary
# consumer (the GUI trace) reads the same constant, never a second literal.
MIN_CAPTURE_S = 0.5


@dataclass(frozen=True)
class LiveEvent:
    """Typed observation emitted by a LiveTake.

    `kind` is one of: "block" (capture telemetry), "pitch" (one YIN
    frame), "note" (current note state), "note_closed" (a note just
    ended; `grid` carries the live take's BeatGrid), "stopped"
    (capture drained, signal final), "failed" (finalization error,
    emitted once), "finalized" (AnalysisResult ready).
    """

    kind: str
    time_s: float = 0.0
    rms: float = 0.0
    pitch: tuple[float, float] | None = None
    confirmed_pitch: tuple[float, float] | None = None
    heard_pitch: tuple[float, float] | None = None
    note: "LiveNote | None" = None
    grid: BeatGrid | None = None
    signal: np.ndarray | None = None
    result: AnalysisResult | None = None
    error: str | None = None


class LiveTake:
    """Own one microphone capture from start through final analysis.

    Qt and CLI are adapters: they subscribe to events and decide how to
    render or print them. The capture queue, streaming segmenter, stop/drain
    order, final signal and final AnalysisResult have one owner here — and
    so has the LIVE GRID POLICY (the approximate BPM from the notes so far,
    the meter detected from their accents, the figures labeled on its
    BeatGrid): `note_closed` publishes it, and adapters consume ready-labeled
    notes and a ready grid, never re-derive.
    """

    def __init__(
        self,
        min_conf: float = LIVE_CONF,
        sr: int = audio_io.SR,
        on_event: Callable[[LiveEvent], None] | None = None,
        stream_factory: "audio_io.StreamFactory | None" = None,
    ) -> None:
        self.min_conf = min_conf
        self.sr = sr
        self.on_event = on_event
        # None = the real microphone; a replay injects its own stream opener
        # (the injection, not a global patch, is what keeps a GUI replay safe)
        self.stream_factory = stream_factory
        self.capture: LiveCapture | None = None
        self.stream: LivePitchStream | None = None
        self._signal: np.ndarray | None = None
        self._result: AnalysisResult | None = None
        self._grid = BeatGrid(None)  # live grid: approximate BPM, no meter
        self._state = "idle"
        self._error_emitted = False
        self._failure: Exception | None = None
        self._snapped = 0  # live notes the key-aware snap retuned (see _apply_grid_policy)

    @property
    def state(self) -> str:
        return self._state

    @property
    def segmenter(self) -> "LiveNoteSegmenter | None":
        return self.stream.seg if self.stream is not None else None

    @property
    def elapsed(self) -> float:
        return self.capture.elapsed() if self.capture is not None else 0.0

    @property
    def signal(self) -> np.ndarray:
        if self._signal is not None:
            return self._signal
        return self.capture.signal() if self.capture is not None else np.zeros(0, dtype=np.float32)

    @property
    def result(self) -> AnalysisResult | None:
        return self._result

    @property
    def grid(self) -> BeatGrid:
        """The take's live BeatGrid: approximate BPM, meter when detected.

        `meter` falls back on the default 4/4 (the grid's rule) until the
        accents of at least `meter.MIN_NOTES` notes say otherwise; adapters
        that need the displayed beat count read `grid.meter.n_slots`.
        """
        return self._grid

    @property
    def error_emitted(self) -> bool:
        return self._error_emitted

    def _emit(self, event: LiveEvent) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def start(self) -> None:
        """Open the microphone; a stopped take cannot be revived."""
        if self._state == "recording":
            return
        if self._state == "stopped":
            raise RuntimeError(tr("err take stopped"))
        self._snapped = 0  # a NEW take starts with its own snap history
        self.capture = LiveCapture(
            on_block=self._on_block, sr=self.sr, stream_factory=self.stream_factory
        )
        self.capture.start()
        self._state = "recording"

    def _on_block(self, block: np.ndarray, t: float) -> None:
        """One captured block: pitch-track it and publish what it revealed.

        Figures are labeled at note END (approximate BPM from the onsets so
        far) so the closed note lands on the roll already labeled; the meter
        is detected the same moment, from the same notes' attack strengths.
        """
        if self.capture is None:
            return
        if self.stream is None:
            # gate_rms getattr: the capture is the injected seam (fakes in
            # tests may predate the attribute — the class default is the
            # one policy, audio_io.GATE_RMS)
            self.stream = LivePitchStream(
                self.capture.samplerate,
                gate_rms=getattr(self.capture, "gate_rms", audio_io.GATE_RMS),
            )
        before = self.stream.elapsed_notes()
        rms, frames = self.stream.feed(block, t)
        self._emit(LiveEvent("block", time_s=t, rms=rms))
        for frame_t, hz in frames:
            self._emit(LiveEvent("pitch", time_s=frame_t, pitch=(frame_t, hz)))
        segmenter = self.stream.seg
        confirmed = getattr(segmenter, "confirmed", None)
        heard = getattr(segmenter, "heard", None)
        self._emit(
            LiveEvent(
                "note",
                time_s=t,
                pitch=segmenter.current(),
                confirmed_pitch=confirmed() if confirmed is not None else None,
                heard_pitch=heard() if heard is not None else None,
            )
        )
        if self.stream.elapsed_notes() > before:
            self._apply_grid_policy(notes=self.stream.seg.notes, now_s=t)

    def _apply_grid_policy(self, notes: list[LiveNote], now_s: float) -> None:
        """Write the take's live grid once, then publish it on `note_closed`.

        THE grid policy, in one place: approximate BPM from the notes so
        far (the dotted-pulse reading, if any, is already corrected inside
        the estimate — the meter must be tested at the tempo the grid
        uses), the meter from their accents, figures labeled on the grid —
        all BEFORE the event, so every adapter consumes a ready note and a
        ready meter.
        """
        bpm = estimate_bpm_live(notes)
        if bpm:
            # with_meter accepts the detector's None (no conclusion yet):
            # the grid falls back on the default 4/4 by itself
            grid = BeatGrid(bpm).with_meter(detect_meter(notes, bpm))
            label_notes(notes, grid)
            self._grid = grid
        # the key-aware snap, applied to the CONFIRMED notes as they close:
        # the live twin of pipeline's snap_to_key. The offline pass absorbs
        # the global tuning offset BEFORE rounding, the live segmenter rounds
        # its running median raw — the same plateau read by two roundings is
        # the dominant family of live/final diffs (the ±1 labels measured on
        # the real takes). Snapping live toward the key's degree, as soon as
        # the evidence (the take's own notes so far) can read one, closes
        # that gap where the singer watches it — the counter and the final
        # JSON then tell the same story. LiveNote carries its residual
        # deviation as `dev`, the snap's dev_attr twin of Note.deviation.
        if notes:
            key = detect_key(notes)
            self._snapped += snap_to_key(notes, key.best, dev_attr="dev")
        self._emit(
            LiveEvent("note_closed", time_s=now_s, note=notes[-1], grid=self._grid)
        )

    @property
    def snapped_notes(self) -> int:
        """How many live notes the key-aware pull has retuned so far."""
        return self._snapped

    @property
    def dropped_blocks(self) -> int:
        """Blocks the drain abandoned at stop (the capture's honesty counter).

        A block QUEUED by the driver is not a block HANDLED: a stop past
        the drain's join timeout abandons the backlog, and those blocks
        never reached the DSP — the live note count reads hollow while
        the WAV keeps everything (the final analysis reads the take
        whole). The announcement at stop turns the number visible.
        """
        return self.capture.dropped if self.capture is not None else 0

    @property
    def drain_min(self) -> float:
        """The capture's worst mid-take drain progress (1.0 = never behind)."""
        return self.capture.drain_min if self.capture is not None else 1.0

    def stop(self) -> np.ndarray:
        """Stop once, drain captured blocks, flush segmentation, keep signal."""
        if self._state == "stopped":
            return self.signal
        if self.capture is None:
            self._state = "stopped"
            self._signal = np.zeros(0, dtype=np.float32)
            self._emit(LiveEvent("stopped", signal=self._signal))
            return self._signal
        self.capture.stop()
        if self.stream is not None:
            self.stream.seg.flush(self.capture.elapsed())
        self._signal = self.capture.signal()
        self._state = "stopped"
        self._emit(LiveEvent("stopped", time_s=self.elapsed, signal=self._signal))
        return self._signal

    def finalize(self) -> tuple[AnalysisResult, np.ndarray]:
        """Analyze the captured signal once; repeated calls return the same result.

        Idempotent in both directions: a successful analysis is reused, and
        a failure re-raises its error WITHOUT re-emitting the event (the
        adapter already showed it once). The result carries the take's OWN
        live/final decomposition (from the segmenter's confirmed notes,
        through the ONE matching policy): the JSON block, the results-panel
        explainer and the per-take archive read the same dict.
        """
        if self._result is not None:
            return self._result, self.signal
        if self._failure is not None:
            raise self._failure
        signal = self.stop()
        if len(signal) < self.sr * MIN_CAPTURE_S:
            self._fail(ValueError(tr("err ui too short")))
        try:
            result = analyze_signal(
                signal,
                self.sr,
                source="live",
                opts=AnalyzeOptions(min_conf=self.min_conf),
            )
        except Exception as exc:
            self._fail(exc)
        try:
            from .compare import breakdown, invalidate

            result.live_vs_final = breakdown(
                self.segmenter.notes if self.segmenter else [],
                result.notes,
                result.grid,
                # the honesty counters travel IN the block: the report JSON
                # and the archived sidecar read this one dict
                dropped=self.dropped_blocks,
                drain_min=self.drain_min,
            )
        except Exception:
            # the attach itself can fail (a result that cannot even carry the
            # attribute): the nicety must not cost the take in ANY case
            try:
                invalidate(result)
            except Exception:
                pass
        self._result = result
        self._emit(LiveEvent("finalized", signal=signal, result=result))
        return result, signal

    def _fail(self, error: Exception) -> None:
        """Report a finalization failure once, then remember it for retries.

        The event carries the take's signal (the same payload the
        "finalized" event would have): the failure diagnostic — the trace
        that says WHY the take died — needs the audio to name the folder
        and the duration, and only the take has it.
        """
        self._failure = error
        self._error_emitted = True
        self._emit(LiveEvent("failed", error=str(error), signal=self._signal))
        raise error

    def close(self) -> None:
        if self._state == "recording":
            self.stop()
        self.on_event = None
