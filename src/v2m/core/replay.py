"""Replay ONE take through the REAL live chain: the engine behind every consumer.

The daily follow-up measured by hand (a fake mic feeding the true
`LiveCapture` at real-time speed, the final analysis, the live/final
comparison) has ONE owner here, and `scripts/replay_takes.py` and the
GUI's `ReplayWorker` are both thin adapters over it.

What a replay is, and the two measured rules it inherits:

- **Real-time delivery or nothing**: the fake `InputStream` delivers the
  take's WAV in 1024-sample blocks at 1x real time. Delivering faster is
  MEASURED to lie (7 notes instead of 24 on take 140425): the segmenter
  confirms on the blocks' time, not on their content.
- **The chain, not a shortcut**: the WAV feeds `LiveCapture` exactly like a
  real microphone (44 100 Hz, resampled to the analysis rate inside), the
  final reading is `LiveTake.finalize`, and the gap between the two
  readings is `core.compare.breakdown` — the ONE matching policy. A replay
  that re-implemented any of it would diagnose an engine that does not
  exist.

The stream is INJECTED (`stream_factory`, same contract as
`sd.InputStream`), never patched globally: a GUI replay must not swap the
module's `sd` while the user's real microphone reads it. The audio output
is optional (`output_factory=None` skips it) so a headless machine replays
without a speaker.

A replay NEVER writes artifacts: it reads the take's WAV, it does not
touch anything beside it. The archive of the gap is the CALLER's decision
(`ReplayWorker` writes the sidecar; the script only prints).

The final reading reuses the take's CACHED evidence
(`<stem>.observation.npz`) when it matches the WAV (`matches_take`: rate,
duration, size+mtime): the pYIN pass — ~0.3x the take's real time, the
longest stage — is skipped and the interpretation reads the same evidence
`analyze_signal` would have built, so the reading is identical by
construction (the cache only ever saves time, it cannot change a
decision). A stale, corrupt or absent cache falls back to
`LiveTake.finalize`; a replay that cannot read its own evidence must never
fail because of it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import audio_io
from .compare import breakdown

# The mic's rate: a real capture opens its device there, so a replay does
# too (LiveCapture resamples to the analysis rate exactly as for a voice).
REPLAY_SR = 44_100


def engine_fingerprint() -> str:
    """A hash of the replay engine's OWN code: what `--changed` compares.

    The sources of the modules a replay actually runs through (replay,
    live_session, liveseg, compare, meter) are hashed together — a change
    in ANY of them means the previous audits were measured by another
    engine and must be re-derived. Data files are excluded on purpose:
    only CODE changes invalidate a verdict. Deterministic within a
    checkout (sorted names, read bytes); good enough to answer "did the
    engine change since this audit", not a security primitive.
    """
    import hashlib

    from . import __file__ as _pkg_root  # type: ignore[attr-defined]

    root = Path(_pkg_root).parent  # .../core/__init__.py -> v2m/
    names = [
        "core/replay.py", "core/live_session.py", "core/liveseg.py",
        "core/compare.py", "core/meter.py", "core/pipeline.py",
    ]
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        try:
            digest.update((root / name).read_bytes())
        except OSError:
            digest.update(b"<missing>")  # a moved module invalidates too
    return digest.hexdigest()[:16]


def _fake_input_stream(audio: np.ndarray, sr: int, stop: Optional[Callable[[], bool]] = None):
    """Build a replacement `InputStream` that replays `audio` at 1x.

    Same shape as the test harnesses' fakes: the PortAudio callback gets
    1024-sample blocks on the real clock, `stop()` closes the tap. A
    provided `stop` predicate (the worker's flag) breaks the delivery so a
    cancelled replay does not keep sleeping through the whole take.
    """

    class ReplayStream:
        def __init__(self, samplerate, channels, blocksize, callback, **kwargs):
            self.rate = samplerate
            self.block = blocksize
            self.callback = callback
            self._stop = threading.Event()
            self._thread: threading.Thread | None = None
            self._audio = audio
            self._cancel = stop

        def start(self) -> None:
            self._thread = threading.Thread(target=self._play, daemon=True)
            self._thread.start()

        def _play(self) -> None:
            delay = self.block / self.rate
            for i in range(0, len(self._audio), self.block):
                if self._stop.is_set() or (self._cancel and self._cancel()):
                    return
                blk = self._audio[i : i + self.block]
                if len(blk) < self.block:
                    blk = np.pad(blk, (0, self.block - len(blk)))
                self.callback(blk.reshape(-1, 1), self.block, None, None)
                time.sleep(delay)

        def stop(self) -> None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(2)

        def close(self) -> None:
            pass

    return ReplayStream


def _final_reading(take, wav_path: Path, y: np.ndarray):
    """The replay's offline reading: cached evidence when possible, else finalize.

    `LiveTake.finalize` runs its own `analyze_signal` (pYIN included); the
    cache path skips exactly that pass. Both consume the same evidence
    through `interpret_observation`, so the two readings agree by
    construction — whoever answers, the audit reads THE take. The tempo is
    re-detected (cheap, librosa beat_track — a fraction of pYIN) because it
    is the caller's argument to the seam, not part of the evidence.
    """
    from .pipeline import LIVE_CONF, AnalyzeOptions, cached_observation, detect_tempo, interpret_observation

    try:
        cached = cached_observation(
            str(wav_path), y, audio_io.SR, AnalyzeOptions(min_conf=LIVE_CONF)
        )
    except Exception:
        cached = None  # a corrupt cache is worth exactly a missing one
    if cached is not None:
        try:
            tempo = detect_tempo(y, audio_io.SR)
            return interpret_observation(
                cached,
                tempo=tempo,
                source="live",
                opts=AnalyzeOptions(min_conf=LIVE_CONF),
                report=lambda f, m: None,
            )
        except Exception:
            pass  # fall through to finalize — same rule as a missing cache
    result = take.finalize()[0]
    # seed the take's evidence cache: THIS replay's pYIN pass becomes the
    # NEXT analysis's fast path (the same auto-save `analyze_file`
    # performs). Best-effort twice over — a read-only folder or an
    # observation the result does not carry changes nothing today.
    try:
        observation = getattr(result, "observation", None)
        if observation is not None:
            stat = wav_path.stat()
            observation.source_size = stat.st_size
            observation.source_mtime = stat.st_mtime
            from .takes import evidence_path

            observation.save(evidence_path(wav_path))
    except Exception:
        pass
    return result


def replay_take(
    wav_path: Path,
    stream_factory: Optional[Callable] = None,
    output_factory: Optional[Callable] = None,
    stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Replay one take's WAV through the GUI chain; return its audit row.

    `stream_factory` opens the input (default: the built-in real-time
    replayer); `output_factory` opens an optional monitoring output, skipped
    when None; `stop` is an optional predicate polled between blocks so a
    cancelled replay returns early. The row carries the same numbers as the
    script's table (live/final counts, contours, rate, tempo) plus the
    error key when the take could not be replayed.
    """
    from .live_session import LiveTake

    y = audio_io.load_file(str(wav_path), sr=REPLAY_SR)
    factory = stream_factory or _fake_input_stream(y, REPLAY_SR, stop)
    take = LiveTake(stream_factory=factory)
    take.start()
    if output_factory is not None:
        # the monitoring output is best-effort: no speaker must cost the audit
        try:
            output_factory(samplerate=REPLAY_SR)
        except Exception:
            pass
    # LiveCapture reads the injected stream at the real clock; wait until it
    # consumed the file (elapsed counts device samples). Delivering faster
    # is measured to lie: the wait IS the replay's real-time cost.
    audio_s = len(y) / REPLAY_SR
    deadline = time.monotonic() + max(30.0, audio_s * 2.5)
    while (
        take.state == "recording"
        and take.elapsed < audio_s * 0.98
        and not (stop and stop())
    ):
        if time.monotonic() > deadline:
            break
        time.sleep(0.05)
    seg_notes = list(take.segmenter.notes) if take.segmenter else []
    take.stop()
    result = _final_reading(take, Path(wav_path), y)
    final_notes = result.notes if result is not None else []
    block = breakdown(seg_notes, final_notes, result.grid if result else None)
    return {
        "stem": Path(wav_path).stem,
        "duration_s": round(audio_s, 2),
        "live_notes": len(seg_notes),
        "final_notes": len(final_notes),
        "contours": block["contours"],
        "off1": block["off1"],
        "moved": block["moved"],
        "absorbed": block["absorbed"],
        "rate": block["rate"],
        "tempo_bpm": result.tempo.bpm if result and result.tempo else None,
    }
