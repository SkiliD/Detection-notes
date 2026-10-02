"""The live path's clock contract (ADR 0009).

The stream expresses its windows in TIME units: the same duration at any
sample rate, and the frame period it derives from them is published to
the segmenter, which reads its frame-written ceilings there.

A windows-in-SAMPLES stream fed at 44 100 Hz would run at half its
tuned duration (11.6 ms instead of the 23 ms tuned at 22 050): the
sequence would manufacture phantom low notes on top of the real ones —
the exact family the anti-rumble gate repairs. The contract is pinned
at both rates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v2m.core import audio_io
from v2m.core.liveseg import (
    HOP_MS,
    SPLIT_FRAME_S,
    WINDOW_MS,
    LiveNoteSegmenter,
    LivePitchStream,
)

# Real takes are gitignored: the parity test SKIPS where they are absent
# (the same rule as the reference-artifact fingerprints).
REAL_TAKES = ("chant.wav", "prise2.wav")


def test_the_window_is_a_duration_not_a_sample_count():
    """Both rates get the SAME duration, and the calibrated counts exactly."""
    a = LivePitchStream(22_050, gate_rms=audio_io.GATE_RMS)
    b = LivePitchStream(44_100, gate_rms=audio_io.GATE_RMS)
    assert a.window / a.sr == pytest.approx(b.window / b.sr, abs=1e-4)
    assert a.hop / a.sr == pytest.approx(b.hop / b.sr, abs=1e-4)
    # the settings the measurements were taken at: 512/256 at 22 050 Hz
    # (harness, sweep, audit) and 1024/512 at 44 100 Hz (the device default)
    assert (a.window, a.hop) == (512, 256)
    assert (b.window, b.hop) == (1024, 512)
    assert a.hop / a.sr == pytest.approx(HOP_MS / 1000.0, abs=1e-4)
    assert a.window / a.sr == pytest.approx(WINDOW_MS / 1000.0, abs=1e-4)


def test_the_stream_publishes_its_frame_period_to_the_segmenter():
    """The frame-based ceilings read the period of the stream, not a literal."""
    for sr in (22_050, 44_100):
        stream = LivePitchStream(sr, gate_rms=audio_io.GATE_RMS)
        assert stream.seg.frame_s == pytest.approx(stream.hop / stream.sr)
    # a segmenter built by HAND keeps the analysis-rate default: that is the
    # frame economics of every documented measurement
    assert LiveNoteSegmenter().frame_s == pytest.approx(SPLIT_FRAME_S)
    # At 44 100 Hz the DERIVED period coincides with that default (512/44100
    # == 256/22050 — the same duration, which is the whole point), so the
    # derivation is what a THIRD rate tells apart: at 48 000 Hz the period
    # must be the stream's own hop / sr, never the analysis-rate literal.
    other = LivePitchStream(48_000, gate_rms=audio_io.GATE_RMS)
    assert other.seg.frame_s == pytest.approx(other.hop / other.sr)
    assert other.seg.frame_s != pytest.approx(SPLIT_FRAME_S, rel=1e-9)


def _notes_at(path: str, sr: int) -> list:
    """Replay a take through the live chain at `sr` (device-style blocks)."""
    y = audio_io.load_file(path, sr=sr)
    stream = LivePitchStream(sr, gate_rms=audio_io.GATE_RMS)
    block = 1024  # LiveCapture.blocksize: the device hands SAMPLES, not ms
    for i in range(0, len(y), block):
        blk = y[i : i + block]
        if len(blk) < block:
            blk = np.pad(blk, (0, block - len(blk)))
        stream.feed(blk, (i + len(blk)) / sr)
    stream.seg.flush(len(y) / sr)
    return stream.seg.notes


@pytest.mark.slow
@pytest.mark.parametrize("path", REAL_TAKES)
def test_both_rates_give_the_same_notes_on_a_real_take(path: str):
    """A real take reads the same music at 22 050 and 44 100 Hz.

    Not the same samples: the same PITCHES, in the same number, with the
    boundaries inside one frame of each other (measured: 23 ms of worst-case
    difference on chant.wav, 35 ms on prise2.wav, 0 ms typical — the frame
    grid of the two chains, not a segmentation difference).
    """
    if not Path(path).exists():
        pytest.skip(f"{path} absent (real take, gitignored)")
    low = _notes_at(path, 22_050)
    high = _notes_at(path, 44_100)
    assert low and high
    assert [n.midi for n in low] == [n.midi for n in high]
    assert len(low) == len(high)
    assert max(abs(a.start - b.start) for a, b in zip(low, high, strict=True)) < 0.06
