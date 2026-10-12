"""Pitch tests: a known sine -> the right detected frequency."""

import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import SR, sine_note

from v2m.core import pitch
from v2m.core.music import hz_to_midi
from v2m.core.pitch import YinBlock, track_pitch_offline, yin_pitch

sys.path.insert(0, str(Path(__file__).parent))
from voicetruth import SungNote, detect_notes, measure, synthesize  # noqa: E402


def test_yin_block_440hz():
    """Block YIN on a 440 Hz A -> ~440."""
    frame = sine_note(440.0, 1024 / SR, sr=SR)
    hz = yin_pitch(frame, SR)
    assert hz == hz  # not NaN
    assert abs(hz - 440.0) / 440.0 < 0.02  # < 2%


def test_yin_block_silence_nan():
    """Silence -> NaN (unvoiced)."""
    hz = yin_pitch(np.zeros(1024, dtype=np.float32), SR)
    assert np.isnan(hz)


def _stream(y: np.ndarray, block: int, **kw) -> list[tuple[float, float]]:
    """Push a signal block by block, like the live capture does."""
    yin = YinBlock(SR, **kw)
    out: list[tuple[float, float]] = []
    for i in range(0, len(y), block):
        blk = y[i : i + block]
        if len(blk) < block:
            blk = np.pad(blk, (0, block - len(blk)))
        out.extend(yin.push(blk))
    return out


def test_yin_block_times_are_absolute_and_monotonic():
    """Frame times must advance, even when the buffer is trimmed.

    After a trim the frames must keep their ABSOLUTE times: restarting
    from the buffer's start freezes the clock and the live segmenter
    confirms no note at all.
    """
    frames = _stream(sine_note(440.0, 2.0, sr=SR), 1024, window=2048, hop=1024)
    times = [t for t, _ in frames]
    assert len(times) > 40
    diffs = np.diff(times)
    assert diffs.min() > 0  # strictly increasing
    assert diffs == pytest.approx([1024 / SR] * len(diffs), rel=1e-6)
    # covers the full 2 s, not only the first blocks
    assert times[-1] > 1.8


def test_yin_block_stream_memory_is_bounded():
    """The sliding buffer must not grow with the capture's duration."""
    yin = YinBlock(SR, window=2048, hop=1024)
    y = sine_note(440.0, 3.0, sr=SR)
    for i in range(0, len(y), 1024):
        yin.push(y[i : i + 1024])
        assert len(yin.buffer) <= 2048 + 1024 + 1024


def test_yin_block_stream_tracks_pitch():
    """The detected pitch stays right block after block (live config)."""
    frames = _stream(sine_note(440.0, 1.0, sr=SR), 1024, window=2048, hop=1024)
    hz = np.array([h for _, h in frames], dtype=float)
    voiced = hz[~np.isnan(hz)]
    assert len(voiced) > 0
    assert abs(float(np.median(voiced)) - 440.0) < 8.0


def test_yin_block_confirms_a_live_note():
    """End to end: the live segmenter must see a note on a sine.

    The LIVE CHAIN's window is a DURATION (~23 ms, not 11.6 ms —
    ADR 0009), i.e. 512/256 samples at 22 050 Hz: that step is what
    matters, because at a 46 ms hop (2048/1024) three frames span 93 ms
    and the confirm_s window (0.09 s) can then never hold three — no
    note confirms at that coarse step. Production goes through
    LivePitchStream, which derives these numbers from the rate (512/256
    at 22 050 Hz, 1024/512 at 44 100 Hz).
    """
    from v2m.core.liveseg import LiveNoteSegmenter

    y = np.concatenate([sine_note(440.0, 0.5, sr=SR), np.zeros(SR // 2, dtype=np.float32)])
    seg = LiveNoteSegmenter()
    for t, hz in _stream(y, 1024, window=512, hop=256):
        seg.push(t, hz)
    seg.flush(1.0)
    assert [n.midi for n in seg.notes] == [69]


def test_a_low_rumble_never_becomes_a_note_offline():
    """The low mic rumble at the start of a live take (the "Sol#2" read on
    the 21:23 take, 1.08 s) must NOT become a note: a low-pass rumble has
    neither periodicity (d' 0.73-1.12) nor energy above 500 Hz (part
    0.003) — yet pYIN returns an Hz, and segmentation engraved it. The
    two gates (MIN_PITCH_QUALITY, HF_SHARE_MIN) kill all its frames; the
    real notes of the rest of the take survive.
    """
    from scipy.signal import butter, filtfilt

    truth = [SungNote(64, 1.4, 0.6), SungNote(67, 2.1, 0.6), SungNote(71, 2.8, 0.6)]
    y = synthesize(truth, glide_s=0.09, consonant_s=0.05)
    rng = np.random.default_rng(3)
    n = int(1.2 * SR)
    b, a = butter(4, 250 / (SR / 2), "low")
    noise = filtfilt(b, a, rng.standard_normal(n))
    y[:n] += (noise * np.linspace(0.15, 1.0, n) * 0.18).astype(np.float32)
    notes = detect_notes(y, SR)
    assert measure(notes, truth).exact, [(x.midi, round(x.start, 2)) for x in notes]
    assert all(n.midi > 50 for n in notes)  # no more false low note


def test_a_low_rumble_never_becomes_a_live_note():
    """The same rumble through the LIVE chain: the 512 window lets a few
    "periodic" frames through by chance (measured 24/87, never more than
    5 consecutive = 57 ms), but NO note can confirm — confirm_s (0.09 s)
    demands a continuous hold of voiced frames the noise never has.
    Before the gates, the whole rumble was voiced and the segmenter
    engraved a false low note (the take's "Sol#2").
    """
    from scipy.signal import butter, filtfilt

    from v2m.core.liveseg import LiveNoteSegmenter

    rng = np.random.default_rng(3)
    n = int(1.0 * SR)
    b, a = butter(4, 250 / (SR / 2), "low")
    noise = filtfilt(b, a, rng.standard_normal(n))
    y = (noise * np.linspace(0.15, 1.0, n) * 0.25).astype(np.float32)
    frames = _stream(y, 1024, window=512, hop=256)
    voiced = [hz for _, hz in frames if hz == hz and hz > 0]
    assert len(voiced) < len(frames) / 2, (len(voiced), len(frames))
    # end to end: the segmenter confirms NOTHING on this noise
    seg = LiveNoteSegmenter()
    for t, hz in frames:
        seg.push(t, hz)
    seg.flush(1.0)
    assert seg.notes == [], [(x.midi, round(x.start, 2)) for x in seg.notes]


def test_pyin_440hz_median():
    """pYIN over 1 s of 440 -> median ~440, MIDI note ~69."""
    y = sine_note(440.0, 1.0, sr=SR)
    track = track_pitch_offline(y, SR)
    voiced = track.hz[~np.isnan(track.hz)]
    assert len(voiced) > 20
    med = float(np.median(voiced))
    assert abs(hz_to_midi(med) - 69.0) < 0.5


def test_pyin_confidence_is_the_voicing_probability():
    """`confidence` carries pYIN's voicing PROBABILITY, not the binary flag.

    The old code threw the vprob (3rd return of librosa.pyin) away for a
    1.0 flag wherever f0 existed: `clean_pitch` then had NOTHING to mask
    — the MusicXML colors and MIDI velocities were constant, and the
    doubtful voicing of a breathy voice passed for a sure voice. Measured
    on the voice harness: a clear hold ~0.95, breathy (noise 0.10, the
    contract) >= 0.83, a breath frame on a hold 0.49-0.59.
    """
    import sys

    import voicetruth

    sys.path.insert(0, "tests")
    truth = [voicetruth.SungNote(64, 0.2, 0.5)]
    y = voicetruth.synthesize(truth, consonant_s=0.05, glide_s=0.03)
    track = track_pitch_offline(y, SR)
    note_frames = (track.times >= 0.25) & (track.times <= 0.6) & ~np.isnan(track.hz)
    assert note_frames.sum() >= 10
    # the old binary flag = exactly 1.0; the vprob, it, varies
    assert track.confidence[note_frames].max() < 0.999
    assert float(np.median(track.confidence[note_frames])) >= 0.8  # sure note
    # and the frames WITHOUT f0 stay at 0 (the mask's contract)
    assert float(track.confidence[np.isnan(track.hz)].max(initial=0.0)) == 0.0


def test_pyin_two_octaves():
    """Two successive notes (220 then 880 Hz) well separated."""
    from conftest import silence

    y = np.concatenate([sine_note(220.0, 0.6), silence(0.2), sine_note(880.0, 0.6)])
    track = track_pitch_offline(y, SR)
    voiced = ~np.isnan(track.hz)
    assert voiced.sum() > 30
    first_half = np.median(track.hz[voiced][: len(voiced) // 2])
    last_half = np.median(track.hz[voiced][-len(voiced) // 2 :])
    assert abs(hz_to_midi(first_half) - 57.0) < 0.7   # A3 = 220 Hz
    assert abs(hz_to_midi(last_half) - 81.0) < 0.7    # A5 = 880 Hz


# ---------------------------------------------------------------------
# the swallow rescue: the block-wise second reading (finalization)
# ---------------------------------------------------------------------
def _burst_signal(sr: int = SR, note_hz: float = 220.0, burst_s: float = 0.12,
                  at_s: float = 0.5, noise: float = 0.10) -> np.ndarray:
    """Silence with one short voiced burst: the swallowed-note shape.

    The burst carries noise (a voice's breathiness), not a pure sine: a
    pure sine is already voiced by pYIN (measured 11 frames on 120 ms) —
    the swallow needs the noise-voiced shape.
    """
    y = np.zeros(sr * 2, dtype=np.float32)
    n = int(burst_s * sr)
    i0 = int(at_s * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    y[i0:i0 + n] = (
        0.3 * np.sin(2 * np.pi * note_hz * t)
        + noise * rng.standard_normal(n)
    ).astype(np.float32)
    return y


def test_rescue_grafts_a_swallowed_voiced_burst_into_the_curve():
    """A 120 ms voiced burst pYIN leaves NaN becomes a voiced span.

    The long 2048 window reads no period there; the short-window sweep
    (the live geometry) does, and the curve inherits it — voiced frames
    AT the burst's pitch, the swallow the finalization used to miss.
    """
    y = _burst_signal()
    import librosa
    f0, _vf, vprob = librosa.pyin(
        y, fmin=pitch.FMIN, fmax=pitch.FMAX, sr=SR, frame_length=2048,
        hop_length=441, fill_na=np.nan,
    )
    rescued_s = pitch._rescue_swallowed_spans(y, SR, 441, f0, vprob)
    assert rescued_s > 0.0
    voiced = ~np.isnan(f0)
    assert voiced.sum() >= 4
    hz = f0[voiced]
    assert np.all(np.abs(hz - 220.0) / 220.0 < 0.05)


def test_rescue_refuses_a_true_silence_and_a_breath_noise():
    """Below the absolute gate, and noise without sustained period: no graft.

    A true silence stays a silence (the level gate is absolute, ADR 0003's
    no-relative-floor rule); breath-shaped noise (no periodicity at the
    short window) is never voiced — voice presence is answered by a curve.
    """
    y = np.zeros(SR * 2, dtype=np.float32)
    # a quiet NON-periodic burst (noise) at gate-clearing level
    i0 = int(0.5 * SR)
    rng = np.random.default_rng(7)
    y[i0:i0 + int(0.12 * SR)] = (0.15 * rng.standard_normal(int(0.12 * SR))).astype(np.float32)
    import librosa
    f0, _vf, vprob = librosa.pyin(
        y, fmin=pitch.FMIN, fmax=pitch.FMAX, sr=SR, frame_length=2048,
        hop_length=441, fill_na=np.nan,
    )
    rescued_s = pitch._rescue_swallowed_spans(y, SR, 441, f0, vprob)
    assert rescued_s == 0.0
    assert np.isnan(f0).all()


def test_rescue_refuses_a_legato_ramp_between_two_pitches():
    """A ramp relaying the neighbouring plateaus is a glide, not a note.

    The measured intruder shape: grafting a ramp's median sprouted a fake
    note between two sung ones. The spread bound refuses a wide ramp; the
    bridge guard refuses a narrow one that starts on one neighbour's pitch
    and ends on the other's.
    """
    sr = SR
    y = np.zeros(sr * 2, dtype=np.float32)
    seg0 = sine_note(220.0, 0.8, sr=sr)  # A3
    seg1 = sine_note(293.66, 0.8, sr=sr)  # D4
    y[: len(seg0)] = seg0
    y[len(y) - len(seg1):] = seg1
    # a 120 ms ramp sliding 220 -> 293.66 between them
    n_ramp = int(0.12 * sr)
    i0 = len(seg0)
    t = np.arange(n_ramp) / sr
    f_start, f_end = 220.0, 293.66
    phase = 2 * np.pi * np.cumsum(f_start + (f_end - f_start) * (t / t[-1]) / sr)
    y[i0:i0 + n_ramp] = (0.3 * np.sin(phase)).astype(np.float32)
    import librosa
    f0, _vf, vprob = librosa.pyin(
        y, fmin=pitch.FMIN, fmax=pitch.FMAX, sr=sr, frame_length=2048,
        hop_length=441, fill_na=np.nan,
    )
    rescued_s = pitch._rescue_swallowed_spans(y, sr, 441, f0, vprob)
    assert rescued_s == 0.0


def test_rescue_reads_only_fully_absent_spans():
    """A voiced-but-doubtful frame keeps pYIN's reading: never overwritten.

    Overwriting a low-confidence-but-present reading re-dates boundaries
    (measured: a real take's BPM moved 5 %) — the swallow shape is a span
    the long window read NOWHERE (NaN throughout).
    """
    # the real take's own evidence: a rescue call must not touch any frame
    # pYIN voiced, and on a fully-voiced span list there is nothing to read
    import librosa
    y = sine_note(220.0, 1.0, sr=SR)
    f0, _vf, vprob = librosa.pyin(
        y, fmin=pitch.FMIN, fmax=pitch.FMAX, sr=SR, frame_length=2048,
        hop_length=441, fill_na=np.nan,
    )
    snapshot = f0.copy()
    rescued_s = pitch._rescue_swallowed_spans(y, SR, 441, f0, vprob)
    assert rescued_s == 0.0
    assert np.array_equal(f0, snapshot)


def test_yin_pitch_closed_form_matches_the_reference_loop():
    """The vectorized d'(tau) IS the literal YIN loop, within float64 eps.

    The closed form (cumsum + one FFT autocorrelation) replaced the
    per-tau `np.mean` loop that cost 57 % of the live DSP (profiled on a
    real take: 655k mean dispatches for 981 frames). The thresholds
    (0.15 / MIN_PITCH_QUALITY 0.35 / the 0.6 fallback) are calibrated on
    the loop's MEAN definition, so this pin refuses any drift of the
    table they read — the calibration survives the optimization.
    """
    from v2m.core.pitch import FMIN

    def reference_dprime(x: np.ndarray) -> np.ndarray:
        w = len(x)
        tau_max = min(w - 1, int(SR / FMIN))
        d = np.zeros(tau_max + 1)
        for tau in range(1, tau_max + 1):
            diff = x[: w - tau] - x[tau:]
            d[tau] = np.mean(diff * diff)
        cum = np.cumsum(d[1:]) / np.arange(1, tau_max + 1)
        return d[1:] / np.where(cum == 0, 1e-12, cum)

    def closed_dprime(x: np.ndarray) -> np.ndarray:
        w = len(x)
        tau_max = min(w - 1, int(SR / FMIN))
        csum = np.concatenate(([0.0], np.cumsum(x * x)))
        total = csum[w]
        n_fft = 1
        while n_fft < 2 * w:
            n_fft <<= 1
        spec = np.fft.rfft(x, n_fft)
        r = np.fft.irfft(spec * np.conj(spec), n_fft)[:w]
        taus = np.arange(1, tau_max + 1)
        d = np.empty(tau_max + 1)
        d[0] = 0.0
        d[1:] = (csum[w - taus] + (total - csum[taus]) - 2.0 * r[taus]) / (w - taus)
        cum = np.cumsum(d[1:]) / taus
        return d[1:] / np.where(cum == 0, 1e-12, cum)

    rng = np.random.default_rng(11)
    for hz in (82.4, 220.0, 440.0, 1046.5):
        t = np.arange(1024) / SR
        x = np.sin(2 * np.pi * hz * t) + 0.05 * rng.standard_normal(1024)
        ref, closed = reference_dprime(x), closed_dprime(x)
        assert np.abs(closed - ref).max() < 1e-9
        # the function reads THAT table and stays on pitch
        quality: dict = {}
        hz_read = yin_pitch(x, SR, quality_out=quality)
        assert hz_read == hz_read
        assert abs(hz_read - hz) / hz < 0.01


def test_batched_pyin_matches_librosa_candidate_and_viterbi_tracks():
    """The compiled candidate batches preserve pYIN's full pitch reading."""
    import librosa

    from v2m.core.pitch import FMAX, FMIN, _pyin_batched

    # Include voiced sections, a silence, and a changing pitch so this pins
    # threshold fallback, candidate-bin collisions, and Viterbi transitions.
    parts = []
    for hz, duration in ((220.0, 0.28), (None, 0.09), (261.6256, 0.31), (329.6276, 0.25)):
        count = round(SR * duration)
        times = np.arange(count) / SR
        tone = np.zeros(count) if hz is None else 0.3 * np.sin(2 * np.pi * hz * times)
        parts.append(tone)
    signal = np.concatenate(parts).astype(np.float32)
    hop = 441
    reference = librosa.pyin(
        signal,
        fmin=FMIN,
        fmax=FMAX,
        sr=SR,
        frame_length=2048,
        hop_length=hop,
        fill_na=np.nan,
    )

    for batch_size in (1, 17, 384):
        actual = _pyin_batched(signal, SR, hop, batch_size=batch_size)
        assert np.array_equal(actual[0], reference[0], equal_nan=True)
        assert np.array_equal(actual[1], reference[1])
        assert np.allclose(actual[2], reference[2], rtol=0.0, atol=1e-12)


def test_batched_pyin_rejects_nonpositive_batch_size():
    """A zero batch must fail early instead of silently returning an empty track."""
    from v2m.core.pitch import _pyin_batched

    with pytest.raises(ValueError, match="batch_size must be positive"):
        _pyin_batched(np.ones(SR, dtype=np.float32), SR, 441, batch_size=0)
