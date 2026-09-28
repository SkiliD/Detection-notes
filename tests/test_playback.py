"""Tests for the local playback renderer (playback.render_notes)."""

import threading
import time
import types

import numpy as np
import pytest

from v2m.core import audio_io, playback
from v2m.core.notes import Note
from v2m.core.playback import PlaybackSession, NotePlayer, Timeline, render_notes, pitch_shift_signal


def test_render_produces_nonempty_signal():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9),
             Note(start=0.5, end=1.0, midi=64, confidence=0.9)]
    y = render_notes(notes, sr=22050)
    assert len(y) > 0
    assert np.max(np.abs(y)) > 0.1
    assert np.issubdtype(y.dtype, np.floating)


def test_render_duration_covers_last_note():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9),
             Note(start=1.5, end=2.0, midi=67, confidence=0.9)]
    y = render_notes(notes, sr=22050)
    assert len(y) >= int(2.0 * 22050)


def test_render_normalizes_peak():
    notes = [Note(start=0.0, end=1.0, midi=60, confidence=0.9)]
    y = render_notes(notes, sr=22050)
    peak = float(np.max(np.abs(y)))
    assert 0.5 < peak <= 1.0  # normalized, not clipped


def test_render_transpose_shifts_content():
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    y_ref = render_notes(notes, sr=22050, transpose=0)
    y_up = render_notes(notes, sr=22050, transpose=12)  # one octave up
    assert len(y_ref) == len(y_up)
    assert not np.allclose(y_ref, y_up)


def test_render_empty_notes():
    y = render_notes([], sr=22050)
    assert len(y) == 0


def test_render_gain_scales_the_piano():
    """The volume setting scales the render, peak stays bounded."""
    notes = [Note(start=0.0, end=0.5, midi=60, confidence=0.9)]
    ref = render_notes(notes, sr=22050, gain=1.0)
    quiet = render_notes(notes, sr=22050, gain=0.2)
    loud = render_notes(notes, sr=22050, gain=2.0)
    assert np.max(np.abs(quiet)) == pytest.approx(0.2 * float(np.max(np.abs(ref))), abs=1e-4)
    assert np.max(np.abs(loud)) <= 1.0  # never clips the float32 range
    assert len(quiet) == len(ref)


def test_note_player_is_a_rendered_source():
    """The render engine prepares buffers; playing is the session's job.

    The old class also played its own signal in a thread — a second way of
    starting a sound that made the GUI arbitrate two clocks. It is a plain
    rendered source now, handed to the PlaybackSession.
    """
    sr = 22050
    y = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    p = NotePlayer(signal=y, sr=sr)
    assert p.duration == pytest.approx(1.0, abs=0.01)
    assert np.array_equal(p.signal, y)
    q = NotePlayer()  # no notes, no signal: an empty source is legal
    assert len(q.signal) == 0 and q.duration == 0.0
    # the playback half is gone, by design — one player only
    assert not hasattr(p, "play") and not hasattr(p, "stop")
    assert not hasattr(p, "playing") and not hasattr(p, "bind_session")


def test_render_sample_exact_tone_length():
    """Regression: tone() re-derived its length through a seconds round-trip
    (int(dur * sr)) and lost one sample on some durations -> ValueError while
    mixing (30870 vs 30869 samples). `end` below makes i1 - i0 == 30870."""
    sr = audio_io.SR
    end = (30870 + 0.5) / sr  # int(end * sr) == 30870 exactly
    notes = [Note(start=0.0, end=end, midi=60, confidence=0.9)]
    y = render_notes(notes, sr=sr)  # used to raise ValueError
    assert len(y) > 0


def test_render_prime_and_edge_lengths():
    """Off-by-one-prone slice lengths must all render without a shape error."""
    for n in (1, 7, 30869, 30870, 30871):
        notes = [Note(start=0.0, end=(n + 0.5) / 22050, midi=60, confidence=0.9)]
        assert len(render_notes(notes, sr=22050)) >= n


def test_render_arbitrary_durations_never_crash():
    """A random sweep of note durations must never mis-shape the mix.

    ~9% of float durations lost a sample in the old round-trip, so a
    handful of notes was enough to hit the crash; 200 sweeps pin it for
    good at the project sample rate.
    """
    sr = audio_io.SR
    rng = np.random.default_rng(7)
    notes, t = [], 0.0
    for _ in range(200):
        dur = float(rng.uniform(0.02, 2.5))
        notes.append(Note(start=t, end=t + dur, midi=60, confidence=0.9))
        t += dur + 0.03
    y = render_notes(notes, sr=sr)
    assert len(y) >= int(t * sr)
    assert np.issubdtype(y.dtype, np.floating)


# ----------------------------------------------------------------------
# PlaybackSession: THE player (one OutputStream, one clock)
# ----------------------------------------------------------------------

class FakeOutputStream:
    """Records what a player streams; simulates a slow device drain."""

    instances: list["FakeOutputStream"] = []

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.kwargs = kwargs
        self.written = 0
        self.closed = False
        self.started = False
        FakeOutputStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def write(self, data) -> None:
        self.written += len(data)
        time.sleep(0.0005)  # the device drains slower than RAM

    def stop(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


def _fake_sd(monkeypatch) -> list[FakeOutputStream]:
    FakeOutputStream.instances = []
    monkeypatch.setattr(
        playback, "sd", types.SimpleNamespace(OutputStream=FakeOutputStream)
    )
    return FakeOutputStream.instances


def _notes(n: int = 30) -> list[Note]:
    return [
        Note(start=0.1 * i, end=0.1 * i + 0.09, midi=60 + (i % 5), confidence=0.9)
        for i in range(n)
    ]


def test_replaying_replaces_the_single_stream(monkeypatch):
    """Each play() owns exactly one OutputStream; the previous one is closed."""
    streams = _fake_sd(monkeypatch)
    session = PlaybackSession(sr=22050)
    session.set_sources(
        piano=render_notes(_notes(), sr=22050),
        voice=np.zeros(22050, dtype=np.float32),
    )
    assert session.play("piano")
    assert session.playing
    session.play("voice")  # re-playing closes the previous stream first
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not all(s.closed for s in streams[:-1]):
        time.sleep(0.005)
    assert session.playing
    assert len(streams) == 2
    assert all(s.closed for s in streams[:-1]), "replaying closes the previous stream"
    session.stop()
    assert not session.playing


def test_stop_ends_playback_and_finishes_at_duration(monkeypatch):
    _fake_sd(monkeypatch)
    session = PlaybackSession(sr=22050)
    session.set_sources(piano=render_notes(_notes(10), sr=22050))
    assert session.play("piano")
    assert session.playing
    session.stop()
    assert not session.playing
    assert session.position() == session.duration  # finished, not mid-signal
    # play() again after a stop must work (the generation is re-armed)
    assert session.play("piano")
    assert session.playing
    session.stop()


def test_playback_position_advances(monkeypatch):
    _fake_sd(monkeypatch)
    session = PlaybackSession(sr=22050)
    session.set_sources(piano=render_notes(_notes(200), sr=22050))  # ~20 s
    assert session.play("piano")
    time.sleep(0.05)
    assert session.playing, "the playback must last long enough to be observed"
    first = session.position()
    time.sleep(0.05)
    assert session.position() > first
    session.stop()


def test_pitch_shift_keeps_duration_and_changes_pitch():
    """The voice transpose: same timeline, shifted content."""
    sr = 22050
    y = sine_note_like(220.0, 1.0, sr)
    up = pitch_shift_signal(y, 3.0, sr)
    assert len(up) == len(y), "the duration must not move (sync with the piano)"
    assert not np.allclose(up, y)
    assert np.max(np.abs(up)) > 0.1
    same = pitch_shift_signal(y, 0, sr)
    assert np.array_equal(same, y)  # semitones == 0 is the identity


def _peak_frequency(signal: np.ndarray, sr: int) -> float:
    """Dominant frequency away from OLA edges."""
    x = signal[sr // 5 : -sr // 5]
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return float(freqs[np.argmax(np.abs(np.fft.rfft(x)))])


def test_pitch_shift_has_the_requested_frequency_and_keeps_the_tail():
    """Transposition is musical, not merely a different same-length signal."""
    sr = 22050
    y = sine_note_like(220.0, 1.5, sr)
    up = pitch_shift_signal(y, 3.0, sr)
    expected = 220.0 * 2.0 ** (3.0 / 12.0)
    assert _peak_frequency(up, sr) == pytest.approx(expected, abs=3.0)
    assert np.max(np.abs(up[-sr // 10 :])) > 0.1
    assert float(np.sqrt(np.mean(up**2))) == pytest.approx(
        float(np.sqrt(np.mean(y**2))), rel=0.2
    )


def test_pitch_shift_supports_downward_transposition():
    sr = 22050
    y = sine_note_like(330.0, 1.0, sr)
    down = pitch_shift_signal(y, -4.0, sr)
    expected = 330.0 * 2.0 ** (-4.0 / 12.0)
    assert _peak_frequency(down, sr) == pytest.approx(expected, abs=3.0)


def test_timeline_clamps_seconds_and_samples():
    timeline = Timeline(samples=22050, sr=22050)
    assert timeline.duration == pytest.approx(1.0)
    assert timeline.sample_at(-1.0) == 0
    assert timeline.sample_at(2.0) == 22050
    assert timeline.clamp(2.0) == 1.0


# ---------------------------------------------------------------------------
# A piano that does not sound mechanical (user feedback: "trop robotique")
# ---------------------------------------------------------------------------


def _one_note(midi=60, strength=0.5, start=0.0, dur=0.5):
    return Note(start=start, end=start + dur, midi=midi, confidence=0.9, strength=strength)


def test_render_follows_the_take_dynamics():
    """Every key's velocity follows the take's measured accent: WITHIN one
    render, a loud note comes out louder than a soft one — the global
    normalization aligns the render's peak but keeps the ratios between
    notes (a lone note is renormalized, context-free, which is not
    audible)."""
    notes = [
        _one_note(start=0.0, dur=0.4, strength=1.0),
        _one_note(start=0.5, dur=0.4, strength=0.0),
    ]
    y = render_notes(notes, sr=22050)

    def rms(t0, t1):
        win = y[int(t0 * 22050) : int(t1 * 22050)].astype(np.float64)
        return float(np.sqrt(np.mean(win**2)))

    loud, soft = rms(0.05, 0.35), rms(0.55, 0.85)
    assert loud > soft * 1.5


def test_render_is_deterministic_and_humanized():
    """Two renders of the same notes are IDENTICAL (humanization is not
    randomness), and a SHORT note (< 0.35 s) does not occupy its whole
    written duration: the damper lifts the key a little early."""
    notes = [_one_note(dur=0.3)]
    a = render_notes(notes, sr=22050)
    b = render_notes([_one_note(dur=0.3)], sr=22050)
    assert np.array_equal(a, b)
    # the tail is quiet before the nominal end: the key was lifted early
    tail = a[int(0.293 * 22050) : int(0.299 * 22050)]
    assert float(np.max(np.abs(tail))) < 0.2 * float(np.max(np.abs(a)))


def test_a_held_note_rings_through_a_pedal_tail():
    """The pedal (user request): a HELD note (>= 0.35 s) keeps resonating
    after the key is lifted — the tail decays but stays audible beyond
    the written end, then dies out (the pedal does not ring forever). A
    short note, meanwhile, stays damped at the written end (previous
    test).
    """
    from v2m.core import playback as pb

    notes = [_one_note(dur=0.8)]
    a = render_notes(notes, sr=22050)
    dur = int(0.8 * 22050)
    # beyond the written end (signal continues -> the render is longer now)
    assert len(a) > dur + int(0.1 * 22050)
    after = a[dur + int(0.03 * 22050) : dur + int(0.08 * 22050)]
    assert float(np.max(np.abs(after))) > 0.1 * float(np.max(np.abs(a)))
    # ...and the tail actually dies before its end (bounded pedal)
    far = a[dur + int(pb.PEDAL_TAIL_MAX_S * 22050) - 50 : dur + int(pb.PEDAL_TAIL_MAX_S * 22050)]
    assert float(np.max(np.abs(far))) < 0.05


def test_velocity_brightens_the_tone():
    """Velocity brightens the timbre (user request): at equal envelope
    attack, a LOUD note keeps more energy in the upper partials than a
    soft one (the partials' amplitude exponent depends on velocity).
    Measured on the spectrum of the same instant: the 2..6 kHz / 0..1 kHz
    energy ratio is larger for the loud note (the render is normalized:
    the global level cannot lie).
    """
    loud = render_notes([_one_note(strength=1.0, dur=0.45)], sr=22050)
    soft = render_notes([_one_note(strength=0.0, dur=0.45)], sr=22050)

    def brightness(y):
        win = y[int(0.05 * 22050) : int(0.20 * 22050)].astype(np.float64)
        spec = np.abs(np.fft.rfft(win * np.hanning(len(win)))) ** 2
        freqs = np.fft.rfftfreq(len(win), 1 / 22050)
        hi = float(spec[(freqs >= 2000) & (freqs <= 6000)].sum())
        lo = float(spec[freqs <= 1000].sum())
        return hi / max(lo, 1e-12)

    assert brightness(loud) > brightness(soft) * 1.15, (brightness(loud), brightness(soft))


def test_render_notes_do_not_overlap_the_next_onset_too_much():
    """The human offset (a soft note starts ~18 ms late) stays under the
    collision window: two consecutive eighths do not overlap to the point
    of blurring the next attack."""
    notes = [_one_note(start=0.0, dur=0.3, strength=0.0), _one_note(start=0.3, dur=0.3, strength=0.0)]
    y = render_notes(notes, sr=22050)
    assert len(y) > int(0.62 * 22050)  # the shifted second note still renders


def test_playback_session_mixes_with_one_stream(monkeypatch):
    streams = _fake_sd(monkeypatch)
    session = PlaybackSession(sr=22050)
    session.set_sources(
        piano=np.ones(22050, dtype=np.float32) * 0.2,
        voice=np.ones(22050, dtype=np.float32) * 0.3,
    )
    assert session.play("both")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not session.playing:
        time.sleep(0.005)
    session.stop()
    assert len(streams) == 1
    assert streams[0].written > 0


def sine_note_like(hz: float, dur: float, sr: int) -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * hz * np.arange(int(dur * sr)) / sr)).astype(np.float32)
