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
    assert all(s.closed for s in streams[:-1]), "rejouer ferme le flux précédent"
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
    assert session.playing, "la lecture doit durer assez pour être observée"
    first = session.position()
    time.sleep(0.05)
    assert session.position() > first
    session.stop()


def test_pitch_shift_keeps_duration_and_changes_pitch():
    """The voice transpose: same timeline, shifted content."""
    sr = 22050
    y = sine_note_like(220.0, 1.0, sr)
    up = pitch_shift_signal(y, 3.0, sr)
    assert len(up) == len(y), "la durée ne doit pas bouger (synchro avec le piano)"
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
# Un piano qui ne sonne pas mécanique (retour utilisateur : "trop robotique")
# ---------------------------------------------------------------------------


def _one_note(midi=60, strength=0.5, start=0.0, dur=0.5):
    return Note(start=start, end=start + dur, midi=midi, confidence=0.9, strength=strength)


def test_render_follows_the_take_dynamics():
    """La vélocité de chaque touche suit l'accent mesuré de la prise :
    DANS un même rendu, une note forte sort plus forte qu'une faible — la
    normalisation globale aligne le sommet du rendu mais garde les
    rapports entre notes (une note seule est renormalisée, sans contexte,
    ce qui ne se entend pas)."""
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
    """Deux rendus des mêmes notes sont IDENTIQUES (l'humanisation n'est
    pas du hasard), et une note de plus d'un temps n'occupe pas toute sa
    durée écrite (le damper lève la touche un peu tôt)."""
    notes = [_one_note(dur=0.8)]
    a = render_notes(notes, sr=22050)
    b = render_notes([_one_note(dur=0.8)], sr=22050)
    assert np.array_equal(a, b)
    # the tail is quiet before the nominal end: the key was lifted early
    tail = a[int(0.75 * 22050) : int(0.79 * 22050)]
    assert float(np.max(np.abs(tail))) < 0.2 * float(np.max(np.abs(a)))


def test_render_notes_do_not_overlap_the_next_onset_too_much():
    """Le décalage humain (une note faible part ~18 ms tard) reste sous la
    fenêtre de collision : deux croches consécutives ne se recouvrent pas
    au point de brouiller l'attaque de la suivante."""
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
