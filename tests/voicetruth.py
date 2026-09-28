"""Ground truth: a synthetic voice whose notes are all known.

A pure sine tests nothing useful here: it has no transition, no consonant, no
vibrato — none of what makes segmentation fail. This module builds a "sung"
signal (glottal harmonics, portamento between notes, wide-band consonants,
vibrato, background noise) whose events are known exactly, then compares what
the pipeline detects against that truth.

Two uses:

    from voicetruth import sequence, synthesize, detect_notes, measure
    truth = sequence([60, 62, 64])              # legato quarters, 100 BPM
    acc = measure(detect_notes(synthesize(truth)), truth)
    assert acc.exact

`scripts/score_accuracy.py` uses it to score a real take.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 22_050


@dataclass(frozen=True)
class SungNote:
    """A note actually sung (the reference).

    `amp`: the note's RELATIVE amplitude in the take (1.0 by default) —
    it is what manufactures the accents the attack strength must read.
    """

    midi: int
    start_s: float
    dur_s: float
    amp: float = 1.0

    @property
    def hz(self) -> float:
        return 440.0 * 2.0 ** ((self.midi - 69) / 12.0)


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #


def sequence(
    midis,
    start_s: float = 0.0,
    dur_s: float = 0.6,
    gap_s: float = 0.0,
    legato_s: float = 0.0,
) -> list[SungNote]:
    """Successive notes: `legato_s` > 0 chains them (portamento, no silence)."""
    step = dur_s + gap_s - legato_s
    return [SungNote(m, start_s + i * step, dur_s) for i, m in enumerate(midis)]


def _harmonic_voice(f0: np.ndarray, sr: int, f_max: float = 4000.0) -> np.ndarray:
    """Harmonic sum with a glottal slope (1/k^1.4), capped near 4 kHz."""
    phase = 2.0 * np.pi * np.cumsum(f0) / sr
    base = float(np.mean(f0))
    n_harm = max(1, int(f_max / max(base, 1.0)))
    out = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        out += np.sin(k * phase) / k ** 1.4
    return out


def synthesize(
    notes: list[SungNote],
    sr: int = SR,
    glide_s: float = 0.09,
    consonant_s: float = 0.05,
    consonant_amp: float = 1.4,
    vibrato_st: float = 0.0,
    vibrato_hz: float = 5.5,
    noise: float = 0.0,
    amp: float = 0.35,
    seed: int = 0,
    closure_s: float = 0.0,
    tremolo_depth: float = 0.0,
    tremolo_hz: float = 5.5,
) -> np.ndarray:
    """Mono float32 signal carrying `notes`.

    `glide_s`: duration of the attack portamento (0 = crisp attack). The
        portamento only exists between LEGATO notes: after a silence, the
        singer re-strikes the note (that is what a real détaché does).
    `consonant_s`: duration of the wide-band attack burst (0 = none).
    `vibrato_st`: vibrato depth in semitones (±value), 0 = steady hold.
    `noise`: amplitude of the background noise added everywhere.
    `closure_s`: glottal closure at the seam of each LEGATO note (the
        first excepted): a level notch (~5% for the given duration) that
        models the singer's re-emission. Without it, two glued notes are
        one continuous envelope — no detector, human or machine, can see
        a seam there (measured: level flat at ±5%, tonal spectrum); a
        real singer interrupts the airflow, and it is THAT interruption
        the live segmentation cuts on, via its mute frames
        (SPLIT_FRAMES).
    `tremolo_depth`: amplitude modulation (0.30 = ±30%) — the case that
        must NEVER fragment into "notes" at every cycle.
    `tremolo_hz`: modulation speed (5.5 Hz = vibrato of amplitude).
    """
    if not notes:
        return np.zeros(0, dtype=np.float32)
    end = max(n.start_s + n.dur_s for n in notes) + 0.3
    y = np.zeros(int(end * sr), dtype=np.float64)
    rng = np.random.default_rng(seed)
    prev_hz: float | None = None
    prev_end = 0.0
    phase0 = 0.0

    for note in notes:
        i0 = int(note.start_s * sr)
        ln = max(1, int(note.dur_s * sr))
        t = np.arange(ln) / sr
        hz = note.hz
        f0 = np.full(ln, hz)
        legato = note.start_s <= prev_end + 0.05
        if prev_hz and glide_s > 0 and legato:
            g = min(int(glide_s * sr), ln)
            f0[:g] = np.linspace(prev_hz, hz, g)  # linear portamento in Hz
        if vibrato_st > 0:
            depth = np.clip((t - 0.12) / 0.12, 0.0, 1.0)  # vibrato settling in
            f0 = f0 * 2.0 ** (vibrato_st * np.sin(2 * np.pi * vibrato_hz * t) * depth / 12.0)

        # phase continuous from one note to the next: no click at the seam
        phase = phase0 + 2.0 * np.pi * np.cumsum(f0) / sr
        phase0 = float(phase[-1] + 2.0 * np.pi * f0[-1] / sr)
        env = _envelope(t, ln / sr)
        if closure_s > 0 and legato:
            env = env * _glottal_closure(t, ln / sr, closure_s)
        if tremolo_depth > 0:
            env = env * (1.0 + tremolo_depth * np.sin(2 * np.pi * tremolo_hz * (t + note.start_s)))
        body = _harmonic_voice(f0, sr) * env

        if consonant_s > 0:
            cn = min(int(consonant_s * sr), ln)
            burst = rng.standard_normal(cn) * _consonant_envelope(cn, sr)
            body[:cn] += consonant_amp * burst
        y[i0 : i0 + ln] += body * note.amp
        prev_hz = hz
        prev_end = note.start_s + note.dur_s

    y *= amp
    if noise > 0:
        y += noise * amp * rng.standard_normal(len(y))
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * amp).astype(np.float32)


def _envelope(t: np.ndarray, dur_s: float) -> np.ndarray:
    """25 ms attack, 40 ms release — no click at either end."""
    env = np.ones_like(t)
    a = min(int(0.025 * len(t) / max(dur_s, 1e-9)), len(t) // 2)
    r = min(int(0.040 * len(t) / max(dur_s, 1e-9)), len(t) // 2)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def _glottal_closure(t: np.ndarray, dur_s: float, closure_s: float) -> np.ndarray:
    """Glottal-closure notch at the start of a legato note (re-emission).

    The level falls to ~5% for `closure_s` (~30 ms, measured on sung
    repetitions), with 12 ms edges — not a silence (the breath and the
    note's body remain; it is a re-emission, not a restart after
    silence) but a dip the level measure reads.
    """
    env = np.ones_like(t)
    n = int(closure_s * len(t) / max(dur_s, 1e-9))
    f = min(int(0.012 * len(t) / max(dur_s, 1e-9)), n // 3)
    if n > 0 and f > 0:
        env[:f] = np.linspace(1.0, 0.05, f)
        env[f : n - f] = 0.05
        env[n - f : n] = np.linspace(0.05, 1.0, f)
    return env


def _consonant_envelope(n: int, sr: int) -> np.ndarray:
    """Percussive burst: very short, fast decay."""
    t = np.arange(n) / sr
    return np.exp(-t / max(0.012, 1e-9))


def add_breath(y: np.ndarray, at: float, dur_s: float = 0.09, amp: float = 0.25,
               seed: int = 7) -> None:
    """Adds an air intake (windowed noise burst) into `y`, in place.

    The singer breathes: in the gap of a legato line (just before the
    re-strike), in the silence between two phrases, or as the continuous
    breath of a breathy voice (synthesize's `noise` parameter).
    Segmentation must NEVER turn it into a note.
    """
    i0 = int(at * SR)
    n = int(dur_s * SR)
    breath = np.random.default_rng(seed).standard_normal(n) * amp * np.hanning(n)
    y[i0 : i0 + n] += breath.astype(np.float32)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


@dataclass
class Accuracy:
    """What the chain made of the sung notes."""

    truth_n: int
    detected_n: int
    matched: int
    missing: int
    extra: int
    intruders: int          # extra notes INSIDE a sung note
    worst_start_ms: float
    worst_pitch_st: float

    @property
    def exact(self) -> bool:
        """Every note found, none extra (hence no fragment)."""
        return self.matched == self.truth_n and self.extra == 0

    def line(self) -> str:
        return (
            f"{self.matched}/{self.truth_n} notes, {self.extra} extra "
            f"({self.intruders} inside), attack gap <= "
            f"{self.worst_start_ms:.0f} ms"
        )


def detect_notes(y: np.ndarray, sr: int = SR, with_attacks: bool = True):
    """Full chain pitch -> attacks -> segmentation."""
    from v2m.core.notes import interpret
    from v2m.core.observation import VoiceObservation

    # one observation (cleaned curve + attacks + clock), then interpretation:
    # the evidence is built by the module that owns it, and this harness feeds
    # the SAME two doors as the pipeline
    observation = VoiceObservation.from_signal(y, sr, use_attacks=with_attacks)
    return interpret(observation)


def measure(
    detected,
    truth: list[SungNote],
    tol_s: float = 0.12,
    pitch_tol_st: float = 0.5,
) -> Accuracy:
    """Matches detected notes to sung notes (same attack, same pitch).

    A transition fragment (an extra note between two real ones) counts
    as `intruders`: it is exactly the defect attack detection must
    remove, and it would hide inside a plain total note count.
    """
    used = [False] * len(detected)
    matched = 0
    worst_start = 0.0
    worst_pitch = 0.0
    for ev in truth:
        best_j: int | None = None
        best_dt = tol_s
        for j, d in enumerate(detected):
            if used[j] or d.midi != ev.midi:
                continue
            dt = abs(d.start - ev.start_s)
            if dt <= best_dt:
                best_j, best_dt = j, dt
        if best_j is None:
            continue
        used[best_j] = True
        matched += 1
        worst_start = max(worst_start, best_dt * 1000.0)
        worst_pitch = max(worst_pitch, abs(detected[best_j].midi - ev.midi))

    intruders = 0
    for j, d in enumerate(detected):
        if used[j]:
            continue
        for ev in truth:
            if ev.start_s - tol_s <= d.start <= ev.start_s + ev.dur_s + tol_s:
                intruders += 1
                break

    return Accuracy(
        truth_n=len(truth),
        detected_n=len(detected),
        matched=matched,
        missing=len(truth) - matched,
        extra=sum(1 for j in range(len(detected)) if not used[j]),
        intruders=intruders,
        worst_start_ms=round(worst_start, 1),
        worst_pitch_st=round(worst_pitch, 2),
    )


def describe(acc: Accuracy) -> str:
    """Readable line for script reports."""
    flag = "OK " if acc.exact else "→  "
    return f"{flag}{acc.line()}"
