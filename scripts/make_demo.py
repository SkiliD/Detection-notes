"""Generate demo.wav: a voice-like melody (harmonics + vibrato) at 120 BPM."""

import numpy as np
import soundfile as sf

SR = 22_050
BPM = 120.0
BEAT = 60.0 / BPM  # 0.5 s


def voice_note(f0: float, dur: float, sr: int = SR, amp: float = 0.4) -> np.ndarray:
    """A voice-like note: decaying harmonics + 5 Hz ±20 cents vibrato."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    vib_st = 0.2 * np.sin(2 * np.pi * 5.0 * t)          # ±0.2 semitone = ±20 cents
    freq = f0 * 2.0 ** (vib_st / 12.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    y = np.zeros(n)
    for k, a in enumerate((1.0, 0.5, 0.25, 0.12, 0.06), start=1):
        y += a * np.sin(k * phase)
    # envelope: 20 ms attack, 60 ms release
    atk = int(0.02 * sr)
    rel = int(0.06 * sr)
    env = np.ones(n)
    env[:atk] = np.linspace(0, 1, atk)
    env[-rel:] = np.linspace(1, 0, rel)
    return (amp * y * env).astype(np.float32)


# C major scale up + down (C4..C5..C4), long final tonic
scale_hz = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25]
seq = scale_hz + list(reversed(scale_hz[:-1])) + [261.63]
durs = [BEAT] * (len(seq) - 1) + [2 * BEAT]

parts = []
for hz, d in zip(seq, durs):
    parts.append(voice_note(hz, d - 0.04))
    parts.append(np.zeros(int(0.04 * SR), dtype=np.float32))  # articulation micro-gap

y = np.concatenate(parts)
sf.write("demo.wav", y, SR)
print(f"demo.wav written: {len(y)/SR:.2f} s, a {len(seq)}-note melody at {BPM} BPM")
