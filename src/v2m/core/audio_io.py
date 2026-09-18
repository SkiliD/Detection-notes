"""Chargement audio (fichier) et capture micro (temps réel)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - pas de périphérique audio dans CI
    sd = None

SR = 22_050  # fréquence de travail pour l'analyse


def load_file(path: str, sr: int = SR) -> np.ndarray:
    """Charge un fichier audio en mono, resamplé à `sr` (float32 dans [-1, 1])."""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32)


@dataclass
class LiveCapture:
    """Capture micro par blocs, avec callback utilisateur.

    Les blocs sont resamplés grossièrement vers `sr` (linéaire) pour rester
    cohérents avec le moteur d'analyse.
    """

    samplerate: int = 44_100          # fréquence du périphérique
    blocksize: int = 1024             # ~23 ms par bloc
    sr: int = SR                      # fréquence d'analyse
    on_block: Optional[Callable[[np.ndarray, float], None]] = None
    chunks: list = field(default_factory=list, repr=False)
    t0: float = 0.0
    _stream: object = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[micro] {status}")
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        self.chunks.append(mono.copy())
        if self.on_block is not None:
            self.on_block(mono, self.elapsed())

    def elapsed(self) -> float:
        n = sum(len(c) for c in self.chunks)
        return n / self.samplerate

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice indisponible (pas de périphérique audio ?)")
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def signal(self) -> np.ndarray:
        """Signal complet enregistré, resamplé vers `sr` (pour analyse finale)."""
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        y = np.concatenate(self.chunks).astype(np.float32)
        if self.samplerate == self.sr:
            return y
        ratio = self.sr / self.samplerate
        n_out = int(len(y) * ratio)
        idx = np.linspace(0, len(y) - 1, n_out)
        return np.interp(idx, np.arange(len(y)), y).astype(np.float32)

    @property
    def duration(self) -> float:
        return len(self.signal()) / self.sr if self.chunks else 0.0
