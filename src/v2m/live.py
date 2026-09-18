"""Session temps réel (CLI) : capture micro -> affichage live -> export."""

from __future__ import annotations

import time

import numpy as np

from .core import audio_io
from .core.audio_io import LiveCapture
from .core.liveseg import LiveNoteSegmenter
from .core.music import midi_to_name
from .core.pipeline import AnalyzeOptions, analyze_signal, export_result
from .core.pitch import YinBlock

GATE_RMS = 0.008  # en dessous : considéré comme silence


class LiveSession:
    """Capture au micro, affiche les notes détectées, exporte à l'arrêt."""

    def __init__(self, min_conf: float = 0.5, transpose: int = 0, out_stem: str | None = None):
        self.min_conf = min_conf
        self.transpose = transpose
        self.out_stem = out_stem or f"live_{time.strftime('%Y%m%d_%H%M%S')}"
        self.seg = LiveNoteSegmenter()
        self.yin: YinBlock | None = None
        self.capture: LiveCapture | None = None

    # ------------------------------------------------------------------
    def _on_block(self, block: np.ndarray, t: float) -> None:
        if self.yin is None:
            self.yin = YinBlock(self.capture.samplerate, window=2048, hop=1024)
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        frames = self.yin.push(block)
        for ft, hz in frames:
            if rms < GATE_RMS:
                hz = float("nan")
            ev = self.seg.push(ft, hz)
            if ev == "on":
                pass  # l'affichage courant suffit
            elif ev == "off" and self.seg.notes:
                n = self.seg.notes[-1]
                print(f"  ♪ {midi_to_name(n.midi)}  ({n.duration:.2f} s)")

    # ------------------------------------------------------------------
    def run(self) -> None:
        print("🎤 Mode temps réel — chantez ! (Ctrl+C pour arrêter et exporter)")
        self.capture = LiveCapture(on_block=self._on_block)
        self.capture.start()
        print(f"  Périphérique : {self.capture.samplerate} Hz, blocs de {self.capture.blocksize}")
        try:
            while True:
                time.sleep(0.1)
                cur = self.seg.display_text()
                print(f"\r  {self.capture.elapsed():7.1f} s | {cur:>16} ", end="", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            self.capture.stop()
            self.seg.flush(self.capture.elapsed())
        print("\nArrêt de la capture.")
        self.export()

    # ------------------------------------------------------------------
    def export(self) -> str | None:
        y = self.capture.signal() if self.capture else np.zeros(0)
        if len(y) < audio_io.SR // 2:
            print("Enregistrement trop court : rien à exporter.")
            return None
        print("Analyse finale (tempo, tonalité, notes)…")
        res = analyze_signal(y, audio_io.SR, source="live", opts=AnalyzeOptions(min_conf=self.min_conf))
        export_result(res, self.out_stem, transpose=self.transpose)
        print(f"Tempo    : {res.tempo.bpm} BPM")
        print(f"Tonalité : {res.key.best.label}")
        print(f"Notes    : {len(res.notes)}")
        print(f"Export   : {self.out_stem}.mid, {self.out_stem}.musicxml, {self.out_stem}.json, {self.out_stem}.csv")
        return f"{self.out_stem}.mid"
