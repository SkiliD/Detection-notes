"""Real-time session (CLI): microphone capture -> live display -> export."""

from __future__ import annotations

import time

import numpy as np

from .core import audio_io
from .core.audio_io import LiveCapture
from .core.figures import label_notes
from .core.i18n import note_display_pair, tr, tr_figure
from .core.liveseg import LivePitchStream
from .core.pipeline import AnalyzeOptions, analyze_signal
from .core.report import export_result
from .core.tempo import estimate_bpm_live


class LiveSession:
    """Capture from the mic, display detected notes, export on stop."""

    def __init__(self, min_conf: float = 0.5, transpose: int = 0, out_stem: str | None = None):
        self.min_conf = min_conf
        self.transpose = transpose
        self.out_stem = out_stem or f"live_{time.strftime('%Y%m%d_%H%M%S')}"
        self.stream: LivePitchStream | None = None
        self.capture: LiveCapture | None = None

    @property
    def seg(self):
        """The live segmenter (owned by the shared live chain)."""
        return self.stream.seg if self.stream is not None else None

    # ------------------------------------------------------------------
    def _on_block(self, block: np.ndarray, t: float) -> None:
        if self.stream is None:
            self.stream = LivePitchStream(self.capture.samplerate, gate_rms=audio_io.GATE_RMS)
        closed = self.stream.elapsed_notes()
        self.stream.feed(block, t)
        if self.stream.elapsed_notes() == closed:
            return  # no note ended in this block
        notes = self.stream.seg.notes
        n = notes[-1]
        # label at note END so the figure lands with the note itself
        bpm = estimate_bpm_live(notes)
        if bpm:
            label_notes(notes, bpm)
        fig = f"  [{tr_figure(n.figure)}]" if n.figure else ""
        print(f"  ♪ {note_display_pair(n.midi)}  ({n.duration:.2f} s){fig}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        print(tr("cli live banner"))
        self.capture = LiveCapture(on_block=self._on_block)
        self.capture.start()
        print(f"  Périphérique : {self.capture.samplerate} Hz, blocs de {self.capture.blocksize}")
        try:
            while True:
                time.sleep(0.1)
                if self.seg is None:
                    continue
                cur = self.seg.display_text()
                total = len(self.seg.notes)
                print(
                    f"\r  {self.capture.elapsed():7.1f} s | {cur:>24} | {total} notes ",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.capture.stop()
            if self.seg is not None:
                self.seg.flush(self.capture.elapsed())
        print(tr("cli stop capture"))
        self.export()

    # ------------------------------------------------------------------
    def export(self) -> str | None:
        y = self.capture.signal() if self.capture else np.zeros(0)
        if len(y) < audio_io.SR // 2:
            print(tr("cli too short"))
            return None
        print(tr("key final analysis"))
        res = analyze_signal(y, audio_io.SR, source="live", opts=AnalyzeOptions(min_conf=self.min_conf))
        export_result(res, self.out_stem, transpose=self.transpose)
        print(f"{tr('key tempo')}    : {res.tempo.bpm} BPM")
        print(f"{tr('key key label')} : {res.key.best.label}")
        if res.chord:
            print(f"{tr('key chord label')} : {res.chord.best.label}")
        print(f"{tr('key notes label')} : {len(res.notes)}")
        print(f"Export   : {self.out_stem}.mid, {self.out_stem}.musicxml, {self.out_stem}.json, {self.out_stem}.csv")
        return f"{self.out_stem}.mid"
