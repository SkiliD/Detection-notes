"""Real-time CLI adapter for the shared live take lifecycle."""

from __future__ import annotations

import time

from .core.i18n import note_display_pair, tr, tr_figure
from .core.live_session import LiveEvent, LiveTake
from .core.report import export_result, live_export_dir, live_take_name


class LiveSession:
    """CLI adapter: print a shared LiveTake and export its final result."""

    def __init__(self, min_conf: float = 0.5, transpose: int = 0, out_stem: str | None = None):
        self.min_conf = min_conf
        self.transpose = transpose
        self.out_stem = out_stem or live_take_name()
        self.take = LiveTake(min_conf=min_conf, on_event=self._on_event)

    def _on_event(self, event: LiveEvent) -> None:
        if event.kind == "note_closed" and event.note is not None:
            note = event.note
            # The figures are already labeled by LiveTake (the policy's
            # single owner) BEFORE this event is emitted: the adapter only
            # prints them.
            fig = f"  [{tr_figure(note.figure)}]" if note.figure else ""
            print(f"  ♪ {note_display_pair(note.midi)}  ({note.duration:.2f} s){fig}")

    def run(self) -> None:
        print(tr("cli live banner"))
        self.take.start()
        try:
            while True:
                time.sleep(0.1)
                seg = self.take.segmenter
                if seg is None:
                    continue
                cur = seg.display_text()
                total = len(seg.notes)
                print(
                    f"\r  {self.take.elapsed:7.1f} s | {cur:>24} | {total} notes ",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.take.stop()
        print(tr("cli stop capture"))
        self.export()

    def export(self) -> str | None:
        if self.take.state != "stopped":
            self.take.stop()
        if self.take.result is None and self.take.signal.size < self.take.sr // 2:
            print(tr("cli too short"))
            return None
        print(tr("key final analysis"))
        # The analysis has ONE path: LiveTake.finalize owns the drain, the
        # length gate and the idempotency; the adapter only exports what it
        # returns (a short take re-raises the take's own 'failed' error).
        res, _ = self.take.finalize()
        if res.tempo is None or res.key is None:
            raise ValueError("analyse live incomplète")
        take_dir = live_export_dir(self.out_stem)
        export_result(res, take_dir / self.out_stem, transpose=self.transpose)
        print(f"{tr('key tempo')}    : {res.tempo.bpm} BPM")
        print(f"{tr('key key label')} : {res.key.best.label}")
        if res.chord:
            print(f"{tr('key chord label')} : {res.chord.best.label}")
        print(f"{tr('key notes label')} : {len(res.notes)}")
        print(
            f"Export   : {take_dir / self.out_stem}.mid, {take_dir / self.out_stem}.musicxml, "
            f"{take_dir / self.out_stem}.json, {take_dir / self.out_stem}.csv"
        )
        return str(take_dir / f"{self.out_stem}.mid")
