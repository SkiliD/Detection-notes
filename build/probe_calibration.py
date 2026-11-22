"""Throwaway probe v3: real chain measurements, mode by mode (build/, ignored).

Modes:
  chain   one held tone through HeldMic -> live segmenter -> pitchFrame
  seam    two tones back to back, frames around the change
  hold    the card's rounds with the real 400 ms tick (hold rule)
  old     the HEAD rule (freshness only) fed the same frames
"""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

BASE = 440.0


def cents(hz: float, base: float) -> float:
    return 1200.0 * math.log2(hz / base)


def pump(app, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def pump_until(app, predicate, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def make_window():
    from v2m.core import mic_cal
    from v2m.gui import main_window as mw

    mic_cal.CAL_FILE = pathlib.Path(tempfile.mkdtemp()) / "cal.json"
    win = mw.MainWindow()
    win.show()
    return win, mw


def open_live(qapp, fake=None):
    import sounddevice as sd

    from tests.test_calibration_e2e import HeldMic

    fake = fake or HeldMic
    fake.instances = []
    original = sd.InputStream
    sd.InputStream = fake
    win, mw = make_window()
    win.on_start_live()
    live = win.live_thread
    assert live is not None
    assert pump_until(qapp, lambda: bool(fake.instances) and live.elapsed() >= 0.4, 30)
    return win, mw, fake.instances[-1], original


class ChangeMic:
    """A fake InputStream that can sing a CONTINUOUS change between two tones.

    `HeldMic` switches abruptly at a block boundary. A singer slides between
    two notes (a glissando, phase continuous) or a chord change puts both tones
    in the same window (a crossfade); both are rendered here sample by sample,
    so the chain meets a real passage instead of a splice.
    """

    instances: list = []

    def __init__(self, samplerate, channels, blocksize, callback, **kwargs):  # noqa: ANN001
        self.rate = samplerate
        self.block = blocksize
        self.callback = callback
        self._parts: list[tuple[str, tuple[float, float, float, float], float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        ChangeMic.instances.append(self)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        import numpy as np

        quiet = np.zeros(self.block, np.float32).reshape(-1, 1)
        while not self._stop.is_set():
            if self._parts:
                kind, args, seconds = self._parts.pop(0)
                self._emit(kind, args, seconds)
            else:
                self.callback(quiet, self.block, None, None)
                time.sleep(0.005)

    def _render(self, kind: str, args: tuple[float, float, float, float], seconds: float):
        import numpy as np

        n = int(self.rate * seconds)
        t = np.arange(n) / self.rate
        f0 = args[0] * 2.0 ** (args[1] / 1200.0)
        f1 = args[2] * 2.0 ** (args[3] / 1200.0)
        if kind == "tone":
            return 0.35 * np.sin(2 * np.pi * f0 * t)
        if kind == "glide":
            # A musical glide is linear in LOG frequency, and a real voice has
            # no phase jump at the block boundary: integrate the instantaneous
            # frequency instead of restarting a sine per block.
            f = f0 * (f1 / f0) ** (t / seconds)
            return 0.35 * np.sin(2 * np.pi * np.cumsum(f) / self.rate)
        # fade: two steady tones, one linear crossfade between them
        ramp = t / seconds
        return 0.35 * (
            (1.0 - ramp) * np.sin(2 * np.pi * f0 * t)
            + ramp * np.sin(2 * np.pi * f1 * t)
        )

    def _emit(self, kind: str, args: tuple[float, float, float, float], seconds: float) -> None:
        import numpy as np

        tone = self._render(kind, args, seconds).astype(np.float32)
        for i in range(0, len(tone), self.block):
            if self._stop.is_set():
                return
            blk = tone[i : i + self.block]
            if len(blk) < self.block:
                blk = np.pad(blk, (0, self.block - len(blk)))
            self.callback(blk.reshape(-1, 1), self.block, None, None)
            time.sleep(0.005)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)

    def close(self) -> None:
        pass

    def sing(self, hz: float, cents: float = 0.0, seconds: float = 4.0) -> None:
        self._parts.append(("tone", (hz, cents, hz, cents), seconds))

    def glide(self, hz_from: float, hz_to: float, seconds: float) -> None:
        self._parts.append(("glide", (hz_from, 0.0, hz_to, 0.0), seconds))

    def fade(self, hz_from: float, hz_to: float, seconds: float) -> None:
        self._parts.append(("fade", (hz_from, 0.0, hz_to, 0.0), seconds))


def close(win, original) -> None:
    import sounddevice as sd

    if original is not None:
        sd.InputStream = original
    if win.live_thread is not None and win.live_thread.isRunning():
        win.live_thread.stop()
        win.live_thread.wait(6000)
    win.close()


class Recorder:
    def __init__(self) -> None:
        self.frames: list[tuple[float, float]] = []
        self.t0 = time.monotonic()

    def __call__(self, t: float, hz: float) -> None:
        self.frames.append((time.monotonic() - self.t0, hz))

    def summary(self, label: str, base: float) -> list[float]:
        good = [(t, hz) for t, hz in self.frames if math.isfinite(hz) and hz > 0]
        print(f"  {label}: {len(self.frames)} emits, {len(good)} finite")
        if not good:
            return []
        values = [cents(hz, base) for _, hz in good]
        print(f"    first t={good[0][0]:.3f} cents={values[0]:+.2f}")
        print(f"    last  t={good[-1][0]:.3f} cents={values[-1]:+.2f}")
        median = sorted(values)[len(values) // 2]
        print(f"    min={min(values):+.2f} max={max(values):+.2f} median={median:+.2f}")
        for t, hz in good[:6]:
            print(f"    + t={t:.3f} hz={hz:.3f} c={cents(hz, base):+.2f}")
        return values


def mode_chain(qapp) -> int:
    for hz in (BASE * 2 ** (-22 / 1200), 261.63 * 2 ** (-22 / 1200)):
        print(f"== held tone {hz:.2f} Hz from silence ==")
        win, mw, mic, original = open_live(qapp)
        rec = Recorder()
        try:
            live = win.live_thread
            assert live is not None
            live.pitchFrame.connect(rec)
            pump(qapp, 0.6)
            mic.sing(hz, 0.0, 3.0)
            pump(qapp, 3.6)
            mic.sing(hz, 0.0, 1.0)  # a second round of the same note
            pump(qapp, 1.4)
        finally:
            close(win, original)
        rec.summary("frames", hz)
    return 0


def mode_seam(qapp) -> int:
    print("== two tones back to back: A4-122c then A4-22c ==")
    win, mw, mic, original = open_live(qapp)
    rec = Recorder()
    try:
        live = win.live_thread
        assert live is not None
        live.pitchFrame.connect(rec)
        pump(qapp, 0.5)
        mic.sing(BASE * 2 ** (-122 / 1200), 0.0, 1.5)
        mic.sing(BASE * 2 ** (-22 / 1200), 0.0, 2.0)
        pump(qapp, 4.2)
    finally:
        close(win, original)
    rec.summary("all frames vs 440", BASE)
    print("  the change is between +6 c (old tone) and -22 c (new tone):")
    for t, hz in rec.frames:
        if math.isfinite(hz) and hz > 0:
            c = cents(hz, BASE)
            if -115.0 < c < -10.0:
                print(f"    t={t:.3f} hz={hz:.3f} c={c:+.2f}")
    return 0


def mode_change(qapp) -> int:
    """A CONTINUOUS change (glide, then crossfade): what the seam frames read.

    The abrupt switch of `HeldMic` is one shape of a note change; a voice
    slides, and a chord change sums both tones in one window. Both are fed to
    the same live chain, and every frame of the passage is printed in cents
    from the new target, so a figure claimed in a test can be checked against
    a passage that really produces it.
    """
    old = BASE * 2 ** (-122 / 1200)
    new = BASE * 2 ** (-22 / 1200)
    for kind, seconds in (("glide", 0.20), ("fade", 0.15)):
        print(f"== continuous {kind}: -122c -> -22c over {seconds * 1000:.0f} ms ==")
        win, mw, mic, original = open_live(qapp, ChangeMic)
        rec = Recorder()
        try:
            live = win.live_thread
            assert live is not None
            live.pitchFrame.connect(rec)
            pump(qapp, 0.6)
            mic.sing(old, 0.0, 1.0)
            if kind == "glide":
                mic.glide(old, new, seconds)
            else:
                mic.fade(old, new, seconds)
            mic.sing(new, 0.0, 1.6)
            pump(qapp, 3.4)
        finally:
            close(win, original)
        values = [
            (t, cents(hz, BASE))
            for t, hz in rec.frames
            if math.isfinite(hz) and hz > 0
        ]
        print(f"  {len(values)} finite frames, cents from the new target (target = -22):")
        for t, c in values:
            print(f"    t={t:.3f} c={c:+.2f} (vs target {c - (-22):+.2f})")
        if not values:
            print("  NO FRAMES")
            continue
        # The seam frame is pinned by the suite
        # (`test_a_150_ms_crossfade_seam_reads_minus_36_42_cents`): this probe
        # prints where the passage lands, it does not grade it.
        closest = min(values, key=lambda item: abs(item[1] + 36.42))
        print(
            f"  seam frame closest to -36.42: {closest[1]:+.2f} c at "
            f"t={closest[0]:.3f} (gap {closest[1] + 36.42:+.2f} c)"
        )
        transition = [(t, c) for t, c in values if -115.0 < c < -10.0]
        if transition:
            lo = min(transition, key=lambda item: item[1])
            hi = max(transition, key=lambda item: item[1])
            print(
                f"  transition frames: {len(transition)}, "
                f"range {hi[1]:+.2f} .. {lo[1]:+.2f} c"
            )
    return 0


def mode_hold(qapp) -> int:
    from v2m.core import mic_cal

    print("== hold rule: card + real 400 ms tick ==")
    win, mw, mic, original = open_live(qapp)
    try:
        win.stack.setCurrentWidget(win.page_home)
        win.page_home.btn_calibrate.click()
        from tests.test_calibration_e2e import HeldMic

        card = win.page_home.calibration_card
        print(
            f"  timer armed: {win._calibration_timer.isActive()} "
            f"mics: {len(HeldMic.instances)} own thread: {win._calibration_thread is not None}"
        )
        heard, targets = [], []
        for index in range(mic_cal.ROUNDS):
            target = card._targets[-1]
            targets.append(target)
            ok = False
            for _attempt in range(4):
                if card._round != index:
                    break
                for fake in HeldMic.instances:
                    fake.sing(target, -22.0, 1.2)
                    fake.sing(target, -22.0, 1.2)
                ok = pump_until(qapp, lambda i=index: card._round == i + 1, 4.0)
            print(
                f"  round {index + 1}: target={target:.2f} advanced={ok} "
                f"rounds={card._round} heard={card.heard_line.text()!r}"
            )
            if not ok:
                break
            heard.append(card._heard[-1])
        values = [cents(h, t) for h, t in zip(heard, targets, strict=False)]
        print("  cents:", ", ".join(f"{v:+.2f}" for v in values) or "(none)")
        if values:
            print(f"  mean = {sum(values) / len(values):+.2f} c")
        print("  prompt:", card.prompt.text())
        return 0 if len(values) == mic_cal.ROUNDS else 1
    finally:
        win._on_calibrate_toggled(False)
        close(win, original)


def mode_old(qapp) -> int:
    from v2m.core import mic_cal

    print("== HEAD rule (freshness only) ==")
    win, mw = make_window()
    try:
        win.stack.setCurrentWidget(win.page_home)
        win.page_home.btn_calibrate.click()
        card = win.page_home.calibration_card
        win._calibration_timer.stop()

        def old_tick() -> None:
            hz = win._calibration_hz
            now = time.monotonic()
            if hz is not None and now - win._calibration_at > mw.MainWindow.CALIBRATION_FRESH_S:
                hz = None
            card.set_heard(hz)
            if hz is None:
                return
            if card.feed(hz):
                win._calibration_round_started()

        seam_cents = float(sys.argv[2]) if len(sys.argv) > 2 else -36.05
        for index in range(mic_cal.ROUNDS):
            target = card._targets[-1]
            sung = seam_cents if index == 3 else -22.0
            win._remember_calibration_pitch(target * 2 ** (sung / 1200))
            old_tick()
            print(
                f"  round {index + 1}: target={target:.2f} sung={sung:+.2f}c "
                f"-> round {card._round}"
            )
        values = [
            cents(h, t) for h, t in zip(card._heard, card._targets, strict=False)
        ]
        print("  cents:", ", ".join(f"{v:+.2f}" for v in values))
        if values:
            mean = sum(values) / len(values)
            print(f"  mean = {mean:+.2f} c (rounded {round(mean, 1):+.1f})")
        print("  prompt:", card.prompt.text())
    finally:
        win.close()
    return 0


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    qapp = QApplication.instance() or QApplication([])
    mode = sys.argv[1] if len(sys.argv) > 1 else "chain"
    return {
        "chain": mode_chain,
        "seam": mode_seam,
        "change": mode_change,
        "hold": mode_hold,
        "old": mode_old,
    }[mode](qapp)


if __name__ == "__main__":
    raise SystemExit(main())
