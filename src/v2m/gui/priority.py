"""Background priority for the GUI's behind-the-scenes DSP threads.

The pYIN finalization of the previous take and the replay worker's
real-time DSP compete with the next capture's drain thread for CPU — the
capture is the singer's real-time product, so everything that only has
to finish EVENTUALLY (a finalization, a replay, the retention) runs at a
lower OS priority and the machine's scheduler puts the drain first
without any code change on the hot path.

Best-effort by design: a platform without a usable priority API (or a
policy that refuses the call) silently keeps the default priority —
never a crash. The function must be called FROM the thread being lowered
(the handle is the CALLER's thread; a cross-thread call would lower the
wrong one, and the test pins that contract by running inside the worker).
"""

from __future__ import annotations

import sys
import threading

_local = threading.local()


def lower_background_priority() -> bool:
    """Lower the CALLING thread's OS priority; True when it took effect.

    Called once per background worker thread at its start (thread-local:
    each new worker is lowered again). Windows uses the current thread's
    priority (THREAD_MODE_BACKGROUND_BEGIN falls back to the "lowest"
    step when refused); POSIX best-effort adjusts the nice value — Linux
    applies it per-thread, macOS per-process, which is still the right
    direction for a desktop box where the only foreground work IS the
    capture.
    """
    if getattr(_local, "done", False):
        return False  # one call per thread: re-lowering is refused
    try:
        if sys.platform == "win32":
            import ctypes

            GetCurrentThread = ctypes.windll.kernel32.GetCurrentThread
            GetCurrentThread.restype = ctypes.c_void_p
            handle = GetCurrentThread()  # pseudo-handle: valid IN THIS thread
            THREAD_MODE_BACKGROUND_BEGIN = 0x00010000
            ok = ctypes.windll.kernel32.SetThreadPriority(
                ctypes.c_void_p(handle), THREAD_MODE_BACKGROUND_BEGIN
            )
            if not ok:
                # Vista+: background mode can be refused (foreground window
                # rules) — degrade to the "lowest" step, still below the GUI
                ok = ctypes.windll.kernel32.SetThreadPriority(
                    ctypes.c_void_p(handle), -2  # THREAD_PRIORITY_LOWEST
                )
            _local.done = bool(ok)
            return _local.done
        import os

        if hasattr(os, "nice"):
            os.nice(10)  # POSIX: yield to every default-priority thread
            _local.done = True
            return True
    except Exception:
        pass  # best-effort: the status quo is an acceptable fallback
    return False
