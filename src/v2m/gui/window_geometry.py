"""How big the window opens on the screen it actually runs on.

A fixed opening size and a fixed floor are screen-blind, and the floor
makes the failure permanent: on a 1366x768 laptop at 150 % Windows scaling
Qt reports an available rectangle of 911x512 logical px, so a 1280x780
window opens 369 px wider and 268 px taller than the desktop (its title
bar and bottom edge land off-screen) and a 560 px floor leaves the user
unable to shrink it back on screen either.

This module is the ONE place that decides, for one screen's available
rectangle:

- the OPENING size: the app's comfortable default on the screens that fit
  it, shrinking to the usable area on a small one and growing up to a
  comfort ceiling on a large one;
- the FLOOR: the readability floor, clamped to the usable area, so a window
  can ALWAYS be made to fit the screen it is on (every stacked page
  scrolls, so nothing needs the floor to be tall);
- the POSITION: centered in the available rectangle rather than at the
  screen's top-left corner.

Qt reports LOGICAL pixels — Windows scaling and the device pixel ratio are
already applied — which is the unit this layout uses everywhere (fonts are
sized in pt), so no extra DPI arithmetic belongs here.

Deliberately pure numbers, no Qt import: the policy is pinned by
`tests/test_windowed_layout.py` without a QApplication, and `MainWindow` is
only the adapter that applies the returned `WindowPlan`.
"""

from __future__ import annotations

from typing import NamedTuple

# The comfort ceiling: a bigger screen stops growing the window here, and it
# doubles as the app's long-standing default size — the results page is laid
# out for it (its body sits side by side from BODY_SIDE_BY_SIDE_MIN_W 1180).
# Measured at 1600x1000: both results columns already run well past their
# floors (LEFT_COLUMN_MIN_W 520 / RIGHT_COLUMN_MIN_W 620) and the edit
# actions stay in ONE column, so a larger default buys no readability —
# it only starts to look like a maximized window and hides the desktop.
MAX_W = 1600
MAX_H = 1000

# The readability floor: below this the roll, the staff and the piano
# keyboard stop being readable. It is a CEILING on the minimum, never a
# promise: on a screen too small to hold it the floor follows the screen.
FLOOR_W = 900
FLOOR_H = 560

# Qt's geometry is the CLIENT area; the frame (title bar, borders) is drawn
# OUTSIDE it. A window sized to the raw available rectangle therefore loses
# its title bar off the top and its bottom edge off the bottom, so the
# usable area subtracts this frame allowance first. Measured on Windows at
# 150 % scaling: a frame reported as a few px still paints ~46 px of title
# bar, which the client-rectangle arithmetic does not see.
FRAME_W = 24
FRAME_H = 72

# The share of the usable area the opening size fills. It leaves the frame
# AND a visible strip of desktop on every side, so the window can still be
# grabbed and dragged — the reason this is not 1.0.
FILL = 0.85


class WindowPlan(NamedTuple):
    """The geometry one screen gives the window: size, floor and position."""

    width: int
    height: int
    min_width: int
    min_height: int
    x: int
    y: int


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def window_plan(available_w: int, available_h: int, x: int = 0, y: int = 0) -> WindowPlan:
    """The plan for ONE screen's available rectangle (logical px).

    `available_w`/`available_h` are what Qt's `availableGeometry()` reports
    (the desktop minus the panel); `x`/`y` are that rectangle's origin, so
    the returned position centers the window on the right monitor.
    """
    usable_w = max(1, int(available_w) - FRAME_W)
    usable_h = max(1, int(available_h) - FRAME_H)
    # the floor follows the screen when the screen is smaller than the floor
    min_w = min(FLOOR_W, usable_w)
    min_h = min(FLOOR_H, usable_h)
    width = _clamp(int(round(usable_w * FILL)), min_w, MAX_W)
    height = _clamp(int(round(usable_h * FILL)), min_h, MAX_H)
    return WindowPlan(
        width=width,
        height=height,
        min_width=min_w,
        min_height=min_h,
        x=int(x) + (int(available_w) - width) // 2,
        y=int(y) + (int(available_h) - height) // 2,
    )
