"""The page's own names for a capture that never opened.

`web/app.js` tells a refused permission, a missing device and a device another
application holds apart, and keeps a neutral sentence for everything else. The
desktop used to show `str(error)` — "Error opening InputStream: Device
unavailable [-9985]" — which names the transport instead of the thing the
singer must DO. PortAudio's message is the only evidence available before a
stream opens, so this maps the signatures it can PROVE and never guesses: an
unrecognised failure keeps the neutral sentence, detail included.

The keys are the translation keys themselves: the caller shows `tr(key)`, so a
new name cannot exist on one side of the language tables only.
"""

from __future__ import annotations

from .i18n import tr

BLOCKED = "ui mic blocked"
MISSING = "ui mic missing"
BUSY = "ui mic busy"
NEUTRAL = "err ui mic"

# PortAudio's own codes travel inside the message sounddevice raises.
_INVALID_DEVICE = "-9996"  # no such input device: unplugged, stale index
_DEVICE_UNAVAILABLE = "-9985"  # the endpoint is known and cannot be opened now
# Windows reports a refused microphone as E_ACCESSDENIED, wrapped by PortAudio
# in a generic -9999 "Unanticipated host error".
_WINDOWS_DENIED = "0x80070005"


def capture_failure_key(error: BaseException | str) -> str:
    """Which sentence this capture failure earns.

    Only the three causes a reader can act on are named. The codes come first
    because PortAudio's wording around them changes between platforms and
    versions, while the codes do not — and a sentence that sends the singer to
    close an application when the device was simply unplugged is worse than the
    neutral one, so an unknown shape stays neutral.
    """
    text = str(error).lower()
    if _WINDOWS_DENIED in text or "access is denied" in text or "permission denied" in text:
        return BLOCKED
    if _INVALID_DEVICE in text or "invalid device" in text or "no default input device" in text:
        return MISSING
    if _DEVICE_UNAVAILABLE in text or "device unavailable" in text:
        return BUSY
    return NEUTRAL


def capture_failure_message(error: BaseException | str, key: str) -> str:
    """The sentence to show for a capture failure, in the running language.

    `key` is the one a capture adapter worked out when the stream refused to
    open (`LiveThread.failure_key`), and "" when the failure came AFTER it did:
    a take too short to analyze keeps its own wording, because prefixing it with
    "microphone unavailable" would send the reader to check a device that
    worked. Only the neutral sentence carries the transport text — the three
    named causes already say what to do with it.
    """
    if not key:
        return str(error)
    if key == NEUTRAL:
        return tr(NEUTRAL).format(e=error)
    return tr(key)
