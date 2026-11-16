"""A microphone that never opened is named the way the page names it.

`web/app.js` tells a refused permission, a missing device and a device another
application holds apart, and keeps a neutral sentence for everything else. The
desktop showed the driver's own line — "Error opening InputStream: Device
unavailable [-9985]" — which names the transport instead of what the singer
must DO. These pin the mapping, the neutral sentence that keeps the detail, and
the banner the reader actually gets; a failure that happens AFTER the stream
opened (a take too short to analyze) keeps its own wording, because the
microphone is not what failed.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from v2m.core import capture_failure
from v2m.core.i18n import tr


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class RefusingStream:
    """An opener that refuses the way PortAudio's do, and counts the attempts.

    `LiveCapture` retries once at the device's own rate before giving up, so the
    message is what a real refusal leaves behind on the second attempt too.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def __call__(self, **_options: object) -> object:
        self.calls += 1
        raise OSError(self.message)


# The wordings PortAudio and sounddevice actually raise (the code first: the
# prose around it changes between platforms and versions).
@pytest.mark.parametrize(
    ("message", "key"),
    [
        (
            "Error opening InputStream: Unanticipated host error "
            "[PaErrorCode -9999] [0x80070005]",
            capture_failure.BLOCKED,
        ),
        ("Error opening InputStream: Invalid device [PaErrorCode -9996]", capture_failure.MISSING),
        ("PortAudioError: No Default Input Device Available", capture_failure.MISSING),
        (
            "Error opening InputStream: Device unavailable [PaErrorCode -9985]",
            capture_failure.BUSY,
        ),
        # A rate the endpoint refuses is not one of the three causes, and
        # naming a busy microphone for it would send the reader to close an
        # application that is not holding anything.
        (
            "Error opening InputStream: Invalid sample rate [PaErrorCode -9997]",
            capture_failure.NEUTRAL,
        ),
    ],
)
def test_each_refusal_earns_the_pages_own_name(message, key):
    assert capture_failure.capture_failure_key(message) == key


def test_the_neutral_sentence_keeps_what_the_driver_said():
    # The three named causes say what to do and drop the transport text; the
    # neutral one is the only place a bug report still finds it on screen.
    named = capture_failure.capture_failure_message(
        "Error opening InputStream: Device unavailable [PaErrorCode -9985]",
        capture_failure.BUSY,
    )
    assert named == tr("ui mic busy")
    assert "-9985" not in named

    neutral = capture_failure.capture_failure_message("boom", capture_failure.NEUTRAL)
    assert neutral == tr("err ui mic").format(e="boom")

    # No key means the stream opened: the failure is the take's, and its own
    # wording must survive untouched.
    assert capture_failure.capture_failure_message("boom", "") == "boom"


def test_a_capture_that_never_opens_shows_the_pages_sentence(qapp):
    from v2m.gui.main_window import LiveThread, MainWindow

    win = MainWindow()
    stream = RefusingStream("Error opening InputStream: Device unavailable [PaErrorCode -9985]")
    thread = LiveThread(stream_factory=stream)
    win.live_thread = thread
    thread.failed.connect(lambda error, signal: win._live_thread_failed(error, signal))
    thread.run()  # inline: the refusal happens inside start()

    assert thread.failure_key == capture_failure.BUSY
    assert not win.error_banner.isHidden(), "the failure was never announced"
    # the banner draws its own leading mark; the sentence is what it says
    assert win.error_banner.text().endswith(tr("ui mic busy"))


def test_a_failure_after_the_stream_opened_keeps_its_own_text(qapp):
    # A take too short to analyze is the take's failure, not the microphone's:
    # the banner must not send the reader to check a device that worked.
    from v2m.gui.main_window import MainWindow

    win = MainWindow()
    win.live_thread = None  # no thread key: nothing says the microphone failed
    win._live_thread_failed("capture too short to analyze (0.412 s)")

    assert win.error_banner.text().endswith("capture too short to analyze (0.412 s)")
