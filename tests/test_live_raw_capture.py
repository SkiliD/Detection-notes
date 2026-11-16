"""Raw (exclusive) capture: asked for, refused safely, never assumed.

Shared mode runs the endpoint's audio effects chain before the app sees the
samples. WASAPI's exclusive mode is the way past it, but a device refuses for
ordinary reasons (it is busy, the backend has no exclusive mode, the driver
will not take the format). A refused raw open must cost the MODE, never the
take — and the take must report which one it got.
"""

from __future__ import annotations

import types

import pytest

from v2m.core import audio_io


class FakeStream:
    """A stand-in `sounddevice.InputStream` that records its options."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)
        self.samplerate = kwargs.get("samplerate")
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class RefusingClock(FakeStream):
    """A stream that opens and then refuses to start (a refused raw clock)."""

    def start(self) -> None:
        raise RuntimeError("clock refused")


class Factory:
    """Every open attempt, and how the RAW one fails (`None` = it works)."""

    def __init__(self, refuse: str | None = None) -> None:
        self.refuse = refuse
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> FakeStream:
        self.calls.append(dict(kwargs))
        raw = "extra_settings" in kwargs
        if raw and self.refuse == "open":
            raise RuntimeError("exclusive mode refused")
        stream_type = RefusingClock if raw and self.refuse == "start" else FakeStream
        return stream_type(**kwargs)


def _sounddevice(**extra: object) -> types.SimpleNamespace:
    api: dict[str, object] = {
        "InputStream": FakeStream,
        "query_devices": lambda index: {"default_samplerate": 48_000.0},
    }
    api.update(extra)
    return types.SimpleNamespace(**api)


def _wasapi_settings(**kwargs: object) -> tuple[str, dict[str, object]]:
    return ("wasapi", kwargs)


def test_raw_is_asked_for_at_the_endpoint_own_rate(monkeypatch):
    factory = Factory()
    monkeypatch.setattr(audio_io, "sd", _sounddevice(WasapiSettings=_wasapi_settings))
    cap = audio_io.LiveCapture(samplerate=44_100, device=3, exclusive=True, stream_factory=factory)
    cap.start()
    try:
        assert factory.calls[0]["extra_settings"] == ("wasapi", {"exclusive": True})
        # exclusive takes only the DEVICE's format, never the shared-mode one
        assert factory.calls[0]["samplerate"] == 48_000
        assert cap.samplerate == 48_000  # the take's clock is what OPENED
        assert cap.exclusive_opened is True
    finally:
        cap.stop()


def test_a_refused_raw_open_still_captures_in_shared_mode(monkeypatch):
    factory = Factory(refuse="open")
    monkeypatch.setattr(audio_io, "sd", _sounddevice(WasapiSettings=_wasapi_settings))
    cap = audio_io.LiveCapture(samplerate=44_100, device=3, exclusive=True, stream_factory=factory)
    cap.start()
    try:
        assert cap.exclusive_opened is False
        assert len(factory.calls) == 2  # the raw attempt, then the shared one
        assert "extra_settings" not in factory.calls[1]
        assert factory.calls[1]["samplerate"] == 44_100
        assert cap.samplerate == 44_100
        assert cap._stream.started
    finally:
        cap.stop()


def test_a_raw_clock_that_refuses_reopens_shared(monkeypatch):
    factory = Factory(refuse="start")
    monkeypatch.setattr(audio_io, "sd", _sounddevice(WasapiSettings=_wasapi_settings))
    cap = audio_io.LiveCapture(samplerate=44_100, device=3, exclusive=True, stream_factory=factory)
    cap.start()
    try:
        assert cap.exclusive_opened is False
        assert len(factory.calls) == 2
        assert cap._stream.started  # the SECOND stream is the one running
    finally:
        cap.stop()


def test_a_shared_open_refused_at_the_requested_rate_retries_the_device_rate(monkeypatch):
    """Measured: a virtual endpoint refuses 44.1 kHz in shared mode too.

    A refused raw open must cost the MODE; ending at a shared endpoint that
    also refuses the requested rate must cost neither, or the fallback itself
    loses the take.
    """

    class RefusingRate(Factory):
        def __call__(self, **kwargs: object) -> FakeStream:
            if kwargs.get("samplerate") == 44_100:
                self.calls.append(dict(kwargs))
                raise RuntimeError("invalid sample rate")
            return super().__call__(**kwargs)

    factory = RefusingRate()
    monkeypatch.setattr(audio_io, "sd", _sounddevice(WasapiSettings=_wasapi_settings))
    cap = audio_io.LiveCapture(samplerate=44_100, device=3, stream_factory=factory)
    cap.start()
    try:
        assert [call["samplerate"] for call in factory.calls] == [44_100, 48_000]
        assert cap.samplerate == 48_000  # the endpoint's own rate is the clock
        assert cap.exclusive_opened is False
        assert cap._stream.started
    finally:
        cap.stop()


def test_a_device_that_refuses_every_rate_still_fails_the_take(monkeypatch):
    """The rate retry is not a blanket swallow: an unusable device must say so."""

    class Dead(Factory):
        def __call__(self, **kwargs: object) -> FakeStream:
            self.calls.append(dict(kwargs))
            raise RuntimeError("no such device")

    factory = Dead()
    monkeypatch.setattr(audio_io, "sd", _sounddevice(WasapiSettings=_wasapi_settings))
    cap = audio_io.LiveCapture(samplerate=44_100, device=3, stream_factory=factory)
    with pytest.raises(RuntimeError):
        cap.start()
    assert len(factory.calls) == 2  # the requested rate, then the device's


def test_a_shared_capture_never_asks_for_raw_settings(monkeypatch):
    factory = Factory()
    monkeypatch.setattr(audio_io, "sd", _sounddevice(WasapiSettings=_wasapi_settings))
    cap = audio_io.LiveCapture(samplerate=44_100, device=3, stream_factory=factory)
    cap.start()
    try:
        assert all("extra_settings" not in call for call in factory.calls)
        assert cap.exclusive_opened is False
    finally:
        cap.stop()
