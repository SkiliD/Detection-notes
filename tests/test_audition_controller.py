"""The short-audition lifecycle has ONE owner (the approved spec):
`gui/audition_controller.AuditionController`, tested here through a fake
playback adapter — no Qt, no audio, no clock.

The pins are the spec's semantics: same note identity toggles off, any
other audition replaces, a drag plays exactly two clips ordered old then
moved, a failed start leaves no identity, and every transition tells the
window which readout to clear and whether an audition is still active.
"""

from __future__ import annotations

from v2m.gui.audition_controller import (
    AuditionController,
    AuditionRequest,
)

PIANO = "piano"


class FakePlayback:
    """The PlaybackSession surface the controller touches — nothing else."""

    def __init__(self) -> None:
        self.piano = None
        self.voice = None
        self.play_calls: list[str] = []
        self.fail = False
        self.stopped = 0

    def set_sources(self, piano=None, voice=None) -> None:
        self.piano = piano
        self.voice = voice

    def play(self, mode: str = PIANO, t: float = 0.0) -> bool:
        if self.fail:
            return False
        self.play_calls.append(mode)
        return True

    def stop(self) -> None:
        self.stopped += 1


def _note(index: int = 3) -> AuditionRequest:
    return AuditionRequest(kind="note", identity=("result", index), clips=(_CLIP_A,), mode=PIANO)


def _live_note(index: int = 2) -> AuditionRequest:
    return AuditionRequest(
        kind="live_note", identity=("captured", index), clips=(_CLIP_A,), mode=PIANO
    )


def _drag() -> AuditionRequest:
    return AuditionRequest(
        kind="drag", identity=("result", 5), clips=(_CLIP_A, _CLIP_B), mode=PIANO
    )


_CLIP_A = object()
_CLIP_B = object()


def test_a_single_start_sets_sources_then_plays_once():
    playback = FakePlayback()
    controller = AuditionController(playback)

    transition = controller.start(_note())

    assert playback.piano is _CLIP_A
    assert playback.play_calls == [PIANO]
    assert controller.active
    assert (transition.previous, transition.active, transition.active_kind) == (
        None,
        ("result", 3),
        "note",
    )


def test_a_note_clip_stages_into_the_voice_slot():
    """The session's `_selected` reads `voice` for a voice-mode play: the
    note clips are staged THERE (a piano-slot staging would play silence)."""
    playback = FakePlayback()
    controller = AuditionController(playback)

    controller.start(
        AuditionRequest(kind="note", identity=("result", 1), clips=(_CLIP_A,), mode="voice")
    )

    assert playback.voice is _CLIP_A
    assert playback.play_calls == ["voice"]


def test_the_same_note_identity_toggles_off_without_replaying():
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_note(index=3))

    transition = controller.start(_note(index=3))

    assert playback.play_calls == [PIANO]  # the first start only
    assert playback.stopped == 1
    assert not controller.active
    assert (
        transition.previous,
        transition.previous_kind,
    ) == (("result", 3), "note")


def test_the_same_captured_identity_toggles_off_too():
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_live_note(index=2))

    transition = controller.start(_live_note(index=2))

    assert playback.play_calls == [PIANO]
    assert not controller.active
    assert (transition.previous, transition.active) == (("captured", 2), None)


def test_a_different_note_identity_replaces_and_reports_the_old_one():
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_note(index=3))

    transition = controller.start(_note(index=7))

    assert playback.play_calls == [PIANO, PIANO]
    assert controller.active
    assert (transition.previous, transition.active) == (("result", 3), ("result", 7))


def test_every_other_kind_replaces_rather_than_toggles():
    playback = FakePlayback()
    controller = AuditionController(playback)
    kinds = [
        AuditionRequest(kind="finder", identity=64, clips=(_CLIP_A,), mode=PIANO),
        AuditionRequest(kind="key", identity=65, clips=(_CLIP_A,), mode=PIANO),
        AuditionRequest(kind="phrase_original", identity=1, clips=(_CLIP_A,), mode=PIANO),
        AuditionRequest(kind="phrase_replacement", identity=1, clips=(_CLIP_A,), mode=PIANO),
    ]
    for request in kinds:
        before = controller.active
        transition = controller.start(request)
        # a repeated click on the SAME finder/key identity still restarts
        assert transition.active == request.identity
        assert before or transition.previous is None


def test_a_drag_starts_only_its_first_clip():
    playback = FakePlayback()
    controller = AuditionController(playback)

    transition = controller.start(_drag())

    assert playback.piano is _CLIP_A
    assert playback.play_calls == [PIANO]
    assert controller.active
    assert transition.active == ("result", 5)  # identity only, no readout


def test_one_completion_starts_the_second_clip_through_the_adapter():
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_drag())

    transition = controller.note_clip_completed()

    assert playback.piano is _CLIP_B
    assert playback.play_calls == [PIANO, PIANO]
    assert controller.active
    assert transition.active == ("result", 5)


def test_the_second_completion_spends_the_state():
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_drag())
    controller.note_clip_completed()

    transition = controller.note_clip_completed()

    assert playback.play_calls == [PIANO, PIANO]
    assert not controller.active
    assert (transition.previous, transition.active) == (("result", 5), None)


def test_a_single_clip_completion_ends_the_audition():
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_note())

    transition = controller.note_clip_completed()

    assert not controller.active
    assert (transition.previous, transition.active) == (("result", 3), None)


def test_a_failed_second_clip_leaves_no_identity():
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_drag())
    controller.note_clip_completed()
    playback.fail = True

    transition = controller.note_clip_completed()

    assert not controller.active
    assert (transition.previous, transition.active) == (("result", 5), None)


def test_a_failed_start_clears_the_old_identity_and_starts_nothing():
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_note(index=3))
    playback.fail = True

    transition = controller.start(_note(index=7))

    assert playback.play_calls == [PIANO]  # the failed start played nothing
    assert not controller.active
    assert (transition.previous, transition.active) == (("result", 3), None)


def test_a_failed_start_silences_the_replaced_audition():
    """A failed `play` leaves NOTHING sounding: the replaced audition's
    clip would otherwise keep playing with no identity observing it.
    """
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_note(index=3))
    playback.fail = True

    transition = controller.start(_note(index=7))

    # staging happened first (the failed clip), then the contract's stop
    assert playback.piano is _CLIP_A
    assert playback.stopped == 1
    assert (transition.previous, transition.active) == (("result", 3), None)


def test_cancel_stops_the_adapter_and_clears_the_state():
    playback = FakePlayback()
    controller = AuditionController(playback)
    controller.start(_note())

    transition = controller.cancel()

    assert playback.stopped == 1
    assert not controller.active
    assert (transition.previous, transition.active) == (("result", 3), None)


def test_cancel_and_completion_on_an_idle_controller_are_no_ops():
    playback = FakePlayback()
    controller = AuditionController(playback)

    cancelled = controller.cancel()
    completed = controller.note_clip_completed()

    assert playback.stopped == 0
    assert not controller.active
    assert (cancelled.previous, cancelled.active) == (None, None)
    assert (completed.previous, completed.active) == (None, None)
