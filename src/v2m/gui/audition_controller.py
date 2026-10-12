"""The short auditions' lifecycle has ONE owner (the approved spec,
2026-10-06-playback-audition-ownership-design).

A short audition is any prepared clip the singer hears for a DECISION:
a result-note clip (Space), a captured-note clip, a finder/keyboard tone,
an original/replacement phrase clip, or a drag's old/moved comparison.
Today that lifecycle is spread over four loose `MainWindow` fields and
seven handlers; this module concentrates identity, toggle/replace, the
two-clip drag sequence and completion, so the window only prepares clips
and applies transition facts.

The controller is Qt-free and page-free on purpose: no timer, no widget,
no take or result reference crosses this seam — callers pass prepared
clip buffers and an opaque identity, and the injected playback adapter
(the production adapter is `core.playback.PlaybackSession`) stays the
only thing that touches audio. The window keeps its 33 ms observer and
reads `PlaybackSession.playing`; when the adapter reports idle, the
window calls `note_clip_completed()` and applies the returned facts.
"""

from __future__ import annotations

from dataclasses import dataclass

# The kinds that answer "the SAME audition again": only the two note
# identities toggle off on a repeat. Every other kind (finder, keyboard,
# phrases, drag) restarts — a repeated click must not read as a stop.
_TOGGLE_KINDS = frozenset({"note", "live_note"})


@dataclass(frozen=True)
class AuditionRequest:
    """One prepared audition: identity, mode and its ordered clips.

    A single audition carries ONE clip; a drag comparison carries TWO,
    ordered old note then moved note. The identity is an opaque key (the
    caller's take/note or pitch naming) — the controller compares it, it
    never dereferences a take or a result.
    """

    kind: str
    identity: object
    clips: tuple
    mode: str


@dataclass(frozen=True)
class AuditionTransition:
    """The facts the window applies after one controller operation.

    `previous` names the identity whose note-only readout must clear and
    `previous_kind` which PAGE readout that is; `active`/`active_kind`
    name the identity whose readout must show. The facts travel together
    so the readout ownership stays one-way: the pages render, the window
    applies, the controller decides.
    """

    previous: object | None
    previous_kind: str | None
    active: object | None
    active_kind: str | None


class AuditionController:
    """Owns the single active short audition and its sequence, Qt-free.

    The playback adapter's surface is exactly what the controller needs:
    `set_sources(piano=...)` to stage a clip, `play(mode) -> bool` to
    start it (False = failed to start), `stop()` to silence. Completion
    is OBSERVED by the window's timer (the adapter reports not-playing)
    and REPORTED here through `note_clip_completed()`; the controller
    advances a sequence exactly once per observed completion and clears
    its state on the final one.
    """

    def __init__(self, playback) -> None:
        self._playback = playback
        self._kind: str | None = None
        self._identity: object | None = None
        self._mode: str | None = None
        self._remaining: tuple = ()

    @property
    def active(self) -> bool:
        return self._identity is not None

    @property
    def active_identity(self):
        """The active audition's identity, or None — the fact tests and the
        window's readout logic read, never the internals."""
        return self._identity

    @property
    def active_kind(self) -> str | None:
        """The active audition's kind, or None — the window maps it to the
        one page readout that kind owns."""
        return self._kind

    def start(self, request: AuditionRequest) -> AuditionTransition:
        """Start, toggle off or replace — the one entry for every kind.

        The SAME result/captured note identity toggles the audition off
        (the Space behavior); any other kind or identity replaces the
        current one and reports it as `previous` so its readout clears.
        A failed adapter start leaves no identity.
        """
        if (
            request.kind in _TOGGLE_KINDS
            and self.active
            and self._kind == request.kind
            and self._identity == request.identity
        ):
            return self.cancel()
        previous, previous_kind = self._identity, self._kind
        self._kind = request.kind
        self._identity = request.identity
        self._mode = request.mode
        self._remaining = tuple(request.clips[1:])
        if not self._start_clip(request.clips[0], request.mode):
            # a failed start must leave NOTHING sounding: the adapter may
            # still hold a stream from the replaced audition, and no
            # identity remains to observe it
            self._playback.stop()
            return self._end_spent(previous, previous_kind)
        return AuditionTransition(
            previous=previous,
            previous_kind=previous_kind,
            active=self._identity,
            active_kind=self._kind,
        )
        return AuditionTransition(
            previous=previous,
            previous_kind=previous_kind,
            active=self._identity,
            active_kind=self._kind,
        )

    def note_clip_completed(self) -> AuditionTransition:
        """One observed clip completion: advance a sequence, or end.

        A two-clip drag starts its second clip HERE through the adapter —
        the caller never selects or starts a sequence clip. The final
        completion spends the state; a second clip that fails to start
        ends the audition and reports the identity so the readout clears.
        """
        if not self.active:
            return self._end(None, None)
        if self._remaining:
            next_clip = self._remaining[0]
            self._remaining = self._remaining[1:]
            if self._start_clip(next_clip, self._mode):
                return AuditionTransition(
                    previous=None,
                    previous_kind=None,
                    active=self._identity,
                    active_kind=self._kind,
                )
        return self._end_spent(self._identity, self._kind)

    def cancel(self) -> AuditionTransition:
        """Explicit stop/navigation/new take: silence, state cleared."""
        if not self.active:
            return self._end(None, None)
        previous, previous_kind = self._identity, self._kind
        self._playback.stop()
        return self._end_spent(previous, previous_kind)

    # ------------------------------------------------------------------
    # internals

    def _start_clip(self, clip, mode: str | None) -> bool:
        # the stage slot follows the mode: the session's `_selected` reads
        # `voice` for a voice-mode play, `piano` for everything else — a
        # note clip staged into the wrong slot would play silence
        if mode == "voice":
            self._playback.set_sources(voice=clip)
        else:
            self._playback.set_sources(piano=clip)
        return bool(self._playback.play(mode))

    def _end(self, previous, previous_kind) -> AuditionTransition:
        """The one inactive transition shape: the identity that JUST left
        (named so its readout clears) or (None, None) when nothing did."""
        return AuditionTransition(
            previous=previous,
            previous_kind=previous_kind,
            active=None,
            active_kind=None,
        )

    def _end_spent(self, previous, previous_kind) -> AuditionTransition:
        """End an ACTIVE audition: clear the identity, then report the leave."""
        self._clear()
        return self._end(previous, previous_kind)

    def _clear(self) -> None:
        self._kind = None
        self._identity = None
        self._mode = None
        self._remaining = ()
