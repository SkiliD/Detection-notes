"""The localhost companion: the desktop's offline half, offered to the web page.

`web/` runs the live half in the browser and cannot run pYIN (librosa and numba
are desktop-only), so the page has no tempo, no meter, no key, no notation. This
module is the OPT-IN bridge: `v2m serve` listens on loopback, and a page that
the user has allowed posts a file to `/analyze` when the companion is running.

Why it looks the way it does (ADR 0042, and the browser study it records):

- loopback only, never `0.0.0.0` — the socket exists for this machine's page,
  and a listening port on a LAN is a different product;
- an Origin allowlist, because ANY page in the browser can POST to a local
  port (CSRF, DNS rebinding): the browser's own Local Network Access permission
  gates the CALL, never the server;
- codes in the error body, never sentences — the desktop translates through
  `~/.v2m_lang` and the page has its own two languages, so a translated
  message would arrive in whichever language the other side is not using;
- no artifacts: the answer is `report.report_dict`, the export's own schema
  built without touching the disk (a route that writes files is a route that
  can fill a disk from a browser tab);
- one analysis at a time (429 while busy): the DSP is CPU-bound in one thread,
  and three concurrent pYIN runs on a laptop is not a service.

The routes:

- `/analyze` — audio bytes in, the export's own report payload out;
- `/render` — the page's CORRECTED note list in, the MusicXML the desktop would
  engrave out (`musicxml.build_score_xml`, the string half of the export's
  writer). The page can retune, move and delete notes but cannot lay out a
  staff; this is the route that turns its corrections into a score, and it
  deliberately does not queue behind a running analysis (layout is
  milliseconds, pYIN is not).
- `/compare` — an exported `notes.json` and the take it was read from in, the
  difference with a FRESH desktop reading out. The page's live chain (YIN)
  and the desktop's (pYIN) are two readings of one take; `reading_diff`
  already pairs two such readings for the window, and this route is the same
  comparison over the wire, so a singer can re-read a take they exported
  without re-recording it. The audio travels as base64 inside the JSON: one
  body, no multipart, inside the same cap an upload already has.
"""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from .. import __version__
from .report import IncompleteAnalysis

if TYPE_CHECKING:  # pragma: no cover - the helpers import their types lazily
    from .grid import Meter
    from .keys import KeyResult
    from .pipeline import AnalysisResult

APP_ID = "v2m"
API = 1
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# A take of ten minutes at 44.1 kHz is ~50 MB: the cap refuses a body that is
# already absurd for a voice memo, before a byte of it reaches the analyser.
MAX_BODY_BYTES = 60 * 1024 * 1024
FEATURES = ("analyze", "render", "compare")
# A developer's own static server serves the page from one of these; the
# published page needs its origin named with `--allow-origin`.
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
# Per route: `/analyze` takes bytes, `/render` and `/compare` take the page's JSON
# (a comparison carries its audio INSIDE that JSON, as base64).
MEDIA_TYPES = {
    "/analyze": ("audio/", "application/octet-stream"),
    "/render": ("application/json",),
    "/compare": ("application/json",),  # the audio travels base64 inside the JSON
}
# Held for the whole analysis, taken WITHOUT blocking: a second request is
# answered "busy" instead of queueing behind a take it cannot see.
ANALYSIS_LOCK = threading.Lock()


def origin_allowed(origin: str | None, allowed: tuple[str, ...]) -> bool:
    """Whether a page origin may read the answer.

    No `Origin` header at all is a non-browser client (the contract tests, a
    `curl` from the user's shell) and is let through: the allowlist exists to
    stop one PAGE from driving the port, and such a client is not a page.
    """
    if origin is None:
        return True
    if origin in allowed:
        return True
    parsed = urlsplit(origin)
    return parsed.scheme in ("http", "https") and (parsed.hostname or "") in LOOPBACK_HOSTS


def health() -> dict:
    """What a page asks before it trusts the rest: identity, version, routes."""
    return {
        "app": APP_ID,
        "api": API,
        "version": __version__,
        "features": list(FEATURES),
    }


def decode_upload(data: bytes):
    """The posted bytes as the analysis signal (mono, the analysis rate)."""
    import librosa

    from .audio_io import SR

    try:
        signal, _rate = librosa.load(io.BytesIO(data), sr=SR, mono=True)
    except Exception as error:  # noqa: BLE001 - every decoder failure is the payload's
        # soundfile raises `LibsndfileError` (a RuntimeError), librosa raises
        # ParametersError: one code for all of them, because the caller can act
        # on none of the variants differently
        raise ValueError("undecodable upload") from error
    return signal


class BadRender(ValueError):
    """The page's render request is unusable; `code` names the field to fix."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _meter_from(label) -> "Meter":
    """`"4/4"` as the grid's own Meter (4/4 when the page sends none)."""
    from .grid import DEFAULT_METER, Meter

    if label is None:
        return DEFAULT_METER
    if not isinstance(label, str) or "/" not in label:
        raise BadRender("meter")
    numerator, _, denominator = label.partition("/")
    try:
        meter = Meter(int(numerator), int(denominator))
    except (TypeError, ValueError):
        raise BadRender("meter") from None
    if meter.beats_per_measure <= 0 or meter.beat_unit not in (2, 4, 8, 16):
        raise BadRender("meter")
    return meter


def _key_from(payload_key) -> "KeyResult | None":
    """The report's `{"best": "C major"}` — or the bare label — as a KeyResult.

    The label is parsed rather than trusted: `KeyCandidate.label` is English on
    both sides (`"C major"`, never the localized `tr_key_label`), so the page can
    hand back what `/analyze` gave it. A label the desktop cannot spell is a
    400, never a silent C major.
    """
    from .keys import KeyCandidate, KeyResult
    from .music import NOTE_NAMES

    if payload_key is None:
        return None
    label = payload_key.get("best") if isinstance(payload_key, dict) else payload_key
    if not isinstance(label, str):
        raise BadRender("key")
    tonic, _, mode = label.partition(" ")
    if tonic not in NOTE_NAMES or mode not in ("major", "minor"):
        raise BadRender("key")
    return KeyResult(best=KeyCandidate(tonic, mode, 1.0), alternatives=[])


def _note_objects(items, fail) -> list:
    """The page's note rows as the core's own Note objects.

    `fail` is a FACTORY naming the caller's field: `/render` refuses a bad
    row as `notes`, `/compare` refuses an exported document as `exported`.
    One validator, two codes — two routes never drift apart on what a note
    is, which is the kind of divergence a caller discovers in production.
    """
    from .notes import Note

    if not isinstance(items, list) or not items:
        raise fail()
    notes = []
    for item in items:
        if not isinstance(item, dict):
            raise fail()
        try:
            start = float(item["start_s"])
            end = float(item["end_s"])
            midi = int(item["midi"])
            confidence = float(item.get("confidence", 1.0))
            strength = float(item.get("strength", 0.0))
        except (KeyError, TypeError, ValueError):
            raise fail() from None
        if midi < 0 or midi > 127 or start < 0.0 or end <= start:
            raise fail()
        notes.append(
            Note(start=start, end=end, midi=midi, confidence=confidence, strength=strength)
        )
    return notes


def _notes_from(items):
    """The page's corrected rows, for the notation writer."""
    return _note_objects(items, lambda: BadRender("notes"))


def render_upload(payload: dict) -> dict:
    """The page's corrected notes, as the MusicXML the desktop would write.

    A corrected take lives only in the browser: the page can retune, move and
    delete notes, but it cannot lay a staff out (measures, figures, key
    signature, note colours) — and that layout IS the desktop's job.
    `build_score_xml` is the string-returning half of the export's own writer,
    so the answer is the very score `v2m analyze` engraves for those notes, not
    a second notation that could disagree with it. Nothing touches the disk: the
    caller receives the text and decides what to do with it.

    The request mirrors what `/analyze` answered (`tempo`, `meter`, `key`), so a
    page that only edited `notes` can post its report back with the corrections:
    anything else in the payload is ignored, never echoed.
    """
    from .grid import BeatGrid
    from .musicxml import build_score_xml

    if not isinstance(payload, dict):
        raise BadRender("body")
    tempo = payload.get("tempo")
    bpm = tempo.get("bpm") if isinstance(tempo, dict) else tempo
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not bpm > 0:
        raise BadRender("tempo")
    notes = _notes_from(payload.get("notes"))
    meter = _meter_from(payload.get("meter"))
    key = _key_from(payload.get("key"))
    try:
        grid = BeatGrid(float(bpm), meter)
    except ValueError:
        # the grid's own sane_bpm guard refused the tempo: the page's field,
        # not a server fault
        raise BadRender("tempo") from None
    return {
        "api": API,
        "musicxml": build_score_xml(notes, grid, key, title="Transcription"),
        "n_notes": len(notes),
        "meter": meter.label,
        "key": key.best.label if key else None,
    }


def analyze_upload(data: bytes) -> dict:
    """The posted audio as the report the page can use — and nothing on disk."""
    from .audio_io import SR
    from .pipeline import analyze_signal
    from .report import report_dict

    signal = decode_upload(data)
    if not len(signal):
        raise ValueError("empty upload")
    return report_dict(analyze_signal(signal, SR, source="upload"))


class BadCompare(ValueError):
    """The comparison request is unusable; `code` names the field to fix."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _exported_reading(document) -> "AnalysisResult":
    """The page's exported `notes.json`, as a reading this desktop can compare.

    A REAL `AnalysisResult` with no tempo, no key and no observation: those are
    what an exported note list does not carry, and inventing them would let the
    comparison claim evidence it never had (a suspect list needs the pitch
    track, a grid needs the tempo). `reading_diff.compare_readings` reads
    exactly these fields through `getattr`, so an absent one is "unavailable",
    never "clean".
    """
    from .pipeline import AnalysisResult

    if not isinstance(document, dict):
        raise BadCompare("exported")
    notes = _note_objects(document.get("notes"), lambda: BadCompare("exported"))
    return AnalysisResult(
        source="exported",
        duration_s=max(note.end for note in notes),
        notes=notes,
        tempo=None,
        key=None,
    )


def compare_export(payload: dict) -> dict:
    """An exported take re-read by the desktop, and the difference explained.

    The page reads the live half (YIN, in the browser) and exports what it
    measured; this route reads the offline half (pYIN) of the SAME audio and
    compares the two. `reading_diff.compare_readings` is the ONE pairing policy
    for two readings of one take (one beat of association, the visible window
    for a change), so this route adds no second notion of "the same note" — a
    stable pair means the two engines agree, and a changed one names its reasons.

    Only presentation data travels back: spans, status codes and reason CODES
    (`pitch`, `attack`, `duration`) — never a sentence, because the page
    translates in its own language and the desktop in `~/.v2m_lang`.
    """
    import base64
    import binascii

    from .audio_io import SR
    from .pipeline import analyze_signal
    from .reading_diff import (
        ADDED,
        CHANGED,
        MISSING,
        NEAR,
        STABLE,
        compare_readings,
    )

    if not isinstance(payload, dict):
        raise BadCompare("body")
    exported = _exported_reading(payload.get("exported"))
    encoded = payload.get("audio_base64")
    if not isinstance(encoded, str) or not encoded:
        raise BadCompare("audio")
    try:
        audio = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise BadCompare("audio") from None
    if not audio:
        raise BadCompare("audio")
    try:
        signal = decode_upload(audio)
    except ValueError:
        raise BadCompare("audio") from None
    if not len(signal):
        raise BadCompare("audio")
    # `analyze_signal` alone, not `report_dict`: a take with no tempo or no key
    # still has notes to compare, and 422 would answer "nothing to say" for a
    # request that was perfectly answerable.
    reading = analyze_signal(signal, SR, source="upload")
    comparison = compare_readings(exported, reading)

    def span(note) -> dict:
        return {
            "start_s": round(float(note.start), 4),
            "end_s": round(float(note.end), 4),
            "midi": int(note.midi),
        }

    return {
        "api": API,
        "exported": {
            "n_notes": len(exported.notes),
            "source": payload["exported"].get("source"),
            "generated_at": payload["exported"].get("generated_at"),
        },
        "reading": {
            "n_notes": len(reading.notes),
            "duration_s": round(float(reading.duration_s), 3),
            "bpm": getattr(reading.tempo, "bpm", None),
            "key": reading.key.best.label if reading.key else None,
            "meter": reading.meter.label if reading.meter else None,
        },
        "counts": {
            STABLE: comparison.count(STABLE),
            NEAR: comparison.count(NEAR),
            CHANGED: comparison.count(CHANGED),
            MISSING: len(comparison.missing),
            ADDED: len(comparison.added),
        },
        "pairs": [
            {
                **span(pair.before),
                "after_start_s": round(pair.after.start, 4),
                "after_end_s": round(pair.after.end, 4),
                "after_midi": pair.after.midi,
                "status": pair.status,
                "reasons": list(pair.reasons),
                "pitch_delta": pair.pitch_delta,
                "onset_delta_s": round(pair.onset_delta_s, 4),
                "end_delta_s": round(pair.end_delta_s, 4),
            }
            for pair in comparison.pairs
        ],
        "missing": [span(note) for note in comparison.missing],
        "added": [span(note) for note in comparison.added],
    }


class CompanionServer(ThreadingHTTPServer):
    """The loopback server, carrying its own allowlist."""

    daemon_threads = True

    def __init__(self, address, allowed_origins: tuple[str, ...] = ()):
        super().__init__(address, CompanionHandler)
        self.allowed_origins = allowed_origins


class CompanionHandler(BaseHTTPRequestHandler):
    """`GET /health`, the three POST routes, and a preflight that answers honestly."""

    server_version = f"v2m/{__version__}"
    sys_version = ""  # no interpreter banner: the page has no use for it
    protocol_version = "HTTP/1.1"
    server: CompanionServer

    # --- plumbing ---------------------------------------------------------

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - the stdlib signature
        """Silence: a page polling `/health` must not fill the user's console."""

    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if origin and origin_allowed(origin, self.server.allowed_origins):
            self.send_header("Access-Control-Allow-Origin", origin)
            # The allowlist depends on the header: a cache must not reuse an
            # answer granted to one origin for another.
            self.send_header("Vary", "Origin")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _origin_refused(self) -> bool:
        """403 for a page the user never allowed; the body names the code."""
        origin = self.headers.get("Origin")
        if origin_allowed(origin, self.server.allowed_origins):
            return False
        self._json(403, {"error": "origin"})
        return True

    # --- routes -----------------------------------------------------------

    def do_OPTIONS(self) -> None:
        # Only a preflighted origin reaches here; answering 204 keeps the
        # browser from guessing which methods exist.
        if self._origin_refused():
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self._origin_refused():
            return
        if self.path.split("?")[0] == "/health":
            self._json(200, health())
            return
        if self.path.split("?")[0] in MEDIA_TYPES:
            self._json(405, {"error": "method"})
            return
        self._json(404, {"error": "path"})

    def do_POST(self) -> None:
        if self._origin_refused():
            return
        path = self.path.split("?")[0]
        if path == "/health":
            self._json(405, {"error": "method"})
            return
        if path not in MEDIA_TYPES:
            self._json(404, {"error": "path"})
            return
        media = (self.headers.get("Content-Type") or "").lower()
        if not any(media.startswith(kind) for kind in MEDIA_TYPES[path]):
            self._json(415, {"error": "media-type"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._json(400, {"error": "length"})
            return
        if length > MAX_BODY_BYTES:
            # Refused before reading: the body is never buffered, and the
            # connection is closed so an unread stream cannot poison the next
            # request on this socket.
            self.close_connection = True
            self._json(413, {"error": "too-large"})
            return
        data = self.rfile.read(length)
        if path == "/render":
            self._render(data)
            return
        if path == "/compare":
            self._compare(data)
            return
        self._analyze(data)

    def _render(self, data: bytes) -> None:
        """Layout is milliseconds, so it does NOT wait behind a pYIN run: only
        the analysis owns the lock, and a page correcting notes stays
        responsive while a long take is still being analysed."""
        try:
            answer = render_upload(json.loads(data))
        except json.JSONDecodeError:
            self._json(400, {"error": "body"})
            return
        except BadRender as error:
            self._json(400, {"error": error.code})
            return
        except Exception:  # noqa: BLE001 - one bad request must not kill the server
            self._json(500, {"error": "internal"})
            return
        self._json(200, answer)

    def _compare(self, data: bytes) -> None:
        """A comparison runs pYIN, so it waits for (or is refused by) the lock.

        Unlike `/render`, this IS analysis work: queuing a second re-reading
        behind a take the caller cannot see is the situation 429 exists for.
        """
        if not ANALYSIS_LOCK.acquire(blocking=False):
            self._json(429, {"error": "busy"})
            return
        try:
            answer = compare_export(json.loads(data))
        except json.JSONDecodeError:
            self._json(400, {"error": "body"})
            return
        except BadCompare as error:
            self._json(400, {"error": error.code})
            return
        except Exception:  # noqa: BLE001 - one bad request must not kill the server
            self._json(500, {"error": "internal"})
            return
        finally:
            ANALYSIS_LOCK.release()
        self._json(200, answer)

    def _analyze(self, data: bytes) -> None:
        if not ANALYSIS_LOCK.acquire(blocking=False):
            self._json(429, {"error": "busy"})
            return
        try:
            report = analyze_upload(data)
        except IncompleteAnalysis:
            # the bytes decoded and the analysis ran, but there is no tempo or
            # no key to report: a payload the schema cannot describe (422, not
            # the 400 a decoder would answer)
            self._json(422, {"error": "incomplete"})
            return
        except ValueError:
            self._json(400, {"error": "audio"})
            return
        except Exception:  # noqa: BLE001 - one bad take must not kill the server
            self._json(500, {"error": "internal"})
            return
        finally:
            ANALYSIS_LOCK.release()
        self._json(200, {"api": API, "report": report})


def make_server(port: int = DEFAULT_PORT, allowed_origins: tuple[str, ...] = ()) -> CompanionServer:
    """A loopback server; port 0 asks the OS for a free one (the tests' way)."""
    return CompanionServer((HOST, port), allowed_origins)
