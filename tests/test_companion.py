"""Contract tests for the loopback companion (`v2m serve`).

The contract is what the page codes against, so it is exercised over a REAL
loopback socket on port 0 rather than by calling the handler's methods: the
status codes, the CORS headers and the preflight are part of the promise, and a
unit call would never prove them.

The audio is synthesised here (four notes at 120 BPM) so the suite needs no
`demo.wav` and no fixture file: the point is the route, not the analysis.

Refusals are pinned too, because they are the security decision of ADR 0042: a
foreign page origin, a body over the cap, a second concurrent analysis, an
undecodable body — each one has a code, never a sentence.

`/render` gets the same treatment from the other side: the page's CORRECTED
notes in, the MusicXML the desktop would engrave out. The tests read the score
back (steps, octaves, key signature, beats) rather than trusting a string, so a
route that echoed the take's original notes, or that engraved a second notation
of its own, fails.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import threading
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from v2m import __version__, cli
from v2m.core import companion

PAGE_ORIGIN = "https://page.example"


def _phrase_wav() -> bytes:
    """Four sung-ish notes, 120 BPM, 2 s: enough for a tempo and a key."""
    rate = 22_050
    buffer = io.BytesIO()
    signal = np.zeros(int(rate * 2.0), dtype=np.float32)
    for index, midi in enumerate((60, 64, 67, 72)):
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        start = int(index * 0.5 * rate)
        length = int(0.45 * rate)
        t = np.arange(length) / rate
        tail = (length - np.arange(length)) / (0.03 * rate)
        ramp = np.minimum(1.0, t / 0.02) * np.minimum(1.0, tail)
        burst = 0.3 * ramp * (np.sin(2 * np.pi * hz * t) + 0.4 * np.sin(4 * np.pi * hz * t))
        signal[start:start + length] += burst.astype(np.float32)
    sf.write(buffer, signal, rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def wav_bytes() -> bytes:
    return _phrase_wav()


@pytest.fixture()
def server():
    """A real server on an OS-chosen port, torn down with the test."""
    running = companion.make_server(0, (PAGE_ORIGIN,))
    thread = threading.Thread(target=running.serve_forever, daemon=True)
    thread.start()
    try:
        yield running
    finally:
        running.shutdown()
        thread.join(timeout=5)
        running.server_close()


def _call(
    server,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict | None = None,
) -> tuple[int, dict[str, str], Any]:
    """One request, its status, headers and decoded JSON body (None if empty).

    The body is `Any`: a contract test subscribes into whatever the route
    answered, and an Optional there turned every `body["error"]` into a
    pyright error that said nothing about the route.
    """
    port = server.server_address[1]
    connection = http.client.HTTPConnection(companion.HOST, port, timeout=60)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        parsed: Any = json.loads(payload) if payload else None
        return response.status, dict(response.getheaders()), parsed
    finally:
        connection.close()


# --- it answers, and only for the pages the user allowed --------------------


def test_health_names_the_companion_and_its_api(server):
    status, headers, body = _call(server, "GET", "/health")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert body == {
        "app": companion.APP_ID,
        "api": companion.API,
        "version": __version__,
        "features": ["analyze", "render", "compare"],
    }


def test_a_page_origin_the_user_named_is_granted(server):
    status, headers, _body = _call(server, "GET", "/health", headers={"Origin": PAGE_ORIGIN})
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == PAGE_ORIGIN
    assert headers["Vary"] == "Origin"


def test_any_other_page_origin_is_refused_without_a_grant(server):
    status, headers, body = _call(server, "GET", "/health", headers={"Origin": "https://other.example"})
    assert status == 403
    assert "Access-Control-Allow-Origin" not in headers, (
        "a refused origin must not be granted: the browser would read the answer"
    )
    assert body == {"error": "origin"}


def test_a_loopback_page_needs_no_flag_but_a_published_one_does(server):
    """The developer's own static server is always welcome; a public page is not."""
    status, headers, _body = _call(server, "GET", "/health", headers={"Origin": "http://localhost:8768"})
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "http://localhost:8768"
    status, _headers, _body = _call(server, "GET", "/health", headers={"Origin": "https://someone.github.io"})
    assert status == 403


def test_a_client_with_no_origin_is_served(server):
    """`curl` and the tests are not pages: there is no origin to police."""
    assert _call(server, "GET", "/health")[0] == 200


def test_preflight_answers_the_methods_the_companion_has(server):
    status, headers, _body = _call(server, "OPTIONS", "/analyze", headers={"Origin": PAGE_ORIGIN})
    assert status == 204
    assert "POST" in headers["Access-Control-Allow-Methods"]
    assert "Content-Type" in headers["Access-Control-Allow-Headers"]
    assert headers["Access-Control-Allow-Origin"] == PAGE_ORIGIN


# --- /analyze: the export's schema, and nothing on disk ---------------------


def test_analyze_answers_the_export_schema_without_writing_anything(
    server, wav_bytes, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    status, _headers, body = _call(
        server,
        "POST",
        "/analyze",
        body=wav_bytes,
        headers={"Origin": PAGE_ORIGIN, "Content-Type": "audio/wav"},
    )
    assert status == 200, body
    assert body["api"] == companion.API
    report = body["report"]
    # the same payload the `.json` artifact carries, key for key
    assert {"source", "tempo", "key", "meter", "notes", "n_notes"} <= set(report)
    assert report["source"] == "upload"
    assert report["n_notes"] >= 3
    assert report["tempo"]["bpm"] > 0
    assert list(tmp_path.iterdir()) == [], "the route answers in memory: no artifact may appear"


def test_an_undecodable_body_is_the_callers_fault(server):
    status, _headers, body = _call(
        server, "POST", "/analyze", body=b"not audio at all",
        headers={"Content-Type": "audio/wav"},
    )
    assert status == 400
    assert body == {"error": "audio"}


def test_a_body_over_the_cap_is_refused_before_it_is_read(server):
    port = server.server_address[1]
    connection = http.client.HTTPConnection(companion.HOST, port, timeout=30)
    try:
        connection.putrequest("POST", "/analyze")
        connection.putheader("Content-Type", "audio/wav")
        connection.putheader("Content-Length", str(companion.MAX_BODY_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 413
        assert json.loads(response.read()) == {"error": "too-large"}
    finally:
        connection.close()


def test_a_second_analysis_is_refused_while_one_runs(server, wav_bytes, monkeypatch):
    """One DSP at a time: the second caller is told, not queued behind a mystery."""
    busy = threading.Lock()
    busy.acquire()
    monkeypatch.setattr(companion, "ANALYSIS_LOCK", busy)
    try:
        status, _headers, body = _call(
            server, "POST", "/analyze", body=wav_bytes, headers={"Content-Type": "audio/wav"}
        )
    finally:
        busy.release()
    assert status == 429
    assert body == {"error": "busy"}


# --- /render: the page's corrections, engraved by the desktop's writer -----

RENDER_BODY = {
    "tempo": {"bpm": 120.0},
    "meter": "4/4",
    "key": {"best": "C major", "score": 0.81},
    "notes": [
        {"start_s": 0.0, "end_s": 0.5, "midi": 60, "confidence": 0.9},
        {"start_s": 0.5, "end_s": 1.0, "midi": 67, "confidence": 0.8},
    ],
}


def _report(server, wav: bytes) -> dict:
    """The `/analyze` report, or a failure naming what came back instead."""
    status, _headers, payload = _call(
        server, "POST", "/analyze", body=wav, headers={"Content-Type": "audio/wav"}
    )
    assert status == 200, payload
    assert isinstance(payload, dict) and isinstance(payload.get("report"), dict), payload
    return payload["report"]


def _post_json(server, payload, headers: dict | None = None, path: str = "/render"):
    merged = {"Content-Type": "application/json"}
    merged.update(headers or {})
    body = json.dumps(payload).encode("utf-8")
    return _call(server, "POST", path, body=body, headers=merged)


def test_render_answers_the_score_the_desktop_would_engrave(server):
    status, _headers, body = _post_json(server, RENDER_BODY)
    assert status == 200, body
    assert body["api"] == companion.API
    assert body["n_notes"] == 2
    assert body["meter"] == "4/4"
    assert body["key"] == "C major"
    xml = body["musicxml"]
    assert xml.startswith("<?xml")
    assert "<score-partwise" in xml
    # two PITCHED notes: the exporter also writes a trailing rest to close the
    # measure, which carries no <step>
    assert xml.count("<step>") == 2
    assert "<fifths>0</fifths>" in xml
    assert "<beats>4</beats>" in xml
    # the pitches the page sent, spelled by the desktop's own notation
    assert "<step>C</step>" in xml and "<octave>4</octave>" in xml
    assert "<step>G</step>" in xml and "<octave>4</octave>" in xml


def test_render_renders_the_corrections_it_is_given(server):
    """A page that retuned a note must get ITS note, not the take's original."""
    corrected = json.loads(json.dumps(RENDER_BODY))
    corrected["notes"][0]["midi"] = 62  # C4 retuned to D4
    status, _headers, body = _post_json(server, corrected)
    assert status == 200, body
    assert "<step>D</step>" in body["musicxml"]
    assert "<step>C</step>" not in body["musicxml"]


def test_render_signs_the_key_the_page_carries(server):
    status, _headers, body = _post_json(server, dict(RENDER_BODY, key={"best": "F major"}))
    assert status == 200, body
    assert body["key"] == "F major"
    assert "<fifths>-1</fifths>" in body["musicxml"]


def test_render_answers_in_memory_only(server, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _post_json(server, RENDER_BODY)[0] == 200
    assert list(tmp_path.iterdir()) == [], "the score is returned, never written"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"notes": [], "tempo": {"bpm": 120}}, "notes"),
        ({"notes": [{"midi": 60, "start_s": 0.0, "end_s": 0.0}], "tempo": {"bpm": 120}}, "notes"),
        ({"notes": [{"midi": 200, "start_s": 0.0, "end_s": 1.0}], "tempo": 120}, "notes"),
        ({"notes": [{"midi": 60, "start_s": 0.0, "end_s": 1.0}]}, "tempo"),
        (dict(RENDER_BODY, tempo={"bpm": -3}), "tempo"),
        (dict(RENDER_BODY, meter="4"), "meter"),
        (dict(RENDER_BODY, key={"best": "H major"}), "key"),
    ],
)
def test_render_names_the_field_it_cannot_use(server, payload, code):
    status, _headers, body = _post_json(server, payload)
    assert status == 400
    assert body == {"error": code}


def test_render_refuses_a_body_that_is_not_json(server):
    status, _headers, body = _call(
        server, "POST", "/render", body=b"not json", headers={"Content-Type": "application/json"}
    )
    assert status == 400
    assert body == {"error": "body"}


def test_render_takes_json_not_audio(server):
    status, _headers, body = _call(
        server, "POST", "/render", body=b"{}", headers={"Content-Type": "audio/wav"}
    )
    assert status == 415
    assert body == {"error": "media-type"}


def _rows(report) -> list[dict]:
    """The report's notes as an exported `notes.json` carries them."""
    return [
        {"start_s": note["start_s"], "end_s": note["end_s"], "midi": note["midi"]}
        for note in report["notes"]
    ]


def _encoded(wav: bytes) -> str:
    """The take inside the JSON body, base64 — how `/compare` carries audio."""
    return base64.b64encode(wav).decode("ascii")


ONE_NOTE = {"notes": [{"start_s": 0, "end_s": 1, "midi": 60}]}


def test_compare_pairs_an_exported_reading_with_a_fresh_one(server, wav_bytes):
    """The same audio, read twice: every note pairs, and nothing moves.

    The export is built from what `/analyze` itself answered, so this is the
    honest case — the page exported a reading of THIS take and asks the desktop
    to read it again. A comparison that reported drift here would be inventing
    differences between two runs of one engine.
    """
    exported = {
        "format": "v2m-web-bundle/1",
        "generated_at": "2026-11-14-22-40",
        "source": "demo",
        "notes": _rows(_report(server, wav_bytes)),
    }
    status, _headers, body = _post_json(
        server, {"exported": exported, "audio_base64": _encoded(wav_bytes)}, path="/compare"
    )
    assert status == 200, body
    assert body["api"] == companion.API
    assert body["exported"]["n_notes"] == len(exported["notes"])
    assert body["exported"]["source"] == "demo"
    assert body["reading"]["n_notes"] == len(exported["notes"])
    assert body["reading"]["bpm"] > 0
    assert body["counts"] == {
        "stable": len(exported["notes"]),
        "near": 0,
        "changed": 0,
        "missing": 0,
        "added": 0,
    }, "two readings of one take must pair one to one, unchanged"
    assert len(body["pairs"]) == len(exported["notes"])
    assert all(pair["reasons"] == [] for pair in body["pairs"])


def test_compare_names_what_the_fresh_reading_moved(server, wav_bytes):
    """A drifted export is reported, not smoothed over: the reason is a CODE."""
    rows = _rows(_report(server, wav_bytes))
    rows[0] = dict(rows[0], midi=rows[0]["midi"] + 2)  # a wrong pitch
    rows[-1] = dict(rows[-1], start_s=rows[-1]["start_s"] + 0.4)  # a wrong attack
    status, _headers, body = _post_json(
        server, {"exported": {"notes": rows}, "audio_base64": _encoded(wav_bytes)}, path="/compare"
    )
    assert status == 200, body
    reasons = [reason for pair in body["pairs"] for reason in pair["reasons"]]
    assert "pitch" in reasons, body["pairs"]
    assert "attack" in reasons, body["pairs"]
    assert body["counts"]["changed"] >= 2
    counts = body["counts"]
    assert counts["stable"] == len(rows) - counts["changed"] - counts["missing"]


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"audio_base64": "AAAA"}, "exported"),
        ({"exported": {"notes": []}, "audio_base64": "AAAA"}, "exported"),
        ({"exported": {"notes": [{"midi": 60}]}, "audio_base64": "AAAA"}, "exported"),
        ({"exported": ONE_NOTE}, "audio"),
        ({"exported": ONE_NOTE, "audio_base64": "not base64 !"}, "audio"),
        ({"exported": ONE_NOTE, "audio_base64": "//79"}, "audio"),
    ],
)
def test_compare_names_the_field_it_cannot_use(server, payload, code):
    status, _headers, body = _post_json(server, payload, path="/compare")
    assert status == 400
    assert body == {"error": code}


def test_compare_refuses_a_body_that_is_not_json(server):
    status, _headers, body = _call(
        server,
        "POST",
        "/compare",
        body=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert status == 400
    assert body == {"error": "body"}


def test_compare_takes_json_not_audio(server):
    status, _headers, body = _call(
        server, "POST", "/compare", body=b"{}", headers={"Content-Type": "audio/wav"}
    )
    assert status == 415
    assert body == {"error": "media-type"}


def test_compare_is_refused_while_an_analysis_runs(server, monkeypatch):
    """It reads with pYIN: a second run is told, not queued."""
    busy = threading.Lock()
    busy.acquire()
    monkeypatch.setattr(companion, "ANALYSIS_LOCK", busy)
    try:
        status, _headers, body = _post_json(
            server, {"exported": ONE_NOTE, "audio_base64": "AAAA"}, path="/compare"
        )
    finally:
        busy.release()
    assert status == 429
    assert body == {"error": "busy"}


def test_compare_answers_in_memory_only(server, wav_bytes, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exported = {"notes": _rows(_report(server, wav_bytes))}
    status, _headers, _body = _post_json(
        server, {"exported": exported, "audio_base64": _encoded(wav_bytes)}, path="/compare"
    )
    assert status == 200
    assert list(tmp_path.iterdir()) == [], "the answer is a comparison, never an artifact"


def test_render_does_not_queue_behind_a_running_analysis(server, monkeypatch):
    """Layout is milliseconds: correcting notes stays usable mid-analysis."""
    busy = threading.Lock()
    busy.acquire()
    monkeypatch.setattr(companion, "ANALYSIS_LOCK", busy)
    try:
        status, _headers, body = _post_json(server, RENDER_BODY)
    finally:
        busy.release()
    assert status == 200, body


# --- routing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("GET", "/analyze", 405),  # presets only
        ("POST", "/health", 405),
        ("GET", "/render", 405),  # a render is a POST, never a fetch
        ("GET", "/compare", 405),  # and so is a comparison
        ("POST", "/nope", 404),
    ],
)
def test_the_routes_are_the_ones_it_has(server, method, path, expected):
    status, _headers, body = _call(server, method, path, body=b"" if method == "POST" else None)
    assert status == expected
    assert body["error"] in {"method", "path"}


# --- the two decisions a reviewer would re-open -----------------------------


def test_the_socket_never_listens_beyond_this_machine():
    assert companion.HOST == "127.0.0.1"
    running = companion.make_server(0)
    try:
        assert running.server_address[0] == "127.0.0.1"
    finally:
        running.server_close()


def test_the_cli_exposes_the_companion_with_its_port_and_origins():
    args = cli.build_parser().parse_args(["serve", "--port", "8790", "--allow-origin", PAGE_ORIGIN])
    assert args.command == "serve"
    assert args.port == 8790
    assert args.allow_origin == [PAGE_ORIGIN]
    default = cli.build_parser().parse_args(["serve"])
    assert default.port == companion.DEFAULT_PORT
    assert default.allow_origin == []
