"""The page's exports and its parity fixtures, pinned away from its layout.

The archive's shape is decided in `web/bundle.js` and `web/zip.js`, and the
reading those exports carry is only worth comparing if the port reads audio
the way the desktop does. Two contracts live here:

- `tests/data/parity.wav` + `tests/reference/parity.json` must still be what
  `scripts/export_web_fixtures.py` writes and reads: a committed sung phrase
  and the desktop's LIVE chain over that same audio, at every rate a browser
  can decode it to, so the page's own `notes.json` can be compared with the
  desktop instead of with a fixture of pitch frames;
- the archive keeps the recording first, compresses only where the browser
  offers a raw stream, falls back to the stored writer, releases the
  concatenation it built for the pack, and reads its cents from the one
  shared unit.

The rest of the port (constants, layout, strings, workflows) stays in
`tests/test_web_build.py`; `web/test/file.test.mjs` drives the archive a
reader actually downloads through its raw headers in a browser.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
sys.path.insert(0, str(REPO))


def test_the_page_can_export_the_whole_take_as_one_archive():
    # The MIDI says what should be PLAYED; only the recording says what was
    # SUNG, and only the JSON says how it was measured. Re-analysing a take,
    # comparing two of them or improving the algorithm needs all three together
    # — so the page writes one archive, and the notice inside it says what each
    # file is without assuming the reader knows a format.
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="bundle"' in html and 'data-i18n="bundle"' in html, (
        "the page lost its whole-take export button"
    )
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert 'import { buildBundle } from "./bundle.js";' in app, "app.js no longer builds a bundle"
    assert '"click", exportBundle' in app, "the bundle button is not wired"
    # A control that never enables is a dead button; the audio is required, so a
    # take with notes but no kept signal cannot offer an archive.
    assert '$("bundle").disabled = notes.length === 0 || takeChunks.length === 0;' in app, (
        "the bundle button is enabled without a take to put in it"
    )

    # The AUDIO must be the first entry: the notes are a reading of it and the
    # MIDI a reading of those, so an archive that opens on `notes.json` invites
    # exactly the mistake this feature exists to prevent.
    source = (WEB / "bundle.js").read_text(encoding="utf-8")
    order = [source.index(f"name: {name}") for name in ("AUDIO_NAME", "NOTES_NAME", "MIDI_NAME")]
    assert order == sorted(order), "the archive no longer puts the recording first"

    # The two languages carry the same keys and the page's own requests are
    # covered by `test_web_build.test_both_languages_carry_every_key_the_page_asks_for`,
    # so one language is enough here: what this adds is that each notice key
    # carries TEXT, since an empty value passes every key-set check.
    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    for key in (
        "bundle",
        "bundle_readme_intro",
        "bundle_readme_audio",
        "bundle_readme_notes",
        "bundle_readme_midi",
    ):
        assert strings["fr"].get(key), f"{key} is missing or empty"


def test_the_archive_compresses_when_the_browser_can_and_stores_when_it_cannot():
    # A take is a few seconds of 16-bit audio: deflate roughly halves the
    # download for every browser that offers a raw stream, and a browser that
    # does not must still get a valid archive. The stored fallback is the
    # writer this page shipped before compression, byte for byte, so an archive
    # written today and one written in that browser are the same file. What the
    # guard refuses is the silent middle: compressing WITHOUT probing first,
    # zlib-wrapping a stream into a file an unzip tool turns into garbage, and
    # a CRC taken over the compressed bytes.
    source = (WEB / "zip.js").read_text(encoding="utf-8")
    assert 'new CompressionStream("deflate-raw")' in source, (
        "the archive no longer asks for RAW deflate"
    )
    probe = source.index('new CompressionStream("deflate-raw")')
    assert "try {" in source[:probe] and "catch {" in source[probe:], (
        "the raw-deflate probe lost its try/catch: an unsupported format throws"
    )
    assert 'new CompressionStream("deflate")' not in source, (
        "a zlib-wrapped stream is not ZIP method 8"
    )
    assert "export async function zipArchive(" in source
    assert "compress = browserDeflate" in source, "the compressor is not the browser's by default"
    assert "packed.length < bytes.length" in source, (
        "an entry that does not shrink is deflated anyway, and grows the archive"
    )
    assert "crc32(entry.bytes)" in source, "the CRC must be taken over the original bytes"
    fallback = source[source.index("export function zipStore(") :]
    assert "method: 0" in fallback, "the stored fallback is gone"

    bundle = (WEB / "bundle.js").read_text(encoding="utf-8")
    assert "export async function buildBundle(" in bundle
    assert "await zipArchive(entries" in bundle, "the bundle no longer awaits its archive"
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "async function exportBundle(" in app and "await buildBundle({" in app, (
        "the download still reads a promise as bytes"
    )
    assert 'setStatus("error_bundle");' in app, "a failed archive has no message of its own"
    strings = json.loads((WEB / "strings.json").read_text(encoding="utf-8"))
    for language in ("fr", "en"):
        assert strings[language].get("error_bundle"), f"{language}: error_bundle is missing"
        assert strings[language]["error_bundle"] != strings[language]["error_file"], (
            f"{language}: the archive failure borrows another failure's sentence"
        )


def test_a_take_is_held_once_after_its_archive():
    """The export's concatenation is released, so the take is ONE copy.

    `takeVoice()` builds a full copy of the blocks for a mix playback and for
    the archive. Keeping it after an export made the page hold the take twice
    for the rest of the session (measured: +141 MB for a 69 MB signal, where a
    single copy was +72). The archive has copied what it packs, so the view goes
    on the exit path — a FAILED export must release it too.
    """
    app = (WEB / "app.js").read_text(encoding="utf-8")
    body = app[app.index("async function exportBundle(") :]
    body = body[: body.index("\nasync function ")]
    assert "takeVoice()" in body, "the archive no longer keeps the take's signal"
    assert "finally {" in body, "the release is not on the exit path"
    assert "takeVoiceCache = null;" in body[body.index("finally {") :], (
        "the archive's concatenation survives the export: the page holds the take twice"
    )
    # ... and the budget that MEASURES it still exists, still opt-in, still
    # bounded against the captured take rather than a second copy. The ceiling
    # IS the contract: how it samples, at which rate and with which factors is
    # the browser check's own business, so no assertion here quotes its code.
    budget = (WEB / "test" / "long-take.test.mjs").read_text(encoding="utf-8")
    assert "V2M_LONG_TAKE" in budget
    assert "const exportCeiling = captureMb + EXPORT_SLACK_MB;" in budget, (
        "the export ceiling allows a second copy of the take again"
    )


def test_the_archive_exports_the_very_number_the_table_prints():
    """`notes.json` and the table read the SAME cents unit.

    `web/cents.js` is the one conversion (whole signed cents). The archive
    importing it is the fact: a second conversion written beside it is what a
    reader would see drift, and the hardware recipe's gesture 4 asks a reviewer
    to compare the two surfaces. HOW each surface formats the number is the
    page's own business, so no assertion here quotes its expressions.
    """
    bundle = (WEB / "bundle.js").read_text(encoding="utf-8")
    assert 'import { centsOf } from "./cents.js";' in bundle, (
        "the archive converts cents on its own instead of reading the shared unit"
    )


def _parity_covers_the_chain(notes) -> bool:
    """Whether a frozen reading still exercises what the phrase exists for.

    A tone-only phrase would pass every boundary bound while pinning nothing:
    the reading must hold at least five notes, TWO of them on the same pitch
    (the consonant's re-articulation) and at least four distinct pitches (a
    glide read as a note of its own, or a lost note, changes that set).
    """
    pitches = [note["midi"] for note in notes]
    repeats = sum(1 for before, after in zip(pitches, pitches[1:], strict=False) if before == after)
    return len(pitches) >= 5 and repeats >= 1 and len(set(pitches)) >= 4


def test_the_page_reads_a_sung_phrase_as_the_desktop_live_chain_does():
    """The page's reading of committed audio, against the desktop's own.

    `cases.json` hands the port pitch frames and `yin.json` samples: neither
    says what the WHOLE chain decides about a phrase someone would sing. The
    page has no offline pipeline, so the twin is the desktop's
    `LivePitchStream` fed the page's own 2048-sample blocks — not
    `analyze_file`, whose look-ahead the page never claims.
    """
    from scripts.export_web_fixtures import (
        PARITY_BLOCK,
        PARITY_GOLDEN,
        PARITY_WAV,
        parity_reference,
        parity_wav_bytes,
    )

    stale = (
        "tests/data/parity.wav and tests/reference/parity.json are stale: run "
        "`./.venv/Scripts/python.exe -m scripts.export_web_fixtures`, then "
        "`node --test web/test/parity.test.mjs`"
    )
    assert PARITY_WAV.read_bytes() == parity_wav_bytes(), stale
    checked = json.loads(PARITY_GOLDEN.read_text(encoding="utf-8"))
    assert parity_reference() == checked, stale

    # One reading per rate the browser can decode to: `decodeAudioData`
    # resamples to the DEVICE's rate (measured 96 kHz for a 44.1 kHz file), and
    # the glide hands over 70 ms later at 48 kHz than at 44.1 — a one-rate
    # golden would read a rate as a drift. The two capture rates and the
    # measured desk rate are mandatory, not "whatever the machine recorded".
    recorded = {rate["sr"] for rate in checked["rates"]}
    for sr in (44_100, 48_000, 96_000):
        assert sr in recorded, f"the golden lost its {sr} Hz reading: {sorted(recorded)}"
    for rate in checked["rates"]:
        assert _parity_covers_the_chain(rate["notes"]), (
            f"the phrase reads {[note['midi'] for note in rate['notes']]} at {rate['sr']} Hz: "
            "it lost the re-articulation, the glide or a note"
        )

    # The geometry compared is the page's own feed, and BOTH sides declare the
    # block size — the exporter as `PARITY_BLOCK`, the page as `FILE_BLOCK`. A
    # golden written at one size against a page reading another would compare
    # two FEEDS instead of two implementations of the same chain.
    page = (WEB / "app.js").read_text(encoding="utf-8")
    declared = re.search(r"^const FILE_BLOCK = (\d+);$", page, re.M)
    assert declared, "web/app.js no longer declares its file block"
    assert int(declared.group(1)) == PARITY_BLOCK, (
        "the page reads a file in another block than the golden was written from"
    )
    assert checked["chain"]["block"] == PARITY_BLOCK, "the golden moved its block geometry"
    assert "LivePitchStream" in checked["chain"]["reader"]

    # And the browser side reads the page's RATE, with a bound in frames.
    source = (WEB / "test" / "parity.mjs").read_text(encoding="utf-8")
    assert "golden.rates.find((rate) => rate.sr === sr)" in source, (
        "the comparator stopped picking the reading the browser's rate decoded to"
    )
    assert "PARITY_MAX_SHIFT_S = 0.024;" in source, "the boundary bound was widened"


def test_every_fixture_writer_asks_for_lf():
    """Every fixture keeps the exporter's line endings, on every platform.

    On Windows, `Path.write_text` without `newline="\n"` turns every line
    ending into CRLF: the frozen artifact would then be the one the writing
    machine happened to produce. `web/constants.js` is compared as TEXT
    against the regenerated source, so a CRLF copy reads as stale; the four
    JSONs are parsed, and a checkout carrying CR is still the same divergence
    — `.gitattributes` pins every fixture to `eol=lf` so a fresh clone cannot
    bring one back.
    """
    source = (REPO / "scripts" / "export_web_fixtures.py").read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_text"
    ]
    assert calls, "the fixture writer no longer writes a text artifact"
    for call in calls:
        asked = [word for word in call.keywords if word.arg == "newline"]
        assert asked, f"a fixture is written without newline at line {call.lineno}"
        assert ast.literal_eval(asked[0].value) == "\n", (
            f"a fixture keeps the platform's line endings at line {call.lineno}"
        )

    attributes = (REPO / ".gitattributes").read_text(encoding="utf-8")
    for path in FIXTURE_TEXT:
        raw = path.read_bytes()
        assert b"\r" not in raw, f"{path.name} carries a CR on disk: the bytes are compared"
        assert raw.endswith(b"\n"), f"{path.name} does not end on a line break"
        relative = path.relative_to(REPO).as_posix()
        assert f"{relative} text eol=lf" in attributes, (
            f"{relative} is not pinned to LF: a Windows checkout rewrites it"
        )


def test_the_parity_guards_bite():
    """A tone-only reading and a phrase missing its re-attack must be refused."""
    assert not _parity_covers_the_chain([{"midi": 69}, {"midi": 74}]), "two tones passed"
    assert not _parity_covers_the_chain(
        [{"midi": 60}, {"midi": 62}, {"midi": 64}, {"midi": 65}, {"midi": 67}]
    ), "a phrase with no re-attack passed"


FIXTURE_JSONS = ("cases.json", "intonation.json", "yin.json", "yin-rates.json")
FIXTURE_JSON_PATHS = tuple(WEB / "test" / name for name in FIXTURE_JSONS)
# The text artifacts `_write_text` owns: the four JSONs plus the ported constants.
FIXTURE_TEXT = (WEB / "constants.js", *FIXTURE_JSON_PATHS)
FIXTURE_WRITER = "scripts/export_web_fixtures.py"
# The parity golden is the FIFTH JSON `_write_text` writes (beside its WAV): a
# copy naming another writer reads as fresh there too, so it carries the name.
PARITY_JSON = REPO / "tests" / "reference" / "parity.json"


def _fixture_names_its_writer(payload: object) -> bool:
    """True when a fixture records the one entry point allowed to have written it."""
    return isinstance(payload, dict) and payload.get("generated_by") == FIXTURE_WRITER


def _fixture_label(path: Path) -> str:
    """The fixture's name in a failure, repo-relative so the line stays readable."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def test_every_fixture_names_the_one_script_that_writes_it():
    """A fixture written by another entry point reads as fresh, and the page trusts it.

    The FIVE JSONs the generator writes -- the four frozen from the desktop
    chain plus the parity golden -- each carry `generated_by`, and a reader
    loads these bytes as they are: a name that is not this file leaves no way
    to tell a re-export from a stale copy. The path is checked on disk too, so
    a rename fails here instead of silently blessing the old spelling.
    """
    assert (REPO / FIXTURE_WRITER).is_file(), f"{FIXTURE_WRITER} is gone"
    for fixture in (*FIXTURE_JSON_PATHS, PARITY_JSON):
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        assert _fixture_names_its_writer(payload), (
            f"{_fixture_label(fixture)} says "
            f"generated_by={payload.get('generated_by')!r}, "
            f"not {FIXTURE_WRITER!r}"
        )


def test_the_generator_name_guard_bites():
    """Both ways a fixture can name the wrong writer must be refused."""
    assert not _fixture_names_its_writer({"generated_by": "scripts.export_web_fixtures"})
    assert not _fixture_names_its_writer({})
    assert _fixture_names_its_writer({"generated_by": FIXTURE_WRITER})
