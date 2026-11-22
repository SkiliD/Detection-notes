// The take a reader CHOOSES, through the browser's own decoder.
//
// Every other check hands the page its samples (the demo synthesises them) or a
// microphone (Chrome's fake device): the file path — a WAV decoded by
// `decodeAudioData` and then fed through the same chain as a capture — was the
// one reading a person can pick that no machine exercised. `DOM.setFileInputFiles`
// puts a real file in `#file`, which is exactly what choosing a recording does,
// so the decoder, the take's own sample rate and the reading behind the table
// are all the page's, not a fixture's.
//
// The archive is read here by its RAW HEADERS (end record → central directory →
// local header → data), as an unzip tool does: a page that exported a file its
// own writer can read has proven nothing about what the reader downloads.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const { chooseFile, download, startBrowser } = await import("./browser.mjs");
const { readArchive } = await import("./archive.mjs");
import { tonesWav } from "./tone.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8"));
const WORK = await mkdtemp(join(tmpdir(), "v2m-file-"));
const TAKE = join(WORK, "prise.wav");
const JUNK = join(WORK, "illisible.wav");

// The same phrase the capture check feeds Chrome's fake microphone (A4, a short
// gap, D5), from the one writer in `tone.mjs`: the two entries into the chain
// can then be held to the same reading.
await writeFile(TAKE, tonesWav());
// Text under an audio name: no container the browser's decoder recognises, so
// it refuses at once — the sentence is the point of this file.
await writeFile(JUNK, Buffer.from("ceci n'est pas un enregistrement\n", "utf8"));

let driver = null;
async function page() {
  driver ??= await startBrowser({ root: WEB, width: 1280, height: 900 });
  return driver;
}

test.after(async () => {
  if (driver) await driver.close();
});

/** The first four bytes of an entry, read as the text a format starts with. */
const magicOf = (bytes) => Buffer.from(bytes.subarray(0, 4)).toString("latin1");

/** The page's own status sentence, trimmed. */
const statusOf = async () =>
  (await page()).evaluate('document.querySelector("#status").textContent.trim()');

test("a file the browser cannot decode names the file, not the microphone", async () => {
  await chooseFile(await page(), JUNK, async () => (await statusOf()) === STRINGS.fr.error_file);
  const status = await statusOf();
  assert.equal(
    status,
    STRINGS.fr.error_file,
    `an unreadable file reads as "${status}"; the microphone's own sentence sends the reader to a device`,
  );
});

test("a chosen file is read, exported and bundled as one take", async () => {
  const rowsOf = async () => (await page()).evaluate('document.querySelectorAll("#rows tr").length');
  await chooseFile(
    await page(),
    TAKE,
    async () => (await rowsOf()) > 0 && (await statusOf()) === STRINGS.fr.ready,
  );
  const rows = await rowsOf();
  assert.ok(rows > 0, "the chosen file produced no notes");

  const status = await statusOf();
  assert.equal(status, STRINGS.fr.ready, `the reading finished reading as "${status}"`);

  // The archive, from outside: audio first (the only entry nothing can rebuild),
  // then the measurements, then the MIDI, then the notice.
  const archive = readArchive(await download(await page(), "bundle", join(WORK, "bundle"), ".zip"));
  const names = archive.map((entry) => entry.name);
  assert.equal(names[0], "prise.wav", `the archive starts with ${names[0]}`);
  assert.equal(names[1], "notes.json", `the second entry is ${names[1]}`);
  assert.equal(names[2], "prise.mid", `the third entry is ${names[2]}`);
  assert.equal(names.length, 4, `the archive holds ${names.length} entries`);
  assert.equal(magicOf(archive[0].data), "RIFF", "the archive's WAV is not a WAV");
  assert.equal(magicOf(archive[2].data), "MThd", "the archive's MIDI is not a MIDI");

  // The reading itself: the two notes the file holds, at the length it holds
  // them, in the SAME table the page shows.
  const document = JSON.parse(Buffer.from(archive[1].data).toString("utf8"));
  assert.equal(document.source, "file", `the archive names its source "${document.source}"`);
  // `decodeAudioData` RESAMPLES to the browser's own output rate (measured at
  // 96 kHz on a desk with a 44.1 kHz file), so the take's rate is the decoded
  // one — what the page must keep is the DURATION the file holds.
  assert.ok(document.sample_rate_hz >= 22050, `the take was read at ${document.sample_rate_hz} Hz`);
  assert.ok(
    Math.abs(document.duration_s - 1.8) < 0.05,
    `the take lasts ${document.duration_s} s instead of the file's 1.8`,
  );
  assert.deepEqual(document.notes.map((note) => note.midi), [69, 74], "the file's notes moved");
  for (const note of document.notes) {
    assert.ok(Math.abs(note.duration_s - 0.8) < 0.12, `a note lasts ${note.duration_s} s instead of 0.8`);
  }
  assert.equal(rows, document.notes.length, "the table and the export disagree on how many notes there are");

  // And the MIDI download is a file, not an empty click.
  const midi = await download(await page(), "export", join(WORK, "midi"), ".mid");
  assert.equal(magicOf(midi), "MThd", "the MIDI export is not a MIDI file");

  // A SECOND archive, after the first released the concatenation it packed:
  // the take is rebuilt from its own blocks, so the audio comes back byte for
  // byte and the reading is the same one. An export that spent the take to
  // save memory would fail here.
  const again = readArchive(await download(await page(), "bundle", join(WORK, "bundle-2"), ".zip"));
  assert.deepEqual(
    again.map((entry) => entry.name),
    names,
    "the second archive does not carry the same four members",
  );
  assert.ok(
    Buffer.from(again[0].data).equals(Buffer.from(archive[0].data)),
    "the take was not rebuilt: the second export's audio differs from the first",
  );
  const redone = JSON.parse(Buffer.from(again[1].data).toString("utf8"));
  assert.deepEqual(redone.notes, document.notes, "the second export reads the take differently");
});
