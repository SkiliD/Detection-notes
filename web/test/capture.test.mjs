// The capture path, on a browser that cannot possibly have a microphone.
//
// Everything else in `web/test/` replays samples the page is GIVEN (the demo,
// the conformance cases, the YIN frames): no check ever pressed "Sing" with a
// working input, so the worklet, the fake-mic plumbing and the export after a
// capture were the one user path no machine exercised. Chrome can be told to
// pretend: `--use-fake-device-for-media-stream` publishes a microphone and
// `--use-file-for-fake-audio-capture` feeds it a file, so the page runs its real
// chain — permission, worklet, YIN, segmenter, MIDI — exactly as on a desk.
//
// The audio OUTPUT is the catch: `AudioContext.resume()` settles only when an
// output device can start, and a headless runner with no sound card leaves it
// pending forever, which is why the page now bounds that wait and reports the
// engine failure instead of sitting on "microphone open". This file asserts
// that reported failure where no output exists, and skips the capture itself
// with the measured reason — `V2M_REQUIRE_AUDIO=1` turns that skip into a
// failure, which is what CI sets once it has given the runner a null sink.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const { startBrowser, pressDemo, sleep } = await import("./browser.mjs");

const WEB = fileURLToPath(new URL("..", import.meta.url));
const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8"));
const WORK = await mkdtemp(join(tmpdir(), "v2m-capture-"));
const WAV = join(WORK, "tones.wav");

/** A 16-bit mono 44.1 kHz take: A4, a short gap, D5 — two notes any YIN finds. */
function tonesWav(rate = 44100) {
  const parts = [
    { hz: 440, seconds: 0.8 },
    { hz: 0, seconds: 0.2 },
    { hz: 587.33, seconds: 0.8 },
  ];
  const samples = [];
  for (const part of parts) {
    for (let i = 0; i < Math.round(part.seconds * rate); i += 1) {
      samples.push(part.hz === 0 ? 0 : Math.sin((2 * Math.PI * part.hz * i) / rate) * 0.35);
    }
  }
  const buffer = Buffer.alloc(44 + samples.length * 2);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + samples.length * 2, 4);
  buffer.write("WAVEfmt ", 8);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(1, 22); // mono
  buffer.writeUInt32LE(rate, 24);
  buffer.writeUInt32LE(rate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(samples.length * 2, 40);
  samples.forEach((value, index) => buffer.writeInt16LE(Math.round(value * 32767), 44 + index * 2));
  return buffer;
}

await writeFile(WAV, tonesWav());

let driver = null;
async function page() {
  driver ??= await startBrowser({
    root: WEB,
    width: 1200,
    height: 900,
    args: [
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream", // no prompt to answer in a headless run
      `--use-file-for-fake-audio-capture=${WAV}`,
    ],
  });
  return driver;
}

test.after(async () => {
  if (driver) await driver.close();
});

/**
 * The state this browser's audio graph reaches, under the page's own bound.
 *
 * No output device means `resume()` never settles and the state stays
 * `suspended` — the exact fact `startedContext` gives up on, measured here
 * rather than assumed.
 */
async function graphState(driver) {
  const expression = `
  (async () => {
    const context = new AudioContext({ latencyHint: "interactive" });
    await Promise.race([
      context.resume().catch(() => {}),
      new Promise((done) => setTimeout(done, 2500)),
    ]);
    return context.state;
  })()
  `;
  return driver.evaluate(expression);
}

/**
 * The notes a type-0 MIDI file carries, in seconds.
 *
 * The page writes 480 ticks per quarter at a fixed 120 BPM (`midi.js`), so a
 * tick is 60 / (120 * 480) s either way. Reading the file back is the point:
 * the comparison below is about the TRANSCRIPTION, not about bytes.
 */
function notesOf(bytes) {
  const division = (bytes[12] << 8) | bytes[13];
  const seconds = (ticks) => (ticks * 60) / (120 * division);
  let at = 22; // past "MThd" (14 bytes) and "MTrk" + length (8)
  let tick = 0;
  const open = new Map();
  const notes = [];
  const readVar = () => {
    let value = 0;
    while (true) {
      const byte = bytes[at];
      at += 1;
      value = (value << 7) | (byte & 0x7f);
      if (!(byte & 0x80)) return value;
    }
  };
  while (at < bytes.length) {
    tick += readVar();
    const status = bytes[at];
    if (status === 0xff) break; // end of track
    at += 1;
    if (status === 0xc0) {
      at += 1; // program change
      continue;
    }
    const pitch = bytes[at];
    const velocity = bytes[at + 1];
    at += 2;
    if ((status & 0xf0) === 0x90 && velocity > 0) {
      open.set(pitch, tick);
    } else if (open.has(pitch)) {
      notes.push({ pitch, start: seconds(open.get(pitch)), duration: seconds(tick - open.get(pitch)) });
      open.delete(pitch);
    }
  }
  return notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

/**
 * The same transcription, measured from its own first note.
 *
 * When the fake stream starts relative to the click is wall-clock, not music:
 * subtracting the first onset keeps the comparison on what the page promises
 * (these notes, this long, this far apart) and off when the browser opened the
 * microphone.
 */
const fromFirstNote = (notes) => notes.map((note) => ({ ...note, start: note.start - notes[0].start }));

/** Exports the take as MIDI and answers its bytes (the page is one download). */
async function exportedMidi(driver, folder) {
  await mkdir(folder, { recursive: true });
  await driver.send("Page.setDownloadBehavior", { behavior: "allow", downloadPath: folder });
  await driver.evaluate("document.querySelector('#export').click()");
  for (let waited = 0; waited < 8000; waited += 200) {
    const names = (await readdir(folder)).filter((name) => name.endsWith(".mid"));
    if (names.length) return readFile(join(folder, names[0]));
    await sleep(200);
  }
  throw new Error("the page exported no MIDI");
}

test("the reader recovers exactly what the page's writer wrote", async () => {
  // The live comparison below decodes MIDI; a reader that quietly dropped a
  // note would make two WRONG takes look identical. This pins it to the real
  // writer, note for note, with no browser in between.
  const { writeMidi } = await import("../midi.js");
  const written = writeMidi([
    { start: 0.25, end: 1.0, midi: 69, strength: 0.6 },
    { start: 1.5, end: 2.0, midi: 72, strength: 0.4 },
  ]);
  const notes = notesOf(written);
  assert.deepEqual(notes.map((note) => note.pitch), [69, 72]);
  // One tick is 1.04 ms: the writer's rounding, not a different note.
  assert.ok(Math.abs(notes[0].start - 0.25) < 0.002, `starts at ${notes[0].start}`);
  assert.ok(Math.abs(notes[0].duration - 0.75) < 0.002, `lasts ${notes[0].duration}`);
  assert.ok(Math.abs(notes[1].start - 1.5) < 0.002, `second starts at ${notes[1].start}`);
});

test("the demo take exports the same MIDI twice", async () => {
  // The export end of the capture path: same notes, same bytes, twice. It also
  // proves the download plumbing the microphone test depends on.
  const driver = await page();
  await pressDemo(driver);
  const first = await exportedMidi(driver, join(WORK, "demo-1"));
  const second = await exportedMidi(driver, join(WORK, "demo-2"));
  assert.equal(first.subarray(0, 4).toString("latin1"), "MThd", "not a MIDI file");
  // Byte equality is legal HERE and only here: the demo is fed frame by frame,
  // no microphone, no clock, no block can go missing between the two exports.
  assert.ok(first.equals(second), "two exports of one take differ");
  assert.ok(first.length > 14, "the export carries no events");
  const demo = notesOf(first);
  assert.ok(demo.length >= 3, `the demo transcription read back ${demo.length} notes`);
  assert.ok(demo.every((note) => note.pitch >= 21 && note.pitch <= 108), "a pitch left the piano");
  assert.ok(demo.every((note) => note.duration > 0), "a note has no length");
});

test("the fake microphone opens the page's capture and two takes export the same MIDI", async (t) => {
  const driver = await page();
  const state = await graphState(driver);
  if (state !== "running") {
    // No output device: the page must SAY so — the promise it used to wait on
    // never settles, and the reader was left on "microphone open" for good.
    await driver.evaluate("document.querySelector('#record').click()");
    await sleep(2600);
    const status = await driver.evaluate("document.querySelector('#status').textContent.trim()");
    assert.ok(
      [STRINGS.fr.error_mic, STRINGS.en.error_mic].includes(status),
      `a graph that never started reads as "${status}", not as the engine failure`,
    );
    const why = `no audio output device (context ${state})`;
    if (process.env.V2M_REQUIRE_AUDIO === "1") {
      assert.fail(`the capture path did not run: ${why}`);
    }
    t.skip(`needs an audio output device: ${why}`);
    return;
  }

  const take = async (index) => {
    await driver.evaluate("document.querySelector('#record').click()");
    let live = false;
    for (let waited = 0; waited < 15000 && !live; waited += 100) {
      live = await driver.evaluate("!document.querySelector('#stop').disabled");
      if (!live) await sleep(100);
    }
    assert.ok(live, "the page never reached a live capture");
    const status = await driver.evaluate("document.querySelector('#status').textContent.trim()");
    assert.ok(
      [STRINGS.fr.listening, STRINGS.en.listening].includes(status),
      `a live capture reads as "${status}"`,
    );
    await sleep(2200);
    await driver.evaluate("document.querySelector('#stop').click()");
    await sleep(800);
    const rows = await driver.evaluate("document.querySelectorAll('#rows tr').length");
    assert.ok(rows > 0, `the fake take produced no notes (take ${index})`);
    return exportedMidi(driver, join(WORK, `take-${index}`));
  };

  const first = fromFirstNote(notesOf(await take(1)));
  const second = fromFirstNote(notesOf(await take(2)));
  // What the page PROMISES is the transcription: the same notes, at the same
  // pitches, a frame or two apart. It does NOT promise byte-identical files —
  // the live path timestamps blocks as they arrive, so a stalled runner that
  // loses a block shifts a tick or two while the take is still right. Byte
  // equality stays where it is guaranteed: the demo export above, which is fed
  // frame by frame with no clock in between.
  assert.equal(
    first.length,
    second.length,
    `two takes of one recording gave ${first.length} and ${second.length} notes`,
  );
  assert.ok(first.length > 0, "the fake take transcribed nothing");
  const LIMIT_S = 0.06; // about five 11.6 ms hops: a lost block, not a wrong note
  first.forEach((note, index) => {
    const other = second[index];
    assert.equal(note.pitch, other.pitch, `note ${index + 1}: different pitch`);
    assert.ok(
      Math.abs(note.start - other.start) <= LIMIT_S,
      `note ${index + 1} starts ${(note.start - other.start).toFixed(3)} s apart`,
    );
    assert.ok(
      Math.abs(note.duration - other.duration) <= LIMIT_S,
      `note ${index + 1} lasts ${(note.duration - other.duration).toFixed(3)} s longer`,
    );
  });
});
