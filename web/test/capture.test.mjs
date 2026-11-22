// The capture path, on a browser that cannot possibly have a microphone.
//
// Everything else in `web/test/` replays samples the page is GIVEN (the demo,
// the conformance cases, the YIN frames). This file is the one place the
// worklet, the fake-mic plumbing and the export after a capture run together.
// Chrome can be told to pretend: `--use-fake-device-for-media-stream` publishes
// a microphone and `--use-file-for-fake-audio-capture` feeds it a file, so the
// page runs its real chain — permission, worklet, YIN, segmenter, MIDI —
// exactly as on a desk.
//
// The audio OUTPUT is the catch: `AudioContext.resume()` settles only when an
// output device can start, and a headless runner with no sound card leaves it
// pending forever. The page bounds that wait and reports the engine failure,
// and this file asserts that failure where no output exists, then skips the
// capture itself with the measured reason — `V2M_REQUIRE_AUDIO=1` turns that
// skip into a failure, which is what CI sets once it has given the runner a
// null sink.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";

import { NOTE_NAMES } from "../constants.js";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const { audioState, exportedBytes, fakeMicrophone, pressDemo, rowTrace, sleep, startBrowser, traceRows } =
  await import("./browser.mjs");
import { tonesWav } from "./tone.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8"));
const WORK = await mkdtemp(join(tmpdir(), "v2m-capture-"));
const WAV = join(WORK, "tones.wav");

// Three tones, a gap after each, and a long silence at the end. The fake stream
// LOOPS this file from an arbitrary phase, so a capture can open in the middle
// of a tone: the take then starts with one PARTIAL note, and every later tone
// is whole.
//
// The stop must therefore be decided by what the PAGE heard, never by a fixed
// capture window: a wall-clock stop lands wherever the loop happens to be, and
// a tone still sounding when the click lands is closed by the take's own flush,
// so its length reports the click instead of the music — two takes of ONE
// recording then disagree by more than a lost block. The page's own reading,
// `#rows` and an empty Hz line, is the signal that the tone is over.
const PARTS = [
  { hz: 440, seconds: 0.8 },
  { hz: 0, seconds: 0.6 },
  { hz: 587.33, seconds: 0.8 },
  { hz: 0, seconds: 0.6 },
  { hz: 698.46, seconds: 0.8 },
  { hz: 0, seconds: 2.4 },
];
const TONES = PARTS.filter((part) => part.hz > 0).length;
// The loop is 6.0 s and its three tones end by 3.8 s, so the silence after the
// third tone is 2.2 s long: a stop taken once the page has heard one more note
// than the file has tones always lands in that gap.

await writeFile(WAV, tonesWav({ parts: PARTS }));

let driver = null;
async function page() {
  driver ??= await startBrowser({
    root: WEB,
    width: 1200,
    height: 900,
    // A muted browser opens no output stream, and an AudioContext that cannot
    // start never records: this is the one caller that needs its graph to run.
    mute: false,
    args: fakeMicrophone(WAV),
  });
  return driver;
}

test.after(async () => {
  if (driver) await driver.close();
});

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
 * The MIDI number a readout slot NAMES (`C4` → 60), or null when it names none.
 *
 * `#note` carries `midiToName`'s spelling on every path that writes it, so a
 * reader can hold it against the pitches a transcription exported.
 */
function pitchOfName(text) {
  const match = /^([A-G])(#?)(-?\d)$/.exec(text);
  if (!match) return null;
  return 12 * (Number(match[3]) + 1) + NOTE_NAMES.indexOf(match[1] + match[2]);
}

/**
 * Exports the take as MIDI and answers the bytes the PAGE built.
 *
 * The download itself is a browser artifact — a file that appears after a
 * rename, at a moment no page fact names, and a full run can starve that write
 * long after the export happened. These two checks are about what the page
 * wrote, so they read the Blob inside the page; the file a reader actually gets
 * is pinned by `file.test.mjs`, `keyboard.test.mjs` and `parity.test.mjs`.
 */
async function exportedMidi(driver) {
  const { bytes } = await exportedBytes(driver, "export");
  return bytes;
}

test("the reader recovers exactly what the page's writer wrote", async () => {
  // The live comparison below decodes MIDI; a reader that quietly dropped a
  // note would make two WRONG takes look identical. This pins it to the real
  // writer, note for note, with no browser in between.
  const { writeMidi } = await import("../midi.js");
  const written = writeMidi([
    { start: 0.25, end: 1.0, midi: 69, strength: 0.6 },
    { start: 1.5, end: 2.0, midi: 72, strength: 0.4 },
    // A re-attack of the first pitch exactly where it ends: the segmenter
    // clamps a note's end to the next start, so this shared tick is the normal
    // shape of two adjacent takes of one pitch.
    { start: 1.0, end: 1.4, midi: 69, strength: 0.6 },
  ]);
  const notes = notesOf(written);
  assert.deepEqual(
    notes.map((note) => `${note.pitch}@${note.start.toFixed(2)}+${note.duration.toFixed(2)}`),
    ["69@0.25+0.75", "69@1.00+0.40", "72@1.50+0.50"],
    "a note lost its length at a shared tick",
  );
  // One tick is 1.04 ms: the writer's rounding, not a different note.
  assert.ok(Math.abs(notes[0].start - 0.25) < 0.002, `starts at ${notes[0].start}`);
});

test("the demo take exports the same MIDI twice", async (t) => {
  // The export end of the capture path: one take, the same bytes twice, through
  // the export the microphone test uses.
  const driver = await page();
  await pressDemo(driver);

  const first = await exportedMidi(driver);
  const second = await exportedMidi(driver);
  assert.equal(first.subarray(0, 4).toString("latin1"), "MThd", "not a MIDI file");
  // Byte equality is legal HERE and only here: the demo is fed frame by frame,
  // no microphone, no clock, no block can go missing between the two exports.
  assert.ok(first.equals(second), "two exports of one take differ");
  assert.ok(first.length > 14, "the export carries no events");
  const demo = notesOf(first);
  assert.ok(demo.length >= 3, `the demo transcription read back ${demo.length} notes`);
  assert.ok(demo.every((note) => note.pitch >= 21 && note.pitch <= 108), "a pitch left the piano");
  assert.ok(demo.every((note) => note.duration > 0), "a note has no length");
  // The demo path writes `#note` too — not while `feedTake` segments, but
  // through the PLAYBACK the demo button starts: `tickPlayhead` names the note
  // under the playhead and EMPTIES the Hz line, because no frame is being
  // measured there. Both slots come from ONE snapshot, so a page that dropped
  // one of them cannot pass: a name from another take or a hertz line a live
  // capture left behind shows here. It needs the page's audio graph — without
  // one the playhead never starts — and this file reports that rather than
  // waiting on it.
  const graph = await audioState(driver);
  if (graph !== "running") {
    const why = `no audio output device (context ${graph})`;
    if (process.env.V2M_REQUIRE_AUDIO === "1") {
      assert.fail(`the demo readout did not run: ${why}`);
    }
    t.diagnostic(`the demo readout needs an audio output device: ${why}`);
  } else {
    const pitches = new Set(demo.map((note) => note.pitch));
    let shown = { note: "", hz: null };
    let named = false;
    for (let waited = 0; waited < 12000 && !named; waited += 100) {
      shown = await driver.evaluate(`({
        note: document.querySelector("#note").textContent.trim(),
        hz: document.querySelector("#hz").textContent.trim(),
      })`);
      const pitch = pitchOfName(shown.note);
      named = pitch !== null && pitches.has(pitch) && shown.hz === "";
      if (!named) await sleep(100);
    }
    assert.ok(
      named,
      `the demo readout showed "${shown.note}" beside "${shown.hz}" at its playhead`,
    );
  }
  // The URL is released too — kept alive past the click, not leaked. The page
  // waits a second; anything released sooner would be a download cancelled
  // before the browser read it.
  await sleep(1200);
  const blobs = await driver.evaluate("window.__exports");
  assert.ok(blobs.length >= 2, `the two exports created ${blobs.length} blob URL(s)`);
  const released = blobs.map((item) =>
    item.revokedAt === null ? "never" : Math.round(item.revokedAt - item.createdAt),
  );
  assert.deepEqual(
    released.filter((ms) => ms === "never" || ms < 900),
    [],
    `an export released its blob URL too early (ms after creation: ${released.join(", ")})`,
  );
});

test("the fake microphone opens the page's capture and two takes export the same MIDI", async (t) => {
  const driver = await page();
  const state = await audioState(driver);
  if (state !== "running") {
    // No output device: the page must SAY so — a graph that never starts never
    // settles, so the answer cannot be a status that waits on it.
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
    // `#rows` belongs to ONE take. The page that runs this test already holds
    // the demo's transcription from the export check above, and the stop guard
    // counts rows: a list that kept the previous take's lines would let a
    // capture pass with notes it never heard. The minimum is traced with an
    // observer instead of polled, so the empty state cannot be missed between
    // two reads.
    await traceRows(driver);
    await driver.evaluate("document.querySelector('#record').click()");
    let live = false;
    for (let waited = 0; waited < 15000 && !live; waited += 100) {
      live = await driver.evaluate("!document.querySelector('#stop').disabled");
      if (!live) await sleep(100);
    }
    assert.ok(live, "the page never reached a live capture");
    // The empty state must OCCUR while this take is being set up; whether it
    // lands before or after the stop action is enabled is an ordering the test
    // does not need to fix.
    let trace = null;
    for (let waited = 0; waited < 5000; waited += 100) {
      trace = await rowTrace(driver);
      if (trace.min === 0) break;
      await sleep(100);
    }
    assert.equal(
      trace.min,
      0,
      `#rows never emptied when take ${index} started (${trace.start} row(s) of the previous take stayed)`,
    );
    const status = await driver.evaluate("document.querySelector('#status').textContent.trim()");
    assert.ok(
      [STRINGS.fr.listening, STRINGS.en.listening].includes(status),
      `a live capture reads as "${status}"`,
    );
    // The live readout: `onBlock` writes the confirmed note AND the measured
    // hertz through one `setReadout` call, and empties both slots when the tone
    // stops. Both are read from ONE snapshot, while this take sounds, because
    // two protocol reads could straddle a frame and pair a note with the next
    // tone's hertz. The note is the confirmed plateau while the Hz line is the
    // frame being measured, so at an attack they can straddle a semitone: the
    // promise is that they end up naming the SAME tone, and a page that wrote
    // one slot and dropped the other (an empty live region, a name from the
    // previous take) never does.
    const pitchOfHz = (hz) => 69 + 12 * Math.log2(hz / 440);
    let readout = { note: "", hz: "" };
    let sameTone = false;
    for (let waited = 0; waited < 2000 && !sameTone; waited += 100) {
      readout = await driver.evaluate(`({
        note: document.querySelector("#note").textContent.trim(),
        hz: document.querySelector("#hz").textContent.trim(),
      })`);
      if (!/^\d+[.,]\d Hz$/.test(readout.hz)) {
        await sleep(100);
        continue;
      }
      const heardHz = Number(readout.hz.replace(",", ".").replace(" Hz", ""));
      const named = pitchOfName(readout.note);
      sameTone = named !== null && Math.abs(named - pitchOfHz(heardHz)) <= 1;
      if (!sameTone) await sleep(100);
    }
    assert.match(readout.hz, /^\d+[.,]\d Hz$/, `the live readout wrote "${readout.hz}" in the Hz line`);
    assert.ok(
      sameTone,
      `the live readout named "${readout.note}" for ${readout.hz}`,
    );
    // One note more than the file has tones: whatever the loop phase, the
    // extra note is the partial the stream may open on, so ALL three whole
    // tones have been heard by then.
    const wanted = TONES + 1;
    let heard = 0;
    for (let waited = 0; waited < 20000 && heard < wanted; waited += 100) {
      heard = await driver.evaluate("document.querySelectorAll('#rows tr').length");
      if (heard < wanted) await sleep(100);
    }
    assert.ok(heard >= wanted, `take ${index} heard ${heard} notes, not ${wanted}`);
    // An empty Hz line is the page saying the last tone has stopped: the 2.2 s
    // silence that follows is the only place a stop can land without closing a
    // note itself.
    let quiet = false;
    for (let waited = 0; waited < 4000 && !quiet; waited += 100) {
      quiet = (await driver.evaluate('document.querySelector("#hz").textContent.trim()')) === "";
      if (!quiet) await sleep(100);
    }
    assert.ok(quiet, `take ${index} never fell silent after ${heard} notes`);
    await driver.evaluate("document.querySelector('#stop').click()");
    await sleep(800);
    const rows = await driver.evaluate("document.querySelectorAll('#rows tr').length");
    assert.ok(rows >= wanted, `the stopped take closed ${rows} of ${wanted} notes`);
    return exportedMidi(driver);
  };

  /** The whole tone heard for each pitch: the LONGEST note at that pitch. */
  const whole = (notes) => {
    const byPitch = new Map();
    for (const note of notes) {
      byPitch.set(note.pitch, Math.max(byPitch.get(note.pitch) ?? 0, note.duration));
    }
    return byPitch;
  };
  // What the page PROMISES is the transcription: these three tones, each as
  // long as it sounded. It does NOT promise byte-identical files — the live
  // path stamps each block with the samples it has heard, so a stalled runner
  // shifts a tick or two while the take is still right. Byte equality stays
  // where it is guaranteed: the demo export above, which is fed frame by frame
  // with no clock in between.
  const raw = [notesOf(await take(1)), notesOf(await take(2))];
  const dump = raw.map(
    (notes) =>
      notes.map((note) => `${note.pitch}@${note.start.toFixed(3)}+${note.duration.toFixed(3)}`).join(" "),
  );
  const first = whole(raw[0]);
  const second = whole(raw[1]);
  // A capture that opened mid-tone also carries a SHORT partial of it: that is
  // the loop phase, not a reading, and the longest note per pitch is the whole
  // tone the comparison is about.
  const PITCHES = [69, 74, 77]; // the file's A4, D5, F5
  for (const [name, take_] of [
    ["take 1", first],
    ["take 2", second],
  ]) {
    assert.deepEqual(
      [...take_.keys()].sort((left, right) => left - right),
      PITCHES,
      `${name} transcribed ${[...take_.keys()].join(", ")} instead of the file's three tones`,
    );
  }
  const LIMIT_S = 0.06; // about five 11.6 ms hops: a lost block, not a wrong note
  for (const pitch of PITCHES) {
    const mine = first.get(pitch);
    const other = second.get(pitch);
    // Each tone lasts 0.8 s: the segmenter may add a release frame or two, so
    // the bound is a tenth of a second, not the millisecond.
    assert.ok(
      Math.abs(mine - 0.8) <= 0.12,
      `pitch ${pitch} lasted ${mine.toFixed(3)} s in take 1 (heard: ${dump[0]})`,
    );
    assert.ok(
      Math.abs(other - 0.8) <= 0.12,
      `pitch ${pitch} lasted ${other.toFixed(3)} s in take 2 (heard: ${dump[1]})`,
    );
    assert.ok(
      Math.abs(mine - other) <= LIMIT_S,
      `pitch ${pitch} lasts ${(mine - other).toFixed(3)} s longer in take 1`,
    );
  }
});
