// The page's main promises, replayed in the engines a singer actually holds.
//
// `web/test/*.test.mjs` drives ONE browser — the runner's Chrome — and the
// page's worst bug hid outside it: WebKit has no `Symbol.asyncIterator` on
// `ReadableStream`, so the archive's compressor threw, every entry fell back to
// STORED, the ZIP stayed valid at 1.5× the size, and a green Chromium said
// nothing. Each engine here replays the five checks that matter and reads every
// result from the page itself — the last of them against the DESKTOP's own
// reading of one sung phrase, not against a fixture of pitch frames.
//
// The driver is borrowed by the workflow, so this file lives in `tests/` rather
// than in `web/`: the page ships as a dependency-free folder.
//
//   node tests/web_matrix/run.mjs chromium|firefox|webkit
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium, firefox, webkit } from "playwright";

import { wavBytes } from "../../web/bundle.js";
import { DEMO_NOTES } from "../../web/demo.js";
import { readArchive } from "../../web/test/archive.mjs";
import { servePage } from "../../web/test/browser.mjs";
import { PARITY_GOLDEN, PARITY_TAKE, assertParity } from "../../web/test/parity.mjs";

const WEB = fileURLToPath(new URL("../../web", import.meta.url));
const ENGINES = { chromium, firefox, webkit };
const engine = process.argv[2];
if (!(engine in ENGINES)) {
  console.error(`usage: node tests/web_matrix/run.mjs ${Object.keys(ENGINES).join("|")}`);
  process.exit(2);
}

const WORK = await mkdtemp(join(tmpdir(), `v2m-${engine}-`));

/** A4, a gap, D5 as mono floats: two notes any YIN finds. */
function toneSamples(rate = 44100) {
  const parts = [
    [440, 0.8],
    [0, 0.2],
    [587.33, 0.8],
  ];
  const seconds = parts.reduce((total, [, held]) => total + held, 0);
  const samples = new Float32Array(Math.round(seconds * rate));
  let at = 0;
  for (const [hz, held] of parts) {
    for (let i = 0; i < Math.round(held * rate); i += 1) {
      samples[at + i] = hz === 0 ? 0 : Math.sin((2 * Math.PI * hz * i) / rate) * 0.35;
    }
    at += Math.round(held * rate);
  }
  return samples;
}

/** Waits for a reading to land and for both exports to open. */
async function waitForTake(page) {
  await page.waitForFunction(
    () =>
      document.querySelectorAll("#rows tr").length > 0 &&
      !document.querySelector("#bundle").disabled &&
      !document.querySelector("#export").disabled,
    null,
    { timeout: 30000 },
  );
}

/** Clicks a download button and answers the bytes the page wrote. */
async function downloaded(page, button) {
  const [download] = await Promise.all([page.waitForEvent("download"), page.click(`#${button}`)]);
  const path = join(WORK, `${button}-${Date.now()}`);
  await download.saveAs(path);
  return readFile(path);
}

/**
 * The ported segmenter against the desktop's conformance cases.
 *
 * Same fixtures (`cases.json`, produced by the Python engine), same call, but
 * the decisions are computed HERE, by this engine's numbers — which is the
 * point: `Math.log2`, `Math.round` (ties!) and float parsing are the parts of a
 * port an engine can change under you.
 */
async function segmenter(page) {
  const report = await page.evaluate(async () => {
    const payload = await (await fetch("/test/cases.json")).json();
    const { LiveNoteSegmenter } = await import("/segmenter.js");
    const problems = [];
    for (const item of payload.cases) {
      const seg = new LiveNoteSegmenter(item.params);
      const events = [];
      let lastT = 0;
      for (const [t, hz, attack, strength, weak, quality] of item.frames) {
        lastT = t;
        events.push(
          seg.push(t, hz === null ? NaN : hz, {
            attack: attack !== 0,
            strength,
            weak: weak !== 0,
            quality: quality === null ? NaN : quality,
          }),
        );
      }
      if (item.flush) seg.flush(lastT + payload.frame_s);
      if (events.length !== item.events.length) {
        problems.push(`${item.name}: ${events.length} frames decided, ${item.events.length} expected`);
      }
      if (seg.notes.length !== item.notes.length) {
        problems.push(`${item.name}: ${seg.notes.length} notes, ${item.notes.length} expected`);
        continue;
      }
      seg.notes.forEach((note, index) => {
        const want = item.notes[index];
        for (const field of ["start", "end", "midi", "strength", "dev"]) {
          if (Math.abs(note[field] - want[field]) > 1e-6) {
            problems.push(`${item.name} note ${index}: ${field} ${note[field]} != ${want[field]}`);
          }
        }
        if (note.articulated !== want.articulated) {
          problems.push(`${item.name} note ${index}: articulated differs`);
        }
      });
    }
    return { cases: payload.cases.length, problems };
  });
  assert.deepEqual(report.problems, [], `the port diverges in ${engine}:\n  ${report.problems.join("\n  ")}`);
  return `${report.cases} desktop cases agree`;
}

/**
 * YIN at the live geometry, on the demo's own synthesis.
 *
 * The held pitch is compared to what the phrase SINGS (C4, deliberately eight
 * cents sharp), never to another engine's run of the same module: the samples
 * come from `Math.sin`, which every engine may round differently, so a tight
 * agreement with ourselves would prove nothing a note does not.
 */
async function yin(page) {
  const reading = await page.evaluate(async () => {
    const { YinBlock } = await import("/yin.js");
    const { median } = await import("/segmenter.js");
    const { DEMO_SR, demoTake } = await import("/demo.js");
    const block = new YinBlock(DEMO_SR, { window: 1024, hop: 256 });
    const frames = block.pushWithQuality(demoTake()).filter((f) => Number.isFinite(f.hz));
    const pitches = frames.map((f) => f.hz);
    return {
      hz: median(frames.filter((f) => f.t > 0.15 && f.t < 0.45).map((f) => f.hz)),
      low: Math.min(...pitches),
      high: Math.max(...pitches),
    };
  });
  const note = DEMO_NOTES[0];
  assert.ok(reading.hz, "the engine's YIN found no pitch in the demo's first note");
  const sung = 440 * 2 ** ((note.midi - 69) / 12) * 2 ** (note.cents / 1200);
  const cents = Math.abs(1200 * Math.log2(reading.hz / sung));
  assert.ok(
    cents < 10,
    `the first note reads ${reading.hz.toFixed(2)} Hz in ${engine}, ${cents.toFixed(1)} cents from the ${sung.toFixed(2)} Hz it sings`,
  );
  assert.ok(
    [reading.low, reading.high].every((hz) => hz > 200 && hz < 600),
    `a frame left the demo's octave: ${reading.low.toFixed(0)}..${reading.high.toFixed(0)} Hz`,
  );
  return `${reading.hz.toFixed(2)} Hz, ${cents.toFixed(2)} cents from the sung C4`;
}

/**
 * The archive — the check WebKit's bug went through.
 *
 * An engine that ADVERTISES `CompressionStream("deflate-raw")` must be seen
 * USING it: the page's fallback is silent by design (a compressor that throws
 * costs the compression, never the archive), so only the METHOD can tell
 * "stored because it does not shrink" from "stored because the stream could not
 * be read" — which is exactly the WebKit failure that shipped.
 */
async function bundle(page) {
  await page.click("#demo");
  await waitForTake(page);
  const deflates = await page.evaluate(() => {
    try {
      new CompressionStream("deflate-raw");
      return true;
    } catch {
      return false;
    }
  });
  const archive = readArchive(await downloaded(page, "bundle"));
  assert.deepEqual(
    archive.map((entry) => entry.name).slice(0, 3),
    ["prise.wav", "notes.json", "prise.mid"],
    "the archive's order moved",
  );
  assert.equal(
    Buffer.from(archive[0].data.subarray(0, 4)).toString("latin1"),
    "RIFF",
    "the archive's WAV is not a WAV",
  );
  if (deflates) {
    assert.equal(
      archive[0].method,
      8,
      `${engine} advertises deflate-raw and the page stored the audio anyway — the stream was not readable`,
    );
  }
  const { notes } = JSON.parse(Buffer.from(archive[1].data).toString("utf8"));
  assert.deepEqual(
    notes.map((note) => note.midi),
    DEMO_NOTES.map((note) => note.midi),
    "the demo's phrase was read differently in this engine",
  );
  return `${archive.length} entries, audio ${deflates ? "deflated" : "stored"} (engine ${deflates ? "supports" : "lacks"} deflate-raw)`;
}

/** A file the reader chooses, decoded by this engine's own `decodeAudioData`. */
async function file(page) {
  // A COLD page: the demo take above already opened both exports, so a check
  // that only waits for them would read the previous take (measured in Firefox,
  // where the file had not landed yet and the archive still said "demo").
  await page.reload();
  await page.waitForSelector("#demo");
  const take = join(WORK, "prise.wav");
  await writeFile(take, wavBytes(toneSamples(), 44100));
  await page.setInputFiles("#file", take);
  await waitForTake(page);
  const archive = readArchive(await downloaded(page, "bundle"));
  const document = JSON.parse(Buffer.from(archive[1].data).toString("utf8"));
  assert.equal(document.source, "file", `the archive names its source "${document.source}"`);
  assert.deepEqual(
    document.notes.map((note) => note.midi),
    [69, 74],
    "the chosen file's two notes were read differently in this engine",
  );
  assert.ok(
    Math.abs(document.duration_s - 1.8) < 0.05,
    `the decoded take lasts ${document.duration_s} s instead of the file's 1.8`,
  );
  return `${document.notes.length} notes read at ${document.sample_rate_hz} Hz`;
}

/**
 * A SUNG phrase, read by this engine and by the desktop's own live chain.
 *
 * `file` proves the path on two tones any YIN can find; this is the one check
 * that holds the two IMPLEMENTATIONS to one reading of one piece of audio —
 * glide, re-attack on one pitch and a breath included. The golden is the
 * Python chain's (`scripts/export_web_fixtures.py`), never the page's.
 */
async function parity(page) {
  await page.reload();
  await page.waitForSelector("#demo");
  await page.setInputFiles("#file", PARITY_TAKE);
  await waitForTake(page);
  const archive = readArchive(await downloaded(page, "bundle"));
  const entry = archive.find((item) => item.name === "notes.json");
  assert.ok(entry, `the archive holds ${archive.map((item) => item.name).join(", ")}`);
  const document = JSON.parse(Buffer.from(entry.data).toString("utf8"));
  assert.equal(document.source, "file", `the archive names its source "${document.source}"`);
  return assertParity(document, JSON.parse(await readFile(PARITY_GOLDEN, "utf8")));
}

const browser = await ENGINES[engine].launch();
const server = await servePage(WEB);
const origin = `http://127.0.0.1:${server.address().port}/`;
const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 900 } });
const page = await context.newPage();
const trouble = [];
page.on("pageerror", (error) => trouble.push(`thrown: ${error.message}`));
page.on("console", (message) => {
  if (message.type() === "error") trouble.push(`error: ${message.text()}`);
});

const results = [];
try {
  await page.goto(origin);
  await page.waitForSelector("#demo");

  for (const [label, run] of [
    ["segmenter", segmenter],
    ["yin", yin],
    ["bundle", bundle],
    ["file", file],
    ["parity", parity],
  ]) {
    try {
      results.push(`${label.padEnd(9)} ${await run(page)}`);
    } catch (error) {
      results.push(`${label.padEnd(9)} FAILED: ${error.message}`);
      process.exitCode = 1;
    }
  }
} finally {
  await context.close();
  await browser.close();
  await new Promise((closed) => server.close(closed));
}

console.log(engine);
for (const line of results) console.log(`  ${line}`);
if (trouble.length) {
  console.log(`  page said:\n    ${trouble.join("\n    ")}`);
  process.exitCode = 1;
}
