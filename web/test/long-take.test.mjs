// What a LONG take costs the page, measured and held to a stated ceiling.
//
// Every other web check uses a take of a second or two. That is not the shape
// of a real session: the page keeps the whole signal (every captured block),
// and an export concatenates it once more before packing the archive. A leak or
// a second copy per export is invisible at two seconds and decisive at three
// minutes, on the phone the page is meant for.
//
// It is OPT-IN (`V2M_LONG_TAKE=1`) because it spends three minutes of real
// time: the fake device feeds the page in real time, exactly like a microphone.
// A push gate cannot pay that on every run, and a budget measured once per
// release answers the question the suite cannot.
//
// The number is the browser's OWN accounting
// (`performance.measureUserAgentSpecificMemory`), not `Runtime.getHeapUsage`:
// the take lives in Float32Array backing stores, which sit OUTSIDE the V8 heap
// (measured: a 32 MB Float32Array leaves the DevTools heap unchanged). That API
// demands a cross-origin-isolated page, so this caller serves the page with
// COOP/COEP — headers that change nothing the page does, and nothing else in
// `web/test/` asks for them.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { download, fakeMicrophone, sleep, startBrowser } from "./browser.mjs";
import { readArchive } from "./archive.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const RUN = process.env.V2M_LONG_TAKE === "1";

/** The low end of the 3–5 minute band: the gate costs three minutes, not five. */
const TAKE_SECONDS = 180;

// The signal a mono take holds is 4 bytes per sample (Float32): 34.6 MB at
// 48 kHz for three minutes, 69.1 MB at 96 kHz. The take is ONE copy while it is
// sung, with room for notes, the roll's canvas and the page itself: measured
// 0.405 MB/s over one minute and 0.399 over three, i.e. the blocks plus ~3 MB,
// and the browser's own breakdown puts the blocks in the JavaScript bucket
// (canvas 0.9 MB, shared 0.9 MB, the table's rows the only part that grows).
const CAPTURE_FACTOR = 1.05;
const SLACK_MB = 14;

// An export CONCATENATES the blocks to pack the archive, and must release that
// view once it is packed: the take stays playable and re-exportable from its
// own blocks. During the pack it holds blocks + view + a 16-bit WAV + the
// archive, so a sampler inside the page holds the peak under PEAK_FACTOR. The
// archive's bytes may still be on their way out while its download finishes,
// so the export is allowed them — never a second take.
const EXPORT_SLACK_MB = 12;

// An export holds the blocks, the concatenation it builds, the 16-bit WAV it
// writes and the compressed archive at once — about 2.5x the captured take,
// measured at one minute. The ceiling leaves room for a bigger archive and
// nothing for a fourth copy; the floor refuses a sampler that measured nothing.
const PEAK_FACTOR = 2.8;
const PEAK_FLOOR_FACTOR = 1.5;

// A device whose rate is not the one this machine happens to present:
// `V2M_LONG_TAKE_RATE=48000` opens every graph the page builds at that rate.
// The page reads its clock from `context.sampleRate` and its ceilings are
// expressed against the signal that rate produces, so the run becomes a 48 kHz
// run — which is what a phone (and most interfaces) actually reports. Without
// it the browser chooses (96 kHz on this machine), and an unsupported rate is
// refused loudly rather than measured as if it had been honoured.
const RATE = Number(process.env.V2M_LONG_TAKE_RATE || 0);

const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8"));

/** A 16-bit mono 44.1 kHz take: one second of A4, a quarter second of silence. */
function toneTake(rate = 44100) {
  const seconds = TAKE_SECONDS;
  const samples = Math.round(seconds * rate);
  const buffer = Buffer.allocUnsafe(44 + samples * 2);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + samples * 2, 4);
  buffer.write("WAVEfmt ", 8);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(1, 22); // mono
  buffer.writeUInt32LE(rate, 24);
  buffer.writeUInt32LE(rate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(samples * 2, 40);
  const period = Math.round(rate * 1.25);
  const tone = Math.round(rate * 1.0);
  for (let i = 0; i < samples; i += 1) {
    const inTone = i % period < tone;
    const value = inTone ? Math.sin((2 * Math.PI * 440 * i) / rate) * 0.35 : 0;
    buffer.writeInt16LE(Math.round(value * 32767), 44 + i * 2);
  }
  return buffer;
}

/**
 * The page's footprint in MB, as the browser accounts for it.
 *
 * Chrome may refuse a second reading inside its cooldown window; the reading IS
 * the budget, so the wait is paid once rather than guessed at.
 */
async function memoryMb(driver) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      const bytes = await driver.evaluate(
        "performance.measureUserAgentSpecificMemory().then((found) => found.bytes)",
      );
      return bytes / 1e6;
    } catch (error) {
      if (attempt === 1) throw error;
      await sleep(21000);
    }
  }
}

/** The take's transcription, read from the archive rather than from the page. */
function readingOf(archive) {
  const notes = archive.find((entry) => entry.name === "notes.json");
  assert.ok(notes, "the archive carries no notes.json");
  return JSON.parse(Buffer.from(notes.data).toString("utf8"));
}

test(
  "three minutes of capture and its archive stay inside the memory ceiling",
  { skip: RUN ? false : "opt-in: set V2M_LONG_TAKE=1 (three minutes of real-time capture)" },
  async (t) => {
    const work = await mkdtemp(join(tmpdir(), "v2m-long-take-"));
    const wav = join(work, "long-take.wav");
    await writeFile(wav, toneTake());

    const driver = await startBrowser({
      root: WEB,
      width: 1200,
      height: 900,
      // A muted browser opens no output stream, and a graph that cannot start
      // never captures: this check needs the page's audio path to run.
      mute: false,
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
      args: fakeMicrophone(wav),
    });
    try {
      assert.equal(
        await driver.evaluate("crossOriginIsolated"),
        true,
        "the page is not cross-origin isolated: the browser refuses its own memory accounting",
      );
      if (RATE) {
        await driver.evaluate(`(() => {
          const Real = window.AudioContext;
          window.AudioContext = class extends Real {
            constructor(options = {}) {
              super({ ...options, sampleRate: ${RATE} });
            }
          };
        })()`);
      }
      const sr = await driver.evaluate(`(async () => {
        const probe = new AudioContext();
        const rate = probe.sampleRate;
        await probe.close();
        return rate;
      })()`);
      const before = await memoryMb(driver);

      await driver.evaluate("document.querySelector('#record').click()");
      let live = false;
      for (let waited = 0; waited < 15000 && !live; waited += 100) {
        live = await driver.evaluate("!document.querySelector('#stop').disabled");
        if (!live) await sleep(100);
      }
      if (!live) {
        const status = await driver.evaluate("document.querySelector('#status').textContent.trim()");
        const why = `the capture never started (status: ${status})`;
        if (process.env.V2M_REQUIRE_AUDIO === "1") assert.fail(why);
        t.skip(`needs an audio output device: ${why}`);
        return;
      }

      await sleep(TAKE_SECONDS * 1000);
      await driver.evaluate("document.querySelector('#stop').click()");
      let stopped = "";
      for (let waited = 0; waited < 20000; waited += 200) {
        stopped = await driver.evaluate("document.querySelector('#status').textContent.trim()");
        if ([STRINGS.fr.stopped, STRINGS.en.stopped].includes(stopped)) break;
        await sleep(200);
      }
      assert.ok(
        [STRINGS.fr.stopped, STRINGS.en.stopped].includes(stopped),
        `the take never closed (status: "${stopped}")`,
      );
      const rows = await driver.evaluate("document.querySelectorAll('#rows tr').length");
      assert.ok(rows > 20, `three minutes of tones produced ${rows} notes`);

      await sleep(500);
      const afterCapture = await memoryMb(driver);

      // The archive's PEAK is invisible after the fact — the transient copies
      // are gone by the time a before/after pair is read — and it is what a
      // phone's tab must survive once. The sampler lives IN the page so the
      // copies cannot slip between two harness round trips, which is how a
      // 150 ms poll from here missed them entirely.
      // The outer function returns NOTHING: `evaluate` awaits a returned
      // promise, which would run every reading BEFORE the archive is clicked
      // and measure the take alone. The loop has to keep running beside the
      // click. Each reading walks the whole heap AND the ArrayBuffer stores, so
      // this is a BOUNDED set of readings, not a poll: a continuous loop starves
      // the compression it is meant to watch and the archive never appears.
      await driver.evaluate(`
        (() => {
          window.__peak = 0;
          window.__sampling = true;
          (async () => {
            for (let i = 0; i < 10 && window.__sampling; i += 1) {
              const found = await performance.measureUserAgentSpecificMemory();
              if (found.bytes > window.__peak) window.__peak = found.bytes;
              await new Promise((resolve) => setTimeout(resolve, 150));
            }
          })();
        })();
      `);
      const zip = await download(driver, "bundle", join(work, "archive"), ".zip", {
        timeoutMs: 120000,
      });
      const peakMb =
        (await driver.evaluate("(window.__sampling = false, window.__peak)")) / 1e6;
      const archive = readArchive(zip);
      const notes = readingOf(archive);
      // The budget is only meaningful if the take it measured is the long one:
      // a capture that died early would pass every ceiling below.
      assert.ok(
        notes.duration_s >= TAKE_SECONDS - 8,
        `the archive holds ${notes.duration_s} s, not ~${TAKE_SECONDS} s`,
      );
      assert.equal(notes.sample_rate_hz, sr, "the archive's rate is not the one the page recorded");
      if (RATE) {
        // A browser may hand back another rate or clamp the request: `sr` must
        // equal the forced rate, or the ceilings describe a take this run never
        // recorded.
        assert.equal(sr, RATE, `the browser opened ${sr} Hz, not the forced ${RATE} Hz`);
      }
      assert.ok(notes.notes.length > 20, `the archive holds ${notes.notes.length} notes`);

      // Give the blob a moment to leave: the budget is what the page KEEPS, not
      // what the download pipeline still holds while it finishes.
      await sleep(2000);
      const afterExport = await memoryMb(driver);

      const signalMb = (TAKE_SECONDS * sr * 4) / 1e6;
      const captureMb = afterCapture - before;
      const exportMb = afterExport - before;
      const captureCeiling = CAPTURE_FACTOR * signalMb + SLACK_MB;
      const exportCeiling = captureMb + EXPORT_SLACK_MB;
      const peak = peakMb - before;
      const peakCeiling = captureMb * PEAK_FACTOR;
      console.log(
        `long take: ${notes.notes.length} notes, ${notes.duration_s} s at ${sr} Hz, ` +
          `signal ${signalMb.toFixed(1)} MB; page ${before.toFixed(1)} MB, ` +
          `after capture +${captureMb.toFixed(1)} MB (ceiling ${captureCeiling.toFixed(1)}), ` +
          `export peak +${peak.toFixed(1)} MB (ceiling ${peakCeiling.toFixed(1)}), ` +
          `after archive +${exportMb.toFixed(1)} MB (ceiling ${exportCeiling.toFixed(1)})`,
      );

      // A green that measured nothing would pass forever: the take's own signal
      // must be there before any ceiling is worth checking.
      assert.ok(
        captureMb >= 0.8 * signalMb,
        `the take's signal is not in the measurement: +${captureMb.toFixed(1)} MB ` +
          `for a ${signalMb.toFixed(1)} MB signal`,
      );
      assert.ok(
        captureMb <= captureCeiling,
        `three minutes of capture hold +${captureMb.toFixed(1)} MB, past the ` +
          `${captureCeiling.toFixed(1)} MB ceiling (${CAPTURE_FACTOR} x the ` +
          `${signalMb.toFixed(1)} MB signal + ${SLACK_MB} MB)`,
      );
      assert.ok(
        peak >= PEAK_FLOOR_FACTOR * captureMb,
        `the archive's peak was never sampled: +${peak.toFixed(1)} MB for a ` +
          `${captureMb.toFixed(1)} MB take (floor ${(PEAK_FLOOR_FACTOR * captureMb).toFixed(1)} MB)`,
      );
      assert.ok(
        peak <= peakCeiling,
        `the archive peaks at +${peak.toFixed(1)} MB, past the ${peakCeiling.toFixed(1)} MB ` +
          `ceiling (${PEAK_FACTOR} x the captured take): a third copy of the signal ` +
          "is being held while it packs",
      );
      assert.ok(
        exportMb <= exportCeiling,
        `the take plus its archive hold +${exportMb.toFixed(1)} MB, past the ` +
          `${exportCeiling.toFixed(1)} MB ceiling (the captured take's ` +
          `${captureMb.toFixed(1)} MB + ${EXPORT_SLACK_MB} MB of archive still ` +
          "leaving): an export that keeps the concatenation it packed holds a " +
          "fourth copy",
      );
    } finally {
      await driver.close();
    }
  },
);
