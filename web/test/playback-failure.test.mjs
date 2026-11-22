// A machine with no output device must SAY so when the take is asked to play.
//
// `AudioContext.resume()` settles only once an output stream can start. A
// headless browser, or a machine whose audio service is off, leaves the promise
// pending for good: an unbounded await leaves the page on its preparing-playback
// sentence with no clock, no sound and no failure. The capture path carries the
// same bound; this file is the playback half.
//
// It needs no sound card of its own: the dead graph is built in the page —
// `state` reads `suspended` forever and `resume()` never settles — which is
// exactly what a browser without an output stream does. The page must give up
// after its bound and name `error_play`, not sit on "preparing".
//
// The bound is a WAIT, not a verdict: a slow output (a Bluetooth headset, a
// phone waking its audio service) opens after a second or more and must still
// play. The second test opens after 1.2 s, so shortening the bound below that
// fails here instead of on the device of the one singer whose headset is slow.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { chooseFile, startBrowser, sleep } from "./browser.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const TAKE = fileURLToPath(new URL("../../tests/data/parity.wav", import.meta.url));
const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8")).fr;

/** The five-note take, read and tabled before the output is asked for. */
async function loadParity(driver) {
  await chooseFile(
    driver,
    TAKE,
    async () => (await driver.evaluate("document.querySelectorAll('#rows tr').length")) === 5,
    // The helper's one clean document for a stalled decode is what a just-closed
    // browser's held output device needs; the retried load still has to table
    // all five notes.
    { timeoutMs: 12000 },
  );
}

test("a playback graph that never starts is reported instead of awaited forever", async () => {
  const driver = await startBrowser({ root: WEB, width: 1200, height: 900 });
  try {
    await loadParity(driver);

    // The dead output, in the page: no device to start, and a resume that never
    // answers — the two facts a runner without a sound card presents together.
    await driver.evaluate(`
      Object.defineProperty(AudioContext.prototype, "state", {
        configurable: true,
        get: () => "suspended",
      });
      AudioContext.prototype.resume = () => new Promise(() => {});
    `);

    const started = Date.now();
    await driver.evaluate("document.querySelector('#listen').click()");
    await sleep(200); // the page says it is preparing first
    assert.equal(
      await driver.evaluate("document.querySelector('#status').textContent.trim()"),
      STRINGS.preparing,
      "the page never said it was preparing the playback",
    );

    let status = "";
    for (let waited = 0; waited < 6000; waited += 100) {
      status = await driver.evaluate("document.querySelector('#status').textContent.trim()");
      if (status === STRINGS.error_play) break;
      await sleep(100);
    }
    assert.equal(
      status,
      STRINGS.error_play,
      `the page stayed on "${status}": a dead output is not named`,
    );
    const waited = Date.now() - started;
    // A device that needs a moment still gets its chance: the bound is a wait,
    // not an instant refusal. Measured ~2 s (the capture path's own bound).
    assert.ok(waited >= 1800, `the page gave up after ${waited} ms, before its bound`);
    assert.ok(waited < 6000, `the page waited ${waited} ms, longer than its bound`);

    // The console names the step that was running and the graph's own state, so
    // a report of a dead output carries the machine's facts.
    const reason = driver.notes().filter((note) => /^error: playback failed/.test(note));
    assert.equal(
      reason.length,
      1,
      `the console never said why the playback failed: ${driver.notes().join(" | ")}`,
    );
    assert.match(
      reason[0],
      /^error: playback failed \[audio engine\]: AudioContext suspended; Error: the bound expired before the output started$/,
      `the console line is not readable: ${reason[0]}`,
    );

    // Nothing is left half-started: no clock, and the button says again what it
    // will do.
    assert.equal(
      await driver.evaluate("document.querySelector('#clock').hidden"),
      true,
      "the clock stayed after the playback failed",
    );
    assert.equal(
      await driver.evaluate("document.querySelector('#listen').textContent.trim()"),
      STRINGS.listen,
      "the listening button still reads as if the take were sounding",
    );
    const said = driver.notes().filter((note) => /^thrown/.test(note));
    assert.deepEqual(said, [], `the page threw: ${said.join(" | ")}`);
  } finally {
    await driver.close();
  }
});

test("a slow output still plays instead of being written off", async () => {
  const driver = await startBrowser({ root: WEB, width: 1200, height: 900 });
  try {
    await loadParity(driver);

    // A real `resume()`, answered after 1.2 s: the graph reads `suspended`
    // until it settles, like an output that takes a moment to open (a headset
    // pairing, a phone waking its audio service). The bound is a wait, so this
    // must play — shortening it under the delay fails HERE, not on the device.
    await driver.evaluate(`
      const realResume = AudioContext.prototype.resume;
      Object.defineProperty(AudioContext.prototype, "state", {
        configurable: true,
        get() {
          return this.__open ? "running" : "suspended";
        },
      });
      AudioContext.prototype.resume = function resume() {
        return new Promise((done) => {
          setTimeout(() => {
            this.__open = true;
            done(realResume.call(this));
          }, 1200);
        });
      };
    `);

    const started = Date.now();
    await driver.evaluate("document.querySelector('#listen').click()");
    let status = "";
    for (let waited = 0; waited < 6000; waited += 100) {
      status = await driver.evaluate("document.querySelector('#status').textContent.trim()");
      if (status === STRINGS.playing) break;
      await sleep(100);
    }
    const waited = Date.now() - started;
    assert.equal(status, STRINGS.playing, `the page left the slow output on "${status}"`);
    assert.ok(waited >= 1100, `playback began after ${waited} ms, before the output even opened`);
    assert.equal(
      await driver.evaluate("document.querySelector('#clock').hidden"),
      false,
      "the clock never appeared on a slow output",
    );
    assert.equal(
      await driver.evaluate("document.querySelector('#listen').textContent.trim()"),
      STRINGS.listen_stop,
      "the listening button does not read as sounding",
    );
    const said = driver.notes().filter((note) => /^thrown/.test(note));
    assert.deepEqual(said, [], `the page threw: ${said.join(" | ")}`);
  } finally {
    await driver.close();
  }
});
