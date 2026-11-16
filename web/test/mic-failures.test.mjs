// What the page SAYS when it cannot listen, on screen and in the console.
//
// `test_a_failed_capture_names_its_stage_and_the_error_in_the_console` pins the
// stage ordering as source text; this file presses the button and reads both
// surfaces the reader and a bug report have: `#status`, and the
// `capture failed [stage]: name: message` line. The audio graph is stubbed on
// purpose — a runner without a sound card never starts a real `AudioContext` —
// while what the page SAYS has nothing to do with the hardware: the browser's
// own error name, the stage that was running, or the page's secure-context
// test decides it.
//
// The stage is the fact the sentence cannot carry: a permission refusal and a
// worklet whose module never loaded are both `NotAllowedError`, and only the
// line between the brackets tells which half of the page broke.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { networkInterfaces } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const { startBrowser, servePage, sleep } = await import("./browser.mjs");

const WEB = fileURLToPath(new URL("..", import.meta.url));
const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8"));

// The three refusals a reader can act on. An unmapped name keeps the neutral
// sentence, and the Python suite pins that fallback in the source.
const REFUSALS = [
  ["NotAllowedError", "error_mic_blocked"],
  ["NotFoundError", "error_mic_missing"],
  ["NotReadableError", "error_mic_busy"],
];

/**
 * A page whose audio graph is up and whose microphone rejects with `name`.
 *
 * Only the graph is faked: `getUserMedia` is the browser's own property being
 * replaced, so the page runs its real `startCapture` — guard, context, worklet,
 * request, catch — and only the answer changes.
 */
const refusal = (name) => `
  window.AudioContext = class {
    constructor() {
      this.state = "running";
      this.sampleRate = 44100;
      this.destination = {};
      this.audioWorklet = { addModule: async () => {} };
    }
    createMediaStreamSource() { return { connect() {} }; }
    close() { return Promise.resolve(); }
  };
  window.AudioWorkletNode = class {
    constructor() { this.port = { onmessage: null }; }
    connect() {}
    disconnect() {}
  };
  Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
    configurable: true,
    value: () => Promise.reject(new DOMException("injected refusal", ${JSON.stringify(name)})),
  });
`;

/**
 * An output that never starts, behind a microphone that must never be asked.
 *
 * `resume()` rejects the way a closed context does, so the page gives up at
 * once instead of spending its two-second bound — and the stream request stays
 * unspent, which is the point: a dead output is not a permission problem.
 */
const deadEngine = `
  window.__asked = false;
  window.AudioContext = class {
    constructor() { this.state = "suspended"; this.sampleRate = 44100; this.destination = {}; }
    resume() { return Promise.reject(new DOMException("closed", "InvalidStateError")); }
    close() { return Promise.resolve(); }
  };
  window.AudioWorkletNode = class {
    constructor() { this.port = {}; }
    connect() {}
    disconnect() {}
  };
  Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
    configurable: true,
    value: () => { window.__asked = true; return Promise.reject(new DOMException("asked", "NotAllowedError")); },
  });
`;

/** A graph that runs, whose worklet module refuses — the other `NotAllowedError`. */
const brokenWorklet = `
  window.AudioContext = class {
    constructor() {
      this.state = "running";
      this.sampleRate = 44100;
      this.destination = {};
      this.audioWorklet = {
        addModule: () => Promise.reject(new DOMException("injected", "NotAllowedError")),
      };
    }
    close() { return Promise.resolve(); }
  };
  window.AudioWorkletNode = class {
    constructor() { this.port = {}; }
    connect() {}
    disconnect() {}
  };
`;

let driver = null;
let insecureServer = null;

async function page() {
  driver ??= await startBrowser({ root: WEB, width: 1200, height: 900 });
  return driver;
}

test.after(async () => {
  if (driver) await driver.close();
  if (insecureServer) await new Promise((done) => insecureServer.close(done));
});

/** The page is ready when its translated words are there, not when it loads. */
async function loaded(driver) {
  for (let waited = 0; waited < 20000; waited += 100) {
    const ready = await driver.evaluate(
      "Boolean(document.querySelector('#theme')?.textContent?.trim())",
    );
    if (ready) return;
    await sleep(100);
  }
  throw new Error("the page never translated");
}

/** Presses Sing and answers what the reader sees and what the console kept. */
async function sing(driver) {
  const before = driver.notes().length;
  await driver.evaluate("document.querySelector('#record').click()");
  await sleep(600);
  const shown = JSON.parse(
    await driver.evaluate(`JSON.stringify({
      status: document.querySelector("#status").textContent.trim(),
      lang: document.documentElement.lang || "fr",
    })`),
  );
  const language = shown.lang in STRINGS ? shown.lang : "fr";
  return { status: shown.status, language, said: driver.notes().slice(before).join("\n") };
}

test("each refusal shows its own sentence and names itself in the console", async () => {
  const driver = await page();
  for (const [name, key] of REFUSALS) {
    await driver.send("Page.navigate", { url: driver.origin });
    await loaded(driver);
    await driver.evaluate(refusal(name));
    const { status, language, said } = await sing(driver);
    assert.equal(status, STRINGS[language][key], `${name} reads as "${status}", not ${key}`);
    // The stage comes first because one name means two things here: this is the
    // microphone's refusal, and the line must say so without being read as a
    // worklet that never loaded.
    assert.match(
      said,
      new RegExp(`capture failed \\[microphone\\]: ${name}:`),
      `${name} left no console line: ${said || "(nothing)"}`,
    );
  }
});

test("a dead output and a broken worklet name their own stage", async () => {
  const driver = await page();
  const cases = [
    // The stage the page reports, the name it must carry, the sentence it must
    // show, and the script that produces it.
    { stage: "audio engine", said: "AudioContext did not start", key: "error_mic", script: deadEngine },
    { stage: "worklet", said: "NotAllowedError", key: "error_mic_blocked", script: brokenWorklet },
  ];
  for (const one of cases) {
    await driver.send("Page.navigate", { url: driver.origin });
    await loaded(driver);
    await driver.evaluate(one.script);
    const { status, language, said } = await sing(driver);
    assert.equal(status, STRINGS[language][one.key], `${one.stage} reads as "${status}"`);
    assert.match(
      said,
      new RegExp(`capture failed \\[${one.stage}\\]: ${one.said}`),
      `${one.stage} left no line of its own: ${said || "(nothing)"}`,
    );
    if (one.stage === "audio engine") {
      // The engine branch returns before the stream request: the console line
      // must exist without a permission prompt the reader would see for nothing.
      assert.equal(await driver.evaluate("window.__asked"), false, "a dead output asked for a microphone");
    }
  }
});

test("a page outside a secure context says so, and the console says why", async (t) => {
  // The one refusal no stub can produce: the browser hides
  // `navigator.mediaDevices` on plain http, so the page must be SERVED from an
  // address that is not loopback. Chrome treats 127.0.0.1 as secure, which is
  // exactly why every other check here can run on it.
  const lan = Object.values(networkInterfaces())
    .flat()
    .find((entry) => entry.family === "IPv4" && !entry.internal);
  if (!lan) {
    t.skip("no non-loopback IPv4 address to serve a plain-http origin from");
    return;
  }
  const driver = await page();
  insecureServer = await servePage(WEB, "0.0.0.0");
  await driver.send("Page.navigate", {
    url: `http://${lan.address}:${insecureServer.address().port}/`,
  });
  await loaded(driver);
  const { status, language, said } = await sing(driver);
  assert.equal(status, STRINGS[language].error_insecure, `an insecure page reads as "${status}"`);
  assert.match(said, /capture failed \[microphone\]: insecure context:/, `no console line: ${said}`);
  // The permission is not the problem, and saying it were would send the reader
  // to a setting that cannot change anything.
  assert.notEqual(status, STRINGS[language].error_mic_blocked);
});
