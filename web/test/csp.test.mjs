// The page's own Content-Security-Policy, and the proof it refuses nothing the
// page needs.
//
// The page is a static folder with no CDN, no inline script and no eval, so the
// policy can be the narrow one: everything from `'self'`, plus the ONE `blob:`
// the audio worklet is built from in memory (`app.js`). A policy that is too
// wide is a promise nobody checked; a policy that is too narrow breaks the page
// in the one place a test that only reads the DOM never looks. So the same file
// does both halves: it pins the exact directives, and it DRIVES the page — a
// take from the demo, a played passage, an archive, and the real capture with
// its worklet — while the page records every `securitypolicyviolation` and the
// browser's own log is read for a refusal.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  audioState,
  download,
  fakeMicrophone,
  pressDemo,
  rowTrace,
  sleep,
  startBrowser,
  traceRows,
} from "./browser.mjs";
import { tonesWav } from "./tone.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const HTML = await readFile(join(WEB, "index.html"), "utf8");
const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8")).fr;

/** The one policy the page declares, as its `content` attribute spells it. */
function declaredPolicy() {
  const metas = [...HTML.matchAll(/<meta\b[^>]*http-equiv="Content-Security-Policy"[^>]*>/gi)];
  assert.equal(metas.length, 1, `the page declares ${metas.length} CSP metas, not one`);
  const head = HTML.slice(0, HTML.search(/<\/head>/i));
  assert.ok(head.includes(metas[0][0]), "the CSP meta sits after </head>: it would govern nothing");
  const content = /content="([^"]*)"/i.exec(metas[0][0]);
  assert.ok(content, "the CSP meta has no content attribute");
  return content[1];
}

/** A refusal as the BROWSER logged it (the page keeps its own record too). */
function loggedRefusals(driver) {
  return driver
    .logs()
    .filter((entry) => /content security policy|refused to (load|execute|apply|create)/i.test(entry.text))
    .map((entry) => `${entry.source}: ${entry.text}`);
}

const work = await mkdtemp(join(tmpdir(), "v2m-csp-"));
const TONE = join(work, "tone.wav");
await writeFile(TONE, tonesWav({ parts: [{ hz: 440, seconds: 1.2 }, { hz: 0, seconds: 0.2 }] }));

let driver = null;
async function page() {
  driver ??= await startBrowser({
    root: WEB,
    width: 1200,
    height: 900,
    // The capture half opens the page's audio graph with the worklet: a muted
    // browser never starts one, and then the `blob:` being tested never loads.
    mute: false,
    args: fakeMicrophone(TONE),
  });
  return driver;
}

test.after(async () => {
  if (driver) await driver.close();
});

test("the policy is 'self' plus one blob:, and nothing wider", () => {
  const policy = declaredPolicy();
  for (const directive of [
    "default-src 'self'",
    "script-src 'self' blob:",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'none'",
  ]) {
    assert.ok(policy.includes(directive), `the policy lost "${directive}": ${policy}`);
  }
  assert.ok(
    !/unsafe-inline|unsafe-eval/i.test(policy),
    `the policy allows an escape hatch: ${policy}`,
  );
  assert.ok(!policy.includes("*"), `the policy allows a wildcard: ${policy}`);
  assert.ok(!/https?:/i.test(policy), `the policy allows a remote origin: ${policy}`);

  // The reason it can stay this narrow: the head carries NO inline script. An
  // inline copy would need 'unsafe-inline' or a hash that every edit silently
  // invalidates, so the theme's pre-paint script lives in its own file.
  const head = HTML.slice(0, HTML.search(/<\/head>/i));
  const inline = [...head.matchAll(/<script\b(?![^>]*\bsrc=)[^>]*>/gi)];
  assert.equal(inline.length, 0, "an inline script in the head needs a policy exception");
  assert.match(head, /<script src="theme-init\.js"><\/script>/);
  assert.ok(existsSync(join(WEB, "theme-init.js")), "the head names a script that is not there");
});

test("nothing the page loads or exports is refused by its own policy", async () => {
  const page_ = await page();
  await sleep(400);
  assert.deepEqual(await page_.violations(), [], "the page refused something while it opened");

  // The busiest device-free path: the demo (a take, a roll, a status), one
  // played passage, and the archive built from a blob: URL.
  await pressDemo(page_);
  await page_.evaluate("document.querySelector('#listen').click()");
  await sleep(500);
  await page_.evaluate("document.querySelector('#listen').click()");
  const zip = await download(page_, "bundle", join(work, "archive"), ".zip");
  assert.ok(zip.length > 0, "the archive came back empty");
  assert.deepEqual(await page_.violations(), [], "the page refused a take, a playback or an export");

  const logged = loggedRefusals(page_);
  assert.deepEqual(logged, [], `the browser logged a refusal: ${logged.join(" | ")}`);
});

test("the capture worklet's in-memory blob: module is allowed", async (t) => {
  const page_ = await page();
  const state = await audioState(page_);
  if (state !== "running") {
    const why = `no audio output device (context ${state})`;
    if (process.env.V2M_REQUIRE_AUDIO === "1") assert.fail(`the worklet half did not run: ${why}`);
    t.skip(`needs an audio output device: ${why}`);
    return;
  }

  // "Stop" only un-disables after the worklet module LOADED and the stream
  // opened (`startCapture`), so an enabled Stop is the proof that the policy
  // let the blob: through — not that the path was avoided. The table already
  // holds the demo's transcription, so its traces are watched from here: the
  // take's own notes are what the check below is about.
  await traceRows(page_);
  await page_.evaluate("document.querySelector('#record').click()");
  let live = false;
  for (let waited = 0; waited < 15000 && !live; waited += 100) {
    live = await page_.evaluate("!document.querySelector('#stop').disabled");
    if (!live) await sleep(100);
  }
  assert.ok(live, "the capture never reached the worklet");
  await sleep(1200);
  assert.equal(
    await page_.evaluate("document.querySelector('#status').textContent.trim()"),
    STRINGS.listening,
    "the capture is not open while the fake microphone sings",
  );
  await page_.evaluate("document.querySelector('#stop').click()");
  // Stop is ASYNC on the page: it closes the audio graph, THEN flushes the take
  // and redraws the table. Reading the rows after a fixed wait answers before
  // the page finished on a loaded machine, and the take looks empty though
  // nothing was wrong. Wait for the page's own facts: the record action enabled
  // again (the stop completed) with a filled table.
  let rows = 0;
  for (let waited = 0; waited < 10000; waited += 100) {
    const state = await page_.evaluate(`({
      closed: !document.querySelector("#record").disabled,
      rows: document.querySelectorAll("#rows tr").length,
    })`);
    rows = state.rows;
    if (state.closed && rows > 0) break;
    await sleep(100);
  }
  assert.deepEqual(await page_.violations(), [], "the worklet's blob: module was refused");
  const logged = loggedRefusals(page_);
  assert.deepEqual(logged, [], `the browser logged a refusal: ${logged.join(" | ")}`);
  const trace = await rowTrace(page_);
  assert.equal(
    trace.min,
    0,
    `#rows never emptied when the capture started (${trace.start} row(s) of the demo stayed)`,
  );
  assert.ok(rows > 0, "the fake take transcribed nothing");
});
