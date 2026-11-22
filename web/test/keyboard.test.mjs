// The page with no mouse: Tab, Enter and Space, from the file picker to the
// export.
//
// The rest of `web/test/` presses buttons through `element.click()`, which is
// not a keyboard gesture at all: it skips the focus chain, and it never meets
// the page's ONE shortcut. Space is the trap. A browser already CLICKS the
// focused button on Space, so a document-level handler that also toggled
// playback made one press do two things — start and stop, or click Export and
// toggle a take. `web/shortcuts.js` states the rule (a control that owns the key
// keeps it), and this file is the browser proof of it.
//
// The journey is driven with REAL key events over the DevTools protocol: Tab
// moves focus as the browser does, Enter opens the file chooser (intercepted
// here, because headless Chrome never draws the native dialog), and Space
// activates whatever has focus. Every step is read back from the page.
//
// The Space half needs an audio OUTPUT device — the page's playback context
// awaits `resume()`, which a machine with no sound card never settles — so it
// is its own test, skipped with the measured reason where no device exists.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, readdir, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { audioCounts, markAudio, stopWatchingAudio, watchAudio } from "./audio.mjs";
import { audioState, chooseFile, startBrowser, sleep } from "./browser.mjs";
import { hold, press } from "./keys.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const TAKE = fileURLToPath(new URL("../../tests/data/parity.wav", import.meta.url));
const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8")).fr;

/**
 * The controls Tab reaches before any take: the disabled ones are SKIPPED.
 *
 * The tutorial's summary is a stop of its own; its buttons are behind the
 * closed sheet and can never take focus while it is closed. The notes table is
 * hidden until it holds rows, so the chain ends at the demo. What a browser
 * does AFTER the last control (a stop on the document, or straight back to the
 * first control) is its own business, so this chain stops there.
 */
const FOCUS_EMPTY = ["theme", "lang", "summary", "record", "file", "demo"];

/**
 * The same walk once a take exists, from the file input onwards.
 *
 * Stop stays disabled (nothing is being captured). The roll is now a stop: it
 * is the page's seek surface, and a control promised by a hint must be
 * reachable without a mouse. Then the listening and export row joins the
 * chain, in the order a reader needs it: hear the take, hear voice and piano,
 * export MIDI, export the archive, balance, reset — the table last, because it
 * is a reading surface, not an action.
 */
const FOCUS_TAKE = [
  "demo",
  "roll",
  "listen",
  "voice",
  "export",
  "bundle",
  "balance",
  "balance-reset",
  "notes-table",
];

/** Where focus is, by the name the chains above use. */
function focused(driver) {
  return driver.evaluate(
    "(() => { const a = document.activeElement; " +
      "return a && a !== document.body ? (a.id || a.tagName.toLowerCase()) : 'body'; })()",
  );
}

function status(driver) {
  return driver.evaluate("document.querySelector('#status').textContent.trim()");
}

/** Waits for the page to say `wanted`; an error sentence is answered as it is.
 *  Measured 0–119 ms, so 1 s is the budget; the audio-stop callers pass their
 *  own 3 s because a stopped graph settles later than a status line. */
async function waitForStatus(driver, wanted, timeoutMs = 1000) {
  let seen = "";
  for (let waited = 0; waited < timeoutMs; waited += 100) {
    seen = await status(driver);
    if (seen === wanted || seen === STRINGS.error_play) return seen;
    await sleep(100);
  }
  return seen;
}

function clock(driver) {
  return driver.evaluate(
    "({ shown: !document.querySelector('#clock').hidden, text: document.querySelector('#clock-pos').textContent })",
  );
}

// The page writes the clock in its own language: a French page separates
// tenths with a comma.
const secondsOf = (text) => {
  const [minutes, rest] = text.split(":");
  return Number(minutes) * 60 + Number(rest.replace(",", "."));
};

// The seek sentence, split around the second it names.
const SEEK_PREFIX = STRINGS.seek_at.split("{time}")[0];

/** The next seek announcement, skipping the one already on screen. */
async function seekAnnouncement(driver, previous = null, timeoutMs = 8000) {
  let said = "";
  for (let waited = 0; waited < timeoutMs; waited += 100) {
    said = await status(driver);
    if (said === STRINGS.error_play || said === STRINGS.listened) return said;
    if (said.startsWith(SEEK_PREFIX) && said !== previous) return said;
    await sleep(100);
  }
  return said;
}

/** The second the seek sentence names, in seconds. */
function seekSeconds(said) {
  const time = said.slice(SEEK_PREFIX.length).trim().replace(/\.$/, "");
  return /^\d+:\d{2}[,.]\d$/.test(time) ? secondsOf(time) : Number.NaN;
}

/** What the roll declares about itself, plus the clock it must agree with. */
function seekSurface(driver) {
  return driver.evaluate(`(() => {
    const roll = document.querySelector("#roll");
    return {
      role: roll.getAttribute("role"),
      tabindex: roll.getAttribute("tabindex"),
      min: Number(roll.getAttribute("aria-valuemin")),
      max: Number(roll.getAttribute("aria-valuemax")),
      now: Number(roll.getAttribute("aria-valuenow")),
      text: roll.getAttribute("aria-valuetext"),
      clock: document.querySelector("#clock-pos").textContent,
      label: roll.getAttribute("aria-label"),
    };
  })()`);
}

/** One keyboard activation, then the single file it must have produced. */
async function activateAndDownload(driver, id, key, folder, extension) {
  await mkdir(folder, { recursive: true });
  await driver.send("Page.setDownloadBehavior", { behavior: "allow", downloadPath: folder });
  await driver.evaluate(`document.querySelector('#${id}').focus()`);
  assert.equal(await focused(driver), id, `#${id} did not take focus`);
  await press(driver, key);
  // The budget absorbs a machine running every page file at once; the elapsed
  // time is printed either way, so a slow export is read in the log instead of
  // being mistaken for a broken one (or hidden by a generous wait).
  const started = Date.now();
  let names = [];
  for (let waited = 0; waited < 30000; waited += 200) {
    names = (await readdir(folder)).filter((name) => name.endsWith(extension));
    if (names.length) break;
    await sleep(200);
  }
  if (names.length) {
    console.log(`#${id} wrote ${names[0]} after ${key} in ${((Date.now() - started) / 1000).toFixed(1)} s`);
  } else {
    const what = await status(driver);
    throw new Error(
      `#${id} wrote no ${extension} within 30 s after ${key} (status: "${what}")`,
    );
  }
  // A second copy would be a double activation: the browser would have written
  // `prise-web-….mid` and `prise-web-… (1).mid` in the same folder.
  await sleep(800);
  names = (await readdir(folder)).filter((name) => name.endsWith(extension));
  assert.equal(names.length, 1, `one ${key} press wrote ${names.length} ${extension} files`);
  return { name: names[0], bytes: await readFile(join(folder, names[0])) };
}

const work = await mkdtemp(join(tmpdir(), "v2m-keyboard-"));
let driver = null;

async function page() {
  driver ??= await startBrowser({
    root: WEB,
    width: 1200,
    height: 900,
    // Playback is one of the gestures, and a muted browser opens no output
    // stream: the Space test below needs the page's audio graph to start.
    mute: false,
  });
  return driver;
}

test.after(async () => {
  if (driver) await driver.close();
});

test("the file, the take and the exports are reachable by Tab, Enter and Space", async () => {
  const page_ = await page();
  await page_.send("Page.enable");
  await page_.send("DOM.enable");
  await sleep(300);
  await page_.evaluate("document.body.focus()");

  // 1. The reachable controls, before any take.
  const empty = [];
  for (let step = 0; step < FOCUS_EMPTY.length; step += 1) {
    await press(page_, "Tab");
    await sleep(60);
    empty.push(await focused(page_));
  }
  assert.deepEqual(
    empty,
    FOCUS_EMPTY,
    "the keyboard chain before a take moved: a control was added, removed or reordered",
  );

  // 2. Tab to the file input and open it with Enter, as a reader does. The
  //    chooser is the browser's own; headless Chrome never draws it, so the
  //    dialog is intercepted and answered with the committed phrase. The
  //    event is the proof that Enter opened it.
  await page_.send("Page.setInterceptFileChooserDialog", { enabled: true });
  await page_.evaluate("document.querySelector('#file').focus()");
  assert.equal(await focused(page_), "file");
  let chooser = null;
  for (let attempt = 0; attempt < 8 && !chooser; attempt += 1) {
    const opened = page_.waitFor("Page.fileChooserOpened", 2500).catch(() => null);
    await press(page_, "Enter");
    chooser = await opened;
  }
  assert.ok(chooser, "Enter on the file input never opened a file chooser");
  await page_.send("DOM.setFileInputFiles", {
    backendNodeId: chooser.backendNodeId,
    files: [TAKE],
  });

  let rows = 0;
  for (let waited = 0; waited < 20000 && rows !== 5; waited += 200) {
    rows = await page_.evaluate("document.querySelectorAll('#rows tr').length");
    if (rows !== 5) await sleep(200);
  }
  assert.equal(rows, 5, `the chosen file produced ${rows} rows, not 5`);
  assert.equal(await status(page_), STRINGS.ready, "the page never said the file was read");
  const names = await page_.evaluate(
    "Array.from(document.querySelectorAll('#rows tr td:nth-child(4)')).map((cell) => cell.textContent.trim())",
  );
  assert.deepEqual(names, ["C4", "E4", "E4", "D4", "G4"], "the keyboard-chosen take is not the phrase");

  // 3. The chain a take opens, from the file input on.
  await page_.evaluate("document.querySelector('#file').focus()");
  const withTake = [];
  for (let step = 0; step < FOCUS_TAKE.length; step += 1) {
    await press(page_, "Tab");
    await sleep(60);
    withTake.push(await focused(page_));
  }
  assert.deepEqual(
    withTake,
    FOCUS_TAKE,
    "the keyboard chain after a take moved: the listening and export row must follow the order a reader needs",
  );

  // 4. Enter exports the MIDI, one file for one press. What the file HOLDS is
  //    `file.test.mjs`'s subject; the gesture is this file's.
  const midi = await activateAndDownload(page_, "export", "Enter", join(work, "midi"), ".mid");
  assert.equal(midi.bytes.subarray(0, 4).toString("latin1"), "MThd", "the export is not a MIDI file");

  // 5. Space exports the archive too: Space activates a button like Enter, and
  //    the page's shortcut does not add a gesture of its own.
  const zip = await activateAndDownload(page_, "bundle", "Space", join(work, "zip"), ".zip");
  assert.equal(zip.bytes.subarray(0, 2).toString("latin1"), "PK", "the archive is not a zip");

  // 6. A control that OWNS Space keeps it: the balance is a range input, and
  //    one Space there must move neither the take nor the playback.
  await page_.evaluate("document.querySelector('#balance').focus()");
  assert.equal(await focused(page_), "balance");
  await press(page_, "Space");
  await sleep(600);
  assert.equal(await status(page_), STRINGS.ready, "Space on the balance moved the take");
  assert.equal((await clock(page_)).shown, false, "Space on the balance opened the clock");

  const said = page_.notes().filter((note) => /^(thrown|error)/.test(note));
  assert.deepEqual(said, [], `the page reported errors: ${said.join(" | ")}`);
});

test("Space starts the take once, and never while a control owns the key", async (t) => {
  const page_ = await page();
  if ((await page_.evaluate("document.querySelectorAll('#rows tr').length")) !== 5) {
    // This test does not depend on the walk above; it needs a take, and the
    // demo takes no key of its own.
    await chooseFile(page_, TAKE, async () => {
      const rows = await page_.evaluate("document.querySelectorAll('#rows tr').length");
      return rows === 5;
    });
  }

  // A graph that can actually sound, so the clock may be expected to move.
  if ((await audioState(page_)) !== "running") {
    const why = "no audio output device (the playback context never runs)";
    if (process.env.V2M_REQUIRE_AUDIO === "1") assert.fail(`the Space half did not run: ${why}`);
    t.skip(`needs an audio output device: ${why}`);
    return;
  }

  // 7. Space on a button is ONE activation. The page's shortcut must leave the
  //    key to the focused control — a second toggle would stop what the first
  //    one started, and the status would say so.
  await page_.evaluate("document.querySelector('#listen').focus()");
  assert.equal(await focused(page_), "listen");
  await press(page_, "Space");
  const playing = await waitForStatus(page_, STRINGS.playing);
  assert.equal(playing, STRINGS.playing, "one Space on Écouter did not start the take exactly once");
  const firstClock = await clock(page_);
  assert.ok(firstClock.shown, "the clock is hidden while the take sounds");
  await sleep(900);
  assert.equal(await status(page_), STRINGS.playing, "the take stopped by itself: Space did a second action");
  const later = await clock(page_);
  assert.ok(
    secondsOf(later.text) > secondsOf(firstClock.text),
    `the clock did not move (${firstClock.text} -> ${later.text})`,
  );
  // The label changes with the state, so the control says what Space will do.
  assert.match(await page_.evaluate("document.querySelector('#listen').textContent"), /arrêter/i);
  await press(page_, "Space");
  assert.equal(
    await waitForStatus(page_, STRINGS.stopped, 3000),
    STRINGS.stopped,
    "the second Space did not stop the take",
  );
  assert.equal((await clock(page_)).shown, false, "the clock stayed after the take stopped");

  // 8. With nothing focused, Space is the page's own shortcut: it starts, then
  //    the same press again stops.
  await page_.evaluate("document.activeElement.blur()");
  await press(page_, "Space");
  assert.equal(
    await waitForStatus(page_, STRINGS.playing),
    STRINGS.playing,
    "Space with nothing focused did not start the take",
  );
  await press(page_, "Space");
  assert.equal(
    await waitForStatus(page_, STRINGS.stopped, 3000),
    STRINGS.stopped,
    "Space with nothing focused did not stop the take",
  );
});

test("the roll is a Tab stop whose arrows seek, and says the second", async (t) => {
  const page_ = await page();
  if ((await page_.evaluate("document.querySelectorAll('#rows tr').length")) !== 5) {
    await chooseFile(page_, TAKE, async () => {
      const rows = await page_.evaluate("document.querySelectorAll('#rows tr').length");
      return rows === 5;
    });
  }
  if ((await audioState(page_)) !== "running") {
    const why = "no audio output device (the playback context never runs)";
    if (process.env.V2M_REQUIRE_AUDIO === "1") assert.fail(`the roll seek did not run: ${why}`);
    t.skip(`needs an audio output device: ${why}`);
    return;
  }

  // The roll follows the transport's last button — a hint that promises a
  // gesture must leave that gesture on the keyboard's path.
  await page_.evaluate("document.querySelector('#demo').focus()");
  await press(page_, "Tab");
  await sleep(80);
  assert.equal(await focused(page_), "roll", "Tab from Démonstration did not land on the roll");

  const idle = await seekSurface(page_);
  assert.equal(idle.role, "slider", "the roll is not announced as a seek slider");
  assert.equal(idle.tabindex, "0", "the roll left the Tab chain");
  assert.match(idle.label, /flèches/i, "the roll's name does not teach the arrows");
  assert.equal(idle.min, 0, "the seek range does not start at zero");
  assert.ok(idle.max > 0, "the seek slider carries no range");
  assert.ok(
    idle.now >= idle.min && idle.now <= idle.max,
    `aria-valuenow (${idle.now}) is outside [${idle.min}, ${idle.max}]`,
  );

  // The take's own sources are the only honest witness: the take is short
  // enough that repeated seeks all clamp at its end, so the POSITION cannot
  // tell one start from twelve. What ONE start costs is measured first, on the
  // single press below; the held arrow then has to cost exactly that. The
  // prototype is handed back after the hold, so the Home and End steps that
  // follow are not instrumented.
  await watchAudio(page_);

  // Right hears the take from one second in; the status and the slider agree.
  await press(page_, "ArrowRight");
  const first = await seekAnnouncement(page_);
  assert.ok(first.startsWith(SEEK_PREFIX), `the arrow did not announce a seek: ${first}`);
  assert.ok(Math.abs(seekSeconds(first) - 1) <= 0.6, `the first arrow resumed at ${first}`);
  const started = await seekSurface(page_);
  assert.equal(started.text, started.clock, "the slider reads another second than the clock");
  assert.ok(started.now >= 0.5 && started.now <= 2.5, `aria-valuenow says ${started.now}`);
  const oneStart = (await audioCounts(page_)).sources;

  // A second press steps from the playhead, not from zero.
  await press(page_, "ArrowRight");
  const second = await seekAnnouncement(page_, first);
  assert.ok(
    seekSeconds(second) > seekSeconds(first),
    `the second arrow did not move forward (${first} -> ${second})`,
  );
  assert.ok(
    seekSeconds(second) <= seekSeconds(first) + 1.6,
    `the second arrow jumped more than a second (${first} -> ${second})`,
  );

  // A HELD arrow is ONE resume: the browser's auto-repeat is one gesture, not
  // a stop-render-restart per repeat, and two starts landing together would
  // reach the page's "playback could not start" sentence. What one start
  // costs is measured on the single press above, so the hold is compared to
  // it rather than to a number.
  const notesBeforeHold = page_.notes().length;
  await markAudio(page_);
  await hold(page_, "ArrowRight");
  await sleep(700);
  const heldSources = (await audioCounts(page_)).sources;
  await stopWatchingAudio(page_);
  assert.ok(
    oneStart >= 1,
    `one arrow press created ${oneStart} sources: the count measured nothing`,
  );
  assert.equal(
    heldSources,
    oneStart,
    `a held arrow created ${heldSources} sources where one start creates ${oneStart}: ` +
      "an auto-repeat ran a whole seek",
  );
  assert.notEqual(
    await status(page_),
    STRINGS.error_play,
    "a held arrow raced two seeks into a playback failure",
  );
  const saidWhileHeld = page_
    .notes()
    .slice(notesBeforeHold)
    .filter((note) => /^(thrown|error)/.test(note));
  assert.deepEqual(saidWhileHeld, [], `a held arrow reported errors: ${saidWhileHeld.join(" | ")}`);

  // Home and End name the two ends of the take.
  await press(page_, "Home");
  const home = await seekAnnouncement(page_, second);
  assert.ok(Math.abs(seekSeconds(home)) <= 0.3, `Home resumed at ${home}`);

  // End puts the head on the range's last second. When that second is the
  // take's END, there is nothing left to hear and the page says the take is
  // over instead of a seek — the slider still reports where the head is.
  await press(page_, "End");
  await sleep(400);
  const ended = await status(page_);
  assert.ok(
    ended.startsWith(SEEK_PREFIX) || ended === STRINGS.listened,
    `End answered neither a seek nor the end of the take: ${ended}`,
  );
  const last = await seekSurface(page_);
  assert.ok(
    Math.abs(last.now - last.max) <= 0.2,
    `End left the slider at ${last.now} of ${last.max}`,
  );

  const said = page_.notes().filter((note) => /^(thrown|error)/.test(note));
  assert.deepEqual(said, [], `the page reported errors: ${said.join(" | ")}`);
});
