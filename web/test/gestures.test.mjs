// The playback gestures as a SEQUENCE, held to three invariants.
//
// `keyboard.test.mjs` proves each gesture once, in isolation: one Space starts
// the take, one arrow seeks, a held arrow does not restart it twelve times.
// This file drives gestures that OVERLAP: a seek is async (its graph may have
// to start), so a click landing while an arrow's restart is in flight can run a
// second stop/render/start and leave "playback could not start" on screen.
//
// So this file drives a legal sequence a singer actually produces — Space, a
// click in the roll, arrows, a click DURING a restart, a held arrow, Space
// again — and after every step asserts the invariants:
//
// 1. AT MOST ONE audio graph is open and ONE buffer source is sounding (the
//    page plays the piano alone here): a leaked source from a previous take is
//    audible, and it is exactly what a lost race sounds like;
// 2. a legal sequence NEVER says `error_play` and never throws — the failure
//    that overlap produces is a sentence, so the sentence is the assertion;
// 3. the CLOCK agrees with itself and with the seek surface: both ends in the
//    page's own `m:ss,d` shape, position inside the total, `aria-valuetext`
//    reading the very second the clock shows, and the second a seek announces
//    being the second the clock counts from.
//
// The counters come from the page's own Web Audio objects (wrapped before the
// first gesture), not from a private field: `playbackContext` and `player` are
// module state, and a test that read them would pass on a page whose second
// take sounds over the first.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { audioCounts, watchAudio } from "./audio.mjs";
import { audioState, chooseFile, sleep, startBrowser } from "./browser.mjs";
import { hold, press } from "./keys.mjs";
import { tonesWav } from "./tone.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8")).fr;

// A take long enough for four different seeks to land apart: every tone
// re-attacks after a silence, so the page reads separate notes and the roll
// spans ~4.9 s.
const PARTS = [
  { hz: 440, seconds: 1.0 },
  { hz: 0, seconds: 0.3 },
  { hz: 587.33, seconds: 1.0 },
  { hz: 0, seconds: 0.3 },
  { hz: 440, seconds: 1.0 },
  { hz: 0, seconds: 0.3 },
  { hz: 587.33, seconds: 1.0 },
];

/** A real click in the roll at `fraction` of its width. */
async function clickRoll(page, fraction) {
  const box = await page.evaluate(`(() => {
    const r = document.querySelector("#roll").getBoundingClientRect();
    return { left: r.left, top: r.top, width: r.width, height: r.height };
  })()`);
  const x = Math.round(box.left + box.width * fraction);
  const y = Math.round(box.top + box.height / 2);
  await page.send("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x,
    y,
    button: "left",
    clickCount: 1,
  });
  await page.send("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x,
    y,
    button: "left",
    clickCount: 1,
  });
}

/**
 * Two clicks in ONE task, landing while the first seek is still restarting.
 *
 * Separate protocol events are separate tasks, and the page's restart can
 * finish between them: an overlap this file cannot time from the outside. Two
 * dispatches in a single `evaluate` cannot interleave with the seek's own
 * continuation, so the guard is exercised every run.
 */
async function clickRollTwice(page) {
  await page.evaluate(`(() => {
    const roll = document.querySelector("#roll");
    const rect = roll.getBoundingClientRect();
    for (const fraction of [0.35, 0.55]) {
      roll.dispatchEvent(new MouseEvent("click", {
        clientX: rect.left + rect.width * fraction,
        clientY: rect.top + rect.height / 2,
        bubbles: true,
      }));
    }
  })()`);
}

/** Everything the three invariants read, after one gesture has settled. */
async function checkpoint(page) {
  const audio = await audioCounts(page);
  const surface = await page.evaluate(`(() => {
    const roll = document.querySelector("#roll");
    return {
      status: document.querySelector("#status").textContent.trim(),
      shown: !document.querySelector("#clock").hidden,
      pos: document.querySelector("#clock-pos").textContent.trim(),
      total: document.querySelector("#clock-total").textContent.trim(),
      text: roll.getAttribute("aria-valuetext") ?? "",
      min: Number(roll.getAttribute("aria-valuemin")),
      max: Number(roll.getAttribute("aria-valuemax")),
      now: Number(roll.getAttribute("aria-valuenow")),
    };
  })()`);
  return { ...audio, ...surface };
}

const CLOCK = /^\d+:\d{2},\d$/;

/** `m:ss,d` as seconds; the page writes a comma in French. */
function secondsOf(text) {
  const [minutes, rest] = text.split(":");
  return Number(minutes) * 60 + Number(rest.replace(",", "."));
}

/** The second a seek sentence names, or NaN. */
function seekSeconds(status) {
  const prefix = STRINGS.seek_at.split("{time}")[0];
  if (!status.startsWith(prefix)) return Number.NaN;
  const time = status.slice(prefix.length).trim().replace(/\.$/, "");
  return CLOCK.test(time) ? secondsOf(time) : Number.NaN;
}

/** Invariants 1 and 2, at every step of the sequence. */
function assertInvariants(state, label) {
  assert.ok(state.open <= 1, `${label}: ${state.open} audio graphs are open at once`);
  assert.ok(state.live <= 1, `${label}: ${state.live} buffer sources are sounding at once`);
  assert.notEqual(
    state.status,
    STRINGS.error_play,
    `${label}: a legal gesture sequence said playback could not start`,
  );
}

/** Invariant 3: the clock, the slider and the seek sentence are one story. */
function assertClock(state, label) {
  if (!state.shown) return;
  assert.match(state.pos, CLOCK, `${label}: the clock's position reads "${state.pos}"`);
  assert.match(state.total, CLOCK, `${label}: the clock's total reads "${state.total}"`);
  assert.ok(
    secondsOf(state.pos) <= secondsOf(state.total) + 1e-9,
    `${label}: the clock stands at ${state.pos}, past its own ${state.total}`,
  );
  assert.equal(state.text, state.pos, `${label}: the slider reads a second the clock does not`);
  assert.ok(
    state.now >= state.min && state.now <= state.max + 0.05,
    `${label}: the slider stands at ${state.now} outside [${state.min}, ${state.max}]`,
  );
}

const work = await mkdtemp(join(tmpdir(), "v2m-gestures-"));
const take = join(work, "gestes.wav");
await writeFile(take, tonesWav({ parts: PARTS }));

const page = await startBrowser({ root: WEB, width: 1200, height: 900, mute: false });
test.after(async () => {
  await page.close();
});

test("a legal gesture sequence keeps one graph, one source and one clock", async (t) => {
  if ((await audioState(page)) !== "running") {
    const why = "no audio output device (the playback context never runs)";
    if (process.env.V2M_REQUIRE_AUDIO === "1") assert.fail(`the gesture sequence did not run: ${why}`);
    t.skip(`needs an audio output device: ${why}`);
    return;
  }

  await chooseFile(page, take, async () => {
    const rows = await page.evaluate("document.querySelectorAll('#rows tr').length");
    return rows >= 4;
  });
  const rows = await page.evaluate("document.querySelectorAll('#rows tr').length");
  assert.ok(rows >= 4, `the take produced ${rows} rows: the roll would have nothing to seek`);

  // Counted from here on: the take's analysis may have opened a decoder, but
  // no PLAYBACK graph exists until the first gesture.
  await watchAudio(page);
  const errorsAtStart = page.notes().length;

  const steps = [];
  const after = async (label, { delta }) => {
    await sleep(350);
    const state = await checkpoint(page);
    assertInvariants(state, label);
    assertClock(state, label);
    if (delta !== undefined) {
      assert.equal(
        state.sources,
        steps.length ? steps[steps.length - 1].sources + delta : delta,
        `${label}: the sequence created ${state.sources - (steps.at(-1)?.sources ?? 0)} sources, not ${delta}`,
      );
    }
    steps.push({ label, ...state });
    return state;
  };

  // 1. Space, nothing focused: the page's own shortcut starts the take.
  await page.evaluate("document.activeElement.blur()");
  await press(page, "Space");
  const started = await after("Space starts", { delta: 1 });
  assert.equal(started.shown, true, "Space starts: the clock stayed hidden while the take sounds");
  assert.equal(started.live, 1, "Space starts: the take sounds from one source");
  assert.match(started.status, new RegExp(`^${STRINGS.playing}`, "i"), "Space starts: status");

  // 2. A click in the roll: the take resumes from that second, not from zero.
  await clickRoll(page, 0.3);
  const clicked = await after("click in the roll", { delta: 1 });
  const clickedSecond = seekSeconds(clicked.status);
  assert.ok(Number.isFinite(clickedSecond), `the click did not announce a seek: ${clicked.status}`);
  assert.ok(
    Math.abs(secondsOf(clicked.pos) - clickedSecond) <= 1.0,
    `the click announced ${clickedSecond} s and the clock counts from ${clicked.pos}`,
  );

  // 3. Two clicks in ONE task, the second landing mid-restart: the guard
  //    refuses it, so exactly one restart runs and nothing fails.
  await clickRollTwice(page);
  const burst = await after("two clicks in one task", { delta: 1 });
  assert.ok(
    secondsOf(burst.pos) >= clickedSecond - 0.1,
    `the burst moved the take backwards (${clicked.pos} -> ${burst.pos})`,
  );

  // 4. An arrow steps from the playhead, and the clock follows it.
  const beforeArrow = secondsOf(burst.pos);
  await page.evaluate("document.querySelector('#roll').focus()");
  await press(page, "ArrowRight");
  const arrowed = await after("arrow right", { delta: 1 });
  assert.ok(
    secondsOf(arrowed.pos) >= beforeArrow - 0.2,
    `the arrow did not move forward (${burst.pos} -> ${arrowed.pos})`,
  );

  // 5. A HELD arrow is one restart: every auto-repeat is refused.
  await hold(page, "ArrowRight");
  const held = await after("held arrow", { delta: 1 });
  assert.ok(
    secondsOf(held.pos) >= secondsOf(arrowed.pos) - 0.2,
    `the held arrow moved the take backwards (${arrowed.pos} -> ${held.pos})`,
  );

  // 6. Home names the take's start.
  await press(page, "Home");
  const home = await after("Home", { delta: 1 });
  assert.ok(
    secondsOf(home.pos) <= 1.0,
    `Home left the clock at ${home.pos}, not at the take's start`,
  );

  // 7. Space, again with nothing focused: the same press stops the take and
  //    releases every resource it opened.
  await page.evaluate("document.activeElement.blur()");
  await press(page, "Space");
  const stopped = await after("Space stops", { delta: 0 });
  assert.equal(stopped.status, STRINGS.stopped, "Space stops: the page did not say it stopped");
  assert.equal(stopped.live, 0, `Space stops: ${stopped.live} sources kept sounding`);
  assert.equal(stopped.open, 0, `Space stops: ${stopped.open} audio graphs stayed open`);
  assert.equal(stopped.shown, false, "Space stops: the clock stayed on screen");

  // 8. And it starts again — from the beginning of a fresh graph, still one.
  await press(page, "Space");
  const restarted = await after("Space starts again", { delta: 1 });
  assert.equal(restarted.live, 1, "Space starts again: the take does not sound from one source");
  assert.equal(restarted.open, 1, `Space starts again: ${restarted.open} graphs are open`);

  // The sequence is over: no gesture may have thrown or logged an error.
  const said = page.notes().slice(errorsAtStart).filter((note) => /^(thrown|error)/.test(note));
  assert.deepEqual(said, [], `the gesture sequence reported errors: ${said.join(" | ")}`);
  assert.ok(steps.length >= 8, `only ${steps.length} steps were measured`);
});
