// The notes table follows the playing row — in the PAGE, on a take longer than
// the window.
//
// `table-follow.test.mjs` pins the DECISION (`playingRowScroll`) with rect
// edges and no browser; this file is the wiring that decision exists for,
// measured on a real take: the page travels down to a row below the fold, back
// up to a row above it when the take is seeked to its start, stays still while
// the table is off screen (the reader is watching the roll), and never resets
// the table's own sideways scroll.
//
// The take is synthetic and deliberately longer than the window: forty tones
// with gaps, ~23 s, so the notes table is ~1 750 px tall inside a 500 px
// window — three and a half windows. A take of a few notes fits on screen and
// would pass this file while following nothing: the row would never leave the
// fold.
//
// Playback needs the page's audio graph: without an output device the playhead
// never ticks, so the run reports that instead of waiting on it. The `capture`
// job of `.github/workflows/pages.yml` runs this file on its null sink with
// `V2M_REQUIRE_AUDIO=1`, which turns that report into a failure; the glob in
// `tests.yml` has no output device, so there the file can only report.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { audioState, chooseFile, sleep, startBrowser } from "./browser.mjs";
import { press } from "./keys.mjs";
import { tonesWav } from "./tone.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));
const WORK = await mkdtemp(join(tmpdir(), "v2m-follow-"));
const WAV = join(WORK, "long-take.wav");

// Eight pitches, cycled: every row shows a different note, so the table's
// rows are real transcriptions rather than forty repeats of one tone.
const PITCHES = [220, 246.94, 261.63, 293.66, 329.63, 349.23, 392, 440];
const TONES = 40;
const PARTS = [];
for (let index = 0; index < TONES; index += 1) {
  PARTS.push({ hz: PITCHES[index % PITCHES.length], seconds: 0.45 });
  PARTS.push({ hz: 0, seconds: 0.12 });
}
await writeFile(WAV, tonesWav({ parts: PARTS }));

/** A take of forty notes must fill more rows than a 500 px window can show. */
const LEAST_ROWS = 30;

const driver = await startBrowser({ root: WEB, width: 900, height: 500, mute: false });

test.after(async () => {
  await driver.close();
});

/**
 * The page's own facts, read in ONE snapshot: which row sounds, where it and
 * the table sit, what the page and the table have scrolled to.
 *
 * Two reads could straddle a frame and pair a row with the next one, which is
 * exactly the disagreement this file is about.
 */
const state = () =>
  driver.evaluate(`(() => {
    const rows = [...document.querySelectorAll("#rows tr")];
    const index = rows.findIndex((row) => row.classList.contains("playing"));
    const row = index >= 0 ? rows[index].getBoundingClientRect() : null;
    const table = document.getElementById("notes-table");
    const box = table.getBoundingClientRect();
    return {
      rows: rows.length,
      playing: index,
      y: Math.round(window.scrollY),
      vh: innerHeight,
      rowTop: row ? Math.round(row.top) : null,
      rowBottom: row ? Math.round(row.bottom) : null,
      tableTop: Math.round(box.top),
      tableBottom: Math.round(box.bottom),
      height: Math.round(box.height),
      left: table.scrollLeft,
      scrollWidth: table.scrollWidth,
      clientWidth: table.clientWidth,
    };
  })()`);

// Measured: the followed-row states arrive within 1.39 s at worst (a seek to a
// row off screen under load), so 5 s leaves that margin without waiting twenty
// seconds on a follow that never happened.
async function waitFor(predicate, timeoutMs = 5000, stepMs = 100) {
  const end = Date.now() + timeoutMs;
  let last = null;
  while (Date.now() < end) {
    last = await state();
    if (predicate(last)) return last;
    await sleep(stepMs);
  }
  return { ...last, timedOut: true };
}

/** Parks the page so the table starts `offset` px below the window's top. */
const parkTable = (offset) =>
  driver.evaluate(`(() => {
    const table = document.getElementById("notes-table");
    window.scrollTo(0, table.getBoundingClientRect().top + window.scrollY - ${offset});
  })()`);

/**
 * Stops the take, so the page stops travelling with its row while the test
 * aims the next gesture.
 *
 * A sounding take follows its row every few hundred milliseconds; a page that
 * moves under the pointer between reading the roll's rect and clicking it is
 * how a click misses the roll and the test blames the follow.
 */
async function stopTake() {
  const label = await driver.evaluate("document.getElementById('listen').textContent");
  if (/arrêter/i.test(label)) {
    await driver.evaluate("document.querySelector('#listen').click()");
  }
  await waitFor((s) => s.playing === -1, 5000);
}

/**
 * Clicks the roll at `fraction` of its width: the page's own seek gesture.
 *
 * The roll is scrolled into view first. A mouse event is delivered at VIEWPORT
 * coordinates, so a roll parked above the fold receives nothing at all — the
 * click misses, the take plays on, and a test that only watches the rows would
 * call that a follow.
 */
async function clickRoll(fraction) {
  await driver.evaluate(
    "document.getElementById('roll').scrollIntoView({ block: 'center' })",
  );
  await sleep(150);
  const rect = await driver.evaluate(`(() => {
    const box = document.getElementById("roll").getBoundingClientRect();
    return { x: box.left, y: box.top, width: box.width, height: box.height };
  })()`);
  const x = Math.round(rect.x + rect.width * fraction);
  const y = Math.round(rect.y + rect.height / 2);
  await driver.send("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x,
    y,
    button: "left",
    clickCount: 1,
  });
  await driver.send("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x,
    y,
    button: "left",
    clickCount: 1,
  });
}

test("the notes table follows the take, and leaves the page alone when it is off screen", async (t) => {
  const graph = await audioState(driver);
  if (graph !== "running") {
    const why = `no audio output device (context ${graph})`;
    if (process.env.V2M_REQUIRE_AUDIO === "1") {
      assert.fail(`the follow check did not run: ${why}`);
    }
    t.diagnostic(`the follow check needs an audio output device: ${why}`);
    return;
  }

  await chooseFile(driver, WAV, async () => {
    const rows = await driver.evaluate("document.querySelectorAll('#rows tr').length");
    return rows >= LEAST_ROWS;
  });
  const loaded = await state();
  assert.ok(loaded.rows >= LEAST_ROWS, `the long take transcribed ${loaded.rows} notes`);
  assert.ok(
    loaded.height > loaded.vh,
    `the table is ${loaded.height} px tall in a ${loaded.vh} px window: nothing can leave the fold`,
  );

  // The take sounds from its start: the playhead is the only thing that marks
  // a row, and the follow reads that mark.
  await driver.evaluate("document.querySelector('#listen').click()");
  const sounding = await waitFor((s) => s.playing >= 0);
  assert.ok(!sounding.timedOut, "the take never started sounding");

  await sleep(1200); // the take's own start settles before a seek restarts it

  // --- 1. a row BELOW the fold: a seek late in the take, and the page travels
  //        down to the row it started.
  //
  // The seek happens while the roll is on screen — the table is below the fold
  // there, so the new row is NOT followed yet, and the page is parked on the
  // table right after. The next row change is then below the fold, and it is
  // the page's own follow that has to bring it back.
  await stopTake();
  await clickRoll(0.7);
  await parkTable(60);
  await sleep(150);
  const below = await state();
  const followedDown = await waitFor(
    (s) => s.playing >= 18 && s.rowBottom === s.vh && s.y > below.y + 5,
  );
  assert.ok(
    !followedDown.timedOut,
    `the page did not follow a row below the fold (playing ${below.playing} -> ${followedDown.playing}, y ${below.y} -> ${followedDown.y})`,
  );
  assert.equal(
    followedDown.rowBottom,
    followedDown.vh,
    `the followed row's bottom edge sits at ${followedDown.rowBottom}, not at the window's ${followedDown.vh}`,
  );

  // --- 2. a row ABOVE the fold: the take is seeked back to its start while the
  //        page is parked below it, and the page follows up.
  const wasFar = await state();
  assert.ok(wasFar.playing > 0, "the take is already at its first row");
  await stopTake();
  // Home belongs to the roll, and focusing the roll may scroll it into view,
  // so the page is parked AFTER the focus and the key itself never moves it.
  await driver.evaluate("document.getElementById('roll').focus()");
  await parkTable(-300);
  await sleep(150);
  const above = await state();
  assert.ok(
    above.tableTop < 0 && above.playing === -1,
    `the table starts at ${above.tableTop} px with row ${above.playing} marked`,
  );
  await press(driver, "Home");
  const followedUp = await waitFor(
    (s) => s.playing === 0 && s.rowTop === 0 && s.y < above.y - 5,
  );
  assert.ok(
    !followedUp.timedOut,
    `the page did not follow a row above the fold (playing ${wasFar.playing} -> ${followedUp.playing}, y ${above.y} -> ${followedUp.y})`,
  );
  assert.equal(
    followedUp.rowTop,
    0,
    `the followed row's top edge sits at ${followedUp.rowTop}, not at the window's top`,
  );

  // --- 3. the table out of view: the rows advance and the page stays put.
  await driver.evaluate("window.scrollTo(0, 0)");
  await sleep(200);
  const parked = await state();
  assert.ok(
    parked.tableTop > parked.vh,
    `the table starts at ${parked.tableTop} px, inside the ${parked.vh} px window`,
  );
  // A single frame staying still would also pass a page that jumps on the
  // next row, so the rows are watched through two full notes.
  let furthest = parked.playing;
  for (let waited = 0; waited < 2000; waited += 100) {
    const now = await state();
    furthest = Math.max(furthest, now.playing);
    assert.equal(now.y, 0, `the page moved to ${now.y} px while the table was off screen`);
    await sleep(100);
  }
  assert.ok(furthest >= 1, `the take's row never advanced past ${furthest}`);

  // --- 4. the table's own sideways scroll survives a follow.
  await driver.send("Emulation.setDeviceMetricsOverride", {
    width: 240,
    height: 500,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await sleep(400);
  const narrow = await state();
  assert.ok(
    narrow.scrollWidth > narrow.clientWidth,
    `the table does not overflow sideways at 240 px (${narrow.scrollWidth} <= ${narrow.clientWidth})`,
  );
  await driver.evaluate(`(() => {
    const table = document.getElementById("notes-table");
    table.scrollLeft = 80;
  })()`);
  await stopTake();
  await clickRoll(0.7);
  await parkTable(60);
  await sleep(150);
  const beforeLateral = await state();
  assert.ok(beforeLateral.left > 0, "the table was not scrolled sideways");
  const lateralFollow = await waitFor(
    (s) => s.playing >= 18 && s.rowBottom === s.vh && s.y > beforeLateral.y + 5,
  );
  assert.ok(!lateralFollow.timedOut, "a sideways-scrolled table stopped the follow");
  assert.equal(
    lateralFollow.left,
    beforeLateral.left,
    `the follow reset the table's sideways scroll (${beforeLateral.left} -> ${lateralFollow.left})`,
  );
});
