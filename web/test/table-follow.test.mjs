// When the notes list may move the PAGE to show the playing row.
//
// The row itself is always MARKED; this is the other half of the behaviour, and
// it is the half that can be wrong in a way nobody notices until they are
// reading: `scrollIntoView` on the row scrolls every ancestor, the DOCUMENT
// included, so a list that follows its row unconditionally takes the roll and
// its playhead out of view while a take sounds.
//
// The decision takes four rect edges and answers the vertical gap itself, so
// every arrangement is measured here without a browser and without audio. The
// WIRING — that the page asks this question at all, and scrolls by the returned
// gap — is pinned by `tests/test_web_build.py`.
import test from "node:test";
import assert from "node:assert/strict";

import { playingRowScroll } from "../playhead.js";

const VIEW = 700;
const ROW = 30; // one 30 px table row
const FILLING = [-600, 2500]; // a table taller than the window, on screen

/** A row whose top edge sits at `top`, and a table spanning `[top, bottom]`. */
const gap = (top, tableTop, tableBottom, viewHeight = VIEW) =>
  playingRowScroll({
    rowTop: top,
    rowBottom: top + ROW,
    tableTop,
    tableBottom,
    viewHeight,
  });

test("a row anywhere on screen moves nothing", () => {
  // The reader is following the line: nudging the page by a few pixels to
  // "perfectly" centre it is exactly the movement that loses the place.
  assert.equal(gap(300, ...FILLING), 0, "a centred row moved the page");
  assert.equal(gap(-10, ...FILLING), 0, "a row cut at the top moved the page");
  assert.equal(gap(VIEW - 10, ...FILLING), 0, "a row cut at the bottom moved the page");
  assert.equal(gap(0, ...FILLING), 0, "a row flush with the top edge moved the page");
});

test("an off-screen row is revealed while the table is being read", () => {
  // One whole row of the table is visible, so the list is what the reader is
  // looking at: following the take there is help, not a hijack.
  assert.equal(gap(-200, ...FILLING), -200, "a row above the table in view was not revealed");
  assert.equal(
    gap(VIEW + 100, ...FILLING),
    ROW + 100,
    "a row below the table in view was not revealed",
  );
  assert.equal(
    gap(VIEW + 100, VIEW - ROW, 2500),
    ROW + 100,
    "a table showing one whole row must follow",
  );
});

test("the gap's sign names which edge comes back into view", () => {
  // Above the window the row's TOP is the distance back; below it the row's
  // BOTTOM is. The caller scrolls by this number alone, so both signs are the
  // rule — a positive value must never be produced for a row overhead.
  const above = gap(-200, ...FILLING);
  const below = gap(VIEW + 100, ...FILLING);
  assert.ok(above < 0, `a row above the window answered ${above}`);
  assert.ok(below > 0, `a row below the window answered ${below}`);
  assert.equal(above, -200, "the row above did not bring its top edge back");
  assert.equal(below, ROW + 100, "the row below did not bring its bottom edge back");
});

test("the roll keeps the page when the table is not what is being read", () => {
  // The take sounds while the eye is on the roll and the table sits under the
  // fold: the page must not travel at all.
  assert.equal(gap(VIEW + 300, VIEW + 70, 2500), 0, "a table under the fold took the page");
  assert.equal(gap(-300, -1000, -40), 0, "a table scrolled past took the page");
  // A table peeking over the fold with less than one row is not a reading
  // surface yet: ten pixels of header must not scroll the roll away.
  assert.equal(gap(VIEW + 300, VIEW - 10, 2500), 0, "a peeking table took the page");
  assert.equal(gap(VIEW + 300, VIEW + 300, VIEW + 600), 0, "an empty viewport-far table took the page");
});
