// The page's numbers, in both languages.
//
// Every number the page shows — the clock's tenths, the Hz line, the table's
// seconds, the roll's second marks — goes through `formatNumber`, which is one
// `Intl.NumberFormat` per language and decimal count. `toFixed` writes a dot
// everywhere, so a French page said "3.2 s" beside a clock that said "0:03,2".
//
// The live readout and the intonation sentence are checked through the pure
// functions that write them, so a harmless refactor of `app.js` cannot fail
// this file. The rendered table is checked in `craft-floor.test.mjs`; the
// roll's marks are canvas text, so their wiring alone is read at the source.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { formatClock, formatHz, formatNumber, intonationNumbers } from "../readout.js";

test("one formatter writes every page number, and French writes commas", () => {
  assert.equal(formatNumber(3.2, 1, "fr"), "3,2");
  assert.equal(formatNumber(3.2, 1, "en"), "3.2");
  assert.equal(formatNumber(12, 2, "fr"), "12,00");
  assert.equal(formatNumber(12, 2, "en"), "12.00");
  assert.equal(formatNumber(0, 2, "fr"), "0,00");
  assert.equal(formatNumber(440, 0, "fr"), "440");
  // No grouping inside a measure: 1200 s stays one number on a dense axis.
  assert.equal(formatNumber(1200, 0, "fr"), "1200");
  // A reading that never arrived prints as zero, not as "NaN".
  assert.equal(formatNumber(Number.NaN, 1, "fr"), "0,0");
});

test("the clock follows the page's language", () => {
  assert.equal(formatClock(3.2), "0:03.2");
  assert.equal(formatClock(3.2, "fr"), "0:03,2");
  assert.equal(formatClock(12, "fr"), "0:12,0");
  // Rounding the tenths first is what keeps a minute from printing 0:60,0.
  assert.equal(formatClock(59.96, "fr"), "1:00,0");
  assert.equal(formatClock(Number.NaN, "fr"), "0:00,0");
});

test("the live Hz line writes the page's separator", () => {
  assert.equal(formatHz(440, "fr"), "440,0 Hz");
  assert.equal(formatHz(440, "en"), "440.0 Hz");
  // A measure, not money: four digits stay ungrouped.
  assert.equal(formatHz(1234.56, "fr"), "1234,6 Hz");
  // An empty slot, not "NaN Hz", when no finite reading arrived.
  assert.equal(formatHz(Number.NaN, "fr"), "");
  assert.equal(formatHz(undefined, "fr"), "");
});

test("the intonation line writes the summary's counts and whole percent", () => {
  const summary = { inTune: 3, n: 5, share: 0.6 };
  assert.deepEqual(intonationNumbers(summary, "fr"), { in_tune: "3", n: "5", share: "60" });
  assert.deepEqual(intonationNumbers(summary, "en"), { in_tune: "3", n: "5", share: "60" });
  // The sentence prints a WHOLE percent: rounding happens before the string.
  assert.equal(intonationNumbers({ inTune: 2, n: 3, share: 2 / 3 }, "fr").share, "67");
  // Counts are measures too, so they never gain a thousands separator.
  assert.deepEqual(intonationNumbers({ inTune: 1200, n: 1200, share: 1 }, "fr"), {
    in_tune: "1200",
    n: "1200",
    share: "100",
  });
});

test("the roll's second marks share the formatter", async () => {
  // The marks are painted into the canvas, so their numbers cannot be read from
  // the DOM; this is the one line whose wiring is still read at the source.
  const app = await readFile(new URL("../app.js", import.meta.url), "utf8");
  assert.match(app, /formatNumber\(second, 0, lang\)/, "the roll's marks write their own dot");
});
