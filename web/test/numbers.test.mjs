// Every number the page SHOWS, recensed in both languages on one take.
//
// `format.test.mjs` pins the formatter's own separators; this file reads the
// page that uses it, because the two defects of this family were both call
// SITES, not formatters: the table's seconds kept a dot beside a French clock
// that said `0:03,2`, and the readout did the same after the first repair. A
// unit test of `formatNumber` cannot see a line that never calls it.
//
// The inventory is the RENDERED page, not the source:
//
// - `innerText` answers only what a reader can see, so the hidden clock's
//   markup default is not counted as a shown number;
// - `title` and `aria-valuetext` are read too: they are the two attributes a
//   reader HEARS, and `aria-valuetext` is where the clock's own format reaches
//   the DOM even while the clock is folded away;
// - the canvas is hooked through `fillText`, because the roll's second marks
//   and its octave names are painted and appear in no text node at all.
//
// Both directions are asserted. French must carry no decimal dot, AND English
// must carry the same numbers with dots: a page that showed no number, or a
// harness that collected nothing, would otherwise pass this file forever.
import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

import { pressDemo, sleep, startBrowser } from "./browser.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));

// Records what the roll paints. Installed once, from the page itself, so the
// canvas keeps drawing with the browser's own implementation underneath.
const RECORD_PAINTED_TEXT = `(() => {
  window.__numberCensus = [];
  const proto = CanvasRenderingContext2D.prototype;
  if (proto.__numberCensusHooked) return;
  const fillText = proto.fillText;
  proto.fillText = function (text, ...rest) {
    window.__numberCensus.push(String(text));
    return fillText.call(this, text, ...rest);
  };
  proto.__numberCensusHooked = true;
})()`;

// Everything a reader meets as text, joined for one pass: the visible body,
// the two spoken attributes (on VISIBLE elements only — a hidden tooltip is
// not a number anyone reads), and the canvas's painted strings.
const READ_VISIBLE_NUMBERS = `(() => {
  const parts = [document.body.innerText];
  for (const element of document.querySelectorAll("[title], [aria-valuetext]")) {
    if (!element.getClientRects().length) continue;
    parts.push(element.getAttribute("title") ?? "");
    parts.push(element.getAttribute("aria-valuetext") ?? "");
  }
  parts.push(...(window.__numberCensus ?? []));
  return parts.join("\\n");
})()`;

/** The numeric tokens of one reading, separator included. */
const NUMBERS = /\d+(?:[.,]\d+)?/g;

function tokensOf(text) {
  return text.match(NUMBERS) ?? [];
}

/** One language's numbers, from the page as it stands. */
async function readLanguage(page) {
  const text = await page.evaluate(READ_VISIBLE_NUMBERS);
  return { text, tokens: tokensOf(text) };
}

// One take, two readings. The browser is opened, driven and closed here so the
// two tests below read a single inventory: re-driving the demo per test would
// spend a second take to prove the same page.
const page = await startBrowser({ root: WEB });
let french = { text: "", tokens: [] };
let english = { text: "", tokens: [] };
try {
  await pressDemo(page);
  // Let a playback attempt land or expire BEFORE any reading: the page bounds
  // a dead output at 2 s, and a clock that appeared between the two languages
  // would make them differ for a reason this file does not test.
  await sleep(3000);
  // On a machine WITH an output the demo loops, and its clock ticks between
  // the two readings. Stop it through the page's own control (`#listen` while
  // something is listening IS the stop), so both languages describe one take
  // at rest; the clock's format still reaches this census through
  // `aria-valuetext`, which the language gesture rewrites.
  if (await page.evaluate("!document.querySelector('#clock').hidden")) {
    await page.evaluate("document.querySelector('#listen').click()");
    await sleep(250);
  }

  // The roll's marks are painted, so a repaint is forced through the page's own
  // language gesture (it applies the language and redraws) — and the array is
  // cleared first, so each reading holds exactly one language's paint.
  await page.evaluate(RECORD_PAINTED_TEXT);
  await page.evaluate("document.querySelector('#lang').click()"); // fr -> en
  await page.evaluate("window.__numberCensus = []");
  await page.evaluate("document.querySelector('#lang').click()"); // en -> fr
  french = await readLanguage(page);

  await page.evaluate("window.__numberCensus = []");
  await page.evaluate("document.querySelector('#lang').click()"); // fr -> en
  await sleep(120);
  english = await readLanguage(page);
} finally {
  await page.close();
}

test("the French page writes no decimal dot anywhere it shows a number", () => {
  const dotted = french.tokens.filter((token) => token.includes("."));
  assert.deepEqual(
    dotted,
    [],
    `the French page shows a dot as its decimal separator: ${dotted.join(", ")}\n${french.text}`,
  );
  // A net that collected nothing would pass this test forever: the take's own
  // table is two decimals wide and its summary counts notes.
  assert.ok(french.tokens.length >= 12, `only ${french.tokens.length} numbers were read in French`);
  const separated = french.tokens.filter((token) => token.includes(","));
  assert.ok(
    separated.length >= 4,
    `the French take showed ${separated.length} decimal(s): a page with none would pass this file`,
  );
});

test("the English page proves the same inventory sees decimals", () => {
  const dotted = english.tokens.filter((token) => token.includes("."));
  assert.ok(
    dotted.length >= 4,
    `the English page shows ${dotted.length} decimal(s); the French check would pass on a page with none`,
  );
  const commas = english.tokens.filter((token) => token.includes(","));
  assert.deepEqual(commas, [], `the English page shows a comma as its decimal separator: ${commas}`);
});

test("both languages show the same numbers, one separator apart", () => {
  // The separator is the ONLY difference a language may make: normalising it
  // must leave the two readings identical, so a line that stopped being
  // rendered, or a number that changed while switching, fails here instead of
  // hiding behind a shorter list.
  const normalised = (tokens) => tokens.map((token) => token.replace(",", ".")).sort();
  assert.deepEqual(
    normalised(english.tokens),
    normalised(french.tokens),
    "the two languages do not show the same numbers",
  );
});
