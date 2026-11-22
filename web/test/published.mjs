// The page AFTER publication: the one thing no checkout can prove.
//
// Every other check in `web/test/` serves `web/` from this machine, so it can
// only show that the FOLDER works. A deployment has its own failure modes —
// GitHub Pages answers 404 until the build propagates, a project site lives
// under `/Detection-notes/` rather than `/`, a module served as
// `text/plain` is refused by the browser's own import rule, `strings.json` can
// be missing from the artifact — and each of them is invisible from here.
//
// So this script reads the PUBLISHED url, over the network, exactly as a
// visitor's browser would: the headers, the two files the page cannot work
// without, the demo that runs the real chain, and the language switch, which
// only moves once the page's own `strings.json` fetch has landed. It is run by
// `pages.yml` right after `deploy-pages`, never by the checkout's own tests.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const { pressDemo, sleep, startBrowser } = await import("./browser.mjs");

const WEB = fileURLToPath(new URL("..", import.meta.url));
const STRINGS = JSON.parse(await readFile(join(WEB, "strings.json"), "utf8"));

const url = process.argv[2];
if (!url) {
  console.error("usage: node web/test/published.mjs <published-url>");
  process.exit(2);
}

/**
 * Answers once the deployment has propagated.
 *
 * A Pages deployment returns before its own CDN does: the first seconds are a
 * 404 for a url that will exist. Retrying is the difference between "not yet"
 * and "broken", and the failure keeps the LAST answer so a real 404 is readable.
 */
async function untilAnswered(target, { tries = 60, waitMs = 1000 } = {}) {
  let last = "no answer";
  for (let attempt = 0; attempt < tries; attempt += 1) {
    try {
      const response = await fetch(target, { redirect: "follow" });
      if (response.ok) return response;
      last = `HTTP ${response.status}`;
    } catch (error) {
      last = error.message;
    }
    await sleep(waitMs);
  }
  throw new Error(`${target} never answered: ${last}`);
}

/** The type a browser needs to RUN the answer, lowercase, without parameters. */
const typeOf = (response) =>
  (response.headers.get("content-type") ?? "none").toLowerCase().split(";")[0].trim();

const page = await untilAnswered(url);
const pageUrl = page.url;
assert.equal(typeOf(page), "text/html", `the page is served as "${typeOf(page)}"`);
const markup = await page.text();
assert.ok(markup.includes('id="demo"'), "the published markup has no demo button");

// The two files the page cannot work without, read from the PUBLISHED origin
// and at the url the page itself would resolve — which is what makes a project
// site under `/Detection-notes/` fail here instead of in a visitor's console.
const script = await fetch(new URL("app.js", pageUrl));
assert.equal(script.status, 200, "app.js is missing from the deployment");
assert.match(
  typeOf(script),
  /javascript/,
  `app.js is served as "${typeOf(script)}": a browser refuses to import it`,
);
// The head's CLASSIC script, fetched before any module runs. It applies the
// stored theme to <html> so the page never paints its light default first; a
// deployment that lost the file (or serves it as "text/plain") still renders,
// so nothing else here would notice the flash a visitor sees every load.
const bootstrap = await fetch(new URL("theme-init.js", pageUrl));
assert.equal(bootstrap.status, 200, "theme-init.js is missing from the deployment");
assert.match(
  typeOf(bootstrap),
  /javascript/,
  `theme-init.js is served as "${typeOf(bootstrap)}": the head's script is refused`,
);
const table = await fetch(new URL("strings.json", pageUrl));
assert.equal(table.status, 200, "strings.json is missing from the deployment");
assert.match(typeOf(table), /json/, `strings.json is served as "${typeOf(table)}"`);
const published = await table.json();
assert.deepEqual(Object.keys(published).sort(), ["en", "fr"], "the published words are not bilingual");

// And now the page itself, driven: the demo runs the REAL chain (YIN, attacks,
// segmenter, roll, table), so a page that renders but cannot listen fails here.
const driver = await startBrowser({ url: pageUrl, width: 1280, height: 900 });
try {
  // A secure context is a PRECONDITION, not a detail: outside one the browser
  // hides `navigator.mediaDevices`, so the microphone — the page's whole point
  // — can never be asked for, and a page that renders perfectly is still
  // unusable. Read from the browser on the published origin, before the demo,
  // so "Settings → Pages → Enforce HTTPS" is a measured fact of every
  // deployment instead of a setting we hope is still on.
  const context = await driver.evaluate(
    "({ secure: isSecureContext, protocol: location.protocol })",
  );
  assert.equal(
    context.secure,
    true,
    `the published page is NOT a secure context (served over ${context.protocol}): the browser hides the microphone`,
  );
  assert.equal(
    context.protocol,
    "https:",
    `the published page answered over ${context.protocol}, not https: (is Enforce HTTPS on?)`,
  );

  const rows = await pressDemo(driver);
  assert.ok(rows > 0, "the published demo produced no notes");
  // The demo LOADS and, being one gesture, starts sounding: the sentence it
  // leaves is a success one either way, so the check refuses the failure
  // sentences instead of pinning which success it stopped on.
  const status = await driver.evaluate('document.querySelector("#status").textContent.trim()');
  const failures = Object.entries(STRINGS.fr)
    .filter(([key]) => key.startsWith("error_"))
    .map(([, text]) => text);
  assert.ok(!failures.includes(status), `the published demo reported "${status}"`);

  // The proof that the page's OWN fetch of `strings.json` reached this origin:
  // the markup ships French copy, and only a table that arrived can answer in
  // English.
  // `#record` is read, not `#demo`: the demo is a TOGGLE, so its own label has
  // already moved to "stop" and would compare against the wrong key.
  await driver.evaluate('document.querySelector("#lang").click()');
  await sleep(300);
  const label = await driver.evaluate('document.querySelector("#record").textContent.trim()');
  assert.equal(label, STRINGS.en.record, "the language switch did not reach the page");

  const trouble = driver.notes().filter((note) => /^(thrown|error):?/.test(note));
  assert.deepEqual(trouble, [], `the published page reported:\n  ${trouble.join("\n  ")}`);
} finally {
  await driver.close();
}

console.log(`the published page answers and reads a take: ${pageUrl}`);
