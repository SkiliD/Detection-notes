// The accessibility audit, on every push: both palettes, every state the page
// is seen in, and every resolution class.
//
//   node web/test/axe-audit.mjs --axe /path/to/axe.min.js     CHROME=…
//
// Why a browser and not a unit test: axe reads the RENDERED DOM — the computed
// colours of a verdict on its own row wash, the canvas' accessible name, a
// table's header cells. The page ships two palettes and three visible states
// (no take, a take loaded, a row being heard), and it reshapes itself at four
// widths: a rule can be broken in any ONE of those combinations, because a
// control that overlaps its neighbour at 600 px is a control a screen reader's
// user cannot see either. Every combination is audited here, and any violation
// AND any incomplete result fails the run, because a check axe could not decide
// is a check nobody has made either.
//
// axe-core is NOT vendored: the workflow pins its version and the page stays
// dependency-free. The browser is `browser.mjs`'s — a local server and the
// DevTools protocol, nothing installed.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { pressDemo, sleep, startBrowser } from "./browser.mjs";
import { RESOLUTIONS } from "./resolutions.mjs";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const STATES = ["empty", "take", "playing"];

/** Where axe-core is: `--axe <path>` or `AXE=<path>`. */
function axePath() {
  const at = process.argv.indexOf("--axe");
  if (at > -1 && process.argv[at + 1]) return process.argv[at + 1];
  return process.env.AXE ?? null;
}

/**
 * The page is reloaded per palette AND per size, so it is awaited every time.
 *
 * Ready means TRANSLATED, not loaded: `#demo` exists in the markup, so it
 * answers while the new document is still white — and auditing there finds a
 * theme button with no accessible name and a table of headers with no rows,
 * neither of which is what a reader meets. The words arrive by fetch, so the
 * wait is on the words.
 */
async function waitForPage(page) {
  for (let waited = 0; waited < 30000; waited += 100) {
    const ready = await page.evaluate(
      `document.readyState === "complete" &&
       Boolean(document.querySelector("#theme")?.textContent.trim()) &&
       Boolean(document.querySelector("#status")?.textContent.trim())`,
    );
    if (ready) return;
    await sleep(100);
  }
  throw new Error("the page never finished painting its words");
}

/** One audit of whatever is on screen now; a failure names the rule and nodes. */
async function audit(page, axe, label, problems) {
  // axe attaches itself to the document, so it is injected once per reload.
  await page.evaluate(axe);
  const result = await page.evaluate(
    'axe.run(document, { resultTypes: ["violations", "incomplete"] })',
  );
  const refused = [
    ...result.violations.map((rule) => ({ ...rule, kind: "violation" })),
    ...result.incomplete.map((rule) => ({ ...rule, kind: "incomplete" })),
  ];
  for (const rule of refused) {
    const where = rule.nodes
      .slice(0, 3)
      .map((node) => node.target.join(" "))
      .join(" | ");
    problems.push(`${label}: ${rule.kind} [${rule.impact ?? "?"}] ${rule.id} — ${rule.help} (${where})`);
  }
  console.log(
    `${label}: ${result.violations.length} violations, ${result.incomplete.length} incomplete, ` +
      `${result.passes.length} checks passed`,
  );
}

async function main() {
  const source = axePath();
  if (!source) {
    console.error("usage: node web/test/axe-audit.mjs --axe <axe.min.js>   (or AXE=<path>)");
    process.exitCode = 2;
    return;
  }
  const axe = await readFile(source, "utf8");
  const first = RESOLUTIONS[0];
  const page = await startBrowser({ root: ROOT, width: first.width, height: first.height });
  const problems = [];
  try {
    for (const theme of ["light", "dark"]) {
      // The stored theme is what the page paints before its first frame, so the
      // palette is chosen the way a reader's is: reloaded, not recoloured.
      await page.evaluate(`localStorage.setItem("v2m_theme", "${theme}")`);
      for (const size of RESOLUTIONS) {
        await page.send("Emulation.setDeviceMetricsOverride", {
          width: size.width,
          height: size.height,
          deviceScaleFactor: 1,
          mobile: false,
        });
        await page.send("Page.navigate", { url: page.origin });
        await waitForPage(page);
        const painted = await page.evaluate("document.documentElement.dataset.theme");
        if (painted !== theme) throw new Error(`asked for the ${theme} palette, got ${painted}`);

        for (const state of STATES) {
          if (state === "take") await pressDemo(page);
          if (state === "playing") {
            // A row is heard through the page's own class: headless Chrome will
            // not start an AudioContext, so the wash that a verdict is read on
            // is applied directly — it is the state's only difference.
            await page.evaluate('document.querySelectorAll("tbody tr")[1].classList.add("playing")');
          }
          await audit(page, axe, `${theme}/${size.name}/${state}`, problems);
        }
      }
    }
    if (problems.length) {
      throw new Error(`the audit refused the page:\n  ${problems.join("\n  ")}`);
    }
  } finally {
    await page.close();
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
