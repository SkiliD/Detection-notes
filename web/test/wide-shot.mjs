// A 1600 px screenshot of the page, taken on every push.
//
// The wide layout is the one no unit test can see: the 1040 px measure, the
// notes table's flush-right numbers, the masthead's band. A picture in the
// run's artifacts is what a reviewer opens when one of them moves — and the
// script also fails on the two facts the wide page exists for (the measure is
// applied, the numeric columns are set right), so an obvious break stops the
// deploy instead of riding along as a file nobody opened.
//
//   node web/test/wide-shot.mjs [out.png]        CHROME=/path/to/chrome …
//
// The browser is `browser.mjs`'s: served from this checkout, driven over the
// DevTools protocol. The take is loaded by pressing the page's own demo button,
// so the screenshot shows what a reader gets, not a fixture.
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { pressDemo, sleep, startBrowser } from "./browser.mjs";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const OUT = resolve(process.argv[2] ?? "wide-1600.png");
const WIDTH = 1600;
const HEIGHT = 1000;

async function main() {
  const page = await startBrowser({ root: ROOT, width: WIDTH, height: HEIGHT });
  try {
    await pressDemo(page);

    const facts = await page.evaluate(`(() => {
      const first = document.querySelector("tbody tr");
      const cell = (index) => getComputedStyle(first.children[index]).textAlign;
      return {
        window: window.innerWidth,
        measure: Math.round(document.querySelector(".wrap").getBoundingClientRect().width),
        rows: document.querySelectorAll("tbody tr").length,
        numbers: [cell(0), cell(1), cell(2), cell(4)],
        words: [cell(3), cell(5)],
      };
    })()`);

    const wrong = [];
    if (facts.window !== WIDTH) wrong.push(`the page measured ${facts.window} px, not ${WIDTH}`);
    if (facts.measure !== 1040) wrong.push(`the wide measure is ${facts.measure} px, not 1040`);
    if (facts.rows < 1) wrong.push("the demo take produced no notes");
    if (facts.numbers.some((align) => align !== "right")) {
      wrong.push(`the numeric columns read ${facts.numbers.join(", ")}`);
    }
    if (facts.words.some((align) => align !== "left")) {
      wrong.push(`the text columns read ${facts.words.join(", ")}`);
    }
    if (wrong.length) throw new Error(`the wide page regressed:\n  ${wrong.join("\n  ")}`);

    // The first screen stops above the notes table, and the table is what the
    // wide check is about: the shot is the WHOLE page at 1600 px wide (grown to
    // its own height, capped so a runaway document cannot fill the runner), so
    // the columns a reviewer must judge are in the picture.
    const tall = Math.min(
      Math.max(await page.evaluate("Math.ceil(document.documentElement.scrollHeight)"), HEIGHT),
      4000,
    );
    await page.send("Emulation.setDeviceMetricsOverride", {
      width: WIDTH,
      height: tall,
      deviceScaleFactor: 1,
      mobile: false,
    });
    await sleep(150);
    const shot = await page.send("Page.captureScreenshot", { format: "png" });
    await writeFile(OUT, Buffer.from(shot.data, "base64"));
    console.log(
      `${OUT}: ${WIDTH} x ${tall} px, measure ${facts.measure} px, ` +
        `${facts.rows} notes, numbers ${facts.numbers.join("/")}, words ${facts.words.join("/")}`,
    );
  } finally {
    await page.close();
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
