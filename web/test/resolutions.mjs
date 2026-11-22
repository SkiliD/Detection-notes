// The page at every resolution class, MEASURED in a browser on every push.
//
//   node web/test/resolutions.mjs                    CHROME=/path/to/chrome …
//
// `tests/test_web_build.py` can only grep the stylesheet for its four queries:
// a rule that is declared but loses to another one, a wrap that overflows by
// two pixels, a control that ends under the 24 px tap floor — none of that is
// visible in the CSS. So the layout is measured where it is painted, at the
// widths a reader actually uses, and every class has to hold on its own.
//
// The matrix is DATA, and the accessibility audit imports it: the same eight
// classes are audited in both palettes, so a width cannot be checked for one
// and forgotten by the other.
//
// Expectations are layout facts only (paddings, measures, computed positions),
// never text metrics: the runner's fonts differ from a desktop's, and a check
// that pins 46ch in pixels would fail on Linux for no reason.
import { fileURLToPath, pathToFileURL } from "node:url";
import { pressDemo, sleep, startBrowser } from "./browser.mjs";
import { VISIBLE_CONTROLS } from "./controls.mjs";

const ROOT = fileURLToPath(new URL("../", import.meta.url));

/** Every class the layout has a rule for, plus the two edges of the wide one. */
export const RESOLUTIONS = [
  // A small phone: the narrowest window the page claims to support.
  { name: "phone 320", width: 320, height: 640, wrap: 292, tools: "static", mastheadTop: 40 },
  // The common phone, portrait.
  { name: "phone 390", width: 390, height: 844, wrap: 362, tools: "static", mastheadTop: 40 },
  // Just under the 620 px query: the pills have left the title's corner.
  { name: "small 600", width: 600, height: 900, wrap: 572, tools: "static", mastheadTop: 40 },
  // A phone held SIDEWAYS: a HEIGHT class, where the header is trimmed.
  { name: "landscape 812x375", width: 812, height: 375, wrap: 720, tools: "absolute", mastheadTop: 20 },
  // The first wide pixel: the measure is already wider than a column and has
  // not yet reached `--measure-wide`.
  { name: "wide edge 1000", width: 1000, height: 800, wrap: 928, tools: "absolute", mastheadTop: 46 },
  // The measure is capped from here on.
  { name: "laptop 1280", width: 1280, height: 900, wrap: 1040, tools: "absolute", mastheadTop: 46 },
  // The width the screenshot is taken at.
  { name: "desk 1600", width: 1600, height: 1000, wrap: 1040, tools: "absolute", mastheadTop: 46 },
  // An ultra-wide screen: the paper must not stretch with it.
  { name: "ultrawide 2560", width: 2560, height: 1440, wrap: 1040, tools: "absolute", mastheadTop: 46 },
];

/** The smallest a control may be clicked with, in CSS pixels. */
const TAP_FLOOR = 24;

/** Everything the checks below need, read from the rendered page. */
const MEASURE = `(() => {
  const px = (selector) => {
    const element = document.querySelector(selector);
    return element ? Math.round(element.getBoundingClientRect().width) : -1;
  };
  const box = (selector) => {
    const element = document.querySelector(selector);
    return element ? Math.round(element.getBoundingClientRect().height) : -1;
  };
  // The controls are the SHARED enumeration in controls.mjs: the same list
  // pins the floor in craft-floor.test.mjs, including the notes scroller's
  // tabindex, which a hand-copied list had left behind here. The hidden file
  // input stays out; its label is the target a thumb hits.
  const controls = ${VISIBLE_CONTROLS}.map((control) => ({
    name: control.name,
    box: control.height,
  }));
  const table = document.querySelector(".table-scroll");
  // Elements that stick out of the window — the notes table is ALLOWED to be
  // one, because it scrolls inside its own box; nothing else is.
  const spills = [...document.querySelectorAll("body *")]
    .filter((element) => !table?.contains(element))
    .filter((element) => element.getBoundingClientRect().right > window.innerWidth + 1)
    .map((element) => element.id || element.className || element.tagName)
    .slice(0, 5);
  return {
    window: window.innerWidth,
    scroll: Math.round(document.documentElement.scrollWidth),
    wrap: px(".wrap"),
    deck: px(".deck"),
    mastheadHeight: box(".masthead"),
    mastheadTop: parseInt(getComputedStyle(document.querySelector(".masthead")).paddingTop, 10),
    tools: getComputedStyle(document.querySelector(".tools")).position,
    rows: document.querySelectorAll("tbody tr").length,
    short: controls.filter((control) => control.box < ${TAP_FLOOR}).map((control) => control.name + "=" + control.box + "px"),
    spills,
  };
})()`;

async function main() {
  const page = await startBrowser({ root: ROOT, width: RESOLUTIONS[0].width, height: RESOLUTIONS[0].height });
  const problems = [];
  try {
    // A take on screen: the notes table and its own scroller are part of every
    // class, and an empty page would measure a layout nobody reads.
    await pressDemo(page);
    const seen = new Map();
    for (const size of RESOLUTIONS) {
      await page.send("Emulation.setDeviceMetricsOverride", {
        width: size.width,
        height: size.height,
        deviceScaleFactor: 1,
        mobile: false,
      });
      await sleep(250);
      const facts = await page.evaluate(MEASURE);
      seen.set(size.name, facts);
      const wrong = [];
      if (facts.window !== size.width) wrong.push(`the page measured ${facts.window} px`);
      if (facts.scroll > facts.window) {
        wrong.push(`the page scrolls sideways (${facts.scroll} px of content in ${facts.window})`);
      }
      if (facts.wrap !== size.wrap) wrong.push(`the measure is ${facts.wrap} px, not ${size.wrap}`);
      if (facts.tools !== size.tools) wrong.push(`the pills are ${facts.tools}, not ${size.tools}`);
      if (facts.mastheadTop !== size.mastheadTop) {
        wrong.push(`the masthead pads ${facts.mastheadTop} px, not ${size.mastheadTop}`);
      }
      if (facts.deck > facts.wrap) {
        wrong.push(`the deck overflows its column (${facts.deck} px in ${facts.wrap})`);
      }
      if (facts.rows < 1) wrong.push("the demo take produced no notes");
      if (facts.short.length) wrong.push(`controls under ${TAP_FLOOR} px: ${facts.short.join(", ")}`);
      if (facts.spills.length) wrong.push(`content past the right edge: ${facts.spills.join(", ")}`);
      if (wrong.length) problems.push(`${size.name} (${size.width}×${size.height}): ${wrong.join("; ")}`);
      else console.log(`${size.name}: ${facts.window} px, measure ${facts.wrap} px, ${facts.rows} notes`);
    }

    // Cross-class facts, which no single size can state: the prose keeps its
    // own measure on a wide screen (a widened page must not widen a line), and
    // a phone held sideways trades its header for the controls below it.
    const desk = seen.get("desk 1600");
    const ultrawide = seen.get("ultrawide 2560");
    if (ultrawide.deck !== desk.deck) {
      problems.push(
        `the prose stretches with the window: the deck is ${desk.deck} px at 1600 and ` +
          `${ultrawide.deck} px at 2560`,
      );
    }
    const portrait = seen.get("phone 390");
    const landscape = seen.get("landscape 812x375");
    if (landscape.mastheadHeight >= portrait.mastheadHeight) {
      problems.push(
        `a landscape phone spends ${landscape.mastheadHeight} px on a masthead that costs ` +
          `${portrait.mastheadHeight} px in portrait`,
      );
    }

    if (problems.length) throw new Error(`the page breaks by resolution:\n  ${problems.join("\n  ")}`);
    console.log(`${RESOLUTIONS.length} resolution classes hold`);
  } finally {
    await page.close();
  }
}

// Imported for `RESOLUTIONS` by the audit, the file must not launch a browser
// of its own: the sweep runs when this file IS the command.
if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
