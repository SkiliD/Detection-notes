// The page's FLOOR rules, pinned on the rendered page.
//
// Three rules from AGENTS.md's "UI craft gates" are checkable without a design
// review: a control is at least 24×24 CSS px (WCAG 2.2 AA 2.5.8), every control
// reachable by Tab shows a visible focus state, and `prefers-reduced-motion:
// reduce` removes the page's transitions. `resolutions.mjs` measures the target
// floor across eight widths; this file pins it beside the other two, so a
// control that ships without a focus ring fails a test that names it.
//
// The sizes are MEASURED, not read from the stylesheet: padding and font size
// add up to a box, and a declaration can be overridden by a later rule. The
// focus rings are read the same way, after real Tab presses — `:focus-visible`
// is what the page styles, and only a keyboard press makes the browser match it.
// A ring the page draws is `outline-style: solid`; the browser's own fallback
// (`outline: auto`) is refused, because it renders near-white (#eee) on this
// paper — it reports as a ring without being a visible one.
//
// The hidden file input is not a target: it is 1×1 so its label can paint the
// control, and the label is measured instead. Every other control has to clear
// the floor on both axes.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { pressDemo, sleep, startBrowser } from "./browser.mjs";
import { press } from "./keys.mjs";
import { VISIBLE_CONTROLS } from "./controls.mjs";

const WEB = fileURLToPath(new URL("..", import.meta.url));

/** The smallest a control may be clicked with, in CSS pixels. */
const FLOOR = 24;

// Every visible control's box, by the name a failure can repeat. The list lives
// in `controls.mjs`, shared with `resolutions.mjs`: the notes scroller had
// joined one of the two copies only, so a control could be measured by the
// floor at one width and forgotten by the other check.
const CONTROL_BOXES = VISIBLE_CONTROLS;

/** One real Tab, then a beat: programmatic focus skips :focus-visible. */
async function pressTab(page) {
  await press(page, "Tab");
  await sleep(60);
}

/** The focus ring the page actually paints on the element focus landed on. */
const FOCUS_RING = `(() => {
  const active = document.activeElement;
  if (!active || active === document.body) return null;
  // The file input is hidden on purpose: its label carries the ring.
  const target = active.id === "file" ? document.querySelector(".file span") : active;
  const style = getComputedStyle(target);
  const width = Number.parseFloat(style.outlineWidth) || 0;
  const alpha = style.outlineColor.startsWith("rgba")
    ? Number.parseFloat(style.outlineColor.split(",")[3])
    : 1;
  return {
    name: active.id || active.tagName.toLowerCase(),
    paintedBy: target.id || target.className || target.tagName.toLowerCase(),
    ring: style.outlineStyle === "solid" && width > 0 && alpha > 0,
    outline: style.outlineStyle + " " + style.outlineWidth + " " + style.outlineColor,
  };
})()`;

let page = null;
let loaded = false;

/** The page once, with a take on screen: the listening and export row exists then. */
async function readyPage() {
  page ??= await startBrowser({ root: WEB, width: 1200, height: 900 });
  if (!loaded) {
    await pressDemo(page);
    loaded = true;
  }
  return page;
}

test.after(async () => {
  if (page) await page.close();
});

test("every visible control clears the 24x24 px floor", async () => {
  const page_ = await readyPage();
  const boxes = await page_.evaluate(CONTROL_BOXES);
  // A net that matched nothing would pass forever: the empty page alone has
  // eleven controls, and this one carries a take on top.
  assert.ok(boxes.length >= 14, `only ${boxes.length} controls were measured`);

  const short = boxes.filter(
    (control) => control.name !== "file" && (control.width < FLOOR || control.height < FLOOR),
  );
  assert.deepEqual(
    short.map((control) => `${control.name} ${control.width}x${control.height}`),
    [],
    `controls under the ${FLOOR} px floor`,
  );
});

test("every control reached by Tab paints a visible focus ring", async () => {
  const page_ = await readyPage();
  // Start from the top: the walk is the browser's own chain, not a list.
  await page_.evaluate("document.body.focus()");
  const visited = new Map();
  for (let step = 0; step < 30; step += 1) {
    await pressTab(page_);
    const ring = await page_.evaluate(FOCUS_RING);
    if (!ring) break; // the chain ended on the document
    if (visited.has(ring.name)) break; // wrapped around to the first control
    visited.set(ring.name, ring);
  }

  assert.ok(visited.size >= 12, `the Tab chain stopped after ${visited.size} controls`);
  const blind = [...visited.values()]
    .filter((ring) => !ring.ring)
    .map((ring) => `${ring.name} (${ring.outline})`);
  assert.deepEqual(blind, [], "controls with no visible focus ring");
});

/** One selector's declarations, whether it opens a rule or shares one. */
function bodyFor(css, selector) {
  const rule = [...css.matchAll(/([^{}]+)\{([^}]*)\}/g)].find(([, selectors]) =>
    selectors
      .split(",")
      .map((entry) => entry.trim())
      .includes(selector),
  );
  assert.ok(rule, `style.css no longer declares \`${selector}\``);
  return rule[2];
}

/** The declarations of the rule that opens with `selector` alone on its line. */
function ruleBody(css, selector) {
  const at = css.indexOf(`${selector} {`);
  assert.notEqual(at, -1, `style.css has no \`${selector}\` rule`);
  return css.slice(at, css.indexOf("}", at));
}

/** Every style rule of a stylesheet TEXT, inside an @media block or not. */
function styleRules(css) {
  const text = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const rules = [];
  const preludes = [];
  let start = 0;
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "{") {
      preludes.push(text.slice(start, index).trim());
      start = index + 1;
    } else if (text[index] === "}") {
      const prelude = preludes.pop() ?? "";
      // An at-rule's prelude is not a selector: its inner rules were pushed
      // and popped as rules of their own.
      if (prelude && !prelude.startsWith("@")) {
        rules.push({ selector: prelude, body: text.slice(start, index) });
      }
      start = index + 1;
    }
  }
  return rules;
}

/** A selector list, split on the commas that are NOT inside (), [] or a string. */
function selectorList(selector) {
  const parts = [];
  let current = "";
  let depth = 0;
  let quote = null;
  for (const char of selector) {
    if (quote) {
      current += char;
      if (char === quote) quote = null;
      continue;
    }
    if (char === '"' || char === "'") {
      quote = char;
      current += char;
      continue;
    }
    if (char === "(" || char === "[") depth += 1;
    if (char === ")" || char === "]") depth -= 1;
    if (char === "," && depth === 0) {
      parts.push(current.trim());
      current = "";
      continue;
    }
    current += char;
  }
  if (current.trim()) parts.push(current.trim());
  return parts;
}

/**
 * Every selector list of `css` that names a `-webkit-` pseudo-element AND a
 * `-moz-` one.
 *
 * Such a list is invalid WHOLE in each engine — the engine does not know the
 * other one's pseudo-element — so both drop the rule and `style.css` still
 * reads as declared. The CSSOM cannot show it (Chrome keeps no trace of the
 * dropped rule), so the file itself is read, @media blocks included.
 */
function mixedVendorSelectors(css) {
  return styleRules(css)
    .map((rule) => rule.selector)
    .filter((selector) => {
      const entries = selectorList(selector);
      return (
        entries.some((entry) => /::-webkit-/.test(entry)) &&
        entries.some((entry) => /::-moz-/.test(entry))
      );
    })
    .map((selector) => selector.replace(/\s+/g, " "));
}

/**
 * Every rule of the page's OWN sheet, with its declarations, `@media` blocks
 * flattened: ONE probe, read by the disabled-balance test and by the
 * vendor-list guard, so both judge the sheet the browser really parsed
 * instead of two copies that can drift apart.
 */
const PARSED_RULES = `(() => {
  const sheet = [...document.styleSheets].find((entry) =>
    (entry.href ?? "").endsWith("style.css"),
  );
  if (!sheet) return null;
  const rules = [];
  const walk = (list) => {
    for (const rule of list) {
      if (rule instanceof CSSStyleRule) {
        rules.push({
          selector: rule.selectorText,
          background: rule.style.getPropertyValue("background").trim(),
          shadow: rule.style.getPropertyValue("box-shadow").trim(),
        });
      } else if (rule.cssRules) {
        walk(rule.cssRules);
      }
    }
  };
  walk(sheet.cssRules);
  return rules;
})()`;

test("every interactive control matches a :focus-visible rule", async () => {
  // Comments sit between rules, and a selector line above a block would
  // otherwise be read as part of it.
  const css = (await readFile(join(WEB, "style.css"), "utf8")).replace(/\/\*[\s\S]*?\*\//g, "");
  // Every block that styles keyboard focus, with the pseudo-class stripped:
  // what is left still has to NAME the control it draws the ring on, whether
  // or not that control is reachable in the state on screen (disabled buttons
  // and the closed tutorial's own buttons are not).
  const focusRules = [...css.matchAll(/([^{}]+)\{([^}]*)\}/g)].filter(([, selectors]) =>
    selectors.includes(":focus-visible"),
  );
  assert.ok(focusRules.length >= 4, `only ${focusRules.length} focus rules were found`);
  const group = focusRules.find(([, selectors]) => selectors.includes("button:focus-visible"));
  assert.ok(group, "the grouped focus rule is gone");
  assert.match(
    group[2],
    /outline:\s*2px solid var\(--copper\)/,
    "the grouped focus rule still exists but draws no ring",
  );
  const selectors = focusRules.flatMap(([, list]) =>
    list
      .split(",")
      .map((entry) => entry.trim())
      // A compound entry still counts: the file input's ring is the label's,
      // `.file input:focus-visible + span`, which does not end on the pseudo.
      .filter((entry) => entry.includes(":focus-visible"))
      .map((entry) => entry.replace(":focus-visible", "").trim()),
  );
  assert.ok(selectors.length >= 5, `only ${selectors.length} focus selectors were parsed`);

  const page_ = await readyPage();
  const checked = await page_.evaluate(`(() => {
    const selectors = ${JSON.stringify(selectors)};
    // The file input's ring is painted on the label beside it, and that label
    // is checked in its place.
    const nodes = [...document.querySelectorAll("button, a, input, summary, [tabindex], .file span")]
      .filter((element) => element.type !== "file");
    return {
      total: nodes.length,
      missing: nodes
        .filter((element) => !selectors.some((selector) => {
          try { return element.matches(selector); } catch { return false; }
        }))
        .map((element) => element.id || element.className || element.tagName.toLowerCase()),
    };
  })()`);
  assert.ok(checked.total >= 12, `only ${checked.total} controls were checked`);
  assert.deepEqual(
    checked.missing,
    [],
    "controls whose focus-ring rule no longer names them (a renamed selector)",
  );
});

/**
 * The ring on the focused control and the surface it is drawn against.
 *
 * The ring sits 2 px OUTSIDE the control (`outline-offset`), so the colour
 * that has to carry it is the nearest ancestor painting an opaque background —
 * the paper, or the masthead's night band for its two pills.
 */
const FOCUS_RING_CONTRAST = `(() => {
  const active = document.activeElement;
  if (!active || active === document.body) return null;
  const target = active.id === "file" ? document.querySelector(".file span") : active;
  const style = getComputedStyle(target);
  let node = target.parentElement;
  let background = "rgb(255, 255, 255)";
  while (node) {
    const painted = getComputedStyle(node).backgroundColor;
    const alpha = painted.startsWith("rgba") ? Number.parseFloat(painted.split(",")[3]) : 1;
    if (painted !== "transparent" && alpha === 1) {
      background = painted;
      break;
    }
    node = node.parentElement;
  }
  return {
    name: active.id || active.tagName.toLowerCase(),
    colour: style.outlineColor,
    background,
    width: Number.parseFloat(style.outlineWidth) || 0,
    line: style.outlineStyle,
  };
})()`;

test("the focus ring is 2 px of copper and clears 3:1 on both palettes", async () => {
  // A renamed selector empties its rule without deleting the file, so the ring
  // is pinned twice: its declaration must still name the controls and the
  // copper token, and the browser must still paint it at non-text contrast
  // (WCAG 1.4.11 — a ring nobody can see is not a focus state). The surface
  // behind the ring is measured, not assumed: the pills sit on the night band.
  const css = (await readFile(join(WEB, "style.css"), "utf8")).replace(/\/\*[\s\S]*?\*\//g, "");
  const focusRules = [...css.matchAll(/([^{}]+)\{([^}]*)\}/g)].filter(([, selectors]) =>
    selectors.includes(":focus-visible"),
  );
  const ringRules = focusRules.filter(([, , body]) => /outline:/.test(body));
  assert.ok(ringRules.length >= 2, `only ${ringRules.length} rules draw the ring`);
  for (const [, selectors, body] of ringRules) {
    assert.match(
      body,
      /outline:\s*2px solid var\(--copper\)/,
      `${selectors.trim()}: the ring lost its width, its style or its one colour`,
    );
  }
  // A compound entry still counts — the file input's ring is its label's — but
  // a suffix is refused: `#roll:focus-visible-old` names nothing the browser
  // matches, so it must not pass as the ring's owner.
  const names = (selectors, control) =>
    selectors
      .split(",")
      .map((entry) => entry.trim())
      .some((entry) => entry === control || entry.startsWith(`${control} `));
  for (const control of [
    "button:focus-visible",
    "#roll:focus-visible",
    ".guide > summary:focus-visible",
    ".file input:focus-visible",
    'input[type="range"]:focus-visible',
    ".table-scroll:focus-visible",
  ]) {
    assert.ok(
      focusRules.some(([, selectors]) => names(selectors, control)),
      `${control} no longer names its control: a renamed selector would leave it unringed`,
    );
  }

  const page_ = await readyPage();
  const original = await page_.evaluate("document.documentElement.dataset.theme ?? ''");
  try {
    for (const theme of ["light", "dark"]) {
      await page_.evaluate(`document.documentElement.dataset.theme = "${theme}"`);
      await page_.evaluate("document.body.focus()");
      const visited = new Map();
      for (let step = 0; step < 30; step += 1) {
        await pressTab(page_);
        const ring = await page_.evaluate(FOCUS_RING_CONTRAST);
        if (!ring) break; // the chain ended on the document
        if (visited.has(ring.name)) break; // wrapped around to the first control
        visited.set(ring.name, ring);
      }
      assert.ok(visited.size >= 12, `${theme}: only ${visited.size} focus rings were measured`);
      const faint = [...visited.values()]
        .map((ring) => ({ ...ring, ratio: contrast(ring.colour, ring.background) }))
        .filter((ring) => ring.line !== "solid" || ring.width < 2 || ring.ratio < 3)
        .map(
          (ring) =>
            `${ring.name}: ${ring.line} ${ring.width}px ${ring.colour} on ${ring.background}` +
            ` = ${ring.ratio.toFixed(2)}:1`,
        );
      assert.deepEqual(faint, [], `${theme}: focus rings that are not 2 px of copper at 3:1`);
    }
  } finally {
    await page_.evaluate(
      `document.documentElement.dataset.theme = ${JSON.stringify(original)}`,
    );
  }
});

test("prefers-reduced-motion silences every transition, the meter's included", async () => {
  const css = await readFile(join(WEB, "style.css"), "utf8");
  const block = css.match(/@media \(prefers-reduced-motion: reduce\)\s*\{([\s\S]*?)\n\}/);
  assert.ok(block, "the reduced-motion block is gone from style.css");
  assert.match(block[1], /transition:\s*none/, "the block no longer removes the transitions");
  // Every family that animates anywhere must be named here: a selector left
  // out of the block keeps moving for a reader who asked it not to.
  for (const selector of ["button", ".link", ".meter-fill"]) {
    assert.ok(
      block[1].includes(selector),
      `${selector} animates but the reduced-motion block does not silence it`,
    );
  }
  // The block only silences what these rules declare, so the declarations are
  // pinned too: renaming one would otherwise empty its rule in silence.
  assert.match(ruleBody(css, "button"), /transition:\s*background-color 90ms ease-out/);
  assert.match(ruleBody(css, ".link"), /transition:\s*background-color 90ms ease-out/);
  assert.match(ruleBody(css, ".meter-fill"), /transition:\s*width 80ms linear/);
});

test("reduced motion really computes 0s, in both palettes", async () => {
  // Reading the block proves the rule EXISTS; this proves the browser APPLIES
  // it. Chrome emulates the reader's setting, and the computed
  // `transition-duration` is the value the compositor will use — a selector
  // renamed inside the media query leaves the transition at 90/80 ms here even
  // though the block still names it.
  const page_ = await readyPage();
  const readTransitions = () =>
    page_.evaluate(`(() => {
      const read = (selector) => {
        const style = getComputedStyle(document.querySelector(selector));
        return { duration: style.transitionDuration, delay: style.transitionDelay };
      };
      return {
        button: read("button"),
        link: read(".link"),
        meter: read(".meter-fill"),
      };
    })()`);

  // First the page as every reader gets it: if these were already 0s, the
  // emulated check below would pass on a page that never animates at all.
  const moving = await readTransitions();
  assert.equal(moving.button.duration, "0.09s", "the button lost its transition");
  assert.equal(moving.link.duration, "0.09s", "the masthead pill lost its transition");
  assert.equal(moving.meter.duration, "0.08s", "the level meter lost its smoothing");

  const original = await page_.evaluate("document.documentElement.dataset.theme ?? ''");
  await page_.send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-motion", value: "reduce" }],
  });
  try {
    for (const theme of ["light", "dark"]) {
      await page_.evaluate(`document.documentElement.dataset.theme = "${theme}"`);
      const still = await readTransitions();
      for (const [name, value] of Object.entries(still)) {
        assert.equal(value.duration, "0s", `${theme}: ${name} still transitions for ${value.duration}`);
        assert.equal(value.delay, "0s", `${theme}: ${name} still waits ${value.delay}`);
      }
    }
  } finally {
    await page_.send("Emulation.setEmulatedMedia", { features: [] });
    await page_.evaluate(
      `document.documentElement.dataset.theme = ${JSON.stringify(original)}`,
    );
  }

  // Both borrowings belong to THIS test: every later test in this file drives
  // the same page, so an emulation or a palette left behind would silence
  // their transitions and measure the wrong sheet. Restoring without checking
  // it took effect is how that happens quietly.
  const handedBack = await page_.evaluate(`(() => ({
    reduced: matchMedia("(prefers-reduced-motion: reduce)").matches,
    theme: document.documentElement.dataset.theme ?? "",
  }))()`);
  assert.equal(handedBack.reduced, false, "the emulated reduced motion outlived its test");
  assert.equal(handedBack.theme, original, "the palette outlived its test");
});

/** WCAG relative luminance of a computed `rgb(r, g, b)` colour. */
function luminance(color) {
  const [r, g, b] = color
    .match(/[\d.]+/g)
    .slice(0, 3)
    .map(Number)
    .map((channel) => {
      const value = channel / 255;
      return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** The WCAG contrast ratio between two computed colours. */
function contrast(foreground, background) {
  const [lighter, darker] = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
  return (lighter + 0.05) / (darker + 0.05);
}

test("headings balance their lines, running prose stays pretty", async () => {
  const css = await readFile(join(WEB, "style.css"), "utf8");
  for (const selector of ["h1", "h2"]) {
    assert.match(ruleBody(css, selector), /text-wrap:\s*balance/, `${selector} does not balance`);
  }
  for (const selector of [".deck", ".fineprint"]) {
    assert.match(ruleBody(css, selector), /text-wrap:\s*pretty/, `${selector} does not wrap pretty`);
  }
});

test("the disabled balance keeps a quiet track and a readable thumb", async () => {
  // The slider is a control too: at 45 % its track and thumb faded into the
  // paper together, so an inactive balance looked like a smudge instead of a
  // control that is waiting. It follows the button pattern — a quiet field and
  // an ink that still reads — and the colours come from the SAME disabled pair
  // every other control uses, in both palettes.
  const css = (await readFile(join(WEB, "style.css"), "utf8")).replace(/\/\*[\s\S]*?\*\//g, "");
  const disabled = bodyFor(css, '.balance-row input[type="range"]:disabled');
  assert.match(disabled, /cursor:\s*default/);
  assert.ok(!/opacity/.test(disabled), "the disabled balance fades itself again");
  for (const pseudo of ["::-webkit-slider-runnable-track", "::-moz-range-track"]) {
    const track = bodyFor(css, `.balance-row input[type="range"]:disabled${pseudo}`);
    assert.match(track, /background:\s*var\(--button-off\)/, `${pseudo} lost its quiet field`);
    assert.match(track, /box-shadow:\s*none/, `${pseudo} keeps the active track's shadow`);
  }
  for (const pseudo of ["::-webkit-slider-thumb", "::-moz-range-thumb"]) {
    const thumb = bodyFor(css, `.balance-row input[type="range"]:disabled${pseudo}`);
    assert.match(thumb, /background:\s*var\(--button-off-ink\)/, `${pseudo} lost its readable ink`);
    assert.match(thumb, /box-shadow:\s*none/, `${pseudo} keeps the active knob's ring`);
  }

  const page_ = await readyPage();

  // The stylesheet TEXT is not the sheet the browser APPLIES: a rule grouped
  // with the other engine's pseudo-element is dropped WHOLE by the engine that
  // does not know that pseudo, so `bodyFor` above still finds the disabled
  // track in `style.css` while the rendered slider keeps its enabled one. The
  // rules this engine actually parses are looked up in the CSSOM and must
  // still carry the quiet field or the readable ink they declare; that no rule
  // of the file shares a selector list with the other engine is pinned for the
  // WHOLE sheet by the guard below. The `-moz` rules can only be read as text:
  // no Chromium sheet ever contains them, which is the reason they are separate.
  const parsed = await page_.evaluate(PARSED_RULES);
  assert.ok(parsed, "the page no longer loads style.css as its own sheet");
  for (const [pseudo, expected] of [
    ["::-webkit-slider-runnable-track", "--button-off"],
    ["::-webkit-slider-thumb", "--button-off-ink"],
  ]) {
    const selector = `.balance-row input[type="range"]:disabled${pseudo}`;
    const compact = selector.split(" ").join("");
    const rule = parsed.find((entry) => entry.selector.split(" ").join("") === compact);
    assert.ok(
      rule,
      `${selector} is not in the parsed stylesheet: the browser dropped a grouped rule`,
    );
    assert.equal(
      rule.background,
      `var(${expected})`,
      `${selector} lost its ${expected} fill in the sheet the browser parses`,
    );
    assert.equal(rule.shadow, "none", `${selector} keeps the active shadow/ring in the parsed sheet`);
  }

  const painted = await page_.evaluate(`(() => {
    const slider = document.querySelector("#balance");
    const was = slider.disabled;
    slider.disabled = true;
    const style = getComputedStyle(slider);
    const read = { opacity: style.opacity, cursor: style.cursor };
    slider.disabled = was;
    return read;
  })()`);
  assert.equal(painted.opacity, "1", "the disabled balance fades its whole control again");
  assert.equal(painted.cursor, "default", "the disabled balance still offers a grab cursor");

  // The pair is measured where it is painted: `--button-off-ink` on
  // `--button-off`, resolved by the browser in each palette.
  const original = await page_.evaluate("document.documentElement.dataset.theme ?? ''");
  try {
    for (const theme of ["light", "dark"]) {
      const tokens = await page_.evaluate(`(() => {
        document.documentElement.dataset.theme = "${theme}";
        const root = getComputedStyle(document.documentElement);
        const probe = document.createElement("span");
        probe.style.color = root.getPropertyValue("--button-off-ink").trim();
        probe.style.background = root.getPropertyValue("--button-off").trim();
        document.body.append(probe);
        const style = getComputedStyle(probe);
        const pair = { ink: style.color, field: style.backgroundColor };
        probe.remove();
        return pair;
      })()`);
      const ratio = contrast(tokens.ink, tokens.field);
      assert.ok(ratio >= 4.5, `${theme}: the disabled thumb reads ${ratio.toFixed(2)}:1 on its track`);
    }
  } finally {
    await page_.evaluate(`document.documentElement.dataset.theme = ${JSON.stringify(original)}`);
  }
});

test("no rule of the sheet groups a -webkit- pseudo-element with a -moz- one", async () => {
  // The balance slider carried this defect: one selector list naming both
  // engines' pseudo-elements is invalid for each of them, so each drops the
  // rule and the file still reads as declared. The guard covers EVERY rule of
  // `style.css`, not just the slider, and it is pinned twice: the source may
  // not mix the two engines in one list, and every `::-webkit-` rule the file
  // declares must still be in the sheet the browser parses — a grouped rule is
  // exactly the one that vanishes from there.
  const css = await readFile(join(WEB, "style.css"), "utf8");
  const grouped = mixedVendorSelectors(css);
  assert.deepEqual(
    grouped,
    [],
    "selector lists mixing a -webkit- pseudo-element with a -moz- one: each engine drops the whole rule",
  );

  const engineRules = styleRules(css)
    .map((rule) => rule.selector.replace(/\s+/g, " "))
    .filter((selector) => /::-(webkit|moz)-/.test(selector));
  assert.ok(
    engineRules.length >= 8,
    `only ${engineRules.length} engine rules were found: the guard lost its net`,
  );

  const page_ = await readyPage();
  const parsed = await page_.evaluate(PARSED_RULES);
  assert.ok(parsed, "the page no longer loads style.css as its own sheet");
  const compacted = parsed.map((rule) => rule.selector.split(" ").join(""));
  const dropped = engineRules
    .filter((selector) => selector.includes("::-webkit-"))
    .filter((selector) => !compacted.includes(selector.split(" ").join("")));
  assert.deepEqual(
    dropped,
    [],
    "rules the browser dropped from style.css: they share a selector list the engine cannot parse",
  );
});

test("the vendor-list guard bites on a rule nobody has written yet", () => {
  // A rule from a FUTURE change, naming pseudo-elements no page rule uses, has
  // to be caught — inside an @media too, and with the lookalike in a comment
  // ignored. A pair of adjacent single-engine rules must NOT be caught: that is
  // the shape the whole sheet uses, and a guard that bans it would be useless.
  const future = `
    /* .x::-webkit-track, .x::-moz-track is a comment, not a rule */
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-moz-number-spin-box { color: var(--copper); }
    @media (min-width: 620px) {
      .card::-webkit-scrollbar, .card::-moz-scrollbar { width: 6px; }
    }
  `;
  assert.deepEqual(mixedVendorSelectors(future), [
    'input[type="number"]::-webkit-inner-spin-button, input[type="number"]::-moz-number-spin-box',
    ".card::-webkit-scrollbar, .card::-moz-scrollbar",
  ]);

  const separate = `
    .x::-webkit-slider-thumb { color: red; }
    .x::-moz-range-thumb { color: red; }
    .a:is(.b, .c)::-webkit-scrollbar { width: 2px; }
  `;
  assert.deepEqual(mixedVendorSelectors(separate), []);
});

test("both disabled text links keep their word at 4.5:1 and drop the underline", async () => {
  // `#balance-reset` and the tutorial's `#guide-demo` disable during a take;
  // they are buttons painted as links, so the inactive state is DECLARED on
  // them (a quiet `--button-off` field, a readable `--button-off-ink` label)
  // instead of being faded. The rendered pair is measured in BOTH palettes,
  // because a palette switch is exactly when inherited ink lands on a field
  // from the other one. The LINE is measured too: an underlined word reads as
  // a live link, so the disabled state must drop it — and the enabled state
  // must keep it, or the check would pass on a page where links lost theirs.
  const page_ = await readyPage();
  const original = await page_.evaluate("document.documentElement.dataset.theme ?? ''");
  const painted = [];
  try {
    for (const theme of ["light", "dark"]) {
      await page_.evaluate(`document.documentElement.dataset.theme = "${theme}"`);
      for (const id of ["balance-reset", "guide-demo"]) {
        const link = await page_.evaluate(`(() => {
          const element = document.querySelector("#${id}");
          const was = element.disabled;
          element.disabled = false;
          const enabledUnderline = getComputedStyle(element).textDecorationLine;
          element.disabled = true;
          const style = getComputedStyle(element);
          const painted = {
            color: style.color,
            background: style.backgroundColor,
            opacity: style.opacity,
            underline: style.textDecorationLine,
            enabledUnderline,
          };
          element.disabled = was;
          return painted;
        })()`);
        painted.push({ theme, id, ...link, ratio: contrast(link.color, link.background) });
      }
    }
  } finally {
    await page_.evaluate(`document.documentElement.dataset.theme = ${JSON.stringify(original)}`);
  }

  assert.equal(painted.length, 4, "both links were not measured in both palettes");
  const faded = painted
    .filter((link) => link.opacity !== "1")
    .map((link) => `${link.theme} ${link.id}`);
  assert.deepEqual(faded, [], "a disabled text link fades its own label again");
  const faint = painted
    .filter((link) => link.ratio < 4.5)
    .map(
      (link) =>
        `${link.theme} ${link.id}: ${link.color} on ${link.background}` +
        ` = ${link.ratio.toFixed(2)}:1`,
    );
  assert.deepEqual(faint, [], "disabled text links under 4.5:1 on their own field");
  const lined = painted
    .filter((link) => link.underline !== "none")
    .map((link) => `${link.theme} ${link.id}: ${link.underline}`);
  assert.deepEqual(lined, [], "a disabled text link still draws the line it cannot keep");
  const unlined = painted
    .filter((link) => link.enabledUnderline !== "underline")
    .map((link) => `${link.theme} ${link.id}: ${link.enabledUnderline}`);
  assert.deepEqual(unlined, [], "an enabled text link lost the underline it acts through");
});

test("the readout announces the note, not every frame", async () => {
  const html = await readFile(join(WEB, "index.html"), "utf8");
  const note = html.match(/<p id="note"[^>]*>/)?.[0] ?? "";
  assert.match(note, /aria-live="polite"/, "the note name is not announced");
  assert.match(note, /aria-atomic="true"/, "the note name is announced by fragments");
  const section = html.match(/<section class="readout"[^>]*>/)?.[0] ?? "";
  assert.ok(section, "the readout section is gone");
  assert.ok(
    !section.includes("aria-live"),
    "the whole readout is a live region: the Hz line speaks with the note",
  );

  // Both writes go through the change gate. A direct assignment puts the frame
  // rate back into the announcement, which is what the region exists to avoid.
  const app = await readFile(join(WEB, "app.js"), "utf8");
  assert.match(app, /if \(note\.textContent !== name\)/, "the note is re-assigned every frame");
  assert.match(app, /if \(line\.textContent !== hz\)/, "the Hz line is re-assigned every frame");
});

test("the table's seconds follow the page's language", async () => {
  const page_ = await readyPage();
  const starts = () =>
    page_.evaluate(
      "Array.from(document.querySelectorAll('#rows tr td:nth-child(2)'))" +
        ".map((cell) => cell.textContent.trim())",
    );

  const french = await starts();
  assert.ok(french.length >= 5, `only ${french.length} rows were measured`);
  for (const cell of french) assert.match(cell, /^\d+,\d{2} s$/, `French cell: ${cell}`);

  // The same rows, re-rendered in English: the separator is the only change.
  await page_.evaluate("document.querySelector('#lang').click()");
  await sleep(250);
  const english = await starts();
  assert.equal(english.length, french.length, "the language switch changed the table");
  for (const cell of english) assert.match(cell, /^\d+\.\d{2} s$/, `English cell: ${cell}`);
});
