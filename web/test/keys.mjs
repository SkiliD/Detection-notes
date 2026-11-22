// The keyboard gestures, in ONE place.
//
// The focus walk (`keyboard.test.mjs`), the legal sequence
// (`gestures.test.mjs`) and the focus-ring walk (`craft-floor.test.mjs`) press
// the same keys over the DevTools protocol. A second copy is how one suite
// quietly stops sending what a reader's browser sends — and `hold` is the one
// that matters most: `event.repeat` only exists because the held key really
// auto-repeats, so a "hold" made of separate taps would let a broken guard
// pass.
import { sleep } from "./browser.mjs";

/** The DevTools parameters of one key, by the name the suites use. */
export const KEYS = {
  Tab: { key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 },
  Enter: { key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, text: "\r" },
  Space: { key: " ", code: "Space", windowsVirtualKeyCode: 32, text: " " },
  ArrowRight: { key: "ArrowRight", code: "ArrowRight", windowsVirtualKeyCode: 39 },
  Home: { key: "Home", code: "Home", windowsVirtualKeyCode: 36 },
  End: { key: "End", code: "End", windowsVirtualKeyCode: 35 },
};

/** One real key press: `Tab` is a raw key (no text), the others are keystrokes. */
export async function press(page, name) {
  const spec = KEYS[name];
  await page.send("Input.dispatchKeyEvent", {
    type: name === "Tab" ? "rawKeyDown" : "keyDown",
    ...spec,
  });
  await page.send("Input.dispatchKeyEvent", { type: "keyUp", ...spec });
}

/**
 * One key HELD down: the first press, then the browser's own auto-repeats.
 *
 * `autoRepeat: true` is what makes the page see `event.repeat`, so this is the
 * same stream a reader produces by resting on an arrow — not a list of taps.
 */
export async function hold(page, name, repeats = 12) {
  const spec = KEYS[name];
  await page.send("Input.dispatchKeyEvent", { type: "keyDown", ...spec });
  for (let step = 0; step < repeats; step += 1) {
    await page.send("Input.dispatchKeyEvent", { type: "keyDown", autoRepeat: true, ...spec });
    await sleep(20);
  }
  await page.send("Input.dispatchKeyEvent", { type: "keyUp", ...spec });
}
