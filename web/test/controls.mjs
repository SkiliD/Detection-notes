// The page's controls, enumerated ONCE for every craft floor.
//
// `resolutions.mjs` measures the 24 px target at eight widths and
// `craft-floor.test.mjs` measures it on the page it already drives. Two copied
// lists drift: `[tabindex]` had joined one of them only, so the notes scroller
// was measured by one check and forgotten by the other. The selector list and
// the visible geometry therefore live here — a control added to the page joins
// BOTH measurements by existing.
//
// The hidden file input is not a target: it is 1×1 so its label can paint the
// control, and the label (`.file span`) is what a finger hits.
export const CONTROL_SELECTORS = [
  "button",
  "a",
  "input",
  "summary",
  "[tabindex]",
  ".file span",
];

/** Every VISIBLE control, named, with the box both floors measure. */
export const VISIBLE_CONTROLS = `(() => {
  const nodes = [...document.querySelectorAll(${JSON.stringify(CONTROL_SELECTORS.join(", "))})]
    .filter((element) => element.id !== "file");
  return nodes
    .map((element) => {
      const rect = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      return {
        name: element.id || element.className || element.tagName.toLowerCase(),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        visible:
          rect.width > 0 &&
          rect.height > 0 &&
          style.visibility !== "hidden" &&
          style.display !== "none",
      };
    })
    .filter((control) => control.visible);
})()`;
