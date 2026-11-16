// Which key events may toggle playback, kept pure so the rule is testable.
//
// Space is the page's only shortcut and the browser already gives it a
// meaning everywhere: it CLICKS a focused button, types a space in a field,
// and scrolls the document. A global handler that just took every Space
// would fire a focused button AND toggle playback in the same press, and
// would break typing. So the rule is stated once, here:
//
// - only a bare Space (no Ctrl/Meta/Alt, and no Shift: Shift+Space is the
//   page-scroll gesture);
// - never while a control that owns the key has focus — a button, an input
//   (the file picker), a textarea, a select, a link, or editable content.

const OWNS_SPACE = new Set(["BUTTON", "INPUT", "TEXTAREA", "SELECT", "A"]);

export function spaceTogglesPlayback({
  key,
  tagName = "",
  editable = false,
  ctrl = false,
  meta = false,
  alt = false,
  shift = false,
} = {}) {
  if (key !== " " && key !== "Spacebar") return false; // "Spacebar": old browsers
  if (ctrl || meta || alt || shift) return false;
  if (editable) return false;
  return !OWNS_SPACE.has(String(tagName).toUpperCase());
}
