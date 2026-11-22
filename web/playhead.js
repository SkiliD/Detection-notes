// Where the playhead is, which note it is inside, and where that note's row
// sits in the list.
//
// Kept pure and separate from the canvas and the DOM so the decisions a
// playhead can get wrong are testable without a browser:
//
// - the lookup is HALF-OPEN (`start <= t < end`): an inclusive end would name
//   the previous note during a gap and hide a real silence;
// - the answer is an INDEX, not an object: the roll and the notes table show
//   the same take, so one number lights the bar AND the row, and they cannot
//   disagree;
// - the pixel mapping never leaves the roll, because a reading past the last
//   sample must sit on the edge instead of disappearing;
// - the notes list follows the playing row only when the reader is reading it:
//   a row with any part on screen moves nothing, and one off screen moves the
//   page by the vertical gap alone — `scrollIntoView` scrolls every ancestor,
//   the document included, whatever the reader was watching.
//   The gap is this module's answer, so the page cannot disagree with the rule
//   that allowed the move.

/** Index of the note sounding at `t`, or -1. Half-open, so gaps stay silent. */
export function noteIndexAt(notes, t) {
  for (let index = 0; index < notes.length; index++) {
    const note = notes[index];
    if (t >= note.start && t < note.end) return index;
  }
  return -1;
}

/**
 * The vertical gap the notes list may scroll the PAGE by to show the playing
 * row: 0 when nothing moves, negative when the row is above, positive below.
 *
 * The row is always MARKED; this only decides whether the page also travels.
 * Three answers matter and each has a case:
 *
 * - a row with any part on screen moves nothing, so a take read in place never
 *   nudges the line the eye is following;
 * - a row off screen is revealed only when at least one whole row of the table
 *   sits in the viewport: a table merely peeking over the fold is not what the
 *   reader is watching, and following it there would scroll the roll and its
 *   playhead out of view;
 * - when it does move, the gap is the row's own edge: its top above the fold,
 *   its bottom below it. Returning the number rather than a yes/no keeps the
 *   direction here, where the decision was made — a caller free to rebuild it
 *   is a second owner of the same rule.
 *
 * The rects are all it needs, so every arrangement is measurable without a
 * browser (`web/test/table-follow.test.mjs`).
 */
export function playingRowScroll({ rowTop, rowBottom, tableTop, tableBottom, viewHeight }) {
  const rowVisible = rowBottom > 0 && rowTop < viewHeight;
  if (rowVisible) return 0;
  const shown = Math.min(tableBottom, viewHeight) - Math.max(tableTop, 0);
  if (shown < rowBottom - rowTop) return 0;
  return rowBottom <= 0 ? rowTop : rowBottom - viewHeight;
}

/** x in pixels for `seconds` inside the [t0, t1] window, clamped to the width. */
export function playheadX(seconds, t0, t1, width) {
  const span = t1 - t0;
  if (!(span > 0)) return 0; // a degenerate window must not divide by zero
  const ratio = (seconds - t0) / span;
  return Math.min(width, Math.max(0, ratio * width));
}

/**
 * The time a click at `px` means inside the SAME window.
 *
 * The exact inverse of `playheadX`, so a click and the playhead it starts
 * agree: a click on the pixel the playhead is drawn at lands on the second
 * the playhead reads. Clamped to the window, because a click on the canvas
 * edge is still a click (and a gesture outside the take must not seek past
 * the recording).
 */
export function timeAtX(px, t0, t1, width) {
  const span = t1 - t0;
  if (!(span > 0) || !(width > 0)) return t0;
  const ratio = Math.min(1, Math.max(0, px / width));
  return t0 + ratio * span;
}
