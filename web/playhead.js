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
//   sample must sit on the edge instead of disappearing.

/** Index of the note sounding at `t`, or -1. Half-open, so gaps stay silent. */
export function noteIndexAt(notes, t) {
  for (let index = 0; index < notes.length; index++) {
    const note = notes[index];
    if (t >= note.start && t < note.end) return index;
  }
  return -1;
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
/**
 * `m:ss.s` for the playback clock, in TENTHS, not in seconds.
 *
 * Rounding in floating seconds rolls 59.96 into "0:60.0"; rounding whole
 * tenths first makes the minute carry exact, and the decimal separator is a
 * parameter because the page is bilingual (French writes a comma).
 */
export function formatClock(seconds, separator = ".") {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  let tenths = Math.round(safe * 10);
  const minutes = Math.floor(tenths / 600);
  tenths -= minutes * 600;
  const rest = (tenths / 10).toFixed(1).padStart(4, "0");
  return `${minutes}:${rest.replace(".", separator)}`;
}

export function timeAtX(px, t0, t1, width) {
  const span = t1 - t0;
  if (!(span > 0) || !(width > 0)) return t0;
  const ratio = Math.min(1, Math.max(0, px / width));
  return t0 + ratio * span;
}
