// How far the sung note sat from the key the take names it after, in cents,
// and which of the desktop's three intonation verdicts that distance earns.
//
// The segmenter already publishes the distance: `dev` is the note's plateau
// median minus its rounded key, in semitones, clamped to ±0.5. Converting it
// here is a DISPLAY decision only — nothing re-reads the pitch — so the table
// cannot disagree with the note the take decided, and the cents of a row are
// the same evidence that put the row there.
//
// Whole cents with an explicit sign: the question a lit row raises is "sharp
// or flat, and by how much?", and `+0` would read as a measurement where `0`
// reads as "on the key". The minus is U+2212 — a hyphen beside tabular digits
// is half the width of the plus and makes the column jitter.
//
// The bands are the desktop's (`core.intonation.IN_TUNE_CENTS` 10,
// `gui.widgets.NEAR_CENTS` 25 — green / amber / red), and they read the SAME
// rounded cents the column prints, so a number and its colour can never
// disagree. `tests/test_web_build.py` fails if either band drifts.
import { IN_TUNE_CENTS, NEAR_CENTS } from "./constants.js";

export { IN_TUNE_CENTS, NEAR_CENTS };

/**
 * The printed value of a deviation: whole signed cents, or null.
 *
 * Everything the page says about a note's intonation reads THIS number — the
 * printed cents, the verdict's colour, the take's in-tune share — so a note
 * painted "in tune" can never be counted as sharp in the line beside it.
 */
export function centsOf(dev) {
  if (!Number.isFinite(dev)) return null;
  return Math.round(dev * 100);
}

/** `+7` / `−12` / `0`, or null when the note carries no measurement. */
export function formatCents(dev) {
  const cents = centsOf(dev);
  if (cents === null) return null;
  if (cents === 0) return "0";
  return `${cents > 0 ? "+" : "\u2212"}${Math.abs(cents)}`;
}

/** "ok" | "near" | "off" — the desktop's verdict, or null without a reading. */
export function centsVerdict(dev) {
  const cents = centsOf(dev);
  if (cents === null) return null;
  const distance = Math.abs(cents);
  if (distance <= IN_TUNE_CENTS) return "ok";
  if (distance <= NEAR_CENTS) return "near";
  return "off";
}
