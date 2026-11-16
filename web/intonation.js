// The take's intonation, summarised the way the desktop summarises it.
//
// `core.intonation.summarize` is the reference: the share of notes sitting ON
// their key, and a tendency named only when the evidence backs it. Both rules
// are ported rather than invented, because a summary that calls two notes a
// habit is worse than no summary at all:
//
// - MIN_NOTES (4): fewer notes than that in a group means nothing systematic;
// - MIN_CENTS (12): a mean smaller than that is intonation, not a habit, so
//   the take is "even" rather than leaning.
//
// The deviation each note contributes is the cents the TABLE prints
// (`cents.centsOf`), not the raw float: the line describes the column the
// singer is reading, so a note painted "in tune" cannot be missing from the
// in-tune count one line below it.

import { IN_TUNE_CENTS, MIN_CENTS, MIN_NOTES } from "./constants.js";
import { centsOf } from "./cents.js";

// The two bars (fewer notes than MIN_NOTES: nothing is systematic, say
// nothing; a mean under MIN_CENTS is intonation, not a habit) are the
// desktop's own, declared once in the generated `constants.js`.
export { MIN_CENTS, MIN_NOTES };

/**
 * One group's lean, or null when it is not one: the desktop's two conditions.
 * `{ n, mean }` where `mean` is in printed cents.
 */
function tendencyOf(values) {
  if (values.length < MIN_NOTES) return null;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  if (Math.abs(mean) < MIN_CENTS) return null;
  return { n: values.length, mean };
}

/**
 * `{ n, inTune, share, sharp, flat, tendency, mean, byRegister, byPosition }`
 * for a take.
 * `tendency` is "even", "sharp", "flat", or null when the take is too short
 * to say — never a guess. `byRegister` divides the take's MIDI EXTENT into
 * three equal-width bands (never its note count), as the desktop does: a
 * singer who sits flat in the low register only is a fact about registers,
 * and an identically-pitched take has no extent to divide, so every band
 * stays null instead of pretending. `byPosition` splits the take in NOTE-COUNT
 * halves (start/end) and asks the same question there: a singer who drifts flat
 * through the phrase can show a start lean the take-wide mean washes out, and
 * the count — not the clock — is the desktop's rule, so a slow note does not
 * weigh more than a fast one.
 */
export function summarizeIntonation(notes) {
  const cents = [];
  for (const note of notes) {
    const value = centsOf(note.dev);
    if (value !== null) cents.push(value);
  }
  const n = cents.length;
  if (!n) {
    return {
      n: 0,
      inTune: 0,
      share: 0,
      sharp: 0,
      flat: 0,
      tendency: null,
      mean: 0,
      byRegister: { low: null, mid: null, high: null },
      byPosition: { start: null, end: null },
    };
  }
  const onNote = cents.filter((value) => Math.abs(value) <= IN_TUNE_CENTS).length;
  const sharp = cents.filter((value) => value > IN_TUNE_CENTS).length;
  const flat = n - onNote - sharp;
  const mean = cents.reduce((sum, value) => sum + value, 0) / n;
  let tendency = null;
  if (n >= MIN_NOTES) {
    tendency = "even";
    if (Math.abs(mean) >= MIN_CENTS) tendency = mean > 0 ? "sharp" : "flat";
  }

  const midis = notes.filter((note) => centsOf(note.dev) !== null).map((note) => note.midi);
  const lowest = Math.min(...midis);
  const highest = Math.max(...midis);
  const bands = { low: [], mid: [], high: [] };
  const names = ["low", "mid", "high"];
  if (highest > lowest) {
    for (const note of notes) {
      const value = centsOf(note.dev);
      if (value === null) continue;
      const index = Math.min(2, Math.trunc((3 * (note.midi - lowest)) / (highest - lowest)));
      bands[names[index]].push(value);
    }
  }
  const byRegister = {
    low: tendencyOf(bands.low),
    mid: tendencyOf(bands.mid),
    high: tendencyOf(bands.high),
  };
  // An odd count gives its extra note to the END half: the last note is the
  // end of the phrase, where a drift has had time to show.
  const half = Math.trunc(n / 2);
  const byPosition = {
    start: tendencyOf(cents.slice(0, half)),
    end: tendencyOf(cents.slice(half)),
  };
  return {
    n,
    inTune: onNote,
    share: onNote / n,
    sharp,
    flat,
    tendency,
    mean,
    byRegister,
    byPosition,
  };
}
