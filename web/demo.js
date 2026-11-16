// A short SUNG take, generated in the page, so the listening path can be
// tried without a microphone (and on a machine with none at all).
//
// The important decision is that this is a SIGNAL, not a note list: the demo
// feeds these samples through the same `LivePitchStream` the microphone and a
// dropped file use, so the notes the page shows are the ones the real chain
// decides. A hand-written note table would demonstrate the table, not the
// reading, and could disagree with it forever without anybody noticing —
// `web/test/segmenter.test.mjs` runs the chain over this signal and pins the
// notes it finds.
//
// The synthesis keeps the traits the segmenter is sensitive to: a fast level
// rise at each onset (the attack tracker's evidence), a harmonic stack around
// a steady fundamental (YIN's periodicity), a vibrato shallow enough to stay
// inside the hold window, a short silence between notes (so each one closes
// where a singer would breathe), and a few cents of detuning per note so the
// cents column has something true to say.

export const DEMO_SR = 44100;

const VIBRATO_HZ = 5.0;
const VIBRATO_CENTS = 12; // peak: 24 cents peak-to-peak stays inside HOLD_ST
const PEAK = 0.22;
const ATTACK_S = 0.012;
const RELEASE_S = 0.04;
const GAP_S = 0.07;
const HARMONICS = [1, 0.45, 0.2, 0.1];

/**
 * Whether the demo should start another pass of itself.
 *
 * The loop belongs to the DEMO, not to playback: an ordinary take ends where
 * it ends, and only the demo button asks for a repeat. Keeping the decision
 * here rather than inline makes the cases that must not loop (a stopped
 * playback, a take that produced no notes) a testable fact.
 */
export function demoRepeats({ looping, mode, notes }) {
  return Boolean(looping) && mode === "mix" && notes.length > 0;
}

/**
 * Whether the demo loop survives a seek.
 *
 * A click in the roll is a REPOSITION, not a stop: the singer wants to hear
 * that passage again (and again) while they move the balance. Only a real
 * stop gesture — the button, Space, the listen buttons — ends the loop.
 */
export function demoLoopsAfterSeek({ looping, playing }) {
  return Boolean(looping) && Boolean(playing);
}

/** The phrase, in order. `cents` is where the singer sat off the key. */
export const DEMO_NOTES = [
  // The four low notes lean sharp together (measured ~+15 cents each): that is
  // what makes the take's LOW register a named habit, which the demo a page
  // shipped without would never show. They still sit inside the "close" band
  // one by one — the habit is the group's mean, not any single note.
  { midi: 60, seconds: 0.55, cents: 8 },
  { midi: 62, seconds: 0.55, cents: 6 },
  { midi: 64, seconds: 0.7, cents: -9 },
  { midi: 62, seconds: 0.55, cents: 4 },
  { midi: 60, seconds: 0.55, cents: 6 },
  // Deliberately sharp, far enough to earn the "off" verdict: a demo phrase
  // that stayed in its two friendly bands would never show the third one on a
  // real measurement, and the colour scale would go untested in the page.
  { midi: 67, seconds: 0.6, cents: 25 },
];

/**
 * The demo take as mono float samples at `sr`, concatenated exactly as a
 * capture would arrive: silence, note, silence, note...
 */
export function demoTake({ sr = DEMO_SR } = {}) {
  const total = Math.ceil(DEMO_NOTES.reduce((sum, note) => sum + note.seconds + GAP_S, 0) * sr);
  const samples = new Float32Array(total);
  const weight = HARMONICS.reduce((sum, value) => sum + value, 0);
  let cursor = 0;
  for (const note of DEMO_NOTES) {
    const length = Math.round(note.seconds * sr);
    const attack = Math.max(1, Math.round(ATTACK_S * sr));
    const release = Math.max(1, Math.round(RELEASE_S * sr));
    let phase = 0;
    for (let i = 0; i < length && cursor + i < total; i++) {
      const t = i / sr;
      // The note's own intonation plus a shallow vibrato around it.
      const cents = note.cents + VIBRATO_CENTS * Math.sin(2 * Math.PI * VIBRATO_HZ * t);
      const hz = 440 * Math.pow(2, (note.midi - 69 + cents / 100) / 12);
      phase += (2 * Math.PI * hz) / sr;
      let value = 0;
      for (let h = 0; h < HARMONICS.length; h++) value += HARMONICS[h] * Math.sin((h + 1) * phase);
      // A fast rise, a flat body, a short fall: the shape the attack tracker
      // and the segmenter are built to hear.
      const rising = Math.min(1, i / attack);
      const falling = Math.min(1, (length - i) / release);
      samples[cursor + i] = (PEAK * value * Math.min(rising, falling)) / weight;
    }
    cursor += length + Math.round(GAP_S * sr);
  }
  return samples;
}
