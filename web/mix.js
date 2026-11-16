// Voice + piano levels, for judging a take against what it should be.
//
// The page plays the two signals as TWO sources through their own GainNodes,
// so the balance moves the faders while the take keeps sounding — a mix baked
// into one buffer could only change by restarting it. What is shared with the
// desktop is the intent: the piano is normalized by its own renderer, the
// captured voice is brought to a fixed peak (a microphone's level is the
// room's, not the take's), and the piano sits under it.
//
// The crossfade is flat at its middle: each side reaches its reference at 0.5
// and the other fades out over the remaining half, so moving the slider never
// lifts the whole mix, and 0.5 reproduces the balance the page shipped with.
//
// Clipping: the baked mixer could scale the whole sum because it saw it. Live
// faders cannot, so the graph ends in the browser's limiter (app.js) — it only
// touches peaks that would have clipped.

const VOICE_PEAK = 0.85;
const PIANO_GAIN = 0.55;

/** 0.5 = voice at its peak, piano under it. */
export const BALANCE_DEFAULT = 0.5;

/** Largest absolute sample: the reference a level is expressed against. */
export function peakOf(samples) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const value = Math.abs(samples[i]);
    if (value > peak) peak = value;
  }
  return peak;
}

/**
 * The captured voice at a fixed peak, so a gain ratio means the same thing
 * on every take. Silence stays silent instead of being amplified.
 */
export function normalizeVoice(voice, { peak = VOICE_PEAK } = {}) {
  const level = peakOf(voice);
  const samples = new Float32Array(voice.length);
  if (level <= 0) return { samples, peak: 0 };
  const gain = peak / level;
  for (let i = 0; i < voice.length; i++) samples[i] = voice[i] * gain;
  return { samples, peak };
}

/**
 * The two GainNode values for a balance, as RATIOS of each source's peak.
 * Out-of-range input clamps, so a bad stored value cannot invert the sources.
 */
export function balanceGains(balance, { voicePeak = 0, pianoPeak = 0 } = {}) {
  const share = Math.min(1, Math.max(0, Number.isFinite(balance) ? balance : BALANCE_DEFAULT));
  const voiceTarget = VOICE_PEAK * Math.min(1, 2 * share);
  const pianoTarget = PIANO_GAIN * Math.min(1, 2 * (1 - share));
  return {
    voice: voicePeak > 0 ? voiceTarget / voicePeak : 0,
    piano: pianoPeak > 0 ? pianoTarget / pianoPeak : 0,
  };
}
