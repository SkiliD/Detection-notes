// A faithful port of the live half of `src/v2m/core/pitch.py`.
//
// The browser cannot run librosa/numba, so the desktop's YIN is reimplemented
// here at the SAME geometry (window/hop in time, ADR 0009). What is mirrored
// is not a detail: `quality` is YIN's own normalized difference at the chosen
// lag, the number `MIN_PITCH_QUALITY` gates and the voice doubt reads, so the
// two front-ends must agree on it or the same voice would be doubted on one
// and trusted on the other.
//
// The measurements, the traps and the reasons live in pitch.py; this file
// carries only what the port needs to reproduce them.

import { FMAX, FMIN, HOP_MS, MIN_PITCH_QUALITY, WINDOW_MS } from "./constants.js";
import { autocorrelation } from "./fft.js";

// The bounds and the frame geometry are declared once, in the generated
// `constants.js`; re-exported here because this is the module their readers
// already import (the segmenter and the two fixture tests).
export { FMAX, FMIN, HOP_MS, MIN_PITCH_QUALITY, WINDOW_MS };

/**
 * One frame's pitch, or NaN. `quality` is d'(tau*) — 0 = perfectly periodic.
 * Returns {hz, quality}; quality is NaN when no lag was accepted.
 */
export function yinPitch(frame, sr, { fmin = FMIN, fmax = FMAX } = {}) {
  const x = frame;
  const width = x.length;
  let peak = 0;
  for (let i = 0; i < width; i++) peak = Math.max(peak, Math.abs(x[i]));
  if (peak < 1e-4) return { hz: NaN, quality: NaN };

  const tauMin = Math.max(2, Math.trunc(sr / fmax));
  const tauMax = Math.min(width - 1, Math.trunc(sr / fmin));
  if (tauMax <= tauMin) return { hz: NaN, quality: NaN };

  // d(tau) = A + B - 2R, the same prefix-sum form as the desktop so both
  // engines subtract the same quantity from the same correlation.
  const cumulative = new Float64Array(width + 1);
  for (let i = 0; i < width; i++) cumulative[i + 1] = cumulative[i] + x[i] * x[i];
  const total = cumulative[width];
  const correlation = autocorrelation(x);

  const difference = new Float64Array(tauMax + 1);
  let running = 0;
  const dPrime = new Float64Array(tauMax);
  for (let tau = 1; tau <= tauMax; tau++) {
    difference[tau] =
      (cumulative[width - tau] + (total - cumulative[tau]) - 2.0 * correlation[tau]) /
      (width - tau);
    running += difference[tau];
    const mean = running / tau;
    dPrime[tau - 1] = difference[tau] / (mean === 0 ? 1e-12 : mean);
  }

  // The FIRST dip under 0.15 that is not still descending: a later dip is a
  // multiple of the period, and picking it halves the pitch.
  let best = -1;
  for (let index = tauMin - 1; index < tauMax - 1; index++) {
    if (dPrime[index] < 0.15) {
      if (index + 1 < dPrime.length && dPrime[index] > dPrime[index + 1]) continue;
      best = index;
      break;
    }
  }
  if (best < 0) {
    const region = dPrime.subarray(tauMin - 1, tauMax);
    if (!region.length) return { hz: NaN, quality: NaN };
    let arg = 0;
    for (let i = 1; i < region.length; i++) if (region[i] < region[arg]) arg = i;
    best = tauMin - 1 + arg;
    if (dPrime[best] > 0.6) return { hz: NaN, quality: NaN };
  }

  const quality = dPrime[best];
  let tau = best + 1;
  if (best > 0 && best < dPrime.length - 1) {
    const left = dPrime[best - 1];
    const middle = dPrime[best];
    const right = dPrime[best + 1];
    const denominator = 2.0 * (left + right - 2.0 * middle);
    if (Math.abs(denominator) > 1e-12) tau += (left - right) / denominator;
  }
  return { hz: sr / tau, quality };
}

/**
 * Incremental YIN with a clock that survives a buffer trim.
 *
 * `offset` must move with the trim: without it the frame clock restarts near
 * 0 and the segmenter never reaches its confirmation window.
 */
export class YinBlock {
  constructor(sr, { window = 1024, hop = 256 } = {}) {
    this.sr = sr;
    this.window = window;
    this.hop = hop;
    this.buffer = new Float64Array(0);
    this.pos = 0;
    this.offset = 0;
    this.prevPeriodic = false;
  }

  push(block) {
    return this.pushWithQuality(block).map((f) => ({ t: f.t, hz: f.hz }));
  }

  /** Same frames WITH YIN's periodicity score (0 = perfect period). */
  pushWithQuality(block) {
    const appended = new Float64Array(this.buffer.length + block.length);
    appended.set(this.buffer);
    appended.set(block, this.buffer.length);
    this.buffer = appended;
    const out = [];
    while (this.pos + this.window <= this.buffer.length) {
      const frame = this.buffer.subarray(this.pos, this.pos + this.window);
      const t = (this.offset + this.pos + this.window / 2) / this.sr;
      const { hz: measured, quality } = yinPitch(frame, this.sr);
      let hz = measured;
      const periodic = Number.isFinite(hz) && quality <= MIN_PITCH_QUALITY;
      // A single unpitched frame inside a periodic run is still a frame of
      // that run: the neighbour rule the desktop applies, unchanged.
      if (Number.isFinite(hz) && !(periodic || this.prevPeriodic)) hz = NaN;
      this.prevPeriodic = periodic;
      out.push({ t, hz, quality });
      this.pos += this.hop;
    }
    const cut = Math.max(0, this.pos - this.window);
    if (cut) {
      this.buffer = this.buffer.slice(cut);
      this.pos -= cut;
      this.offset += cut;
    }
    return out;
  }
}
