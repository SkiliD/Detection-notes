// Streaming attack evidence, ported from `src/v2m/core/onsets.py`.
//
// Two voices, both measured there: a LEVEL rise above the recent floor (with
// a DEEP dip required first, so an amplitude tremolo never splits a held
// note) and a NOISE-FLAT BURST that replaces the vowel inside its own block.
// The tuning, the refused alternatives (spectral flux goes blind on a
// vibrato) and every margin live in onsets.py; this port carries the same
// arithmetic so a web take hears its re-articulations where the desktop does.

import {
  DEEP_DIP_RATIO,
  DEEP_DIP_WINDOW_S,
  DIP_RATIO,
  FLATNESS_TRIGGER,
  MIN_GAP_S,
  STREAM_ALPHA,
  STREAM_ALPHA_DOWN,
  STREAM_FLOOR,
  STREAM_LEVEL,
  SUBWINDOW_DIP,
} from "./constants.js";
import { rfftPower } from "./fft.js";

// The tuning of this tracker is declared once, in the generated
// `constants.js` (read from onsets.py); re-exported here for its readers.
export {
  DEEP_DIP_RATIO,
  DEEP_DIP_WINDOW_S,
  DIP_RATIO,
  FLATNESS_TRIGGER,
  MIN_GAP_S,
  STREAM_ALPHA,
  STREAM_ALPHA_DOWN,
  STREAM_FLOOR,
  STREAM_LEVEL,
  SUBWINDOW_DIP,
};

// Page-local sizes: buffer lengths of this port, with no Python twin.
export const MAX_RECENT = 256;
export const SUBWINDOWS = 8;

export class AttackTracker {
  constructor({
    levelRatio = STREAM_LEVEL,
    floorRms = STREAM_FLOOR,
    minGapS = MIN_GAP_S,
    alpha = STREAM_ALPHA,
    alphaDown = STREAM_ALPHA_DOWN,
    hfRatio = FLATNESS_TRIGGER,
  } = {}) {
    this.levelRatio = levelRatio;
    this.floorRms = floorRms;
    this.minGapS = minGapS;
    this.alpha = alpha;
    this.alphaDown = alphaDown;
    this.hfRatio = hfRatio;
    this.strength = 0;
    this.peakRatio = 0;
    this.maxSilentRatio = 0;
    this.recent = [];
    this.baselineRms = 0;
    this.floorRmsSeen = 0;
    this.lastT = -1e9;
    this.primed = false;
    this.hfFloor = 0;
    this.peakRms = 0;
    this.lastDeepT = -1e9;
  }

  static level(block) {
    if (!block.length) return 0;
    let sum = 0;
    for (let i = 0; i < block.length; i++) sum += block[i] * block[i];
    return Math.sqrt(sum / block.length);
  }

  /** Spectral flatness: 1 = white noise, ~0 = a sung vowel's peaks. */
  static flatness(block) {
    if (block.length < 8) return 0;
    // Hann, exactly numpy's `np.hanning` (0.5 - 0.5 cos(2 pi n / (N-1)))
    const windowed = new Float64Array(block.length);
    const denom = block.length - 1;
    for (let i = 0; i < block.length; i++) {
      windowed[i] = block[i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / denom));
    }
    const power = rfftPower(windowed);
    let logSum = 0;
    let sum = 0;
    for (let i = 0; i < power.length; i++) {
      const p = power[i] + 1e-12;
      logSum += Math.log(p);
      sum += p;
    }
    return Math.exp(logSum / power.length) / (sum / power.length);
  }

  /** One captured block; true when it carries an attack. */
  push(t, block) {
    const rms = AttackTracker.level(block);
    if (!this.primed) {
      this.baselineRms = rms;
      this.floorRmsSeen = rms;
      this.peakRms = rms;
      this.hfFloor = this.hfRatio;
      this.primed = true;
      return false;
    }
    const hf = AttackTracker.flatness(block);
    if (hf < this.hfFloor) this.hfFloor = hf;
    else this.hfFloor += 0.35 * (hf - this.hfFloor);

    if (rms > this.peakRms) this.peakRms = rms;
    else this.peakRms += 0.02 * (rms - this.peakRms);

    if (rms < Math.max(DEEP_DIP_RATIO * this.peakRms, this.floorRms)) this.lastDeepT = t;

    let attack = false;
    // Declared OUTSIDE the gate block: the accent below reads it after the
    // block that decides it, and a block-scoped `let` is a ReferenceError on
    // the one path that matters — the block where an attack was found.
    let loud = false;
    const burstLike = hf > FLATNESS_TRIGGER && rms > 0.3 * this.peakRms;
    if (rms >= this.floorRms || burstLike) {
      if (rms <= this.baselineRms && !burstLike) {
        this.floorRmsSeen = Math.min(this.floorRmsSeen, rms);
      } else if (!burstLike) {
        this.floorRmsSeen += 0.5 * (this.baselineRms - this.floorRmsSeen);
      }
      loud = rms > this.levelRatio * this.floorRmsSeen;
      let dips = rms < DIP_RATIO * this.peakRms;
      if (burstLike && dips) {
        // A burst replaces the vowel INSIDE its block; a release block has
        // its silence at the edge, not in the middle.
        const usable = Math.floor(block.length / SUBWINDOWS) * SUBWINDOWS;
        let minSub = Infinity;
        for (let s = 0; s < usable; s += SUBWINDOWS) {
          let acc = 0;
          for (let i = 0; i < SUBWINDOWS; i++) acc += block[s + i] * block[s + i];
          minSub = Math.min(minSub, Math.sqrt(acc / SUBWINDOWS));
        }
        dips = minSub < SUBWINDOW_DIP * this.peakRms;
      }
      const spaced = t - this.lastT >= this.minGapS;
      const deepOk = t - this.lastDeepT <= DEEP_DIP_WINDOW_S;
      loud = loud && deepOk;
      attack = (loud || (burstLike && dips)) && spaced;
    }
    if (attack) {
      this.lastT = t;
      this.recent.push(t);
      if (this.recent.length > MAX_RECENT) this.recent.splice(0, this.recent.length - MAX_RECENT);
      const ratio = rms / Math.max(this.floorRmsSeen, 1e-9);
      this.peakRatio = Math.max(this.peakRatio, ratio);
      this.maxSilentRatio = Math.max(this.maxSilentRatio, ratio);
      // The accent is the level against the recent PEAK, not the re-armed
      // floor: every gap empties that floor, so a floor reference read every
      // mid-take attack as "risen from silence" (onsets.py measured it).
      this.strength = Math.min(1.0, rms / Math.max(this.peakRms, 1e-9));
      this.hfFloor = hf;
      if (loud) {
        this.baselineRms = rms;
        this.floorRmsSeen = rms;
        this.lastDeepT = -1e9;
      }
    } else if (rms < this.baselineRms) {
      this.baselineRms += this.alphaDown * (rms - this.baselineRms);
      this.floorRmsSeen = Math.min(this.floorRmsSeen, this.baselineRms);
    } else {
      this.baselineRms += this.alpha * (rms - this.baselineRms);
    }
    return attack;
  }

  reset() {
    this.baselineRms = 0;
    this.floorRmsSeen = 0;
    this.peakRatio = 0;
    this.maxSilentRatio = 0;
    this.strength = 0;
    this.hfFloor = this.hfRatio;
    this.peakRms = 0;
    this.recent = [];
    this.lastT = -1e9;
    this.primed = false;
    this.lastDeepT = -1e9;
  }
}
