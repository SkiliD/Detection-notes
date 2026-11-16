// The desktop's piano, ported from `src/v2m/core/playback.py`.
//
// The page must let a singer JUDGE a take by ear without exporting it, and a
// different instrument would judge a different take: this is the same
// additive model — six inharmonic partials whose higher ranks die faster, a
// hammer attack that is fast on a loud key and slow on a soft one, a timbre
// that brightens with the take's own accents, a damper for short notes and a
// pedal tail for held ones, keys tuned a few cents apart deterministically,
// weak notes landing a few ms late. The measurements behind those choices
// live in playback.py; the arithmetic here mirrors it sample for sample,
// which `web/test/cases.json` pins against the Python renderer.
//
// The ONE thing the browser cannot reproduce is the room: the desktop
// convolves a noise impulse response from numpy's PCG64 generator. The page
// generates its own deterministic noise of the same shape and length (same
// 120 ms exponential, same -20 dB wet level), so the room is equivalent in
// level and decay but not sample-identical — the conformance case compares
// the DRY render for that reason.

import {
  BRIGHT_EXPO,
  DARK_EXPO,
  HELD_MIN_S,
  PEDAL_TAIL_FACTOR,
  PEDAL_TAIL_MAX_S,
} from "./constants.js";
import { fft } from "./fft.js";

// The damper and the brightness curve are the desktop piano's own numbers
// (playback.py), declared once in the generated `constants.js`.
export { BRIGHT_EXPO, DARK_EXPO, HELD_MIN_S, PEDAL_TAIL_FACTOR, PEDAL_TAIL_MAX_S };

const PARTIALS = 6;
const ROOM_S = 0.12;
const ROOM_DECAY_S = 0.035;
const ROOM_WET = 0.09;

/** 0..1 velocity from the take's own attack accent, neutral when absent. */
function velocity(note) {
  const value = note.strength;
  if (!Number.isFinite(value)) return 0.5;
  return Math.min(1, Math.max(0, value));
}

/** Deterministic PRNG: the room must sound the same on every play. */
function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  // Box-Muller: numpy uses a ziggurat, but the DISTRIBUTION is what the
  // room's shape needs, not the bit stream.
  let u = 0;
  while (u === 0) u = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

function addTone(y, i0, i1, freq, vel, liftS, sr) {
  const n = i1 - i0;
  const env = new Float64Array(n).fill(1);
  // hammer attack: a loud key strikes fast (3 ms), a soft one slow (12 ms)
  const a = Math.min(n, Math.max(1, Math.trunc((0.003 + 0.009 * (1 - vel)) * sr)));
  for (let i = 0; i < a; i++) env[i] = i / a;
  const expo = DARK_EXPO - (DARK_EXPO - BRIGHT_EXPO) * vel;
  const r = Math.min(n - a, Math.trunc((0.06 + 0.04 * (1 - vel)) * sr));
  if (liftS === null) {
    // a SHORT note: the damper sits on the string and cuts the decay
    if (r > 0) {
      for (let j = 0; j < r; j++) env[n - r + j] *= r === 1 ? 0 : 1 - j / (r - 1);
    }
  } else {
    // a HELD note: the pedal lets it ring, the hammer's direct sound steps
    // down to ~60 % and the tail fades over its second half
    const lift = Math.min(n, Math.trunc(liftS * sr));
    if (lift < n) {
      const step = Math.min(n - lift, Math.max(1, Math.trunc(0.025 * sr)));
      for (let j = 0; j < step; j++) {
        env[lift + j] *= step === 1 ? 1 : 1 - (0.4 * j) / (step - 1);
      }
      for (let j = lift + step; j < n; j++) env[j] *= 0.6;
      const rest = n - lift - step;
      if (rest > 0) {
        const half = Math.floor(rest / 2);
        const count = rest - half;
        for (let j = 0; j < count; j++) env[lift + step + half + j] *= 1 - j / count;
      }
    }
  }
  const scale = 0.25 + 0.75 * vel;
  for (let k = 1; k <= PARTIALS; k++) {
    // a stiff string's partials sit slightly SHARP: the piano's metallic edge
    const fk = freq * k * Math.sqrt(1 + 0.0004 * k * k);
    if (fk > sr / 2) break;
    const tau = 0.9 / Math.pow(k, 0.8);
    const amp = Math.pow(k, -expo);
    const omega = (2 * Math.PI * fk) / sr;
    for (let i = 0; i < n; i++) {
      y[i0 + i] += scale * env[i] * amp * Math.exp(-(i / sr) / tau) * Math.sin(omega * i);
    }
  }
}

function inverseFft(re, im) {
  for (let i = 0; i < im.length; i++) im[i] = -im[i];
  fft(re, im);
  const n = re.length;
  for (let i = 0; i < n; i++) {
    re[i] /= n;
    im[i] = -im[i] / n;
  }
}

/** Overlap-add convolution, truncated to `x.length` (as the desktop's slice). */
function convolveTruncated(x, h) {
  let size = 1;
  while (size < 2 * h.length) size <<= 1;
  const hr = new Float64Array(size);
  const hi = new Float64Array(size);
  hr.set(h);
  fft(hr, hi);
  const out = new Float64Array(x.length);
  const step = size - h.length + 1;
  const xr = new Float64Array(size);
  const xi = new Float64Array(size);
  for (let start = 0; start < x.length; start += step) {
    xr.fill(0);
    xi.fill(0);
    const len = Math.min(x.length - start, step);
    for (let i = 0; i < len; i++) xr[i] = x[start + i];
    fft(xr, xi);
    for (let k = 0; k < size; k++) {
      const re = xr[k] * hr[k] - xi[k] * hi[k];
      const im = xr[k] * hi[k] + xi[k] * hr[k];
      xr[k] = re;
      xi[k] = im;
    }
    inverseFft(xr, xi);
    for (let i = 0; i < size && start + i < out.length; i++) out[start + i] += xr[i];
  }
  return out;
}

function roomTail(y, sr) {
  const random = mulberry32(20177);
  const count = Math.trunc(ROOM_S * sr);
  const ir = new Float64Array(count);
  let peak = 0;
  for (let i = 0; i < count; i++) {
    ir[i] = gaussian(random) * Math.exp(-i / (ROOM_DECAY_S * sr));
    peak = Math.max(peak, Math.abs(ir[i]));
  }
  if (peak > 0) for (let i = 0; i < count; i++) ir[i] = (ir[i] / peak) * ROOM_WET;
  const wet = convolveTruncated(y, ir);
  const out = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) out[i] = y[i] + wet[i];
  return out;
}

/**
 * Render notes to a mono Float32Array, as `render_notes` does.
 * `room: false` gives the dry signal the conformance case compares.
 */
export function renderNotes(notes, { sr = 44100, gain = 1, room = true } = {}) {
  if (!notes.length) return new Float32Array(0);
  const last = Math.max(...notes.map((n) => n.end));
  const length = Math.trunc(last * sr) + Math.trunc(PEDAL_TAIL_MAX_S * sr) + Math.trunc(sr / 10);
  const y = new Float64Array(length);

  for (const note of notes) {
    const vel = velocity(note);
    // A live note carries `duration` as a getter; a plain {start, end} (the
    // conformance fixture, an imported JSON) must not silently render silence.
    const duration = Number.isFinite(note.duration) ? note.duration : note.end - note.start;
    // human placement: a weak note lands a few ms late, a strong one on time,
    // and every key is tuned a few cents off its neighbour deterministically
    const start = note.start + 0.018 * (1 - vel);
    const dur = Math.max(0.03, duration * 0.97);
    const held = duration >= HELD_MIN_S;
    const tail = held ? Math.min(PEDAL_TAIL_MAX_S, PEDAL_TAIL_FACTOR * duration) : 0;
    const end = start + dur + tail;
    const cents = 4.0 * Math.sin(note.start * 12.9898) + 2.5 * Math.sin(note.start * 4.898);
    const frequency = 440 * Math.pow(2, (Math.trunc(note.midi) - 69 + cents / 100) / 12);
    const i0 = Math.max(0, Math.trunc(start * sr));
    const i1 = Math.min(length, Math.trunc(end * sr));
    if (i1 <= i0) continue;
    addTone(y, i0, i1, frequency, vel, held ? dur : null, sr);
  }

  let peak = 0;
  for (let i = 0; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]));
  if (peak > 0) for (let i = 0; i < y.length; i++) y[i] *= 0.85 / peak;
  const tailed = room ? roomTail(y, sr) : y;

  const out = new Float32Array(tailed.length);
  for (let i = 0; i < tailed.length; i++) {
    out[i] = Math.max(-1, Math.min(1, tailed[i] * gain));
  }
  return out;
}
