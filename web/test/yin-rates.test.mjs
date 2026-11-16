// The browser's YIN against the desktop's at the two CAPTURE rates.
//
// `web/test/yin.json` proves the ARITHMETIC at the analysis rate (22 050 Hz,
// a 512/256 frame). A device never runs there: the live settings are
// DURATIONS (ADR 0009), so 44.1 kHz opens a 1024/512 frame and 48 kHz a
// 1115/557 one — and 1115 is NOT a power of two, so the difference function's
// autocorrelation is zero-padded to a 4096-point radix-2 transform. A port
// that kept the analysis-rate window, or that assumed a power-of-two frame,
// would pass the 22 050 Hz fixture and still misread every real capture.
// `web/test/yin-rates.json` freezes both geometries with the frames the REAL
// Python estimator reads on the same excerpt.
//
// The fixture stores no audio. Both sides upsample the same float32 excerpt of
// `demo.wav` by linear interpolation, so the two engines read identical
// samples and the comparison is about the YIN, never about two resamplers.
//
// Measured on a 0.95 s window of the take's TAIL (80 frames per rate, 76
// voiced): 0.000013 cents and 5e-7 in d' at both rates. The window is 0.35 s of
// the tonic the last note follows — a short lag, then the JUMP to the pedal's —
// plus 0.6 s of that pedal: a low C2 whose period sits at `tau_max` (339 at
// 22 050 Hz, 678 at 44.1 kHz, 738 at 48 kHz), whose vibrato dips under FMIN, so
// some frames read exactly `sr / tau_max`. A C4..C5 excerpt never reaches that
// edge, and a pedal-only one never exercises a short lag. The bounds below are
// the analysis-rate ones, because a capture-rate reading has no more room for
// drift than any other.
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";
import assert from "node:assert/strict";

import { FMIN } from "../constants.js";
import { HOP_MS, WINDOW_MS, YinBlock } from "../yin.js";
import { readPcm16Mono } from "./wav.mjs";

const FIXTURE = fileURLToPath(new URL("./yin-rates.json", import.meta.url));
const TAKE = fileURLToPath(new URL("../../demo.wav", import.meta.url));

const MAX_CENTS = 0.5;
const MAX_QUALITY = 0.01;

/**
 * The same linear interpolation as `_upsample_linear` in the export script.
 *
 * Every step is the identical double arithmetic on both sides (an integer
 * product, one division, a lerp), and the result is stored back to float32, so
 * the two engines see bit-identical samples.
 */
function upsampleLinear(samples, srIn, srOut) {
  const count = Math.floor((samples.length * srOut) / srIn);
  const out = new Float32Array(count);
  const last = samples.length - 1;
  for (let i = 0; i < count; i++) {
    const position = (i * srIn) / srOut;
    const index = Math.floor(position);
    const fraction = position - index;
    const left = samples[index];
    const right = samples[Math.min(index + 1, last)];
    out[i] = left + (right - left) * fraction;
  }
  return out;
}

const fixture = JSON.parse(readFileSync(FIXTURE, "utf8"));

for (const rate of fixture.rates) {
  test(`the desktop's YIN geometry at ${rate.sr} Hz is the page's own`, () => {
    // The page converts the millisecond settings itself; the frozen window and
    // hop are worth nothing if they no longer come from that rule.
    assert.equal(
      Math.max(64, Math.round((rate.sr * WINDOW_MS) / 1000)),
      rate.window,
      `the ${rate.sr} Hz window no longer follows WINDOW_MS`,
    );
    assert.equal(
      Math.max(64, Math.round((rate.sr * HOP_MS) / 1000)),
      rate.hop,
      `the ${rate.sr} Hz hop no longer follows HOP_MS`,
    );

    let fftSize = 1;
    while (fftSize < 2 * rate.window) fftSize <<= 1;
    assert.equal(fftSize, rate.fft_size, "the difference function's transform size moved");
    assert.ok(rate.tau_min > 0 && rate.tau_max > rate.tau_min, "the lag span is empty");
  });

  test(`every frame's pitch matches the desktop's YIN at ${rate.sr} Hz`, (t) => {
    if (!existsSync(TAKE)) {
      t.skip("demo.wav is absent: run `python scripts/make_demo.py`");
      return;
    }
    const digest = createHash("sha256").update(readFileSync(TAKE)).digest("hex");
    assert.equal(
      digest,
      fixture.source.sha256,
      `${fixture.source.name} is not the take this fixture was built from`,
    );

    const { sampleRate, samples } = readPcm16Mono(TAKE);
    assert.equal(sampleRate, fixture.source.sr);
    assert.equal(samples.length, fixture.source.samples);

    const excerpt = samples.subarray(
      fixture.source.excerpt_start,
      fixture.source.excerpt_start + fixture.source.excerpt_samples,
    );
    const upsampled = upsampleLinear(excerpt, fixture.source.sr, rate.sr);
    assert.equal(upsampled.length, rate.samples, "the two sides built different signals");

    const yin = new YinBlock(rate.sr, { window: rate.window, hop: rate.hop });
    const frames = yin.pushWithQuality(upsampled).map((frame) => [
      frame.t,
      Number.isFinite(frame.hz) ? frame.hz : null,
      Number.isFinite(frame.quality) ? frame.quality : null,
    ]);
    assert.equal(frames.length, rate.frames.length, "the two estimators disagree on the frame count");
    // A fixture of silence would pass every bound below and pin nothing.
    assert.ok(
      rate.frames.some(([, hz]) => hz !== null),
      `${rate.sr} Hz: every frozen frame is unvoiced`,
    );

    let worstCents = 0;
    let worstQuality = 0;
    for (const [index, expected] of rate.frames.entries()) {
      const [frameTime, hz, quality] = frames[index];
      assert.ok(
        Math.abs(frameTime - expected[0]) < 1e-6,
        `frame ${index}: time ${frameTime} vs ${expected[0]}`,
      );
      if (expected[1] === null || hz === null) {
        // A voicing disagreement is not a rounding difference: one engine calls
        // it a note and the other calls it silence, which the take would show.
        assert.equal(hz, expected[1], `frame ${index}: voicing differs (${hz} vs ${expected[1]})`);
        continue;
      }
      const cents = Math.abs(1200 * Math.log2(hz / expected[1]));
      worstCents = Math.max(worstCents, cents);
      worstQuality = Math.max(worstQuality, Math.abs(quality - expected[2]));
    }
    assert.ok(worstCents <= MAX_CENTS, `${rate.sr} Hz: worst pitch difference ${worstCents.toFixed(4)} cents`);
    assert.ok(worstQuality <= MAX_QUALITY, `${rate.sr} Hz: worst periodicity difference ${worstQuality.toFixed(5)}`);
  });
}

test("the excerpt reaches the low end of the lag search", () => {
  // A C4..C5 melody never does: the take's final C2 pedal is what lands a
  // reading at `tau_max`, where a port that widened or narrowed the span
  // reports a different pitch (or calls the frame voiced when the desktop
  // calls it silence). Losing the pedal would leave that limit unexercised
  // while every pitch comparison above still passed.
  for (const rate of fixture.rates) {
    const low = rate.frames.filter(([, hz]) => hz !== null && hz <= FMIN * 1.05);
    assert.ok(low.length, `${rate.sr} Hz: no reading near FMIN — the excerpt lost its low note`);
    // ... and the CROSSING: a short lag (the tonic, above 100 Hz) followed by a
    // reading at the pedal's long one. A pedal-only window reaches the edge above
    // and pins no jump, and neither does a tonic-only one or a take that climbs
    // instead of landing — the order IS the thing this window proves.
    let seenShort = false;
    let crossing = false;
    for (const [, hz] of rate.frames) {
      if (hz === null) continue;
      if (hz <= FMIN * 1.05 && seenShort) {
        crossing = true;
        break;
      }
      if (hz > 100) seenShort = true;
    }
    assert.ok(
      crossing,
      `${rate.sr} Hz: no short-then-long crossing — the lag jump is outside the window`,
    );
  }
});

test("one capture rate opens a window that is not a power of two", () => {
  // This is why the fixture exists: 44.1 kHz lands on 1024 by coincidence, and
  // only 48 kHz's 1115 forces the padded transform. Losing it would leave the
  // padding path untested while the fixture still passed.
  const padded = fixture.rates.filter((rate) => (rate.window & (rate.window - 1)) !== 0);
  assert.ok(padded.length, "no capture rate exercises the FFT padding path");
  for (const rate of padded) {
    assert.ok(
      rate.fft_size > 2 * rate.window,
      `${rate.sr} Hz: the transform no longer pads past the window`,
    );
  }
});
