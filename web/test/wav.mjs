// The one WAV reader the page's tests share.
//
// `yin.test.mjs` and `yin-rates.test.mjs` both compare the browser's estimator
// to the desktop's on `demo.wav`, and both must feed it the very same samples:
// a second copy of this parser is a second place for a bug to hide, and the
// comparison would then be measuring two decoders instead of two YINs.
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

/** A 16-bit mono PCM WAV as float samples: the same numbers the desktop reads. */
export function readPcm16Mono(path) {
  const bytes = readFileSync(path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 12;
  let sampleRate = 0;
  let channels = 0;
  let bits = 0;
  let data = null;
  while (offset + 8 <= bytes.length) {
    const id = String.fromCharCode(...bytes.subarray(offset, offset + 4));
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
    }
    if (id === "data") data = bytes.subarray(body, body + size);
    offset = body + size + (size % 2);
  }
  // A stereo or float take would be interleaved garbage here: say so instead
  // of comparing two different signals and reporting a pitch disagreement.
  assert.equal(channels, 1, `${path} is not mono`);
  assert.equal(bits, 16, `${path} is not 16-bit PCM`);
  const dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const samples = new Float32Array(data.length / 2);
  for (let i = 0; i < samples.length; i++) samples[i] = dataView.getInt16(i * 2, true) / 32768;
  return { sampleRate, samples };
}
