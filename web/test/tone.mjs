// The one 16-bit WAV writer for the checks that feed Chrome's fake microphone.
//
// `capture.test.mjs` drives the real capture path with a tone and `csp.test.mjs`
// drives it to prove the worklet's `blob:` module is allowed; both need the
// same exact bytes a browser accepts as a capture file, and a second writer is
// a second place for the header to drift.
const DEFAULT_PARTS = [
  { hz: 440, seconds: 0.8 },
  { hz: 0, seconds: 0.2 },
  { hz: 587.33, seconds: 0.8 },
];

/** A 16-bit mono PCM WAV: tones and silences, at `rate`. */
export function tonesWav({ rate = 44100, parts = DEFAULT_PARTS } = {}) {
  const samples = [];
  for (const part of parts) {
    for (let i = 0; i < Math.round(part.seconds * rate); i += 1) {
      samples.push(part.hz === 0 ? 0 : Math.sin((2 * Math.PI * part.hz * i) / rate) * 0.35);
    }
  }
  const buffer = Buffer.alloc(44 + samples.length * 2);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + samples.length * 2, 4);
  buffer.write("WAVEfmt ", 8);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(1, 22); // mono
  buffer.writeUInt32LE(rate, 24);
  buffer.writeUInt32LE(rate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(samples.length * 2, 40);
  samples.forEach((value, index) => buffer.writeInt16LE(Math.round(value * 32767), 44 + index * 2));
  return buffer;
}
