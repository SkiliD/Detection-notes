// The exported archive, checked from the OUTSIDE.
//
// A writer tested with its own reader would prove only that the two agree. So
// the checks below are independent: the checksums are compared with the
// PUBLISHED CRC-32 vectors, and the archive is walked through its raw headers
// (end record -> central directory -> local header -> data) exactly the way an
// unzip tool does. These checks read the BYTES a reader gets, so a layout that
// drifts fails here even while the writer still agrees with itself.
import assert from "node:assert/strict";
import test from "node:test";
import { deflateRawSync, inflateRawSync } from "node:zlib";

import { buildBundle, notesDocument, wavBytes } from "../bundle.js";
import { browserDeflate, crc32, zipArchive, zipStore } from "../zip.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/**
 * Walk an archive by its headers, as a reader would.
 *
 * `inflate` is the independent decompressor: zlib's own, never the writer's.
 */
function readArchive(bytes, { inflate = (raw) => new Uint8Array(inflateRawSync(raw)) } = {}) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const endAt = bytes.length - 22;
  assert.equal(view.getUint32(endAt, true), 0x06054b50, "no end of central directory record");
  const count = view.getUint16(endAt + 10, true);
  const directoryAt = view.getUint32(endAt + 16, true);
  assert.equal(directoryAt + view.getUint32(endAt + 12, true), endAt, "the directory size is wrong");

  const entries = [];
  let at = directoryAt;
  for (let i = 0; i < count; i++) {
    assert.equal(view.getUint32(at, true), 0x02014b50, `entry ${i}: bad central signature`);
    const crc = view.getUint32(at + 16, true);
    const method = view.getUint16(at + 10, true);
    const packedSize = view.getUint32(at + 20, true);
    const size = view.getUint32(at + 24, true);
    const nameLength = view.getUint16(at + 28, true);
    const localAt = view.getUint32(at + 42, true);
    const name = decoder.decode(bytes.subarray(at + 46, at + 46 + nameLength));

    assert.equal(view.getUint32(localAt, true), 0x04034b50, `${name}: bad local signature`);
    assert.equal(view.getUint32(localAt + 14, true), crc, `${name}: the two headers disagree on the CRC`);
    assert.equal(view.getUint16(localAt + 8, true), method, `${name}: the two headers disagree on the method`);
    const dataAt = localAt + 30 + view.getUint16(localAt + 26, true);
    const raw = bytes.subarray(dataAt, dataAt + packedSize);
    const data = method === 8 ? inflate(raw) : raw;
    assert.equal(data.length, size, `${name}: the entry is not ${size} bytes long`);
    entries.push({ name, crc, method, raw, data, size });
    at += 46 + nameLength;
  }
  return entries;
}

const FIXED_DATE = new Date(2026, 10, 14, 21, 12, 30);

test("the CRC is the published one", () => {
  // IEEE 802.3 / ZIP CRC-32, from the reference vectors: a wrong polynomial or
  // a missing final inversion agrees with itself and with no reader.
  assert.equal(crc32(encoder.encode("123456789")), 0xcbf43926);
  assert.equal(crc32(encoder.encode("hello")), 0x3610a686);
  assert.equal(crc32(new Uint8Array()), 0);
});

test("the archive is readable by its own raw structure", () => {
  const entries = [
    { name: "a.txt", bytes: "first" },
    { name: "dossier/b.bin", bytes: new Uint8Array([0, 1, 2, 250]) },
  ];
  const archive = zipStore(entries, { date: FIXED_DATE });
  const read = readArchive(archive);

  assert.deepEqual(
    read.map((entry) => entry.name),
    ["a.txt", "dossier/b.bin"],
    "the entries must keep the order they were written in",
  );
  assert.equal(decoder.decode(read[0].data), "first");
  assert.deepEqual([...read[1].data], [0, 1, 2, 250]);
  for (const entry of read) {
    assert.equal(entry.crc, crc32(entry.data), `${entry.name}: the stored CRC is not of its data`);
  }
});

test("the same entries and date give the same bytes", () => {
  const entries = [{ name: "x.txt", bytes: "same" }];
  const first = zipStore(entries, { date: FIXED_DATE });
  const second = zipStore(entries, { date: FIXED_DATE });
  assert.deepEqual([...first], [...second], "the archive is not reproducible");
  const later = zipStore(entries, { date: new Date(2026, 10, 15, 8, 0, 0) });
  assert.notDeepEqual([...first], [...later], "the timestamp never reaches the archive");
});

test("the WAV is 16-bit mono at the take's rate", () => {
  const samples = new Float32Array([0, 0.5, -0.5, 1.5, -1.5]);
  const bytes = wavBytes(samples, 44100);
  const view = new DataView(bytes.buffer);

  assert.equal(decoder.decode(bytes.subarray(0, 4)), "RIFF");
  assert.equal(decoder.decode(bytes.subarray(8, 12)), "WAVE");
  assert.equal(decoder.decode(bytes.subarray(12, 16)), "fmt ");
  assert.equal(view.getUint16(20, true), 1, "not PCM");
  assert.equal(view.getUint16(22, true), 1, "not mono");
  assert.equal(view.getUint32(24, true), 44100);
  assert.equal(view.getUint16(34, true), 16, "not 16-bit");
  assert.equal(decoder.decode(bytes.subarray(36, 40)), "data");
  assert.equal(view.getUint32(40, true), samples.length * 2, "the data chunk is the wrong size");

  // and the samples survive: half scale (the negative side lands one step
  // inside, because PCM scales by 32767 and rounds toward +∞ — one LSB of
  // asymmetry, pinned here so it cannot drift silently), and out-of-range input
  // is CLAMPED rather than wrapped: a wrap turns a loud note into a loud click.
  assert.equal(view.getInt16(44, true), 0);
  assert.equal(view.getInt16(46, true), 16384);
  assert.equal(view.getInt16(48, true), -16383);
  assert.equal(view.getInt16(50, true), 32767);
  assert.equal(view.getInt16(52, true), -32767);
});

test("notes.json carries the measurements AND the settings behind them", () => {
  const document = notesDocument({
    notes: [
      { start: 0.25, end: 0.75, midi: 62, strength: 0.7, dev: 0.08 },
      { start: 0.8, end: 1.2, midi: 64, dev: null, voiceDoubt: true },
    ],
    sr: 44100,
    seconds: 1.5,
    source: "file",
    language: "fr",
    stamp: "2026-11-14-21-12",
  });

  assert.equal(document.format, "v2m-web-bundle/1");
  assert.equal(document.source, "file");
  assert.equal(document.duration_s, 1.5);
  assert.equal(document.settings.fmin_hz > 0, true);
  assert.equal(document.settings.window_ms > 0, true);

  const [first, second] = document.notes;
  assert.deepEqual(
    [first.index, first.start_s, first.duration_s, first.midi, first.name, first.cents, first.strength],
    [1, 0.25, 0.5, 62, "D4", 8, 0.7],
  );
  assert.equal(second.cents, null, "an unmeasured note must not export a zero");
  assert.equal(second.voice_doubt, true);
});

test("a deflated entry carries raw DEFLATE and the CRC of the original", async () => {
  // Method 8 is a raw DEFLATE stream: a zlib-wrapped one would be copied into
  // the file with its header as data, so the compressor is pinned to the raw
  // format here, and the header's CRC is checked against the ORIGINAL bytes —
  // a CRC taken over the compressed ones agrees with itself and with no reader.
  const text = "x".repeat(200);
  const entries = [
    { name: "a.txt", bytes: text },
    { name: "tiny.bin", bytes: new Uint8Array([0, 1, 2, 250]) },
  ];
  const archive = await zipArchive(entries, {
    date: FIXED_DATE,
    compress: async (bytes) => new Uint8Array(deflateRawSync(bytes)),
  });
  const read = readArchive(archive);

  assert.equal(read[0].method, 8, "the compressible entry was stored");
  assert.ok(read[0].raw.length < read[0].size, "the packed entry is not smaller");
  assert.equal(decoder.decode(read[0].data), text);
  assert.equal(read[0].crc, crc32(encoder.encode(text)), "the CRC is not the original's");
  // Four bytes cannot get smaller: deflating them would GROW the archive.
  assert.equal(read[1].method, 0, "a tiny entry was deflated anyway");
});

test("the stored fallback is the archive this page always wrote", async () => {
  // A browser without `CompressionStream` must still get a valid ZIP, and the
  // stored path writes the same bytes as `zipStore` always has: an archive does
  // not depend on whether the browser can compress.
  const entries = [
    { name: "a.txt", bytes: "first" },
    { name: "dossier/b.bin", bytes: new Uint8Array([0, 1, 2, 250]) },
  ];
  const fallback = await zipArchive(entries, { date: FIXED_DATE, compress: async () => null });
  assert.deepEqual([...fallback], [...zipStore(entries, { date: FIXED_DATE })]);
});

test("a compressor that throws costs the compression, not the archive", async () => {
  // The one path a failed `CompressionStream` takes: the singer asked for a
  // download, and a browser that cannot compress must not answer with nothing.
  const entries = [{ name: "a.txt", bytes: "y".repeat(64) }];
  const refused = await zipArchive(entries, {
    date: FIXED_DATE,
    compress: async () => {
      throw new Error("no deflate here");
    },
  });
  assert.deepEqual([...refused], [...zipStore(entries, { date: FIXED_DATE })]);
  // ... and a compressor that GROWS an entry is ignored the same way.
  const grown = await zipArchive(entries, {
    date: FIXED_DATE,
    compress: async (bytes) => new Uint8Array(bytes.length + 10),
  });
  assert.deepEqual([...grown], [...zipStore(entries, { date: FIXED_DATE })]);
});

test("the deflate reads a stream that has no async iterator", async () => {
  // WebKit's `ReadableStream` is not async-iterable: `for await` over the
  // compressor throws there, the failure is caught as designed, and every
  // entry is stored — still a valid archive. The loop must use a reader, so
  // the archive compresses in every engine that has raw DEFLATE.
  const native = globalThis.CompressionStream;
  globalThis.CompressionStream = class extends native {
    constructor(format) {
      super(format);
      Object.defineProperty(this.readable, Symbol.asyncIterator, { value: undefined });
    }
  };
  const text = "z".repeat(4000);
  try {
    const packed = await browserDeflate(encoder.encode(text));
    assert.ok(packed, "the compressor was skipped altogether");
    assert.ok(packed.length < 100, `the entry did not compress (${packed.length} bytes)`);
    assert.equal(inflateRawSync(Buffer.from(packed)).toString(), text);
  } finally {
    globalThis.CompressionStream = native;
  }
});

test("the real browser deflate produces an archive zlib can read", async () => {
  // No injected compressor: this is `CompressionStream("deflate-raw")` itself,
  // decompressed by zlib, so the two implementations have to agree on the
  // format. Skipped where the runtime has no raw deflate.
  let probe = null;
  try {
    probe = new CompressionStream("deflate-raw");
  } catch {
    probe = null;
  }
  if (!probe) {
    // The page's own fallback path is covered above; nothing to compare here.
    assert.ok(true);
    return;
  }
  const samples = new Float32Array(2000);
  for (let i = 0; i < samples.length; i++) samples[i] = 0.3 * Math.sin(i / 20);
  const notes = [{ start: 0, end: 0.04, midi: 60, dev: 0 }];
  const bundle = await buildBundle({
    notes,
    samples,
    sr: 44100,
    midi: new Uint8Array([0x4d, 0x54, 0x68, 0x64]),
    readme: "notice\n",
    readmeName: "LISEZ-MOI.txt",
    source: "demo",
    language: "fr",
    stamp: "2026-11-14-21-12",
    date: FIXED_DATE,
  });
  const read = readArchive(bundle.bytes);
  assert.equal(read[0].name, "prise.wav");
  assert.equal(read[0].method, 8, "the audio was not deflated");
  assert.deepEqual([...read[0].data], [...wavBytes(samples, 44100)]);
  assert.ok(read[0].raw.length < read[0].size, "the audio did not shrink");
  // The archive as a whole, against the SAME four entries stored: comparing it
  // with a one-entry archive would only say the take is bigger than a WAV.
  const stored = await buildBundle({
    notes,
    samples,
    sr: 44100,
    midi: new Uint8Array([0x4d, 0x54, 0x68, 0x64]),
    readme: "notice\n",
    readmeName: "LISEZ-MOI.txt",
    source: "demo",
    language: "fr",
    stamp: "2026-11-14-21-12",
    date: FIXED_DATE,
    compress: async () => null,
  });
  assert.ok(stored.bytes.length > 0, "the stored bundle came back empty");
  assert.ok(bundle.bytes.length < stored.bytes.length, "the archive is not smaller than a stored one");
});

test("the bundle holds the audio first, then the reading, then the MIDI", async () => {
  const samples = new Float32Array(441);
  const notes = [{ start: 0, end: 0.01, midi: 60, dev: 0 }];
  const bundle = await buildBundle({
    notes,
    samples,
    sr: 44100,
    midi: new Uint8Array([0x4d, 0x54, 0x68, 0x64]),
    readme: "notice\n",
    readmeName: "LISEZ-MOI.txt",
    source: "demo",
    language: "fr",
    stamp: "2026-11-14-21-12",
    date: FIXED_DATE,
  });

  assert.equal(bundle.name, "prise-web-2026-11-14-21-12.zip");
  const read = readArchive(bundle.bytes);
  assert.deepEqual(
    read.map((entry) => entry.name),
    ["prise.wav", "notes.json", "prise.mid", "LISEZ-MOI.txt"],
  );

  // the audio is the one entry that cannot be rebuilt from the others
  assert.equal(decoder.decode(read[0].data.subarray(0, 4)), "RIFF");
  const document = JSON.parse(decoder.decode(read[1].data));
  assert.equal(document.notes.length, 1);
  assert.equal(document.duration_s, 0.01, "the JSON's duration must be the audio's");
  assert.deepEqual([...read[2].data], [0x4d, 0x54, 0x68, 0x64]);
  assert.equal(decoder.decode(read[3].data), "notice\n");
});
