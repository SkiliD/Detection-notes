// What an exported archive contains, read from OUTSIDE it.
//
// Two drivers take an archive apart — Chrome in `file.test.mjs` and
// `parity.test.mjs`, Playwright in `tests/web_matrix` — and a reader written
// three times is a reader that drifts. The walk is by RAW HEADERS (end record
// → central directory → local header → data), with zlib's own inflate and CRC
// as the independent half: `CompressionStream("deflate-raw")` is the thing
// being checked, so the page's writer may never be its own reader.
import assert from "node:assert/strict";
import { crc32, inflateRawSync } from "node:zlib";

/** The entries of a ZIP, in the order its central directory lists them. */
export function readArchive(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const endAt = bytes.length - 22;
  assert.equal(view.getUint32(endAt, true), 0x06054b50, "no end of central directory record");
  const count = view.getUint16(endAt + 10, true);
  let at = view.getUint32(endAt + 16, true);
  const entries = [];
  for (let index = 0; index < count; index += 1) {
    assert.equal(view.getUint32(at, true), 0x02014b50, `entry ${index}: bad central signature`);
    const method = view.getUint16(at + 10, true);
    const packedSize = view.getUint32(at + 20, true);
    const size = view.getUint32(at + 24, true);
    const nameLength = view.getUint16(at + 28, true);
    const localAt = view.getUint32(at + 42, true);
    const name = bytes.subarray(at + 46, at + 46 + nameLength).toString("utf8");
    assert.equal(view.getUint32(localAt, true), 0x04034b50, `${name}: bad local signature`);
    const dataAt = localAt + 30 + view.getUint16(localAt + 26, true);
    const raw = bytes.subarray(dataAt, dataAt + packedSize);
    const data = method === 8 ? new Uint8Array(inflateRawSync(raw)) : raw;
    assert.equal(data.length, size, `${name}: the entry is not ${size} bytes long`);
    assert.equal(crc32(data), view.getUint32(at + 16, true), `${name}: the stored CRC is not of its data`);
    entries.push({ name, method, data });
    at += 46 + nameLength;
  }
  return entries;
}
