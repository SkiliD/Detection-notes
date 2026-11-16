// A ZIP, written by the page itself: no library, no build, no server.
//
// The archive is DEFLATED (method 8) when the browser offers a raw deflate
// stream, and STORED (method 0) otherwise: a take is a few seconds of 16-bit
// audio plus a short JSON, so compression halves a download that matters to the
// singer, but an archive this page cannot write is worth far less than a larger
// one it can. The fallback is the same writer with every entry stored, so a
// browser without `CompressionStream` produces the exact bytes this page wrote
// before compression existed. Determinism matters as much: the same entries and
// the same `date` must give the same bytes (per engine — two implementations
// may compress differently, which is why the CRC is computed on the
// UNCOMPRESSED data and never depends on the compressor).
const utf8 = new TextEncoder();

let crcTable = null;

function table() {
  if (crcTable) return crcTable;
  crcTable = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crcTable[n] = c >>> 0;
  }
  return crcTable;
}

/** CRC-32 (the ZIP one, IEEE 802.3), unsigned. */
export function crc32(bytes) {
  const t = table();
  let c = 0xffffffff;
  for (const byte of bytes) c = t[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const asBytes = (value) => (typeof value === "string" ? utf8.encode(value) : value);

// MS-DOS time and date: ZIP's own clock, two-second resolution, 1980 epoch.
function dosTime(date) {
  return {
    time: (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2),
    day: ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate(),
  };
}

/**
 * The three structures every reader requires, over already-decided entries.
 *
 * An entry arrives as `{ name, bytes, method, data }`: `bytes` is the original
 * (what the CRC is computed over and what the header says the size is), `data`
 * is what sits in the file. The two are the same object for a stored entry.
 */
function assemble(entries, { date }) {
  const { time, day } = dosTime(date);
  const chunks = [];
  const central = [];
  let offset = 0;

  for (const entry of entries) {
    const nameBytes = utf8.encode(entry.name);
    const crc = crc32(entry.bytes);

    const local = new Uint8Array(30 + nameBytes.length);
    const localView = new DataView(local.buffer);
    localView.setUint32(0, 0x04034b50, true); // local file header
    localView.setUint16(4, 20, true); // version needed (2.0 covers deflate)
    localView.setUint16(6, 0x0800, true); // UTF-8 name
    localView.setUint16(8, entry.method, true); // 0 stored, 8 deflated
    localView.setUint16(10, time, true);
    localView.setUint16(12, day, true);
    localView.setUint32(14, crc, true);
    localView.setUint32(18, entry.data.length, true); // compressed
    localView.setUint32(22, entry.bytes.length, true); // uncompressed
    localView.setUint16(26, nameBytes.length, true);
    localView.setUint16(28, 0, true); // no extra field
    local.set(nameBytes, 30);
    chunks.push(local, entry.data);

    const record = new Uint8Array(46 + nameBytes.length);
    const recordView = new DataView(record.buffer);
    recordView.setUint32(0, 0x02014b50, true); // central directory entry
    recordView.setUint16(4, 20, true); // version made by
    recordView.setUint16(6, 20, true); // version needed
    recordView.setUint16(8, 0x0800, true);
    recordView.setUint16(10, entry.method, true);
    recordView.setUint16(12, time, true);
    recordView.setUint16(14, day, true);
    recordView.setUint32(16, crc, true);
    recordView.setUint32(20, entry.data.length, true);
    recordView.setUint32(24, entry.bytes.length, true);
    recordView.setUint16(28, nameBytes.length, true);
    recordView.setUint16(30, 0, true); // extra
    recordView.setUint16(32, 0, true); // comment
    recordView.setUint16(34, 0, true); // disk
    recordView.setUint16(36, 0, true); // internal attributes
    recordView.setUint32(38, 0, true); // external attributes
    recordView.setUint32(42, offset, true); // where its local header sits
    record.set(nameBytes, 46);
    central.push(record);

    offset += local.length + entry.data.length;
  }

  const centralSize = central.reduce((size, record) => size + record.length, 0);
  const end = new Uint8Array(22);
  const endView = new DataView(end.buffer);
  endView.setUint32(0, 0x06054b50, true); // end of central directory
  endView.setUint16(8, entries.length, true); // on this disk
  endView.setUint16(10, entries.length, true); // total
  endView.setUint32(12, centralSize, true);
  endView.setUint32(16, offset, true); // where the directory starts
  endView.setUint16(20, 0, true); // no archive comment

  const total = offset + centralSize + end.length;
  const out = new Uint8Array(total);
  let at = 0;
  for (const chunk of [...chunks, ...central, end]) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
}

/**
 * A ZIP archive of `[{ name, bytes }]`, in the order given, every entry STORED.
 *
 * Flags carry UTF-8 names (bit 11) so a file called `LISEZ-MOI.txt` survives an
 * accented sibling.
 */
export function zipStore(entries, { date = new Date() } = {}) {
  return assemble(
    entries.map((entry) => {
      const bytes = asBytes(entry.bytes);
      return { name: entry.name, bytes, method: 0, data: bytes };
    }),
    { date },
  );
}

let rawDeflate = null; // null = not probed, false = the browser has no RAW deflate

/**
 * The browser's own deflate, in the form ZIP wants, or null.
 *
 * `CompressionStream("deflate-raw")` is raw DEFLATE (no zlib header, no
 * checksum), which is exactly what method 8 stores. The plain `"deflate"`
 * format wraps the stream in a zlib header that an unzip tool would copy into
 * the file as garbage, so it is not a substitute. Older browsers, and some
 * hardened ones, have no `CompressionStream` at all: then the caller stores.
 */
export async function browserDeflate(bytes) {
  if (rawDeflate === null) {
    try {
      new CompressionStream("deflate-raw");
      rawDeflate = true;
    } catch {
      rawDeflate = false; // the format is not supported here, whatever the class says
    }
  }
  if (!rawDeflate) return null;
  const stream = new CompressionStream("deflate-raw");
  const writer = stream.writable.getWriter();
  // Write and close without awaiting before reading: a stream is a pipe, and
  // waiting on the write while nothing drains it would wait forever.
  writer.write(bytes).then(() => writer.close());
  // A reader, not `for await`: WebKit's `ReadableStream` has no async
  // iterator, and iterating it throws there — the archive stays valid because
  // the caller stores, but it stores every entry, at 1.5x the size.
  const reader = stream.readable.getReader();
  const parts = [];
  let length = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parts.push(value);
    length += value.length;
  }
  const out = new Uint8Array(length);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

/**
 * A ZIP archive, DEFLATED entry by entry when a compressor is available.
 *
 * `compress(bytes)` may be injected (the tests compare against another
 * implementation); it returns the packed bytes, or null to store that entry.
 * An entry is stored when compressing did not make it smaller, and when the
 * compressor throws — a broken or refused compressor must cost the
 * compression, never the archive.
 */
export async function zipArchive(entries, { date = new Date(), compress = browserDeflate } = {}) {
  const prepared = [];
  for (const entry of entries) {
    const bytes = asBytes(entry.bytes);
    let packed = null;
    try {
      packed = await compress(bytes);
    } catch {
      packed = null;
    }
    const smaller = packed && packed.length < bytes.length;
    prepared.push({
      name: entry.name,
      bytes,
      method: smaller ? 8 : 0,
      data: smaller ? packed : bytes,
    });
  }
  return assemble(prepared, { date });
}
