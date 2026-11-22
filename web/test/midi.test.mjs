// The page's MIDI writer: the ORDER of events inside one tick.
//
// `writeMidi` is what the export button hands the browser, and two of its
// promises are invisible in a round-trip through a player: two events of ONE
// instant must be written note-off first (a re-attack of one pitch shares the
// tick with the end of the note before it, and an ON first would close the NEW
// note), and two events of DIFFERENT instants that round to the same tick must
// keep their time order — a global (tick, kind) sort moves the later one ahead
// of the earlier one.
import test from "node:test";
import assert from "node:assert/strict";

import { writeMidi } from "../midi.js";

const PPQ_EXPECTED = 480;
const TICKS_PER_SECOND = 960; // 120 BPM, 480 PPQ: one second is two beats

/** Every note event of the file's one track, with its absolute tick. */
function noteEvents(bytes) {
  assert.equal(String.fromCharCode(...bytes.slice(0, 4)), "MThd", "not a MIDI file");
  const ppq = (bytes[12] << 8) | bytes[13];
  assert.equal(ppq, PPQ_EXPECTED);
  let i = 14;
  assert.equal(String.fromCharCode(...bytes.slice(i, i + 4)), "MTrk", "no track");
  const length = (bytes[i + 4] << 24) | (bytes[i + 5] << 16) | (bytes[i + 6] << 8) | bytes[i + 7];
  i += 8;
  const end = i + length;

  const events = [];
  let tick = 0;
  const vlq = () => {
    let value = 0;
    for (;;) {
      const byte = bytes[i];
      i += 1;
      value = (value << 7) | (byte & 0x7f);
      if (!(byte & 0x80)) return value;
    }
  };
  while (i < end) {
    tick += vlq();
    const status = bytes[i];
    if (status === 0xff) {
      i += 1;
      const type = bytes[i];
      i += 1;
      const metaLength = vlq();
      i += metaLength;
      if (type === 0x2f) break; // end of track
      continue;
    }
    const kind = status & 0xf0;
    if (kind === 0xc0 || kind === 0xd0) {
      i += 2; // one data byte
      continue;
    }
    const note = bytes[i + 1];
    const velocity = bytes[i + 2];
    i += 3;
    if (kind === 0x90 && velocity > 0) events.push({ tick, kind: "on", note });
    else if (kind === 0x80 || kind === 0x90) events.push({ tick, kind: "off", note });
  }
  return events;
}

const at = (events, tick) => events.filter((event) => event.tick === tick).map((e) => [e.kind, e.note]);

test("two events of different instants that share a tick keep their time order", () => {
  // 0.5000 s and 0.5004 s both round to tick 480; the earlier one is an ON.
  // A global (tick, kind) sort would put the OFF first and move the later
  // event ahead of the earlier one.
  const bytes = writeMidi([
    { start: 0.5, end: 1.0, midi: 60, strength: 0.6 },
    { start: 0.0, end: 0.5004, midi: 64, strength: 0.6 },
  ]);
  assert.equal(Math.round(0.5 * TICKS_PER_SECOND), 480);
  assert.deepEqual(at(noteEvents(bytes), 480), [
    ["on", 60],
    ["off", 64],
  ]);
});

test("a re-attack of one pitch writes the off before the on at that instant", () => {
  const bytes = writeMidi([
    { start: 0.0, end: 0.5, midi: 60, strength: 0.6 },
    { start: 0.5, end: 1.0, midi: 60, strength: 0.6 },
  ]);
  assert.deepEqual(at(noteEvents(bytes), 480), [
    ["off", 60],
    ["on", 60],
  ]);
});

/** Every span one pitch sounds for, as [on, off] tick pairs, in file order. */
function spans(events, note) {
  const out = [];
  let open = null;
  for (const event of events) {
    if (event.note !== note) continue;
    if (event.kind === "on") open = event.tick;
    else if (open !== null) {
      out.push([open, event.tick]);
      open = null;
    }
  }
  return out;
}

test("a same-pitch repeat that overlaps releases the note before it", () => {
  // The analysis leaves a 20 ms shared window: the repeat starts at 0.61 s and
  // the note before it ends at 0.63 s. Written as it stands, that OFF lands
  // AFTER the repeat's ON and the reader closes the repeat at its own attack,
  // so a 0.18 s note is heard as 0.02 s. The notes are handed out of order on
  // purpose: the release follows their times, not the array's.
  const repeat = { start: 0.61, end: 0.79, midi: 53, strength: 0.6 };
  const held = { start: 0.47, end: 0.63, midi: 53, strength: 0.6 };
  const events = noteEvents(writeMidi([repeat, held]));

  // 0.47/0.61/0.79 s at 960 ticks per second (rounded): 451, 586, 758.
  assert.deepEqual(spans(events, 53), [
    [451, 586],
    [586, 758],
  ]);
  assert.deepEqual(
    at(events, 586),
    [
      ["off", 53],
      ["on", 53],
    ],
    "the release and the repeat must share one instant, off first",
  );
});

test("a repeat of another pitch is never released", () => {
  // The release is keyed by pitch: a note the next one merely overlaps in time
  // is a harmony of the take, not a repeat, and keeps its written length.
  const events = noteEvents(
    writeMidi([
      { start: 0.0, end: 1.0, midi: 60, strength: 0.6 },
      { start: 0.5, end: 0.9, midi: 67, strength: 0.6 },
    ]),
  );
  assert.deepEqual(spans(events, 60), [[0, 960]]);
  assert.deepEqual(spans(events, 67), [[480, 864]]);
});
