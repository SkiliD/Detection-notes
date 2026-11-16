// A standard MIDI file (format 0) writer: the web page's one output format.
//
// The desktop writes MIDI through `mido`; the browser has no package manager
// at runtime, and a type-0 file with one track is small enough to write by
// hand. Ticks come from the take's own seconds at a fixed 120 BPM, which is
// what the offline pipeline uses when the tempo is unknown — the web page
// does no tempo detection, and inventing one would be worse than saying 120.

const PPQ = 480;
const BPM = 120;

function vlq(value) {
  let v = Math.max(0, Math.round(value));
  const bytes = [v & 0x7f];
  v >>= 7;
  while (v > 0) {
    bytes.unshift((v & 0x7f) | 0x80);
    v >>= 7;
  }
  return bytes;
}

const ticks = (seconds) => Math.round((seconds * BPM * PPQ) / 60);

/**
 * `notes` need {start, end, midi, strength?}. Returns a Uint8Array.
 * Note-offs sort before note-ons at the same tick so a repeated pitch never
 * swallows its own retrigger.
 */
export function writeMidi(notes) {
  const events = [];
  for (const n of notes) {
    const velocity = Math.round(64 + 56 * Math.min(1, Math.max(0, n.strength ?? 0.6)));
    events.push({ tick: ticks(n.start), order: 0, data: [0x90, n.midi & 0x7f, velocity] });
    events.push({ tick: ticks(n.end), order: 1, data: [0x80, n.midi & 0x7f, 64] });
  }
  events.sort((a, b) => a.tick - b.tick || a.order - b.order);

  const track = [];
  let now = 0;
  // Acoustic Grand Piano on channel 0: the page exports a sung line, not a
  // General MIDI arrangement.
  track.push(...vlq(0), 0xc0, 0x00);
  for (const ev of events) {
    track.push(...vlq(ev.tick - now), ...ev.data);
    now = ev.tick;
  }
  track.push(...vlq(0), 0xff, 0x2f, 0x00); // end of track

  const header = [
    0x4d, 0x54, 0x68, 0x64, // MThd
    0, 0, 0, 6,
    0, 0, // format 0
    0, 1, // one track
    (PPQ >> 8) & 0xff, PPQ & 0xff,
  ];
  const trackHeader = [
    0x4d, 0x54, 0x72, 0x6b, // MTrk
    (track.length >>> 24) & 0xff,
    (track.length >>> 16) & 0xff,
    (track.length >>> 8) & 0xff,
    track.length & 0xff,
  ];
  return new Uint8Array([...header, ...trackHeader, ...track]);
}
