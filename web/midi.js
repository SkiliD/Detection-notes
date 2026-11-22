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
 * The notes as WRITTEN: a note released at the attack of the same pitch that
 * follows it, when that attack lands before the note's own end.
 *
 * The analysis can leave a same-pitch repeat overlapping the note before it
 * (measured 20 ms on chant.wav and on demo.wav), and written as it stands the
 * earlier OFF lands AFTER the repeat's ON: the reader closes the repeat at its
 * own attack and a 0.18 s note is heard as 0.02 s. Releasing the earlier note
 * at that attack puts both events on ONE instant, where the OFF sorts first,
 * and each note keeps the length that was written for it. The desktop writes
 * its file by the same rule.
 *
 * Copies first: the segmenter's notes are live objects the page still reads,
 * so shortening one in place would edit the take being displayed.
 */
function withRepeatsReleased(notes) {
  const written = notes.map((n) => ({
    start: n.start,
    end: n.end,
    midi: n.midi,
    strength: n.strength,
  }));
  const byStart = written
    .map((_, index) => index)
    .sort((a, b) => written[a].start - written[b].start || written[a].end - written[b].end);
  // One key sounds one note at a time: only the LAST note of a pitch can still
  // be sounding when the next one starts. A different pitch is never touched.
  const sounding = new Map();
  for (const index of byStart) {
    const note = written[index];
    const held = sounding.get(note.midi);
    if (held !== undefined && written[held].end > note.start) written[held].end = note.start;
    sounding.set(note.midi, index);
  }
  return written;
}

/**
 * `notes` need {start, end, midi, strength?}. Returns a Uint8Array.
 *
 * Events are ordered by TIME first, and the kind only decides between two
 * events of the SAME instant: a take's notes meet exactly at their boundary
 * (the segmenter clamps each end to the next start), so a re-attack of one
 * pitch shares a tick with the end of the note before it, and an ON written
 * first would land on the pitch that is still sounding — the reader would
 * close the NEW note instead of the old one, and only its fragment survives.
 *
 * The kind must NOT sort the whole tick: two events of DIFFERENT instants can
 * round to one tick (an ON at 0.5000 and an OFF at 0.5004 both land on tick
 * 480 at 120 BPM), and a global (tick, kind) sort would move the later event
 * ahead of the earlier one. The desktop writes its tick by the same rule.
 */
export function writeMidi(notes) {
  const events = [];
  for (const n of withRepeatsReleased(notes)) {
    const velocity = Math.round(64 + 56 * Math.min(1, Math.max(0, n.strength ?? 0.6)));
    events.push({ time: n.start, order: 1, data: [0x90, n.midi & 0x7f, velocity] });
    events.push({ time: n.end, order: 0, data: [0x80, n.midi & 0x7f, 64] });
  }
  events.sort((a, b) => a.time - b.time || a.order - b.order);
  for (const ev of events) ev.tick = ticks(ev.time);
  // Stable by tick alone: the time order above survives inside a shared tick.
  events.sort((a, b) => a.tick - b.tick);

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
