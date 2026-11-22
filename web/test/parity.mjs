// The page's reading of ONE sung phrase, held against the desktop's.
//
// `cases.json` hands the port pitch frames and `yin.json` hands it samples;
// neither says what the WHOLE chain — YIN, gate, attacks, segmenter, flush —
// decides about audio someone would actually record. So `tests/data/` holds a
// short phrase as a WAV, and the desktop's own live chain over that same audio
// as `tests/reference/parity.json`. Both are written by
// `scripts/export_web_fixtures.py`;
// `tests/test_web_build.py` refuses a golden that is not what the Python engine
// decides today.
//
// The page is compared on its own terms: `notes.json` — the file a reader
// downloads — through the SAME archive reader the other checks use. Its rate is
// the BROWSER's, not the file's: `decodeAudioData` resamples to the device's
// rate (measured at 96 kHz on a desk whose file is 44.1), so the golden carries
// one frozen reading per rate and this module picks the page's. Measured: at
// each rate the desktop's notes are the page's, boundary for boundary.
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

/** The committed phrase and the desktop readings of it. */
export const PARITY_TAKE = fileURLToPath(new URL("../../tests/data/parity.wav", import.meta.url));
export const PARITY_GOLDEN = fileURLToPath(
  new URL("../../tests/reference/parity.json", import.meta.url),
);

/**
 * How far a note boundary may sit from the desktop's, in seconds.
 *
 * One frame is 11.61 ms, and the two sides read DIFFERENT samples: the browser
 * resamples the file with its own filter, the golden uses a linear one. The
 * bound is two frames, so float noise and that filter cannot spend it, while a
 * moved threshold or a lost guard — which land on frame boundaries further
 * apart than this, or change the notes themselves — cannot hide inside it.
 */
export const PARITY_MAX_SHIFT_S = 0.024;

/** `C4 C4 E4 …`: the labels, so a count or a pitch difference reads at once. */
const labels = (notes) => notes.map((note) => note.name ?? note.midi).join(" ");

/** The desktop reading for the rate the browser decoded at. */
function readingAt(golden, sr) {
  const reading = golden.rates.find((rate) => rate.sr === sr);
  assert.ok(
    reading,
    `the page decoded the file at ${sr} Hz, where no desktop reading exists ` +
      `(recorded: ${golden.rates.map((rate) => rate.sr).join(", ")} Hz). Add the rate to ` +
      "`PARITY_RATES` in `scripts/export_web_fixtures.py` and re-run it",
  );
  return reading;
}

/**
 * Compares the page's `notes.json` with the desktop's reading; returns a line.
 *
 * Throws with both readings in the message: a parity failure that says only
 * "5 != 4" sends the reader back to a browser to find out what moved.
 */
export function assertParity(document, golden) {
  const expected = readingAt(golden, document.sample_rate_hz).notes;
  const read = document.notes.map((note) => ({
    start: note.start_s,
    end: note.end_s,
    midi: note.midi,
    name: note.name,
  }));

  assert.equal(
    read.length,
    expected.length,
    `the page read ${read.length} notes (${labels(read)}) where the desktop read ` +
      `${expected.length} (${labels(expected)}) — this is the whole chain, not a rounding`,
  );
  assert.deepEqual(
    read.map((note) => note.midi),
    expected.map((note) => note.midi),
    `the pitch sequence moved: the page sang ${labels(read)}, the desktop ${labels(expected)}`,
  );

  let worst = 0;
  read.forEach((note, index) => {
    const target = expected[index];
    const shift = Math.max(Math.abs(note.start - target.start), Math.abs(note.end - target.end));
    worst = Math.max(worst, shift);
    assert.ok(
      shift <= PARITY_MAX_SHIFT_S,
      `note ${index + 1} (${note.name}) sits ${(shift * 1000).toFixed(0)} ms from the ` +
        `desktop's — the tolerance is ${(PARITY_MAX_SHIFT_S * 1000).toFixed(0)} ms at ` +
        `${document.sample_rate_hz} Hz: page ${note.start.toFixed(3)}–${note.end.toFixed(3)} s, ` +
        `desktop ${target.start.toFixed(3)}–${target.end.toFixed(3)} s`,
    );
  });
  // The reading must cover the file, not a prefix of it: a chain that stopped
  // early would still pass every bound above on the notes it did read.
  assert.ok(
    Math.abs(document.duration_s - golden.source.samples / golden.source.sr) < 0.05,
    `the take lasts ${document.duration_s} s instead of the file's ` +
      `${(golden.source.samples / golden.source.sr).toFixed(2)} s`,
  );
  return (
    `${read.length} notes (${labels(read)}), worst boundary ` +
    `${(worst * 1000).toFixed(0)} ms at ${document.sample_rate_hz} Hz`
  );
}
