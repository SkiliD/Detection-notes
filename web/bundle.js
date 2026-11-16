// The take as a FOLDER: one archive holding the voice, the measurements, the
// MIDI and a plain-language note, so the same take can be re-analysed later,
// compared with another one, or handed to someone who owns no browser.
//
// Everything here is pure: samples in, bytes out, no DOM, no clock of its own
// (the caller passes `stamp` and, for a reproducible archive, `date`).
import { centsOf } from "./cents.js";
import { FMAX, FMIN, HOP_MS, MIN_PITCH_QUALITY, WINDOW_MS } from "./constants.js";
import { midiToName } from "./segmenter.js";
import { zipArchive } from "./zip.js";

export const BUNDLE_FORMAT = "v2m-web-bundle/1";
export const AUDIO_NAME = "prise.wav";
export const NOTES_NAME = "notes.json";
export const MIDI_NAME = "prise.mid";

const writeAscii = (view, at, text) => {
  for (let i = 0; i < text.length; i++) view[at + i] = text.charCodeAt(i);
};

/**
 * A 16-bit mono WAV of the samples the page kept.
 *
 * 16-bit because that is what the desktop's reader and every audio editor open
 * without asking; the page's own chain works in float, so the conversion is
 * where a sample or two of rounding is lost — the `notes.json` keeps the
 * numbers the page actually measured, so the archive never claims more than the
 * audio proves.
 */
export function wavBytes(samples, sr) {
  const data = samples.length * 2;
  const out = new Uint8Array(44 + data);
  const view = new DataView(out.buffer);
  writeAscii(out, 0, "RIFF");
  view.setUint32(4, 36 + data, true);
  writeAscii(out, 8, "WAVE");
  writeAscii(out, 12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono: the page reads one voice
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(out, 36, "data");
  view.setUint32(40, data, true);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(clamped * 32767), true);
  }
  return out;
}

/**
 * `notes.json`: what the page measured, plus the settings that produced it.
 *
 * The settings are the point of the file. A note list alone cannot be compared
 * with another reading — a different window, a different floor or a different
 * pitch band would explain every difference — so the thresholds travel with the
 * measurements, and `format` names the layout for whatever reads it next.
 */
export function notesDocument({ notes, sr, seconds, source, language, stamp }) {
  return {
    format: BUNDLE_FORMAT,
    generated_at: stamp,
    source,
    language,
    sample_rate_hz: sr,
    duration_s: Number(seconds.toFixed(3)),
    settings: {
      window_ms: WINDOW_MS,
      hop_ms: HOP_MS,
      fmin_hz: FMIN,
      fmax_hz: FMAX,
      min_pitch_quality: MIN_PITCH_QUALITY,
    },
    notes: notes.map((note, index) => {
      const cents = centsOf(note.dev);
      return {
        index: index + 1,
        start_s: Number(note.start.toFixed(4)),
        end_s: Number(note.end.toFixed(4)),
        duration_s: Number((note.end - note.start).toFixed(4)),
        midi: note.midi,
        name: midiToName(note.midi),
        cents: cents === null ? null : Number(cents.toFixed(1)),
        strength: note.strength ?? null,
        voice_doubt: Boolean(note.voiceDoubt),
      };
    }),
  };
}

/**
 * The archive: audio, measurements, MIDI, notice — in that order.
 *
 * The audio comes FIRST because it is the only entry that cannot be rebuilt:
 * the notes are a reading of it, and the MIDI a second reading of the notes.
 *
 * Asynchronous because the entries are compressed on the way in: the browser's
 * deflate is a stream, and a page that blocked on it would freeze while the
 * singer waits for a download. `compress` is injectable so the tests can pack
 * with another implementation and force the stored fallback.
 */
export async function buildBundle({
  notes,
  samples,
  sr,
  midi,
  readme,
  readmeName,
  source,
  language,
  stamp,
  date,
  compress,
}) {
  const document = notesDocument({
    notes,
    sr,
    seconds: samples.length / sr,
    source,
    language,
    stamp,
  });
  const entries = [
    { name: AUDIO_NAME, bytes: wavBytes(samples, sr) },
    { name: NOTES_NAME, bytes: `${JSON.stringify(document, null, 2)}\n` },
    { name: MIDI_NAME, bytes: midi },
    { name: readmeName, bytes: readme },
  ];
  return {
    name: `prise-web-${stamp}.zip`,
    bytes: await zipArchive(entries, { ...(date ? { date } : {}), compress }),
  };
}
