// Every number the page WRITES, from its one formatter.
//
// `toFixed` writes a dot in every language, so the clock, the Hz line, the
// table's seconds and the roll's second marks disagreed with a French page that
// reads `3,2`. One `Intl.NumberFormat` per language and decimal count, grouping
// OFF — these are seconds and hertz inside a measure, never money — and the
// three lines built on it: the playback clock, the live reading's hertz, and
// the intonation sentence. Pure, so a separator is checked without a browser.
//
// `tests/test_web_numbers.py` refuses a second formatter and a displayed number
// built by hand at its call site; `web/test/format.test.mjs` pins these
// functions; `web/test/numbers.test.mjs` reads the rendered page in both
// languages.

const NUMBER_FORMATS = new Map();

export function formatNumber(value, digits, lang) {
  const key = `${lang}:${digits}`;
  let format = NUMBER_FORMATS.get(key);
  if (!format) {
    format = new Intl.NumberFormat(lang === "fr" ? "fr-FR" : "en-GB", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      useGrouping: false,
    });
    NUMBER_FORMATS.set(key, format);
  }
  return format.format(Number.isFinite(value) ? value : 0);
}

/**
 * `m:ss.s` for the playback clock, in TENTHS, not in seconds.
 *
 * Rounding in floating seconds rolls 59.96 into "0:60.0"; rounding whole
 * tenths first makes the minute carry exact, and `formatNumber` writes the
 * tenths, so the clock shares the page's one separator.
 */
export function formatClock(seconds, lang = "en") {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  let tenths = Math.round(safe * 10);
  const minutes = Math.floor(tenths / 600);
  tenths -= minutes * 600;
  const rest = formatNumber(tenths / 10, 1, lang).padStart(4, "0");
  return `${minutes}:${rest}`;
}

/** The Hz line, or an empty slot when no finite reading ever arrived. */
export function formatHz(hz, lang) {
  return Number.isFinite(hz) ? `${formatNumber(hz, 1, lang)} Hz` : "";
}

/**
 * The three fill values of the intonation sentence.
 *
 * The share is the whole percent the sentence prints, rounded BEFORE it becomes
 * a string, so a future decimal would follow the language instead of a dot.
 */
export function intonationNumbers(summary, lang) {
  return {
    in_tune: formatNumber(summary.inTune, 0, lang),
    n: formatNumber(summary.n, 0, lang),
    share: formatNumber(Math.round(summary.share * 100), 0, lang),
  };
}
