// The page's remembered choices: what survives a visit.
//
// A singer tunes the page to a room and a habit — how loud the voice sits
// under the piano, which language they read, which comparison they were
// listening to. All three are worth keeping, and all three are the same kind
// of value: a string in `localStorage` that the page must be able to ACT on.
// Keeping the keys and the tolerant readers here means no control has to know
// how its choice is stored.
//
// Every read is tolerant on purpose. A stored value outlives the code that
// wrote it, so absent, garbage and out-of-range readings all fall back to the
// caller's default instead of reaching a GainNode or a translation table.

/** Language of the page's own words: "fr" or "en". */
export const LANG_KEY = "v2m-web-lang";

/** The listening mode the roll and Space resume: "piano" or "mix". */
export const MODE_KEY = "v2m-web-mode";

/** 0 = piano alone, 1 = voice alone. */
export const BALANCE_KEY = "v2m-web-balance";

/**
 * The page's paper: "light" or "dark", under the desktop's OWN name.
 *
 * The desktop keeps the appearance in `~/.v2m_theme` and writes exactly those
 * two words (a malformed value falls back to dark). A static page cannot read
 * that file — there is no filesystem in a browser — so the shared contract is
 * the vocabulary: the same name and the same two values, which is what makes
 * a value carried from one to the other — a launcher, a person, a future
 * bridge — mean the same thing without translation.
 */
export const THEME_KEY = "v2m_theme";

/** The two themes the page can paint. */
export const THEMES = ["light", "dark"];

/**
 * The theme the page opens in.
 *
 * A stored choice always wins: a singer who turned the lights down at night
 * does not want the page to follow the sun at the next launch. Without one the
 * page follows the system, which is the first visit's only evidence of what
 * the room looks like.
 */
export function themeFor({ stored, prefersDark }) {
  return readStoredChoice(stored, THEMES) ?? (prefersDark ? "dark" : "light");
}

/** The modes a listen gesture can resume. */
export const LISTEN_MODES = ["piano", "mix"];

/**
 * The mode a Space press or a roll click resumes, from the remembered choice.
 *
 * The stored value decides, with one repair: a remembered "mix" needs a
 * recording to put over the piano, and a take without one would resume into
 * a playback that never starts. The preference itself is NOT rewritten — the
 * next take with a voice should still open in the mode the singer chose.
 */
export function resumeMode({ stored, hasVoice }) {
  const mode = readStoredChoice(stored, LISTEN_MODES) ?? "piano";
  return mode === "mix" && !hasVoice ? "piano" : mode;
}

/** A stored choice, but only one of the values the page can act on. */
export function readStoredChoice(raw, allowed) {
  if (typeof raw !== "string") return null;
  return allowed.includes(raw) ? raw : null;
}

/**
 * A stored balance, or null when it cannot be trusted.
 *
 * `Number(null)` and `Number("")` are 0, which would silently pin the mix to
 * "piano only" — hence the explicit emptiness check rather than a bare
 * coercion.
 */
export function readStoredBalance(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0 || value > 1) return null;
  return value;
}
