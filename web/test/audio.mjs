// The page's own Web Audio objects, counted from OUTSIDE.
//
// `playbackContext` and `player` are module state: a test that read them would
// pass on a page whose second take sounds over the first. These wrappers count
// what the page ACTUALLY opens and starts — a graph is "open" until `close()`,
// a source "sounding" between `start()` and `stop()`/`ended` — and they are
// installed before the first gesture, so nothing the page creates later
// escapes the count.
//
// `sources` reads as the number created since the last `markAudio`, so a walk
// can measure one gesture against the one before it without resetting the id
// counter the live set depends on.

/** Wraps `window.AudioContext` and `createBufferSource`, and starts counting. */
export async function watchAudio(page) {
  await page.evaluate(`(() => {
    window.__audio = { open: 0, contexts: 0, sources: 0, mark: 0, live: new Set() };
    window.__audioOriginals = {
      context: window.AudioContext,
      createBufferSource: AudioContext.prototype.createBufferSource,
    };
    const Native = window.AudioContext;
    const makeContext = function (...args) {
      const context = new Native(...args);
      window.__audio.contexts += 1;
      window.__audio.open += 1;
      const close = context.close.bind(context);
      context.close = () => {
        // A graph opened while watching can outlive the watch (a take still
        // sounding when the walk ends): its close runs against the restored
        // page, where the counters are already gone. Counting is best-effort
        // there; closing is not.
        if (window.__audio) window.__audio.open -= 1;
        return close();
      };
      return context;
    };
    makeContext.prototype = Native.prototype;
    window.AudioContext = makeContext;

    const create = Native.prototype.createBufferSource;
    Native.prototype.createBufferSource = function (...args) {
      const source = create.apply(this, args);
      const id = ++window.__audio.sources;
      const start = source.start.bind(source);
      const stop = source.stop.bind(source);
      source.start = (...rest) => {
        window.__audio.live.add(id);
        return start(...rest);
      };
      source.stop = (...rest) => {
        window.__audio.live.delete(id);
        return stop(...rest);
      };
      source.addEventListener("ended", () => window.__audio.live.delete(id));
      return source;
    };
  })()`);
}

/** Graphs open now, sources sounding now, and sources created since the mark. */
export async function audioCounts(page) {
  return page.evaluate(`({
    open: window.__audio.open,
    live: window.__audio.live.size,
    sources: window.__audio.sources - window.__audio.mark,
  })`);
}

/** Makes `sources` count from here, for one gesture's own cost. */
export async function markAudio(page) {
  await page.evaluate("window.__audio.mark = window.__audio.sources");
}

/** Hands the page's prototypes back; the counters stop moving. */
export async function stopWatchingAudio(page) {
  await page.evaluate(`(() => {
    const originals = window.__audioOriginals;
    if (!originals) return;
    window.AudioContext = originals.context;
    AudioContext.prototype.createBufferSource = originals.createBufferSource;
    delete window.__audioOriginals;
    delete window.__audio;
  })()`);
}
