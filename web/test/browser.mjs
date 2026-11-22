// The one browser the web checks need.
//
// `web/test/*.mjs` is the only place a dependency could hide, so the page is
// served from this checkout (a plain static server: GitHub Pages does no more)
// and Chrome is driven over the DevTools protocol through the WebSocket Node 22
// already ships. The 1600 px screenshot and the axe audit share this file, so
// neither owns a copy of the launch flags, the port dance or the readiness
// rule — and the layout width is FIXED by the metrics override rather than
// guessed from a window size. Chrome's fake microphone and the audio-graph
// probe live here for the same reason: the capture, policy, keyboard and
// long-take checks all need them, and a second copy is a second set of flags
// to drift.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { availableParallelism, tmpdir } from "node:os";
import { extname, join, normalize } from "node:path";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wav": "audio/wav",
  ".png": "image/png",
};

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

/** The browser to drive: the environment's choice first, then the usual names. */
function chromeCandidates() {
  const named = process.env.CHROME ? [process.env.CHROME] : [];
  if (process.platform === "win32") {
    return [
      ...named,
      "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
      "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    ];
  }
  if (process.platform === "darwin") {
    return [...named, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"];
  }
  return [...named, "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"];
}

/**
 * The page folder, served as GitHub Pages serves it: no build, no rewrite.
 *
 * `headers` is the ONE caller's exception: the memory budget asks the browser
 * for its own accounting (`performance.measureUserAgentSpecificMemory`), which
 * the platform refuses outside a cross-origin-isolated page. The two headers
 * change nothing the page does — they only make that reading legal — and every
 * other caller serves exactly what Pages sends.
 */
export function servePage(root, host = "127.0.0.1", headers = {}) {
  const server = createServer(async (request, response) => {
    const asked = decodeURIComponent(new URL(request.url, "http://x").pathname);
    const name = asked === "/" ? "index.html" : asked.slice(1);
    const path = join(root, normalize(name));
    if (!path.startsWith(root)) {
      response.writeHead(403).end("outside the page");
      return;
    }
    try {
      const body = await readFile(path);
      response.writeHead(200, {
        "Content-Type": TYPES[extname(path)] ?? "application/octet-stream",
        ...headers,
      });
      response.end(body);
    } catch {
      response.writeHead(404).end("not found");
    }
  });
  // `host` exists for the one check that needs an ORIGIN that is not a secure
  // context: a page on 0.0.0.0 answers on the machine's own address, which is
  // plain http to Chrome. Every other caller keeps loopback.
  return new Promise((ready) => server.listen(0, host, () => ready(server)));
}

/** One DevTools connection: `send` answers the matching reply, events are noted. */
class Session {
  constructor(url) {
    this.url = url;
    this.waiting = new Map();
    // One-off event waits: the keyboard check has to see the browser OPEN a
    // file chooser, which is an event, not a reply.
    this.waitingEvents = new Map();
    this.last = 0;
    // What the page said while it was being driven: a check that fails with
    // "Uncaught" and nothing else wastes the run it exists for.
    this.notes = [];
    // The browser's own log (`Log.entryAdded`): a Content-Security-Policy
    // refusal is reported here, not through `console.error`, so a page that
    // only reads `console` would call a broken policy quiet.
    this.entries = [];
  }

  open() {
    this.socket = new WebSocket(this.url);
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      const call = this.waiting.get(message.id);
      if (!call && message.method === "Runtime.consoleAPICalled") {
        const text = message.params.args.map((arg) => arg.value ?? arg.description).join(" ");
        this.notes.push(`${message.params.type}: ${text}`);
        return;
      }
      if (!call && message.method === "Log.entryAdded") {
        this.entries.push(message.params.entry);
        return;
      }
      if (!call && message.method === "Runtime.exceptionThrown") {
        const details = message.params.exceptionDetails;
        this.notes.push(`thrown: ${details.exception?.description ?? details.text}`);
        return;
      }
      const waiting = !call && this.waitingEvents.get(message.method);
      if (waiting) {
        this.waitingEvents.delete(message.method);
        waiting.done(message.params);
        return;
      }
      if (!call) return;
      this.waiting.delete(message.id);
      if (message.error) call.fail(new Error(message.error.message));
      else call.done(message.result);
    });
    return new Promise((ready, broken) => {
      this.socket.addEventListener("open", () => ready());
      this.socket.addEventListener("error", () => broken(new Error(`cannot reach ${this.url}`)));
    });
  }

  send(method, params = {}) {
    const id = ++this.last;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((done, fail) => this.waiting.set(id, { done, fail }));
  }

  /** Resolves on the next event of that name, or rejects after `timeoutMs`.
   *  Measured: 1 ms alone and 5–16 ms under every file at once, so 1 s is
   *  ~60× the worst contention and a stuck event fails in a second. */
  waitFor(method, timeoutMs = 1000) {
    return new Promise((done, fail) => {
      const timer = setTimeout(() => {
        this.waitingEvents.delete(method);
        fail(new Error(`no ${method} within ${timeoutMs} ms`));
      }, timeoutMs);
      this.waitingEvents.set(method, {
        done: (params) => {
          clearTimeout(timer);
          done(params);
        },
      });
    });
  }

  close() {
    this.socket.close();
  }
}

/** The browser's own port, read from the line it prints when it is ready. */
function devtoolsPort(child) {
  return new Promise((ready, broken) => {
    let seen = "";
    const timer = setTimeout(() => broken(new Error(`no DevTools line:\n${seen}`)), 30000);
    child.stderr.on("data", (chunk) => {
      seen += chunk;
      const found = seen.match(/DevTools listening on ws:\/\/[^:]+:(\d+)\//);
      if (!found) return;
      clearTimeout(timer);
      ready(Number(found[1]));
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      broken(new Error(`the browser exited (${code}) before it was ready:\n${seen}`));
    });
  });
}

async function startChrome(profile, args = [], muteFlags = []) {
  const failures = [];
  for (const binary of chromeCandidates()) {
    const child = spawn(
      binary,
      [
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        // NOT `--mute-audio`: a muted browser never opens an output stream, so
        // an AudioContext cannot leave `suspended` and the capture check would
        // report a dead graph on a machine that has one. Silence is asked for
        // where it matters (`mute`, on for every caller but the capture).
        //
        // The pilot presses buttons through `element.click()`, which is NOT a
        // user gesture: without this policy Chrome keeps the page's audio
        // contexts suspended whatever the hardware is.
        "--autoplay-policy=no-user-gesture-required",
        `--user-data-dir=${profile}`,
        "--remote-debugging-port=0",
        // Caller flags last: the capture check adds Chrome's fake microphone,
        // and no other caller passes any.
        ...muteFlags,
        ...args,
        "about:blank",
      ],
      { stdio: ["ignore", "ignore", "pipe"] },
    );
    if (child.pid === undefined) {
      failures.push(`${binary}: could not start`);
      continue;
    }
    try {
      const port = await devtoolsPort(child);
      return { child, port };
    } catch (error) {
      child.kill();
      failures.push(`${binary}: ${error.message.split("\n")[0]}`);
    }
  }
  throw new Error(`no browser to drive:\n  ${failures.join("\n  ")}`);
}

// ------------------------------------------------------------- browser slots

/**
 * How many browsers this machine can feed at once.
 *
 * Every test file starts its own Chrome, and `node --test` schedules one file
 * per core by default: eighteen browsers on a sixteen-core machine starve the
 * page's REAL-TIME audio graph and the exports' CPU, so a suite that is green
 * alone times out when the glob runs. A quarter of the cores, never fewer than
 * two, is what a page check can take and still answer in a second or two —
 * measured on sixteen cores: four browsers kept every export and file choice
 * under three seconds and the whole run near forty.
 *
 * The cap lives in FILES because the scheduling is not ours: `node --test`
 * gives each test file its own process, so an in-process counter would limit
 * nothing a sibling process could see. Node offers no project-level
 * `--test-concurrency` (the config file is only read with `--config-file`, and
 * `NODE_OPTIONS` refuses the flag), which is why the bare glob needs this.
 */
const BROWSER_SLOTS = Math.max(2, Math.floor(availableParallelism() / 4));
const SLOT_DIR = join(tmpdir(), "v2m-web-browsers");
const SLOT_STALE_MS = 15 * 60 * 1000;

/** One free slot, or an error naming the directory a killed run left behind. */
async function takeBrowserSlot(timeoutMs = 10 * 60 * 1000) {
  await mkdir(SLOT_DIR, { recursive: true });
  const started = Date.now();
  for (;;) {
    for (let slot = 0; slot < BROWSER_SLOTS; slot += 1) {
      const path = join(SLOT_DIR, `slot-${slot}`);
      try {
        // ONE call, exclusive creation: a sibling that finds the file must see
        // the owner's PID, never the empty file between a create and a write.
        await writeFile(path, String(process.pid), { flag: "wx" });
        return path;
      } catch (error) {
        if (error.code !== "EEXIST") throw error;
        await dropStaleSlot(path);
      }
    }
    if (Date.now() - started > timeoutMs) {
      throw new Error(
        `no browser slot after ${(timeoutMs / 60000).toFixed(0)} min: ` +
          `${SLOT_DIR} still holds all ${BROWSER_SLOTS} slots`,
      );
    }
    await sleep(100 + Math.floor(Math.random() * 100));
  }
}

/**
 * Removes a slot whose owner is gone, or one far older than any file lives.
 *
 * A slot is a claim made by a process: killing a run (`Ctrl+C`, a timeout)
 * leaves the file with a dead PID, and the next run must not wait ten minutes
 * for a browser that will never close. A LIVE owner is never evicted by age
 * alone — the clock is the fallback for a recycled PID, not the rule.
 *
 * A slot with NO readable PID is not stale: it belongs to a run that has just
 * claimed it, and evicting it there would put two browsers in one slot (the
 * measured race — the fifth of six requests opened a browser while four were
 * still alive). Only age frees an unreadable slot.
 */
async function dropStaleSlot(path) {
  try {
    const [owner, info] = await Promise.all([readFile(path, "utf8"), stat(path)]);
    const pid = Number(owner);
    const aged = Date.now() - info.mtimeMs > SLOT_STALE_MS;
    if (aged) {
      await rm(path, { force: true });
      return;
    }
    if (!pid) return; // claimed a moment ago, owner not read yet
    try {
      process.kill(pid, 0);
    } catch (error) {
      if (error.code === "ESRCH") await rm(path, { force: true });
    }
  } catch {
    // The slot moved between the read and the stat: the next round sees it.
  }
}

/** Gives one slot back, at most once, however many cleanup paths ask. */
function slotReleaser(path) {
  let held = path;
  return async () => {
    const current = held;
    held = null;
    if (current) await rm(current, { force: true }).catch(() => {});
  };
}

/**
 * Opens the page at a FIXED layout size, ready to be driven.
 *
 * `width`/`height` are the layout's, not the window's: the metrics override is
 * what makes "1600 px wide" mean 1600 and not the runner's screen. The page is
 * considered ready when its demo button exists, which is the first moment a
 * caller can act on it — the module's handlers are still bound afterwards, so
 * callers that press that button retry until the page answers (see `pressDemo`).
 */
export async function startBrowser({
  root,
  url = null,
  width = 1600,
  height = 1000,
  args = [],
  mute = true,
  headers = {},
}) {
  const profile = await mkdtemp(join(tmpdir(), "v2m-web-"));
  // A caller that names a URL is reading a page it does not own — the
  // PUBLISHED site — so nothing is served from this checkout and the browser
  // speaks to the real origin, over the network, as a visitor's would.
  const server = url ? null : await servePage(root, "127.0.0.1", headers);
  const origin = url ?? `http://127.0.0.1:${server.address().port}/`;
  let chrome = null;
  let session = null;
  let releaseSlot = async () => {};
  try {
    // A browser is taken against the machine-wide cap BEFORE it is spawned:
    // the wait is what keeps eighteen files from opening eighteen Chromes.
    releaseSlot = slotReleaser(await takeBrowserSlot());
    // A page can sound (the audit's "a row being heard" state does): the run
    // stays quiet unless the caller NEEDS an output stream to exist, which is
    // exactly the capture check.
    chrome = await startChrome(profile, args, mute ? ["--mute-audio"] : []);
    const targets = await (await fetch(`http://127.0.0.1:${chrome.port}/json/list`)).json();
    const page = targets.find((target) => target.type === "page");
    if (!page) throw new Error("the browser opened no page to drive");

    session = new Session(page.webSocketDebuggerUrl);
    await session.open();
    await session.send("Page.enable");
    await session.send("Runtime.enable");
    await session.send("Log.enable");
    // Record every refusal the policy makes, in the page itself, from the first
    // document on: a violation that fires before a test looks is still counted.
    await session.send("Page.addScriptToEvaluateOnNewDocument", {
      source: `window.__cspViolations = [];
        document.addEventListener("securitypolicyviolation", (event) => {
          window.__cspViolations.push({
            directive: event.violatedDirective,
            blocked: event.blockedURI,
            source: event.sourceFile || "",
            line: event.lineNumber || 0,
          });
        });`,
    });
    await session.send("Emulation.setDeviceMetricsOverride", {
      width,
      height,
      deviceScaleFactor: 1,
      mobile: false,
    });

    const evaluate = async (expression) => {
      const answer = await session.send("Runtime.evaluate", {
        expression,
        returnByValue: true,
        awaitPromise: true,
      });
      if (answer.exceptionDetails) {
        const details = answer.exceptionDetails;
        throw new Error(`the page threw: ${details.exception?.description ?? details.text}`);
      }
      return answer.result.value;
    };

    await session.send("Page.navigate", { url: origin });
    for (let waited = 0; waited < 30000; waited += 100) {
      if (await evaluate("Boolean(document.querySelector('#demo'))")) break;
      await sleep(100);
    }

    return {
      origin,
      send: (method, params) => session.send(method, params),
      waitFor: (method, timeoutMs) => session.waitFor(method, timeoutMs),
      evaluate,
      note: (text) => session.notes.push(text),
      /** Every `securitypolicyviolation` the page recorded, in order. */
      violations: () => evaluate("window.__cspViolations ?? []"),
      /** The browser's own log entries, as `{ source, level, text }`. */
      logs: () =>
        session.entries.map((entry) => ({
          source: entry.source,
          level: entry.level,
          text: entry.text,
        })),
      // What the page wrote to its console, in order: a check that reads the
      // sentence must be able to read the line beside it.
      notes: () => [...session.notes],
      close: async () => {
        const said = session.notes.filter((note) => /^(thrown|error):?/.test(note));
        try {
          session.close();
          chrome.child.kill();
          if (server) await new Promise((closed) => server.close(closed));
          await rm(profile, { recursive: true, force: true, maxRetries: 3 }).catch(() => {});
        } finally {
          await releaseSlot();
          if (said.length) console.error(`page said:\n  ${said.join("\n  ")}`);
        }
      },
    };
  } catch (error) {
    session?.close();
    chrome?.child.kill();
    if (server) await new Promise((closed) => server.close(closed));
    await rm(profile, { recursive: true, force: true, maxRetries: 3 }).catch(() => {});
    await releaseSlot();
    const said = session?.notes.length ? `\n  page said: ${session.notes.join("\n  page said: ")}` : "";
    throw new Error(`${error.message}${said}`);
  }
}

/**
 * Chrome's own fake microphone, fed from the WAV at `path`.
 *
 * `--use-fake-ui-for-media-stream` answers the permission prompt a headless run
 * can never draw, and the file feeds the device in REAL time, as a microphone
 * does. This is a flag list, not a browser: the caller still asks for
 * `mute: false`, because a muted browser opens no output stream and a graph that
 * cannot start never captures.
 */
export function fakeMicrophone(path) {
  return [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream", // no prompt to answer in a headless run
    `--use-file-for-fake-audio-capture=${path}`,
  ];
}

/**
 * The state the page's audio graph reaches, under the page's OWN bound.
 *
 * `AudioContext.resume()` settles only once an output stream can start, so a
 * runner without a sound card leaves the promise pending for good and the state
 * stays `suspended` — the exact fact `startedContext` gives up on, measured here
 * rather than assumed. The wait is 2.5 s, the page's 2 s bound plus a margin, so
 * a slow output is not read as a dead one.
 */
export async function audioState(page, timeoutMs = 2500) {
  return page.evaluate(`
  (async () => {
    const context = new AudioContext({ latencyHint: "interactive" });
    await Promise.race([
      context.resume().catch(() => {}),
      new Promise((done) => setTimeout(done, ${timeoutMs})),
    ]);
    return context.state;
  })()`);
}

/**
 * Presses the page's own demo button until the take lands.
 *
 * The module binds its handlers after the markup is parsed, and a click on a
 * button nobody has wired yet is a no-op — so the press is retried instead of
 * assuming the page was ready when it looked ready. Returns the row count.
 *
 * Measured: 263–264 ms alone and 1.30 s with every file at once under sixteen
 * spinning processes, so 5 s is 3.8× the worst contention measured.
 */
export async function pressDemo(page, timeoutMs = 5000) {
  let rows = 0;
  for (let waited = 0; waited < timeoutMs && rows === 0; waited += 250) {
    await page.evaluate("document.querySelector('#demo').click()");
    await sleep(250);
    rows = await page.evaluate("document.querySelectorAll('tbody tr').length");
  }
  if (rows === 0) {
    const status = await page.evaluate("document.querySelector('#status').textContent.trim()");
    throw new Error(`the demo take produced no notes (status: ${status})`);
  }
  return rows;
}

/**
 * Puts a real file in the page's own input, the way a reader chooses one.
 *
 * Repeated until `ready()` answers: the module binds its `change` handler
 * after the markup is parsed, and a file set before that listener exists is a
 * silent no-op — the same race `pressDemo` retries for the demo button. The
 * input node is re-read every attempt, and a refused node id is retried like
 * a silent no-op: the page rebuilds its DOM when a reading starts, and its
 * FIRST document is still being committed when the harness connects, so a
 * node id read a moment earlier can already be gone by the time it is used.
 *
 * A retry only means something if the page can SEE the file again, so the
 * input is emptied before each set: setting the same path on an input that
 * already holds it fires no `change`, and a first set that landed before the
 * handler was bound would otherwise be retried silently for the whole run.
 *
 * A document that never answers gets ONE clean document (`Page.reload`) before
 * the helper gives up. A browser that just closed can still hold the output
 * device the page's DECODER opens, and then the decode never answers and
 * `ready()` stays false for the rest of that document — a harness transient,
 * not a fact any caller is testing. Every caller that chooses a file goes
 * through here, so the second chance is not a per-test copy. The failure names
 * the page's own status and whatever it threw or logged, because "never
 * answered" alone leaves the next reader guessing.
 *
 * The 10 s budget is the export wait's, for the same contention (measured
 * 0.3–0.5 s alone, 4.7 s with the machine at twice its core count), and it is
 * spent PER ATTEMPT: at most two documents, so a dead harness fails in 20 s.
 */
export async function chooseFile(page, path, ready, { timeoutMs = 10000 } = {}) {
  await page.send("DOM.enable");
  const notesAtStart = page.notes().length;
  const started = Date.now();
  let last = "the page never answered";
  // Two documents at most: the one the harness connected to, then one clean one.
  for (let attempt = 0; attempt < 2; attempt += 1) {
    if (attempt > 0) {
      await page.send("Page.reload");
      // `Page.reload` answers before the new document is parsed, and `ready()`
      // reads the page's own nodes: wait for the markup the harness already
      // treats as ready rather than letting the predicate throw on a
      // half-committed page.
      for (let waited = 0; waited < 10000; waited += 100) {
        let there = false;
        try {
          there = await page.evaluate("Boolean(document.querySelector('#file'))");
        } catch {
          there = false; // a context destroyed by the navigation: look again
        }
        if (there) break;
        await sleep(100);
      }
    }
    for (let waited = 0; waited < timeoutMs; waited += 250) {
      try {
        // Empty the input FIRST: `change` only fires when the value moves, so
        // an unchanged retry would reach nobody. Clearing fires nothing itself.
        await page.evaluate(`(() => {
          const input = document.querySelector("#file");
          if (input) input.value = "";
        })()`);
        const { root } = await page.send("DOM.getDocument", { depth: 1 });
        const { nodeId } = await page.send("DOM.querySelector", {
          nodeId: root.nodeId,
          selector: "#file",
        });
        assert.ok(nodeId, "the page has no file input to press");
        await page.send("DOM.setFileInputFiles", { nodeId, files: [path] });
      } catch (error) {
        last = error.message; // a stale node: the next attempt re-reads it
      }
      await sleep(250);
      if (await ready()) {
        const took = ((Date.now() - started) / 1000).toFixed(1);
        console.log(`chose ${path} in ${took} s`);
        return;
      }
    }
  }
  let status = "unreadable";
  try {
    status = await page.evaluate('document.querySelector("#status")?.textContent.trim() ?? ""');
  } catch {}
  const said = page
    .notes()
    .slice(notesAtStart)
    .filter((note) => /^(thrown|error)/.test(note));
  const why = said.length ? `; page said: ${said.join(" | ")}` : "";
  const took = ((Date.now() - started) / 1000).toFixed(1);
  throw new Error(
    `choosing ${path} never reached the page in ${took} s (last: ${last}; status: "${status}"${why})`,
  );
}

/**
 * Clicks an export button and answers the bytes the page BUILT.
 *
 * A file in a folder is the BROWSER's artifact: it appears after a rename, at a
 * moment no page fact names, and a full run can starve that write long after the
 * export happened. The page's own promise is the Blob it hands to the anchor,
 * so the blob is read inside the page — no folder, no wait, nothing to race.
 * The download-to-disk path keeps its own checks where a file is the subject:
 * `file.test.mjs`, `keyboard.test.mjs` and `parity.test.mjs` all read one.
 *
 * The wrappers are the ONE place blob URLs are observed: they record creation
 * and revocation, so a caller can also check the page kept the URL alive past
 * the click. A disabled button builds no blob, and the failure names the status.
 *
 * Measured: 2–8 ms alone and 40–152 ms with every file at once under sixteen
 * spinning processes — 1 s is 6.5× the worst contention measured.
 */
export async function exportedBytes(page, button, { timeoutMs = 1000 } = {}) {
  await page.evaluate(`(() => {
    window.__exports = window.__exports ?? [];
    // Installed ONCE: a second wrapper would record every blob twice and
    // revoke only the first entry, leaving a phantom "never released".
    if (window.__exportsWrapped) return;
    window.__exportsWrapped = true;
    const create = URL.createObjectURL.bind(URL);
    const revoke = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = (blob) => {
      const url = create(blob);
      window.__exports.push({ url, blob, createdAt: performance.now(), revokedAt: null });
      return url;
    };
    URL.revokeObjectURL = (url) => {
      const entry = window.__exports.find((item) => item.url === url);
      if (entry && entry.revokedAt === null) entry.revokedAt = performance.now();
      return revoke(url);
    };
  })()`);
  const before = await page.evaluate("window.__exports.length");
  await page.evaluate(`document.querySelector("#${button}").click()`);
  for (let waited = 0; waited < timeoutMs; waited += 50) {
    const answer = await page.evaluate(`(async () => {
      if (window.__exports.length <= ${before}) return null;
      const entry = window.__exports[${before}];
      const bytes = new Uint8Array(await entry.blob.arrayBuffer());
      return { bytes: Array.from(bytes), createdAt: entry.createdAt, revokedAt: entry.revokedAt };
    })()`);
    // A Buffer, as `download()` answers: the callers decode with `.equals`,
    // `.subarray` and `toString("latin1")` on the bytes they get back.
    if (answer) return { ...answer, bytes: Buffer.from(answer.bytes) };
    await sleep(50);
  }
  let status = "unreadable";
  try {
    status = await page.evaluate('document.querySelector("#status")?.textContent.trim() ?? ""');
  } catch {}
  throw new Error(`#${button} built no file in ${timeoutMs} ms (status: "${status}")`);
}

/**
 * Watches `#rows` from now on: `start` is what it holds before the next take,
 * `min` the lowest count seen after.
 *
 * A capture test must know that the table it reads belongs to the CURRENT take,
 * and the page already holds an earlier transcription (a demo, a previous take).
 * An observer catches the empty state between the two, which a poll can miss.
 */
export async function traceRows(page) {
  await page.evaluate(`(() => {
    const body = document.querySelector("#rows");
    const read = () => body.querySelectorAll("tr").length;
    window.__rowTrace = { start: read(), min: read() };
    window.__rowTraceObserver?.disconnect();
    window.__rowTraceObserver = new MutationObserver(() => {
      const n = read();
      if (n < window.__rowTrace.min) window.__rowTrace.min = n;
    });
    window.__rowTraceObserver.observe(body, { childList: true, subtree: true });
  })()`);
}

/** The trace `traceRows` installed. */
export function rowTrace(page) {
  return page.evaluate("window.__rowTrace");
}

/**
 * Clicks a download button and answers the bytes the page wrote.
 *
 * The budget is for CONTENTION, not for hiding a slow export: `node --test`
 * runs one Chrome per file, and a machine with fewer cores than files (or a
 * runner shared with other work) can leave an export queued far past the few
 * seconds it takes alone. Measured on 16 cores: 0.2–0.6 s alone, 2.2 s with
 * every file at once under 16 spinning processes, 4.7 s under 32 (twice the
 * cores) — so the budget is 10 s, 2.1× the worst measured, where the old 30 s
 * would have hidden a genuinely stuck export for half a minute. Every export
 * PRINTS what it took, so a wait that grows shows its number even in a green
 * run, and a timeout names the budget and the page's own status.
 */
export async function download(page, button, folder, extension, { timeoutMs = 10000 } = {}) {
  await mkdir(folder, { recursive: true });
  // A folder that already holds one of these would answer this wait with the
  // previous run's file — the one way a wait can "succeed" on an export that
  // never happened. Every export therefore gets its own folder, and a caller
  // that reuses one is refused here rather than served a stale file.
  const stale = (await readdir(folder)).filter((name) => name.endsWith(extension));
  assert.equal(
    stale.length,
    0,
    `${folder} already holds ${stale.join(", ")}: the wait would answer with a stale file`,
  );
  await page.send("Page.setDownloadBehavior", { behavior: "allow", downloadPath: folder });
  await page.evaluate(`document.querySelector("#${button}").click()`);
  const started = Date.now();
  for (let waited = 0; waited < timeoutMs; waited += 200) {
    const names = (await readdir(folder)).filter((name) => name.endsWith(extension));
    if (names.length) {
      const path = join(folder, names[0]);
      // The final name IS the finished file: Chrome writes `<name>.crdownload`
      // and renames it when the download completes (measured on a 20 MB blob:
      // the temporary name at 72 ms, the final one, full size, at 107 ms), so
      // a name ending in `extension` is never half-written and is read at once.
      console.log(`#${button} wrote ${names[0]} in ${((Date.now() - started) / 1000).toFixed(1)} s`);
      return readFile(path);
    }
    await sleep(200);
  }
  let status = "unreadable";
  try {
    status = await page.evaluate('document.querySelector("#status")?.textContent.trim() ?? ""');
  } catch {}
  throw new Error(
    `the page produced no ${extension} within ${(timeoutMs / 1000).toFixed(0)} s (status: "${status}")`,
  );
}

export { sleep };
