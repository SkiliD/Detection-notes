// The one browser the web checks need.
//
// `web/test/*.mjs` is the only place a dependency could hide, so the page is
// served from this checkout (a plain static server: GitHub Pages does no more)
// and Chrome is driven over the DevTools protocol through the WebSocket Node 22
// already ships. The 1600 px screenshot and the axe audit share this file, so
// neither owns a copy of the launch flags, the port dance or the readiness
// rule — and the layout width is FIXED by the metrics override rather than
// guessed from a window size.
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
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

/** The page folder, served as GitHub Pages serves it: no build, no rewrite. */
export function servePage(root) {
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
      response.writeHead(200, { "Content-Type": TYPES[extname(path)] ?? "application/octet-stream" });
      response.end(body);
    } catch {
      response.writeHead(404).end("not found");
    }
  });
  return new Promise((ready) => server.listen(0, "127.0.0.1", () => ready(server)));
}

/** One DevTools connection: `send` answers the matching reply, events are noted. */
class Session {
  constructor(url) {
    this.url = url;
    this.waiting = new Map();
    this.last = 0;
    // What the page said while it was being driven: a check that fails with
    // "Uncaught" and nothing else wastes the run it exists for.
    this.notes = [];
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
      if (!call && message.method === "Runtime.exceptionThrown") {
        const details = message.params.exceptionDetails;
        this.notes.push(`thrown: ${details.exception?.description ?? details.text}`);
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

async function startChrome(profile) {
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
        "--mute-audio",
        `--user-data-dir=${profile}`,
        "--remote-debugging-port=0",
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

/**
 * Opens the page at a FIXED layout size, ready to be driven.
 *
 * `width`/`height` are the layout's, not the window's: the metrics override is
 * what makes "1600 px wide" mean 1600 and not the runner's screen. The page is
 * considered ready when its demo button exists, which is the first moment a
 * caller can act on it — the module's handlers are still bound afterwards, so
 * callers that press that button retry until the page answers (see `pressDemo`).
 */
export async function startBrowser({ root, width = 1600, height = 1000 }) {
  const profile = await mkdtemp(join(tmpdir(), "v2m-web-"));
  const server = await servePage(root);
  const origin = `http://127.0.0.1:${server.address().port}/`;
  let chrome = null;
  let session = null;
  try {
    chrome = await startChrome(profile);
    const targets = await (await fetch(`http://127.0.0.1:${chrome.port}/json/list`)).json();
    const page = targets.find((target) => target.type === "page");
    if (!page) throw new Error("the browser opened no page to drive");

    session = new Session(page.webSocketDebuggerUrl);
    await session.open();
    await session.send("Page.enable");
    await session.send("Runtime.enable");
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
      evaluate,
      note: (text) => session.notes.push(text),
      close: async () => {
        const said = session.notes.filter((note) => /^(thrown|error):?/.test(note));
        try {
          session.close();
          chrome.child.kill();
          await new Promise((closed) => server.close(closed));
          await rm(profile, { recursive: true, force: true, maxRetries: 3 }).catch(() => {});
        } finally {
          if (said.length) console.error(`page said:\n  ${said.join("\n  ")}`);
        }
      },
    };
  } catch (error) {
    session?.close();
    chrome?.child.kill();
    await new Promise((closed) => server.close(closed));
    await rm(profile, { recursive: true, force: true, maxRetries: 3 }).catch(() => {});
    const said = session?.notes.length ? `\n  page said: ${session.notes.join("\n  page said: ")}` : "";
    throw new Error(`${error.message}${said}`);
  }
}

/**
 * Presses the page's own demo button until the take lands.
 *
 * The module binds its handlers after the markup is parsed, and a click on a
 * button nobody has wired yet is a no-op — so the press is retried instead of
 * assuming the page was ready when it looked ready. Returns the row count.
 */
export async function pressDemo(page, timeoutMs = 30000) {
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

export { sleep };
