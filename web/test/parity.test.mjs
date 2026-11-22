// The page's reading of a SUNG phrase, against the desktop's own.
//
// `file.test.mjs` proves the file path works on two tones any YIN can find; it
// cannot say whether the port DECIDES what the desktop decides once the audio
// has a glide, a re-attack and a breath in it. `tests/data/parity.wav` is that
// phrase and `tests/reference/parity.json` is the desktop's live chain read
// over the very same file (`scripts/export_web_fixtures.py` writes both), so
// this is the one check that holds the two implementations to one reading of
// one piece of audio. `tests/web_matrix` replays it in all three engines.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const { chooseFile, download, startBrowser } = await import("./browser.mjs");
const { readArchive } = await import("./archive.mjs");
const { PARITY_GOLDEN, PARITY_TAKE, assertParity } = await import("./parity.mjs");

const WEB = fileURLToPath(new URL("..", import.meta.url));
const WORK = await mkdtemp(join(tmpdir(), "v2m-parity-"));

let driver = null;
async function page() {
  driver ??= await startBrowser({ root: WEB, width: 1280, height: 900 });
  return driver;
}

test.after(async () => {
  if (driver) await driver.close();
});

test("a chosen phrase is read as the desktop's live chain reads it", async () => {
  const page_ = await page();
  await chooseFile(page_, PARITY_TAKE, async () => {
    const rows = await page_.evaluate('document.querySelectorAll("#rows tr").length');
    const bundled = !(await page_.evaluate('document.querySelector("#bundle").disabled'));
    return rows > 0 && bundled;
  });

  const archive = readArchive(await download(page_, "bundle", join(WORK, "bundle"), ".zip"));
  const entry = archive.find((item) => item.name === "notes.json");
  assert.ok(entry, `the archive holds ${archive.map((item) => item.name).join(", ")}`);
  const document = JSON.parse(Buffer.from(entry.data).toString("utf8"));
  assert.equal(document.source, "file", `the archive names its source "${document.source}"`);
  assertParity(document, JSON.parse(await readFile(PARITY_GOLDEN, "utf8")));
});
