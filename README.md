**English** | [Français](README.fr.md)

# Voice → MIDI

Turn a sung melody into notes. Sing into the microphone or open an audio file, correct the transcription, then export MIDI, MusicXML, JSON, CSV — and a PDF when a score tool is installed. The audio is analysed locally and never uploaded.

Three ways in: the desktop app (`v2m gui`), the live command (`v2m live`), and a static web page in `web/` that runs the listening half in the browser. The desktop reads files with pYIN and therefore also gives tempo, meter, key, rests, notation and the PDF; the page uses the faster YIN on the live path only, which is why the two can disagree about a note boundary.

## Install

Python 3.12 is recommended, on Windows, macOS or Linux.

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

The repository installs from pyproject's ranges; a local constraint file (kept out of the repository) can pin the library versions the audio references were measured on, because without pins a resolver may install a newer librosa or numba and report a different engine as a regression.

Below, `python` and `pytest` mean that environment (on Windows without activation, use `.venv/Scripts/python.exe`).

## Use the app

```bash
v2m gui
```

- **Sing** records a take, which stays on the results page until you export it. Corrections, note moves and deletions are one undoable transaction, and a failed one leaves the reading and the history untouched.
- **Review** offers one next action: listen to the passages the coherence check flagged, in order, then export. A doubt is never an automatic correction, and **Skip to export** keeps every step optional.
- **Views**: Notes (the piano roll, where a take is corrected), Score and Waveform, one at a time. **Show details** adds the staff, the diagnostics and the export options, and starts collapsed.
- **Replace**: select a range, re-sing it, then keep or cancel the replacement; the take before it is kept.
- **Intonation**: the share of notes sung in tune, a named tendency, and leans by register (low / mid / high) and by half (start / end) — always from the measured cents. A detected key renames a note without rewriting its measurement.
- **Export** opens Save As (MIDI, MusicXML, PDF, JSON, notes CSV) and remembers the last folder; a missing PDF tool skips only the PDF. Live takes also keep their full audit archive in `prises_live/`.
- **Note finder**: sing a note or click a piano key to identify it. The answer card holds the name, the tuning needle and the mic's reception; confirmed notes stay as playable chips.
- **Appearance**: Dark or Light, saved in `~/.v2m_theme`. Both palettes keep every word legible, including the ones the app paints itself.
- **Language and controls**: the window keeps the web page's craft floors. French numbers read the French way (`1,0 s`, not `1.0 s`) because one formatter writes every decimal, the roll's second marks included; every control measures at least 24 px in both directions; and each focusable control draws the same 2 px copper ring (3:1 against the surface behind it) when the keyboard reaches it.

### Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| Tab | Walk the page's actions in task order |
| Ctrl+Z / Ctrl+Y | Undo / redo the last note edit (Ctrl+Shift+Z also redoes) |
| Up / Down | Move the selection one note |
| Home / End | Jump to the first / last note |
| Space | Audition the selected note, or stop it |
| Left / Right | Choose a pitch on the focused piano (it plays); Enter applies it, Escape cancels |
| Delete | Delete the selected range |

## Web version

`web/` is a static page — no build, no dependency, nothing uploaded — that runs the listening half in the browser: pitch, notes, accents, a piano roll, the intonation lines and a type-0 MIDI export. **Everything in one archive** downloads the take as one ZIP: `prise.wav`, `notes.json`, `prise.mid` and a short notice, deflated when the browser can and stored when it cannot.

To look at it locally (the page loads its strings from a file):

```bash
cd web
python -m http.server 8000
```

To replay the page's two measurements on their own, without reading the workflow (Node 22 and a Chrome/Chromium the harness can find; `CHROME` overrides it):

```bash
mkdir -p build
node web/test/resolutions.mjs
curl -sSfL -o build/axe.min.js https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js
node web/test/axe-audit.mjs --axe build/axe.min.js
```

The `curl` line downloads the same pinned axe-core 4.10.2 the workflow uses, and `node web/test/axe-audit.mjs` runs the audit itself, which covers both palettes in three visible states at all eight widths.

To publish it, set **Settings → Pages → Source** to **GitHub Actions**, then push to `main`: the workflow replays the port's conformance cases, photographs the page at 1600 px, measures eight resolution classes (320 px to 2560 px, a landscape phone included), audits both palettes in three visible states at each of them with axe-core, then deploys `web/` — and finally reads the published URL back, over the network: headers, both files the page needs, the demo chain and the language switch, so a page served from a repository subfolder as well as from a site root has to work for a visitor, not just in a checkout. It also fails on a page that is not a **secure context** (`isSecureContext` and `https:`), because outside one the browser hides the microphone entirely: **Settings → Pages → Enforce HTTPS** is a measured precondition of every deployment, not a setting taken on trust. That reading runs **after** the deploy, so it judges a page that is already online: a red run means the site is broken for visitors, and nothing is withdrawn automatically — no rollback, no unpublish. The remedy is to correct it and push again.

Publication also requires the **microphone** job: the page's recording path is driven on Chrome's own fake device with a generated tone, and a red capture now blocks the deploy instead of warning beside it. The only way past that gate is the repository variable `MIC_CAPTURE_WAIVED=true` — a deliberate, logged exception for a broken runner recipe, not for a broken page; the run then carries a warning naming what was not proven, and the waiver is to be taken back as soon as the job is green. Only a maintainer can set it — it is a repository setting, not a file — and it is one command each way, run from this checkout:

```bash
gh variable set MIC_CAPTURE_WAIVED --body true
gh variable delete MIC_CAPTURE_WAIVED
```

The same two values live in **Settings → Secrets and variables → Actions → Variables** when `gh` is not at hand. The waiver takes effect on the next run: set it, re-run the failed deployment, then remove it once `capture` is green.

The page's own tests drive **one** engine, the runner's Chrome. A WebKit-only defect already shipped that way: its `ReadableStream` has no async iterator, so the archive's compressor threw and every entry was stored — a valid ZIP, 1.5× too big, with a green suite. `.github/workflows/tests.yml` therefore replays the five checks that matter (the ported segmenter against the desktop's cases, YIN at the live geometry, the exported archive, a file you choose, and a short sung phrase read by the desktop's own live chain) in Chromium, Firefox **and** WebKit. The phrase is `tests/data/parity.wav`, committed with the desktop's reading of it in `tests/reference/parity.json` at every rate a browser can decode it to, so the page's `notes.json` is compared with the desktop note for note instead of with a fixture of pitch frames. The harness lives in `tests/web_matrix/` and borrows a pinned Playwright: `web/` keeps no dependency.

A long take's **memory budget** is measured, not assumed: on demand, the page is driven through a **three-minute** capture on Chrome's fake device and the browser's own accounting is read before the take, after it and after the archive — the take is held once while it is sung, and an export releases the concatenation it packed, so the archive adds no second copy — the run fails past an explicit ceiling. The steady cost is **4 bytes per sample** (11.5 MB per minute at 48 kHz, 23 MB at 96 kHz) plus a few MB of page, and an export peaks at about **2.5× the take** while it packs — measured at 178 MB for the three-minute take and held to a 2.8× ceiling by the run itself — so budget that headroom once on a phone. It is opt-in because three minutes of real time are too much for every push:

```bash
V2M_LONG_TAKE=1 node --test web/test/long-take.test.mjs
V2M_LONG_TAKE=1 V2M_LONG_TAKE_RATE=48000 node --test web/test/long-take.test.mjs
```

A machine with no audio output device skips it, and `V2M_REQUIRE_AUDIO=1` turns that skip into a failure. `V2M_LONG_TAKE_RATE` forces the graph's rate, so the same budget can be re-run where the device reports 48 kHz (measured there: 34.6 MB signal, +36.3 MB after capture, +36.4 MB after the archive) instead of trusting the 96 kHz one; a browser that refuses the rate fails rather than measuring the old one.

The page also ships its own **Content-Security-Policy**, declared in the head: everything from `'self'`, plus the one `blob:` the audio worklet's module is built from — no CDN, no inline script, no `eval`, so the policy needs no `'unsafe-inline'`. `web/test/csp.test.mjs` pins the directives and drives the page (demo, playback, the archive, the real capture) while the page records every `securitypolicyviolation` and the browser's own log is read for a refusal.

The page is also walked **by keyboard alone** (`web/test/keyboard.test.mjs`): real Tab, Enter and Space events pin the focus order twice — before a take and once it holds notes — open the file chooser from its input, and export both files. Space is the page's only shortcut and it is refused while a control owns the key, so one Space on **Listen to the take** starts it once, and one Space on the balance moves nothing. A held arrow resumes the take once: its auto-repeats are ignored and a seek already restarting the take refuses a second one, so holding the key cannot race two starts into "playback could not start". Every control that takes focus draws the page's own 2 px copper ring, and `tests/test_web_build.py` reads that ring from the sheet beside the markup for each one, so a renamed selector fails the ordinary suite before the browser walk runs. `web/test/craft-floor.test.mjs` measures the ring the browser actually paints: 3:1 against the surface behind it in both palettes — the masthead pills are measured on the night band, not the paper — and `tests/test_web_build.py` pins the same pair from the tokens (copper on paper: 3.74:1 light, 8.19:1 dark).

What a screen reader hears from the readout is the **note name**, not the frame stream: `#note` is the page's only polite live region, rewritten solely when the name changes, while the Hz line beside it sits outside that region because it moves on every frame. `web/test/craft-floor.test.mjs` pins the placement and both change gates.

Every number the page shows — the clock's tenths, the Hz line, the table's seconds, the roll's second marks, and the intonation line's note counts and percent — is written by one `Intl.NumberFormat`, so a French page reads `3,2 s` where an English one reads `3.2 s`; `web/test/format.test.mjs` pins both and `web/test/craft-floor.test.mjs` checks the rendered table in each language.

A page whose system asks for reduced motion silences every transition — the buttons, the links and the level meter's smoothing included — and `web/test/craft-floor.test.mjs` reads both the block and the rules it names, then emulates the reader's setting in Chrome and requires the computed transitions of `button`, `.link` and `.meter-fill` to be `0s` in both palettes, while `tests/test_web_build.py` re-derives every animating family from the sheet and requires it inside the block, so neither a renamed selector nor a new transition can leave a control moving.

Headings balance their lines and running prose wraps `pretty`, and an inactive control — button, text link or the balance slider's thumb — keeps its word (or its shape) on a quiet field instead of fading the whole control away.

The recording you choose is read by the browser's own decoder: a file goes through the same chain as a take sung into the microphone, and an unreadable file names itself instead of blaming the device.

**Sing** asks for the microphone once, and when it does not open the page names the reason: a permission this site was refused, no usable device, a device another application holds, or a page opened as a local file or over plain http, where the browser never allows the question. The browser console carries the step that failed and the error's own name, for a bug report. Without a microphone, **Demo take** loads a sung phrase read by the same chain. The page plays the notes through the desktop's own piano, follows them with a playhead, and prints as a black-on-white sheet. When the browser cannot start an output graph at all — no sound card, an audio service that is off — **Listen** says the playback could not start instead of leaving its preparing sentence on screen: the page bounds that start the way it already bounds the microphone's (`web/test/playback-failure.test.mjs`). A playback that fails also writes its console line — the step that was running and the audio graph's own state — so a bug report can say why without a screenshot, and a slow output that opens after a second still plays.

## Command line

```bash
v2m analyze chant.wav --quantize
v2m analyze chant.wav --pdf
v2m live
v2m serve
v2m --lang en analyze chant.wav
```

`analyze` accepts `--min-conf`, `--min-note`, `--transpose`, `--quantize`, `--pdf` and `-o`/`--output`. It writes MIDI, MusicXML, JSON and CSV; `--pdf` also writes a PDF when MuseScore or LilyPond is installed — a missing tool skips only the PDF, while a conversion failure exits with code 2. `--lang fr` and `--lang en` select the interface language.

`serve` is the optional loopback companion the page can call, because a browser cannot run pYIN: `v2m serve` binds `127.0.0.1` only and answers `GET /health`, `POST /analyze` (the `.json` export's payload, no file written), `POST /render` (the MusicXML the desktop would engrave, as text) and `POST /compare` (an exported `notes.json` with its audio, answered with how the two readings pair). `--port` picks the port (8765); `--allow-origin` (repeatable) names a page origin allowed to call it — localhost pages and non-browser clients always are, a published page must be named.

## Diagnostics

Optional tools, run from the project root in the activated environment. Each explains itself when called with no arguments; the flags below are the ones that change what it measures.

- `scripts/score_accuracy.py`: read a transcription. `--truth` compares known notes, `--json` writes a report, `--output` selects the artifact folder, `--no-attacks` omits attack evidence, `--compare` checks repeatability, `--golden` checks a saved reference, `--diagnose` lists the thresholds the take exercised.
- `scripts/profile_analysis.py`: time the analysis. `--repeat` sets the runs, `--profile` adds a function profile, `--top` limits that table, `--json` and `--output` write the measurements, `--no-warmup` keeps the first-run cost.
- `scripts/confirm_s_sweep.py`: sweep the live confirmation threshold — `--values`, `--seeds`, `--amps`.
- `scripts/liveseg_sweep.py`: sweep live segmentation — `--holds`, `--reartics`.
- `scripts/guide_capture.py`: record and annotate a checked take; `--audit` checks an existing one, `--annotate` writes a machine draft that an ear check must confirm.
- `scripts/reinterpret.py`: re-read saved evidence without decoding audio or running pYIN — `--save`, `--observation`, `--semitones`, `--min-conf`, `--min-note`, `--sweep`, `--sweep-note`, `--against`, `-o`/`--output`.
- `scripts/coherence_summary.py`: summarise take audits by session; `--lang` selects French or English labels and numbers (`1.25` / `1,25`).
- `scripts/replay_takes.py`: replay takes at real time and save the live/final comparison — `--out`, `--take`, `--root`, and `--changed` for takes measured by an older engine.
- `scripts/engine_drift_watch.py`: refresh outdated comparisons — `--root`, `--report`, `--json`; exits 1 when a reading changed.
- `scripts/compare_takes.py`: compare two takes of one phrase; `--json` writes the comparison.
- `scripts/make_demo.py` generates the gitignored `demo.wav` the GUI smoke test needs.
- `scripts/import_bench.py` reproduces the WSL import measurement: it copies the current venv's packages to the Windows drive and reports the median warm import time from ext4 and /mnt/c.
- `scripts/lots.py`: rebuild one lot's index from `scripts/lots.toml` and prove its tests on a frozen export of HEAD — one commit's worth of files, staged and verified away from the working tree, so "which files belong together" and "that set is green on its own" stop being hand-run commands. It exits 1 when a lot is red; the options it takes are listed in its own docstring.

## Checks

```bash
python -m pytest tests/ -q
pytest tests/ -m "not slow"
QT_QPA_PLATFORM=offscreen python -u scripts/gui_smoke_test.py
```

CI runs all three in `.github/workflows/tests.yml`, on every push and pull request, with the suite's budget raised there for the shared runner. The full suite has a 150-second budget and runs on two pytest-xdist workers — keep 2, since at 4 the real-take golden fingerprints race the evidence cache. The quick loop skips the `slow` tests.

### Running the suite under WSL

The Linux half of the gate runs in a WSL distribution, and its virtualenv lives in the Linux filesystem (`~/.venvs/v2m`), never under /mnt/c: the cost is measured, not assumed — importing the suite's stack (numpy, scipy, librosa, numba, soundfile, PyQt6) from the same `site-packages` (20,158 files, 840 MB) took 2.86 s from /mnt/c against 0.21 s on ext4 (median of 7 warm-cache runs, ×13.5). The checkout itself may stay on the Windows drive — WSL reads the sources across the bridge fine; it is the interpreter and its packages that must be local. Build it from the distribution's own Python with the pinned constraints (the distribution's `python3-venv` is what makes `python3 -m venv` work), add the audio backend (`libportaudio2`, `libasound2t64`, `libjack-jackd2-0`; with no sound server PortAudio opens 0 devices and the suite still passes) and the Qt runtime libraries the GUI smoke test needs (`libegl1`, `libgl1`, `libglib2.0-0t64`, `libxkbcommon0`, `libdbus-1-3`, `libfontconfig1`) — `libglib2.0-0t64` is the one a bare distribution does not get from the others: without it every `PyQt6` import dies on `libglib-2.0.so.0: cannot open shared object file`, measured on a clean `ubuntu-base-24.04.5` (the `t64` suffix is 24.04's name; 22.04 spells it `libglib2.0-0`), while `libsndfile1` is NOT needed, because the `soundfile` wheel carries its own (measured 1.2.2) — then run the suite serially. `scripts/import_bench.py` reproduces the import figure; its first run copies the packages to the Windows public folder, which is the slow part:

```bash
wsl -d Ubuntu -- bash -lc 'sudo apt-get update && sudo apt-get install -y python3-venv libportaudio2 libasound2t64 libjack-jackd2-0 libegl1 libgl1 libglib2.0-0t64 libxkbcommon0 libdbus-1-3 libfontconfig1'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && python3 -m venv ~/.venvs/v2m && ~/.venvs/v2m/bin/pip install -e ".[dev]"'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && QT_QPA_PLATFORM=offscreen XDG_RUNTIME_DIR=/tmp/v2m-run LIBGL_ALWAYS_SOFTWARE=1 V2M_SUITE_BUDGET_S=900 PYTHONFAULTHANDLER=1 ~/.venvs/v2m/bin/python -m pytest tests/ -q -n 0'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && ~/.venvs/v2m/bin/python scripts/make_demo.py'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && install -d -m 700 /tmp/v2m-run && QT_QPA_PLATFORM=offscreen XDG_RUNTIME_DIR=/tmp/v2m-run LIBGL_ALWAYS_SOFTWARE=1 ~/.venvs/v2m/bin/python -u scripts/gui_smoke_test.py'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && ~/.venvs/v2m/bin/python scripts/import_bench.py'
```

The suite and the offscreen GUI smoke test are the two halves of the desktop gate; CI additionally runs the page's Node tests. The smoke test reads the gitignored `demo.wav`, so regenerate it first as CI does, and run it with the same `QT_QPA_PLATFORM`, `XDG_RUNTIME_DIR` and `LIBGL_ALWAYS_SOFTWARE` variables.

`-n 0` is the serial form CI also runs, so a native crash names its test instead of taking a worker's output with it; `QT_QPA_PLATFORM=offscreen` because the distribution has no display; `XDG_RUNTIME_DIR` and `LIBGL_ALWAYS_SOFTWARE` because Qt and PortAudio want a runtime directory and no GPU; `V2M_SUITE_BUDGET_S` because a WSL distribution is slower than the machine the 150-second budget was calibrated on (measured 322 s); `PYTHONFAULTHANDLER` so a native crash prints every thread's stack. Quote the checkout path (it usually holds a space), and keep multi-step recipes in a script: `wsl.exe` consumes `$VAR` and `$?` before bash sees them.

For repository work, use GitHub Issues; the working notes behind the code (`AGENTS.md`, `CONTEXT.md`, `SKILLS.md`, `constraints.txt`, the notes under `docs/project/`, the hardware recipe `docs/recette-materielle-navigateurs.md` and `prises_verite/README.md`) stay machine-local — ignored by `.gitignore`, so git never proposes them. The decisions under `docs/adr/`, dated reports (`*_YYYY-MM-DD.md` at the repository root), captures (`page-prise-*.png`), `docs/engine-drift.md` and `docs/superpowers/` also stay local and are not committed.

No third-party content is vendored into the repository: the interface instructions consulted while building, their lockfile and their license stay machine-local (`vendor/`, ignored by `.gitignore`), and nothing here reads them at runtime.

## License

MIT — see `LICENSE`.
