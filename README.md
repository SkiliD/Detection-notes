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

To publish it, set **Settings → Pages → Source** to **GitHub Actions**, then push to `main`: the workflow replays the port's conformance cases, photographs the page at 1600 px, measures eight resolution classes (320 px to 2560 px, a landscape phone included), audits both palettes in three visible states at each of them with axe-core, then deploys `web/`.

**Sing** asks for the microphone once, and when it does not open the page names the reason: a permission this site was refused, no usable device, a device another application holds, or a page opened as a local file or over plain http, where the browser never allows the question. The browser console carries the step that failed and the error's own name, for a bug report. Without a microphone, **Demo take** loads a sung phrase read by the same chain. The page plays the notes through the desktop's own piano, follows them with a playhead, and prints as a black-on-white sheet.

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
- `scripts/coherence_summary.py`: summarise take audits by session; `--lang` selects French or English labels.
- `scripts/replay_takes.py`: replay takes at real time and save the live/final comparison — `--out`, `--take`, `--root`, and `--changed` for takes measured by an older engine.
- `scripts/engine_drift_watch.py`: refresh outdated comparisons — `--root`, `--report`, `--json`; exits 1 when a reading changed.
- `scripts/compare_takes.py`: compare two takes of one phrase; `--json` writes the comparison.
- `scripts/make_demo.py` generates the gitignored `demo.wav` the GUI smoke test needs.

## Checks

```bash
python -m pytest tests/ -q
pytest tests/ -m "not slow"
QT_QPA_PLATFORM=offscreen python -u scripts/gui_smoke_test.py
```

CI runs all three in `.github/workflows/tests.yml`, on every push and pull request, with the suite's budget raised there for the shared runner. The full suite has a 150-second budget and runs on two pytest-xdist workers — keep 2, since at 4 the real-take golden fingerprints race the evidence cache. The quick loop skips the `slow` tests. For repository work, use GitHub Issues; the design notes behind the code (`AGENTS.md`, `CONTEXT.md`, `docs/adr/`) live in the working tree but are not committed, so a clone does not carry them.

## License

MIT — see `LICENSE`.
