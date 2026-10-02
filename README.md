**English** | [Français](README.fr.md)

# Voice → MIDI

Turn a sung melody into notes. Analyze an audio file or sing into the microphone; review and edit the transcription, then export MIDI, MusicXML, JSON and CSV. PDF export is optional.

The desktop app also includes a note finder: sing a note to identify it, or click a piano key to hear it.

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

The rest of this file assumes the environment is activated: `python` and
`pytest` below are that interpreter (on Windows without activation, use
`.venv/Scripts/python.exe` instead).

## Use the app

A new live transcription stays in the results page until you export it. Re-singing a selected range updates the take: you audition the phrase, then keep it or cancel it (the original take is also kept if recording or re-analysis fails). Note edits (corrections and kept replacements) can be undone and redone; a manual pitch correction keeps the note timing and marks the audio-based coherence check as unavailable for that edited result.

```bash
v2m gui
```

The home screen is a menu: the three actions (open a file, sing into the mic, find the note you are singing) sit in one centered column, with the language toggle and the coherence follow-up below them.

- Open or drop a WAV, MP3, FLAC or OGG file, or choose live recording.
- Review the notes, listen to the piano or voice, and edit a note by selecting it and clicking the correct piano key. Selection also happens directly on the graph: click a note's bar to select it, drag across neighbouring bars to select a range, click empty space to clear. The toolbar shows the current selection (note count and time span); press **Delete** to remove the selected range — one undo step. Undo and redo (Ctrl+Z / Ctrl+Y, or the ↶/↷ buttons) revert note edits; the dropdown above the edit buttons lists every state of the take — pick one to jump straight back to it.
- For a live take, select a continuous range of notes and choose **Replace selected notes**; **Listen to the original** plays the selected phrase as it is now, before you re-sing it. Sing the replacement, finish it, listen to the new phrase, then choose **Keep replacement** (or **Cancel replacement** to restore the take). Nothing is exported until you choose **Export this take**.
- After a live take stops, the notes it closed stay on the live piano roll: click a bar to highlight it and explore the take as it was sung (read-only — the next take starts clean).
- Exports are written beside an input file. Live takes are saved in `prises_live/`.

### Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| Ctrl+Z | Undo the last note edit (results page) |
| Ctrl+Y | Redo the last undone note edit (results page) |
| Delete | Delete the selected note range (results page) |

Ctrl+Y is the redo key; Ctrl+Shift+Z works too. Both shortcuts act from the results page, where the edits live.

## Command line

```bash
v2m analyze chant.wav --quantize
v2m analyze chant.wav --pdf
v2m live
v2m --lang en analyze chant.wav
```

`analyze` accepts `--min-conf`, `--min-note`, `--transpose`, `--quantize`, `--pdf` and `-o`/`--output`. It writes MIDI, MusicXML, JSON and CSV; `--pdf` also writes a PDF when MuseScore or LilyPond is installed. A missing PDF tool skips only the PDF; a conversion failure exits with code 2. `--lang fr` and `--lang en` select the interface language.

## Diagnostics

These tools are optional. Run them from the project root, inside the activated environment.

- `scripts/score_accuracy.py`: inspect a transcription. `--truth` compares known notes; `--json` writes a report; `--output` selects its artifact folder; `--no-attacks` omits attack evidence; `--compare` checks repeatability; `--golden` checks a saved reference; `--diagnose` shows which thresholds the take exercised.
- `scripts/profile_analysis.py`: measure analysis time. `--repeat` sets the number of runs; `--profile` adds a function profile; `--top` limits that table; `--json` writes measurements; `--output` selects the artifact folder; `--no-warmup` includes the first-run cost.
- `scripts/confirm_s_sweep.py`: sweep the live note-confirmation threshold. `--values`, `--seeds` and `--amps` restrict the values, random seeds and test amplitudes.
- `scripts/liveseg_sweep.py`: sweep live segmentation. `--holds` and `--reartics` restrict the hold and re-articulation thresholds.
- `scripts/guide_capture.py`: record and annotate a checked take; `--audit` checks an existing take instead.
- `scripts/reinterpret.py`: reuse saved pitch evidence without decoding audio or running pYIN. `--save` saves evidence; `--observation` loads it; `--semitones`, `--min-conf` and `--min-note` change interpretation; `--sweep` and `--sweep-note` compare thresholds; `--against` compares evidence; `-o`/`--output` exports results.
- `scripts/coherence_summary.py`: summarize take audits by session; pass a takes folder to use a location other than `prises_live/`. `--lang` selects French or English labels.
- `scripts/replay_takes.py`: replay recorded takes at real time and save the live/final comparison. `--out` selects a JSON report; `--take` selects one take; `--changed` replays only takes measured by an older engine; `--root` selects the takes folder.
- `scripts/engine_drift_watch.py`: refresh outdated take comparisons. `--root` selects the takes folder; `--report` selects the Markdown report. Exits 1 if a reading changed. `--json` saves machine-readable results.
- `scripts/compare_takes.py`: compare two takes of one phrase. `--json` writes the comparison.

## Checks

```bash
python -m pytest tests/ -q
pytest tests/ -m "not slow"
QT_QPA_PLATFORM=offscreen python -u scripts/gui_smoke_test.py
```

The full suite has a 150-second budget. The quick loop skips tests marked `slow`. The GUI smoke test uses the gitignored `demo.wav`; generate it with `python scripts/make_demo.py`.
