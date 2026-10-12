**English** | [Français](README.fr.md)

# Voice → MIDI

Turn a sung melody into notes. Analyze an audio file or sing into the microphone; review and edit the transcription, then export MIDI, MusicXML, JSON and CSV. PDF export is optional. In the desktop app, **Export** opens a Save As dialog where you choose the filename and format; it remembers the last destination folder across launches. Live takes also retain their full audit archive in `prises_live/`.

The desktop app also includes a note finder: sing a note to identify it, or click a piano key to hear it. Confirmed notes are kept in its short history, including repeated notes; a held note appears once. The history uses the readings captured with each audio block, so a busy interface does not lose a note that was already confirmed. It tells you whether the sung note is in tune or off, with a cents meter and a named verdict. Its piano answers the keyboard too: Tab to it, choose a key with the arrows, press **Enter** to hear it.

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

The home screen is a menu: the three actions (open a file, sing into the mic, find the note you are singing) sit in one centered column, with the language toggle, the coherence follow-up and a **Calibrate my mic** toggle below them. The calibration walks five held notes, then offers the measured device offset (cents) only when the evidence backs it — scattered runs offer nothing, and each round counts only when the singer reached its asked note (a held pitch graded against the NEXT round's target is exactly the noise the gates refuse). The offset is saved to `~/.v2m_mic_cal` (best-effort) and shows as context on later takes; **recognition never reads it** (a calibration is a display fact, never an engine input).

- In **Find the note I am singing**, a confirmed note is added to the short history even if the interface is busy; a sustained note is listed once, and singing it again after a break adds another entry.
- Open or drop a WAV, MP3, FLAC or OGG file, or choose live recording. During a take, the live page separates **Take settings**, **Your live take** (mic level, confirmed-note count, roll and staff) and **Singing feedback** (recent tuning and its trend), so the stop action and what the microphone is hearing stay easy to find.
- After analysis, a compact **Your transcription** summary keeps note count, duration, tempo and key visible above the main **Export this take** action. Open **Show details** when you need the full diagnostics and options; it starts collapsed so the score and next action are not buried. The entire results-controls column scrolls independently, keeping the diagnostics, note edits, export options and analysis actions reachable at short window heights. **Re-read evidence** quickly applies the minimum-note threshold to cached pitch evidence; **Full reanalysis (pYIN)** rebuilds that evidence from the retained audio in the background, shows stage progress and preserves the current reading if it fails. A successful rerun replaces the reading only when complete; export remains explicit. After a successful re-read or full reanalysis, a **reading diff** appears in place: the previous reading is drawn as translucent ghost bars with the changed notes accented, a compact Before → After count line joins the results panel (identical / changed / near-rounding / missing / added, plus the coherence suspects of the current reading) and an expanded detail lists each change with its causes (pitch, attack, duration). The suspects carry their diff label (NEW / PERSISTS / RESOLVED); the pYIN confidence is shown as context only — never a verdict. Any later boundary (a manual edit, undo/redo, a splice decision, a new take or a new file) clears the diff; exporting stays explicit and writes only the current reading. Review the notes, listen to the piano or voice, and edit a note by selecting it and clicking the correct piano key. Keyboard correction is also available: focus the piano keyboard, use **Left/Right** to choose a pitch (each choice is played), then press **Enter** to apply it (**Escape** cancels). Selection also happens directly on the graph: click a note's bar to select it, drag across neighbouring bars to select a range, click empty space to clear. On the results roll a bar is also **draggable**: slide it to shift its onset and pitch, or grab its right edge to resize it — the move lands as one undo step. Once a bar is picked, the **Left/Right arrows** walk the selection note by note (Shift+arrow extends the range), so the cursor can be held on a note before deleting it. The **notes table** is keyboard-driven too: **Up/Down** step one note, **Home/End** jump to the first or last, **Shift+arrow** extends the range there as well, and **Space** auditions that note's own capture. While it plays, the note name and position are shown with a **Stop** action. The toolbar shows the current selection (note count and time span), the status bar confirms each edit (*"N notes deleted"*, the corrected note's name), and **Delete** removes the selected range — one undo step. Undo and redo (Ctrl+Z / Ctrl+Y, or the ↶/↷ buttons) revert note edits; the dropdown above the edit buttons lists every state of the take — pick one to jump straight back to it.
- The results panel carries a **per-take intonation report**: the share of notes sung in tune, a named tendency (a systematic pull sharp or flat, announced only when the evidence backs it) and the same verdict for each register (low / mid / high). Breaths are part of the score too: silences where an air intake was actually heard are marked **B** on the staff and `Respiration` in the exported MusicXML.
- While you sing, the live page shows the **same intonation feedback**: after each note closes it reads the share of notes sung in tune so far and — once enough notes back it — the take-so-far tendency, in the per-take report's own words and scale, register leans included. Releasing a dragged bar also plays the note **before, then after** the move, so the change is judged by ear like a pitch correction.
- For a live take, select a continuous range of notes and choose **Replace selected notes**; **Listen to the original** plays the selected phrase as it is now, before you re-sing it. Sing the replacement, finish it, listen to the new phrase, then choose **Keep replacement** (or **Cancel replacement** to restore the take). Nothing is exported until you choose **Export this take**.
- After a live take stops, the notes it closed stay on the live piano roll: click a bar to highlight it and explore the take as it was sung (read-only — the next take starts clean). **Space** auditions the highlighted captured note from the take's own audio, with a **Stop** action beside it. From the results page, **Captured notes** opens that same frozen roll — the take as it was captured, with the final reading drawn over it (the notes the offline pass moved carry the accent), so a difference reads at a glance. The **Show final reading** checkbox toggles only that overlay; and **Back to results** returns to the analysis.
- The desktop **Export** action opens Save As: choose the filename and one format (MIDI, MusicXML, PDF, JSON or notes CSV). The dialog reopens in the last folder used, remembered across app launches; the first time it opens beside the source file (or in the current folder for a live take). PDF is listed even when no engraving tool is installed; choosing it then shows a warning and creates no PDF. The selected extension is applied to the filename. Live takes keep their complete audit archive in `prises_live/` and also save the selected file to your chosen location; that archive's live/final audit (the report's `live_vs_final` block and its sidecar) carries the capture's real-time honesty counters: dropped blocks and the worst drain progress sampled during the take. The CLI still writes its full artifact bundle; the desktop and `v2m live` CLI share the same export receipt and live archive contract.

### Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| Tab | Walk the page's actions in task order (live and note-finder pages) |
| Ctrl+Z | Undo the last note edit (results page) |
| Ctrl+Y | Redo the last undone note edit (results page) |
| Up / Down | Move the selection one note (results page) |
| Home / End | Jump to the first / last note (results page) |
| Space | Audition the selected note; press Space again to stop (results page, and the frozen live roll) |
| Left / Right | Choose a pitch on the focused piano (it plays); Enter applies it, Escape cancels |
| Enter | Note finder: hear the key staged on the piano |
| Delete | Delete the selected note range (results page) |

Ctrl+Y is the redo key; Ctrl+Shift+Z works too. Both shortcuts act from the results page, where the edits live.

## Command line

```bash
v2m analyze chant.wav --quantize
v2m analyze chant.wav --pdf
v2m live
v2m --lang en analyze chant.wav
```

`analyze` accepts `--min-conf`, `--min-note`, `--transpose`, `--quantize`, `--pdf` and `-o`/`--output`. It writes MIDI, MusicXML, JSON and CSV; `--pdf` also writes a PDF when MuseScore or LilyPond is installed. A missing PDF tool skips only the PDF; a conversion failure exits with code 2. Held notes in MIDI share one channel-wide sustain pedal: overlapping or adjacent sustain windows are merged so an earlier note cannot lift the damper under a later one. `--lang fr` and `--lang en` select the interface language.

## Diagnostics

These tools are optional. Run them from the project root, inside the activated environment.

- `scripts/score_accuracy.py`: inspect a transcription. `--truth` compares known notes; `--json` writes a report; `--output` selects its artifact folder; `--no-attacks` omits attack evidence; `--compare` checks repeatability; `--golden` checks a saved reference; `--diagnose` shows which thresholds the take exercised.
- `scripts/profile_analysis.py`: measure analysis time. `--repeat` sets the number of runs; `--profile` adds a function profile; `--top` limits that table; `--json` writes measurements; `--output` selects the artifact folder; `--no-warmup` includes the first-run cost. The table ends with a `Rescue` line: the short-window sweep's own cost and how much audio it grafted into the pitch curve (0.0 when the gates refused everything).
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

The full suite has a 150-second budget. The quick loop skips tests marked `slow`. The GUI smoke test uses the gitignored `demo.wav`; generate it with `python scripts/make_demo.py`. For repository work, engineering agents use GitHub Issues and read the domain guidance in `CONTEXT.md` and relevant decisions under `docs/adr/`; see `docs/agents/`.
