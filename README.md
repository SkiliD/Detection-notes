**English** | [Français](README.fr.md)

# 🎤 Voice → MIDI

Desktop application that **detects sung notes** (pitch, note values, meter), the **tempo**, the **key** and the **chords**, then exports a **MIDI file**, a **MusicXML score** (optional PDF) and a **JSON/CSV report**. Works on **audio files** (WAV/MP3/FLAC/OGG) or **live from the microphone**.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)

## Features

- **Pitch detection**: pYIN (librosa) offline, incremental YIN in real time — search range C2–C7, so soprano top notes read at their own octave (the old C6 ceiling folded them one octave down)
- **f0 stabilization**: the pitch derivative marks every moving zone (portamento ramp, note hand-over) as a **transition**, excluded from the pitch estimate — each note reads its **stable plateau**, with the vibrato riding on top
- **Tuning correction**: a singer sitting consistently flat or sharp is re-centered — one global offset (median deviation to the nearest semitone, capped at ±0.35 st) is absorbed **before** rounding, so notes stop rounding on the wrong side; the residual per-note deviation feeds the key-aware snap. The results panel shows the offset in cents with an intonation advice ("in tune" / slightly / clearly off). The JSON carries each note's residual as `deviation_cents` (signed) plus an `intonation` summary that names a habit **only when the evidence supports it** — the take-wide lean, the lean by pitch **register**, and the first half against the second half — so noise is never described as a tendency
- **Rumble gate**: a low-frequency rumble (mic handling noise before the take) reads no pitch — periodicity and high-frequency checks kill the frames both offline and live, so the take no longer opens on a ghost note
- **Key-aware snapping**: a note sung almost halfway between two semitones resolves toward the neighbour it leans to when that neighbour is a degree of the detected key — a mistuned scale degree is re-anchored, a deliberate chromatic shade stays a shade
- **Note segmentation**: hysteresis with a sliding reference + 3-frame confirmation, **driven by attacks** (consonants, glottal stops, vowel changes): a legato glissando is not a note, and the micro-notes it used to create (6 sung notes → 10 detected) are absorbed into the arrival note, at its pitch. Survives vibrato (a held note with vibrato used to trigger 19 "attacks"), **deep vibrato** (±0.6 semitone), fast **trills**, and reads **descending glides** like ascending ones (no duplicate fragment, the arrival starts at its consonant); a note arriving by portamento is read at its plateau pitch, never the ramp's; an explicitly repeated same pitch is read as a repetition even when the singer drifts between repeats
- **Tempo**: librosa beat tracking with ×2 / ÷2 fallback (range 60–180 BPM), then **refinement on note attacks** (least-squares grid → exact BPM)
- **Key**: Krumhansl-Schmuckler profiles (24 major/minor candidates) + 2 alternative candidates
- **Chords**: global recognition, by adjacent 2 s segments and **per measure** (major, minor, dim, aug, 7, maj7, m7) via a weighted histogram — shown in the GUI and the JSON
- **Chord progression**: one chord per **measure** (aligned on the detected meter, identical measures merged) — reads like a chart: `Dm7 | Cmaj7 | G7 | C`. In the French interface the chord labels are translated for display (`Am` is shown **La mineur**) — the JSON, the CSV and the MusicXML keep the canonical `Am`
- **Bilingual note names**: English (`A4`) **and** French (`La4`) side by side in the GUI, the JSON (`name` / `name_fr`) and the CSV
- **fr / en interface**: language chosen in the GUI (🌐 button) or `v2m --lang en` — translates every label **and** the note-name format (`Do Ré Mi` ↔ `A B C`), choice persisted in `~/.v2m_lang`
- **Note values**: each note is labeled (whole → sixteenth, dotted values, **double-dotted** values, triplets) according to the detected tempo, tolerant of human interpretation — `figure` field in the JSON and the CSV, **also in real time** on the mic-mode piano roll. A double dot only engraves on a deliberate length (±1 % of the ideal value): a singer's timing around a plain note never lands one by accident
- **Staff (notation)**: notes and rests written on a **real staff** — treble clef, key signature, time signature, note values (whole → sixteenth, dotted and double-dotted values, triplets), quarter/half/whole rests with stems, flags, dots and **accidentals (sharp, flat AND natural)**, ledger lines, barlines and a tempo · meter · key header. A key-signature note sung natural is spelled natural (F♮ in G major, with its own natural glyph — never a disguised E♯). Shown **after analyzing a file** and **live during capture** (scrolling staff, current note highlighted). Built-in vector engraving: no external software required
- **Meter**: automatic detection — a take on **dotted pulses** is published as **6/8** (tempo corrected ×1.5); otherwise **attack accents** (the notes' relative RMS strengths, contrast required) decide 2/4, 3/4 vs 4/4 — a comparable strong/weak alternation reads as 2/4 (a march), a beat-1-strong / beat-3-secondary hierarchy stays 4/4 — and the ternary grid rescues an accent-less 6/8; with no clear accent, the conservative default is 4/4. The score and the report use the detected meter
- **MusicXML export**: a readable score (detected meter, key signature, tempo, note values, ties across barlines, triplets) that opens in MuseScore, Finale, Sibelius…
- **Every detection on the score**: per-measure chords as **harmony symbols** above the staff **with their readable name** ("La mineur" in the French UI — the file keeps the canonical symbol too), **solfège note names** (Do, Ré, Mi…) under each note, **noteheads colored by confidence** (green = sure → red = doubtful — the color now carries pYIN's real voicing probability, so a breathy note reads dimmer), recap header (tempo · key · meter · chords) in the title
- **MIDI export**: `set_tempo` + `key_signature` meta-events, velocity proportional to confidence, optional 1/16 quantization, ±48 semitone transposition, **sustain pedal events (CC64)** on held notes — files played elsewhere ring like the built-in piano
- **PDF export**: score engraving via MuseScore or LilyPond (auto-detected, install hint otherwise) — `--pdf` on the CLI, a checkbox in the GUI
- **JSON report** (everything, including the pitch curve, the meter, the measure count, each note's measure:beat position, the global chord and the chord segments) + **CSV** (note list with value and position)
- **PyQt6 GUI**: polished dark interface, piano roll, waveform, results panel with per-note (position, value), per-chord-segment and **tuning-offset** rows, drag & drop, **notation mode** (color by note value), **built-in red error banner** (no blocking popups)
- **Real-time mode**: note/Hz/level indicator, **piano roll that follows the capture** (10 s sliding window, time cursor, `mm:ss` clock, note counter and **detected meter** — 3/4, 4/4, 6/8 — shown live), comparison with an already-analyzed score (green/red keys), the held note shows **while you sing it** (~125 ms display latency), note values labeled live, export on stop — pitch tracking runs **outside the audio callback** (dedicated thread), so the driver never drops blocks. A **"Save the take's audio"** checkbox (off by default) also writes the take as a 16-bit WAV (`.wav`) into its `prises_live/<stem>/` folder, ahead of the other artifacts — live only, since a file analysis already has its audio
- **Note finder**: isolated mic recognition, 88-key keyboard and a local history of the last six notes; a key lights **only on a confirmed note** (a mic rumble never does), and there is no analysis nor export
- **Control playback**: ▶ button (built-in local synth, no external software) with a **playhead synchronized** on the piano roll and the waveform; **double-click** the piano roll to start playback from there. The piano follows the take's own dynamics (a strong accent hits harder than a soft note), strikes soft keys slower, breathes a few milliseconds around the grid and sits in a light room — deterministic, so a replay sounds identical

## Installation

Python ≥ 3.10 (3.12 recommended):

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate              # Windows
pip install -e ".[dev]"
```

## Usage

### GUI (recommended)

```bash
v2m gui            # or: python -m v2m gui
```

- **Analyze a file**: button or drag & drop a WAV/MP3/FLAC into the window
- **See the score**: below the piano roll, the staff shows the notes actually detected (with their values) — it adapts to the window size and splits the music into systems with measure numbers
- **Sing into the mic**: a self-scrolling piano roll — the time cursor advances with the capture, the current note's bar grows while you hold it, and the value label appears as soon as the note ends; **the staff scrolls below** and writes down the melody as you go; on stop, final analysis (tempo, key, meter, chords) and export
- Exports are written next to the source file (`.mid`, `.musicxml`, `.json`, `.csv`)
- **Notation mode**: checkbox above the piano roll — each note is colored by its value (whole, half, quarter, eighth…), a short label shows on wide-enough notes
- **Control playback**: ▶ button + synchronized cursor; a **piano keyboard** under the piano roll lights up the keys of the notes sounding during playback

### Command line

```bash
# full analysis with 1/16 quantization on the detected tempo
v2m analyze chant.wav --quantize

# interface and note names in English (fr default; persisted choice)
v2m --lang en analyze chant.wav --quantize

# options
v2m analyze chant.wav --min-conf 0.7 --min-note 80 --transpose -12 -o sortie

# real time (Ctrl+C to stop and export)
v2m live
```

The `analyze` command writes `sortie.mid`, `sortie.musicxml`, `sortie.json` and `sortie.csv` (`--pdf` adds `sortie.pdf` if MuseScore or LilyPond is installed — otherwise the PDF is skipped with an install hint and the exit code stays 0; if engraving fails, exit 2 and the other four artifacts are still written); the JSON carries, for each note, its `figure` field (e.g. `"noire"`, `"croche pointée"` — figure names stay French canonical data), its `position` `[measure, beat]`, its `deviation_cents` (signed residual intonation, once the global offset is absorbed), its `name` / `name_fr` names (English and French), the detected `meter`, `num_measures`, the global `chord` and its `chord_segments`, and the `intonation` summary built from those residuals (mean/median, on-note/sharp/flat shares, by register, by half).

```bash
# with PDF engraving of the score (best-effort: clear message when no tool)
v2m analyze chant.wav --quantize --pdf
```

The MusicXML score embeds **every detection**: chord symbols above the staff (per-measure progression) with their readable name ("La mineur"), solfège names (Do, Ré, Mi…) under each note, noteheads colored by confidence (green = sure → red = doubtful), recap header in the title. Everything opens in MuseScore, Finale or Sibelius.

### Library

```python
from v2m.core.pipeline import analyze_file
from v2m.core.report import export_result

res = analyze_file("chant.wav")
print(res.tempo.bpm, res.key.best.label, len(res.notes))
print(res.notes[0].figure)          # e.g. "noire" (French canonical value)
if res.chord:
    print(res.chord.best.label)     # e.g. "Cmaj7" (None when undetermined)

rep = export_result(res, "out")     # writes out.mid/.musicxml/.json/.csv

# more finely, to build an artifact yourself:
from v2m.core.midi_export import build_midi
mid = build_midi(res.notes, bpm=res.tempo.bpm, key=res.key)
mid.save("out.mid")
```

## Architecture

```
src/v2m/
├── cli.py               # analyze / live / gui subcommands
├── live.py              # CLI real-time session
├── core/
│   ├── audio_io.py      # file loading / mic capture (sounddevice)
│   ├── pitch.py         # pYIN (offline) + block YIN (real time)
│   ├── notes.py         # cleaning (median + anti-vibrato) and segmentation (attacks, transitions)
│   ├── onsets.py        # attacks: offline spectral flux, live level rise
│   ├── tempo.py         # BPM + beats (librosa.beat, ×2/÷2 fallback)
│   ├── keys.py          # Krumhansl-Schmuckler, 24 candidates
│   ├── chords.py        # chords (triads + sevenths, global, segments and per-measure progression)
│   ├── figures.py       # note-value classification; catalog in grid.py
│   ├── meter.py         # meter detection (dotted 6/8, accents 2/4, 3/4 vs 4/4)
│   ├── grid.py          # BeatGrid, Meter, Figure/CATALOGUE and MusicXML divisions
│   ├── playback.py      # note audio rendering + control-playback reader
│   ├── liveseg.py       # real time: blocks → pitch → attacks → notes chain (LivePitchStream)
│   ├── notation.py      # staff: pitch spelling, key signatures, written values, layout
│   ├── midi_export.py   # MIDI writing (tempo, key signature, notes)
│   ├── musicxml.py      # MusicXML score (detected meter, key signature, values, ties,
│   │                    #   <harmony> chords, solfège lyrics, confidence color)
│   ├── pdf_export.py    # PDF via MuseScore/LilyPond (auto-detection)
│   ├── report.py        # export_result: the artifacts + the unified JSON report
│   ├── live_session.py  # LiveTake: one mic take end to end (capture → analysis)
│   └── pipeline.py      # analyze_signal / analyze_file orchestration
└── gui/
    ├── main_window.py   # state, navigation, callbacks, threads and exports
    ├── pages.py         # page assembly and action wiring
    ├── staff.py         # QPainter-drawn staff (clef, key signature, notes, rests, barlines)
    └── widgets.py       # piano roll, waveform, panel, live indicator

scripts/
├── gui_smoke_test.py    # offscreen GUI walkthrough (window, panel, language, keyboard)
├── profile_analysis.py  # diagnostics: analysis time per stage + cProfile
├── score_accuracy.py    # diagnostics: what the transcription did to a real take
├── confirm_s_sweep.py   # diagnostics: live confirm_s threshold sweep, multi-seed
├── liveseg_sweep.py     # diagnostics: live HOLD_ST x REARTIC_ST sweep, multi-seed
├── guide_capture.py     # guided note-by-note recording of a hand-checked take
├── coherence_summary.py # diagnostics: aggregate every take's coherence.csv per session
├── replay_takes.py      # diagnostics: replay the takes' WAVs through the GUI chain
│                        #   (fake real-time mic) and archive the live/final gap
├── engine_drift_watch.py # diagnostics: re-derive every stale engine audit and write
│                        #   a Markdown non-drift report (exit 1 on a moved reading)
└── reinterpret.py       # diagnostics: reinterpret a saved observation, no pYIN

prises_verite/          # hand-checked takes: <name>.wav + <name>.truth.json (see its README)
tests/
└── voicetruth.py        # ground truth: synthetic voice with known notes + accuracy measure
```

## Diagnostics (analysis time)

```bash
# stage table: where does the time go on THIS take?
./.venv/Scripts/python.exe scripts/profile_analysis.py chant.wav

# 3-run average + heaviest functions (cProfile) + JSON dump
./.venv/Scripts/python.exe scripts/profile_analysis.py chant.wav --repeat 3 --profile --top 20 --json mesures.json --output artefacts
```

The stages are not duplicated in the script: it reads the ones the pipeline already announces (`progress`), so a renamed or added stage shows up here on its own. Two columns matter: **Share** (the stage's weight) and **Real-time** (stage duration / audio duration — 0.037× = 27× faster than listening, the threshold to watch on a long take). A 2 s warm-up pass is discarded by default (`--no-warmup` to measure it): without it, the librosa imports and the numba compilation land on the first stage and hide the real cost. `--repeat N` averages N runs (defaults to 1); `--top N` limits the cProfile table to its N heaviest lines; `--json file` writes the measurements to a JSON file; `--output dir` keeps the exported artifacts in that folder (else a temporary one is used). `--profile` adds a cProfile table separating **project code** (what can be optimized) from **libraries** (where called time goes).

On `demo.wav` (8.5 s) decoding dominates; on a 68 s take (repeated 8×) pYIN weighs 53% and `refine_bpm_with_onsets` 16% — the latter is what the table shows on top on the project-code side. The attack computation (one extra STFT) costs 0.12 s on those 68 s, i.e. 0.2% of real time.

## Diagnostics (transcription quality)

```bash
# what the transcription did to THIS take: notes, values, positions…
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav

# dubious-cutting signs, and the attacks-vs-geometry-only comparison
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --json rapport.json --no-attacks

# scored comparison against known notes (ground truth, same harness as the tests)
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --truth chant_verite.json


# does replaying the analysis twice give the same numbers? (drift => exit 1)
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --compare

# golden fingerprint: the take is compared to ITS OWN saved segmentation
# (<audio>.golden.json, created on first run) — drift is caught even when
# two replays agree with each other
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --golden
```

On a real take there is no ground truth: the script first prints what the pipeline found (tempo, meter, key, chords, every note with its value and position), then the **dubious segmentation signs**: orphan attacks (a consonant with no note start — a note may be missing), unattacked starts (the note exists but no attack dates it), fragments (notes shorter than the 120 ms confirmation, typical of a glissando split in two). If `--no-attacks` gives the same note count, the attack evidence is useless on that take (no usable consonants). Ground truth is a simple JSON `{"notes": [{"midi": 60, "start_s": 0.3, "dur_s": 0.6}, …]}` — file names stay language-neutral (`chant_verite.json` in both READMEs); `--truth file.json` scores the transcription against those known notes, `--json file.json` writes the audit report, and `--output dir` keeps the exported artifacts (else a temporary folder).

Two modes protect segmentation **over time**: `--compare` replays the full analysis (re-decoding included) and compares the segmentation fingerprint (tempo, meter, key, notes, rests); `--golden` compares against the fingerprint saved next to the take (`<audio>.golden.json`, created on first run, gitignored like every `.json`) — a segmentation regression on a real take is caught even when two replays agree with each other. Both print the differences field by field and exit with code 1 when the fingerprint moved.

`--diagnose` audits the **thresholds the take actually exercised** — never recalibrate a knob on a take that does not test it: the curve's pitch range against the reading bounds (65 Hz–2100 Hz), the absorbed tuning offset (cents), the normalized attack strengths (deciles, against their floor), the residual pitch deviations that decide the key-aware snap, and the live path replayed as a stream (real display latency, `confirm_s`).

The analysis is split into **two layers**: a `VoiceObservation` (the cleaned pitch curve, the attacks, the frame clock, the accent envelope, the take's tuning offset) and its **interpretation** (notes, key, chords, grid, figures). The evidence can be saved next to the take and replayed **without decoding the audio or running pYIN again** — that is `scripts/reinterpret.py`. `--save` writes the observation to `<take>.observation.npz`; `--observation file.npz` reads a saved one instead of the audio; `--semitones`, `--min-conf` and `--min-note` (in milliseconds) change the interpretation thresholds while the evidence stays the same; `--sweep` and `--sweep-note` scan several values of each on the SAME observation and print one line per reading (note count, tempo), the current calibration starred — choose a threshold from the table, not a hunch; `--against other.npz` diffs TWO observations, the evidence first (curve sample by sample, attacks, clock — where an engine actually changed) then the segmentation; `-o`/`--output dir` re-exports the artifacts and nothing is written without it.

`v2m analyze` and the GUI reuse the take's own `<take>.observation.npz` automatically: every file analysis WRITES the observation it built (so the pYIN cost is paid once per take, not once per analysis), and the next analysis reads it back when the audio has not changed (size + mtime recorded in the file). The status bar then says why it was instant (« evidence reloaded from cache — pYIN skipped »). A stale, unreadable or foreign-layout cache is silently ignored — the fallback is the full evidence build, never a different reading.

```bash
# save the evidence once, then reinterpret it with other thresholds
./.venv/Scripts/python.exe scripts/reinterpret.py chant.wav --save
./.venv/Scripts/python.exe scripts/reinterpret.py chant.wav --observation chant.observation.npz --semitones 0.35 --min-note 80
```

Two more tools complete the corpus work:

```bash
# confirm_s sweep: the 18 live truth cases x 5 synth seeds x 0.05..0.10 s —
# the archive of measured values lives in AGENTS.md
./.venv/Scripts/python.exe scripts/confirm_s_sweep.py

# live segmenter sweep: HOLD_ST x REARTIC_ST over the same cases plus two
# beginner-vibrato (+/-0.5 st) cases, several seeds per cell — decides
# whether a beginner's vibrato deserves a wider hold (measured verdict
# archived in AGENTS.md)
./.venv/Scripts/python.exe scripts/liveseg_sweep.py

# guided recording of a hand-checked take: sing, annotate note by note,
# the bounds are refined on the pitch curve, written under prises_verite/
./.venv/Scripts/python.exe scripts/guide_capture.py prise_verite.wav

# audit it: JSON validation + note-by-note human vs machine comparison
./.venv/Scripts/python.exe scripts/guide_capture.py prise_verite.wav --audit
```

The confirm_s sweep's `--values` narrows the threshold values scanned (comma-separated) and `--seeds` the synthesis seeds (a case can fail on ONE seed only: the noise grain fills the hold window a few ms differently); the segmenter sweep's `--holds`/`--reartics` narrow the two hold axes the same way; the capture's `--audit` re-reads an existing take instead of recording one.

One more tool follows the sessions over time:

```bash
# the swallowed-phrase audit, aggregated: one line per session, one per
# take — are the swallowed phrases decreasing? (a 30-min gap opens a
# session; a day subtotal keeps the calendar readable)
./.venv/Scripts/python.exe -X utf8 scripts/coherence_summary.py
```

The script scans `<folder>/*/*.coherence.csv` (default `prises_live`, pass another folder as the argument). Per session it prints the number of takes, the notes audited (read from each take's `<take>.json` neighbour; shown as `-` when a JSON is missing or unreadable — never guessed), the suspects, the **suspect rate per 100 audited notes** with its **delta against the previous session** (the drop reads without comparing raw counts), the mean **contour rate** (the live/final health indicator, from the takes' `live_vs_final` JSON blocks), the suspects' mean sung range (st) and the longest secondary plateau (s); a second table keeps the per-take detail — each suspect carrying the plateau it hides (its time inside the note, its pitch and the **signed** interval to the label). Every session row also ends with a **trend sparkline** (▁▂▃▄▅▆▇█ — the suspect-rate history up to that session, tall = more suspects), and the multi-week **real-time health trend** closes the report: one line per ISO week with the mean contour rate and the live/final note totals, read from the journal so a deleted take keeps its week's history. The GUI draws that trend as a **pyqtgraph chart** (one point per week, lower is better): clicking a week lists THAT week's replayed takes, and a measured deterioration (more than 10 contour-rate points against the previous week, backed by more than one take) raises a warning on the live page — the one number that says the live view is drifting from the final pass. Every take is verified first (`verify_take`): one whose WAV and JSON diverge is flagged in the table and its counts stay OUT of the rates. The aggregation lives in `core.coherence` (the GUI's follow-up button reads the same module and shows the same trends); a `coherence_journal.json` journal persists each take's audit next to its artifacts, so folder cleanup does not rewrite history. The labels follow the app's language (`v2m --lang`); a folder with no report yet explains where it looked and what to do next (exit 0) while a missing folder exits 2, naming the folder it expects.

And one tool measures the real-time recognition itself, take by take:

```bash
# replay the takes' WAVs through the GUI chain (a fake mic at real-time
# speed — faster delivery is measured to lie) and archive the live/final
# gap: contours, the ±1 rounding family, the real moves, the rate
./.venv/Scripts/python.exe -X utf8 scripts/replay_takes.py [--out rapport.json] [--take live_...] [--changed]
```

For each take owning a `<stem>.wav`, the script feeds the real `LiveCapture` at the real clock, runs the final analysis and classifies every live note through the ONE matching policy (`core.compare` — the same net the GUI outlines and the JSON's `live_vs_final` block reads): identical, ±1 semitone neighbour, moved; the contour rate it prints is the health indicator of real-time recognition. Nothing is exported (a replay never overwrites a take's artifacts); `--out file.json` archives the table; `--take <stem>` restricts the replay to one take; `--changed` re-derives only the takes whose archived audit was measured by an OLDER engine (the engine's code hash travels in the `<take>.live_vs_final.json` sidecar — after a change to the engine, re-measure the verdicts it could have changed). The engine lives in `core.replay` (the stream is injected — a replay never swaps the global `sd.InputStream` the real microphone reads); the GUI reuses it: every live take is replayed automatically when its recording stops, in the background, and the audit replaces the take's `live_vs_final.json` sidecar. The replay's offline reading reuses the take's cached evidence (`<take>.observation.npz`, seeded by the first replay and re-read afterwards) so pYIN — the longest stage — runs at most once per take; a stale or corrupt cache falls back to the full analysis and is re-seeded, never trusted.

The `--changed` follow-up is automated by one script: `engine_drift_watch.py` re-derives every stale audit, compares each fresh row with the sidecar it replaces (per-metric deltas; any note-count movement is a drift, a sub-0.15 rate wobble is the engine's honest session noise) and writes a Markdown non-drift report — exit 1 when a reading moved, so a CI or a pre-commit hook fails loudly when the engine's last change moved a measured verdict. `--root <folder>` points the watch at another takes folder, `--report <file>` names the report (default `docs/engine-drift.md`):

```bash
# after touching the engine: are the archived verdicts still true?
./.venv/Scripts/python.exe -X utf8 scripts/engine_drift_watch.py [--report docs/engine-drift.md]
```

## Project documentation

- **`v2m weekly`** — the weekly recap as text (`--pdf` engraves it with the same tool as the score): the multi-week real-time health, WHEN you sing best (morning/afternoon/evening rates), the day subtotals. The same view lives on the coherence page; the retention (`prises_live/` kept under a documented size budget, oldest takes trimmed whole, history kept in the journal) runs when the recap or an export touches the folder.
- **`CONTEXT.md`** — domain glossary (Analysis, Note, Tempo, Key, Chord, Figure, Meter, Export, Report) and conventions (languages, errors): the reference for any code change.
- **`core/i18n.py`** — the user-visible string catalog (fr/en): every new visible string goes through `tr()`; note names follow the language (`note_display`), stored figures stay canonical (`tr_figure` translates at display time).
- **`AGENTS.md`** — technical learnings and **session rules** (non-regression, docs updated with the code, readability) for coding agents; read automatically when the option is enabled in the client.

## Quality & limits

- Every report carries a **`coherence` block**: each long note is audited for a hidden secondary plateau (> 120 ms held more than a semitone away from the note's label) — a re-sung phrase glued into one continuity shows up there instead of passing silently. The same audit is written next to every take as `<take>.coherence.csv` (semicolon-free, one row per suspect, header only when clean) so sessions can be compared in a spreadsheet. `python -c "from v2m.core.verify import verify_take; print(verify_take('prises_live/<take>/<take>'))"` checks that a take's WAV and JSON describe the SAME signal.

- **Why the final score differs from what you watched**: a live take's report carries a **`live_vs_final` block** decomposing the gap (the ±1 semitone rounding family — a rounding, not a mistake; the real moves; the glides the offline pass absorbs), the results panel turns it into one honest line per cause, and every live take archives it as `<take>.live_vs_final.json` beside its artifacts (the replay cross-check updates it with the re-derived reading).

- **Monophonic voice** (one note at a time) — polyphonic transcription is out of scope
- **Chords on voice alone**: potentially ambiguous (a melody suggests several harmonizations) — the report provides alternatives; fewer than two distinct pitches → no chord
- pYIN ≈ 0.3× the file's real duration
- **Built-in staff**: home-grown engraving (not MuseScore) — monophonic, treble clef, unbeamed notes (flags on eighths) with rests; the exported PDF remains the "professional" engraved version
- The detected tempo is the one of the (human) sung performance; quantization stays optional
- **Transcription validated on synthetic voice** (`tests/voicetruth.py`, 19 cases: legato, détaché, consonants, vibrato, noise, low/high voices, eighths/sixteenths, portamentos): offline segmentation is **exact** — every sung note found, none invented. The harness is synthetic: a real noisy take remains to be measured (`scripts/score_accuracy.py`)
- In real time, segmentation is deliberately coarser than the final analysis (no looking back): sixteenths (≈120 ms) can be merged and a very long portamento (>250 ms) shifts the attack. Pitches stay right, and the final analysis — the one exported — redoes everything from the recorded audio

## Tests

```bash
pytest tests/ -v
```

720+ tests: pitch (YIN/pYIN on known sines, **block YIN stream**: monotonic absolute times and bounded memory), segmentation, tempo (clicks + refinement on attacks), key, **chords** (triads, sevenths, bass, segments), **per-measure progression**, figures, **meter by accents (2/4, 3/4 vs 4/4) and dotted-pulse 6/8**, ×1.5 tempo corrected live and offline, MusicXML (+ **chords/lyrics/colors**), FR/EN note names, **audio rendering**, **PianoView keyboard**, **notation** (spelling in key signature, key signatures, written values, measure/system layout), **staff** (offscreen render, systems, live scrolling), audio-loading errors, MIDI export (meta-events re-read), live segmenter (+ **current note** for real-time tracking), **live piano roll tracking** (sliding window, preview, stable pitch band, recoloring after labeling), **end-to-end live mode** (fake mic → capture → final analysis → export), **live capture off the callback** (queue + thread, blocks never lost, caller error non-fatal), **replay engine** (real chain at the real clock, injected stream — never a global swap —, worker archive, cancellation), **live-vs-final explainer**, **coherence trends** (sparkline, journal-fed weekly contour trend), **CLI `--pdf`** (tool missing → PDF skipped, exit 0; conversion failed → exit 2, artifacts and summary always kept), **min/max waveform envelope**, **file-dialog i18n** (and fr/en key parity), **diagnostics script** (stage table aligned on the real pipeline, averages, cProfile, JSON, input errors), E2E WAV → artifacts pipeline, **transcription accuracy on synthetic voice** (19 cases, offline exact, attack evidence never worse than geometry, no note invented inside a sung note, real time exact on articulated phrases, **breaths never notated**: air intakes, isolated breath, breathy held note), **attacks** (sung onset coverage, a held note with vibrato no longer produces 19 attacks, conservative real-time flux), **transition rules** (glissando absorbed, genuine short note preserved, arrival pitch), **phantom meter** (a 4/4 line is not transcribed as 6/8), **single threshold policy** (`FILE_CONF`/`LIVE_CONF`, no locally rewritten default).

Quick loop while iterating — skips the `slow`-marked tests (the real-time replays and the golden fingerprints):

```bash
pytest tests/ -m "not slow"
```

Measured: ~81 s for the full run against the 150 s suite budget, ~34 s for the quick loop (about 47 s saved, well over half of the run). Every run prints its own cost (`suite cost: ...`); a green run past 150 s fails on purpose, so adding a slow test stays an explicit decision. The `slow` tests stay in the default run — CI runs the full suite, the quick loop is for the local loop.

Headless GUI smoke test:

```bash
QT_QPA_PLATFORM=offscreen python scripts/gui_smoke_test.py
```

## Generate the demo audio

```bash
python scripts/make_demo.py     # demo.wav: a C major scale at 120 BPM
v2m analyze demo.wav --quantize
```
