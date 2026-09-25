**English** | [Français](README.fr.md)

# 🎤 Voice → MIDI

Desktop application that **detects sung notes** (pitch, note values, meter), the **tempo**, the **key** and the **chords**, then exports a **MIDI file**, a **MusicXML score** (optional PDF) and a **JSON/CSV report**. Works on **audio files** (WAV/MP3/FLAC/OGG) or **live from the microphone**.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)

## Features

- **Pitch detection**: pYIN (librosa) offline, incremental YIN in real time — search range C2–C7, so soprano top notes read at their own octave (the old C6 ceiling folded them one octave down)
- **f0 stabilization**: the pitch derivative marks every moving zone (portamento ramp, note hand-over) as a **transition**, excluded from the pitch estimate — each note reads its **stable plateau**, with the vibrato riding on top
- **Tuning correction**: a singer sitting consistently flat or sharp is re-centered — one global offset (median deviation to the nearest semitone, capped at ±0.35 st) is absorbed **before** rounding, so notes stop rounding on the wrong side; the residual per-note deviation feeds the key-aware snap
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
- **MIDI export**: `set_tempo` + `key_signature` meta-events, velocity proportional to confidence, optional 1/16 quantization, ±48 semitone transposition
- **PDF export**: score engraving via MuseScore or LilyPond (auto-detected, install hint otherwise) — `--pdf` on the CLI, a checkbox in the GUI
- **JSON report** (everything, including the pitch curve, the meter, the measure count, each note's measure:beat position, the global chord and the chord segments) + **CSV** (note list with value and position)
- **PyQt6 GUI**: polished dark interface, piano roll, waveform, results panel with per-note (position, value) and per-chord-segment tables, drag & drop, **notation mode** (color by note value), **built-in red error banner** (no blocking popups)
- **Real-time mode**: note/Hz/level indicator, **piano roll that follows the capture** (10 s sliding window, time cursor, `mm:ss` clock, note counter and **detected meter** — 3/4, 4/4, 6/8 — shown live), comparison with an already-analyzed score (green/red keys), the held note shows **while you sing it** (~125 ms display latency), note values labeled live, export on stop — pitch tracking runs **outside the audio callback** (dedicated thread), so the driver never drops blocks
- **Note finder**: isolated mic recognition, 88-key keyboard and a local history of the last six notes; no analysis, no export
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

The `analyze` command writes `sortie.mid`, `sortie.musicxml`, `sortie.json` and `sortie.csv` (`--pdf` adds `sortie.pdf` if MuseScore or LilyPond is installed — otherwise the PDF is skipped with an install hint and the exit code stays 0; if engraving fails, exit 2 and the other four artifacts are still written); the JSON carries, for each note, its `figure` field (e.g. `"noire"`, `"croche pointée"` — figure names stay French canonical data), its `position` `[measure, beat]`, its `name` / `name_fr` names (English and French), the detected `meter`, `num_measures`, the global `chord` and its `chord_segments`.

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
└── score_accuracy.py    # diagnostics: what the transcription did to a real take

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

## Project documentation

- **`CONTEXT.md`** — domain glossary (Analysis, Note, Tempo, Key, Chord, Figure, Meter, Export, Report) and conventions (languages, errors): the reference for any code change.
- **`core/i18n.py`** — the user-visible string catalog (fr/en): every new visible string goes through `tr()`; note names follow the language (`note_display`), stored figures stay canonical (`tr_figure` translates at display time).
- **`AGENTS.md`** — technical learnings and **session rules** (non-regression, docs updated with the code, readability) for coding agents; read automatically when the option is enabled in the client.

## Quality & limits

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

440+ tests: pitch (YIN/pYIN on known sines, **block YIN stream**: monotonic absolute times and bounded memory), segmentation, tempo (clicks + refinement on attacks), key, **chords** (triads, sevenths, bass, segments), **per-measure progression**, figures, **meter by accents (2/4, 3/4 vs 4/4) and dotted-pulse 6/8**, ×1.5 tempo corrected live and offline, MusicXML (+ **chords/lyrics/colors**), FR/EN note names, **audio rendering**, **PianoView keyboard**, **notation** (spelling in key signature, key signatures, written values, measure/system layout), **staff** (offscreen render, systems, live scrolling), audio-loading errors, MIDI export (meta-events re-read), live segmenter (+ **current note** for real-time tracking), **live piano roll tracking** (sliding window, preview, stable pitch band, recoloring after labeling), **end-to-end live mode** (fake mic → capture → final analysis → export), **live capture off the callback** (queue + thread, blocks never lost, caller error non-fatal), **CLI `--pdf`** (tool missing → PDF skipped, exit 0; conversion failed → exit 2, artifacts and summary always kept), **min/max waveform envelope**, **file-dialog i18n** (and fr/en key parity), **diagnostics script** (stage table aligned on the real pipeline, averages, cProfile, JSON, input errors), E2E WAV → artifacts pipeline, **transcription accuracy on synthetic voice** (19 cases, offline exact, attack evidence never worse than geometry, no note invented inside a sung note, real time exact on articulated phrases, **breaths never notated**: air intakes, isolated breath, breathy held note), **attacks** (sung onset coverage, a held note with vibrato no longer produces 19 attacks, conservative real-time flux), **transition rules** (glissando absorbed, genuine short note preserved, arrival pitch), **phantom meter** (a 4/4 line is not transcribed as 6/8), **single threshold policy** (`FILE_CONF`/`LIVE_CONF`, no locally rewritten default).

Quick loop while iterating — skips the 4 `slow` golden/fingerprint tests:

```bash
pytest tests/ -m "not slow"
```

Measured: ~47 s for the full run against the 90 s suite budget, ~31 s for the quick loop (about 16 s saved, roughly a third of the run). Every run prints its own cost (`suite cost: ...`); a green run past 90 s fails on purpose, so adding a slow test stays an explicit decision. The `slow` tests stay in the default run — CI runs the full suite, the quick loop is for the local loop.

Headless GUI smoke test:

```bash
QT_QPA_PLATFORM=offscreen python scripts/gui_smoke_test.py
```

## Generate the demo audio

```bash
python scripts/make_demo.py     # demo.wav: a C major scale at 120 BPM
v2m analyze demo.wav --quantize
```
