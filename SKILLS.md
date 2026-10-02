# SKILLS.md — Voix → MIDI

Step-by-step procedures. The FACTS they act on (thresholds, measured
bounds, invariants) live in `AGENTS.md`; the domain terms in `CONTEXT.md`;
the refusal decisions in `docs/adr/`. A procedure moved here is cited from
AGENTS.md — never duplicated there.

## Skill 1 — Validate a behavior change (the non-regression contract)

Interpreter: the commands show the Windows + Git Bash venv python; on
macOS/Linux use `.venv/bin/python` instead (AGENTS.md, Commands).

1. Run the full suite: `./.venv/Scripts/python.exe -m pytest tests/ -q`
   (budget 150 s — see the `suite cost` trailer; a green over-budget run
   is exit 1).
2. Run the GUI smoke test:
   `QT_QPA_PLATFORM=offscreen ./.venv/Scripts/python.exe -u scripts/gui_smoke_test.py`
   (needs the gitignored `demo.wav`; regenerable per README).
3. Run pyright on the touched files only.
4. If a frozen reference or a pinned test moved: READ WHY first (diff the
   fingerprint / read the failure), then decide. Re-freezing without an
   explanation is forbidden:
   - golden fingerprint: `rm <audio>.golden.json` ONLY after reading what
     moved, then `scripts/score_accuracy.py <audio> --golden`;
   - reference artifacts: `V2M_FREEZE_REFERENCE=1` ONLY after reading what
     moved;
   - say so in the report either way.
5. Add the regression test IN THE SAME CHANGE (a test that passes either
   way pins nothing).

## Skill 2 — Diagnose a suite-budget breach (machine speed vs test bloat)

1. Compare today's `suite cost` with the last calibration's measured
   seconds on the SAME code (history: the dated comments in
   `tests/conftest.py` — the only copy). A green run once measured 106–120 s
   where the same code ran 78.7 s the day before: pure environment variance.
2. Within ~15 % of the last calibration → the budget is NOT the problem;
   find the load (spinning processes, indexing, antivirus).
3. If the speed genuinely moved: re-measure a clean full run, recompute
   `budget = measured × 1.85`, edit `SUITE_BUDGET_S` in conftest.py with a
   dated comment, and update the budget paragraph in AGENTS.md.
4. NEVER re-normalize the budget upward to make a NEW slow test fit —
   adding slow tests must stay an explicit decision.

## Skill 3 — Re-freeze a moving reference (golden / artifacts)

1. Read the drift: `scripts/engine_drift_watch.py` exit 1 names the moved
   verdicts; `score_accuracy.py --compare` prints the fingerprint delta.
2. Read WHY the engine changed the reading (the threshold or stage the
   diff touched). A change with no explanation is a bug until proven
   otherwise.
3. Only then re-freeze (Skill 1, step 4), and say so in the report.

## Skill 4 — Document a new user-facing flag or script

1. Add the flag to `build_parser` (every user-facing flag must be
   documented — and every documented flag must exist).
2. One phrase of semantics per flag, in BOTH `README.md` and
   `README.fr.md`, in the mirrored sections (mirrors are tested:
   `test_readme_mirror.py` fails on heading drift, one-language-only
   flags, broken cross-links, diverging example commands).
3. Scripts run IN-PROCESS by tests when possible (`main()` importable,
   stdout captured): a subprocess boot pays librosa/numba again AND reads
   the real `~/.v2m_lang`. Booting test files must be whitelisted in
   `test_docs_guard.py` (guard 8) with the declared argv.
4. Update the scripts list in AGENTS.md (Commands) and CONTEXT.md if a
   concept was named.

## Skill 5 — Diagnose a failed live take

1. Find the take's folder `prises_live/<stem>/`; read `<stem>.trace.json`
   (written by `LiveSessionController.fail_take` from the take's `failed`
   event — one trace per take; `reason`, `signal_bytes`, `exported`,
   `written_at`).
2. No folder, no trace = the capture never produced audio (a Start press
   against a dead mic): the failure was announced on the status label
   only — there was no take to diagnose.
3. A folder with a trace but no artifacts: the take died before its first
   export — the `reason` line names the phase (mic silent / capture too
   short / capture lost before finalize). The trace reads the signal as
   FLOAT32 (4 B/sample): `signal_bytes / 4 / 22050` ≈ captured seconds.
4. Empty folder WITHOUT a trace: only an app kill mid-export can produce
   that now (no Python ran). Everything else explains itself.
5. The audit tools may skip trace-only folders (a trace is a diagnostic,
   not an artifact).

## Skill 6 — Add a new output path or artifact

1. Write it through the artifact contract in `core/takes.py`
   (`stem_path`/`sidecar_path`/`evidence_path`), never a spelled suffix.
2. Update `.gitignore` in the same commit (anchor root-only patterns with
   a leading `/`); verify with `git status --untracked-files=all`.
3. Update `report.ARTIFACT_EXTENSIONS` if the export announces it, and
   `takes.ARTIFACTS` if the take owns it.
4. Document it (Skill 4) and add the regression test (Skill 1, step 5).
