# ADR 0001 — No shared module under the CLI and live adapters

- **Status**: Accepted (2026-09-23)
- **Deciders**: architecture review of 2026-09-23 (grilling session with the user)
- **Scope**: `src/v2m/live.py` (`LiveSession`, CLI), `src/v2m/gui/main_window.py` (live page), `src/v2m/core/live_session.py` (`LiveTake`), `src/v2m/core/report.py` (`export_result`)

## Context

The architecture review (candidate 3) proposed merging the two consumers of the
live pipeline — the CLI's `LiveSession` and the GUI's live path — behind one
common adapter module that would own "analyze a finished take and export it".

The structure already in place (established by the architecture pass frozen in
AGENTS.md):

- **Core owns musical facts.** `LiveTake` is the single owner of the live grid
  policy: it labels the figures and publishes the `BeatGrid` ON the
  `note_closed` event. `finalize()` runs the final analysis and emits it.
- **Adapters own presentation.** `LiveSession` (CLI) prints and exports;
  `MainWindow` (GUI) renders and exports. Both consume ready-labeled facts —
  the AST guard `tests/test_adapter_purity.py` fails if any adapter re-derives
  one (`estimate_bpm_live`, `detect_meter`, `label_notes`).
- **The seam already exists**: `report.export_result` is the single export
  entry point for CLI, GUI and live; `AnalysisThread` is the one GUI worker
  for finished-take analysis.

## Decision

**Refused**: no common module (and no common protocol/ABC) under the CLI and
live adapters, even though they both call `take.finalize()` then
`export_result()`.

## Rationale — the deletion test

Applying the deletion test: deleting such a module would *concentrate*
complexity (a yes is the signal that a module deserves to exist). Here it
would only *relocate* it: the "shared" module would be a pass-through whose
entire body is `result, signal = take.finalize(); export_result(result, stem)`
— two calls the adapters already make in three lines each. It adds one
concept (a shared-adapter abstraction) without removing any: the CLI still
needs its own printing/export paths, the GUI still needs its Qt-side owner
and rendering callbacks. That is the definition of a **shallow module**.

The design vocabulary's rule "one adapter = hypothetical seam, two = real"
applies to *producers*, not consumers: the two adapters do not form a second
seam that begs unification — they are the two legitimate *consumers* of the
one real seam (`LiveTake`'s events + `export_result`). Each is thin by
construction (the CLI session is ~60 lines; the GUI path owns only rendering
and export callbacks), so duplication cost is bounded and smaller than the
indirection cost.

## Consequences

- Future architecture reviews must not re-propose a common CLI/GUI adapter
  module unless this ADR is reopened first.
- The trigger for revisiting: a **third** real consumer appearing with
  materially duplicated logic beyond `finalize()` + `export_result()` (test
  harnesses do not count — they may consume the adapters directly).
- Adapters stay consumption-only; the AST guard
  (`tests/test_adapter_purity.py`) keeps detecting re-derivation.
