# ADR 0002 — The bilingual READMEs are maintained by hand, with the mirror test as the net

- **Status**: Accepted (2026-09-23)
- **Deciders**: session of 2026-09-23 (bilingual README restructuring)
- **Scope**: `README.md` (English primary), `README.fr.md` (French), `tests/test_readme_mirror.py`

## Context

The README ships in two languages: `README.md` is the English primary,
`README.fr.md` the French version (the original wording). Session rules make
updating the README MANDATORY on every user-facing change — which means every
such change touches BOTH files.

Two structural options existed:

1. **Single source**: one canonical document (or a template with slots) from
   which both languages are generated.
2. **Two hand-maintained files**, kept in sync by a contract test.

## Decision

**Two hand-maintained files** (option 2). No generator, no template, no
single-source pipeline.

## Rationale

- **The repo has no documentation build.** There is no doc toolchain, no
  static-site generator, no markdown pipeline anywhere in the project — a
  generator would introduce the first one, a build step and a dependency,
  for a two-file problem. The deletion test fails: deleting the generator
  would concentrate nothing (the content simply returns to two files).
- **The prose is a translation, not a template.** The French version is the
  original wording; the English is its translation. Sections legitimately
  differ in prose length and phrasing — a slot-based template would force
  either paragraph-level parallelism (brittle, fights the translator) or a
  free-form template (which is just two files again, with extra steps).
- **What must NOT differ is small, enumerable and testable**: the heading
  level sequence, the CLI flags, the scripts referenced, the example
  commands and their sample values, the cross-links. Exactly the surface
  `tests/test_readme_mirror.py` already pins.

## Consequences

- The mirror test is the net: it fails on a heading-level sequence drift, a
  CLI flag or script documented in one language only, an example command
  that diverges (comments inline are prose and may be translated; file names
  may not), and on broken cross-links.
- The test is also bilateral with reality: every documented flag must exist
  in `build_parser` or in the diagnostic scripts, and every user-facing flag
  of those must be documented — the doc cannot drift from the CLI either
  way. Internal flags (`--version`, `--help`) and the documented-once short
  alias `-o` are exempted explicitly in the test.
- Adding a doc generator later would mean revisiting this ADR first; the
  trigger would be a README pair too large to mirror by hand (measured in
  review, not speculation).
