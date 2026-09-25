# ADR 0004 — Chord display: whole-label translator, no per-quality expansion

- **Status**: Revised (2026-09-24) — the whole-label translator this ADR
  anticipated now exists (`tr_chord_label`); the per-suffix table refusal
  stands.
- **Deciders**: i18n closure pass; revised on the user request "whole-label
  chord translator (Am → La mineur) with its guard test"
- **Scope**: `src/v2m/core/i18n.py` (`tr_chord_label` + `_CHORD_QUALITY_FR`
  added; the old per-suffix `tr_chord_quality` stays deleted),
  `src/v2m/core/chords.py` (`CHORD_TEMPLATES` unchanged — canonical storage)

## Context (original decision)

The i18n closure pass aligned every display translator on one design: a
module-level table keyed by the WHOLE canonical name (`_FIGURE_EN` for
figures, `_MODE_FR` for key modes), each pinned by a guard test derived from
its production source (`grid.CATALOGUE`, `keys.KEY_PROFILES`).

The remaining translator, `tr_chord_quality`, mapped chord-quality codes to
French (`"m"` -> `"mineur"`; every other quality mapped to itself). A census
found **zero call sites**: it existed only in its own definition and
`__all__`. It was deleted (no-dead-code rule), with the note that a real
French chord rendering would have to rebuild the whole label (`Am` ->
`La mineur`) — root note names included — and should be specified as that
whole-label translator, not revived from the stub.

## Revision: the whole-label translator, specified as anticipated

**Adopted**: `tr_chord_label(label) -> str`, the third table-backed display
translator.

- The label is PARSED against `music.NOTE_NAMES` (the English table
  `chords.py` builds labels from), two-char roots first so `C#m` is not
  `C` + `#m`; the root is then DISPLAYED through the solfège table
  (`NOTE_NAMES_FR`).
- The quality — the whole suffix after the root, one of the
  `CHORD_TEMPLATES` keys — is looked up in `_CHORD_QUALITY_FR`
  ("" -> "majeur", "m" -> "mineur", "dim" -> "diminué", "aug" -> "augmenté",
  "7" -> "septième", "maj7" -> "septième majeure", "m7" -> "septième
  mineure").
- Closed vocabulary, same contract as the other translators: an unknown root
  or quality comes back unchanged, never half-translated ("Am9" stays
  "Am9").
- Guard test derived from the producer: every quality in
  `chords.CHORD_TEMPLATES` must translate (`test_tr_chord_label_covers_the_
  detection_vocabulary`), mirroring the catalogue/mode guards.
- Wired at the display edges only: CLI summary, live export summary,
  results panel (global chord, alternatives, per-segment and per-measure
  tables). Storage stays canonical: `report.json`, `score.musicxml` and the
  MusicXML `<harmony>` still carry `Am`/`Cmaj7` — a file meant for importers
  must not depend on a UI language.

## Rationale

This resolves the original refusal's condition rather than contradicting it:
the stub could not deliver a translation (`"m"` -> `"mineur"` left `Am`
reading as `Am`), and the ADR demanded the whole-label design *with root
names* before any chord display work. The user asked for exactly that.

## Consequences

- `i18n.py` now has three table-backed display translators, each guard-pinned
  to its producer: `tr_figure` (`grid.CATALOGUE`), `tr_key_label`
  (`keys.KEY_PROFILES`), `tr_chord_label` (`chords.CHORD_TEMPLATES`).
- Chord qualities still stay canonical codes in all stored data; only the
  display edges translate.
- `CHORD_TEMPLATES` remains the single source of the quality vocabulary —
  `_CHORD_QUALITY_FR` is a display mirror, and the guard test is what keeps
  the two in sync (a quality added without its word fails the test, not the
  user).
