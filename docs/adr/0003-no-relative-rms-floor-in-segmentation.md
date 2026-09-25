# ADR 0003 — No relative RMS floor in segmentation

- **Status**: Accepted (2026-09-23)
- **Deciders**: session of 2026-09-23 (breath-boundary measurement)
- **Scope**: `src/v2m/core/liveseg.py`, `src/v2m/core/notes.py` (segmentation), `tests/voicetruth.py` (breath synthesis)

## Context

A breath riding on a held note can fragment it into micro-notes. One proposed
defense: a **relative RMS floor** in the segmenter — a pitch plateau only
counts as a note while it stays above a fraction of the take's own (running
or median) RMS, so breath ripples, being quieter than the voice, fall below.

## Decision

**Refused.** No relative RMS floor in the segmentation path — offline or live.
The breath defense stays what it is: pitch absence + the attack
normalization (take-relative, anchored by consonants).

## Rationale — the measured boundary is two-condition, and a floor helps neither

Measured on synthetic takes (`tests/voicetruth.py`, harness noise units):

1. **A breath in a gap or a silence never becomes a note, at any amplitude**
   — even at peak level. The segmenter requires confirmed pitch before a
   note exists; a breath has none. A level floor adds nothing here.
2. **A breath ON a held note fragments it only when no consonant anchors the
   take-relative attack normalization** (`consonant_s=0`, absolute noise
   0.10 ≈ 3× the harness contract). With consonants present, the contract
   holds up to **3× the contract level** (noise 0.30) without fragmenting
   held notes as long as 2.4 s.

Against that boundary, a relative floor is the wrong tool on both edges:

- It **would not extend** condition 2: the fragmenting take has no anchor
  (`consonant_s=0`), so a floor keyed to the take's level distribution has
  nothing reliable to key on — the ripples *are* part of the level.
- It **breaks the quiet-voice case**: the 19-case contract includes a quiet
  take (amp 0.05). A floor relative to a loud reference (the consonants of
  the same take) rejects the notes of a genuinely quiet take. The refusal
  is what keeps `tests/test_voice_accuracy.py` green.

## Consequences

- The boundary is pinned by `test_breaths_never_become_notes` (breath
  synthesis owned by the harness: `voicetruth.add_breath`); the boundary
  wording lives in AGENTS.md ("Human-error coverage").
- Revisit only with a discriminator that separates breath from voiced
  energy **without a level floor** (e.g. spectral flatness), and re-run the
  19-case sweep plus the quiet-voice case before accepting it.
