# ADR 0005 — No confidence-threshold mask in clean_pitch (vprob published instead)

- **Status**: Accepted (2026-09-24)
- **Deciders**: breath-as-note fix (user report: breaths counted as notes)
- **Scope**: `src/v2m/core/pitch.py` (`track_pitch_offline` now publishes
  pYIN's voicing probability), `src/v2m/core/notes.py` (`clean_pitch` mask)

## Context

The old `track_pitch_offline` kept pYIN's binary voiced FLAG and discarded
its voicing probability (the third return): `confidence` was 1.0 wherever an
f0 existed, so `clean_pitch`'s `min_conf` mask was a no-op against
unvoiced-adjacent doubt, the MusicXML notehead colors and the MIDI
velocities were constant, and a breath's noise rode the harmonic unmasked.

pYIN separates the cases differently than any post-hoc threshold can:
- a clear sung frame: vprob 0.89–0.99;
- a breathy frame at the contract noise (0.10): still ≥ 0.83;
- a breath ON a held note: a few frames at 0.49–0.59 among 0.95 neighbours;
- EVERY frame of a portamento: 0.016–0.62 — the 93 ms window sees a glide
  as noise, so a transition is a COLLECTIVE doubt;
- a breath in a gap: f0 = NaN (already excluded without any mask).

## Decision

**The probability is published as `PitchTrack.confidence`** (traveling into
`Note.confidence`, report.json, the MIDI velocities and the MusicXML
colors — real information where the flag wrote constants), and
**`clean_pitch` masks only frames without an f0** (`confidence == 0`),
never a doubt threshold. `min_conf` is kept as a parameter for API
stability but is deliberately inert.

**Refused** (both measured, both on the harness AND on `chant.wav`):
1. **Hard mask** (`conf >= min_conf`): every transition frame drops, the
   hysteresis segments shatter, `_attach_run` loses its glide fusion —
   5 truth cases fail.
2. **Local-density mask** (mask an isolated doubt, keep a collective one):
   separates the synthetic cases, but on the real take it shifts note
   boundaries enough to move the refined tempo 171.0 → 153.5 BPM (AGENTS.md
   documents 171 as the correct reading) and breaks a measure closure
   (102/96 divisions in measure 7). No threshold exists that excludes the
   breath but keeps the glide — the doubt values overlap (0.49–0.59 vs
   0.016–0.62).

## Consequences

- The voicing doubt travels OUT instead: a breathy note's frames export a
  lower confidence, which is the honest engraving (dimmer notehead color,
  softer MIDI velocity).
- The reference digests moved ONCE for this change (score.json carries the
  new confidence values; counters identical: 29 notes @ 171.02, 15 @
  120.18 — re-frozen after reading exactly that diff).
- The known "breath on a held note fragments it" hole is UNCHANGED and
  stays documented as out of contract (test_voice_accuracy docstring,
  ADR 0003's boundary): the fix there is the attack-side normalization,
  not a confidence mask. A future architecture review re-proposing a
  voicing threshold must first beat measurement #2 above.
- `tests/test_pitch.py::test_pyin_confidence_is_the_voicing_probability`
  pins the publication (max < 0.999 on a sung take — the old flag wrote
  exactly 1.0).
