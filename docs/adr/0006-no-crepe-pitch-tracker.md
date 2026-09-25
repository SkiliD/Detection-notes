# ADR 0006 — No CREPE pitch tracker (pYIN + attack evidence kept)

- **Status**: Accepted (2026-09-25)
- **Deciders**: accuracy overhaul (octaves, vibrato, glissando, voicing,
  quantization review) — the user asked whether CREPE would help
- **Scope**: `src/v2m/core/pitch.py` (the only pitch readers), the live
  DSP budget (`core/audio_io.py`, `core/liveseg.py`), `requirements.lock`

## Context

CREPE (a 6-layer CNN over a 1024-bin spectrogram, trained on ~17 h of
monophonic pitch annotations) is the strongest public pitch tracker on
noisy polyphonic-ish material. The question: would swapping `pyin` for
CREPE (via `torchcrepe`) improve this project's transcription?

The measured facts of THIS project, as of the 2026-09-25 accuracy
overhaul:

1. **The pitch curve was never the fault.** The octave errors (a sung 84
   read as 72, a 79-88 soprano line read an octave down) came from the
   pYIN **search ceiling** sitting exactly on the sung C6
   (`FMAX = 1050.0`), not from the estimator: one constant (FMAX →
   2100 Hz) reads them with pYIN, byte-identical exports on both real
   takes. The half-tone-too-low notes (62→61, 67→66, 72→71) came from an
   inverted condition in the segmentation's glide-attachment rule —
   the curve was correct (median 61.99 for a sung 62, vprob 0.985).
   The duplicate on descending glides came from the re-articulation
   split and the attack-boundary convention. All downstream.
2. **Ground truth is synthetic + two real takes.** CREPE's advantage
   (robustness on real noisy recordings) cannot be evidenced here: the
   harness is a clean harmonic voice, and `chant.wav` has no note-level
   truth (only a golden fingerprint). Swapping estimators would re-freeze
   both references with no accuracy claim to verify against.
3. **The live path is hop-budgeted.** `LivePitchStream` runs YIN per
   23 ms block inside a dedicated DSP thread that must drain faster than
   real time (a slow drain drops driver blocks — AGENTS.md). torchcrepe
   is a per-frame torch forward pass; on CPU it does not fit the
   23 ms/block budget reliably on a modest machine, and it pulls torch
   (~hundreds of MB) into a `python:3.12-slim` image where `requirements`
   are already resolved. A streaming CREPE would also change latency
   characteristics (`confirm_s` was swept to 0.10 against the CURRENT
   curve).
4. **The whole contract is calibrated on the current curve.** `min_conf`
   semantics (ADR 0005), `clean_pitch`'s filters, `HOLD_ST`, the truth
   corpus (23 cases), `ATTACK_TOL_S`, the golden fingerprints. CREPE's
   confidence (POV) is a different distribution; every threshold would
   need re-calibration, and the corpus cannot arbitrate the result.

## Decision

**CREPE is not adopted.** The offline tracker stays `librosa.pyin`
(FMAX raised to 2100 Hz) and the live tracker stays the block-wise YIN
(`pitch.YinBlock`). Attack evidence (`onsets.py`) — not a different pitch
estimator — is what this project's accuracy is built on.

## Revisiting this decision requires

- a REAL note-level ground truth (recorded takes with hand-checked
  note lists), not the synthetic harness;
- a measured live budget proof (torchcrepe forward pass per 23 ms block
  on the target machine class, p99 < block period, torch in the slim
  image);
- a re-calibration plan for every threshold above, with the corpus
  extended to the failure cases CREPE is claimed to win (breathy real
  recordings, low-SNR takes).

## Consequences

- `requirements.lock` stays torch-free; the slim Docker image stays small.
- Pitch-side improvements remain cheap constants/thresholds on pYIN
  (e.g. the 2026-09-25 FMAX raise) instead of a model swap.
- If a future note-level truth corpus exists, run it through BOTH
  trackers before deciding — the ADR refuses the swap without evidence,
  not the evidence itself.
