"""Onset (attack) detection: the evidence the pitch curve cannot give.

A new sung note starts with a transient — a consonant, a glottal stop, a new
vowel, a fresh breath — while a legato glide between two notes has none. The
note segmenter needs that evidence to tell a real note start from a pitch
sweep: measured on a synthetic voice, a 90 ms glide alone turned 6 sung notes
into 10 detected ones, all four extra notes sitting at intermediate pitches.

Two detectors, because the two situations are not the same:

- `detect_attacks(y, sr)`: offline, one STFT (spectral flux in the character
  band plus the RMS rise), then peak picking. It sees the whole take, so it
  can normalize against it — this is what the analysis pipeline uses.
- `AttackTracker`: streaming, one block = one O(1) test against two voices:
  the LEVEL rise (slow baseline floor) and the broadband BURST (a real
  >3 kHz high-pass plus a level dip). Deliberately conservative (see its
  docstring): a false attack splits a held note live.

`compute_onset_strengths` lives here too: the per-note accent (0..1) read
off the audio envelope — the same evidence, quantized per note, that the
meter and the notation consume.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

FRAME_MS = 20.0        # offline frame hop (matches the pitch tracker)
MIN_RMS = 0.006        # absolute floor: silence never produces an attack
MIN_GAP_S = 0.060      # two attacks closer than this are the same event
MIN_SCORE = 1.0        # peak height required, in units of the reference ratio
MIN_STRENGTH = 0.20    # relative floor: a breath-noise ripple is not an attack
BASELINE_MS = 400.0    # slow baseline: a phrase, not a single note
CHARACTER_HZ = 4000.0  # consonant bursts are broadband: look above the harmonics
FLUX_REF = 25.0        # rms-relative character-band flux of a consonant burst
RISE_REF = 0.35        # rms-relative level rise of an attack
STREAM_LEVEL = 1.5     # live: level this much above its own slow baseline
STREAM_FLOOR = 0.006   # live: RMS floor
STREAM_ALPHA = 0.15    # live: baseline tracking speed when rising (~7 blocks)
STREAM_ALPHA_DOWN = 0.35  # live: baseline follows a level DROP faster (~3 blocks)
MAX_RECENT = 256       # live: recent attacks kept for the display/tests
# Live re-articulation trigger, second voice: the consonant burst of a
# re-attacked note leaves the LEVEL nearly flat (measured 1.28x, under
# STREAM_LEVEL — a level-only tracker reads ta-ta-ta-ta on one pitch as two
# notes) but it is NOISE, and noise is spectrally FLAT while a sung vowel is
# made of peaks. Measured on the voice harness (spectral flatness per block):
# held vowel ~0.000, vibrato ~0.000, legato portamento blur ~0.000 (its
# harmonics are still harmonic), breath on silence 1.0 (gated by pitch in
# the segmenter), breathy voice at 3x the noise contract 0.078-0.0995,
# consonant burst 0.40-0.60. Pitch- and amplitude-independent — a quiet
# take's bursts read the same. Anything above 0.20 is not a vowel.
FLATNESS_TRIGGER = 0.20
# But a burst is not just flat: it REPLACES the vowel's energy for its
# block (0.47-0.74x of the recent peak) while the glide blur holds the
# level (0.87-0.96x). The reference is the recent PEAK, not the baseline:
# the baseline follows a dip (alpha_down), so at the burst block it has
# already decayed to the burst's own level and no threshold separates
# anything from it anymore. Measured margins: bursts <= 0.74, glides >= 0.87.
DIP_RATIO = 0.80
# ...and the dip must be INTERNAL to the block: a consonant burst DIPS the
# vowel inside its own block (the burst rides silence — sub-window rms
# measured 0.004-0.007 vs the note's 0.054-0.058), while a RELEASE block
# (vowel cut, then the envelope falls into the gap before the next note's
# consonant) holds its vowel until the edge (sub_min 0.048-0.055). Without
# this second check, every note-end release fired a spurious attack where
# the pitch stream was still flat — and every hand-over got a duplicate
# fragment (measured: the voice-accuracy grave/haute lines, +1 fragment).
SUBWINDOWS = 8
SUBWINDOW_DIP = 0.20


@dataclass
class Attacks:
    """Detected attacks: `times` (s) and `strengths` (0..1)."""

    times: np.ndarray
    strengths: np.ndarray

    def __len__(self) -> int:
        return len(self.times)

    def near(self, t: float, tol_s: float) -> float | None:
        """Strength of the strongest attack within `tol_s` of `t`, else None."""
        if not len(self.times):
            return None
        close = np.abs(self.times - t) <= tol_s
        if not close.any():
            return None
        return float(self.strengths[close].max())

    def in_span(self, t0: float, t1: float) -> int:
        """How many attacks fall in [t0, t1]."""
        if not len(self.times):
            return 0
        return int(np.count_nonzero((self.times >= t0) & (self.times <= t1)))


EMPTY = Attacks(np.zeros(0), np.zeros(0))


def _sharpness(x: np.ndarray, rms: np.ndarray, frame_ms: float) -> np.ndarray:
    """Rise of `x` above its slow baseline, as a share of the CURRENT level.

    Subtracting a slow median baseline keeps only the sharp part of the
    curve. Dividing by the level (not by the take's peak, which a repeated
    artefact would dominate) is what makes the test scale-free — measured on
    a synthetic voice, a held note with vibrato made the character-band flux
    swing as much as 9 per rms unit, a consonant burst 6000.
    """
    if not len(x):
        return np.zeros(0)
    baseline = median_filter(x, size=_odd(BASELINE_MS / frame_ms))
    return np.maximum(0.0, x - baseline) / np.maximum(rms, 1e-12)


def detect_attacks(y: np.ndarray, sr: int, frame_ms: float = FRAME_MS) -> Attacks:
    """Attacks from a spectral flux combined with the amplitude rise.

    The flux catches consonants and vowel changes (a spectrum that suddenly
    gains energy); the RMS rise catches anything percussive or breathy. Both
    go through `_prominence`, so neither dominates on a given take.
    """
    y = np.asarray(y, dtype=np.float64)
    hop = max(64, int(sr * frame_ms / 1000))
    n_fft = max(1024, 2 * hop)
    if len(y) < n_fft:
        return EMPTY

    import librosa

    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    character = spectrum[freqs >= CHARACTER_HZ]
    if not len(character):  # very low sample rate: fall back to the full band
        character = spectrum
    flux = np.maximum(0.0, np.diff(character, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])  # one value per frame, aligned
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]

    # a burst in the character band (a consonant) or a level rise (a breath,
    # a glottal attack), both measured against what the take can show
    score = np.maximum(
        _sharpness(flux, rms, frame_ms) / FLUX_REF,
        _sharpness(rms, rms, frame_ms) / RISE_REF,
    )
    score[rms < MIN_RMS] = 0.0  # never trust a "rise" in near-silence

    peaks, _ = find_peaks(score, height=MIN_SCORE, distance=int(MIN_GAP_S * sr / hop))
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    strengths = score[peaks].astype(float)
    if len(strengths):
        strengths = strengths / float(strengths.max())
        # a noisy take ripples: each ripple above MIN_SCORE would legitimize a
        # spurious note start. The strongest attack of a take IS an attack
        # (a consonant or a glottal onset); against it, breath noise scored
        # <= 0.06 and real onsets >= 0.86 (measured on a breathy voice), so
        # a relative floor removes the ripples without touching consonants.
        keep = strengths >= MIN_STRENGTH
        times, strengths = times[keep], strengths[keep]
    return Attacks(np.asarray(times, dtype=float), strengths)


def compute_onset_strengths(notes, y: np.ndarray, sr: int) -> None:
    """Fill `note.strength` (0..1) from the audio envelope at each onset.

    The signal is cut into small RMS frames; a note's strength is the peak
    envelope around its start, normalized against the recording's own
    quiet/floor percentiles — so accents are RELATIVE within one take,
    which is what meter phase discrimination needs (no absolute loudness).

    The window is deliberately ASYMMETRIC (2 frames back, 7 forward):
    the segmenter dates a note at its first stable-pitch frame, which can
    LEAD the audio onset by tens of ms (measured 37 ms on a dotted-pulse
    take — the 25 ms tone attack sits entirely after the dated start), so
    a symmetric window read the envelope floor and every accent after the
    first note came out 0.0. The tone follows the date; the peak is
    searched where it can be.
    """
    if not notes:
        return
    hop = 256
    n_frames = len(y) // hop
    if n_frames < 4:
        for n in notes:
            n.strength = 0.5
        return
    frames = np.asarray(y[: n_frames * hop], dtype=np.float64).reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    env = np.convolve(rms, np.ones(3) / 3, mode="same")  # ~3-frame smoothing
    floor = float(np.percentile(env, 20))
    span = max(float(np.percentile(env, 98)) - floor, 1e-9)
    for n in notes:
        i = min(n_frames - 1, max(0, int(n.start * sr / hop)))
        lo, hi = max(0, i - 2), min(n_frames, i + 7)
        local = float(np.max(env[lo:hi]))
        n.strength = round(min(1.0, max(0.0, (local - floor) / span)), 3)


def _odd(value: float) -> int:
    n = max(3, int(round(value)))
    return n if n % 2 else n + 1


@dataclass
class AttackTracker:
    """Streaming attack evidence: a level rise above a slow baseline.

    Deliberately CONSERVATIVE, and deliberately not the offline detector.
    Measured on a synthetic voice, the spectral flux — excellent offline —
    cannot be trusted live: a held note with vibrato moves its upper
    harmonics enough to reach a relative flux of 13 while a consonant burst
    reaches 7-11, so no threshold separates them. And a false attack costs
    more live than a missed one: it SPLITS a held note in two, while a missed
    one only dates the onset a little later (the segmenter falls back on the
    moment the pitch settles). The level test has the opposite profile: it is
    quiet on a sung vibrato and fires on a consonant or a fresh breath.

    The baseline is ASYMMETRIC on purpose: it climbs slowly (a note growing
    louder is not an attack) but follows a level drop quickly, and the attack
    is tested against the baseline's FLOOR over the recent past — a slow EMA
    lags a dip by one block, which is exactly the length of the gap between
    two re-articulated notes. Measured: with a plain EMA, a 46 ms gap left
    the baseline at 120% of the next note's rise; against the local floor the
    same rise is 10x. A held note keeps its floor AT its own level, so it
    stays quiet; a vibrato dips less than the ratio above the floor.

    Cost is O(1) per block and memory is bounded (the recent attacks are for
    the display and the tests). The first block primes the baseline instead
    of ramping up from zero, which would fire on everything at the start.
    """

    level_ratio: float = STREAM_LEVEL
    floor_rms: float = STREAM_FLOOR
    min_gap_s: float = MIN_GAP_S
    strength: float = 0.0      # level rise of the LAST attack, 0..1
    peak_ratio: float = 0.0    # loudest attack ratio so far (normalization)
    alpha: float = STREAM_ALPHA
    alpha_down: float = STREAM_ALPHA_DOWN
    recent: deque = field(default_factory=lambda: deque(maxlen=MAX_RECENT))
    baseline_rms: float = 0.0
    floor_rms_seen: float = 0.0   # lowest baseline in the recent past
    hf_ratio: float = FLATNESS_TRIGGER  # flatness the burst must clear
    _last_t: float = -1e9
    _primed: bool = False
    _hf_floor: float = 0.0        # recent LOW of the spectral flatness
    _peak_rms: float = 0.0        # recent HIGH of the level (dip reference)

    @staticmethod
    def level(block: np.ndarray) -> float:
        """RMS of one captured block (0 for an empty block)."""
        x = np.asarray(block, dtype=np.float64)
        return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

    @staticmethod
    def flatness(block: np.ndarray) -> float:
        """Spectral flatness of one block (1 = white noise, ~0 = tonal).

        A consonant burst IS noise; a sung vowel, a vibrato and even a
        legato portamento's blur are still harmonic peaks. Geometric/arithmetic
        mean of the power spectrum of the windowed block — the one
        discriminator measured to separate bursts from glide blurs without
        depending on pitch (a fixed high-pass catches a high note's own
        harmonics) or level (a quiet take reads the same).
        """
        x = np.asarray(block, dtype=np.float64)
        if x.size < 8:
            return 0.0
        ps = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2 + 1e-12
        return float(np.exp(np.mean(np.log(ps))) / np.mean(ps))

    def push(self, t: float, block: np.ndarray) -> bool:
        """Feed one captured block; True when it carries an attack."""
        rms = self.level(block)
        if not self._primed:
            self.baseline_rms = rms
            self.floor_rms_seen = rms
            self._peak_rms = rms
            # unvoiced-first start: no burst seen yet — the reference starts
            # at the trigger so the first voiced vowel sets the real floor
            self._hf_floor = self.hf_ratio
            self._primed = True
            return False
        hf = self.flatness(block)
        # recent LOW of the flatness (the vowel's own colour): a burst must
        # be far ABOVE it; a breath (~1.0, gated by pitch downstream) and a
        # vibrato (~0.0) never are.
        if hf < self._hf_floor:
            self._hf_floor = hf
        else:
            self._hf_floor += 0.35 * (hf - self._hf_floor)
        # recent HIGH of the level, rise instant / decay slow (~1 s): the
        # reference of the dip test (see DIP_RATIO — the decaying BASELINE
        # swallows the very dip it should measure).
        if rms > self._peak_rms:
            self._peak_rms = rms
        else:
            self._peak_rms += 0.02 * (rms - self._peak_rms)
        attack = False
        # a burst rides a live note: the block must carry real energy vs the
        # recent peak (a RATIO, not STREAM_FLOOR — a quiet take's bursts sit
        # an order of magnitude under the floor and every one would be lost)
        burst_like = hf > FLATNESS_TRIGGER and rms > 0.3 * self._peak_rms
        if rms >= self.floor_rms or burst_like:
            # the level reference is the LOW the baseline reached recently:
            # a gap dips the baseline, the next note rises from that dip. A
            # BURST block must not lower it (its dip is the attack itself,
            # not a silence — letting it through made the vowel's recovery,
            # 1.7x the dipped floor, a second rise on the same consonant).
            if rms <= self.baseline_rms and not burst_like:
                self.floor_rms_seen = min(self.floor_rms_seen, rms)
            elif not burst_like:
                self.floor_rms_seen += 0.5 * (self.baseline_rms - self.floor_rms_seen)
            loud = rms > self.level_ratio * self.floor_rms_seen
            # second voice: the consonant of a re-attacked note leaves the
            # level flat (1.28x) while the block turns NOISE-flat — level
            # rise OR (a flat burst with an INTERNAL dip vs the recent peak,
            # which is what keeps a legato portamento's blur — level holds —
            # and a note-end release block — the silence is at the block's
            # edge, not inside — out; see SUBWINDOW_DIP).
            dips = rms < DIP_RATIO * self._peak_rms
            if burst_like and dips:
                x = np.asarray(block, dtype=np.float64)[: len(block) // SUBWINDOWS * SUBWINDOWS]
                sub = np.sqrt((x.reshape(-1, SUBWINDOWS) ** 2).mean(axis=1))
                dips = float(sub.min()) < SUBWINDOW_DIP * self._peak_rms
            spaced = (t - self._last_t) >= self.min_gap_s
            attack = bool((loud or (burst_like and dips)) and spaced)
        if attack:
            self._last_t = t
            self.recent.append(t)
            # the rise above the floor the attack started from, normalized
            # to 0..1 against the loudest attack so far — the offline accents
            # are normalized against the take's own maximum the same way, so
            # the meter's accent contrast means the same thing live as
            # offline. (A saturating rms/threshold map would flatten every
            # attack to 1.0: the threshold is a floor, not a reference.)
            ratio = rms / max(self.floor_rms_seen, 1e-9)
            self.peak_ratio = max(self.peak_ratio, ratio)
            self.strength = min(1.0, ratio / max(self.peak_ratio, 1e-9))
            # re-arm the flatness reference at the burst's own value: the
            # SAME consonant cannot fire twice while its flatness decays back
            # to the vowel's (~2 blocks, alpha above)
            self._hf_floor = hf
            if loud:
                # the level has moved: re-arm from there, otherwise the
                # baseline would keep climbing towards a sustained note and
                # fire again on every block until it got there. An HF-driven
                # attack must NOT re-arm the level: the burst REPLACES the
                # vowel's energy for one block (a dip), and re-arming the
                # references there makes the vowel's recovery a second
                # "rise" — measured, the same ta-ta fired twice (0.51 s and
                # 0.60 s) with the second block at 1.7x the dipped floor.
                self.baseline_rms = rms
                self.floor_rms_seen = rms
        elif rms < self.baseline_rms:
            # the level fell (a gap, a breath between two phrases): follow it
            # DOWN faster than it climbs, so the next note — even at the same
            # strength — rises clearly above the baseline again
            self.baseline_rms += self.alpha_down * (rms - self.baseline_rms)
            self.floor_rms_seen = min(self.floor_rms_seen, self.baseline_rms)
        else:
            self.baseline_rms += self.alpha * (rms - self.baseline_rms)
        return attack

    def reset(self) -> None:
        self.baseline_rms = 0.0
        self.floor_rms_seen = 0.0
        self.peak_ratio = 0.0
        self.strength = 0.0
        self._hf_floor = self.hf_ratio
        self._peak_rms = 0.0
        self.recent.clear()
        self._last_t = -1e9
        self._primed = False
