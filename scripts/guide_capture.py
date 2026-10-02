"""Guided capture of a truth take, note by note.

The `prises_verite/` folder collects sung takes whose every note was
checked BY EAR — the ground truth missing for real voices. This script
builds that corpus:

    python scripts/guide_capture.py ma_prise.wav
      1) captures the microphone continuously (Ctrl+C stops and annotates),
      2) per note: Enter -> sing the note -> Enter,
      3) at the stop, writes `prises_verite/<name>.wav` + `<name>.truth.json`
         with the bounds REFINED by the pitch curve (the human marker
         dates to +-0.3 s; the curve decides at 40 ms),
      4) `--audit` validates the truth and replays the analysis on the take
         (human vs machine comparison, note by note).

The JSON follows the format documented in `prises_verite/README.md`:
midi + start + end per note (+ optional lyric), `checked_by: humain`
(a data value, spelled as the corpus requires).

Without a microphone (CI), only `--audit` works: it captures nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from v2m.core import audio_io  # noqa: E402
from v2m.core.music import hz_to_midi, midi_to_name  # noqa: E402

TRUTH_DIR = ROOT / "prises_verite"
FRAME_S = 0.04      # pitch-curve step (40 ms)
WINDOW_S = 0.08     # YIN analysis window (80 ms)
STABLE_ST = 0.35    # max spread of 3 neighbouring frames for "pitch stable"
REFINE_MARGIN_S = 0.30  # the human marker may be this late/early


# --------------------------------------------------------------------------- #
# Validation / audit
# --------------------------------------------------------------------------- #


def validate_truth(data: dict, audio_name: str) -> list[str]:
    """Blocking errors of a truth file (empty list = valid)."""
    errs = []
    if data.get("audio") != audio_name:
        errs.append(f"audio: {data.get('audio')!r} != {audio_name!r}")
    if data.get("checked_by") != "humain":
        errs.append("checked_by doit valoir 'humain'")  # data value, kept as-is
    notes = data.get("notes")
    if not isinstance(notes, list) or not notes:
        errs.append("notes: liste non vide attendue")
        return errs
    prev_end = -1.0
    for i, n in enumerate(notes):
        missing = [k for k in ("midi", "start", "end") if k not in n]
        if missing:
            errs.append(f"note {i}: champ(s) manquant(s) {missing}")
            continue
        m, s, e = n["midi"], n["start"], n["end"]
        if not (isinstance(m, int) and 21 <= m <= 108):
            errs.append(f"note {i}: midi {m!r} hors piano (21..108)")
        if not (isinstance(s, (int, float)) and isinstance(e, (int, float))):
            errs.append(f"note {i}: start/end doivent être numériques")
            continue
        if e <= s:
            errs.append(f"note {i}: end <= start ({e} <= {s})")
        if s < prev_end:
            errs.append(f"note {i}: start {s} chevauche la note précédente (fin {prev_end})")
        prev_end = max(prev_end, e)
    return errs


def load_truth(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    errs = validate_truth(data, path.stem.replace(".truth", ""))
    if errs:
        raise SystemExit("truth invalide :\n  " + "\n  ".join(errs) + f"\n({path})")
    return data


def _pitch_of(block: np.ndarray, sr: int) -> float:
    """MIDI pitch of a block, NaN under the silence gate or when unvoiced."""
    from v2m.core.pitch import yin_pitch

    if float(np.sqrt(np.mean(block ** 2))) < audio_io.GATE_RMS:
        return float("nan")
    hz = yin_pitch(block.astype(np.float64), sr)
    if hz != hz or hz <= 0:
        return float("nan")
    return float(hz_to_midi(hz))


def _stable_series(y: np.ndarray, sr: int, lo: float, hi: float) -> list[tuple[float, bool]]:
    """[(t, stable)]: is the pitch stable around time t?

    A frame is stable when 3 neighbouring YIN readings (40 ms step, 80 ms
    window) fit within STABLE_ST semitones — the definition of a HELD
    note, what separates singing from breath and from portamento.
    """
    seg = y[int(lo * sr):int(hi * sr)]
    win = int(sr * WINDOW_S)
    hop = int(sr * FRAME_S)
    pitches: list[float] = []
    times: list[float] = []
    for i in range(0, max(0, len(seg) - win), hop):
        pitches.append(_pitch_of(seg[i:i + win], sr))
        times.append(lo + (i + win / 2) / sr)
    out = []
    for i, (t, p) in enumerate(zip(times, pitches)):
        neigh = pitches[max(0, i - 1):i + 2]
        voiced = [q for q in neigh if q == q]
        spread = float(np.ptp(voiced)) if len(voiced) == len(neigh) and neigh else 1e9
        out.append((t, p == p and spread < STABLE_ST))
    return out


def refine_bounds(y: np.ndarray, sr: int, start: float, end: float) -> tuple[float, float]:
    """Refine the human marker's (start, end) on the pitch curve.

    The human marker (Enter -> sing -> Enter) dates to +-0.3 s: we look
    for the RUNS of stable pitch inside the margin and keep the one
    containing the marker's centre (else the nearest one). Taking the
    first/last stable frame of the margin instead caught the NEIGHBOUR
    note (measured: a late marker on note 1 refined INTO note 2). With no
    stable run (breath, silence) or none close enough, the human bound is
    kept — the helper must never degrade a marker.
    """
    lo = max(0.0, start - REFINE_MARGIN_S)
    hi = end + REFINE_MARGIN_S
    runs: list[tuple[float, float]] = []
    cur: list[float] = []
    for t, stable in _stable_series(y, sr, lo, hi):
        if stable:
            cur.append(t)
        elif cur:
            runs.append((cur[0], cur[-1]))
            cur = []
    if cur:
        runs.append((cur[0], cur[-1]))
    if not runs:
        return start, end
    center = (start + end) / 2.0

    def distance(run: tuple[float, float]) -> float:
        a, b = run
        if a <= center <= b:
            return 0.0
        return min(abs(center - a), abs(center - b))

    best = min(runs, key=distance)
    if distance(best) > (end - start) / 2.0 + REFINE_MARGIN_S:
        return start, end  # no run matches this marker
    a, b = best
    if b - a < 0.10:  # a "note" shorter than this makes no sense
        return start, end
    return a, b


# --------------------------------------------------------------------------- #
# Guided capture
# --------------------------------------------------------------------------- #


def capture_and_annotate(out_base: Path) -> None:
    from v2m.core.audio_io import LiveCapture

    chunks: list[np.ndarray] = []
    cap = LiveCapture(on_block=lambda b, t: chunks.append(b))

    print("Capture continue — chantez librement, puis Ctrl+C pour annoter.")
    cap.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
    print(f"\nCapture arrêtée ({cap.duration:.1f} s).")

    notes: list[dict] = []
    print("Pour chaque note : Entree -> chantez -> Entree. (s) pour terminer.")
    while True:
        cmd = input(f"note {len(notes) + 1} [Entree/(s)]: ").strip().lower()
        if cmd == "s":
            break
        t0 = time.time()
        input("   chantez maintenant... Entree pour arreter: ")
        t1 = time.time()
        lyric = input("   syllabe (optionnel): ").strip() or None
        notes.append({"start": round(t0, 3), "end": round(t1, 3), "lyric": lyric})
        print(f"   repere {t0:.2f} -> {t1:.2f}")

    if not notes:
        print("Aucune note annotée — rien à écrire.")
        return

    y = cap.signal()
    sr = audio_io.SR
    capture_end = len(y) / sr
    print("\nRaffinement des bornes sur la courbe de hauteur...")
    refined = []
    for n in notes:
        s = min(n["start"], capture_end)
        e = min(n["end"], capture_end)
        s2, e2 = refine_bounds(y, sr, s, e)
        m = _dominant_midi(y, sr, s2, e2)
        lyric = n.get("lyric")
        refined.append({"midi": m, "start": round(s2, 2), "end": round(e2, 2),
                        **({"lyric": lyric} if lyric else {})})
        print(f"   {midi_to_name(m)}  repere {s:.2f}-{e:.2f}  ->  {s2:.2f}-{e2:.2f}")

    out_wav = out_base.with_suffix(".wav")
    _write_wav(out_wav, y, sr)
    truth = {
        "name": out_base.stem,
        "audio": out_wav.name,
        "recorded": time.strftime("%Y-%m-%d"),
        "checked_by": "humain",
        "notes": refined,
    }
    out_json = out_base.parent / f"{out_base.stem}.truth.json"
    out_json.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nÉcrit : {out_wav}")
    print(f"Écrit : {out_json}")
    print("Réécoutez la prise et corrigez le JSON à la main si besoin "
          "(les bornes raffinées sont une aide, pas une décision).")


def _dominant_midi(y: np.ndarray, sr: int, start: float, end: float) -> int:
    """The note's dominant MIDI pitch (median of the voiced frames)."""
    seg = y[int(start * sr):int(end * sr)]
    hop = int(sr * FRAME_S)
    n = max(1, len(seg) // hop)
    pitches = [_pitch_of(seg[i * hop:(i + 1) * hop], sr) for i in range(n)]
    pitches = [p for p in pitches if p == p]
    if not pitches:
        return 60
    return int(round(float(np.median(pitches))))


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    pcm = np.clip(y, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- #
# Audit: the machine re-reads what the human checked
# --------------------------------------------------------------------------- #


def audit(audio_path: Path) -> None:
    """Validate the truth, replay the analysis, compare note by note."""
    from v2m.core.pipeline import analyze_signal
    from voicetruth import SungNote, measure  # the tested harness, as in score_accuracy

    truth_path = audio_path.with_name(audio_path.stem + ".truth.json")
    if not audio_path.exists():
        raise SystemExit(f"prise introuvable : {audio_path}")
    if not truth_path.exists():
        raise SystemExit(f"vérité introuvable : {truth_path}")
    data = load_truth(truth_path)
    y = audio_io.load_file(str(audio_path))
    res = analyze_signal(y, audio_io.SR, source=str(audio_path))

    truth = [SungNote(n["midi"], n["start"], n["end"] - n["start"])
             for n in data["notes"]]
    acc = measure(res.notes, truth, tol_s=0.15)
    print(f"=== {audio_path.name} — {len(truth)} notes vérifiées à l'oreille")
    print(f"    pipeline : {len(res.notes)} notes, tempo "
          f"{res.tempo.bpm if res.tempo else '?'} BPM")
    print(f"    fidélité : {acc.line()}")
    print(f"    {'vérité':>22} | {'détecté':>22}")
    for n, d in zip(data["notes"], res.notes):
        ly = f" ({n['lyric']})" if n.get("lyric") else ""
        print(f"    {midi_to_name(n['midi']):>7}{ly:>8} "
              f"{n['start']:6.2f}-{n['end']:6.2f} | "
              f"{midi_to_name(d.midi):>7} {d.start:6.2f}-{d.end:6.2f}")
    if acc.exact:
        print("    → la prise peut rejoindre le corpus de vérité.")
    else:
        print("    → divergence : corrigez le JSON (si l'oreille tranche) "
              "ou ouvrez un défaut de segmentation.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="<nom>.wav : sortie de capture, ou prise existante pour --audit")
    parser.add_argument("--audit", action="store_true",
                        help="valide la vérité et rejoue l'analyse sur la prise")
    args = parser.parse_args()

    audio = Path(args.audio)
    if args.audit:
        audit(audio if audio.exists() else TRUTH_DIR / audio.name)
    else:
        capture_and_annotate(TRUTH_DIR / audio.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
