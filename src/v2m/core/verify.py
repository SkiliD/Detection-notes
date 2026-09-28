"""Verify one exported take: do the WAV and the JSON describe the SAME signal?

The 2026-09-26 user report ("the audio doesn't match what I sang") came
from a GUI analysis hand-off that wrote the PREVIOUS take's WAV under the
new take's name: one folder's JSON and WAV told two different takes. This
module is the control the user (or a script) can run on any export: a take
whose WAV duration and JSON duration diverge is a WRONG take, whatever the
analysis says.
"""

from __future__ import annotations

import json
from pathlib import Path

from .i18n import tr

# i18n keys (user-facing output): the table lives in i18n.py like every
# visible string — the AST guard (test_docs_guard) refuses them anywhere else.
VERIFY_OK = "verify ok"
VERIFY_BAD = "verify bad"
VERIFY_WAV = "verify wav"
VERIFY_JSON = "verify json"
VERIFY_GAP = "verify gap"
VERIFY_NONE = "verify none"


def verify_take(stem: str | Path) -> tuple[bool, list[str]]:
    """Compare what a take's WAV and its JSON each tell.

    `stem` is the artifacts' common prefix (`prises_live/<take>/<take>`);
    missing files are simply skipped (a take exported without audio still
    verifies its JSON). Returns (ok, readable lines): the lines say what
    was compared and, when they diverge, the measured gap.
    """
    stem = Path(stem)
    lines: list[str] = []
    ok = True
    wav = Path(str(stem) + ".wav")
    rep = Path(str(stem) + ".json")
    wav_dur: float | None = None
    if wav.exists():
        import soundfile as sf

        info = sf.info(str(wav))
        wav_dur = info.frames / info.samplerate
        lines.append(
            tr(VERIFY_WAV).format(name=wav.name, dur=wav_dur, rate=info.samplerate)
        )
    if rep.exists():
        data = json.loads(rep.read_text(encoding="utf-8"))
        json_dur = float(data.get("duration_s") or 0.0)
        lines.append(
            tr(VERIFY_JSON).format(
                name=rep.name, dur=json_dur, n=data.get("n_notes")
            )
        )
        if wav_dur is not None and abs(wav_dur - json_dur) > 0.1:
            ok = False
            lines.append(
                tr(VERIFY_GAP).format(dur_wav=wav_dur, dur_json=json_dur)
            )
    if not lines:
        return False, [tr(VERIFY_NONE).format(stem=stem)]
    lines.insert(0, tr(VERIFY_OK) if ok else tr(VERIFY_BAD))
    return ok, lines


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    import sys

    for arg in sys.argv[1:]:
        good, out = verify_take(arg)
        print("\n".join(out))
        print()
        sys.exit(0 if good else 1)
