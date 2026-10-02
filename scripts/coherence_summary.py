"""Aggregate every `.coherence.csv` of `prises_live/` into a per-session report.

Every live export writes `<take>.coherence.csv` beside the other artifacts
(one row per suspect note, header only when the take is clean — see
`report.export_result`). The aggregation policy lives in `core.coherence`
(sessions split on a 30 min gap, suspect rate per 100 audited notes, delta
against the previous session, per-take WAV/JSON verification via
`verify_take`); this script is only the COMMAND-LINE ADAPTER — the same
numbers as the GUI page, without launching the application. Labels go
through the fr/en i18n tables (`v2m --lang` / ~/.v2m_lang) like every
visible string.

Usage:
    python -X utf8 scripts/coherence_summary.py
    python -X utf8 scripts/coherence_summary.py other_folder

No artifact is written (the core's `coherence_journal.json` journal updates
itself so the trend survives folder cleanup). A folder without any report
is not an error (an empty table and a message say so).
"""
from __future__ import annotations

import sys
from pathlib import Path

from v2m.core.coherence import LIVE_TAKES_DIR, build_snapshot, render
from v2m.core.i18n import get_language, set_language, tr


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    # the script follows the app's language choice, like every adapter
    if not any(a.startswith("--lang") for a in args):
        get_language()
    root = Path(args[0]) if args else LIVE_TAKES_DIR
    if not root.is_dir():
        print(tr("coh missing folder").format(root=root))
        print(tr("coh missing hint"))
        return 2
    snapshot = build_snapshot(root)
    text = render(
        snapshot.takes,
        root=str(root),
        rows=snapshot.sessions,
        days=snapshot.days,
    )
    # write straight to the buffer, encoded here: a disposable
    # TextIOWrapper over sys.stdout.buffer would CLOSE the underlying buffer
    # at its collection (and with it the calling process's capture)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
