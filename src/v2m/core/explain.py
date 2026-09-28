"""Why the final score differs from what the singer watched: ONE explainer.

The live-vs-final comparison produces a decomposition (the `live_vs_final`
JSON block, from `core.compare.breakdown`); this module turns the NUMBERS
into a short, honest explanation the singer can read on the results panel.
It never invents: every sentence is gated by its number, and an empty or
absent block reads "nothing to explain".
"""

from __future__ import annotations

from .i18n import tr

# i18n keys (user-facing output): the table lives in i18n.py like every
# visible string — the AST guard (test_docs_guard) refuses them anywhere else.
EXPLAIN_NONE = "explain none"
EXPLAIN_HEAD = "explain head"
EXPLAIN_OFF1 = "explain off1"
EXPLAIN_MOVED = "explain moved"
EXPLAIN_ABSORBED = "explain absorbed"
EXPLAIN_MORE = "explain more"


def explain(block: dict) -> list[str]:
    """The explanation lines for one `live_vs_final` block (empty = silence).

    One sentence per cause, only when the cause exists in the block, each
    ending with its number — the singer reads WHY the final picture moved,
    not a judgment: the ±1 family is a rounding, not a mistake.
    """
    if not block or not block.get("live"):
        return []
    lines = [
        tr(EXPLAIN_HEAD).format(
            live=block.get("live", 0), final=block.get("final", 0)
        )
    ]
    off1 = int(block.get("off1") or 0)
    moved = int(block.get("moved") or 0)
    absorbed = int(block.get("absorbed") or 0)
    if off1:
        lines.append(tr(EXPLAIN_OFF1).format(n=off1))
    if moved:
        lines.append(tr(EXPLAIN_MOVED).format(n=moved))
    if absorbed:
        lines.append(tr(EXPLAIN_ABSORBED).format(n=absorbed))
    contours = int(block.get("contours") or 0)
    if len(lines) > 2 and contours and block.get("rate") is not None:
        lines.append(tr(EXPLAIN_MORE).format(n=contours, rate=block["rate"] * 100))
    return lines


def write_breakdown_file(stem, block: dict) -> str | None:
    """Archive the take's live/final breakdown as `<stem>.live_vs_final.json`.

    Written at export time (the caller passes the block from the report);
    the replay script and the multi-week trend can then read the gap back
    per take without re-running anything. The PATH is the artifact
    contract's (`takes.stem_path` + `SIDECAR_SUFFIX`): one spelling, two
    writers must never disagree on the name. Best-effort: an unwritable
    folder returns None, the artifacts are already on disk either way.
    """
    import json

    from .takes import SIDECAR_SUFFIX, stem_path

    if not block:
        return None
    try:
        path = stem_path(stem, SIDECAR_SUFFIX)
        path.write_text(
            json.dumps(block, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)
    except OSError:
        return None
