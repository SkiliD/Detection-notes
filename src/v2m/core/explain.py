"""Why the final score differs from what the singer watched: ONE explainer.

The live-vs-final comparison produces a decomposition (the `live_vs_final`
JSON block — built AND parsed by `core.compare`, the format's owner); this
turns the parsed NUMBERS into a short, honest explanation the singer can
read on the results panel. It never invents: every sentence is gated by
its number, and an absent or empty block reads "nothing to explain".
"""

from __future__ import annotations

from .compare import read_block
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
    parsed = read_block(block)
    if parsed is None or (not parsed.live and not parsed.final):
        return []
    lines = [tr(EXPLAIN_HEAD).format(live=parsed.live, final=parsed.final or 0)]
    if parsed.final == 0:
        lines.append(tr("explain empty final"))
        return lines
    if parsed.live == 0:
        lines.append(tr("explain empty live"))
        return lines
    off1 = parsed.off1 or 0
    moved = parsed.moved or 0
    absorbed = parsed.absorbed or 0
    if off1:
        lines.append(tr(EXPLAIN_OFF1).format(n=off1))
    if moved:
        lines.append(tr(EXPLAIN_MOVED).format(n=moved))
    if absorbed:
        lines.append(tr(EXPLAIN_ABSORBED).format(n=absorbed))
    contours = parsed.contours or 0
    rate = parsed.rate
    if len(lines) > 2 and contours and rate is not None:
        lines.append(tr(EXPLAIN_MORE).format(n=contours, rate=rate * 100))
    return lines
