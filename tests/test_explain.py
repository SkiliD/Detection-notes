"""The live/final pedagogical explanation: numbers -> honest sentences.

`core.explain` turns the `live_vs_final` block (the breakdown of
`core.compare`) into a few lines the singer reads on the results panel.
Three rules pinned here:

1. **silence on absence**: no block, an empty block, or no live notes
   -> no line (a file analysis has nothing to explain);
2. **one sentence per present cause, never an absent one**: each
   sentence is gated by its count — the module invents nothing;
3. **the tone is not a judgment**: the ±1 family is named a "rounding"
   (offline absorbs the tuning offset BEFORE rounding, while live rounds
   its raw median), not an error.
"""

from __future__ import annotations

from v2m.core.explain import explain


def _block(**overrides) -> dict:
    base = {
        "live": 37,
        "final": 31,
        "contours": 18,
        "off1": 11,
        "moved": 6,
        "absorbed": 1,
        "rate": 0.486,
    }
    base.update(overrides)
    return base


def test_an_empty_or_missing_block_explains_nothing():
    assert explain({}) == []
    assert explain({"live": 0, "final": 0}) == []


def test_empty_final_is_explicitly_not_comparable():
    lines = explain(_block(final=0, contours=0, off1=0, moved=0, absorbed=0, rate=0.0))
    assert any("non comparable" in line for line in lines)
    assert all("%" not in line for line in lines)


def test_empty_live_is_not_an_agreement_verdict():
    lines = explain(_block(live=0, final=3, rate=0.0))
    assert any("non comparable" in line for line in lines)
    assert all("%" not in line for line in lines)


def test_one_sentence_per_present_cause_only():
    lines = explain(_block())
    assert len(lines) == 5  # head + off1 + moved + absorbed + more
    assert "37" in lines[0] and "31" in lines[0]
    assert "11" in lines[1]
    assert "6" in lines[2]
    assert "1" in lines[3]
    # the more line carries the rate in percent
    assert "49" in lines[4]  # 48.6 -> 49


def test_an_absent_cause_never_gets_a_sentence():
    lines = explain(_block(off1=0, moved=0, absorbed=0, contours=0, rate=0.0))
    # head alone: the gap decomposes to nothing, the explainer stays honest
    assert len(lines) == 1


def test_the_off1_family_reads_as_a_rounding_not_a_mistake():
    line = explain(_block(moved=0, absorbed=0))[1]
    # the sentence names the recadre/rounding nature (fr or en per language)
    assert ("arrondi" in line) or ("rounding" in line)
