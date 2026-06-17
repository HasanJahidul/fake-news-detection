"""DATA-03 slow gate — implemented in plan 01-07 (PHASE GATE).

The source-stripped leakage probe: a cheap TF-IDF+LogReg trained on the
source-stripped corpus must score meaningfully below a full-content model and
below ~0.95 macro-F1, proving no source artifact leaks the label.
Marked slow so per-task fast runs (`pytest -m "not slow"`) skip it.
"""

import pytest


@pytest.mark.slow
@pytest.mark.skip(reason="implemented in plan 01-07 (leakage_probe.py — PHASE GATE)")
def test_source_stripped_near_chance():
    """Source-stripped probe macro-F1 < full-content model AND < 0.95."""
    raise NotImplementedError("01-07: assert source_stripped.macro_f1 < full.macro_f1 and < 0.95")
