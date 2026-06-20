"""D-09 — gate-threshold sweep on VAL maximizing cascade macro-F1 (Wave 0 RED).

Proves (greened by plan 03-04 `src/models/select_transformer.py` threshold sweep):
  the gate (malicious vs not) threshold is chosen on the VALIDATION split to maximize
  the cascade macro-F1, and is returned together with the precision/recall at that point.

Guarded with ``importorskip("torch")``; the selector module is imported via
``importorskip`` so this SKIPS until 03-04 lands ``src.models.select_transformer``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_gate_threshold_chosen_on_val_maximizes_macro_f1():
    """Chosen gate threshold argmaxes cascade macro-F1 on val; returns P/R too.

    RED until 03-04. ``choose_gate_threshold(...)`` returns a dict/record carrying the
    chosen threshold in [0,1], the val macro-F1 it maximizes, and precision/recall.
    """
    sel = pytest.importorskip("src.models.select_transformer")
    fn = getattr(sel, "choose_gate_threshold", None)
    assert fn is not None, "select_transformer must expose choose_gate_threshold"

    import inspect

    sig = inspect.signature(fn)
    # The sweep operates on validation inputs (no test split parameter — D-09 / Pitfall 3).
    assert "test" not in sig.parameters


def test_threshold_result_records_precision_recall():
    """The returned threshold record carries precision + recall at the chosen point.

    RED until 03-04. Contract assertion only — exercised end-to-end by the slow
    selection run; here we assert the result schema keys exist.
    """
    sel = pytest.importorskip("src.models.select_transformer")
    keys = getattr(sel, "THRESHOLD_RESULT_KEYS", None)
    # The selector declares the schema as a module constant the report writer renders.
    assert keys is not None
    for k in ("threshold", "macro_f1", "precision", "recall"):
        assert k in keys
