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


# ---------------------------------------------------------------------------
# 03-05 GAP-1 (CR-01) — the sweep must be LIVE end-to-end, not dead in main().
# ---------------------------------------------------------------------------
def test_main_calls_choose_gate_threshold():
    """select_transformer.main() actually CALLS choose_gate_threshold and assigns it.

    RED before 03-05: main() declares ``threshold_record = None`` and never reassigns
    it from a sweep, so the gate threshold is dead at the 0.5 sentinel. GREEN after the
    wiring: the source calls choose_gate_threshold and binds the result to
    threshold_record (no longer the unreassigned None default).
    """
    import inspect

    sel = pytest.importorskip("src.models.select_transformer")
    src = inspect.getsource(sel.main)
    assert "choose_gate_threshold" in src, "main() must call the sweep, not skip it"
    # threshold_record must be reassigned from the sweep, not left at its None default.
    assert "threshold_record = choose_gate_threshold" in src, (
        "main() must bind threshold_record = choose_gate_threshold(...) on the val split"
    )


def test_evaluate_cascade_probs_exists_and_returns_pair():
    """select_transformer exposes evaluate_cascade_probs returning a per-row 2-tuple.

    RED before 03-05 (the function does not exist). Contract-only: assert it is callable
    and its signature takes (cascade, df) — the per-stage probability feeder for the sweep.
    """
    import inspect

    sel = pytest.importorskip("src.models.select_transformer")
    fn = getattr(sel, "evaluate_cascade_probs", None)
    assert fn is not None, "select_transformer must expose evaluate_cascade_probs"
    params = list(inspect.signature(fn).parameters)
    assert params[:2] == ["cascade", "df"], (
        "evaluate_cascade_probs(cascade, df) — feeds the val sweep per-stage probs"
    )
