"""D-15 / SC-3 — leakage re-check on transformer test predictions (Wave 0 RED).

Proves (greened by plan 03-04 `src/models/select_transformer.py`):
  the SC-3 leakage smell carries over to the transformer via the SCORE threshold only
  (>= 0.98 macro-F1 ⇒ investigate). Transformers expose no per-token feature weights,
  so the token-tell half of ``src.models.leakage_recheck`` has NO analog here (D-15:
  score-smell only). Reuses ``leakage_recheck.INVESTIGATE_THRESHOLD`` as the constant.

Guarded with ``importorskip("torch")``; the selector module is imported via
``importorskip`` so the transformer-specific path SKIPS until 03-04. The threshold
constant reuse is asserted against the already-present ``leakage_recheck``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_investigate_threshold_constant_reused():
    """The transformer re-check reuses leakage_recheck.INVESTIGATE_THRESHOLD (0.98).

    Runs against the existing ``src.models.leakage_recheck`` (single source of truth);
    asserts the SC-3 score smell constant is 0.98 so the transformer path can reuse it.
    """
    lr = pytest.importorskip("src.models.leakage_recheck")
    assert hasattr(lr, "INVESTIGATE_THRESHOLD")
    assert lr.INVESTIGATE_THRESHOLD == pytest.approx(0.98)


def test_transformer_score_smell_flags_investigate():
    """>=0.98 macro-F1 on transformer test preds ⇒ investigate (score-smell only).

    RED until 03-04. The selector exposes a score-only leakage check (no token-tell
    extraction for transformers, D-15) that flags investigate when the test macro-F1
    meets/exceeds 0.98.
    """
    sel = pytest.importorskip("src.models.select_transformer")
    fn = getattr(sel, "transformer_leakage_smell", None) or getattr(
        sel, "leakage_score_smell", None
    )
    assert fn is not None, "select_transformer must expose a score-only leakage smell check"

    # Contract: a >=0.98 score flags investigate; a clean score does not.
    assert fn(0.99)["investigate"] is True
    assert fn(0.90)["investigate"] is False
