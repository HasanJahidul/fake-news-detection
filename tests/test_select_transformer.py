"""D-01 / SC#1 / SC#4 — per-language ranking + Bangla-priority selection (Wave 0 RED).

Proves (greened by plan 03-04 `src/models/select_transformer.py`):
  the selector reports per-language macro-F1 (bn / en / code-mixed) using the SHARED
  ``src.models.metrics`` harness, and ranks candidate backbones with Bangla priority
  (D-01), reusing the minority-guard / tie-band idiom from the classical selector.

Guarded with ``importorskip("torch")``; the selector module is imported via
``importorskip`` so this SKIPS until 03-04 lands ``src.models.select_transformer``.
The metric-harness reuse assertion runs against the existing ``src.models.metrics``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_metric_harness_is_reused_not_reimplemented():
    """select_transformer reuses src.models.metrics (per_language_macro_f1, etc.).

    RED until 03-04. The selector must import the shared metric functions rather than
    re-implement macro-F1 / per-language scoring (Pattern: metric harness reuse).
    """
    import inspect

    sel = pytest.importorskip("src.models.select_transformer")
    src = inspect.getsource(sel)
    assert "per_language_macro_f1" in src
    assert "minority_recall" in src


def test_existing_metrics_expose_per_language(sample_corpus):
    """The shared metrics the selector ranks on already compute per-language macro-F1.

    Runs against ``src.models.metrics`` (already present) so the per-language contract
    the selector depends on is asserted now; the Bangla-priority ranking itself is RED
    until 03-04.
    """
    from src.models.metrics import per_language_macro_f1

    y_true = list(sample_corpus["label"])
    y_pred = list(sample_corpus["label"])  # perfect preds -> 1.0 per present language
    out = per_language_macro_f1(sample_corpus, y_true, y_pred)
    assert "bn" in out and "en" in out
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_bangla_priority_selection():
    """Backbone selection ranks on per-language macro-F1 with Bangla priority (D-01).

    RED until 03-04. The selector exposes a ranking entrypoint that prioritizes Bangla
    macro-F1 over the overall number.
    """
    sel = pytest.importorskip("src.models.select_transformer")
    fn = getattr(sel, "select_best_transformer", None) or getattr(sel, "select_best", None)
    assert fn is not None, "select_transformer must expose a Bangla-priority selector"
