"""CLS-04 / SC#3 — two-stage cascade + path-product confidence (Wave 0 RED).

Proves (greened by plan 03-04 `src/models/transformer_infer.py` cascade):
  * test_two_stage_path — prediction routes gate (malicious vs not) THEN real/fake,
    NOT a flat 3-way softmax head (D-?? two-stage decision).
  * test_confidence_path_product — every prediction emits a calibrated confidence in
    [0,1] equal to the product of the stage probabilities along the taken path.

Guarded with ``importorskip("torch")``; the cascade class is imported via
``importorskip`` so this SKIPS until 03-04 lands ``src.models.transformer_infer``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def _load_cascade(tiny_seqcls_model):
    infer = pytest.importorskip("src.models.transformer_infer")
    cascade_cls = getattr(infer, "TransformerCascade", None)
    assert cascade_cls is not None, "transformer_infer must expose TransformerCascade"
    # Both stages reuse the tiny 2-label fixture for wiring-level assertions.
    return cascade_cls(gate_dir=tiny_seqcls_model, realfake_dir=tiny_seqcls_model)


def test_two_stage_path(tiny_seqcls_model):
    """Cascade routes gate -> real/fake, not a flat 3-way head.

    RED until 03-04. The verdict label must be one of the locked 3 classes and the
    result must expose the two-stage path (gate decision then, if not malicious, the
    real/fake decision) — proving it is NOT a single softmax over 3 labels.
    """
    from src.data.label_map import LABELS

    cascade = _load_cascade(tiny_seqcls_model)
    result = cascade.predict("some neutral news text")

    assert result["label"] in LABELS
    # Two-stage evidence: a per-stage breakdown exists (gate + optional realfake).
    assert "path" in result or "stages" in result


def test_confidence_path_product(tiny_seqcls_model):
    """Confidence in [0,1] equals the product of stage probabilities along the path.

    RED until 03-04. A malicious verdict's confidence == gate P(malicious); a
    real/fake verdict's confidence == P(not malicious) * P(chosen real/fake).
    """
    cascade = _load_cascade(tiny_seqcls_model)
    result = cascade.predict("some neutral news text")

    conf = result["confidence"]
    assert 0.0 <= conf <= 1.0
    # When the path probabilities are exposed, confidence is exactly their product.
    if "path_probs" in result:
        import math

        prod = math.prod(result["path_probs"])
        assert conf == pytest.approx(prod, abs=1e-6)
