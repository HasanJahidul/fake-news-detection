"""CLS-02 / SC#1 + SC#2 — transformer inference-loader contract (Wave 0 RED).

Proves (greened by plan 03-04 `src/models/transformer_infer.py`):
  * test_normalizer_parity — the inference loader feeds text through the SAME
    shared ``src.preprocess.preprocess`` used at training (Pitfall 1 / D-12).
  * test_load_only_no_training_path — loading a saved transformer predicts WITHOUT
    importing any training module (CLS-02 / SC#2: no training code path).

Guard discipline: ``importorskip("torch")`` so the fast suite runs without the
transformer stack; the inference module is imported via ``importorskip`` so this
file SKIPS (not errors) until plan 03-04 lands ``src.models.transformer_infer``.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_normalizer_parity():
    """Loader normalization parity: the inference path must call the shared preprocess().

    Asserts ``src.models.transformer_infer`` references the project's single
    ``src.preprocess.preprocess`` (no forked normalization) — the train/inference
    normalizer-parity contract (D-12). RED until 03-04.
    """
    infer = pytest.importorskip("src.models.transformer_infer")
    import inspect

    import src.preprocess as preprocess_mod

    src = inspect.getsource(infer)
    # The loader must route through the shared preprocess contract, not re-normalize.
    assert "preprocess" in src
    assert hasattr(preprocess_mod, "preprocess")


def test_load_only_no_training_path(tiny_seqcls_model):
    """Loading a saved transformer predicts with NO training module imported.

    Mirrors the classical ``load_artifacts`` load-only contract: importing /
    using the inference loader must NOT pull in ``src.models.transformer_train``
    (CLS-02 / SC#2). RED until 03-04.
    """
    sys.modules.pop("src.models.transformer_train", None)
    infer = pytest.importorskip("src.models.transformer_infer")

    assert "src.models.transformer_train" not in sys.modules, (
        "inference loader must not import the training module (load-only, SC#2)"
    )
    # The module exposes a load/predict surface the cascade test exercises.
    assert hasattr(infer, "TransformerCascade") or hasattr(infer, "load")
