"""D-14 — class-weighted compute_loss override + minority recall (Wave 0 RED).

Proves (greened by plan 03-02 `src/models/transformer_train.py`):
  the HF ``Trainer`` subclass applies inverse-frequency class weights in a
  ``compute_loss`` override (transformers 4.46 passes ``num_items_in_batch`` so the
  override MUST accept ``**kw``), and the weighting yields minority recall > 0 on the
  tiny fixture (D-14: class weighting, no resampling).

Guarded with ``importorskip("torch")``; the trainer module is imported via
``importorskip`` so this SKIPS until 03-02 lands ``src.models.transformer_train``.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_weighted_trainer_subclasses_hf_trainer():
    """A WeightedTrainer subclasses HF Trainer and overrides compute_loss.

    RED until 03-02. The override must accept ``**kw`` (transformers 4.46 passes
    ``num_items_in_batch``) — assert the signature is variadic-friendly.
    """
    from transformers import Trainer

    train = pytest.importorskip("src.models.transformer_train")
    wt = getattr(train, "WeightedTrainer", None)
    assert wt is not None, "transformer_train must expose WeightedTrainer"
    assert issubclass(wt, Trainer)

    sig = inspect.signature(wt.compute_loss)
    has_varkw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    assert has_varkw, "compute_loss must accept **kw (transformers 4.46 num_items_in_batch)"


@pytest.mark.slow
def test_class_weighting_minority_recall_positive(tiny_seqcls_model):
    """Class-weighted loss drives minority recall > 0 on the tiny fixture.

    RED + slow until 03-02. Trains the tiny 2-label fixture a few steps with the
    weighted trainer on an imbalanced toy set and asserts the minority class is
    recalled at all (recall > 0).
    """
    train = pytest.importorskip("src.models.transformer_train")
    assert hasattr(train, "WeightedTrainer")
    # End-to-end tiny-fit wiring is implemented in 03-02; placeholder RED assertion.
    pytest.fail("transformer_train.WeightedTrainer tiny-fit not implemented yet (03-02)")
