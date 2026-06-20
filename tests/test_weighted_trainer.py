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

    Trains the tiny 2-label fixture a few steps with the weighted trainer on a heavily
    imbalanced toy set (majority class 0, rare class 1) and asserts the minority class is
    recalled at all (recall > 0) — i.e. the inverse-frequency weights prevent the class
    collapse a bare loss would suffer (Pitfall 2 / D-14). Greened by 03-03.
    """
    import tempfile

    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
    )

    train = pytest.importorskip("src.models.transformer_train")
    assert hasattr(train, "WeightedTrainer")

    tokenizer = AutoTokenizer.from_pretrained(tiny_seqcls_model)
    model = AutoModelForSequenceClassification.from_pretrained(tiny_seqcls_model)

    # Heavily imbalanced toy set: many "majority" rows, a few "minority" rows, with a
    # cleanly separable single-token signal per class so the tiny model can learn the
    # boundary at all (the point is no minority collapse, not full convergence).
    majority_texts = ["tok10 tok10 tok10"] * 12
    minority_texts = ["tok40 tok40 tok40"] * 3
    texts = majority_texts + minority_texts
    labels = [0] * len(majority_texts) + [1] * len(minority_texts)

    enc = train.build_tokenized(texts, tokenizer)
    ds = train._TokenizedDataset(enc, labels)
    cw = train.class_weights(labels)

    out_dir = tempfile.mkdtemp(prefix="wt_test_")
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=40,
        learning_rate=1e-2,
        per_device_train_batch_size=4,
        save_strategy="no",
        report_to=[],
        seed=42,
    )
    trainer = train.WeightedTrainer(
        model=model, args=args, train_dataset=ds, class_weights=cw
    )
    trainer.train()

    pred = trainer.predict(ds)
    yhat = np.asarray(pred.predictions).argmax(axis=-1)
    y = np.asarray(labels)

    minority_mask = y == 1
    minority_recall = float((yhat[minority_mask] == 1).mean())
    assert minority_recall > 0.0, (
        "class-weighted loss must recall the minority class at all (no collapse, D-14)"
    )
