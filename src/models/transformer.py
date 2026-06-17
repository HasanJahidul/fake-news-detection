"""Transformer fine-tuning (XLM-RoBERTa primary; BanglaBERT optional) + inference.

Training is GPU-heavy and intended to run in notebooks/02_train_xlmr_colab.ipynb,
but the same `train_transformer()` entrypoint works anywhere a GPU/CPU torch is
installed. Inference is wrapped by `TransformerClassifier` and loaded lazily by
the fusion pipeline; if no fine-tuned model exists, the pipeline degrades to the
classical model only.

Run (Colab):  from src.models.transformer import train_transformer; train_transformer()
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import ID2LABEL, LABEL2ID, LABELS, load_config, path as cfg_path

PROC = cfg_path("processed_dir")
MODELS = cfg_path("models_dir")
REPORTS = cfg_path("reports_dir")
DEFAULT_DIR = MODELS / "xlmr"


def _compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def train_transformer(model_name: Optional[str] = None, out_dir: Optional[Path] = None,
                      tag: str = "xlmr") -> str:
    """Fine-tune a transformer on the 3-class corpus. Returns the saved model dir."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    cfg = load_config()["transformer"]
    model_name = model_name or cfg["model_name"]
    out_dir = Path(out_dir) if out_dir else (MODELS / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(PROC / "train.parquet")
    val_df = pd.read_parquet(PROC / "val.parquet")
    test_df = pd.read_parquet(PROC / "test.parquet")
    for d in (train_df, val_df, test_df):
        d["labels"] = d["label"].map(LABEL2ID)

    tok = AutoTokenizer.from_pretrained(model_name)

    def _tok(batch):
        return tok(batch["text"], truncation=True, max_length=cfg["max_length"])

    ds_train = Dataset.from_pandas(train_df[["text", "labels"]]).map(_tok, batched=True)
    ds_val = Dataset.from_pandas(val_df[["text", "labels"]]).map(_tok, batched=True)
    ds_test = Dataset.from_pandas(test_df[["text", "labels"]]).map(_tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID)

    args = TrainingArguments(
        output_dir=str(out_dir / "_ckpt"),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds_train, eval_dataset=ds_val,
        tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_compute_metrics)
    trainer.train()

    # Final test-set evaluation + confusion matrix for the report.
    from sklearn.metrics import classification_report, confusion_matrix
    pred_logits = trainer.predict(ds_test).predictions
    preds = np.argmax(pred_logits, axis=-1)
    gold = test_df["labels"].to_numpy()
    rep = classification_report([ID2LABEL[i] for i in gold],
                                [ID2LABEL[i] for i in preds],
                                labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(gold, preds, labels=list(range(len(LABELS)))).tolist()

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{tag}_metrics.json").write_text(
        json.dumps({"model_name": model_name, "report": rep, "confusion_matrix": cm},
                   indent=2), encoding="utf-8")
    print(f"Saved fine-tuned model -> {out_dir}")
    print(f"Test macro-F1 = {rep['macro avg']['f1-score']:.4f}  acc = {rep['accuracy']:.4f}")
    return str(out_dir)


# ── Inference wrapper used by the fusion pipeline ───────────────────────────
class TransformerClassifier:
    """Lazy-loads a fine-tuned model dir; returns per-class probabilities."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
        self.dir = Path(model_dir) if model_dir else DEFAULT_DIR
        if not (self.dir / "config.json").exists():
            raise FileNotFoundError(
                f"No fine-tuned transformer at {self.dir}. Train via notebooks/02 first.")
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self._torch = torch

    @staticmethod
    def available(model_dir: Optional[Path] = None) -> bool:
        d = Path(model_dir) if model_dir else DEFAULT_DIR
        return (d / "config.json").exists()

    def predict_proba(self, text: str) -> dict:
        torch = self._torch
        cfg = load_config()["transformer"]
        enc = self.tok(text, truncation=True, max_length=cfg["max_length"],
                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return {ID2LABEL[i]: float(probs[i]) for i in range(len(LABELS))}


if __name__ == "__main__":
    train_transformer()
