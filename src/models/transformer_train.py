"""CLS-02 / CLS-04 — offline transformer fine-tune + cascade export layer.

This is the genuinely-new training code of Phase 3: it composes the 03-02 leaves
(:mod:`src.models.transformer_data` stage views / tokenize / class weights,
:mod:`src.models.calibration` val-only temperature fit) into a trainable + exportable
two-stage cascade. Training is GPU-bound and runs on a Colab free T4 (D-05); only the
wiring is unit-tested here against the tiny GPU-free fixture, so it is provable with no GPU.

Pipeline (per backbone, per stage):

    1. build the stage label view (gate_labels over ALL rows / realfake_frame drop-malicious).
    2. tokenize via the backbone's AutoTokenizer through the SHARED preprocess() (D-12 parity).
    3. compute inverse-frequency class_weights on the stage TRAIN split (D-14, no resampling).
    4. fine-tune with WeightedTrainer (class-weighted compute_loss override — Pattern 3).
    5. fit one temperature scalar on VAL logits per head (Pitfall 3 / T-03-04).
    6. export the locked save_pretrained layout under models/transformer/<backbone>/.

Policy / decision callouts:
  * D-14 — class imbalance is handled by a class-weighted ``compute_loss`` override ONLY
    (HF ``Trainer`` has no ``class_weights`` arg). No SMOTE / oversampling anywhere.
  * D-05 / D-10 — both backbones (banglishbert, xlmr) are fine-tuned end-to-end on Colab,
    exported to Drive, downloaded for LOCAL CPU inference; the loader (03-04) imports NO
    training code.
  * D-04 — BanglaBERT is an optional-stretch third backbone; not in the default set.
  * Pattern 4 / T-03-08 — export anchors output to ``REPO_ROOT / models / transformer /
    <backbone>`` constants; the backbone key comes from the fixed :data:`BACKBONES` dict,
    never user input. ``save_pretrained`` writes safetensors (T-03-07) — never pickle.

``torch`` / ``transformers`` are imported lazily inside the functions that need them so the
module imports without the transformer optional group (mirrors transformer_data/calibration).

CLI: ``python -m src.models.transformer_train`` (real backbones; runs offline on a GPU env).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data.label_map import LABELS  # locked class order — label_map.json is built from this
from src.models.calibration import fit_temperature, save_temperature
from src.models.transformer_data import (
    GATE_CLASSES,
    MAX_LENGTH,
    REALFAKE_CLASSES,
    build_tokenized,
    class_weights,
    gate_labels,
    load_splits,
    realfake_frame,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("transformer_train")

# Repo root = two levels up from src/models/transformer_train.py (mirrors train_classical).
REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

# Backbones fine-tuned end-to-end (D-05/D-10). Keys are the fixed export subdir names; values
# are the CLAUDE.md-named canonical HF repos (T-03-06 — no arbitrary URL loads).
# BanglaBERT ("csebuetnlp/banglabert") is an optional-stretch third backbone (D-04) — add here
# only if the Bangla-heavy path needs it; not in the default both-backbone set.
BACKBONES: dict[str, str] = {
    "banglishbert": "csebuetnlp/banglishbert",
    "xlmr": "xlm-roberta-base",
}

# Fine-tune hyperparameters (Pitfall 4 ranges — recorded for the report; structure is locked).
# Short 2-3 epoch fine-tunes, fp16, batch 8-16, save_strategy=epoch so a lost Colab session
# resumes from the last epoch checkpoint on mounted Drive.
TRAIN_HPARAMS: dict[str, object] = {
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
    "save_strategy": "epoch",
    "max_length": MAX_LENGTH,
    "seed": SEED,
}

# Stage names (the two heads of the cascade, CLS-04).
STAGE_GATE = "gate"
STAGE_REALFAKE = "realfake"


# ---------------------------------------------------------------------------
# Class-weighted Trainer (D-14 / Pattern 3) — HF Trainer has no class_weights arg.
# ---------------------------------------------------------------------------
def _weighted_trainer_cls():
    """Build the ``WeightedTrainer`` subclass lazily (needs torch + transformers present).

    Defined inside a factory so the module imports without the transformer optional group;
    :data:`WeightedTrainer` below is materialized at import time only when the stack exists.
    """
    import torch.nn as nn
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        """``Trainer`` with an inverse-frequency class-weighted cross-entropy loss (D-14).

        HF ``Trainer`` has no ``class_weights`` argument, so the weighting is injected by
        overriding :meth:`compute_loss` with ``nn.CrossEntropyLoss(weight=class_weights)``
        (Pattern 3). The override MUST accept ``**kw`` — transformers 4.46 passes a
        ``num_items_in_batch`` kwarg into ``compute_loss``.
        """

        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = (
                self.class_weights.to(logits.device)
                if self.class_weights is not None
                else None
            )
            loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


try:  # materialize the subclass when the transformer stack is installed (GREEN path)
    WeightedTrainer = _weighted_trainer_cls()
except Exception:  # pragma: no cover - exercised only without torch/transformers
    WeightedTrainer = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Stage label encoding helpers.
# ---------------------------------------------------------------------------
def _stage_view(stage: str, train_df, val_df):
    """Return ``(train_texts, train_y, val_texts, val_y, num_labels)`` for a stage.

    Gate (Stage 1) labels EVERY row (malicious=1 else 0). Real/fake (Stage 2) drops the
    malicious rows and encodes real=0 / fake=1 per :data:`REALFAKE_CLASSES`.
    """
    if stage == STAGE_GATE:
        tr_y = gate_labels(train_df).tolist()
        va_y = gate_labels(val_df).tolist()
        return (
            train_df["text"].tolist(),
            tr_y,
            val_df["text"].tolist(),
            va_y,
            len(GATE_CLASSES),
        )
    if stage == STAGE_REALFAKE:
        tr = realfake_frame(train_df)
        va = realfake_frame(val_df)
        rf_index = {name: i for i, name in enumerate(REALFAKE_CLASSES)}
        tr_y = [rf_index[v] for v in tr["label"]]
        va_y = [rf_index[v] for v in va["label"]]
        return (
            tr["text"].tolist(),
            tr_y,
            va["text"].tolist(),
            va_y,
            len(REALFAKE_CLASSES),
        )
    raise ValueError(f"unknown stage {stage!r}; expected {STAGE_GATE!r} or {STAGE_REALFAKE!r}")


class _TokenizedDataset:
    """Minimal torch ``Dataset`` over a BatchEncoding + integer labels (no extra deps)."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch

        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def train_stage(model_id: str, stage: str, train_df, val_df):
    """Fine-tune one head (``gate`` or ``realfake``) of a backbone with class-weighted loss.

    Builds the stage label view, tokenizes both splits through the SHARED ``preprocess()``
    (D-12 parity) with the backbone's ``AutoTokenizer``, computes inverse-frequency
    ``class_weights`` on the stage TRAIN split (D-14), and runs :data:`WeightedTrainer` with
    the recorded :data:`TRAIN_HPARAMS`. Returns
    ``(model, tokenizer, val_logits, val_labels)`` — the caller fits temperature on the
    returned VAL logits.

    Heavy/GPU path: imports torch + transformers lazily; intended to run on Colab T4.
    """
    import tempfile

    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
    )

    (
        tr_texts,
        tr_y,
        va_texts,
        va_y,
        num_labels,
    ) = _stage_view(stage, train_df, val_df)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tr_enc = build_tokenized(tr_texts, tokenizer)
    va_enc = build_tokenized(va_texts, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

    cw = class_weights(tr_y)

    out_dir = tempfile.mkdtemp(prefix=f"hf_{stage}_")
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=TRAIN_HPARAMS["num_train_epochs"],
        learning_rate=TRAIN_HPARAMS["learning_rate"],
        per_device_train_batch_size=TRAIN_HPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_HPARAMS["per_device_eval_batch_size"],
        warmup_ratio=TRAIN_HPARAMS["warmup_ratio"],
        weight_decay=TRAIN_HPARAMS["weight_decay"],
        fp16=bool(TRAIN_HPARAMS["fp16"]) and torch.cuda.is_available(),
        save_strategy=TRAIN_HPARAMS["save_strategy"],
        seed=SEED,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=_TokenizedDataset(tr_enc, tr_y),
        eval_dataset=_TokenizedDataset(va_enc, va_y),
        class_weights=cw,
    )
    trainer.train()

    pred = trainer.predict(_TokenizedDataset(va_enc, va_y))
    val_logits = torch.as_tensor(np.asarray(pred.predictions)).float()
    val_labels = torch.as_tensor(np.asarray(va_y)).long()
    return model, tokenizer, val_logits, val_labels


# ---------------------------------------------------------------------------
# Export contract (Pattern 4 / CLS-02 / SC#2) — locked save_pretrained layout.
# ---------------------------------------------------------------------------
def _write_label_map(out_dir: Path) -> None:
    """Write ``label_map.json`` built from :data:`LABELS` (never a hardcoded order)."""
    payload = {
        "classes": list(LABELS),
        "gate": list(GATE_CLASSES),
        "realfake": list(REALFAKE_CLASSES),
    }
    (out_dir / "label_map.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_cascade(
    out_dir,
    gate_dir,
    realfake_dir,
    gate_T: float = 1.0,
    realfake_T: float = 1.0,
    gate_threshold: float = 0.5,
) -> Path:
    """Assemble the locked ``models/transformer/<backbone>/`` export from saved heads.

    ``gate_dir`` / ``realfake_dir`` are directories that ALREADY contain a
    ``save_pretrained`` model + tokenizer (the trained heads). This copies each into the
    cascade's ``gate/`` and ``realfake/`` subdirs and writes the two sidecars:
    ``label_map.json`` (from :data:`LABELS`) and ``temperature.json``
    (``{gate, realfake, gate_threshold}``).

    The output is anchored to the caller-supplied ``out_dir`` (the trainer/CLI anchors it to
    ``REPO_ROOT/models/transformer/<backbone>`` — a fixed BACKBONES key, T-03-08). Returns
    the resolved ``out_dir``.
    """
    import shutil

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sub, src in ((STAGE_GATE, Path(gate_dir)), (STAGE_REALFAKE, Path(realfake_dir))):
        dst = out_dir / sub
        if dst.exists():
            shutil.rmtree(dst)
        # copytree mirrors the save_pretrained dir (config.json + safetensors + tokenizer).
        shutil.copytree(src, dst)

    _write_label_map(out_dir)
    save_temperature(
        out_dir / "temperature.json",
        gate=gate_T,
        realfake=realfake_T,
        gate_threshold=gate_threshold,
    )
    log.info("exported cascade -> %s", out_dir)
    return out_dir


# Alternative spelling accepted by the 03-01 export-contract scaffold.
save_pretrained_cascade = export_cascade


def export_backbone(
    backbone_key: str,
    gate_model,
    gate_tokenizer,
    realfake_model,
    realfake_tokenizer,
    gate_T: float,
    realfake_T: float,
    gate_threshold: float,
    out_dir=None,
) -> Path:
    """High-level export: ``save_pretrained`` both in-memory heads then assemble the cascade.

    Persists ``gate_model``/``gate_tokenizer`` into ``<out>/gate`` and
    ``realfake_model``/``realfake_tokenizer`` into ``<out>/realfake`` (mkdir-then-persist,
    safetensors via transformers default — T-03-07), then writes ``label_map.json`` from
    :data:`LABELS` and ``temperature.json``. ``out_dir`` defaults to the project-anchored
    ``REPO_ROOT/models/transformer/<backbone_key>`` (T-03-08); ``backbone_key`` must be a
    fixed :data:`BACKBONES` key, never user input.
    """
    if backbone_key not in BACKBONES:
        raise KeyError(
            f"unknown backbone {backbone_key!r}; known backbones: {sorted(BACKBONES)}"
        )
    if out_dir is None:
        out_dir = REPO_ROOT / "models" / "transformer" / backbone_key
    out_dir = Path(out_dir)

    gate_path = out_dir / STAGE_GATE
    rf_path = out_dir / STAGE_REALFAKE
    gate_path.mkdir(parents=True, exist_ok=True)
    rf_path.mkdir(parents=True, exist_ok=True)

    gate_model.save_pretrained(gate_path)
    gate_tokenizer.save_pretrained(gate_path)
    realfake_model.save_pretrained(rf_path)
    realfake_tokenizer.save_pretrained(rf_path)

    _write_label_map(out_dir)
    save_temperature(
        out_dir / "temperature.json",
        gate=gate_T,
        realfake=realfake_T,
        gate_threshold=gate_threshold,
    )
    log.info("exported backbone %s -> %s", backbone_key, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# CLI — fine-tune both backbones, both heads; fit temperature on val; export.
# ---------------------------------------------------------------------------
def main() -> int:
    """Train both heads of every backbone offline, fit val temperature, export the cascade.

    Heavy/GPU path (Colab T4, D-05/D-10). The gate threshold is left at the default 0.5
    sentinel here; the selection module (03-04) sweeps + records the chosen threshold on val.
    """
    train_df, val_df, _test_df = load_splits()

    for backbone_key, model_id in BACKBONES.items():
        log.info("=== fine-tuning backbone %s (%s) ===", backbone_key, model_id)

        gate_model, gate_tok, gate_logits, gate_labels_v = train_stage(
            model_id, STAGE_GATE, train_df, val_df
        )
        gate_T = fit_temperature(gate_logits, gate_labels_v)

        rf_model, rf_tok, rf_logits, rf_labels_v = train_stage(
            model_id, STAGE_REALFAKE, train_df, val_df
        )
        rf_T = fit_temperature(rf_logits, rf_labels_v)

        export_backbone(
            backbone_key,
            gate_model,
            gate_tok,
            rf_model,
            rf_tok,
            gate_T=gate_T,
            realfake_T=rf_T,
            gate_threshold=0.5,
        )

    log.info("transformer_train complete: exported %s", sorted(BACKBONES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
