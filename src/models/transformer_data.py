"""Transformer data leaf: stage label views + preprocess→tokenize + class weights (CLS-02).

Two stateless concerns the trainer (03-03) and loader/selector (03-04) both compose,
built purely over the locked v1.0 contracts — no normalization fork, no resampling:

  * **Stage label views (Pattern 2).** Stage 1 (gate) and Stage 2 (real/fake) derive
    their label views from the SAME 3-class parquet:
      - :func:`gate_labels`  — malicious → 1 else 0, over ALL rows (binary gate).
      - :func:`realfake_frame` — keep only the real/fake rows (drop malicious).
    The malicious class is English-only (D-01), so the gate has zero Bangla malicious rows.

  * **preprocess → tokenize (D-12 normalizer parity).** :func:`build_tokenized` runs the
    ONE shared :func:`src.preprocess.preprocess` BEFORE the ``AutoTokenizer`` — the exact
    same function inference uses. Skipping this is the most likely silent-accuracy bug
    (BanglaBERT was pretrained on normalized text), so train and inference must not fork.

  * **Inverse-frequency class weights (Pattern 3, D-14).** :func:`class_weights` returns a
    per-stage inverse-frequency ``torch.FloatTensor`` to handle the natural class imbalance
    WITHOUT resampling (D-14 forbids SMOTE / oversampling).

Splits are loaded ONLY via :func:`src.data.schema.read_parquet` from the fixed
``REPO_ROOT / data/processed/`` path (never a user-supplied path; T-03-03), preserving the
``language`` column so the selector can compute per-language macro-F1 (D-13).

``MAX_LENGTH = 256`` is a deliberate head-truncation choice: most of the discriminative
signal is early in news/spam text, and 256 tokens fine-tune far faster on a T4/M4 than 512
(Pitfall 6). Long tails are truncated from the end.

``torch`` is imported lazily inside :func:`class_weights` so this module imports without
torch present (the fast suite installs without the transformer optional group).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.label_map import LABELS  # noqa: F401  (locked class order; never hardcode)
from src.preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("transformer_data")

# Repo root = two levels up from src/models/transformer_data.py (mirrors train_classical).
REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

# Head-truncation length (D-discretion, Pitfall 6 — most signal is early, faster on T4/M4).
MAX_LENGTH = 256

# Per-stage class orders. Index 1 of the gate is the positive (malicious) class.
GATE_CLASSES: tuple[str, str] = ("not_malicious", "malicious")
REALFAKE_CLASSES: tuple[str, str] = ("real", "fake")


def gate_labels(df: pd.DataFrame) -> pd.Series:
    """Stage-1 gate labels: malicious → 1, everything else → 0, over ALL rows (Pattern 2).

    Returns an int Series aligned to ``df.index``. ``not_malicious`` (real/fake) is the
    negative class (0); ``malicious`` is the positive class (1) — matching
    :data:`GATE_CLASSES` index order.
    """
    return (df["label"] == "malicious").astype(int)


def realfake_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Stage-2 view: keep only the real/fake rows, drop every malicious row (Pattern 2).

    Returns a COPY so downstream label re-encoding never mutates the caller's frame.
    The ``language`` column (and all provenance) is preserved.
    """
    return df[df["label"].isin(["real", "fake"])].copy()


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the leak-free, source-disjoint 70/15/15 parquet splits (T-03-03).

    Reads ONLY via :func:`src.data.schema.read_parquet` from the fixed
    ``REPO_ROOT / data/processed/<split>.parquet`` path — never a user-supplied path.
    Returns ``(train, val, test)`` with the ``language`` column preserved (D-13).
    """
    from src.data.schema import read_parquet

    train = read_parquet(REPO_ROOT / "data/processed/train.parquet")
    val = read_parquet(REPO_ROOT / "data/processed/val.parquet")
    test = read_parquet(REPO_ROOT / "data/processed/test.parquet")
    return train, val, test


def build_tokenized(texts, tokenizer, max_length: int = MAX_LENGTH):
    """Run the SHARED ``preprocess()`` on each text, THEN tokenize (D-12 normalizer parity).

    The preprocessing is the EXACT :func:`src.preprocess.preprocess` inference uses — no
    fork. NA-safe like :func:`src.models.vectorizer.texts_from_frame`: a pandas NA cell is
    coalesced to ``None`` so the missing-cell → ``""`` contract holds. Tokenization uses
    head truncation at ``max_length`` with padding.

    Returns the tokenizer's batch encoding (e.g. a ``BatchEncoding`` dict of input_ids /
    attention_mask).
    """
    cleaned = [preprocess(None if pd.isna(t) else t) for t in texts]
    return tokenizer(
        cleaned,
        truncation=True,
        max_length=max_length,
        padding=True,
    )


def class_weights(labels, num_classes: int | None = None):
    """Inverse-frequency class weights as a ``torch.FloatTensor`` (Pattern 3, D-14, WR-03).

    ``labels`` is a sequence of integer class indices (per the relevant stage class order).
    Weight for class ``c`` is ``N / (K * count_c)`` (inverse frequency, normalized so the
    mean weight is ~1) — larger for the rarer class. NO resampling (D-14 forbids SMOTE /
    oversampling); imbalance is handled purely by these loss weights.

    The returned tensor ALWAYS has length ``num_classes`` (WR-03): when a class of the full
    stage label set is absent from this split, its index gets weight ``0.0`` (no rows ⇒ no
    gradient) rather than the vector being shorter than the stage's label count — otherwise a
    ``CrossEntropyLoss(weight=...)`` would crash on a shape mismatch against the head's logits.
    ``num_classes`` defaults to ``max observed index + 1`` to preserve the existing call sites
    where every class is present. ``K`` (the inverse-frequency divisor) is the number of
    PRESENT classes, so present-class weights are unchanged from the prior behavior.

    ``torch`` is imported lazily here so the module imports without torch present.
    """
    import torch

    counts = pd.Series(list(labels)).value_counts()
    n = int(counts.sum())
    k = int(counts.shape[0])  # present classes only — absent classes contribute no gradient

    if num_classes is None:
        num_classes = int(max(counts.index)) + 1 if k else 0

    # Absent indices default to 0.0; present indices keep the inverse-frequency weight.
    weights = [0.0] * num_classes
    for c in counts.index:
        weights[int(c)] = n / (k * int(counts[c]))
    return torch.FloatTensor(weights)
