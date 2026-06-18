"""Shared hybrid word+char TF-IDF vectorizer (D-02).

The single feature representation for every Phase 2 classical baseline. A sklearn
``FeatureUnion`` concatenates two TF-IDF views over the SAME ``preprocess()``-cleaned
bilingual text:

  * ``word``  — ``analyzer="word"``, ``ngram_range=(1,2)``: English phrases / Bangla word units.
  * ``char``  — ``analyzer="char_wb"``, ``ngram_range=(3,5)``: Bangla morphology + code-mixed
    / OOV / sub-word forms, language-agnostic.

The word+char hybrid STRUCTURE is the locked D-02 decision; the exact ngram ranges,
``min_df=2`` and ``max_features=50_000`` are the chosen CPU-friendly defaults (recorded
in the comparison report in plan 02-03).

Input contract (D-02 / D-08): text enters strictly through the shared lossless
``preprocess()`` via :func:`texts_from_frame` — the ONE shared text entry point. We do
NOT re-normalize here (preprocess already runs csebuetnlp/normalizer). TF-IDF's own
``lowercase``/tokenization is the per-consumer cleaning layer ON TOP of preprocess().
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.preprocess import preprocess

# Chosen defaults (tunable per D-02; hybrid structure is locked).
_MIN_DF = 2
_MAX_FEATURES = 50_000


def build_vectorizer() -> FeatureUnion:
    """Return an UNFITTED hybrid word+char TF-IDF ``FeatureUnion`` (D-02).

    The two transformers are named exactly ``"word"`` and ``"char"``. Both share
    ``lowercase=True``, ``min_df=2``, ``max_features=50_000``. Caller fits on text
    produced by :func:`texts_from_frame`.
    """
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=_MIN_DF,
                    max_features=_MAX_FEATURES,
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=_MIN_DF,
                    max_features=_MAX_FEATURES,
                    lowercase=True,
                ),
            ),
        ]
    )


def texts_from_frame(df: pd.DataFrame) -> list[str]:
    """Extract the model-ready text column, routed through the shared ``preprocess()``.

    Returns ``[preprocess(t) for t in df["text"]]`` — the single shared text entry point
    (D-02 / D-08). ``preprocess()`` maps None / empty / whitespace-only to ``""`` and is
    idempotent; we never re-implement normalization here.

    pandas coerces a Python ``None`` in an object column to ``float('nan')``, which is
    not the literal ``None`` that ``preprocess()`` guards against. We coalesce any pandas
    NA to ``None`` at this single shared entry point so the missing-cell -> "" contract
    holds regardless of how the frame stored the missing value.
    """
    return [preprocess(None if pd.isna(t) else t) for t in df["text"]]
