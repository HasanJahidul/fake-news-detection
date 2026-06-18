"""Metric-discipline scoring module (CLS-03 / SC-2).

The project's headline metric contract. **Macro-F1 is the headline** (SC-2 forbids
accuracy as the headline number) — there is deliberately NO ``accuracy()`` helper here;
any accuracy value is secondary and lives in the report writer (plan 02-03), not here.

Every label-ordered function imports the locked class order from
``src.data.label_map.LABELS`` (``("real","fake","malicious")``) — the order is NEVER
hardcoded as a string literal, so the confusion-matrix rows/cols and per-class report
stay consistent with the rest of the project.

Functions
---------
macro_f1                 headline macro-averaged F1.
per_class_report         per-class precision / recall / f1-score (classification_report dict).
confusion               3x3 confusion matrix in locked LABELS order.
per_language_macro_f1    macro-F1 grouped by provenance ``language`` (bn/en/code-mixed),
                         skipping languages with zero rows (malicious is English-only).
minority_recall          per-class recall keyed by LABELS, so the D-03 selection guard
                         can reject a model with collapsed fake/malicious recall.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.data.label_map import LABELS


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """Headline metric: macro-averaged F1 over the 3 classes (SC-2)."""
    return float(f1_score(y_true, y_pred, average="macro"))


def per_class_report(y_true: Sequence, y_pred: Sequence) -> dict:
    """Per-class precision / recall / f1-score as a dict, in locked LABELS order."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )


def confusion(y_true: Sequence, y_pred: Sequence):
    """3x3 confusion matrix whose row/col order is exactly ``LABELS`` (locked order)."""
    return confusion_matrix(y_true, y_pred, labels=list(LABELS))


def per_language_macro_f1(
    df: pd.DataFrame, y_true: Sequence, y_pred: Sequence
) -> dict[str, float]:
    """Macro-F1 per provenance ``language`` group (bn / en / code-mixed).

    Groups the aligned ``(y_true, y_pred)`` by ``df["language"]`` and computes
    :func:`macro_f1` per group. Languages with zero rows are simply absent from the
    result (groupby only yields present groups) — malicious is English-only, so a split
    may legitimately lack a code-mixed group; this must not raise (build_corpus precedent).
    """
    yt = pd.Series(list(y_true)).reset_index(drop=True)
    yp = pd.Series(list(y_pred)).reset_index(drop=True)
    lang = df["language"].reset_index(drop=True)

    out: dict[str, float] = {}
    for group, idx in lang.groupby(lang).groups.items():
        if len(idx) == 0:
            continue
        out[str(group)] = macro_f1(yt.loc[idx], yp.loc[idx])
    return out


def minority_recall(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    """Per-class recall keyed by LABELS, for the D-03 minority-class selection guard.

    Returns recall for every label (incl. the ``fake`` / ``malicious`` minority classes).
    A class the model never predicts yields recall ``0.0`` (``zero_division=0``) — exactly
    the collapsed-minority case the D-03 guard must reject.
    """
    recalls = recall_score(
        y_true, y_pred, labels=list(LABELS), average=None, zero_division=0
    )
    return {label: float(r) for label, r in zip(LABELS, recalls)}
