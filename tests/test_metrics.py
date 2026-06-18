"""Plan 02-01 Task 2 — tests for the metric-discipline module (CLS-03 / SC-2).

Macro-F1 is the headline metric (accuracy is never the headline, SC-2). Tests assert:
macro-F1 matches sklearn, per-class report carries P/R/F1 for every locked label,
the confusion matrix is in LOCKED LABELS order, per-language macro-F1 omits empty groups,
and the minority-recall guard exposes fake/malicious recall (== 0.0 when a class is never
predicted) so the D-03 selection guard can reject collapsed minority recall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.label_map import LABELS
from src.models.metrics import (
    confusion,
    macro_f1,
    minority_recall,
    per_class_report,
    per_language_macro_f1,
)

# Deterministic 3-class fixture (seed 0, balanced enough to exercise every label).
_Y_TRUE = ["real", "fake", "malicious", "real", "fake", "malicious", "real", "fake"]
_Y_PRED = ["real", "fake", "malicious", "fake", "fake", "malicious", "real", "real"]


def test_macro_f1_matches_sklearn():
    expected = float(f1_score(_Y_TRUE, _Y_PRED, average="macro"))
    assert macro_f1(_Y_TRUE, _Y_PRED) == expected


def test_per_class_report_has_all_labels_with_prf():
    rep = per_class_report(_Y_TRUE, _Y_PRED)
    for label in ("real", "fake", "malicious"):
        assert label in rep
        for metric in ("precision", "recall", "f1-score"):
            assert metric in rep[label]


def test_confusion_is_3x3_in_locked_label_order():
    cm = confusion(_Y_TRUE, _Y_PRED)
    cm = np.asarray(cm)
    assert cm.shape == (3, 3)
    # LABELS = ("real","fake","malicious"). One "real" true was predicted "fake"
    # (index 3 in fixture) -> off-diagonal cell [row=real(0), col=fake(1)] must be >= 1.
    real_idx = list(LABELS).index("real")
    fake_idx = list(LABELS).index("fake")
    assert cm[real_idx][fake_idx] >= 1


def test_per_language_macro_f1_omits_empty_group():
    df = pd.DataFrame(
        {
            "language": ["en", "en", "en", "en", "bn", "bn", "bn", "bn"],
            # no code-mixed rows present
        }
    )
    out = per_language_macro_f1(df, _Y_TRUE, _Y_PRED)
    assert set(out.keys()) == {"en", "bn"}
    assert "code-mixed" not in out
    for v in out.values():
        assert isinstance(v, float)


def test_minority_recall_exposes_fake_and_malicious():
    rec = minority_recall(_Y_TRUE, _Y_PRED)
    assert "fake" in rec and "malicious" in rec


def test_minority_recall_zero_when_class_never_predicted():
    y_true = ["malicious", "malicious", "real", "fake"]
    y_pred = ["real", "fake", "real", "fake"]  # never predicts malicious
    rec = minority_recall(y_true, y_pred)
    assert rec["malicious"] == 0.0
