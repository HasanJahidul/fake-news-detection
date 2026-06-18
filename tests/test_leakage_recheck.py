"""CLS-03 leakage re-check tests (SC-3 audit gate, plan 02-02).

The re-check is a THIN reuse layer over Phase 1's ``src/data/leakage_probe.py``:
it extracts each trained classical model's top predictive features and runs the
EXISTING ``leak_tells_in_features`` (``_LEAK_TELL`` outlet/year regex) over them,
then flags any model scoring >= 0.98 (macro-F1/accuracy) for suspected-leakage
investigation before it can be trusted.

Fast tests fit each of LogisticRegression / ComplementNB / RandomForestClassifier
on a tiny TF-IDF matrix and assert:
- a PLANTED outlet/year tell ("reuters" / "2017") is caught -> non-empty leak_tells;
- a CLEAN content-word model is clean -> empty leak_tells (negative control);
- the >= 0.98 score flag fires regardless of leak_tells, and clean low-score is not
  flagged.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

from src.models.leakage_recheck import (
    INVESTIGATE_THRESHOLD,
    recheck_model,
    top_tokens_for_model,
)


# ---------------------------------------------------------------------------
# Fixtures: a tiny labelled corpus where each class has a distinctive token.
# A "leaky" variant plants an outlet name / 4-digit year as the class tell.
# ---------------------------------------------------------------------------

# Genuine content words only — no outlet/year tell. The negative control.
_CLEAN_DOCS = [
    "economy budget policy spending government clean",
    "economy budget policy spending public money",
    "vaccine hospital doctor health patient medicine",
    "vaccine hospital doctor health clinic nurse",
    "phishing password account login verify click",
    "phishing password account login bank urgent",
]
_CLEAN_LABELS = ["real", "real", "fake", "fake", "malicious", "malicious"]

# Same content but each class also carries a planted source-artifact tell:
#   real -> "reuters" (outlet), fake -> "2017" (year), malicious -> "afp" (outlet).
_LEAKY_DOCS = [
    "economy budget policy reuters government",
    "economy budget reuters public money policy",
    "vaccine 2017 doctor health patient medicine",
    "vaccine 2017 doctor health clinic nurse",
    "phishing afp account login verify click",
    "phishing afp account login bank urgent",
]
_LEAKY_LABELS = ["real", "real", "fake", "fake", "malicious", "malicious"]


def _fit(model, docs, labels):
    """Fit a TfidfVectorizer + model; return (fitted_model, feature_names)."""
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 1), min_df=1)
    x = vec.fit_transform(docs)
    model.fit(x, labels)
    return model, list(vec.get_feature_names_out())


def _models():
    return [
        LogisticRegression(class_weight="balanced", max_iter=1000),
        ComplementNB(),
        RandomForestClassifier(n_estimators=20, random_state=0),
    ]


# ---------------------------------------------------------------------------
# top_tokens_for_model: every family yields tokens drawn from feature_names.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", _models())
def test_top_tokens_for_model_returns_vocabulary_tokens(model):
    fitted, names = _fit(model, _CLEAN_DOCS, _CLEAN_LABELS)
    top = top_tokens_for_model(fitted, names, n=5)
    assert isinstance(top, dict) and top, "expected a non-empty per-key dict of tokens"
    name_set = set(names)
    for key, tokens in top.items():
        assert isinstance(tokens, list) and tokens
        for tok in tokens:
            assert tok in name_set, f"{tok!r} not drawn from supplied feature_names"


def test_top_tokens_logreg_keyed_by_class():
    fitted, names = _fit(
        LogisticRegression(class_weight="balanced", max_iter=1000),
        _CLEAN_DOCS,
        _CLEAN_LABELS,
    )
    top = top_tokens_for_model(fitted, names, n=5)
    assert set(top.keys()) == {"real", "fake", "malicious"}


def test_top_tokens_random_forest_is_global():
    fitted, names = _fit(
        RandomForestClassifier(n_estimators=20, random_state=0),
        _CLEAN_DOCS,
        _CLEAN_LABELS,
    )
    top = top_tokens_for_model(fitted, names, n=5)
    # RF importances are not per-class -> a single global key.
    assert list(top.keys()) == ["_global_"]


# ---------------------------------------------------------------------------
# recheck_model: planted tell caught; clean model clean (negative control).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", _models())
def test_recheck_catches_planted_outlet_or_year(model):
    fitted, names = _fit(model, _LEAKY_DOCS, _LEAKY_LABELS)
    out = recheck_model(fitted, names, score=0.80, n=10)
    assert out["leak_tells"], "planted reuters/2017/afp tell should surface in leak_tells"
    assert out["investigate"] is True


@pytest.mark.parametrize("model", _models())
def test_recheck_clean_model_is_clean(model):
    fitted, names = _fit(model, _CLEAN_DOCS, _CLEAN_LABELS)
    out = recheck_model(fitted, names, score=0.80, n=10)
    assert out["leak_tells"] == [], "clean content-word model should report no tells"
    assert out["investigate"] is False


# ---------------------------------------------------------------------------
# >= 0.98 investigation flag.
# ---------------------------------------------------------------------------


def test_investigate_threshold_value():
    assert INVESTIGATE_THRESHOLD == 0.98


def test_high_score_flags_investigate_even_when_clean():
    fitted, names = _fit(
        LogisticRegression(class_weight="balanced", max_iter=1000),
        _CLEAN_DOCS,
        _CLEAN_LABELS,
    )
    out = recheck_model(fitted, names, score=0.99, n=10)
    assert out["leak_tells"] == []
    assert out["investigate"] is True
    assert out["score"] == pytest.approx(0.99)


def test_low_score_clean_is_not_flagged():
    fitted, names = _fit(ComplementNB(), _CLEAN_DOCS, _CLEAN_LABELS)
    out = recheck_model(fitted, names, score=0.80, n=10)
    assert out["investigate"] is False


def test_recheck_return_shape():
    fitted, names = _fit(ComplementNB(), _CLEAN_DOCS, _CLEAN_LABELS)
    out = recheck_model(fitted, names, score=0.5, n=5)
    assert set(out.keys()) == {"top_features", "leak_tells", "score", "investigate"}
    assert isinstance(out["score"], float)
    assert isinstance(out["investigate"], bool)
    assert isinstance(out["leak_tells"], list)
    assert isinstance(out["top_features"], dict)
