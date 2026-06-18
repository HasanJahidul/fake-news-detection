"""CLS-03 / SC-3 — leakage RE-CHECK audit gate (Phase 2).

A thin REUSE layer over Phase 1's :mod:`src.data.leakage_probe`. For each trained
classical model it extracts the top predictive features and runs the EXISTING
``leak_tells_in_features`` (the ``_LEAK_TELL`` outlet/year regex) over them, then flags
any model scoring ``>= 0.98`` (macro-F1/accuracy) for suspected-leakage investigation
before it can be trusted.

The leak-tell definition (outlet names + 4-digit years) is the SINGLE SOURCE OF TRUTH
in ``src.data.leakage_probe`` — this module never redefines it (T-02-03: a divergent copy
could silently weaken the gate). For LogisticRegression we delegate to the Phase-1
``top_features`` ranking logic; ComplementNB uses ``feature_log_prob_`` and
RandomForestClassifier uses ``feature_importances_`` (a single global ranking, since RF
importances are not per-class).

``recheck_model`` returns an explicit ``investigate`` flag + ``leak_tells`` list so the
02-03 report records an auditable trail of which models were cleared / flagged (T-02-04).
"""

from __future__ import annotations

import numpy as np

from src.data.label_map import LABELS  # noqa: F401  (canonical class order; documents intent)
from src.data.leakage_probe import leak_tells_in_features, top_features

# SC-3: any model scoring >= 98% (macro-F1/accuracy) is investigated as suspected leakage
# before being trusted (CONTEXT D-discretion: locked by ROADMAP success criterion 3).
INVESTIGATE_THRESHOLD: float = 0.98

# RF feature_importances_ are a single GLOBAL ranking (not per-class) — keyed under this.
_GLOBAL_KEY = "_global_"


def _rank_tokens(weights: np.ndarray, feature_names: list[str], n: int) -> list[str]:
    """Top-``n`` ``feature_names`` by descending ``weights`` (1-D importance/log-prob row)."""
    order = np.asarray(weights).argsort()[::-1][:n]
    return [str(feature_names[i]) for i in order]


def top_tokens_for_model(model, feature_names, n: int = 20) -> dict:
    """Extract the top-``n`` predictive tokens per class for a fitted classical model.

    - **LogisticRegression** (``.coef_``): delegate to the Phase-1 ranking in
      :func:`src.data.leakage_probe.top_features`. ``top_features`` reads
      ``vectorizer.get_feature_names_out()``; we accept ``feature_names`` directly and
      adapt via a tiny shim exposing that method, so the SAME argsort logic runs whether
      the caller passes a fitted ``FeatureUnion``'s names or a vectorizer's names.
    - **ComplementNB** (``.feature_log_prob_``): one row per class -> per-class top tokens
      keyed by ``model.classes_``.
    - **RandomForestClassifier** (``.feature_importances_``): a single global ranking
      (importances are not per-class) -> top tokens under ``"_global_"``.

    Returns a ``dict`` of ``{class_or_"_global_": [token, ...]}`` with tokens drawn from
    ``feature_names``.
    """
    feature_names = list(feature_names)

    # ComplementNB / other NB: per-class log-probabilities.
    if hasattr(model, "feature_log_prob_"):
        classes = list(model.classes_)
        logp = np.asarray(model.feature_log_prob_)
        return {str(cls): _rank_tokens(logp[ci], feature_names, n) for ci, cls in enumerate(classes)}

    # RandomForest / tree ensembles: single global importance ranking (not per-class).
    if hasattr(model, "feature_importances_"):
        return {_GLOBAL_KEY: _rank_tokens(model.feature_importances_, feature_names, n)}

    # LogisticRegression / linear models with .coef_: reuse the Phase-1 ranking verbatim.
    if hasattr(model, "coef_"):
        class _NamesShim:
            def get_feature_names_out(self):  # mirrors the TfidfVectorizer API top_features needs
                return np.asarray(feature_names, dtype=object)

        return top_features(model, _NamesShim(), n=n)

    raise TypeError(
        f"unsupported model {type(model).__name__}: expected one of "
        "coef_ (linear), feature_log_prob_ (NB), feature_importances_ (tree ensemble)"
    )


def recheck_model(model, feature_names, score: float, n: int = 20) -> dict:
    """Run the SC-3 leak-tell re-check on one fitted model.

    Extracts the top-``n`` features (:func:`top_tokens_for_model`), runs the REUSED
    Phase-1 :func:`src.data.leakage_probe.leak_tells_in_features` over them, and sets
    ``investigate`` when the model's ``score`` reaches :data:`INVESTIGATE_THRESHOLD`
    OR any outlet/year tell surfaces.

    Returns ``{"top_features", "leak_tells", "score", "investigate"}``.
    """
    top = top_tokens_for_model(model, feature_names, n=n)
    tells = leak_tells_in_features(top)
    return {
        "top_features": top,
        "leak_tells": tells,
        "score": float(score),
        "investigate": bool(score >= INVESTIGATE_THRESHOLD or tells),
    }
