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

SCOPE (WR-03): leak-tell matching is restricted to WORD-view feature names; char-view
(``char__``) n-gram fragments are filtered out before the ``_LEAK_TELL`` regex runs,
because that whole-token outlet/year regex cannot match tells fragmented across char
n-grams. Char-view leak detection is therefore out of scope for this gate.
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

# WR-03 — SCOPE: the SC-3 leak-tell re-check inspects WORD-view feature names only. The
# Phase-2 vectorizer is a FeatureUnion of a `word` view and a `char_wb (3,5)` view, so
# ~half the feature names are char n-gram fragments (e.g. `char__ 201`, `char__016`). The
# reused Phase-1 `_LEAK_TELL` regex (\b(?:19|20)\d{2}\b for years + an outlet alternation)
# only matches clean, whole tokens; a year/outlet tell fragmented across char n-grams
# cannot match it. Rather than relax the SINGLE-SOURCE-OF-TRUTH regex (which would risk
# false positives across the Phase-1 gate that also reuses it), we explicitly restrict the
# re-check to word-view names. Char-view leak detection is OUT OF SCOPE for this gate;
# `top_features` still records every extracted token (word + char) for the audit trail.
_CHAR_VIEW_PREFIX = "char__"


def _word_view_only(top: dict) -> dict:
    """Drop char-view (``char__``-prefixed) feature names from a per-key top-token dict.

    The ``_LEAK_TELL`` regex cannot match char n-gram fragments, so passing them through
    adds only noise (WR-03). Word-view names are kept verbatim (including any ``word__``
    prefix, which the regex matches inside the token string).
    """
    return {
        key: [t for t in tokens if not str(t).startswith(_CHAR_VIEW_PREFIX)]
        for key, tokens in top.items()
    }


def _rank_tokens(
    weights: np.ndarray, feature_names: list[str], n: int, *, ascending: bool = False
) -> list[str]:
    """Top-``n`` ``feature_names`` by ``weights`` (1-D importance/log-prob row).

    Descending by default (largest weight = most predictive). Pass ``ascending=True``
    for **ComplementNB** (WR-01): its ``feature_log_prob_`` is a *complement*-class
    statistic, so the features most characteristic of a class are the SMALLEST (most
    negative) values, not the largest — ranking it descending surfaces the wrong tokens.
    """
    order = np.asarray(weights).argsort()
    if not ascending:
        order = order[::-1]
    return [str(feature_names[i]) for i in order[:n]]


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
        # WR-01: ComplementNB's feature_log_prob_ is a COMPLEMENT statistic — rank
        # ascending so the smallest (most class-characteristic) values come first.
        # MultinomialNB and other NB variants keep the standard descending order.
        ascending = type(model).__name__ == "ComplementNB"
        return {
            str(cls): _rank_tokens(logp[ci], feature_names, n, ascending=ascending)
            for ci, cls in enumerate(classes)
        }

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

    SCOPE (WR-03): leak-tell matching runs over WORD-view feature names only; char-view
    (``char__``) n-gram fragments are filtered out first because the ``_LEAK_TELL`` regex
    cannot match year/outlet tells fragmented across char n-grams. ``top_features`` still
    reports every extracted token (word + char) for the auditable trail.

    Returns ``{"top_features", "leak_tells", "score", "investigate"}``.
    """
    top = top_tokens_for_model(model, feature_names, n=n)
    tells = leak_tells_in_features(_word_view_only(top))
    return {
        "top_features": top,
        "leak_tells": tells,
        "score": float(score),
        "investigate": bool(score >= INVESTIGATE_THRESHOLD or tells),
    }
