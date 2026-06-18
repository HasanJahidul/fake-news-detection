"""CLS-01 / CLS-03 — classical-baseline orchestrator tests (plan 02-03).

Mirrors the ``tests/test_leakage_probe.py`` TDD convention: small in-repo fixtures
(``_make_frame`` + a synthetic 3-class corpus), fast unit/contract tests for the
train/select/serialize/SC-3 wiring, and a ``@pytest.mark.slow`` test that runs the
full orchestrator against the REAL built corpus (``data/processed/*.parquet``) — which
must RUN (not skip), propagating a missing-Parquet error.

Contracts asserted (SC-1..SC-4 + D-01/D-03):
  * MODELS = exactly logreg/complement_nb/random_forest; class_weight balancing +
    ComplementNB only; NO imblearn/SMOTE anywhere in the module (D-01).
  * train_and_compare returns per-model val/test macro-F1, per-class report, confusion,
    minority recall, a leakage_recheck entry, and a selected best key.
  * select_best applies the D-03 minority guard (collapsed minority recall cannot win;
    ties break toward higher minority recall).
  * serialize_artifacts + load_artifacts round-trip the fitted vectorizer + best model
    with NO training code path (SC-1 load-only reuse).
  * slow real-corpus run: best test macro-F1 > 0.33, every model fake AND malicious
    recall > 0.0, investigate flag recorded.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src.data.label_map import LABELS
import src.models.train_classical as tc


# ---------------------------------------------------------------------------
# Fixtures (mirror test_leakage_probe._make_frame style)
# ---------------------------------------------------------------------------


def _make_frame(rows):
    return pd.DataFrame(
        rows, columns=["text", "label", "source_dataset", "original_label", "language"]
    )


# Class-separable synthetic bodies so the small fixture trains a non-trivial model:
# each class gets a distinct content vocabulary (NO outlet/year tells — clean content).
_REAL = [
    "parliament approved the annual education budget after a long debate today",
    "the central bank kept the interest rate steady citing stable inflation",
    "scientists published a peer reviewed study on renewable energy storage",
    "the city council opened a new public library in the eastern district",
    "farmers reported a strong harvest season across the northern region",
    "the ministry launched a vaccination drive for school children this month",
]
_FAKE = [
    "shocking miracle cure removes all disease overnight doctors are hiding it",
    "secret government plan to control minds through tap water exposed now",
    "celebrity faked their own death to escape a massive hidden scandal",
    "aliens landed and signed a treaty no media will ever dare report this",
    "drinking this magic potion makes you immortal say anonymous insiders",
    "the moon is actually a hologram projected by a shadowy elite cabal",
]
_MAL = [
    "click this link now to claim your free prize before your account is locked",
    "urgent verify your bank password here or lose access within one hour",
    "you won a lottery send your card number to receive the cash transfer",
    "your parcel is held pay the customs fee via this link immediately today",
    "congratulations selected for a gift card confirm your ssn at this site",
    "security alert login at this address to stop unauthorized charges now",
]


def _class_rows(bodies, label, source, n):
    return [(bodies[i % len(bodies)], label, source, label, "en") for i in range(n)]


def _synthetic_frame(n_per_class: int = 18) -> pd.DataFrame:
    rows = (
        _class_rows(_REAL, "real", "isot", n_per_class)
        + _class_rows(_FAKE, "fake", "isot", n_per_class)
        + _class_rows(_MAL, "malicious", "smsspam", n_per_class)
    )
    return _make_frame(rows)


@pytest.fixture
def splits():
    """Train/val/test synthetic frames drawn from the same separable pools."""
    return _synthetic_frame(18), _synthetic_frame(9), _synthetic_frame(9)


# ---------------------------------------------------------------------------
# MODELS contract (D-01)
# ---------------------------------------------------------------------------


def test_models_exact_set_and_balancing():
    assert set(tc.MODELS) == {"logreg", "complement_nb", "random_forest"}
    from sklearn.naive_bayes import ComplementNB, MultinomialNB

    assert isinstance(tc.MODELS["complement_nb"], ComplementNB)
    assert not isinstance(tc.MODELS["complement_nb"], MultinomialNB)
    assert tc.MODELS["logreg"].class_weight == "balanced"
    assert tc.MODELS["random_forest"].class_weight == "balanced"


def test_no_smote_or_oversampling_imported():
    src = inspect.getsource(tc)
    for forbidden in ("imblearn", "SMOTE", "RandomOverSampler"):
        assert forbidden not in src, f"D-01: {forbidden} must not appear in train_classical"


# ---------------------------------------------------------------------------
# train_and_compare result shape (SC-2 + SC-3)
# ---------------------------------------------------------------------------


def test_train_and_compare_result_shape(splits):
    train, val, test = splits
    res = tc.train_and_compare(train, val, test)
    assert "models" in res and "best" in res
    assert res["best"] in tc.MODELS
    for name in tc.MODELS:
        m = res["models"][name]
        assert 0.0 <= m["val_macro_f1"] <= 1.0
        assert 0.0 <= m["test_macro_f1"] <= 1.0
        assert "per_class" in m
        assert "confusion" in m
        # confusion is 3x3 in LABELS order
        assert len(m["confusion"]) == len(LABELS)
        assert all(len(row) == len(LABELS) for row in m["confusion"])
        assert set(m["minority_recall"]) == set(LABELS)
        # SC-3 leakage re-check wired per model
        assert set(m["leakage_recheck"]) >= {"top_features", "leak_tells", "investigate"}


# ---------------------------------------------------------------------------
# select_best D-03 minority guard
# ---------------------------------------------------------------------------


def _fake_result(val_f1, minority):
    return {"val_macro_f1": val_f1, "minority_recall": minority}


def test_select_best_rejects_collapsed_minority():
    results = {
        # higher val macro-F1 but malicious recall collapsed to 0.0 -> disqualified
        "A": _fake_result(0.90, {"real": 0.95, "fake": 0.80, "malicious": 0.0}),
        # slightly lower macro-F1 but healthy minority recall -> wins
        "B": _fake_result(0.85, {"real": 0.85, "fake": 0.70, "malicious": 0.60}),
    }
    assert tc.select_best(results) == "B"


def test_select_best_tie_breaks_toward_minority_recall():
    results = {
        "A": _fake_result(0.80, {"real": 0.80, "fake": 0.50, "malicious": 0.40}),
        "B": _fake_result(0.80, {"real": 0.80, "fake": 0.70, "malicious": 0.65}),
    }
    assert tc.select_best(results) == "B"


# ---------------------------------------------------------------------------
# serialize / load round-trip (SC-1 load-only reuse)
# ---------------------------------------------------------------------------


def test_serialize_and_load_roundtrip(splits, tmp_path):
    train, val, test = splits
    res = tc.train_and_compare(train, val, test)
    vec = res["vectorizer"]
    best_model = res["fitted_models"][res["best"]]

    tc.serialize_artifacts(vec, best_model, tmp_path)
    assert (tmp_path / "vectorizer.joblib").exists()
    assert (tmp_path / "best_model.joblib").exists()

    loaded_vec, loaded_model = tc.load_artifacts(tmp_path)
    sample = test["text"].tolist()[:5]
    from src.models.vectorizer import texts_from_frame

    X_in = vec.transform([__import__("src.preprocess", fromlist=["preprocess"]).preprocess(t) for t in sample])
    X_loaded = loaded_vec.transform(
        [__import__("src.preprocess", fromlist=["preprocess"]).preprocess(t) for t in sample]
    )
    assert (best_model.predict(X_in) == loaded_model.predict(X_loaded)).all()


def test_load_artifacts_has_no_training_path():
    """SC-1: load_artifacts must not call any training function (load-only reuse)."""
    src = inspect.getsource(tc.load_artifacts)
    assert "train_and_compare" not in src
    assert ".fit(" not in src and "fit_transform" not in src


# ---------------------------------------------------------------------------
# SLOW: full orchestrator on the REAL built corpus (must RUN, not skip)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_corpus_end_to_end():
    """SC-1..SC-3 against data/processed/*.parquet. Built Parquet is a precondition —
    a missing file must PROPAGATE (no skip)."""
    from src.data.schema import read_parquet

    train = read_parquet("data/processed/train.parquet")
    val = read_parquet("data/processed/val.parquet")
    test = read_parquet("data/processed/test.parquet")

    res = tc.train_and_compare(train, val, test)

    # best beats informed chance for 3 (roughly) balanced classes
    best = res["models"][res["best"]]
    assert best["test_macro_f1"] > 0.33, f"best test macro-F1 {best['test_macro_f1']:.4f} <= 0.33"

    # every model has non-trivial minority recall (fake AND malicious)
    for name, m in res["models"].items():
        assert m["minority_recall"]["fake"] > 0.0, f"{name}: fake recall collapsed"
        assert m["minority_recall"]["malicious"] > 0.0, f"{name}: malicious recall collapsed"

    # SC-3 investigate flag recorded per model
    assert "any_investigate" in res
    assert isinstance(res["any_investigate"], bool)
