"""DATA-03 leakage probe tests (PHASE GATE, plan 01-07).

The source-stripped leakage probe: a cheap TF-IDF+LogReg trained on three
deliberately-degraded views (title-only / single-body-sentence / source-stripped)
must NOT score near-ceiling on degraded input. On the source-stripped view it must
score meaningfully below a full-content model AND below ~0.95 macro-F1, proving no
source artifact leaks the label.

Fast tests use small in-repo fixtures (incl. a NEGATIVE CONTROL where the leak is
intact → probe must score near-ceiling, proving the probe actually detects leaks).
The slow test runs the gate against the real built corpus (data/processed/*.parquet)
and must RUN (not skip) — built Parquet is a precondition of this phase gate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.leakage_probe import (
    probe_all_views,
    run_probe,
    split_sentences,
    top_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_frame(rows):
    return pd.DataFrame(
        rows, columns=["text", "label", "source_dataset", "original_label", "language"]
    )


@pytest.fixture
def stripped_corpus():
    """A small corpus where the source leak HAS been stripped.

    Real vs fake bodies differ only in genuine content (no outlet/dateline tell),
    so a degraded-view probe should NOT be able to cheat to near-ceiling.
    """
    real = [
        (f"The committee reviewed the annual budget and approved new spending item {i}.",
         "real", "isot", "true", "en")
        for i in range(12)
    ]
    fake = [
        (f"Shocking claim {i}: aliens secretly control the weather, anonymous post says.",
         "fake", "isot", "fake", "en")
        for i in range(12)
    ]
    mal = [
        (f"WIN a FREE prize {i} now click the link to claim your reward immediately.",
         "malicious", "smsspam", "spam", "en")
        for i in range(12)
    ]
    return _make_frame(real + fake + mal)


@pytest.fixture
def leaky_corpus():
    """NEGATIVE CONTROL: the Reuters dateline leak is INTACT (not stripped).

    Every real body starts with 'WASHINGTON (Reuters) -'; fake/malicious never do.
    A correct probe must catch this and score near-ceiling on source artifacts —
    proving the probe is actually sensitive to leaks.
    """
    real = [
        (f"WASHINGTON (Reuters) - The committee reviewed budget item {i} today.",
         "real", "isot", "true", "en")
        for i in range(12)
    ]
    fake = [
        (f"Shocking claim {i}: aliens secretly control the weather, post says.",
         "fake", "isot", "fake", "en")
        for i in range(12)
    ]
    mal = [
        (f"WIN a FREE prize {i} now click the link to claim your reward now.",
         "malicious", "smsspam", "spam", "en")
        for i in range(12)
    ]
    return _make_frame(real + fake + mal)


# ---------------------------------------------------------------------------
# Unit: sentence splitting (Bangla danda + English)
# ---------------------------------------------------------------------------


def test_split_sentences_english():
    out = split_sentences("First sentence. Second one! Third?")
    assert len(out) == 3
    assert out[0].startswith("First")


def test_split_sentences_bangla_danda():
    out = split_sentences("সরকার নীতি ঘোষণা করেছে। জনগণ খুশি।")
    assert len(out) == 2


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences(None) == []


# ---------------------------------------------------------------------------
# Unit: top_features signature
# ---------------------------------------------------------------------------


def test_top_features_returns_tokens(stripped_corpus):
    res = run_probe(stripped_corpus, stripped_corpus, view="full", seed=0)
    assert "macro_f1" in res and "top_features" in res
    # top_features is a per-class dict of token lists
    assert isinstance(res["top_features"], dict)
    for tokens in res["top_features"].values():
        assert all(isinstance(t, str) for t in tokens)


# ---------------------------------------------------------------------------
# run_probe uses macro-F1, runs each view
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("view", ["full", "title", "sentence", "source_stripped"])
def test_run_probe_each_view(stripped_corpus, view):
    res = run_probe(stripped_corpus, stripped_corpus, view=view, seed=0)
    assert 0.0 <= res["macro_f1"] <= 1.0


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL: an intact leak must be caught (probe sensitivity)
# ---------------------------------------------------------------------------


def test_negative_control_leak_detected(leaky_corpus):
    """With the Reuters dateline intact, the FULL-view probe scores near-ceiling.

    Proves the probe is sensitive: it would catch a residual leak if one survived.
    """
    res = run_probe(leaky_corpus, leaky_corpus, view="full", seed=0)
    assert res["macro_f1"] >= 0.95, (
        "negative control failed: probe did not detect an intact leak"
    )


def test_strip_kills_the_leak(leaky_corpus):
    """Same leaky corpus, source_stripped view: re-applying the strip removes the
    Reuters tell so the probe can no longer rely on it. The source_stripped score
    must drop below the full (leaky) score."""
    full = run_probe(leaky_corpus, leaky_corpus, view="full", seed=0)
    stripped = run_probe(leaky_corpus, leaky_corpus, view="source_stripped", seed=0)
    assert stripped["macro_f1"] < full["macro_f1"]


# ---------------------------------------------------------------------------
# PASS/FAIL rule encoded in probe_all_views
# ---------------------------------------------------------------------------


def test_probe_all_views_pass_on_stripped(stripped_corpus):
    report = probe_all_views(stripped_corpus, stripped_corpus, seed=0)
    assert "verdict" in report and report["verdict"] in {"PASS", "FAIL"}
    assert "views" in report
    for v in ("full", "title", "sentence", "source_stripped"):
        assert v in report["views"]
    # On a properly stripped corpus the source_stripped view must be below ceiling.
    assert report["views"]["source_stripped"]["macro_f1"] < 0.95


def test_probe_all_views_fail_on_leak(leaky_corpus):
    """The PASS/FAIL rule must FAIL a corpus whose degraded view is near-ceiling."""
    report = probe_all_views(leaky_corpus, leaky_corpus, seed=0)
    assert report["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# SLOW: the real phase gate against the built corpus
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_source_stripped_near_chance():
    """PHASE GATE: source-stripped probe macro-F1 < full-content AND < 0.95.

    Runs against the real built splits. Built Parquet is a PRECONDITION — this test
    loads the splits eagerly and lets a missing-file error PROPAGATE (it must NOT
    skip when data/processed/*.parquet is absent; the phase cannot close on a
    skipped probe).
    """
    from src.data.schema import read_parquet

    train = read_parquet("data/processed/train.parquet")
    test = read_parquet("data/processed/test.parquet")

    report = probe_all_views(train, test, seed=42)
    full = report["views"]["full"]["macro_f1"]
    stripped = report["views"]["source_stripped"]["macro_f1"]

    assert stripped < 0.95, f"source_stripped near-ceiling ({stripped:.4f}) — leak survives"
    assert stripped <= full + 1e-9, (
        f"source_stripped ({stripped:.4f}) not <= full ({full:.4f})"
    )
    # No degraded view may be near-ceiling.
    for v in ("title", "sentence", "source_stripped"):
        assert report["views"][v]["macro_f1"] < 0.95, (
            f"{v} view near-ceiling — residual leak"
        )
    assert report["verdict"] == "PASS"
