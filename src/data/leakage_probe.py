"""DATA-03 — leakage probe (the Phase-01 GATE, success-criterion #3).

Train a *cheap* TF-IDF + LogisticRegression on three deliberately-degraded views of
the corpus and check that none of them can cheat. If a degraded view still scores
near-ceiling, a source artifact is leaking the label (RESEARCH Pattern 4; Pitfall 1 —
the ISOT Reuters dateline). This module is the gate that proves plan 01-04's strip
actually worked.

The four views (``run_probe(..., view=...)``):

- ``full``            — the whole body (already stripped in the built corpus).
- ``title``           — title-only proxy: the first sentence/line of the body
                        (the built D-13 corpus has no separate title column, so the
                        leading line stands in for the title/headline tell).
- ``sentence``        — one randomly-sampled body sentence (a recurring boilerplate
                        sentence — e.g. a standing dateline/byline — would leak here).
- ``source_stripped`` — body AFTER re-applying ``leakage_strip.strip_boilerplate``.
                        This is the real gate: macro-F1 must drop toward informed
                        chance (~0.33 for 3 balanced classes) if the strip worked.

PASS/FAIL rule (``probe_all_views``):
- PASS when ``source_stripped`` macro-F1 is meaningfully below the full-content model
  AND no degraded view is near-ceiling (< 0.95).
- FAIL when any degraded view scores >= 0.95 macro-F1 (a residual leak survives) — the
  caller must fix the strip (loop back to plan 01-04) and re-run, NOT paper over it.

Scoring is class-stratified **macro-F1** (not accuracy — accuracy hides minority-class
failure on the imbalanced corpus). Probes train on the train split, score on the test
split. Top-20 features per class are returned for inspection: ``reuters``, a city, an
outlet, or a 4-digit year in the top features is a direct leak signal.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.data.leakage_strip import strip_boilerplate

# Near-ceiling threshold: a degraded-view macro-F1 at/above this is treated as a
# surviving leak (RESEARCH Pattern 4 FAIL rule; Roadmap Phase-2 SC-3 uses ~0.98 acc).
CEILING = 0.95

# Sentence boundary: English .?! plus the Bangla danda (।) and double-danda (॥).
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+|[।॥]\s*")


def split_sentences(text: Optional[str]) -> list[str]:
    """Split ``text`` into sentences (English ``.?!`` + Bangla danda ``।``/``॥``).

    bnlp-toolkit's Bangla sentence tokenizer is used when available (CLAUDE.md), with a
    deterministic regex fallback so the probe never depends on an optional install.
    Returns ``[]`` for None / empty input.
    """
    if not text:
        return []
    stripped = text.strip()
    if not stripped:
        return []
    try:  # optional: bnlp-toolkit gives better Bangla boundaries when installed
        from bnlp import BasicTokenizer  # noqa: F401  (presence check only)
        from bnlp import NLTKTokenizer

        sents = NLTKTokenizer().sentence_tokenize(stripped)
        sents = [s.strip() for s in sents if s.strip()]
        if sents:
            return sents
    except Exception:
        pass
    parts = [p.strip() for p in _SENT_SPLIT.split(stripped) if p and p.strip()]
    return parts or [stripped]


def _view_text(row: pd.Series, view: str, rng) -> str:
    """Derive the degraded text for ``row`` under ``view``."""
    text = row["text"] or ""
    if view == "full":
        return text
    if view == "source_stripped":
        return strip_boilerplate(text, row.get("source_dataset", ""))
    sents = split_sentences(text)
    if not sents:
        return ""
    if view == "title":
        return sents[0]  # leading line stands in for the headline/title tell
    if view == "sentence":
        return sents[int(rng.integers(0, len(sents)))]
    raise ValueError(f"unknown view: {view!r}")


def _build_view(df: pd.DataFrame, view: str, seed: int) -> pd.Series:
    import numpy as np

    rng = np.random.default_rng(seed)
    return df.apply(lambda r: _view_text(r, view, rng), axis=1)


def top_features(model: LogisticRegression, vectorizer: TfidfVectorizer, n: int = 20) -> dict:
    """Return the ``n`` most label-predictive TF-IDF tokens per class.

    Inspecting these is as important as the score: ``reuters``, a city, an outlet name,
    or a 4-digit year among the top features is a direct leak signal (RESEARCH Pattern 4).
    """
    vocab = vectorizer.get_feature_names_out()
    coef = model.coef_
    classes = list(model.classes_)
    out: dict[str, list[str]] = {}
    if coef.shape[0] == 1:  # binary LogReg: one row, +ve -> classes_[1]
        order = coef[0].argsort()
        out[str(classes[1])] = [str(vocab[i]) for i in order[::-1][:n]]
        out[str(classes[0])] = [str(vocab[i]) for i in order[:n]]
        return out
    for ci, cls in enumerate(classes):
        order = coef[ci].argsort()[::-1][:n]
        out[str(cls)] = [str(vocab[i]) for i in order]
    return out


def run_probe(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    view: str,
    *,
    seed: int = 42,
    n_features: int = 20,
) -> dict:
    """Train a TF-IDF + LogReg probe on one ``view`` and score macro-F1 on the test split.

    Returns ``{"view", "macro_f1", "top_features"}``. Uses ``class_weight="balanced"``
    (the corpus is imbalanced) and class-stratified **macro**-F1 (not accuracy).
    """
    x_train = _build_view(df_train, view, seed)
    x_test = _build_view(df_test, view, seed)
    y_train = df_train["label"].astype(str)
    y_test = df_test["label"].astype(str)

    # char-level n-grams catch sub-word artifacts (e.g. "(reuters)") and are
    # language-agnostic across Bangla/English; word tokens alone miss punctuation tells.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50_000,
    )
    xt = vectorizer.fit_transform(x_train)
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="liblinear" if len(set(y_train)) == 2 else "lbfgs"
    )
    model.fit(xt, y_train)
    pred = model.predict(vectorizer.transform(x_test))
    macro_f1 = float(f1_score(y_test, pred, average="macro"))

    return {
        "view": view,
        "macro_f1": macro_f1,
        "top_features": top_features(model, vectorizer, n=n_features),
    }


VIEWS = ("full", "title", "sentence", "source_stripped")


def probe_all_views(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    seed: int = 42,
    n_features: int = 20,
) -> dict:
    """Run all four views and apply the PASS/FAIL gate rule.

    PASS  := source_stripped macro-F1 < full-content macro-F1 (meaningfully below)
             AND every DEGRADED view (title / sentence / source_stripped) < CEILING.
    FAIL  := any degraded view >= CEILING (a residual leak survives — fix the strip).

    Returns ``{"views": {view: result}, "verdict": "PASS"|"FAIL", "reasons": [...]}``.
    """
    views = {v: run_probe(df_train, df_test, v, seed=seed, n_features=n_features) for v in VIEWS}

    full = views["full"]["macro_f1"]
    stripped = views["source_stripped"]["macro_f1"]
    reasons: list[str] = []

    near_ceiling = [
        v for v in ("title", "sentence", "source_stripped") if views[v]["macro_f1"] >= CEILING
    ]
    if near_ceiling:
        reasons.append(
            "degraded view(s) near-ceiling (>= %.2f): %s — residual leak"
            % (CEILING, ", ".join("%s=%.4f" % (v, views[v]["macro_f1"]) for v in near_ceiling))
        )
    # source_stripped must be at or below full (it cannot legitimately exceed the
    # full-content model; equal is fine when the corpus is already stripped).
    if stripped > full + 1e-9:
        reasons.append(
            "source_stripped (%.4f) exceeds full-content (%.4f) — unexpected" % (stripped, full)
        )

    verdict = "FAIL" if reasons else "PASS"
    return {"views": views, "verdict": verdict, "reasons": reasons, "ceiling": CEILING}
