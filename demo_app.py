"""Minimal Streamlit preview of the fake-news / malicious-content detector.

This is an EARLY, classical-model-only preview of the PMICS capstone system —
the closest thing so far to "the app". A user pastes text and gets an explained
real / fake / malicious verdict with a confidence score.

Only the classical TF-IDF + LogisticRegression baseline exists today
(models/best_model.joblib + models/vectorizer.joblib). The FULL Phase-7 UI —
URL input + article extraction, the multilingual transformer, the source /
writing-style / external-verification signals, and the fused multi-signal
explanation — is future work and is NOT in this preview.

Run (from the project root, with the project venv that has streamlit + normalizer):
    .venv/bin/python -m streamlit run demo_app.py
If streamlit is not installed in the venv:  .venv/bin/pip install streamlit

Pipeline (identical contract to demo_predict.py and to training):
  raw text -> src.preprocess.preprocess() -> vectorizer.transform([clean])
           -> model.predict_proba -> argmax via model.classes_ (never hardcoded)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# MANDATORY shared preprocessing contract (D-08): the SAME normalize+clean used at
# train time. Skipping it would silently break the train/inference feature match.
from src.preprocess import preprocess

# Anchor to this file's directory so paths resolve regardless of how streamlit is launched.
_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _ROOT / "models"

# FeatureUnion word-view prefix for the human-readable explanation.
_WORD_PREFIX = "word__"

_VERDICT_BLURB = {
    "real": "Looks like genuine / authentic content.",
    "fake": "Looks like fake / fabricated news.",
    "malicious": "Looks like malicious content (phishing / scam / spam).",
}


@st.cache_resource
def load_artifacts():
    """Load and cache the two SEPARATE joblib artifacts once per session.

    vectorizer.joblib -> fitted sklearn FeatureUnion (hybrid word+char TF-IDF).
    best_model.joblib -> bare classifier (currently LogisticRegression) with
    predict_proba. Cached so the model loads once (the "real-time feel" pattern).
    """
    vectorizer = joblib.load(_MODELS_DIR / "vectorizer.joblib")
    model = joblib.load(_MODELS_DIR / "best_model.joblib")
    return vectorizer, model


def classify(text, vectorizer, model):
    """Return (label, confidence, proba_by_class).

    Label order comes from model.classes_ (sklearn lexicographic sort:
    ['fake','malicious','real']) — NOT the project LABELS order. predict_proba
    column j aligns to model.classes_[j]; we map argmax through it, never hardcode.
    """
    clean = preprocess(text)  # raw -> normalized clean text; "" for empty input
    X = vectorizer.transform([clean])  # bare vectorizer needs its own transform; pass a LIST
    proba = model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label = str(model.classes_[idx])
    confidence = float(proba[idx])
    proba_by_class = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    return label, confidence, proba_by_class


def explain_top_words(text, label, vectorizer, model, top_n=8):
    """Top word-view tokens that pushed THIS input toward its predicted class.

    contribution = coef_[class_row] * tfidf, with the class row from
    model.classes_ (NOT LABELS order). Word view only. Returns [] when the model
    is non-linear (no coef_) or there are no in-vocab word tokens, so the UI can
    omit the explanation gracefully.
    """
    if not hasattr(model, "coef_"):
        return []

    clean = preprocess(text)
    X = vectorizer.transform([clean]).tocsr()
    if X.nnz == 0:
        return []

    cls_idx = list(model.classes_).index(label)
    coef_row = model.coef_[cls_idx]

    names = np.asarray(vectorizer.get_feature_names_out())
    row = X[0]
    idx = row.indices
    tfidf = row.data
    contrib = coef_row[idx] * tfidf

    pairs = [
        (names[i][len(_WORD_PREFIX):], float(c))
        for i, c in zip(idx, contrib)
        if names[i].startswith(_WORD_PREFIX) and c > 0
    ]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[:top_n]


def main():
    st.set_page_config(page_title="Fake News / Malicious Content Detector", page_icon="🛡️")

    st.title("Fake News / Malicious Content Detector")
    st.caption(
        "Early preview — classical model only. Paste text to get an explained "
        "real / fake / malicious verdict. This is NOT the final system."
    )

    vectorizer, model = load_artifacts()
    classes = [str(c) for c in model.classes_]

    default_text = (
        "Your bank account is locked. Verify now at http://secure-login-bank.ru "
        "and enter your PIN and OTP within 2 hours or it will be permanently closed."
    )
    text = st.text_area(
        "Text to analyze (English or Bangla)",
        value=default_text,
        height=160,
        help="Paste a news snippet, message, or SMS. URL extraction is future work.",
    )

    analyze = st.button("Analyze", type="primary")

    if analyze:
        if not preprocess(text):
            st.warning("Please enter some non-empty text to analyze.")
        else:
            label, confidence, proba_by_class = classify(text, vectorizer, model)
            blurb = _VERDICT_BLURB.get(label, "")

            st.subheader(f"Verdict: {label.upper()}")
            if blurb:
                st.write(blurb)

            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.progress(min(max(confidence, 0.0), 1.0))

            st.markdown("**Per-class probabilities**")
            for cls in sorted(proba_by_class):
                p = proba_by_class[cls]
                st.write(f"{cls}: {p * 100:.1f}%")
                st.progress(min(max(p, 0.0), 1.0))

            st.markdown("**Top contributing words (why)**")
            top_words = explain_top_words(text, label, vectorizer, model)
            if top_words:
                st.write(
                    "  ".join(f"`{tok}` (+{c:.2f})" for tok, c in top_words)
                )
                st.caption(
                    "Word-view TF-IDF features that pushed this text toward the "
                    "predicted class (coefficient × term weight). Character n-gram "
                    "features also influence the verdict but are not shown."
                )
            else:
                st.write("_No recognizable word features to explain this input._")

    st.divider()
    st.info(
        "Scope: this preview classifies pasted text with a single classical model "
        f"({type(model).__name__}, classes: {', '.join(classes)}). "
        "The full Phase-7 system — URL input + article extraction, a multilingual "
        "transformer, source-credibility / writing-style / external-verification "
        "signals, and a fused multi-signal explanation — is future work. "
        "Known limits today: Bangla is the weakest language and all malicious "
        "training data is English-only, so Bangla scam/phishing is not covered. "
        "Verdicts are automated estimates, not definitive judgements."
    )


if __name__ == "__main__":
    main()
