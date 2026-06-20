"""Early classical-only demo: text in -> explained verdict + confidence out.

This is an EARLY, classical-baseline-only preview of the PMICS capstone
"AI-Driven Real-Time Detection of Fake News and Malicious Content" system.
Only the classical model exists today (a TF-IDF + LogisticRegression baseline
shipped as models/best_model.joblib). The full system (multilingual transformer,
source-credibility / writing-style / external-verification signals, fusion, and
the Streamlit dashboard) is FUTURE work (Phase 3+ / Phase 7).

What this script does:
  raw text
   -> src.preprocess.preprocess()        # the ONE shared normalize+clean used at train time
   -> vectorizer.transform([clean])      # train-fitted hybrid word+char TF-IDF (FeatureUnion)
   -> model.predict_proba(X)[0]          # per-class probabilities
   -> argmax -> model.classes_[idx]      # label order comes from model.classes_, never hardcoded
   -> top word-view contributors         # coef_ * tfidf, for a human-readable "why"

Usage (run from the project root, with the project venv):
    .venv/bin/python demo_predict.py                 # classify built-in curated examples
    .venv/bin/python demo_predict.py --text "..."    # classify one custom input

Dependencies: only joblib + numpy (already in the project venv) and the project's
own modules. Importing src.preprocess pulls in csebuetnlp/normalizer (git-installed,
present in .venv) — run under that environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

# MANDATORY: the single shared preprocessing contract (D-08). Train AND inference
# call this exact function — skipping it causes a silent train/inference mismatch.
from src.preprocess import preprocess

# Anchor paths to this file's directory so the demo works regardless of cwd.
_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _ROOT / "models"

# FeatureUnion sub-transformer prefix for the human-readable word view. The fitted
# union names features "word__<token>" and "char__<token>"; we show only word-view
# tokens (char n-gram fragments are meaningless to a human reader).
_WORD_PREFIX = "word__"

# Plain-language framing per verdict (transparency is the whole point of the brief).
_VERDICT_BLURB = {
    "real": "looks like genuine/authentic content",
    "fake": "looks like fake / fabricated news",
    "malicious": "looks like malicious content (phishing / scam / spam)",
}

_DISCLAIMER = (
    "Note: automated estimate from an early classical model only — "
    "NOT the final system, and not a definitive judgement."
)


def load_artifacts():
    """Load the two SEPARATE joblib artifacts.

    vectorizer.joblib -> fitted sklearn FeatureUnion (hybrid word+char TF-IDF).
    best_model.joblib -> bare classifier (currently LogisticRegression) with
    predict_proba. The model alone cannot consume raw text; you must transform
    with the vectorizer first.
    """
    vectorizer = joblib.load(_MODELS_DIR / "vectorizer.joblib")
    model = joblib.load(_MODELS_DIR / "best_model.joblib")
    return vectorizer, model


def classify(text, vectorizer, model):
    """Return (label, confidence, proba_by_class) for one input string.

    Label order is taken from model.classes_ (sklearn sorts labels
    lexicographically -> ['fake','malicious','real'], which is NOT the project's
    LABELS order). predict_proba column j aligns to model.classes_[j], so we map
    the argmax through model.classes_ and never hardcode positions.
    """
    clean = preprocess(text)  # raw -> normalized clean text; "" for None/empty/whitespace
    X = vectorizer.transform([clean])  # bare vectorizer needs its own transform; pass a LIST
    proba = model.predict_proba(X)[0]  # per-class probabilities, aligned to model.classes_
    idx = int(np.argmax(proba))
    label = str(model.classes_[idx])
    confidence = float(proba[idx])
    proba_by_class = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    return label, confidence, proba_by_class


def explain_top_words(text, label, vectorizer, model, top_n=6):
    """Top word-view tokens that pushed THIS input toward its predicted class.

    Linear-model explanation: contribution = coef_[class_row] * tfidf_value, where
    the class row index comes from model.classes_ (NOT the LABELS order). We keep
    only the "word__" view (drop char__ n-gram fragments) for readability.

    Returns a list of (token, contribution) sorted descending, or [] if the model
    is not linear (no coef_) or the input has no in-vocab word tokens — in which
    case the caller omits the explanation gracefully.
    """
    # Only valid for linear models exposing signed per-feature coefficients.
    # The shipped best model is LogisticRegression; a retrain could in principle
    # pick ComplementNB / RandomForest, which have no comparable signed per-input
    # push — so we degrade gracefully rather than show a misleading "why".
    if not hasattr(model, "coef_"):
        return []

    clean = preprocess(text)
    X = vectorizer.transform([clean]).tocsr()
    if X.nnz == 0:
        return []

    cls_idx = list(model.classes_).index(label)  # row index for THIS class's coefficients
    coef_row = model.coef_[cls_idx]

    names = np.asarray(vectorizer.get_feature_names_out())
    row = X[0]
    idx = row.indices  # feature indices nonzero for THIS input
    tfidf = row.data  # their TF-IDF values
    contrib = coef_row[idx] * tfidf  # per-feature push toward the predicted class

    pairs = [
        (names[i][len(_WORD_PREFIX):], float(c))  # strip "word__" for display
        for i, c in zip(idx, contrib)
        if names[i].startswith(_WORD_PREFIX) and c > 0  # word view only, positive push
    ]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[:top_n]


# Curated examples covering each class + both languages the brief targets.
_EXAMPLES = [
    # English, genuine-news style
    "The central bank held its benchmark interest rate steady on Tuesday, "
    "citing easing inflation and steady employment figures reported this quarter.",
    # English, fake / sensational
    "SHOCKING: Scientists CONFIRM the moon is secretly made of cheese and the "
    "government has been hiding the truth from you for decades!!!",
    # Bangla, a plain news-style sentence (primary target language)
    "প্রধানমন্ত্রী আজ এক অনুষ্ঠানে নতুন একটি সেতু উদ্বোধন করেছেন এবং উন্নয়ন প্রকল্প নিয়ে কথা বলেছেন।",
    # Phishing SMS asking for credentials / OTP
    "Your bank account is locked. Verify now at http://secure-login-bank.ru "
    "and enter your PIN and OTP within 2 hours or it will be permanently closed.",
    # Lottery / scam message
    "CONGRATULATIONS! You have WON $1,000,000 in the international lottery. "
    "Send your full name, address and bank details to claim your prize today!",
    # English, mundane real-news style (the hard real-vs-fake pair)
    "Officials said the new bridge is expected to open to traffic next month "
    "after final safety inspections are completed by the transport authority.",
]


def _truncate(text, width=58):
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _format_top_words(top_words):
    if not top_words:
        return "(no recognizable word features)"
    return ", ".join(f"{tok} (+{c:.2f})" for tok, c in top_words)


def _print_one(text, vectorizer, model):
    label, confidence, proba_by_class = classify(text, vectorizer, model)
    top_words = explain_top_words(text, label, vectorizer, model)
    blurb = _VERDICT_BLURB.get(label, label)

    print(f"  INPUT      : {_truncate(text)}")
    print(f"  VERDICT    : {label.upper()}  ({blurb})")
    print(f"  CONFIDENCE : {confidence * 100:5.1f}%")
    proba_str = "  ".join(
        f"{c}={p * 100:.1f}%" for c, p in sorted(proba_by_class.items())
    )
    print(f"  ALL CLASSES: {proba_str}")
    print(f"  TOP WORDS  : {_format_top_words(top_words)}")


def main():
    parser = argparse.ArgumentParser(
        description="Early classical-only fake-news / malicious-content demo."
    )
    parser.add_argument(
        "--text",
        help="Classify this single input instead of the built-in examples.",
    )
    args = parser.parse_args()

    vectorizer, model = load_artifacts()

    print("=" * 72)
    print("Fake News / Malicious Content Detector — early classical-model demo")
    print("Classes:", ", ".join(str(c) for c in model.classes_))
    print(_DISCLAIMER)
    print("=" * 72)

    if args.text is not None:
        print()
        _print_one(args.text, vectorizer, model)
        print()
        return

    for i, example in enumerate(_EXAMPLES, start=1):
        print()
        print(f"[Example {i}]")
        _print_one(example, vectorizer, model)

    print()
    print("-" * 72)
    print(_DISCLAIMER)


if __name__ == "__main__":
    main()
