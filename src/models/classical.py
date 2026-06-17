"""Classical ML baselines: TF-IDF features + LogReg / NaiveBayes / RandomForest.

Trains all three, evaluates on the test split, persists the best (by macro-F1)
plus all metrics. The fitted vectorizer + best model are loaded at inference time
by the fusion pipeline.

Run:  python -m src.models.classical
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB

from ..config import LABELS, load_config, path as cfg_path

PROC = cfg_path("processed_dir")
MODELS = cfg_path("models_dir")
REPORTS = cfg_path("reports_dir")


def _load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{name}.parquet")


def _build_vectorizers(cfg):
    t = cfg["classical"]["tfidf"]
    word = TfidfVectorizer(
        analyzer="word", ngram_range=tuple(t["word_ngram"]),
        max_features=t["max_features"], min_df=t["min_df"], sublinear_tf=True)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=tuple(t["char_ngram"]),
        max_features=t["max_features"] // 2, min_df=t["min_df"])
    return word, char


def _vectorize(word, char, texts, fit: bool):
    if fit:
        xw = word.fit_transform(texts)
        xc = char.fit_transform(texts)
    else:
        xw = word.transform(texts)
        xc = char.transform(texts)
    return hstack([xw, xc]).tocsr()


def train() -> dict:
    cfg = load_config()
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    train_df, test_df = _load_split("train"), _load_split("test")
    word, char = _build_vectorizers(cfg)
    Xtr = _vectorize(word, char, train_df["text"], fit=True)
    Xte = _vectorize(word, char, test_df["text"], fit=False)
    ytr, yte = train_df["label"], test_df["label"]

    cw = cfg["classical"]["class_weight"]
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight=cw, n_jobs=-1),
        "naive_bayes": MultinomialNB(),                # NB ignores class_weight
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight=cw, n_jobs=-1, random_state=cfg["data"]["random_state"]),
    }

    results, best_name, best_f1, best_model = {}, None, -1.0, None
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        rep = classification_report(yte, pred, labels=LABELS, output_dict=True, zero_division=0)
        macro_f1 = f1_score(yte, pred, labels=LABELS, average="macro", zero_division=0)
        results[name] = {"macro_f1": macro_f1, "report": rep,
                         "confusion_matrix": confusion_matrix(yte, pred, labels=LABELS).tolist()}
        print(f"{name:22s}  macro-F1 = {macro_f1:.4f}  acc = {rep['accuracy']:.4f}")
        if macro_f1 > best_f1:
            best_name, best_f1, best_model = name, macro_f1, clf

    # Persist artifacts for inference.
    joblib.dump({"word": word, "char": char}, MODELS / "tfidf_vectorizers.joblib")
    joblib.dump(best_model, MODELS / "classical_best.joblib")
    (MODELS / "classical_best.txt").write_text(best_name, encoding="utf-8")
    (REPORTS / "classical_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBest classical model: {best_name} (macro-F1={best_f1:.4f}) -> models/classical_best.joblib")
    return results


# ── Inference helper used by the fusion pipeline ────────────────────────────
class ClassicalClassifier:
    """Loads persisted vectorizers + best model; returns label + class probabilities."""

    def __init__(self) -> None:
        vec = joblib.load(MODELS / "tfidf_vectorizers.joblib")
        self.word, self.char = vec["word"], vec["char"]
        self.model = joblib.load(MODELS / "classical_best.joblib")
        self.name = (MODELS / "classical_best.txt").read_text(encoding="utf-8").strip()

    def predict_proba(self, text: str) -> dict:
        from .classical import _vectorize  # reuse identical feature build
        X = _vectorize(self.word, self.char, [text], fit=False)
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        return {lab: float(proba[classes.index(lab)]) for lab in LABELS}


if __name__ == "__main__":
    train()
