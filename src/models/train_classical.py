"""CLS-01 / CLS-03 — Phase 2 classical-baseline training orchestrator.

This is the Phase-2 integration point: it wires the Wave-1 building blocks
(:mod:`src.models.vectorizer`, :mod:`src.models.metrics`, :mod:`src.models.leakage_recheck`)
into the single ``models/*.joblib`` + ``reports/classical_baselines_report.md`` contract
that Phase 3 consumes as the classical baseline to beat.

Pipeline (leak-safe ORDER):

    1. load data/processed/{train,val,test}.parquet            (schema.read_parquet)
    2. fit the shared hybrid TF-IDF vectorizer ONCE on TRAIN    (build_vectorizer; D-02)
       — never fit on val/test (leak-safe).
    3. transform val + test with the train-fitted vectorizer.
    4. for each model in MODELS: fit on TRAIN features, score val + test
       (macro_f1, per_class_report, confusion, per_language_macro_f1, minority_recall).
    5. run the SC-3 leakage re-check on every fitted model (recheck_model).
    6. select_best by VAL macro-F1 with the D-03 minority guard.
    7. serialize the train-fitted vectorizer + best model to models/ (load-only reuse).
    8. write_report -> reports/classical_baselines_report.md (counts/metrics only).

Policy (CONTEXT decisions):
  * D-01 — imbalance is handled by ``class_weight="balanced"`` (LR/RF) + ``ComplementNB``
    ONLY. No synthetic-oversampling library is used anywhere in this module.
  * D-03 — best model = VAL macro-F1 with a minority-class guard (a model whose fake- or
    malicious-class recall collapsed cannot win; ties break toward higher minority recall;
    CPU latency is a secondary tiebreaker only). Final numbers reported on the held-out TEST.
  * D-04 — the BFN2 satire-vs-fake assumption is recorded as a caveat in the report.

CLI: ``python -m src.models.train_classical``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

from src.data.label_map import LABELS
from src.data.schema import read_parquet
from src.models.leakage_recheck import INVESTIGATE_THRESHOLD, recheck_model
from src.models.metrics import (
    confusion,
    macro_f1,
    minority_recall,
    per_class_report,
    per_language_macro_f1,
)
from src.models.vectorizer import build_vectorizer, texts_from_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train_classical")

# Repo root = two levels up from src/models/train_classical.py.
REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

# Recorded vectorizer hyperparameters (the chosen D-02 defaults; structure is locked).
VECTORIZER_HPARAMS = {
    "word_ngram_range": "(1, 2)",
    "char_wb_ngram_range": "(3, 5)",
    "min_df": 2,
    "max_features": 50_000,
}

# D-01 (LOCKED): LR/RF use class_weight="balanced"; NB uses ComplementNB (handles TF-IDF
# skew natively). No synthetic oversampling — balancing is class_weight only.
MODELS = {
    "logreg": LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs"),
    "complement_nb": ComplementNB(),
    "random_forest": RandomForestClassifier(
        class_weight="balanced", random_state=SEED, n_estimators=200, n_jobs=-1
    ),
}

# Minority classes whose collapsed recall disqualifies a model under the D-03 guard.
_MINORITY_CLASSES = ("fake", "malicious")
# Recall at/under this floor counts as "collapsed" for the D-03 guard.
_MINORITY_FLOOR = 1e-9
# val macro-F1 difference within this band is treated as a tie (break toward minority recall).
_TIE_BAND = 1e-4


# ---------------------------------------------------------------------------
# Model freshness — sklearn estimators are mutable; clone before each fit so a
# re-run never trains on top of an already-fitted estimator.
# ---------------------------------------------------------------------------
def _fresh_models() -> dict:
    from sklearn.base import clone

    return {name: clone(est) for name, est in MODELS.items()}


def _confusion_to_lists(cm) -> list[list[int]]:
    return [[int(v) for v in row] for row in cm]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def train_and_compare(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    """Fit the shared vectorizer once on TRAIN, train every model, score val+test,
    run the SC-3 re-check, and select the best model (D-03).

    Returns a dict::

        {
          "vectorizer": <fitted FeatureUnion>,
          "fitted_models": {name: <fitted estimator>},
          "models": {name: {val_macro_f1, test_macro_f1, per_class, confusion,
                            per_language_macro_f1, minority_recall, leakage_recheck}},
          "best": <name>,
          "any_investigate": <bool>,
          "hparams": {...},
        }

    Leak-safe: the vectorizer is fit on TRAIN text ONLY; val/test are transformed with the
    train-fitted vectorizer (never fit on val/test).
    """
    # 2. Fit the shared hybrid TF-IDF vectorizer ONCE on TRAIN only (leak-safe).
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(texts_from_frame(train_df))
    y_train = list(train_df["label"])
    log.info("fitted vectorizer on %d train rows -> %d features", len(y_train), X_train.shape[1])

    # 3. Transform val + test with the train-fitted vectorizer.
    X_val = vectorizer.transform(texts_from_frame(val_df))
    X_test = vectorizer.transform(texts_from_frame(test_df))
    y_val = list(val_df["label"])
    y_test = list(test_df["label"])

    feature_names = vectorizer.get_feature_names_out()

    fitted = _fresh_models()
    models_out: dict = {}
    any_investigate = False

    for name, est in fitted.items():
        # 4. fit on TRAIN, score val + test.
        est.fit(X_train, y_train)
        val_pred = est.predict(X_val)
        test_pred = est.predict(X_test)

        val_f1 = macro_f1(y_val, val_pred)
        test_f1 = macro_f1(y_test, test_pred)

        # 5. SC-3 leakage re-check on each fitted model (score = test macro-F1).
        recheck = recheck_model(est, feature_names, test_f1)
        any_investigate = any_investigate or bool(recheck["investigate"])

        models_out[name] = {
            "val_macro_f1": val_f1,
            "test_macro_f1": test_f1,
            "per_class": per_class_report(y_test, test_pred),
            "confusion": _confusion_to_lists(confusion(y_test, test_pred)),
            "per_language_macro_f1": per_language_macro_f1(test_df, y_test, test_pred),
            "minority_recall": minority_recall(y_test, test_pred),
            "leakage_recheck": recheck,
        }
        log.info(
            "%s: val_macro_f1=%.4f test_macro_f1=%.4f investigate=%s",
            name, val_f1, test_f1, recheck["investigate"],
        )

    best = select_best(models_out)
    log.info("selected best model (D-03): %s", best)

    return {
        "vectorizer": vectorizer,
        "fitted_models": fitted,
        "models": models_out,
        "best": best,
        "any_investigate": any_investigate,
        "hparams": dict(VECTORIZER_HPARAMS),
    }


def select_best(results: dict) -> str:
    """Pick the best model by VAL macro-F1 with the D-03 minority-class guard.

    Rules (D-03):
      * A model whose ``fake`` or ``malicious`` recall has collapsed (<= a small floor)
        is DISQUALIFIED — it cannot win even with a marginally higher macro-F1.
      * Among qualified models, rank by VAL macro-F1; ties / near-ties (within ``_TIE_BAND``)
        break toward higher minority recall (mean of fake + malicious recall).
      * If EVERY model has collapsed minority recall, fall back to ranking all models
        (so a winner is always returned), still preferring higher minority recall on ties.
    """

    def _min_recall(rec: dict) -> float:
        mr = rec["minority_recall"]
        return sum(mr[c] for c in _MINORITY_CLASSES) / len(_MINORITY_CLASSES)

    def _collapsed(rec: dict) -> bool:
        mr = rec["minority_recall"]
        return any(mr[c] <= _MINORITY_FLOOR for c in _MINORITY_CLASSES)

    qualified = {n: r for n, r in results.items() if not _collapsed(r)}
    pool = qualified or results  # never return empty

    # Rank by EXACT val macro-F1 (minority recall as a stable secondary key), then resolve
    # GENUINE near-ties with an explicit distance check against the top model (WR-02). The
    # previous round(f1 / _TIE_BAND) binning depended on proximity to bin boundaries, not
    # actual distance, so the D-03 minority tiebreak could silently fail to fire for two
    # models a hair apart that straddled a bin edge.
    ranked = sorted(
        pool,
        key=lambda n: (pool[n]["val_macro_f1"], _min_recall(pool[n])),
        reverse=True,
    )
    top = ranked[0]
    ties = [n for n in ranked if abs(pool[n]["val_macro_f1"] - pool[top]["val_macro_f1"]) <= _TIE_BAND]
    return max(ties, key=lambda n: _min_recall(pool[n]))


# ---------------------------------------------------------------------------
# Serialization (SC-1) — mirror schema.write_parquet's mkdir-then-persist shape.
# ---------------------------------------------------------------------------
def serialize_artifacts(vectorizer, best_model, models_dir: str | Path) -> None:
    """Dump the fitted vectorizer + best model to ``models_dir`` via joblib.

    Mirrors ``schema.write_parquet``: ``mkdir(parents=True, exist_ok=True)`` then persist.
    Writes ``vectorizer.joblib`` and ``best_model.joblib``.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, models_dir / "vectorizer.joblib")
    joblib.dump(best_model, models_dir / "best_model.joblib")
    log.info("serialized artifacts -> %s", models_dir)


def load_artifacts(models_dir: str | Path):
    """Reload ``(vectorizer, best_model)`` from ``models_dir`` — LOAD ONLY (SC-1).

    No training code path: this helper only ``joblib.load``s the trusted, project-anchored
    artifacts (T-02-05: never accepts an arbitrary external path from user input).
    """
    models_dir = Path(models_dir)
    vectorizer = joblib.load(models_dir / "vectorizer.joblib")
    best_model = joblib.load(models_dir / "best_model.joblib")
    return vectorizer, best_model


# ---------------------------------------------------------------------------
# Reporting (counts/metrics only — SYS-02: NO raw article text). [Task 2]
# ---------------------------------------------------------------------------
def _per_class_table(per_class: dict) -> str:
    """Render per-class precision/recall/f1 in LABELS order as a markdown table."""
    lines = ["| class | precision | recall | f1 |", "|---|---|---|---|"]
    for label in LABELS:
        row = per_class.get(label, {})
        lines.append(
            f"| {label} | {row.get('precision', 0.0):.4f} | "
            f"{row.get('recall', 0.0):.4f} | {row.get('f1-score', 0.0):.4f} |"
        )
    return "\n".join(lines)


def _confusion_table(cm: list[list[int]]) -> str:
    """Render a 3x3 confusion matrix (rows=true, cols=pred) in LABELS order."""
    header = "| true \\ pred | " + " | ".join(LABELS) + " |"
    sep = "|---|" + "|".join(["---"] * len(LABELS)) + "|"
    lines = [header, sep]
    for i, label in enumerate(LABELS):
        vals = " | ".join(str(int(cm[i][j])) for j in range(len(LABELS)))
        lines.append(f"| {label} | {vals} |")
    return "\n".join(lines)


def _per_language_table(per_lang: dict) -> str:
    """Render the per-language macro-F1 table (bn/en/code-mixed where present)."""
    lines = ["| language | macro-F1 |", "|---|---|"]
    for lang in sorted(per_lang):
        lines.append(f"| {lang} | {per_lang[lang]:.4f} |")
    if len(lines) == 2:
        lines.append("| _(no language groups present)_ | - |")
    return "\n".join(lines)


def write_report(results: dict, report_path: str | Path) -> None:
    """Emit ``reports/classical_baselines_report.md`` — COUNTS / METRICS ONLY (SYS-02).

    Records (SC-4 + D-03 + D-04): per-model macro-F1 (headline) + per-class P/R/F1 (test)
    + confusion matrix per model + per-language macro-F1; the selected best model + its
    test metrics + selection rationale; recorded hyperparameters; the SC-3 leakage re-check
    result; and the BFN2 satire-vs-fake caveat (D-04). No raw article text.
    """
    report_path = Path(report_path)
    models_out = results["models"]
    best = results["best"]
    hp = results["hparams"]

    parts = [
        "# Classical Baselines Report -- Phase 02",
        "",
        "Deterministic, re-runnable training of the Phase-2 classical baselines "
        "(TF-IDF + LogisticRegression / ComplementNB / RandomForest). **Counts / metrics "
        "only -- no raw article text** is recorded here (SYS-02). The fitted "
        "`models/*.joblib` artifacts are gitignored; rebuild with the command below.",
        "",
        "## Build command (deterministic, seed=42)",
        "",
        "```bash",
        "python -m src.models.train_classical",
        "```",
        "",
        "Writes `models/vectorizer.joblib` + `models/best_model.joblib` (load-only reuse) and "
        "this report. **Macro-F1 is the headline metric** -- accuracy is never the headline (SC-2).",
        "",
        "## Per-model macro-F1 (val) + per-class precision/recall/F1 (test)",
        "",
    ]

    summary = ["| model | val macro-F1 | test macro-F1 |", "|---|---|---|"]
    for name in MODELS:
        m = models_out[name]
        summary.append(f"| {name} | {m['val_macro_f1']:.4f} | {m['test_macro_f1']:.4f} |")
    parts += ["\n".join(summary), ""]

    for name in MODELS:
        m = models_out[name]
        parts += [
            f"### {name}",
            "",
            f"- Macro-F1 (val): **{m['val_macro_f1']:.4f}** ; Macro-F1 (test): **{m['test_macro_f1']:.4f}**",
            "",
            "Per-class precision / recall / f1 (test):",
            "",
            _per_class_table(m["per_class"]),
            "",
            "Confusion matrix (test, rows=true, cols=pred, LABELS order):",
            "",
            _confusion_table(m["confusion"]),
            "",
            "Per-language macro-F1 (test):",
            "",
            _per_language_table(m["per_language_macro_f1"]),
            "",
        ]

    best_m = models_out[best]
    parts += [
        "## Selected best model",
        "",
        f"**{best}** -- selected by **validation macro-F1** with the D-03 minority-class guard "
        "(a model whose fake- or malicious-class recall has collapsed cannot win; ties break "
        "toward higher minority recall; CPU latency is a secondary tiebreaker only).",
        "",
        f"- Test macro-F1: **{best_m['test_macro_f1']:.4f}**",
        "",
        "Selected-model per-class precision / recall / f1 (test):",
        "",
        _per_class_table(best_m["per_class"]),
        "",
        "## Recorded hyperparameters",
        "",
        "| component | setting |",
        "|---|---|",
        "| LogisticRegression | solver=lbfgs, max_iter=1000, class_weight=balanced |",
        "| ComplementNB | defaults (handles TF-IDF skew natively; NOT MultinomialNB) |",
        f"| RandomForest | n_estimators=200, class_weight=balanced, random_state={SEED} |",
        f"| TF-IDF word view | analyzer=word, ngram_range={hp['word_ngram_range']} |",
        f"| TF-IDF char view | analyzer=char_wb, ngram_range={hp['char_wb_ngram_range']} |",
        f"| TF-IDF shared | min_df={hp['min_df']}, max_features={hp['max_features']}, lowercase=True |",
        "",
        "Imbalance handling (D-01): `class_weight=\"balanced\"` (LR/RF) + ComplementNB only -- "
        "**no synthetic oversampling**.",
        "",
        "## Leakage re-check (SC-3)",
        "",
        f"Each fitted model's top features are inspected for outlet/dateline/year tells (reused "
        f"Phase-1 `_LEAK_TELL` regex). Any model scoring >= {INVESTIGATE_THRESHOLD:.2f} (macro-F1) "
        "is flagged for suspected-leakage investigation before being trusted.",
        "",
    ]
    recheck_lines = ["| model | >= 0.98 | leak tells | investigate |", "|---|---|---|---|"]
    for name in MODELS:
        rc = models_out[name]["leakage_recheck"]
        ge98 = "yes" if rc["score"] >= INVESTIGATE_THRESHOLD else "no"
        tells = ", ".join(rc["leak_tells"]) if rc["leak_tells"] else "none"
        recheck_lines.append(
            f"| {name} | {ge98} | {tells} | {'yes' if rc['investigate'] else 'no'} |"
        )
    parts += ["\n".join(recheck_lines), ""]
    parts += [
        (
            "**No model was flagged for investigation** -- no outlet/year tells in top features "
            "and no model reached the 0.98 ceiling."
            if not results["any_investigate"]
            else "**At least one model was flagged** -- see the table above; investigate before trusting."
        ),
        "",
        "## Caveats",
        "",
        "- **BFN2 satire-vs-fake assumption (D-04):** the Phase-1 corpus maps BanFakeNews-2.0 "
        "(BFN2) `Label` 0/1/2 -> `fake` and 3 -> `real`, so **satire is treated as `fake`**. This "
        "is a documented known assumption: BFN2 `real` rows are ~100% SHA1-duplicates of "
        "BanFakeNews v1 and dedup away, so the practical impact on fake-class metrics is small. "
        "Revisit only if fake-class error analysis surfaces satire confusion.",
        "- **Malicious class is English-only (Phase-1 D-01):** a documented coverage limitation, "
        "visible in the per-language macro-F1 tables above.",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(parts), encoding="utf-8")
    log.info("wrote report -> %s", report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    """Train LR/NB/RF on data/processed/*.parquet, serialize the best, write the report."""
    train = read_parquet(REPO_ROOT / "data/processed/train.parquet")
    val = read_parquet(REPO_ROOT / "data/processed/val.parquet")
    test = read_parquet(REPO_ROOT / "data/processed/test.parquet")

    results = train_and_compare(train, val, test)
    serialize_artifacts(
        results["vectorizer"], results["fitted_models"][results["best"]], REPO_ROOT / "models"
    )
    write_report(results, REPO_ROOT / "reports" / "classical_baselines_report.md")
    log.info("train_classical complete: best=%s", results["best"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
