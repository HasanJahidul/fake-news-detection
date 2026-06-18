  1. TF-IDF + Logistic Regression, Naive Bayes, and Random Forest are trained on the Phase 1 corpus and the fitted vectorizer + best model are serialized to `models/` for load-only reuse.
  2. Macro-F1, per-class precision/recall, and a confusion matrix are reported per model on the held-out split (accuracy is never the headline metric); minority-class (fake/malicious) recall is non-trivial.
  3. Top predictive features are inspected and confirmed not to be outlet names / datelines / years; any model scoring greater than or equal to 98% is investigated as suspected leakage before being trusted.
  4. A model-comparison report is written to `reports/` and the best classical model is recorded with its metrics.

**Plans**: 3 plans (2 waves)

  - [x] 02-01-PLAN.md — Wave 1: hybrid word+char TF-IDF vectorizer (D-02) + metric-discipline module (macro-F1/per-class/confusion/per-language/minority-guard) (CLS-01, CLS-03)
  - [x] 02-02-PLAN.md — Wave 1: SC-3 leakage re-check reusing leakage_probe leak-tells across LR/NB/RF + >=0.98 investigation flag (CLS-03)
  - [ ] 02-03-PLAN.md — Wave 2: train/select/serialize orchestrator (D-01/D-03) + models/ joblib artifacts + comparison report with BFN2 caveat (CLS-01, CLS-03)

  **Wave structure:** W1 {02-01, 02-02} → W2 {02-03}
