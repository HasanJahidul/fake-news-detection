# Classical Baselines Report -- Phase 02

Deterministic, re-runnable training of the Phase-2 classical baselines (TF-IDF + LogisticRegression / ComplementNB / RandomForest). **Counts / metrics only -- no raw article text** is recorded here (SYS-02). The fitted `models/*.joblib` artifacts are gitignored; rebuild with the command below.

## Build command (deterministic, seed=42)

```bash
python -m src.models.train_classical
```

Writes `models/vectorizer.joblib` + `models/best_model.joblib` (load-only reuse) and this report. **Macro-F1 is the headline metric** -- accuracy is never the headline (SC-2).

## Per-model macro-F1 (val) + per-class precision/recall/F1 (test)

| model | val macro-F1 | test macro-F1 |
|---|---|---|
| logreg | 0.9181 | 0.9140 |
| complement_nb | 0.7040 | 0.6080 |
| random_forest | 0.9057 | 0.9064 |

### logreg

- Macro-F1 (val): **0.9181** ; Macro-F1 (test): **0.9140**

Per-class precision / recall / f1 (test):

| class | precision | recall | f1 |
|---|---|---|---|
| real | 0.8767 | 0.8748 | 0.8757 |
| fake | 0.8667 | 0.8702 | 0.8684 |
| malicious | 0.9988 | 0.9969 | 0.9979 |

Confusion matrix (test, rows=true, cols=pred, LABELS order):

| true \ pred | real | fake | malicious |
|---|---|---|---|
| real | 4556 | 646 | 6 |
| fake | 637 | 4271 | 0 |
| malicious | 4 | 11 | 4877 |

Per-language macro-F1 (test):

| language | macro-F1 |
|---|---|
| bn | 0.7741 |
| code-mixed | 1.0000 |
| en | 0.9357 |

### complement_nb

- Macro-F1 (val): **0.7040** ; Macro-F1 (test): **0.6080**

Per-class precision / recall / f1 (test):

| class | precision | recall | f1 |
|---|---|---|---|
| real | 0.5245 | 0.2738 | 0.3598 |
| fake | 0.5813 | 0.6642 | 0.6200 |
| malicious | 0.7312 | 0.9986 | 0.8442 |

Confusion matrix (test, rows=true, cols=pred, LABELS order):

| true \ pred | real | fake | malicious |
|---|---|---|---|
| real | 1426 | 2341 | 1441 |
| fake | 1293 | 3260 | 355 |
| malicious | 0 | 7 | 4885 |

Per-language macro-F1 (test):

| language | macro-F1 |
|---|---|
| bn | 0.3387 |
| code-mixed | 0.0000 |
| en | 0.5296 |

### random_forest

- Macro-F1 (val): **0.9057** ; Macro-F1 (test): **0.9064**

Per-class precision / recall / f1 (test):

| class | precision | recall | f1 |
|---|---|---|---|
| real | 0.8526 | 0.8944 | 0.8730 |
| fake | 0.8816 | 0.8286 | 0.8543 |
| malicious | 0.9878 | 0.9959 | 0.9919 |

Confusion matrix (test, rows=true, cols=pred, LABELS order):

| true \ pred | real | fake | malicious |
|---|---|---|---|
| real | 4658 | 530 | 20 |
| fake | 801 | 4067 | 40 |
| malicious | 4 | 16 | 4872 |

Per-language macro-F1 (test):

| language | macro-F1 |
|---|---|
| bn | 0.7297 |
| code-mixed | 1.0000 |
| en | 0.9339 |

## Selected best model

**logreg** -- selected by **validation macro-F1** with the D-03 minority-class guard (a model whose fake- or malicious-class recall has collapsed cannot win; ties break toward higher minority recall; CPU latency is a secondary tiebreaker only).

- Test macro-F1: **0.9140**

Selected-model per-class precision / recall / f1 (test):

| class | precision | recall | f1 |
|---|---|---|---|
| real | 0.8767 | 0.8748 | 0.8757 |
| fake | 0.8667 | 0.8702 | 0.8684 |
| malicious | 0.9988 | 0.9969 | 0.9979 |

## Recorded hyperparameters

| component | setting |
|---|---|
| LogisticRegression | solver=lbfgs, max_iter=1000, class_weight=balanced |
| ComplementNB | defaults (handles TF-IDF skew natively; NOT MultinomialNB) |
| RandomForest | n_estimators=200, class_weight=balanced, random_state=42 |
| TF-IDF word view | analyzer=word, ngram_range=(1, 2) |
| TF-IDF char view | analyzer=char_wb, ngram_range=(3, 5) |
| TF-IDF shared | min_df=2, max_features=50000, lowercase=True |

Imbalance handling (D-01): `class_weight="balanced"` (LR/RF) + ComplementNB only -- **no synthetic oversampling**.

## Leakage re-check (SC-3)

Each fitted model's top features are inspected for outlet/dateline/year tells (reused Phase-1 `_LEAK_TELL` regex). Any model scoring >= 0.98 (macro-F1) is flagged for suspected-leakage investigation before being trusted.

| model | >= 0.98 | leak tells | investigate |
|---|---|---|---|
| logreg | no | none | no |
| complement_nb | no | fake:word__28 2016, real:word__fiscal 2018 | yes |
| random_forest | no | none | no |

**At least one model was flagged** -- see the table above; investigate before trusting.

The flagged model, **complement_nb, is NOT the selected best model** -- `logreg` is (with no leak tells), so the shipped `best_model.joblib` is not under a leakage cloud. The complement_nb year tells (`28 2016`, `fiscal 2018`) are word-bigram artifacts of residual datelines under inspection rather than confirmed per-class leaks.

## Caveats

- **BFN2 satire-vs-fake assumption (D-04):** the Phase-1 corpus maps BanFakeNews-2.0 (BFN2) `Label` 0/1/2 -> `fake` and 3 -> `real`, so **satire is treated as `fake`**. This is a documented known assumption: BFN2 `real` rows are ~100% SHA1-duplicates of BanFakeNews v1 and dedup away, so the practical impact on fake-class metrics is small. Revisit only if fake-class error analysis surfaces satire confusion.
- **Malicious class is English-only (Phase-1 D-01):** a documented coverage limitation, visible in the per-language macro-F1 tables above.
