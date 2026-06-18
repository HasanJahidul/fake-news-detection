# Roadmap: AI-Driven Detection of Fake News and Malicious Content

## Overview

This roadmap builds a multi-signal text-classification pipeline for the Bangladesh context (Bangla + English + code-mixed) bottom-up, following the strict offline→online boundary the architecture demands. We start by assembling a de-leaked, balanced, documented 3-class corpus (the gate for all training), then train fast classical baselines to lock metric discipline before spending GPU time on the primary transformer. Once the `ModuleResult` contract is fixed, the credibility / style / malicious-detection signal modules build in parallel, the network-bound verification module is isolated for its own risk-managed phase, fusion + explainability consume all upstream signals (with an ablation gate proving fusion beats the transformer alone), and finally a thin Streamlit app wraps the pipeline into the real-time, explainable verdict that is the project's core value.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Data Foundation** - De-leaked, balanced, documented 3-class corpus + shared preprocessing; locks the label schema and two-stage decision.
- [ ] **Phase 2: Classical Baselines + Metric Discipline** - TF-IDF + LR/NB/RF serialized artifacts with honest macro-F1 reporting; validates the data+artifact path on CPU.
- [ ] **Phase 3: Transformer Fine-Tuning + Model Selection** - Fine-tuned BanglishBERT/BanglaBERT (XLM-R fallback) exported for local inference; two-stage classifier with calibrated confidence; best model selected.
- [ ] **Phase 4: Signal Modules (Contract + Credibility/Style/Malicious)** - `ModuleResult` contract plus credibility, style, and malicious-detection modules conforming to it.
- [ ] **Phase 5: External Verification Module** - Async, timeout-bounded, gracefully-abstaining verification against free fact-check/Wikipedia APIs, with measured Bangla coverage.
- [ ] **Phase 6: Fusion + Explainability** - Weighted-vote + rule-override fusion with calibrated confidence; faithful word highlights + per-module contribution breakdown; ablation gate vs transformer-alone.
- [ ] **Phase 7: Integration + Streamlit UI** - Thin pipeline wrapper + Streamlit app: paste text OR URL, instant explained verdict on local hardware; privacy + disclaimer.

## Phase Details

### Phase 1: Data Foundation

**Goal**: A single, documented, de-leaked, class-balanced 3-class corpus (real/fake/malicious) with grouped train/validation/test splits, plus one shared preprocessing function used identically offline and online.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):

  1. Download scripts fetch every source dataset (BanFakeNews/2.0, ISOT, LIAR, SMS Spam, phishing) from HF/Kaggle/UCI; no raw data is committed to git.
  2. A documented label-mapping table assembles real/fake/malicious from the source corpora, with per-sample source provenance retained for auditing.
  3. A leakage probe (title-only / single-body-sentence / source-stripped) and source-disjoint split are run; boilerplate (Reuters datelines, outlet names, bylines, URLs, image credits) is stripped, and any near-duplicates are removed before splitting.
  4. Class distribution is reported and balanced (class weighting / in-fold resampling), producing leak-free grouped train/validation/test splits.
  5. A shared `preprocess` function (including `csebuetnlp/normalizer`) handles Bangla + English + code-mixed text and is the single importable entry point for both training and inference.

**Plans**: 7 plans (6 waves)

  - [x] 01-01-PLAN.md — Wave 0: pytest scaffold + test stubs + shared preprocess() (DATA-05)
  - [x] 01-02-PLAN.md — Wave 1: dataset acquisition scripts → gitignored data/raw/ (DATA-01)
  - [x] 01-03-PLAN.md — Wave 2: label mapping (LIAR collapse) + provenance schema/Parquet (DATA-02)
  - [x] 01-04-PLAN.md — Wave 2: boilerplate/source-leakage stripping + language tagging (DATA-03)
  - [ ] 01-05-PLAN.md — Wave 3: dedup (exact+fuzzy) + source-disjoint grouped 70/15/15 splits (DATA-04) — depends on 01-03 (confirmed source columns for group keys)
  - [ ] 01-06-PLAN.md — Wave 4: build_corpus orchestrator → Parquet + class-distribution report (DATA-02/04)
  - [ ] 01-07-PLAN.md — Wave 5: leakage probe gate (title/sentence/source-stripped) + report (DATA-03)

  **Wave structure:** W0 {01-01} → W1 {01-02} → W2 {01-03, 01-04} → W3 {01-05} → W4 {01-06} → W5 {01-07}

### Phase 2: Classical Baselines + Metric Discipline

**Goal**: Trained, serialized TF-IDF classical baselines that validate the data pipeline and artifact-persistence path on CPU, with macro-F1 / per-class / confusion-matrix reporting established as the project's metric standard.
**Depends on**: Phase 1
**Requirements**: CLS-01, CLS-03
**Success Criteria** (what must be TRUE):

  1. TF-IDF + Logistic Regression, Naive Bayes, and Random Forest are trained on the Phase 1 corpus and the fitted vectorizer + best model are serialized to `models/` for load-only reuse.
  2. Macro-F1, per-class precision/recall, and a confusion matrix are reported per model on the held-out split (accuracy is never the headline metric); minority-class (fake/malicious) recall is non-trivial.
  3. Top predictive features are inspected and confirmed not to be outlet names / datelines / years; any model scoring greater than or equal to 98% is investigated as suspected leakage before being trusted.
  4. A model-comparison report is written to `reports/` and the best classical model is recorded with its metrics.

**Plans**: TBD

### Phase 3: Transformer Fine-Tuning + Model Selection

**Goal**: A fine-tuned multilingual transformer exported for local inference, wrapped in a two-stage (malicious-gate then real/fake) classifier with calibrated per-prediction confidence, with the primary model selected by measured code-mixed macro-F1 and local latency.
**Depends on**: Phase 2
**Requirements**: CLS-02, CLS-04
**Success Criteria** (what must be TRUE):

  1. BanglishBERT/BanglaBERT (primary) and XLM-R (fallback) are fine-tuned and compared head-to-head on a code-mixed validation set, reporting per-language and code-mixed macro-F1, with the matching `csebuetnlp/normalizer` applied identically at train and inference.
  2. The selected transformer is exported via `save_pretrained` to `models/transformer/` (with tokenizer + label map) and loads for inference without any training code path.
  3. Classification runs as a two-stage approach (malicious-vs-not gate, then real/fake) rather than a flat 3-way softmax, and emits a confidence score with every prediction.
  4. A selection report records the chosen model, its code-mixed/per-language metrics, and measured single-input inference latency on the target local hardware.

**Plans**: TBD

### Phase 4: Signal Modules (Contract + Credibility/Style/Malicious)

**Goal**: A locked uniform `ModuleResult` contract plus three signal modules — source credibility, writing-style/behavioral, and malicious-content detection — that each emit that contract and can be tested in isolation.
**Depends on**: Phase 3
**Requirements**: FUS-01, CRED-01, CRED-02, STY-01, STY-02, MAL-01, MAL-02, MAL-03
**Success Criteria** (what must be TRUE):

  1. A `ModuleResult` contract (label, score, contribution, evidence, available) is defined in `modules/base.py`, and every module returns it so modules are swappable without touching fusion.
  2. The credibility module scores a source from a curated domain-reputation list + false-content history, treats unknown domains as neutral (never "fake"), and emits its result as a `ModuleResult` for text-only input scoring neutral.
  3. The style module detects clickbait-lexicon hits, ALL-CAPS ratio, excessive punctuation, sensational/emotional tone, and repetition across Bangla + English lexicons, emitted as a `ModuleResult`.
  4. The malicious module detects phishing text (credential/urgency cues), scam text (job/lottery/financial fraud), and flags suspicious/malware-distribution URLs via lexical heuristics (IP-host, typosquatting, subdomain depth, punycode, blocklist snapshot) using only features computable at inference, with a tested no-network path.

**Plans**: TBD

### Phase 5: External Verification Module

**Goal**: A verification module that extracts a checkable claim, queries free trusted sources, and returns a uniform support/refute/no-evidence signal — async, timeout-bounded, and gracefully abstaining so it never blocks or coerces the verdict.
**Depends on**: Phase 4
**Requirements**: VER-01, VER-02
**Success Criteria** (what must be TRUE):

  1. The module extracts a short candidate claim and queries free sources (Google Fact Check Tools API, Wikipedia/Wikidata) for related coverage, excluding the input's own domain to avoid circular sourcing.
  2. It returns support / refute / no-evidence as a `ModuleResult`; "no evidence" is a valid, common output that is never coerced into a verdict.
  3. Calls are async and timeout-bounded with caching; on quota/429/timeout/outage the module sets `available=False` and degrades gracefully without blocking the pipeline.
  4. Bangla coverage of the free APIs is measured empirically and documented as a known limitation.

**Plans**: TBD

### Phase 6: Fusion + Explainability

**Goal**: A fusion layer that combines the classifier + credibility + style + verification signals into one calibrated verdict + confidence (proven to beat the transformer alone), with faithful word-level highlights and a per-module contribution breakdown.
**Depends on**: Phase 5
**Requirements**: FUS-02, FUS-03, EXP-01, EXP-02
**Success Criteria** (what must be TRUE):

  1. Fusion combines all available `ModuleResult`s via weighted vote + rule overrides (e.g. a strong malicious hit forces "malicious"), renormalizing weights over present signals so a missing/abstaining module degrades gracefully.
  2. Module scores are calibrated to a comparable scale, and an ablation report shows full-fusion verdict/confidence versus transformer-alone on the held-out set (improvement, or an honest statement that fusion improves explanation without hurting macro-F1).
  3. The explanation highlights the words/phrases that drove the verdict using input-attribution (SHAP / Integrated Gradients / LIME, not raw attention), and a deletion test confirms the highlights are faithful (removing them changes the prediction).
  4. A contributing-factors breakdown shows which modules fired (e.g. low source credibility, clickbait style, refuted by verification).

**Plans**: TBD

### Phase 7: Integration + Streamlit UI

**Goal**: A thin pipeline orchestrator wrapped in a local Streamlit app that accepts pasted text or a URL and returns an instant, clearly-explained verdict + confidence on local hardware, respecting privacy and surfacing a disclaimer.
**Depends on**: Phase 6
**Requirements**: UI-01, UI-02, UI-03, UI-04, SYS-01, SYS-02
**Success Criteria** (what must be TRUE):

  1. The Streamlit app accepts pasted text OR a URL; for a URL it fetches the page and extracts article text (trafilatura) before running the pipeline, falling back to manual paste on extraction failure.
  2. Verdict, confidence, highlighted words, and the per-module contribution breakdown render clearly for a non-technical user, with an "uncertain — verify manually" band shown when confidence is low.
  3. Models/pipeline load once via `@st.cache_resource`; a single-input verdict returns fast on local hardware (verification non-blocking/progressive) so the response feels instant.
  4. The design is modular — modules swap behind the `ModuleResult` contract and models retrain via scripts — and submitted text is not logged/retained, with a "not legal/medical advice, automated estimate" disclaimer surfaced.

**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Foundation | 4/7 | In Progress|  |
| 2. Classical Baselines + Metric Discipline | 0/TBD | Not started | - |
| 3. Transformer Fine-Tuning + Model Selection | 0/TBD | Not started | - |
| 4. Signal Modules (Contract + Credibility/Style/Malicious) | 0/TBD | Not started | - |
| 5. External Verification Module | 0/TBD | Not started | - |
| 6. Fusion + Explainability | 0/TBD | Not started | - |
| 7. Integration + Streamlit UI | 0/TBD | Not started | - |
</content>
