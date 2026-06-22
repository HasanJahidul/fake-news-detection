# Roadmap: AI-Driven Detection of Fake News and Malicious Content

## Milestones

- ✅ **v1.0 — Data + Classical Foundation** — Phases 1–2 (shipped 2026-06-19)
- 📋 **v2.0 — Multi-Signal Detection & Explainable UI** — Phases 3–7 (transformer → modules → verification → fusion/explainability → UI)

## Overview

This roadmap builds a multi-signal text-classification pipeline for the Bangladesh context (Bangla + English + code-mixed) bottom-up, following the strict offline→online boundary the architecture demands. v1.0 assembled a de-leaked, balanced, documented 3-class corpus (the gate for all training) and trained fast classical baselines to lock metric discipline before any GPU spend. v2.0 takes the 22 carried-forward requirements: fine-tune the primary transformer, fix the `ModuleResult` contract and build the credibility / style / malicious-detection signal modules, isolate the network-bound verification module in its own risk-managed phase, fuse all upstream signals with explainability (gated by an ablation proving fusion beats the transformer alone), and finally wrap the pipeline in a thin local Streamlit app — delivering the real-time, explainable verdict that is the project's core value.

## Phases

**Phase Numbering:** Integer phases (1, 2, 3) are planned milestone work; decimal phases (2.1) are urgent insertions (marked INSERTED). Numbering continues across milestones — v2.0 begins at Phase 3.

### ✅ Milestone v1.0 — Data + Classical Foundation (Phases 1–2) — SHIPPED 2026-06-19

<details>
<summary>SHIPPED 2026-06-19 — foundation only (7 of 29 v1 requirements; 22 carried forward to v2.0)</summary>

- [x] **Phase 1: Data Foundation** (7/7 plans) — completed 2026-06-18 — de-leaked, documented 3-class corpus (137,169 rows) + shared preprocess(); leakage gate PASSED (caught + fixed a real Reuters-dateline leak).
- [x] **Phase 2: Classical Baselines + Metric Discipline** (3/3 plans) — completed 2026-06-19 — TF-IDF + LR/NB/RF serialized; best = LogisticRegression, test macro-F1 0.9140; metric discipline + SC-3 leakage re-check locked.

Full detail archived in `milestones/v1.0-ROADMAP.md`.

</details>

### 📋 Milestone v2.0 — Multi-Signal Detection & Explainable UI (Phases 3–7)

- [ ] **Phase 3: Transformer Fine-Tuning + Model Selection** — Fine-tuned BanglishBERT/BanglaBERT (XLM-R fallback) exported for local inference; two-stage classifier with calibrated confidence; best model selected. (pipeline built + unit-proven; verification gaps_found 2026-06-20 — SC-1/SC-4 pending: gap-closure 03-05 (wire D-09 sweep / CR-01) + 03-06 (run Colab fine-tune + populate report))
- [ ] **Phase 4: Signal Modules (Contract + Credibility/Style/Malicious)** — `ModuleResult` contract plus credibility, style, and malicious-detection modules conforming to it.
- [ ] **Phase 5: External Verification Module** — Async, timeout-bounded, gracefully-abstaining verification against free fact-check/Wikipedia APIs, with measured Bangla coverage.
- [ ] **Phase 6: Fusion + Explainability** — Weighted-vote + rule-override fusion with calibrated confidence; faithful word highlights + per-module contribution breakdown; ablation gate vs transformer-alone.
- [ ] **Phase 7: Integration + Streamlit UI** — Thin pipeline wrapper + Streamlit app: paste text OR URL, instant explained verdict on local hardware; privacy + disclaimer.

## Phase Details

### Milestone v2.0 — Multi-Signal Detection & Explainable UI (Phase Details)

#### Phase 3: Transformer Fine-Tuning + Model Selection

**Goal**: A fine-tuned multilingual transformer exported for local inference, wrapped in a two-stage (malicious-gate then real/fake) classifier with calibrated per-prediction confidence, with the primary model selected by per-language macro-F1 (Bangla priority; code-mixed is a qualitative small-N check, latency relaxed per D-11).
**Depends on**: Phase 2
**Requirements**: CLS-02, CLS-04
**Success Criteria** (what must be TRUE):

  1. BanglishBERT/BanglaBERT (primary) and XLM-R (fallback) are fine-tuned and compared head-to-head on a code-mixed validation set, reporting per-language and code-mixed macro-F1, with the matching `csebuetnlp/normalizer` applied identically at train and inference.
  2. The selected transformer is exported via `save_pretrained` to `models/transformer/` (with tokenizer + label map) and loads for inference without any training code path.
  3. Classification runs as a two-stage approach (malicious-vs-not gate, then real/fake) rather than a flat 3-way softmax, and emits a calibrated confidence score with every prediction.
  4. A selection report records the chosen model and its code-mixed/per-language metrics. (Latency relaxed per D-11 — the report confirms the model runs interactively on the M4; no latency benchmark.)

**Plans**: 6 plans (4 built + 2 gap-closure)
Plans:
**Wave 1**

- [x] 03-01-PLAN.md — Install pinned transformer deps (torch 2.6 / transformers 4.46) + import gate + tiny GPU-free fixture + 9 Wave-0 RED test scaffolds.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03-02-PLAN.md — transformer_data.py (gate/realfake label views, preprocess→tokenize, class weights) + calibration.py (val-only temperature scaling).

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03-03-PLAN.md — transformer_train.py (class-weighted WeightedTrainer + save_pretrained export layout) + Colab T4 training notebook.

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03-04-PLAN.md — transformer_infer.py (load-only two-stage cascade + calibrated confidence) + select_transformer.py (Bangla-priority selection, threshold sweep, SC-3 re-check, selection report) + code-mixed probe.

**Wave 5 — gap closure** *(closes verification gaps_found 2026-06-20)*

- [x] 03-05-PLAN.md — GAP 1 / CR-01 (autonomous): wired the dead D-09 gate-threshold sweep into select_transformer.main() — per-stage val probs (evaluate_cascade_probs / cascade.gate_realfake_probs), threshold_record assigned, applied before test eval + written into exported temperature.json; folded in WR-04/IN-01/WR-02/WR-03. RED→GREEN→GREEN, full suite 127 exit 0.

**Wave 6 — gap closure** *(blocked on 03-05; not autonomous)*

- [ ] 03-06-PLAN.md — GAP 2 (manual + auto): human Colab free-T4 fine-tune of both backbones → models/transformer/ exports, then `python -m src.models.select_transformer` populates reports/transformer_selection_report.md (no "(pending Colab run)" cells; chosen model + per-language/code-mixed macro-F1 + swept gate threshold + SC-3 result recorded).

#### Phase 4: Signal Modules (Contract + Credibility/Style/Malicious)

**Goal**: A locked uniform `ModuleResult` contract plus three signal modules — source credibility, writing-style/behavioral, and malicious-content detection — that each emit that contract and can be tested in isolation.
**Depends on**: Phase 3
**Requirements**: FUS-01, CRED-01, CRED-02, STY-01, STY-02, MAL-01, MAL-02, MAL-03
**Success Criteria** (what must be TRUE):

  1. A `ModuleResult` contract (label, score, contribution, evidence, available) is defined in `modules/base.py`, and every module returns it so modules are swappable without touching fusion.
  2. The credibility module scores a source from a curated domain-reputation list + false-content history, treats unknown domains as neutral (never "fake"), and emits its result as a `ModuleResult` scoring neutral for text-only input that has no source.
  3. The style module detects clickbait-lexicon hits, ALL-CAPS ratio, excessive punctuation, sensational/emotional tone, and repetition across Bangla + English lexicons, emitted as a `ModuleResult`.
  4. The malicious module detects phishing text (credential/urgency cues), scam text (job/lottery/financial fraud), and flags suspicious/malware-distribution URLs via lexical heuristics (IP-host, typosquatting, subdomain depth, punycode, blocklist snapshot) using only features computable at inference, with a tested no-network path.

**Plans**: TBD

#### Phase 5: External Verification Module

**Goal**: A verification module that extracts a checkable claim, queries free trusted sources, and returns a uniform support/refute/no-evidence signal — async, timeout-bounded, and gracefully abstaining so it never blocks or coerces the verdict.
**Depends on**: Phase 4
**Requirements**: VER-01, VER-02
**Success Criteria** (what must be TRUE):

  1. The module extracts a short candidate claim and queries free sources (Google Fact Check Tools API, Wikipedia/Wikidata) for related coverage, excluding the input's own domain to avoid circular sourcing.
  2. It returns support / refute / no-evidence as a `ModuleResult`; "no evidence" is a valid, common output that is never coerced into a verdict.
  3. Calls are async and timeout-bounded with caching; on quota/429/timeout/outage the module sets `available=False` and degrades gracefully without blocking the pipeline.
  4. Bangla coverage of the free APIs is measured empirically and documented as a known limitation.

**Plans**: TBD

#### Phase 6: Fusion + Explainability

**Goal**: A fusion layer that combines the classifier + credibility + style + verification signals into one calibrated verdict + confidence (proven to beat the transformer alone), with faithful word-level highlights and a per-module contribution breakdown.
**Depends on**: Phase 5
**Requirements**: FUS-02, FUS-03, EXP-01, EXP-02
**Success Criteria** (what must be TRUE):

  1. Fusion combines all available `ModuleResult`s via weighted vote + rule overrides (e.g. a strong malicious hit forces "malicious"), renormalizing weights over present signals so a missing/abstaining module degrades gracefully.
  2. Module scores are calibrated to a comparable scale, and an ablation report shows full-fusion verdict/confidence versus transformer-alone on the held-out set (improvement, or an honest statement that fusion improves explanation without hurting macro-F1).
  3. The explanation highlights the words/phrases that drove the verdict using input-attribution (SHAP / Integrated Gradients / LIME, not raw attention), and a deletion test confirms the highlights are faithful (removing them changes the prediction).
  4. A contributing-factors breakdown shows which modules fired (e.g. low source credibility, clickbait style, refuted by verification).

**Plans**: TBD

#### Phase 7: Integration + Streamlit UI

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

**Execution Order:** Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Data Foundation | v1.0 | 7/7 | Complete | 2026-06-18 |
| 2. Classical Baselines + Metric Discipline | v1.0 | 3/3 | Complete | 2026-06-19 |
| 3. Transformer Fine-Tuning + Model Selection | v2.0 | 4/6 | Gap closure planned | - |
| 4. Signal Modules (Contract + Credibility/Style/Malicious) | v2.0 | 0/TBD | Not started | - |
| 5. External Verification Module | v2.0 | 0/TBD | Not started | - |
| 6. Fusion + Explainability | v2.0 | 0/TBD | Not started | - |
| 7. Integration + Streamlit UI | v2.0 | 0/TBD | Not started | - |
