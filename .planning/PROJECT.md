# AI-Driven Real-Time Detection of Fake News and Malicious Content

## What This Is

A real-time system that classifies a piece of text (or the article behind a URL) as **real, fake, or malicious**, returning a confidence score and a plain explanation of *why*. It combines classical ML and a multilingual transformer with three supporting signals — source credibility, writing-style/behavioral patterns, and external verification against trusted sources — fused into one verdict. Built for the Bangladesh context (Bangla + English), delivered through a local Streamlit interface. This is the PMICS (Professional Masters in Information and Cyber Security, University of Dhaka) capstone system.

## Core Value

A user pastes text or a URL and instantly gets a trustworthy real/fake/malicious verdict **with an understandable explanation** — accuracy plus transparency is the whole point. If only one thing works, it is: text in → explained verdict + confidence out.

## Current Milestone: v2.0 Multi-Signal Detection & Explainable UI

**Goal:** Turn the shipped classical foundation into the full real-time, explainable detector — a fine-tuned multilingual transformer, the three supporting signal modules, free-API external verification, calibrated decision fusion, faithful explanations, and a local Streamlit UI — delivering the core value end-to-end (text/URL in → explained verdict + confidence out). This milestone takes all **22 requirements carried forward from v1.0** (Phases 3–7).

**Target features:**
- Fine-tuned multilingual transformer (BanglishBERT/BanglaBERT primary, XLM-R fallback) as the primary model — two-stage (malicious-gate → real/fake) classifier with calibrated confidence, exported for local inference *(Phase 3)*
- Malicious-content detection (phishing / scam / suspicious-URL) plus source-credibility and writing-style modules, all behind a uniform `ModuleResult` contract *(Phase 4)*
- External verification against free fact-check / Wikipedia APIs — async, timeout-bounded, gracefully abstaining *(Phase 5)*
- Decision fusion (weighted vote + rule override) with faithful word-level explainability + per-module contribution breakdown, gated by an ablation proving fusion beats the transformer alone *(Phase 6)*
- Local Streamlit UI: paste text OR a URL → instant, clearly-explained verdict + confidence on local hardware *(Phase 7)*

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

**Data Foundation** — *Shipped v1.0 (Phase 01: data-foundation)*
- ✓ DATA-01 — Download scripts for all source datasets; raw not committed — v1.0
- ✓ DATA-02 — Unified 3-class corpus + explicit label mapping + provenance (137,169 rows) — v1.0
- ✓ DATA-03 — Source/publisher leakage stripped; leakage gate PASSED (caught + fixed a real Reuters-dateline leak) — v1.0
- ✓ DATA-04 — Class imbalance handled; leak-free source-disjoint 70/15/15 splits — v1.0
- ✓ DATA-05 — Shared preprocess() (csebuetnlp/normalizer) for Bangla/English/code-mixed; single train=inference entry point — v1.0

**Models (classical)** — *Shipped v1.0 (Phase 02: classical-baselines-metric-discipline)*
- ✓ CLS-01 — Classical baselines (TF-IDF + LR/NB/RF) trained + serialized — v1.0
- ✓ CLS-03 — Compare on macro-F1 / per-class / confusion; select best (LogisticRegression, test macro-F1 0.9140; SC-3 leakage re-check enforced) — v1.0

### Active

<!-- Current scope. Building toward these. Hypotheses until shipped. -->

**Classification core**
- [ ] Classify input text into one of three classes: real / fake / malicious
- [ ] Output a confidence score with every prediction
- [ ] Handle Bangla, English, and mixed Bangla-English text

**Models** *(classical baselines shipped v1.0 → Validated; transformer carried forward)*
- [ ] Multilingual transformer (BanglishBERT / BanglaBERT primary, XLM-R fallback) as primary model — *CLS-02, Phase 3*

**Malicious-content detection** (distinct from fake news)
- [ ] Detect phishing messages (credential/data requests)
- [ ] Detect scam messages (fake job offers, lottery, fraudulent financial schemes)
- [ ] Flag malware-distribution links / suspicious URLs

**Supporting signal modules**
- [ ] Source credibility module: score a source by domain reputation + history of false content
- [ ] Writing-style / behavioral module: detect clickbait words, ALL-CAPS, excessive punctuation, sensational/emotional tone, repetitive phrasing
- [ ] External verification module: fetch related coverage from trusted sources (free APIs — Google Fact Check Tools, Wikipedia) and check claim consistency
- [ ] Decision fusion: combine model output + 3 module signals into one final verdict + confidence

**Explainability**
- [ ] Highlight the words/phrases that drove the verdict
- [ ] Show which factors contributed (which module, e.g. low source credibility, clickbait style)

**Interface & input**
- [ ] Streamlit UI: paste text OR enter a URL, get instant result
- [ ] URL handling: fetch the page, extract article text, run the pipeline
- [ ] Display verdict, confidence, and explanation clearly for non-technical users
- [ ] Real-time response (instant, minimal delay)

**System qualities**
- [ ] Modular design — modules swappable, models retrainable on new datasets

### Out of Scope

<!-- Explicit boundaries with reasoning. -->

- Thesis report writing (chapters, LaTeX) — user handles the academic write-up separately; this project delivers the working system only
- Cloud / public hosting — local Streamlit only for demo and evaluation; no Docker/HF Spaces deploy this milestone
- Paid APIs (NewsAPI paid tier, paid fact-check/LLM services) — free APIs only
- Multimodal detection (images, video, deepfakes) — text-only this milestone
- Social-context / user-network features (follower graphs, spread modeling à la FakeNewsNet social layer) — content + source + verification signals only
- Live streaming / firehose ingestion at scale — single-input real-time, not high-throughput stream processing
- Continuous online/adaptive auto-learning — retraining is manual/batch, not self-updating in production

## Context

- **Academic project**: PMICS capstone, University of Dhaka, Dept. of CSE. Authors: A.S.M Rafiuzzaman Sazin (H-404), Mahi Naz Islam (H-426). Supervisor: Mr Jargis Ahmed; co-supervisor: Mr Md. Faisal Hossain. Brief: `PMICS B4-Proj G11(Mid).pdf`.
- **Motivation**: Bangladesh-specific misinformation problem — high social-media use (Facebook/YouTube), limited digital literacy, Bangla+English mixed content, users vulnerable to both fake news and scams/phishing. Existing fact-checkers (Snopes, FactCheck.org) are reactive/post-hoc; this aims at instant pre-spread detection.
- **Key novelty (from brief)**: a *unified* framework detecting both fake news AND malicious/cyber content in one system; hybrid classical+transformer modeling; multi-layer validation (source credibility + style + external verification) beyond text-only classification.
- **Datasets referenced in the brief** (starting points for research): BanFakeNews (Bangla fake news), FakeNewsNet (English), plus phishing/spam corpora for the malicious classes. Models referenced: BERT, BanglaBERT, RoBERTa.
- **Prior code existed** (classical.py, transformer.py, credibility/style/verify modules, fusion, Streamlit app) but was wiped for a fresh, clean rebuild. Treat as greenfield.
- **Challenges flagged in brief**: context/intent/sarcasm, evolving misinformation patterns, multilingual mixed text, psychological manipulation in malicious content, scarce/noisy/imbalanced labeled data, integration complexity of many modules under real-time constraints.
- **Current state (after v1.0, 2026-06-19)**: Offline foundation shipped — `src/` is 2,579 LOC across 17 Python modules (data pipeline + classical models), 1,658 LOC of tests (100 fast + slow real-corpus gates). 137,169-row 3-class corpus built (gitignored); classical baselines serialized to `models/` (gitignored), best = LogisticRegression (test macro-F1 0.9140). No transformer, malicious-detection, signal modules, verification, fusion, explainability, or UI yet — those are Phases 3–7. 7/29 v1 requirements shipped; 22 carried forward.

## Constraints

- **Tech stack**: Python ML ecosystem; scikit-learn for classical models; Hugging Face Transformers for the transformer; Streamlit for UI. (Confirm exact libs/versions in research.)
- **Language**: Must handle Bangla + English + code-mixed text — drives model choice (multilingual/Bangla-capable transformer) and preprocessing.
- **APIs**: Free tiers only — no paid keys. External verification limited to free fact-check/encyclopedia/news sources.
- **Deployment**: Local machine (Streamlit) only.
- **Data**: Labeled datasets for fake news (esp. Bangla) and malicious content are limited, noisy, and imbalanced — class balancing and careful evaluation required.
- **Ethical AI**: Avoid model bias, aim for fairness across content types/languages, protect user privacy (don't retain submitted text needlessly), keep decisions transparent/explainable.
- **Performance**: Real-time — inference must feel instant in the UI (manage transformer latency on local hardware).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Working system only; report written separately | User scoped deliverable to the running system | — Pending |
| Bangla + English multilingual | Bangladesh context; mixed-language content is core to the problem | — Pending |
| Full pipeline in v1 (all modules + fusion + UI) | Unified multi-layer framework is the project's stated novelty; partial build undercuts the contribution | — Pending |
| Text + URL input, real-time Streamlit UI | Matches brief; URL support needed for real-world news links | — Pending |
| Free APIs only for external verification | No paid keys available | — Pending |
| Local-only deployment | Demo/evaluation context; avoids hosting complexity | — Pending |
| 3-class output: real / fake / malicious | Brief's unified framework spanning misinformation + cyber threats | — Pending |
| Classical baselines before transformer | De-risk the data + artifact-persistence path on CPU before GPU spend | ✓ Good — v1.0 (best classical macro-F1 0.9140; artifact path validated) |
| macro-F1 headline; ≥98% accuracy = suspected leakage | Honest eval on imbalanced 3-class data; accuracy misleads | ✓ Good — v1.0 (leakage gate caught a real Reuters-dateline leak) |
| Natural class distribution + class-weighting (no resampling) | D-03/D-04: store/report real distribution, balance at train time | ✓ Good — v1.0 (every model fake+malicious recall > 0) |
| BanFakeNews-2.0 4-class → 3→real, 0/1/2→fake + report caveat | No in-file codebook; mapping directionally sound, flagged for review | ✓ Good — v1.0 (D-04; documented caveat) |
| Milestone v1.0 = foundation only (Phases 1–2); 22 reqs carried forward | Slice the offline foundation; full product spans Phases 3–7 | ✓ Good — v1.0 (clean foundation shipped; transformer/modules/UI next) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-20 — started milestone v2.0 (Multi-Signal Detection & Explainable UI); 22 carried-forward requirements scoped across Phases 3–7*
