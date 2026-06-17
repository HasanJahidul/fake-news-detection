# AI-Driven Real-Time Detection of Fake News and Malicious Content

## What This Is

A real-time system that classifies a piece of text (or the article behind a URL) as **real, fake, or malicious**, returning a confidence score and a plain explanation of *why*. It combines classical ML and a multilingual transformer with three supporting signals — source credibility, writing-style/behavioral patterns, and external verification against trusted sources — fused into one verdict. Built for the Bangladesh context (Bangla + English), delivered through a local Streamlit interface. This is the PMICS (Professional Masters in Information and Cyber Security, University of Dhaka) capstone system.

## Core Value

A user pastes text or a URL and instantly gets a trustworthy real/fake/malicious verdict **with an understandable explanation** — accuracy plus transparency is the whole point. If only one thing works, it is: text in → explained verdict + confidence out.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

<!-- Current scope. Building toward these. Hypotheses until shipped. -->

**Classification core**
- [ ] Classify input text into one of three classes: real / fake / malicious
- [ ] Output a confidence score with every prediction
- [ ] Handle Bangla, English, and mixed Bangla-English text

**Models**
- [ ] Classical ML baselines: Logistic Regression, Naive Bayes, Random Forest over TF-IDF features
- [ ] Multilingual transformer (BERT-family, e.g. BanglaBERT / multilingual BERT / XLM-R) as primary model
- [ ] Compare models on accuracy / precision / recall / F1; select best

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
*Last updated: 2026-06-17 after initialization*
