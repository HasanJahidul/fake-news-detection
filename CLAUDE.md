<!-- GSD:project-start source:PROJECT.md -->

## Project

**AI-Driven Real-Time Detection of Fake News and Malicious Content**

A real-time system that classifies a piece of text (or the article behind a URL) as **real, fake, or malicious**, returning a confidence score and a plain explanation of *why*. It combines classical ML and a multilingual transformer with three supporting signals — source credibility, writing-style/behavioral patterns, and external verification against trusted sources — fused into one verdict. Built for the Bangladesh context (Bangla + English), delivered through a local Streamlit interface. This is the PMICS (Professional Masters in Information and Cyber Security, University of Dhaka) capstone system.

**Core Value:** A user pastes text or a URL and instantly gets a trustworthy real/fake/malicious verdict **with an understandable explanation** — accuracy plus transparency is the whole point. If only one thing works, it is: text in → explained verdict + confidence out.

### Constraints

- **Tech stack**: Python ML ecosystem; scikit-learn for classical models; Hugging Face Transformers for the transformer; Streamlit for UI. (Confirm exact libs/versions in research.)
- **Language**: Must handle Bangla + English + code-mixed text — drives model choice (multilingual/Bangla-capable transformer) and preprocessing.
- **APIs**: Free tiers only — no paid keys. External verification limited to free fact-check/encyclopedia/news sources.
- **Deployment**: Local machine (Streamlit) only.
- **Data**: Labeled datasets for fake news (esp. Bangla) and malicious content are limited, noisy, and imbalanced — class balancing and careful evaluation required.
- **Ethical AI**: Avoid model bias, aim for fairness across content types/languages, protect user privacy (don't retain submitted text needlessly), keep decisions transparent/explainable.
- **Performance**: Real-time — inference must feel instant in the UI (manage transformer latency on local hardware).

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## TL;DR Recommendations

- **Primary transformer:** `csebuetnlp/banglabert` (BanglaBERT, ELECTRA-discriminator). For the explicitly **code-mixed Bangla+English** requirement, use `csebuetnlp/banglishbert` as a strong sibling and benchmark both. If a single all-rounder is preferred over BUET-CSE models, fall back to **XLM-RoBERTa base**. Do NOT make mBERT the primary — it is the weakest on Bangla of the three (see rationale).
- **Classical baselines:** scikit-learn `TfidfVectorizer` + `LogisticRegression` / `ComplementNB` / `RandomForestClassifier`. Liblinear/saga LR is the strongest classical baseline; keep all three for the required model comparison.
- **Bangla preprocessing:** `csebuetnlp/normalizer` (MANDATORY before BanglaBERT tokenization — model was pretrained on it) + `bnlp-toolkit` for tokenization/cleaning. `indic-nlp-library` only if you need Indic-script transliteration.
- **URL extraction:** `trafilatura` (actively maintained, multilingual, precision-focused). Avoid `newspaper3k` (unmaintained since 2018).
- **Explainability:** `shap` for the classical models + `transformers-interpret` (or `Captum` LayerIntegratedGradients) for word-level highlighting on the transformer. `LIME` as a model-agnostic backup that works on both. Skip raw attention as the explanation (attention ≠ explanation — see What NOT to Use).
- **External verification (free):** Google Fact Check Tools API (free, API-key only, generous quota) + Wikipedia/Wikidata REST APIs (no key). Skip NewsAPI free tier for production verification (24h-delayed, 100 req/day).

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11.x | Runtime | 3.11 is the sweet spot: every lib below ships wheels for it; 3.12/3.13 occasionally lag on PyTorch/native deps. Avoid 3.13 unless all wheels confirmed. |
| PyTorch | 2.6.x (CPU or CUDA) | Transformer training/inference backend | HF Transformers' default backend; mature CPU path matters for the local/real-time constraint. Pin to match your CUDA if a GPU is present. |
| Hugging Face Transformers | 4.4x (4.40+; latest 5.x exists — pin to a tested 4.x for stability) | Load + fine-tune BanglaBERT/XLM-R | Standard for BERT-family fine-tuning; `Trainer` API + `AutoModelForSequenceClassification` cover the whole training loop. Transformers 5.x shipped recently; for a capstone, pin a stable 4.4x to avoid churn. |
| scikit-learn | 1.5.x–1.6.x (1.9.0 is latest; 1.5/1.6 are battle-tested) | Classical models + TF-IDF + metrics | The classical-baseline requirement maps 1:1 to sklearn `Pipeline`. Provides `classification_report`, confusion matrix, cross-val out of the box. |
| Streamlit | 1.3x–1.4x (1.40+) | Local UI: paste text / enter URL → verdict | Required by brief. Single-file app, instant rerun, `st.cache_resource` to load the model once (critical for "real-time feel"). |
| csebuetnlp BanglaBERT | model rev on HF Hub | **Primary** classifier (fine-tuned 3-class) | SOTA on Bangla NLU (NAACL-2022 Findings); ELECTRA discriminator. Best Bangla representation of the candidates. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `normalizer` (csebuetnlp) | git main | Bangla Unicode normalization | **Always** before tokenizing for BanglaBERT/BanglishBERT. Install: `pip install git+https://github.com/csebuetnlp/normalizer`. Skipping it degrades accuracy — the model was pretrained on normalized text. |
| `bnlp-toolkit` | 4.4.0 | Bangla tokenization, cleaning, basic NLP | Classical-pipeline Bangla tokenization, text cleaning, optional Bengali word embeddings. |
| `indic-nlp-library` | 0.92 | Indic script normalization/transliteration | Only if you need Bangla↔Latin transliteration or cross-Indic handling. Not required for the core pipeline. |
| `trafilatura` | 2.1.x | URL → clean article text + metadata | URL input path. Multilingual, precision-focused, actively maintained. |
| `shap` | 0.46.x+ | Word/feature attribution (classical) | Explainability for TF-IDF + LR/NB/RF (`LinearExplainer` / `TreeExplainer` are fast and exact-ish). |
| `transformers-interpret` | 0.10.x | Word-level attribution for the transformer | Wraps Captum Integrated Gradients for HF sequence-classification models; one-call word highlighting. |
| `captum` | 0.7.x | Lower-level attribution (LayerIntegratedGradients) | Use directly if `transformers-interpret` lags your Transformers version. |
| `lime` | 0.2.0.1 | Model-agnostic local explanations | Backup explainer; works identically on classical + transformer via a predict-proba wrapper. Slower than IG. |
| `pandas` | 2.2.x | Dataset loading/wrangling | All dataset ETL. |
| `datasets` (HF) | 2.2x/3.x | Load HF-hosted corpora | Pulling BanFakeNews / English sets from HF Hub when available. |
| `requests` | 2.32.x | Call Fact Check / Wikipedia APIs | External-verification module HTTP. |
| `python-dotenv` | 1.0.x | Load free API keys from `.env` | Keep the Google API key out of source (repo already auto-loads `.env`). |
| `imbalanced-learn` | 0.12.x+ | Class balancing (SMOTE / class weights) | BanFakeNews is heavily imbalanced (~1.2k fake vs ~48k real). Pairs with sklearn `class_weight='balanced'`. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Jupyter / nbconvert | Training notebooks (classical + transformer) | Repo history shows `01_train_classical.ipynb`, `02_train_xlmr_colab.ipynb` — Colab free GPU is the realistic place to fine-tune the transformer. |
| Google Colab (free tier) | GPU for transformer fine-tuning | BanglaBERT/XLM-R base fine-tune comfortably on a free T4; export the model, run inference locally on CPU. |
| `accelerate` | Device placement for `Trainer` | Smooths CPU/GPU handoff; required by recent Transformers `Trainer`. |
| `evaluate` (HF) | accuracy/precision/recall/F1 | Standard metric computation in the `Trainer` loop. |

## Installation

# Core ML + UI

# Bangla NLP (normalizer is git-only and MANDATORY for BanglaBERT)

# optional: pip install indic-nlp-library

# URL extraction

# Explainability

# Class imbalance + API/env

## Primary Transformer: Decision and Rationale

| Candidate | Params | Bangla quality | English quality | Code-mixed | Verdict |
|-----------|--------|----------------|-----------------|------------|---------|
| `csebuetnlp/banglabert` | ~110M (ELECTRA base) | **Best** (SOTA Bangla NLU, NAACL-2022) | Weak (Bangla-only pretraining) | Weak alone | **Primary for Bangla-heavy input** |
| `csebuetnlp/banglishbert` | ~110M | Strong | Strong | **Best** (pretrained on Bangla+English) | **Primary for the code-mixed requirement** |
| `xlm-roberta-base` | ~270M | Good (100-lang, incl. Bangla) | Strong | Good | **Robust fallback / single all-rounder** |
| `bert-base-multilingual-cased` | ~178M | Weakest of the three on Bangla | OK | OK | **Not recommended as primary** |

## Datasets

### Fake News

| Dataset | Lang | Size | Source / How to obtain | License notes |
|---------|------|------|------------------------|---------------|
| **BanFakeNews** | Bangla | ~50k articles (~48.7k authentic, ~1.3k fake) | Kaggle `cryptexcode/banfakenews`; ACL/LREC-2020 paper; GitHub. HF mirrors exist. | CC BY-NC-SA 4.0 (research use). Cite the LREC-2020 paper. **Heavily imbalanced** — balance before training. |
| **BanFakeNews-2.0** | Bangla | ~47k authentic + ~13k manually-annotated fake | Kaggle `hrithikmajumdar/bangla-fake-news` | Newer, far better fake/real balance — **prefer this over v1** if accessible. Verify license on the Kaggle page. |
| **ISOT Fake News** | English | ~45k (~21.4k real / ~23.5k fake, well balanced) | Univ. of Victoria; Kaggle mirrors | Academic/research use; cite ISOT. Well-balanced — good English anchor. |
| **LIAR** | English | ~12.8k short PolitiFact statements, 6-way labels | HF Hub `liar`; UCSB | Public research. 6 fine-grained labels — collapse to true/false (and treat as "fake" class) for the 3-class setup. |
| **FakeNewsNet** | English | Variable (GossipCop + PolitiFact; needs crawling) | GitHub `KaiDMML/FakeNewsNet` | Provides IDs; article text requires crawling (Twitter API now paid). **Use only the content layer**; social layer is out of scope. Higher effort — treat ISOT/LIAR as primary English sources. |

### Malicious Content (phishing / spam / scam)

| Dataset | Type | Size | Source | License notes |
|---------|------|------|--------|---------------|
| **SMS Spam Collection** | SMS spam/ham | 5,574 msgs (747 spam) | UCI ML Repo (id 228); Kaggle `uciml/sms-spam-collection-dataset` | Public, free for research. The standard SMS-spam baseline; maps directly to "scam SMS". English-only. |
| **Nazario Phishing Corpus** | Phishing emails | thousands of raw phishing emails | monkey.org/~jose/phishing (Jose Nazario) | Public research corpus; raw `.mbox` — needs parsing to text. Classic phishing source. |
| **Phishing Email datasets (Kaggle)** | Phishing/legit emails | tens of thousands (varies by set) | Kaggle (multiple curated phishing-email datasets) | License varies per dataset — check each. Easier to consume than raw Nazario. |
| **UCI Phishing Websites** | URL/site features | ~11k instances, 30 features | UCI ML Repo | Public. **Feature-based (not text)** — use it to inform the suspicious-URL/link signal, not the text classifier. |

## Free External-Verification APIs

| API | Auth | Cost / limits | Suitability |
|-----|------|---------------|-------------|
| **Google Fact Check Tools API** (`factchecktools.googleapis.com/v1alpha1/claims:search`) | API key (free, via Google Cloud Console; enable the API) | Free; generous daily quota (default Google API quotas, no billing required). | **Primary verification source.** Returns claim reviews from fact-checkers (incl. some Bangla via partners). Best free option for "has this claim been fact-checked?". |
| **Wikipedia REST API** (`en.wikipedia.org`, `bn.wikipedia.org`) | None | Free; courtesy rate limits (~200 req/s server-wide; set a `User-Agent`). | Pull related coverage / topic summaries for claim-consistency checks. Bangla Wikipedia (`bn.`) supports the Bangla path. |
| **Wikidata API / SPARQL** | None | Free; be polite (User-Agent, throttle). | Entity grounding (people, orgs, events named in the text). |
| **NewsAPI (free tier)** | API key | 100 req/day, **24h-delayed**, dev-only | **Not recommended** for production verification — delay defeats "real-time" and the quota is tiny. Acceptable only for offline corpus building. |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| BanglishBERT / BanglaBERT | XLM-RoBERTa base | When you want ONE model covering all languages + English-heavy malicious classes, and can absorb slower CPU inference. |
| BanglaBERT | mBERT | Only as a third comparison baseline; never as primary. |
| trafilatura | newspaper3k / readability-lxml | newspaper3k only if you specifically need its author/top-image extraction AND accept it's unmaintained. readability-lxml as a lightweight fallback. |
| shap + transformers-interpret | eli5 | eli5 for quick TF-IDF weight inspection on linear models; it's lighter but less actively maintained. |
| LogisticRegression | LinearSVC | SVC if you need a margin-based baseline; loses native predict_proba (needs calibration) — LR is friendlier for confidence scores. |
| ComplementNB | MultinomialNB | ComplementNB handles class imbalance better on TF-IDF; prefer it given BanFakeNews skew. |
| Streamlit `st.cache_resource` | Gradio | Gradio if you later want a quick shareable demo; Streamlit better matches the brief's multi-section dashboard. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `newspaper3k` | Unmaintained since 2018; breaks on malformed HTML/encoding. | `trafilatura` |
| mBERT as primary classifier | Weakest Bangla representation; vocab fragments Bangla. | BanglaBERT / BanglishBERT / XLM-R |
| Raw attention weights as "the explanation" | Attention ≠ faithful explanation (well-documented); misleads non-technical users. | Integrated Gradients (transformers-interpret/Captum), SHAP, LIME |
| BanglaBERT **without** the `normalizer` | Pretraining-inference mismatch → silent accuracy loss. | Always normalize first |
| NewsAPI free tier for live verification | 24h delay + 100 req/day kills real-time. | Google Fact Check + Wikipedia |
| Twitter/X API for FakeNewsNet social layer | Now paid; social layer is out of scope anyway. | Content-only datasets (ISOT/LIAR/BanFakeNews) |
| Bare TF-IDF without `class_weight`/resampling | BanFakeNews ~40:1 skew → model predicts "real" always. | `class_weight='balanced'` + imbalanced-learn |

## Stack Patterns by Variant

- Fine-tune BanglishBERT + BanglaBERT (and optionally XLM-R) on Colab free T4.
- Export best checkpoint; run CPU inference locally for the demo.
- Cache the model with `st.cache_resource` so it loads once.
- Prefer the ~110M ELECTRA models over XLM-R (270M) for lower latency.
- Keep the classical LR model as a fast first-pass; escalate to transformer for the final verdict.
- Treat malicious class as English-leaning; document as a limitation.
- Optionally augment via translation of SMS-spam/phishing into Bangla; flag synthetic data in the report.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| transformers 4.46.x | torch 2.6.x | Confirmed pairing; `Trainer` needs `accelerate>=0.26`. |
| transformers-interpret 0.10.x | transformers 4.x | Can lag newest Transformers; if it errors, drop to Captum directly. |
| csebuetnlp models | normalizer (git) | Hard dependency for correct accuracy. |
| scikit-learn 1.6.x | shap 0.46.x, lime 0.2.x | Stable; pickle models with the same sklearn minor used to train (avoid load warnings). |
| streamlit 1.40.x | python 3.11 | Use `st.cache_resource` (not deprecated `st.cache`). |

## Sources

- https://huggingface.co/csebuetnlp/banglabert — BanglaBERT ELECTRA discriminator, mandatory normalizer pipeline (HIGH)
- https://huggingface.co/csebuetnlp/banglishbert — Bangla+English code-mixed model (HIGH)
- https://github.com/csebuetnlp/banglabert — official release, NAACL-2022 paper, finetuning code (HIGH)
- https://github.com/csebuetnlp/normalizer — required normalization install (HIGH)
- https://www.kaggle.com/datasets/cryptexcode/banfakenews + https://aclanthology.org/2020.lrec-1.349/ — BanFakeNews size/license (HIGH)
- https://www.kaggle.com/datasets/hrithikmajumdar/bangla-fake-news — BanFakeNews-2.0 counts (MEDIUM, verify license on page)
- https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset — ISOT ~45k balanced (HIGH)
- HF Hub `liar` dataset — LIAR ~12.8k, 6 labels (HIGH)
- https://archive.ics.uci.edu/dataset/228/sms+spam+collection — SMS Spam 5,574 (HIGH)
- https://developers.google.com/fact-check/tools/api — Fact Check Tools API auth/quota (MEDIUM — exact limits loosely documented)
- https://trafilatura.readthedocs.io/ — trafilatura 2.1.x, maintained vs newspaper3k 2018 (HIGH)
- https://pypi.org/project/transformers/ , /scikit-learn/ , /bnlp-toolkit/ — current versions (HIGH; pinned to tested stable minors, not bleeding edge)

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
