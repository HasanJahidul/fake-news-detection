# Feature Research

**Domain:** Unified fake-news + malicious-content detection system (text/URL, Bangla+English, local Streamlit, free APIs)
**Researched:** 2026-06-17
**Confidence:** HIGH (table stakes / fusion / explainability), MEDIUM (Bangla-specific lexicons, free-API verification coverage)

## Feature Landscape

The brief defines a **fixed feature set** (PROJECT.md Active list). This research re-categorizes those features against what real-world products/research systems actually ship, exposes concrete feature definitions, flags complexity, and maps dependencies. The categorization below is opinionated: anything the brief *names as the novelty* (unified multi-module + multilingual + fusion + explainability) is treated as a differentiator; the per-input classification basics are table stakes; the explicit Out-of-Scope items are anti-features.

### Table Stakes (Users Expect These)

Features without which "this isn't the system the brief describes."

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| 3-class classification real/fake/malicious | Core value: text in → verdict out | MEDIUM | Single classifier head OR cascade (binary fake/real + malicious detector). Cascade is easier to train given separate datasets (BanFakeNews has no malicious class; phishing corpora have no real-news class). |
| Confidence score per prediction | Verdict alone is untrustworthy; users need "how sure" | LOW | `predict_proba` (classical) / softmax max-prob (transformer). Calibrate (Platt/temperature) — raw softmax is overconfident. |
| Bangla + English + code-mixed handling | Core to the Bangladesh problem; English-only fails the brief | MEDIUM | Drives transformer choice (XLM-R / multilingual BERT / BanglaBERT). Code-mixed text is the hard case — Banglish in Latin script confuses Bangla-script-only models. |
| Classical ML baselines (LR, NB, RF over TF-IDF) | Brief requires; also the honest comparison baseline | LOW | scikit-learn, fast, CPU. Serves as fallback when transformer latency is an issue. |
| Multilingual transformer primary model | Brief's stated primary model; SOTA for the task | HIGH | HF Transformers. Fine-tune on combined corpus. Latency on local CPU is the real-time risk. |
| Model comparison (acc/P/R/F1, select best) | Brief requires; standard ML rigor | LOW | Hold-out + report table. Watch class imbalance — use macro-F1, not accuracy. |
| Text input | Minimum interaction | LOW | Streamlit `text_area`. |
| URL input + article extraction | Brief requires; real news arrives as links | MEDIUM | `trafilatura` or `newspaper3k` to extract main article text from HTML. Handles boilerplate stripping. Failure modes: paywalls, JS-rendered pages, Bangla news sites with odd encodings. |
| Verdict + confidence + explanation display | "Accuracy plus transparency is the whole point" (PROJECT.md) | LOW | Streamlit result panel. Color-coded verdict, confidence bar, explanation section. |
| Real-time / instant response | Brief requires; UX expectation | MEDIUM | Cache model load; lazy-load transformer; consider classical-first with transformer on demand. External-verification API calls are the latency bottleneck — make them async/optional with timeout. |

### Differentiators (Competitive Advantage)

The brief's stated novelty. These are what separate this from a plain text classifier and from reactive fact-checkers (Snopes/FactCheck.org).

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Unified fake-news + malicious-content in one system** | One tool covers misinformation AND cyber-threats (phishing/scam/malware) — most products do only one | MEDIUM | The core differentiator. Requires the malicious sub-detectors below + fusion to reconcile a "fake" vs "malicious" verdict. |
| Phishing detection (credential/data requests) | Catches a class fact-checkers ignore | LOW–MEDIUM | Text-pattern based: keyword/intent lexicon ("verify your account", "login", "update password", "click to confirm") + urgency cues + brand-name-near-action. Combine with URL heuristics below. Bangla phishing/scam lexicon must be hand-built — scarce. |
| Scam detection (fake job, lottery, financial fraud) | High local relevance (Bangladesh scam prevalence) | MEDIUM | Text-pattern + intent lexicon: money promises, "you won", advance-fee, fake-job ("work from home, earn ৳X/day"), payment requests. Largely lexicon + classifier; Bangla examples scarce → curate from local sources. |
| Malware-link / suspicious-URL flagging | Protects against the link-based threat vector | MEDIUM | URL **heuristics** (no paid threat-intel): count of `@`, IP-as-host, hostname length, subdomain depth, URL length, hyphens, suspicious TLDs, punycode/homograph, URL shorteners, brand-lookalike (Levenshtein vs known domains), `https` absence. Optionally a **free blocklist** (e.g. URLhaus, OpenPhish free feed, PhishTank) loaded locally. No live malware scanning. |
| Source credibility module | Adds a signal text alone can't give; flags low-rep domains | MEDIUM | Domain → reputation score. Build a **local curated JSON list** seeded from open MBFC-style factual-accuracy ratings + Bangladesh-specific known-unreliable domains. Score = base domain reputation × (1 − historical false-content frequency). Maintained as a static asset (no live scraping). Unknown domain → neutral prior. |
| Writing-style / behavioral module | Catches sensational/manipulative tone independent of facts | LOW | Cheap, interpretable features (see definitions below). Strong explainability payoff. Lexicons need Bangla translation. |
| External verification module (free APIs) | Moves beyond content-only to evidence-based — the "multi-layer validation" novelty | HIGH | Claim extraction → query Google **Fact Check Tools API** (free, 70+ langs) + Wikipedia/Wikipedia API + (optionally) free news search → cross-source consistency. Highest-risk module: API coverage for Bangla claims is thin; rate limits; latency. Design to degrade gracefully (returns "no evidence found" not failure). |
| Decision fusion (model + 3 modules → 1 verdict + confidence) | The integration novelty; turns 4 signals into one trustworthy answer | MEDIUM–HIGH | **Recommended: weighted-vote / weighted-average of module scores with rule-based overrides**, not learned stacking (insufficient labeled multi-module training data). Rule overrides that work in practice: (1) strong malicious-URL/phishing hit → force `malicious` regardless of news classifier; (2) external verification strongly refutes → push toward `fake`; (3) high source credibility + no refutation → dampen fake score. Stacking/meta-learner is the academic alternative but needs a fusion training set the project likely won't have. |
| Word/phrase highlighting explainability | "Show why" — directly serves Core Value | MEDIUM | Classical: TF-IDF coefficient × token presence, or LIME/SHAP. Transformer: attention or Integrated Gradients/SHAP (slower). Style module: highlight matched clickbait/scam terms directly (free, exact). |
| Per-module contribution display | User sees *which* signal drove the verdict (low credibility vs clickbait vs refuted claim) | LOW | Each module returns (score, reason string). UI lists contributions with weights. Cheap given modules already produce scores — high transparency value. |
| Modular / swappable architecture | Modules retrainable, swappable on new datasets (brief system quality) | LOW–MEDIUM | Common module interface `(input) → {score, label, evidence, reason}`. Enables fusion + isolated testing. |

### Anti-Features (Commonly Requested, Often Problematic)

Explicitly NOT building (most are PROJECT.md Out-of-Scope; restated with the trap and the alternative).

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Multimodal (image/video/deepfake) detection | Misinformation is often visual | Huge scope, separate models, GPU-heavy, out of scope for text-focused capstone | Text-only; extract article text from URL, ignore media. |
| Social-graph / spread modeling (FakeNewsNet social layer) | "Virality predicts fakeness" | Needs social-network data, real-time graph infra, privacy issues; not available for single-input demo | Use content + source + verification signals only. |
| Paid APIs (NewsAPI paid, NewsGuard, paid LLM/fact-check) | Better coverage/accuracy | No keys/budget; not reproducible for evaluation | Free tier only: Google Fact Check Tools, Wikipedia, open blocklists, curated MBFC-derived list. |
| Live blocklist / threat-intel feeds with auto-refresh | "Always current malware list" | Rate limits, network dependency at inference, flaky in demo | Bundle a **snapshot** blocklist as a static asset; refresh manually/batch. |
| Continuous online / adaptive auto-learning | "Model keeps improving" | Concept-drift handling, data poisoning risk, eval instability; explicitly out of scope | Manual/batch retraining only; document retraining procedure. |
| High-throughput streaming / firehose ingestion | "Scale to platform volume" | Out of scope; single-input real-time is the target | Single text/URL per request, optimized for instant response. |
| Cloud/public hosting (Docker/HF Spaces) this milestone | "Make it accessible" | Hosting complexity, secrets handling, out of scope | Local Streamlit for demo/evaluation. |
| LLM-as-judge full-article reasoning (e.g. GPT-4 verdict) | "Just ask an LLM" | Paid/rate-limited, non-reproducible, opaque (undercuts explainability goal), latency | Transparent classical+transformer+rule modules with traceable per-module reasons. |
| Auto-fetch unlimited related coverage per claim | "More evidence = better" | Free-API rate limits + latency kill real-time UX | Bounded queries (top-k results, hard timeout), cache, degrade to "no evidence found." |

## Concrete Feature Definitions (for requirements)

**Writing-style / behavioral features** (per input; each contributes to style score + explanation):
- `caps_ratio` = uppercase-letter count / total-letter count (Latin script); flag if above threshold (e.g. >0.3) on substantive text. (Bangla has no case — apply only to Latin-script tokens.)
- `exclaim_count` / `question_count` and `excess_punct` = runs of `!!!`/`???`/`!?` ; flag repeated punctuation.
- `clickbait_lexicon_hits` = matches against a curated clickbait term list ("shocking", "you won't believe", "insane", "exposed", Bangla equivalents).
- `scam_lexicon_hits` / `phishing_lexicon_hits` = matches against intent lexicons (money/winning/urgency/credential terms).
- `sentiment_extremity` = |sentiment polarity| (extreme polarity ⇒ sensational). Use a multilingual/Bangla-capable sentiment lexicon or model.
- `repetition` = max token/phrase repeat ratio (spammy repetition).
- `intensifier_count`, `superlative_count` = hyperbole markers.

**URL heuristics** (per extracted/embedded URL):
- structural: url length, hostname length, subdomain depth, count of `.` `-` `@` `//`, IP-as-host, port present, `https` present.
- lexical: suspicious TLD, URL-shortener domain, punycode/homograph, digits-in-domain ratio.
- brand-lookalike: min edit distance to a known-brand domain list (typosquatting).
- blocklist hit: membership in bundled OpenPhish/URLhaus/PhishTank snapshot.

**Source credibility** (per domain):
- `domain_reputation` from curated JSON (factual-accuracy tier derived from open MBFC-style ratings + local list); unknown ⇒ neutral.
- `false_content_history` ratio if tracked; credibility_score = reputation × (1 − history_penalty).

**External verification** (per claim):
- claim extraction: take headline / top sentences as query string (lightweight; full claim-detection NLP is optional).
- fetch: Google Fact Check Tools `claims.search` (lang-filtered) + Wikipedia search/summary.
- consistency: do returned fact-checks/articles support, refute, or not mention the claim → {supported, refuted, no-evidence}.

## Feature Dependencies

```
3-class classification ──requires──> classical + transformer models ──requires──> preprocessed Bangla+EN corpus
URL input ──requires──> article extraction ──feeds──> classification + style + source-credibility + verification
phishing/scam detection ──uses──> style/intent lexicons
malware-link flagging ──requires──> URL heuristics + bundled blocklist
source credibility ──requires──> curated domain JSON + URL/domain parse
external verification ──requires──> claim extraction + free-API clients (Fact Check, Wikipedia)
explainability (highlighting) ──requires──> per-token model attributions + style/scam lexicon matches
per-module contribution ──requires──> common module interface (score + reason)

DECISION FUSION ──requires──> ALL OF:
    classification model output
    + source-credibility score
    + writing-style score
    + external-verification result
    + (malicious sub-detector outputs)
        └── fusion is the integration point; it cannot be built or tested until its inputs exist

UI result display ──requires──> fusion output + explainability outputs
```

### Dependency Notes

- **Fusion is the keystone dependency:** it consumes every module's output. Roadmap MUST sequence all modules (models, credibility, style, verification, malicious detectors) before — or with stubbed interfaces ahead of — fusion. Define the common module interface `(input) → {score, label, evidence, reason}` early so modules can be built in parallel and fusion wired against stubs.
- **Explainability depends on the classifier internals:** word highlighting needs model attributions; choosing LIME/SHAP vs attention is a model-coupled decision. Style/scam lexicon highlighting is independent and cheap — ship it first as the explainability MVP.
- **URL input enables three downstream modules** (classification on extracted text, source credibility on the domain, verification on the headline). Article extraction quality gates all three.
- **Verification conflicts with real-time UX:** free-API latency/rate-limits oppose the "instant" requirement. Make verification timeout-bounded and optionally async; never block the verdict on it.
- **Bangla support cross-cuts everything:** every lexicon (clickbait, scam, phishing, sentiment) needs a Bangla variant; verification API coverage for Bangla is weaker. Treat Bangla lexicon curation as its own work item, not a free add-on.

## MVP Definition

### Launch With (v1) — the brief mandates a full pipeline, so MVP ≈ thin-but-complete

- [ ] 3-class classification + confidence — core value
- [ ] Classical baselines + one multilingual transformer + comparison — brief requires
- [ ] Bangla+English text handling — core to problem
- [ ] Text + URL input with article extraction — brief requires
- [ ] Malicious sub-detection: phishing + scam (lexicon) + URL heuristics + bundled blocklist — the unification novelty
- [ ] Source credibility (curated domain JSON) — multi-layer validation
- [ ] Writing-style module (caps/punct/clickbait/scam/sentiment) — cheap, high explainability
- [ ] External verification (Google Fact Check + Wikipedia, bounded, degradable) — multi-layer validation
- [ ] Decision fusion (weighted vote + rule overrides) — integration novelty
- [ ] Explainability: lexicon-match highlighting + per-module contribution panel
- [ ] Streamlit UI: verdict + confidence + explanation

### Add After Validation (v1.x)

- [ ] Model-internal token attribution (LIME/SHAP/IG) — once classifier stable; richer highlighting
- [ ] Confidence calibration (temperature scaling) — once eval shows overconfidence
- [ ] Expanded Bangla scam/phishing lexicons from local data — as misclassifications surface
- [ ] BanglaBERT vs XLM-R head-to-head — once data pipeline solid

### Future Consideration (v2+)

- [ ] Learned stacking/meta-learner fusion — only if a fusion-labeled set becomes available
- [ ] Cloud deployment — out of scope this milestone
- [ ] Multimodal — explicitly out of scope

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| 3-class classification + confidence | HIGH | MEDIUM | P1 |
| Bangla+EN handling | HIGH | MEDIUM | P1 |
| Text + URL input + extraction | HIGH | MEDIUM | P1 |
| Decision fusion (weighted+rules) | HIGH | MEDIUM | P1 |
| Per-module contribution display | HIGH | LOW | P1 |
| Lexicon-match highlighting | HIGH | LOW | P1 |
| Writing-style module | MEDIUM | LOW | P1 |
| Source credibility (curated JSON) | MEDIUM | MEDIUM | P1 |
| Phishing/scam lexicon detection | HIGH | LOW–MEDIUM | P1 |
| Malware-link URL heuristics + blocklist | MEDIUM | MEDIUM | P1 |
| External verification (free APIs) | HIGH | HIGH | P2 (ship bounded/degradable) |
| Model-internal token attribution | MEDIUM | MEDIUM | P2 |
| Confidence calibration | MEDIUM | LOW | P2 |
| Learned stacking fusion | LOW | HIGH | P3 |

**Priority key:** P1 = must have for launch · P2 = should have, add when possible · P3 = future.

## Competitor / Prior-Art Feature Analysis

| Feature | Reactive fact-checkers (Snopes/FactCheck.org) | Phishing/spam tools (Phish Mail Guard, blocklists) | Explainable FND research (XFake/ExFake/VeraCT) | Our Approach |
|---------|-----------------------------------------------|----------------------------------------------------|------------------------------------------------|--------------|
| Misinformation detection | Manual, post-hoc | None | Automated content+context | Automated content+source+verification, instant |
| Malicious/cyber detection | None | URL heuristics + blocklists + text tokens | None | Lexicon + URL heuristics + bundled blocklist |
| Unified mis+malicious | No | No | No | **Yes — the novelty** |
| Source credibility | Implicit | Domain blocklists | Source credibility graphs | Curated MBFC-derived domain JSON |
| External verification | Human | No | RAG / evidence retrieval | Google Fact Check + Wikipedia, bounded |
| Fusion | N/A | Heuristic threshold | Stacking / meta-learning | Weighted vote + rule overrides |
| Explainability | Article prose | Matched-token reasons | Word importance + visualizations | Lexicon highlighting + per-module contributions |
| Multilingual (Bangla) | Limited | Mostly English | Mostly English | **Bangla+English+code-mixed — the novelty** |

## Sources

- Explainable FND systems (XFake, ExFake, VeraCT Scan, evidence-fusion): https://arxiv.org/pdf/1907.07757 , https://arxiv.org/pdf/2311.10784 , https://arxiv.org/pdf/2406.10289 , https://www.mdpi.com/2504-2289/8/10/129
- Phishing/scam detection (URL heuristics, blocklists, text tokens): https://arxiv.org/pdf/2111.01676 , https://www.nature.com/articles/s41598-022-10841-5 , https://www.ijcaonline.org/archives/volume187/number14/phishing-and-spam-detection-based-on-url-heuristics-and-email-text-analysis/
- Google Fact Check Tools API (free, 70+ languages): https://developers.google.com/fact-check/tools/api , https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search
- Decision fusion (weighted vote, rule-based override, stacking): https://arxiv.org/pdf/2507.09174 , https://arxiv.org/pdf/2602.14441 , https://www.researchgate.net/publication/373933617_A_Decision-Fusion-Based_Ensemble_Approach_for_Malicious_Websites_Detection
- Clickbait/style features (caps, punctuation, sensational, sentiment): https://arxiv.org/pdf/2509.10937 , https://www.mdpi.com/2076-3417/13/4/2456 , https://lettersandsciencemag.ucdavis.edu/self-society/if-social-media-post-has-any-these-ten-features-its-probably-clickbait
- Source credibility lists/methodology (MBFC, open domain-quality data): https://mediabiasfactcheck.com/methodology/ , https://link.springer.com/article/10.1140/epjds/s13688-026-00628-3

---
*Feature research for: unified fake-news + malicious-content detection*
*Researched: 2026-06-17*
