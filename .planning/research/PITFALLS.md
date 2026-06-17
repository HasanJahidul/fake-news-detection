# Pitfalls Research

**Domain:** Multilingual (Bangla+English) fake-news + malicious-content detection with hybrid classical/transformer models, multi-signal fusion, external verification, explainability, Streamlit UI
**Researched:** 2026-06-17
**Confidence:** HIGH (dataset leakage, BanFakeNews imbalance, BanglaBERT normalization verified against sources; fusion/eval/ethics pitfalls from established ML practice)

> This domain is unusually prone to *looking* successful while being broken. A 99% accuracy number is the single biggest red flag in this project, not an achievement. Almost every critical pitfall below produces inflated metrics that collapse on real input. Treat high benchmark accuracy as a bug to investigate until proven otherwise.

## Critical Pitfalls

### Pitfall 1: Source-artifact label leakage (model learns the publisher, not the content)

**What goes wrong:**
Classifier hits 99-100% accuracy but only because real and fake samples come from different sources with different surface artifacts. The classic case is the ISOT / Kaggle "Fake-and-Real-News" dataset: all real articles were crawled from Reuters.com and many literally begin with the token `Reuters` / a `(Reuters)` dateline, while fake articles came from flagged sites. The model learns "starts with Reuters → real" — a publisher detector, not a misinformation detector. The same trap exists in BanFakeNews (authentic articles scraped from a fixed set of mainstream Bangla outlets; fake from a different set), and gets *worse* when you naively merge corpora because each source-class is internally homogeneous.

**Why it happens:**
Datasets are assembled by scraping "real" from one place and "fake" from another. Boilerplate (datelines, bylines, outlet names, URLs, "Image: AFP", standard footers, social-share text) correlates perfectly with label. TF-IDF and transformers both happily exploit it.

**How to avoid:**
- Strip source/boilerplate before training: remove `(Reuters)`, datelines, outlet names, bylines, URLs, image credits, standard footers, "Read more" tails. Build an explicit boilerplate-stripping step in preprocessing and apply it identically at inference.
- Run a leakage probe: train on title-only, then on a single sentence sampled from the body, then on text with the first N tokens removed. If accuracy barely drops when you remove the body, you're memorizing artifacts.
- Do source-disjoint evaluation: hold out *entire sources/domains* for the test set so the model can't pass by recognizing a publisher.
- Inspect top TF-IDF coefficients / SHAP features early — if the most predictive tokens are outlet names, dates, or wire-service tags, stop and fix data.

**Warning signs:**
Accuracy ≥ 98% on first try; top features are proper nouns/outlet names/years; model confident on a paraphrase that strips the source tokens flips its answer.

**Phase to address:** Data foundation (corpus build + preprocessing), *before* any model training.

---

### Pitfall 2: Extreme class imbalance reported as accuracy

**What goes wrong:**
BanFakeNews is ~97% authentic / ~3% fake (≈48,678 authentic vs ≈1,299 fake — authentic is ~37x larger). A model that predicts "real" for everything scores ~97% accuracy and ~0% recall on fake — the only class that matters. Adding a third `malicious` class (typically pulled from separate phishing/spam corpora) makes the joint distribution even more skewed and arbitrary.

**Why it happens:**
Accuracy is the default metric and looks great on skewed data. The interesting class (fake/malicious) is the minority, so missing all of it costs almost nothing in accuracy.

**How to avoid:**
- Never report accuracy as the headline. Report per-class precision/recall/F1, macro-F1, and the confusion matrix. Make macro-F1 (or fake-class recall) the model-selection criterion.
- Address imbalance deliberately: class weights (`class_weight='balanced'`, transformer weighted loss), threshold tuning, and/or controlled augmentation. Prefer class weighting over naive oversampling that duplicates leakage. If using SMOTE, do it on TF-IDF features *inside* the CV fold only.
- Consider BanFakeNews-2.0 (~47k authentic / ~13k fake) for a less degenerate balance, or LLM-based augmentation of the fake class — but augmented samples must be held out of the test set.
- Decide deliberately whether `malicious` is a peer class or a separate binary head; a single 3-way softmax over wildly imbalanced, semantically inconsistent sources learns the dataset-of-origin, not the concept.

**Warning signs:**
High accuracy + low recall on fake/malicious; confusion matrix shows an empty minority row; macro-F1 far below accuracy.

**Phase to address:** Classical baseline phase (set metric discipline here) and transformer phase.

---

### Pitfall 3: Train/test contamination and cross-dataset overfit

**What goes wrong:**
Duplicate or near-duplicate articles (same story republished, same fake copy-pasted) land in both train and test, inflating scores. Splitting *after* augmentation/oversampling leaks synthetic twins. And tuning everything on one dataset's test split yields a number that does not generalize to a second dataset or to real pasted text.

**Why it happens:**
`train_test_split` on rows ignores duplicates and source clusters. Many fake-news corpora contain heavy duplication. Single-dataset evaluation is the path of least resistance.

**How to avoid:**
- Dedup (exact + near-dup via minhash/cosine on TF-IDF) before splitting.
- Split by *source/domain group*, not random rows (GroupShuffleSplit).
- Augment/oversample *inside* the training fold only.
- Reserve at least one external/held-out dataset or a small hand-collected real-world set (recent Bangla + English items, including code-mixed) for honest final evaluation — and expect lower numbers there. That lower number is the truthful one.

**Warning signs:**
Test ≈ train accuracy; performance craters when you paste a fresh real-world article; identical texts found in both splits.

**Phase to address:** Data foundation (dedup + grouped split); Evaluation phase (external held-out set).

---

### Pitfall 4: Merging fake-news + phishing/scam datasets with inconsistent label semantics

**What goes wrong:**
The 3-class target (`real / fake / malicious`) is assembled from incompatible sources: long news articles for real/fake, short SMS/email for phishing/scam. The model learns "short text with a link/urgent tone → malicious; long formal text → real/fake" — it classifies *text length and register*, not intent. A long, well-written scam article or a short legitimate news blurb breaks it instantly.

**Why it happens:**
The three classes come from different worlds (news corpora vs. SMS-spam/phishing corpora) with different length, formality, and domain. The join is by convenience, not by a coherent annotation scheme.

**How to avoid:**
- Define the label scheme explicitly and document the decision boundary between `fake` and `malicious` (misinformation vs. intent-to-harm/defraud). Borderline cases (sensational scam-y news) need a rule.
- Normalize length/register confounds: report performance stratified by text length; verify the model isn't just a length classifier (probe with length-matched samples).
- Strongly consider a two-stage architecture: a malicious/benign gate (handles phishing/scam/URL signals) feeding into, or parallel with, a real/fake news classifier — rather than one flat softmax over three incoherent distributions.
- Keep per-source provenance on every sample so you can audit which class is carrying which artifact.

**Warning signs:**
Malicious class predicted almost entirely by text length or presence of a URL; model labels any short message malicious; per-class confusion concentrated between fake and malicious.

**Phase to address:** Data foundation (label schema) and architecture decision (two-stage vs flat) — decide before training.

---

### Pitfall 5: Bangla text not normalized before tokenization

**What goes wrong:**
Bangla Unicode has multiple code-point sequences that render identically (nukta variants, ya-phala/ref ordering, zero-width joiners ZWJ/ZWNJ, Bangla vs ASCII digits, dari `।` vs period). Without normalization, the same word becomes different token sequences, the tokenizer over-fragments into `[UNK]`/subwords, and train/inference drift if you normalize one but not the other. csebuetnlp BanglaBERT/BanglaT5 *require* their normalizer pipeline before tokenizing — skipping it measurably degrades results.

**Why it happens:**
English-centric preprocessing habits (lowercase, strip punctuation) don't cover Bangla. Devs assume HuggingFace tokenizer "just works." Normalization is an extra, easy-to-forget dependency.

**How to avoid:**
- Use the official `csebuetnlp/normalizer` (`pip install git+https://github.com/csebuetnlp/normalizer`) or `banglanlptoolkit` and run it *before* tokenization, for the matching model. Don't pair BanglaBERT with a different normalizer than it was pretrained with.
- Put normalization in ONE shared preprocessing function imported by both training and the Streamlit inference path — never two copies.
- Do NOT lowercase or strip Bangla punctuation blindly; keep dari, normalize ZWJ/ZWNJ, unify digits consistently.
- Smoke-test the tokenizer on real Bangla + code-mixed samples and inspect token counts / `[UNK]` rate before training.

**Warning signs:**
High `[UNK]` rate on Bangla; same word tokenized differently across samples; accuracy on Bangla far below English; inference results differ from eval on identical text.

**Phase to address:** Data foundation / preprocessing — and lock the shared function before the transformer phase.

---

### Pitfall 6: Wrong transformer for Bangla / code-mixed text

**What goes wrong:**
Picking vanilla `bert-base-multilingual-cased` (mBERT) as "the multilingual model" — it is weak on Bangla relative to Bangla-specific models. Or picking a Bangla-only model (BanglaBERT) that handles English poorly, when the project explicitly needs code-mixed Bangla+English. Either way one language silently underperforms.

**Why it happens:**
"Multilingual BERT" sounds sufficient; model choice is made before testing on representative code-mixed input.

**How to avoid:**
- Evaluate candidates head-to-head on a *code-mixed* validation set: BanglaBERT (csebuetnlp), XLM-R (strong multilingual, handles both languages), mBERT (baseline). Report per-language and code-mixed F1, not just overall.
- Match the normalizer to the chosen model (Pitfall 5 dependency).
- Budget for the real-time constraint: XLM-R base is heavier than BanglaBERT-small; measure latency on the target local hardware (Pitfall 13) as part of selection, not after.

**Warning signs:**
One language's F1 lags the other by a wide margin; code-mixed inputs misclassified; model card shows no Bangla benchmark.

**Phase to address:** Transformer model phase (selection experiment).

---

### Pitfall 7: Topic / temporal bias inflating accuracy

**What goes wrong:**
Fake articles cluster on certain topics/time periods (e.g. a single election, COVID, one celebrity) and real on others. Model learns the *topic/era*, not deception. Reported accuracy is high; on a new topic or a later time period it collapses. Especially acute with ISOT (2015-2018 only) and any single-event-heavy Bangla scrape.

**Why it happens:**
Datasets are snapshots. Fake-news generation is bursty and topical. Random splits keep topic/time distribution identical across train/test, hiding the dependency.

**How to avoid:**
- Time-based split where dates exist: train on older, test on newer.
- Inspect topic distribution per class (quick LDA/keyword scan); if topics separate the classes, the model will cheat on them.
- Include the gap explicitly in the report: "evaluated on 2015-2018 distribution; real-world current performance will differ."

**Warning signs:**
Strong predictive tokens are topical proper nouns/years; performance drops on out-of-period or off-topic samples.

**Phase to address:** Evaluation phase; flag in Data foundation.

---

### Pitfall 8: Phishing detection that memorizes keywords + uses inference-unavailable URL features

**What goes wrong:**
(a) Phishing/scam classifier keys on brittle keywords ("lottery", "OTP", "verify your account", "Nagad/bKash bonus") — trivially evaded by rephrasing, and false-positive on legitimate messages mentioning those words. (b) The model is trained with rich URL/network features (WHOIS age, DNS, page HTML, redirect chains, SSL) that are NOT available — or are too slow/unreliable to fetch — at real-time inference in a local Streamlit app. Training scores look great; deployed model can't compute the features it depends on.

**Why it happens:**
Academic phishing datasets ship pre-extracted URL features; copying them yields high accuracy. Nobody checks feature availability at the actual inference moment. Keyword overfit is the default for small spam corpora.

**How to avoid:**
- Define a strict "available-at-inference" feature contract. For pasted text + a URL string, you realistically have: lexical URL features (length, subdomain count, `@`, IP-literal host, punycode, suspicious TLD, look-alike/homoglyph domain, shortener), and the message text — NOT WHOIS/DNS/live page unless you commit to fetching them (then handle timeouts/failures gracefully and degrade).
- Train only on features the inference path can produce; if a feature requires a network call, make it optional with a defined fallback, and test the no-network path.
- Combat keyword overfit with character/subword features, adversarial paraphrase tests, and a held-out set of *legitimate* messages containing scam-adjacent words (banking, prizes) to measure false positives.

**Warning signs:**
Top features are bag-of-keywords; flipping one word changes the verdict; a feature column is null/constant at inference; legit bank SMS flagged as phishing.

**Phase to address:** Malicious-detection module phase (define feature contract first).

---

### Pitfall 9: Stale blocklists / domain-credibility list presented as live truth

**What goes wrong:**
Source-credibility module relies on a hardcoded JSON of "trusted/untrusted domains" and a static malware/phishing blocklist. These go stale fast: new scam domains appear daily, trusted outlets get compromised, and a small hand-built list has near-zero coverage of the Bangla web. The module then contributes confident-but-wrong signal (unknown domain → treated as untrusted, or vice versa).

**Why it happens:**
A static dict is the easy MVP. Nobody schedules refresh or handles the "unknown domain" case.

**How to avoid:**
- Treat unknown domains as *unknown* (neutral / abstain), not as a strong signal in either direction. Never let "not in my list" mean "fake."
- Keep the credibility list small, documented, and explicitly labeled as a heuristic with limited coverage; surface its limitation in the explanation ("source not in our reference list").
- If using any live reputation source, cache with TTL and handle absence gracefully.

**Warning signs:**
Credibility score dominated by a binary in/out-of-list flag; every unknown Bangla domain scored identically; verdict driven by domain alone on content that contradicts it.

**Phase to address:** Source-credibility module phase.

---

### Pitfall 10: Fusion double-counts correlated signals or just echoes the transformer

**What goes wrong:**
The four signals (transformer, credibility, style, verification) are combined, but: (a) style and transformer both react to clickbait/sensational tone → the same evidence counted twice, over-confident verdicts; (b) the transformer is so much stronger that any weighted fusion just tracks it, making the three modules decorative; (c) modules output scores on incomparable scales (one is 0-1 probability, one is a raw count, one is -1/0/+1) so naive weighted-sum is dominated by whichever has the largest range.

**Why it happens:**
Fusion is bolted on last with hand-picked weights and no validation that it beats the transformer alone. Score calibration across heterogeneous modules is overlooked.

**How to avoid:**
- Calibrate every module to a common, comparable scale (e.g. probability via Platt/isotonic, or a documented 0-1 confidence) before fusion. Don't sum raw heterogeneous scores.
- Prove fusion earns its place: report transformer-alone vs. full-fusion on the held-out set. If fusion doesn't improve macro-F1 (or improves explanation without hurting accuracy), say so honestly rather than faking weights.
- Check signal correlation: if style and transformer are highly correlated, down-weight or treat style as an *explanation* layer, not an independent vote.
- Prefer a small learned meta-classifier (logistic regression over module scores, trained on a validation fold) over hand-tuned weights — and freeze it; never tune it on the test set.
- Handle module abstention explicitly (verification often returns "no evidence" — that must not be coerced to "fake").

**Warning signs:**
Full system == transformer-alone within noise; one module's weight makes no difference when ablated; fused confidence pinned near 0/1; verification's "unknown" pushing verdicts.

**Phase to address:** Fusion phase (with an ablation/calibration gate in success criteria).

---

### Pitfall 11: External verification — free-API limits, circular sourcing, claim-extraction failure

**What goes wrong:**
- Google Fact Check Tools API: standard Google API key default quota (~10k req/day, low QPS) and, critically, *sparse coverage of Bangla and of niche/recent claims* — most pasted items return zero matches, so the module abstains almost always and adds little. Wikipedia/news free APIs rate-limit and 429 under any load.
- Claim extraction fails: you need a checkable claim to query; long pasted articles or vague text don't yield a clean query, so the search returns irrelevant hits.
- Circular sourcing: the "trusted source" you verify against may be the same outlet that published the item, or an aggregator that copied it — confirming a claim with itself.
- Latency: live fetch + parse + API round-trips per request destroys the "instant" UX requirement (Pitfall 13).

**Why it happens:**
Verification is the flashiest module and assumed to "just work." Free-tier reality (quota, coverage, latency, 429s) and the hard NLP problem of claim extraction are underestimated.

**How to avoid:**
- Design the module to *abstain gracefully and visibly* — "no external evidence found" is a valid, common output and must not be coerced into a verdict.
- Cache aggressively (per-claim/per-URL TTL cache) to survive quotas and repeated demo inputs; handle 429/timeout/outage with fallback to "verification unavailable."
- Keep verification queries cheap and bounded: extract a short candidate claim (top sentence by entity/keyword density), cap to one or two API calls, run with a strict timeout, and run it async / non-blocking so the verdict can render before verification returns (progressive enhancement).
- Guard against circular sourcing: exclude the input's own domain from "corroborating" sources; require corroboration from a *different* outlet.
- Set expectations in the report/UI: verification is a best-effort signal with low Bangla coverage, not ground truth.

**Warning signs:**
Verification almost always "no result"; 429s in logs; UI hangs for seconds; a claim "verified" by its own source; quota exhausted mid-demo.

**Phase to address:** Verification module phase; UX/latency revisited in Integration phase.

---

### Pitfall 12: Misleading explanations — highlights and "attention" that don't reflect the real reason

**What goes wrong:**
The explanation highlights words that look plausible but aren't actually what drove the verdict — e.g. showing raw transformer attention weights as "evidence." Attention is not explanation (well-established result); attention often spreads to separators/frequent tokens and can be manipulated without changing the prediction. A confident, pretty highlight that's wrong is worse than none — it manufactures false trust, which directly undercuts this project's core value (transparency).

**Why it happens:**
Attention weights are easy to extract and visualize, so they get shipped as "explainability." Faithfulness of the explanation to the actual decision is rarely checked.

**How to avoid:**
- Use a faithfulness-oriented method: for the classical model, TF-IDF coefficients / SHAP are genuinely tied to the decision; for the transformer use input-attribution (Integrated Gradients / SHAP / LIME) rather than raw attention.
- Keep explanations honest about which *module* fired (low credibility / clickbait style / contradicted by source) — the contributing-factor breakdown is often more truthful and useful than token highlights.
- Sanity-check faithfulness: remove the highlighted tokens and confirm the prediction actually changes (deletion test). If it doesn't, the highlight is decorative.
- Never present a single highlighted word as "the reason" with high confidence on a borderline call.

**Warning signs:**
Highlights land on stopwords/punctuation/separators; removing highlighted words doesn't change the verdict; explanation contradicts which module actually voted.

**Phase to address:** Explainability phase.

---

### Pitfall 13: Real-time UX killed by transformer + live-fetch latency

**What goes wrong:**
The project requires "instant" response on local hardware, but a transformer forward pass (especially XLM-R base on CPU) plus URL fetch + article extraction + verification API calls can take many seconds per request. Loading the model on every request (no caching) makes it worse. The "real-time" requirement silently fails in the demo.

**Why it happens:**
Latency isn't measured until the UI is wired up; model and pipeline are re-initialized per request; verification runs synchronously and blocks rendering.

**How to avoid:**
- Cache the model/pipeline in Streamlit (`@st.cache_resource`) — load once, reuse.
- Measure end-to-end latency early on target hardware; pick model size with latency as a selection criterion (Pitfall 6). Consider a smaller/distilled or quantized model if needed.
- Make slow signals (verification, live URL fetch) async/progressive: render the fast text-model verdict immediately, then fill in verification when it returns; show a spinner, never a frozen UI.
- Bound and timeout every network call; truncate very long inputs to the model's max length deliberately (and tell the user).

**Warning signs:**
Multi-second waits per click; CPU pegged; model reloads in logs each request; UI frozen during verification.

**Phase to address:** Integration / Streamlit UI phase (with a latency budget in success criteria).

---

### Pitfall 14: Ethical — dialect/topic bias, over-confident wrong verdicts, privacy of submitted text

**What goes wrong:**
- The model under-performs on regional Bangla dialects, informal/romanized Bangla, or particular topics/communities, systematically mislabeling them — a fairness harm, and the system may flag legitimate minority-dialect content as fake/malicious.
- The system renders a confident `fake`/`malicious` verdict on content it's actually unsure about; users over-trust it (automation bias) and a wrong "fake" label on true content (or "real" on a scam) causes real harm.
- Submitted text (possibly private messages, personal data, the content of pasted scam SMS) is logged/retained, violating the project's own privacy constraint.

**Why it happens:**
Training data skews to formal standard Bangla and mainstream topics. Confidence is reported as raw softmax (uncalibrated, overconfident). Logging is added for debugging and never removed.

**How to avoid:**
- Report performance stratified by language/register where possible; acknowledge dialect/code-mixed gaps explicitly in the report.
- Calibrate confidence (temperature scaling / isotonic) and define an *abstain / "uncertain"* band — the UI should say "uncertain, verify manually" rather than force a confident wrong label. Always pair verdict with a "this is an automated estimate, not a fact-check" disclaimer.
- Privacy by default: do not persist submitted text or URLs beyond the request; no analytics logging of content; if caching verification results, cache the derived claim/result, not raw user input, and document retention = none.

**Warning signs:**
Confidence almost always near 100%; no "uncertain" outcomes ever shown; submitted text appearing in logs/files; complaints that informal Bangla is over-flagged.

**Phase to address:** Cross-cutting — calibration in Fusion phase, privacy + disclaimer in UI phase, fairness reporting in Evaluation phase.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use ISOT/Kaggle real+fake as-is, no boilerplate strip | Instant 99% accuracy, easy demo | Entire system is a publisher detector; collapses on real input; invalidates the thesis claim | Never (the inflated number is the failure) |
| Single 3-way softmax over merged news+phishing data | One model, simple code | Learns length/register/source, not intent; brittle | Only as a throwaway baseline, clearly labeled |
| Hardcoded domain-credibility JSON | Module "works" for demo | Stale, near-zero Bangla coverage, false signal on unknown domains | MVP only, IF unknown→neutral and limitation disclosed |
| Hand-tuned fusion weights | Ships fusion fast | Untestable, likely just echoes transformer, can't justify in report | MVP only, IF ablation vs transformer-alone is reported |
| Raw attention as "explanation" | Easy to render | Misleading/unfaithful, manufactures false trust | Never present as faithful; use only as a rough visual with caveat |
| Synchronous verification in request path | Simpler control flow | Multi-second freezes, quota stalls, fails "real-time" | Never for the blocking path; make it progressive |
| Random train/test split | One line of code | Duplicate + topic + source leakage → inflated metrics | Never for reported numbers; grouped/time split required |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Google Fact Check Tools API | Assume broad coverage; no key quota handling; treat "no match" as "real" | Expect sparse Bangla coverage; cache + handle quota/429; "no evidence" = abstain |
| Wikipedia / free news APIs | Synchronous, no rate-limit handling | Async, cached, timeout, graceful "unavailable" fallback |
| URL article extraction (newspaper3k/trafilatura/readability) | Assume extraction always succeeds; feed page chrome to model | Detect extraction failure; strip nav/ads/boilerplate; fall back to manual-paste prompt |
| BanglaBERT tokenizer | Tokenize without csebuetnlp normalizer; mismatched normalizer/model pair | Run matching official normalizer before tokenizing, same fn at train+inference |
| HuggingFace pipeline in Streamlit | Reload model per request | `@st.cache_resource` load-once |
| Live phishing/reputation lookups | Train on WHOIS/DNS features unavailable at inference | Lexical URL features only, or optional network features with tested fallback |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Transformer forward pass on CPU | Multi-second per verdict | Cache model; smaller/quantized model; truncate input | Immediately on first real use / live demo |
| Model reload per request | Latency + memory spikes each click | `@st.cache_resource` | Every request |
| Synchronous verification API calls | UI freezes seconds; quota stalls | Async/progressive render, timeout, cache | Under any latency in API or repeated demo inputs |
| Live URL fetch + parse in request path | Slow, fails on some sites | Timeout, cache, async, manual-paste fallback | On slow/blocked sites, large pages |
| Unbounded input length to transformer | OOM / very slow on long articles | Truncate to max_length deliberately, inform user | Long pasted articles |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Retaining/logging submitted text | Privacy breach (personal messages, scam SMS with PII) | No persistence of raw input; retention = none; cache derived claims only |
| Fetching arbitrary user URLs server-side | SSRF / fetching internal/malicious endpoints; malware page exposure | Validate/normalize URL, restrict schemes, timeout, sandbox fetch, no auto-execution |
| Trusting URL string as benign feature source | Homoglyph/punycode domains evade lexical checks | Decode punycode; homoglyph/look-alike detection; treat shorteners as suspicious |
| Echoing user input into UI without escaping | XSS via crafted pasted content in Streamlit markdown | Render as text, not unescaped HTML/markdown |
| Hardcoded API keys in repo | Key leak, quota abuse | `.env` / secrets, never commit keys (note: repo already auto-loads `.env`) |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Confident verdict with no uncertainty band | Automation bias; users trust wrong "fake"/"real" calls | Calibrated confidence + explicit "uncertain — verify manually" outcome + disclaimer |
| Verdict with no explanation, or a misleading one | Defeats the project's core transparency value | Faithful highlights + which-module-fired breakdown |
| Frozen UI during verification/fetch | Feels broken; user abandons | Progressive render: instant model verdict, verification fills in |
| Treating "no external evidence" as "fake" | False accusations of legit content | Show "no evidence found" neutrally |
| Over-flagging informal/dialect Bangla | Fairness harm; user distrust | Stratified eval, disclose limits, abstain band |
| Presenting domain-not-in-list as "untrusted" | Penalizes the entire Bangla long-tail web | Unknown → neutral, surfaced as such |

## "Looks Done But Isn't" Checklist

- [ ] **Classifier accuracy:** ≥98% almost always means leakage — verify with source-stripped + title-only + source-disjoint probes before trusting it.
- [ ] **Metrics:** Accuracy shown but per-class P/R/F1, macro-F1, and confusion matrix missing — verify minority (fake/malicious) recall is non-trivial.
- [ ] **Bangla support:** Pipeline runs on English — verify it normalizes, tokenizes (low `[UNK]`), and classifies real Bangla *and* code-mixed text.
- [ ] **Preprocessing parity:** Training preprocesses correctly — verify the *exact same* function runs in the Streamlit inference path.
- [ ] **Fusion:** Produces a verdict — verify it actually beats transformer-alone on held-out (ablation), with calibrated comparable module scores.
- [ ] **Verification module:** Returns results on cherry-picked inputs — verify it abstains gracefully, handles 429/quota/timeout, and has expected Bangla coverage measured.
- [ ] **Explanation:** Highlights render — verify deletion test (removing highlights changes the verdict) so they're faithful, not decorative.
- [ ] **Malicious features:** High train accuracy — verify every feature is computable at inference and the no-network path works.
- [ ] **Latency:** Works on dev box — verify end-to-end response time on target hardware meets the "instant" claim, model cached, verification non-blocking.
- [ ] **Privacy:** No DB — verify no submitted text lands in logs/files; retention documented as none.
- [ ] **Generalization:** Good on its own test split — verify on an external/hand-collected recent Bangla+English set; report the (lower) honest number.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Source-artifact leakage discovered late | MEDIUM | Add boilerplate strip, re-split source-disjoint, retrain, re-report (numbers drop — that's correct) |
| Imbalance reported as accuracy | LOW | Recompute per-class/macro-F1 + confusion matrix from saved preds; add class weights and retrain |
| Train/test contamination | MEDIUM | Dedup, grouped/time re-split, retrain, re-evaluate |
| Wrong transformer / no Bangla normalization | MEDIUM-HIGH | Swap model + add normalizer, retrain transformer (classical stays); re-tune fusion |
| Fusion just echoes transformer | LOW-MEDIUM | Calibrate module scores, run ablation, replace hand weights with small meta-classifier or honestly report transformer-alone |
| Verification unreliable / over-quota | LOW | Add cache + timeout + abstain path; downgrade verification to optional progressive signal |
| Unfaithful attention explanation | LOW-MEDIUM | Replace with SHAP/IG/LIME for transformer, coefficients/SHAP for classical; add deletion test |
| Privacy leak via logs | LOW | Remove content logging; purge stored inputs; document retention=none |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Source-artifact leakage | Data foundation / preprocessing | Source-stripped + title-only + source-disjoint probes; inspect top features |
| Class imbalance as accuracy | Classical baseline (metric discipline) | Per-class + macro-F1 + confusion matrix mandatory in every report |
| Train/test contamination | Data foundation | Dedup report; grouped/time split; external held-out eval |
| Merged inconsistent labels | Data foundation + architecture decision | Length-stratified eval; documented label schema; provenance retained |
| Bangla normalization missing | Preprocessing (shared fn) | `[UNK]` rate check; train==inference on identical text |
| Wrong transformer for Bangla | Transformer selection | Per-language + code-mixed F1 head-to-head |
| Topic/temporal bias | Evaluation (+ flag in Data) | Time-based split; topic-distribution check |
| Phishing keyword/URL-feature trap | Malicious-detection module | Feature-availability contract; adversarial paraphrase + legit-message FP set |
| Stale blocklist/credibility list | Source-credibility module | Unknown→neutral verified; coverage + limitation documented |
| Fusion double-count / echoes model | Fusion (ablation+calibration gate) | Full-system vs transformer-alone on held-out; module ablation |
| Verification limits/circularity/latency | Verification module; UX in Integration | Abstain path, 429/timeout handling, own-domain exclusion, async render |
| Misleading explanation | Explainability | Deletion test for faithfulness |
| Real-time latency | Integration / UI (latency budget) | Measured end-to-end time on target hardware; model cached |
| Ethical: bias / overconfidence / privacy | Cross-cutting (Eval, Fusion, UI) | Stratified eval; calibrated confidence + abstain band; no-retention check |

## Sources

- ISOT / Kaggle Fake-and-Real-News source-artifact leakage (Reuters dateline → label): [ISOT research lab](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/), [Transforming Fake News (arXiv 2109.09796)](https://arxiv.org/pdf/2109.09796), [Kaggle fake-and-real-news](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- BanFakeNews extreme class imbalance (~97% authentic / ~3% fake, ~37x): [Breaking the Curse of Class Imbalance (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3511601), [LLM-Based Dataset Augmentation (arXiv 2605.01292)](https://arxiv.org/html/2605.01292), [BanFakeNews-2.0 (Kaggle)](https://www.kaggle.com/datasets/hrithikmajumdar/bangla-fake-news), [IBFND dataset (IEEE)](https://ieeexplore.ieee.org/document/10212799)
- BanglaBERT mandatory normalization before tokenization: [csebuetnlp/banglabert (GitHub)](https://github.com/csebuetnlp/banglabert), [banglanlptoolkit (PyPI)](https://pypi.org/project/banglanlptoolkit/)
- Google Fact Check Tools API (key required, standard Google API quota, sparse coverage): [Fact Check Tools API](https://developers.google.com/fact-check/tools/api), [claims.search reference](https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search)
- "Attention is not Explanation" (faithfulness of attention-based explanations) — established NLP literature (Jain & Wallace 2019; Wiegreffe & Pinter 2019)
- Established ML practice: imbalanced evaluation (macro-F1/per-class), grouped/temporal splits, model calibration (temperature/isotonic scaling), stacked/meta-classifier fusion

---
*Pitfalls research for: multilingual Bangla+English fake-news + malicious-content detection*
*Researched: 2026-06-17*
