# Architecture Research

**Domain:** Multi-signal fake-news + malicious-content detection pipeline (Bangla/English, local Streamlit)
**Researched:** 2026-06-17
**Confidence:** HIGH (established ML-pipeline patterns; key uncertain points — Streamlit model caching, free verification APIs — web-verified)

## Standard Architecture

This is a **multi-signal scoring pipeline**: one input fans out to N independent signal producers, each emitting a uniform `(label, score, evidence)` record, which a fusion layer collapses into one verdict + confidence + explanation. The decisive structural choice is a **uniform module contract** so fusion and explainability consume every signal the same way and modules stay swappable (a stated project goal).

A second decisive split is **offline vs online**. Dataset prep, training, evaluation, and model selection happen offline in notebooks/scripts and produce **serialized artifacts** (fitted vectorizer, classical model, fine-tuned transformer). The online path loads those artifacts once and runs inference only — this is what keeps the UI "instant" despite a heavy transformer.

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         OFFLINE (notebooks / scripts)                  │
│  data/raw ──► preprocess ──► data/processed ──► train+eval ──► select  │
│                                                       │                │
│                                          serialize artifacts ──► models/│
│   (vectorizer.joblib, classical.joblib, transformer/, label_map.json) │
└──────────────────────────────────────────────────────────────────────┘
                                   │  (artifacts on disk)
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         ONLINE  (Streamlit app, load-once)             │
│                                                                        │
│  Input (text | URL)                                                    │
│       │                                                                │
│       ▼                                                                │
│  ┌──────────────┐   URL? ──► fetch + extract article text             │
│  │  Ingest      │──────────────────────────────────────┐              │
│  └──────┬───────┘                                       │              │
│         ▼                                               │ (raw url     │
│  ┌──────────────┐                                       │  + domain    │
│  │ Preprocess   │  clean / normalize / lang-detect      │  kept for    │
│  └──────┬───────┘  ► cleaned_text                       │  credibility)│
│         │                                               │              │
│         ▼  cleaned_text broadcast to all signal modules ▼              │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌────────────────────────┐  │
│  │Classifier│ │ Credibility│ │  Style   │ │      Verify            │  │
│  │(classical│ │  (domain   │ │(clickbait│ │ (FactCheck API,        │  │
│  │+transfmr)│ │  reputation)│ │ caps,etc)│ │  Wikipedia consistency)│  │
│  └────┬─────┘ └─────┬──────┘ └────┬─────┘ └───────────┬────────────┘  │
│       │  ModuleResult(label, score, evidence) — uniform across all     │
│       └──────────────┬───────────────────────────────┘                │
│                      ▼                                                  │
│              ┌───────────────┐                                         │
│              │  Fusion layer │  weighted combine ► verdict + confidence│
│              └───────┬───────┘                                         │
│                      ▼                                                  │
│              ┌───────────────┐  aggregate evidence from all modules    │
│              │ Explainability│  + token highlights ► reasons[]         │
│              └───────┬───────┘                                         │
│                      ▼                                                  │
│              ┌───────────────┐                                         │
│              │  Streamlit UI │  verdict | confidence | highlights | why│
│              └───────────────┘                                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Ingest** | Accept text or URL; if URL, fetch page + extract main article text; preserve original URL/domain for credibility | `requests` + `trafilatura` (or `newspaper3k`); returns `{text, url, domain}` |
| **Preprocess** | Clean, normalize, detect language (bn/en/mixed); shared by all text modules. Keep raw text too (transformer + highlighting need original) | `src/data/preprocess.py`; regex normalize, Bangla unicode normalize, `fasttext`/heuristic lang-detect |
| **Classifier (signal)** | Predict real/fake/malicious from text. Hybrid: classical (TF-IDF + LR/NB/RF) and transformer (XLM-R / multilingual BERT). One is "primary," other corroborates | scikit-learn pipeline + HF `AutoModelForSequenceClassification`; loads artifacts |
| **Credibility (signal)** | Score source trustworthiness from domain reputation + known-false history. Text-only input scores neutral | Lookup table `domain_credibility.json` + heuristics; no model needed |
| **Style (signal)** | Detect clickbait/sensational markers: ALL-CAPS ratio, excessive `!`/`?`, emotional/clickbait lexicon, repetition | Rule + lexicon scorer `style_lexicon.json`; deterministic, fast |
| **Verify (signal)** | Fetch corroborating coverage from free APIs; check claim consistency. Network-bound, may time out → must degrade gracefully | Google Fact Check Claim Search API + Wikipedia API; cached, timeout-guarded |
| **Fusion** | Combine all `ModuleResult`s into one verdict + calibrated confidence via configurable weights | Weighted vote/score in `src/fusion/fuse.py`; weights in `config.yaml` |
| **Explainability** | Aggregate per-module evidence + token-level highlights into human reasons | Collect `evidence` fields + classical feature weights / attention-or-LIME highlights |
| **UI** | Input form, trigger pipeline, render verdict/confidence/highlights/reasons. Loads models once | Streamlit; `st.cache_resource` for model loading |

## Recommended Project Structure

```
fake-news-detection/
├── config.yaml                  # fusion weights, model names, API toggles, thresholds
├── requirements.txt
├── .env                         # free API keys (gitignored)
│
├── data/
│   ├── raw/                     # downloaded datasets (BanFakeNews, FakeNewsNet, phishing/spam)
│   └── processed/               # cleaned, balanced, train/val/test splits (parquet/csv)
│
├── models/                      # SERIALIZED ARTIFACTS — the offline→online handoff
│   ├── vectorizer.joblib        # fitted TF-IDF
│   ├── classical.joblib         # best classical model (selected offline)
│   ├── transformer/             # fine-tuned XLM-R (HF save_pretrained dir + tokenizer)
│   ├── label_map.json           # {0:"real",1:"fake",2:"malicious"}
│   └── classical_best.txt       # which classical model won + metrics
│
├── notebooks/                   # OFFLINE work — exploratory + heavy training
│   ├── 01_eda_and_preprocess.ipynb
│   ├── 02_train_classical.ipynb
│   └── 03_train_transformer.ipynb   # run on Colab/GPU, export to models/
│
├── src/
│   ├── config.py                # load config.yaml + .env into typed object
│   ├── data/
│   │   ├── download.py          # fetch datasets
│   │   ├── build_corpus.py      # merge sources, label, balance, split
│   │   └── preprocess.py        # SHARED clean/normalize/lang-detect (offline + online)
│   ├── models/
│   │   ├── classical.py         # train + load/predict TF-IDF classical
│   │   └── transformer.py       # train + load/predict transformer
│   ├── modules/                 # the 4 SIGNAL producers — uniform contract
│   │   ├── base.py              # ModuleResult dataclass + SignalModule interface
│   │   ├── classifier.py        # wraps models/ into a signal
│   │   ├── credibility.py
│   │   ├── domain_credibility.json
│   │   ├── style.py
│   │   ├── style_lexicon.json
│   │   └── verify.py
│   ├── fusion/
│   │   └── fuse.py              # combine ModuleResults → verdict+confidence
│   ├── explain.py               # aggregate evidence + highlights → reasons[]
│   └── pipeline.py              # ONLINE orchestrator: ingest→preprocess→signals→fuse→explain
│
├── app/
│   └── streamlit_app.py         # UI only; calls src.pipeline; st.cache_resource loaders
│
└── reports/                     # OFFLINE eval outputs
    ├── model_comparison.{csv,md}
    ├── classical_metrics.json
    └── confusion_*.png
```

### Structure Rationale

- **`models/` is the contract between offline and online.** Anything online needs is a file here. Training never runs inside the app. This is the single most important boundary for "instant UI."
- **`src/modules/` holds the 4 signal producers behind one interface (`base.py`).** New signals drop in without touching fusion — satisfies the "swappable modules" requirement.
- **`preprocess.py` is shared** so offline training and online inference apply identical cleaning (avoids train/serve skew, a classic pitfall).
- **`pipeline.py` is pure orchestration, UI-agnostic.** Streamlit calls it; so could a CLI or test. Keeps UI thin and testable.
- **`notebooks/` for heavy training only** (transformer fine-tune belongs on Colab/GPU); production logic lives in `src/` so it's importable and version-controlled cleanly.

## Architectural Patterns

### Pattern 1: Uniform Signal Module Contract

**What:** Every signal module implements one interface and returns one shape, so fusion and explainability never special-case a module.
**When to use:** Always here — it's the backbone of fusion + swappability.
**Trade-offs:** Slight upfront rigidity (must fit credibility, style, classifier, verify into one shape) in exchange for trivial fusion and a stable explainability consumer. Worth it.

```python
# src/modules/base.py
from dataclasses import dataclass, field

@dataclass
class ModuleResult:
    module: str                       # "classifier" | "credibility" | "style" | "verify"
    label: str                        # "real" | "fake" | "malicious" | "unknown"
    score: float                      # 0..1 confidence in `label`
    contribution: dict = field(default_factory=dict)  # per-class scores, e.g. {"real":.1,"fake":.8,"malicious":.1}
    evidence: list = field(default_factory=list)       # human-readable reasons / highlighted spans
    available: bool = True            # False if module skipped (e.g. verify timed out, no URL)

class SignalModule:
    name: str
    def analyze(self, text: str, ctx: dict) -> ModuleResult: ...
    # ctx carries url, domain, lang so modules that need them get them
```

### Pattern 2: Load-Once Inference (instant UI despite heavy transformer)

**What:** Load every model artifact a single time per process, cache the live object, reuse across all reruns/sessions. Streamlit reruns the script top-to-bottom on every interaction — without caching it would reload the transformer each click.
**When to use:** Any Streamlit app wrapping a heavy model. Mandatory here.
**Trade-offs:** Holds the model in RAM for the process lifetime (fine for local single-machine). First request pays load cost; all later requests are warm and fast.

```python
# app/streamlit_app.py
import streamlit as st

@st.cache_resource          # one object shared across reruns + sessions
def load_pipeline():
    from src.pipeline import Pipeline
    return Pipeline.from_artifacts("models/")   # loads vectorizer, classical, transformer once

pipe = load_pipeline()
result = pipe.run(user_input)   # warm; only forward-pass cost
```

Keep latency down further: run classical + style + credibility (sub-millisecond) always; gate the transformer to one forward pass (no grad, batch=1, CPU `torch.no_grad()`); make `verify` async/timeout-bounded so a slow API never blocks the verdict.

### Pattern 3: Graceful-Degradation Fusion

**What:** Fusion consumes only modules where `available=True` and renormalizes weights over the present signals. A timed-out verify or a text-only input (no domain → credibility neutral) never breaks the verdict.
**When to use:** Any pipeline with network-dependent or input-conditional signals — i.e. this one.
**Trade-offs:** Confidence must be reported honestly lower when signals are missing; slightly more fusion logic, but essential for a real-time UI on free/flaky APIs.

```python
# src/fusion/fuse.py
def fuse(results: list[ModuleResult], weights: dict) -> dict:
    active = [r for r in results if r.available]
    w = {r.module: weights[r.module] for r in active}
    norm = sum(w.values()) or 1.0
    classes = ["real", "fake", "malicious"]
    agg = {c: sum(r.contribution.get(c, 0) * w[r.module] for r in active) / norm
           for c in classes}
    verdict = max(agg, key=agg.get)
    return {"verdict": verdict, "confidence": agg[verdict],
            "class_scores": agg, "used_modules": [r.module for r in active]}
```

## Data Flow

### Request Flow (online, explicit direction)

```
User submits text OR url
        │
        ▼
Ingest:  if url → fetch + extract → {text, url, domain}
         if text → {text, url:None, domain:None}
        │
        ▼
Preprocess: clean + normalize + lang-detect → cleaned_text  (raw text retained for highlights)
        │
        ├──────────────┬──────────────┬──────────────┐   (cleaned_text + ctx broadcast)
        ▼              ▼              ▼              ▼
   Classifier     Credibility       Style         Verify
   (cls+xfmr)     (domain)        (lexicon)    (FactCheck/Wiki)
        │              │              │              │
        └──ModuleResult┴──ModuleResult┴──ModuleResult┘
        │   (uniform (label, score, contribution, evidence, available))
        ▼
Fusion: weighted combine over available modules → {verdict, confidence, class_scores}
        │
        ▼
Explainability: gather evidence[] from every module + classifier token highlights → reasons[]
        │
        ▼
UI: render verdict + confidence bar + highlighted text + ranked reasons
```

Data direction is strictly **left-to-right / top-down**: no module reads another module's output; only fusion aggregates. This keeps modules independent and testable in isolation.

### Offline Data Flow (training)

```
data/raw  ──download──►  build_corpus (merge+label+balance+split)  ──►  data/processed
                                                                            │
                          ┌─────────────────────────────────────────────────┤
                          ▼                                                  ▼
            train classical (TF-IDF+LR/NB/RF)                  fine-tune transformer (Colab/GPU)
                          │  eval on test                                    │ eval on test
                          ▼                                                  ▼
              reports/model_comparison  ◄── select best ──►   models/ (serialized artifacts)
```

### Key Data Flows

1. **Offline→Online handoff:** training serializes to `models/`; the app only ever *loads* from `models/`. No training code path runs in the UI.
2. **Shared preprocessing:** the exact same `preprocess.py` runs in `build_corpus` (offline) and `pipeline` (online) → no train/serve skew.
3. **Context passthrough:** Ingest's `url`/`domain`/`lang` ride alongside text in `ctx` so credibility and verify get what they need without re-parsing.

## Scaling Considerations

This is a **local, single-user, real-time demo** — classic web-scale scaling does not apply. "Scale" here means latency and concurrency on one machine.

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single user (target) | `st.cache_resource` load-once; CPU transformer with `no_grad`, batch=1; verify timeout ~2-3s. Sufficient. |
| Few concurrent demo users | One shared cached model object (already handled by `cache_resource`); serialize transformer calls; cache verify API responses by query hash. |
| Heavier / future | Move transformer to a quantized/distilled checkpoint or a small inference server (FastAPI) the UI calls; out of this milestone's scope. |

### Scaling Priorities

1. **First bottleneck — transformer forward pass.** Fix: load once, CPU `torch.no_grad()`, short sequences (truncate), consider distilled multilingual model if latency hurts.
2. **Second bottleneck — verify API round-trips.** Fix: hard timeout + cache responses; treat verify as best-effort (`available=False` on failure) so it never blocks the verdict.

## Anti-Patterns

### Anti-Pattern 1: Training (or re-fitting vectorizer) inside the app

**What people do:** Call `fit()` / load+train on app start, or refit TF-IDF on the input.
**Why it's wrong:** Re-trains every run, kills "instant," and produces a vectorizer inconsistent with the saved model.
**Do this instead:** Fit offline, `joblib.dump` the vectorizer + model; online only `load` + `transform` + `predict`.

### Anti-Pattern 2: Divergent preprocessing offline vs online (train/serve skew)

**What people do:** Clean text one way in the notebook, another way in the app.
**Why it's wrong:** Model sees inputs unlike its training distribution → silent accuracy loss.
**Do this instead:** Single `preprocess.py` imported by both paths.

### Anti-Pattern 3: Modules with bespoke return shapes

**What people do:** Credibility returns a float, style returns a dict, classifier returns logits — fusion special-cases each.
**Why it's wrong:** Fusion + explainability become a tangle; adding/swapping a module breaks them.
**Do this instead:** Every module returns `ModuleResult`. Fusion and explainability iterate uniformly.

### Anti-Pattern 4: Blocking the verdict on a network call

**What people do:** Await the fact-check API synchronously with no timeout.
**Why it's wrong:** Free/flaky APIs stall the UI; "real-time" promise broken.
**Do this instead:** Timeout-bound verify, mark `available=False` on failure, fuse renormalizes over present signals.

## Integration Points

### External Services (free tiers only)

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Google Fact Check Claim Search API | REST GET with API key (in `.env`); query extracted claim/title | Free; supports 70+ languages incl. Bangla. Returns existing fact-checks → strong signal. Timeout + cache. |
| Wikipedia API (MediaWiki / REST) | Keyless search + summary; check entity/claim consistency | Free, no key. Best-effort corroboration; degrade gracefully. |
| Article extraction (URL input) | `trafilatura`/`newspaper3k` over `requests` | Local lib, not an API. Handle paywalls/fetch failures → fall back to user-pasted text. |
| HF model hub | Download fine-tuned/base weights once → store in `models/transformer/` | Offline step; online loads from disk, no network at inference. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| offline ↔ online | Files in `models/` (joblib + HF dir + json) | The critical seam; nothing else crosses |
| pipeline ↔ each module | Direct call returning `ModuleResult` | Modules independent; no module-to-module calls |
| modules ↔ fusion | List of `ModuleResult` in | Fusion reads `contribution`/`available`/`weights` only |
| fusion+modules ↔ explainability | `evidence[]` + class_scores | Explainability is a read-only consumer |
| pipeline ↔ UI | One `run()` call → result dict | UI never imports models directly; only via cached pipeline |

## Build Order (dependencies between components)

Strict bottom-up; later stages need earlier artifacts/contracts to exist.

1. **`config.py` + `preprocess.py`** — shared foundation; everything imports them.
2. **`data/` (download → build_corpus)** — produces `data/processed`; nothing trains without it.
3. **Models trained offline → `models/` artifacts** — classical first (fast, validates pipeline + metrics), then transformer. Produces the offline→online handoff.
4. **`modules/base.py` (ModuleResult + interface)** — the contract. **Must exist before any module or fusion.**
5. **The 4 signal modules** — classifier (needs `models/`), then credibility, style, verify (independent of each other; build in any order, all conform to `base.py`).
6. **`fusion/fuse.py`** — needs the contract + at least the classifier module to be meaningful; can be built/tested with stub ModuleResults before all 4 modules exist.
7. **`explain.py`** — needs ModuleResults + fusion output to aggregate.
8. **`pipeline.py`** — orchestrates 1–7; needs everything above.
9. **`app/streamlit_app.py`** — thin UI over `pipeline.run()`; built last, needs the whole pipeline.

**Implications for roadmap phasing:** the contract (`base.py`) and shared preprocessing are gating prerequisites — they unblock parallel work on the 4 modules. Fusion and UI are necessarily late (they consume everything). Classical model before transformer de-risks the artifact/persistence path with a fast-to-train model before committing GPU time. Verify (network APIs) carries the most integration risk → flag for deeper research at its phase.

## Model Artifact Persistence

| Artifact | Save (offline) | Load (online) |
|----------|----------------|---------------|
| TF-IDF vectorizer | `joblib.dump(vec, "models/vectorizer.joblib")` | `joblib.load(...)` once, cached |
| Classical model | `joblib.dump(clf, "models/classical.joblib")` | `joblib.load(...)` once, cached |
| Fine-tuned transformer + tokenizer | `model.save_pretrained("models/transformer/")` + `tokenizer.save_pretrained(...)` | `AutoModelForSequenceClassification.from_pretrained("models/transformer/")` once, cached, `.eval()` |
| Label map | write `label_map.json` | load once; map ids→names consistently everywhere |

Persist the **fitted** vectorizer alongside the model (they must match), version artifacts by milestone, and never refit at inference. The transformer dir + joblib files together constitute the complete online dependency set.

## Sources

- [st.cache_resource — Streamlit Docs](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource) — load-once model caching pattern (HIGH)
- [Streamlit caching overview](https://docs.streamlit.io/develop/concepts/architecture/caching) — cache_resource vs cache_data (HIGH)
- [PyTorch model load/inference once — Streamlit forum](https://discuss.streamlit.io/t/pytorch-model-demo-load-the-model-inference-only-once/45118) — class-method caching caveat (MEDIUM)
- [Google Fact Check Tools API](https://developers.google.com/fact-check/tools/api) — free Claim Search, 70+ languages incl. Bangla (HIGH)
- [APIs for fact-checking (curated list)](https://github.com/hearvox/unreliable-news/blob/master/ref/apis-for-fact-checking.md) — free verification API options (MEDIUM)

---
*Architecture research for: multi-signal fake-news + malicious-content detection pipeline*
*Researched: 2026-06-17*
