# Milestones

Shipped versions of the AI-Driven Fake-News & Malicious-Content Detector.

## v1.0 — Data + Classical Foundation

**Shipped:** 2026-06-19
**Phases:** 1–2 (10 plans, ~20 tasks)
**Git range:** 57bf308 → cf7a756 (47 commits)
**Timeline:** 2026-06-17 → 2026-06-19 (3 days)
**Code:** src 2,579 LOC across 17 Python modules; tests 1,658 LOC (100 fast tests + slow real-corpus gates).

**Delivered:** The offline foundation — a de-leaked, documented 3-class corpus with a shared train=inference preprocessing path, and trained + serialized classical TF-IDF baselines with locked metric discipline.

**Key accomplishments:**

1. Unified **137,169-row** 3-class corpus (real/fake/malicious) from 5 sources (BanFakeNews/2.0, ISOT, LIAR, SMS Spam, phishing) — explicit per-source label mapping + provenance, source-disjoint 70/15/15 splits, 37% dedup (exact-SHA1 + MinHashLSH).
2. **Leakage gate genuinely closed** — caught a real bare-Reuters dateline leak on the first run, fixed it, rebuilt deterministically; source_stripped macro-F1 0.9087 (< 0.95 ceiling), no surviving outlet/year tell.
3. **Shared `preprocess()`** (`csebuetnlp/normalizer` + whitespace collapse) — single importable entry point for Bangla + English + code-mixed at both train and inference.
4. **Classical baselines trained + serialized** (TF-IDF + LR/NB/RF) — best = **LogisticRegression, test macro-F1 0.9140**; all minority-class recalls > 0.
5. **Metric discipline locked** — macro-F1 headline (never accuracy), per-class + confusion + per-language + minority-recall guard; SC-3 leakage re-check gate (fired correctly on ComplementNB year-tells).

**Scope:** Ships **7 of 29** v1 product requirements (DATA-01..05, CLS-01, CLS-03). The remaining **22** (transformer, malicious detection, signal modules, verification, fusion, explainability, Streamlit UI — Phases 3–7) are **carried forward** to the next milestone, not shipped.

**Audit:** `milestones/v1.0-MILESTONE-AUDIT.md` — status `passed` (7/7 in-scope reqs, integration 5/5, flows 1/1, Nyquist compliant, security verified). Pre-close open-artifact audit: 1 heuristic flag (Phase 02 UAT, status `passed`/0 pending) — benign false-positive, no real gap.

**Known deferred items at close:** 1 (BanFakeNews-2.0 license web-confirmation — low risk, raw data gitignored). Remaining debt is info-level only.

**Archived:** `milestones/v1.0-ROADMAP.md`, `milestones/v1.0-REQUIREMENTS.md`.
