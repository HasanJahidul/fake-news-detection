# Retrospective

Living retrospective for the AI-Driven Fake-News & Malicious-Content Detector. One section per milestone, plus cross-milestone trends at the end.

## Milestone: v1.0 — Data + Classical Foundation

**Shipped:** 2026-06-19
**Phases:** 2 | **Plans:** 10

### What Was Built

The offline foundation: a unified, de-leaked 3-class corpus (137,169 rows from 5 sources) with explicit per-source label mapping + provenance and source-disjoint 70/15/15 splits; a shared `preprocess()` (csebuetnlp/normalizer) used identically at train and inference; and trained + serialized classical TF-IDF baselines (LR/NB/RF) with locked metric discipline — best = LogisticRegression, test macro-F1 0.9140.

### What Worked

- **Leakage-first discipline paid off.** The leakage probe actually *failed* on the first run (bare-Reuters dateline self-references surviving the strip) and forced a real fix + deterministic rebuild — exactly the kind of silent-accuracy trap the gate exists to catch. Shipping a model that scored high on a leak would have been worse than the delay.
- **Classical-before-transformer** validated the data + artifact-persistence path on CPU before any GPU spend — the serialized `models/` artifact contract is now proven for the transformer to reuse.
- **Single source of truth for leak-tells** — Phase 2's SC-3 re-check reused Phase 1's `leak_tells_in_features` regex rather than duplicating it, so the gate can't silently diverge.
- **TDD throughout** — 100 fast tests + slow real-corpus gates that *run* (not skip) gave high confidence at the milestone gate.

### What Was Inefficient

- **ROADMAP.md got truncated to 12 lines** at some point during Phase 2 and was committed in that state — only caught by the milestone audit, requiring reconstruction from git history before close. A planning artifact silently lost most of its content.
- **WR-01 (ComplementNB descending feature attribution)** shipped into the audit as tech debt and had to be fixed at close rather than during Phase 2.
- **Milestone scope vs. requirements framing collided** — REQUIREMENTS.md framed "v1" as the full 29-req product while the milestone shipped only 7. Resolving "carry 22 forward vs. mark shipped" at close added a decision gate that cleaner scoping up front would have avoided.

### Patterns Established

- macro-F1 headline (never accuracy) + per-class + confusion + per-language + minority-recall guard as the standard report contract.
- ≥0.98 score ⇒ suspected-leakage investigation gate.
- Natural class distribution stored/reported; balance at train time via `class_weight`/ComplementNB, not resampling (D-03/D-04).
- Offline→online boundary via load-only serialized artifacts.

### Key Lessons

- Treat planning artifacts (ROADMAP/REQUIREMENTS) as code: a truncation should be caught by review/CI, not a milestone audit weeks later.
- Define milestone scope as a strict subset of requirements up front, so "complete milestone" never has to reconcile a product-wide requirements doc against a 2-phase slice.
- A verification gate that can fail (and did) is worth more than one that always passes.

### Cost Observations

- Sessions: multiple over 2026-06-17 → 2026-06-19 (3 days).
- Notable: heavy TDD + real-corpus slow gates (leakage probe ~117s) front-loaded cost but caught a real leak.

## Cross-Milestone Trends

_(Populated from v1.1 onward.)_

| Milestone | Phases | Plans | Best headline metric | Notable |
|-----------|--------|-------|----------------------|---------|
| v1.0 | 2 | 10 | classical macro-F1 0.9140 | Leakage gate caught a real leak; foundation only (7/29 reqs) |
