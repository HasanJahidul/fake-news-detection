---
phase: 01
slug: data-foundation
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-17
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Concrete assertions live in `01-RESEARCH.md` → "## Validation Architecture".

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (greenfield — not yet installed) |
| **Config file** | none — Wave 0 (01-01) installs pytest + adds `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest -q -m "not slow"` |
| **Full suite command** | `pytest -q` (includes the slow leakage probe gate) |
| **Estimated runtime** | ~{N} seconds (set after Wave 0) |

---

## Sampling Rate

- **After every task commit:** Run `pytest -q -m "not slow"`
- **After every plan wave:** Run `pytest -q -m "not slow"`
- **Before `/gsd-verify-work`:** Full suite incl. slow leakage probe must be green
- **Max feedback latency:** {N} seconds (set after Wave 0)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01 Task 1 (pytest/gitignore/skeleton) | 01 | 0 | DATA-05 | T-01-ID / T-01-SC | secrets + raw/processed data gitignored (`git check-ignore`); only approved packages installed | infra | `pytest --version && python -c "...slow marker..." && git check-ignore data/raw .env kaggle.json data/processed` | ✅ W0 | ⬜ pending |
| 01-01 Task 2 (preprocess() DATA-05) | 01 | 0 | DATA-05 | — | lossless normalize preserves URLs/CAPS/punct; idempotent | unit | `pytest -q tests/test_preprocess.py` | ✅ W0 | ⬜ pending |
| 01-01 Task 3 (six downstream stubs) | 01 | 0 | DATA-05 | — | N/A (scaffold collects green) | unit | `pytest -q && pytest --collect-only -q tests/` | ✅ W0 | ⬜ pending |
| 01-02 Checkpoint 1 (Kaggle token) | 02 | 1 | DATA-01 | T-02-ID | human-only credential gate (token never committed) | manual | N/A — `checkpoint:human-action` (blocking-human) | ✅ W0 | ⬜ pending |
| 01-02 Task 1 (acquire.py) | 02 | 1 | DATA-01 | T-02-ID / T-02-TM | no hardcoded creds; zip-slip containment guard before extract; content as data only | source/AST | `python -c "ast ... + zip-slip containment guard (commonpath/is_relative_to) source assert"` | ✅ W0 | ⬜ pending |
| 01-02 Task 2 (test_acquire.py + download) | 02 | 1 | DATA-01 | T-02-ID2 | raw gitignored (always-on assert); skips without creds | integration | `pytest -q tests/test_acquire.py && git check-ignore data/raw` | ✅ W0 | ⬜ pending |
| 01-02 Checkpoint 2 (sources downloaded) | 02 | 1 | DATA-01 | T-02-ID2 | nothing raw staged for commit | manual | N/A — `checkpoint:human-verify` (blocking) | ✅ W0 | ⬜ pending |
| 01-03 Task 1 (label_map.py DATA-02) | 03 | 2 | DATA-02 | T-03-TM | only labels ∈ {real,fake,malicious} enter corpus; LIAR half-true + SMS ham dropped | unit | `pytest -q tests/test_schema.py -k "label or LabelMap or collapse"` | ✅ W0 | ⬜ pending |
| 01-03 Task 2 (schema.py Parquet) | 03 | 2 | DATA-02 | T-03-TM | full D-13 schema, no nulls; Bangla Parquet round-trip byte-identical | unit | `pytest -q tests/test_schema.py::test_provenance_complete tests/test_schema.py::test_bangla_roundtrip` | ✅ W0 | ⬜ pending |
| 01-04 Task 1 (leakage_strip.py DATA-03) | 04 | 2 | DATA-03 | T-04-TM | Reuters dateline removed; conservative per-source strip; not in preprocess() (D-09) | unit | `pytest -q tests/test_leakage_strip.py -k "reuters or dateline or strip"` | ✅ W0 | ⬜ pending |
| 01-04 Task 2 (language.py D-02) | 04 | 2 | DATA-03 | T-04-DoS | deterministic bn/en/code-mixed/unknown by Bengali ratio; anchored regex (no backtracking) | unit | `pytest -q tests/test_leakage_strip.py -k "language or Language or codemixed"` | ✅ W0 | ⬜ pending |
| 01-05 Task 1 (dedup.py DATA-04) | 05 | 3 | DATA-04 | T-05-TM / T-05-DoS | exact+fuzzy near-dup cluster before split; sub-quadratic MinHashLSH; dedup_cluster_id retained | unit | `pytest -q tests/test_dedup.py` | ✅ W0 | ⬜ pending |
| 01-05 Task 2 (split.py DATA-04) | 05 | 3 | DATA-04 | T-05-TM | source-disjoint 70/15/15; assert_disjoint fails on cross-split group; all 3 classes per split | unit | `pytest -q tests/test_split_disjoint.py` | ✅ W0 | ⬜ pending |
| 01-06 Task 1 (build_corpus.py orchestrator) | 06 | 4 | DATA-02, DATA-04 | T-06-TM | pipeline hard-orders dedup+strip BEFORE split (source-order check) | source | `python -c "...source order: dedup_dataframe before make_splits..."` | ✅ W0 | ⬜ pending |
| 01-06 Task 2 (build run + report + tests) | 06 | 4 | DATA-02, DATA-04 | T-06-ID | report carries counts only (no raw text); on-disk Parquet has full D-13 schema, no nulls | integration | `pytest -q tests/test_build_corpus.py && test -f reports/corpus_report.md && grep -qiE "class distribution\|per-language" reports/corpus_report.md` | ✅ W0 | ⬜ pending |
| 01-07 Task 1 (leakage_probe.py DATA-03) | 07 | 5 | DATA-03 | T-07-TM | PASS/FAIL rule encoded (source_stripped < full AND < 0.95); negative control proves sensitivity | unit | `pytest -q tests/test_leakage_probe.py -m "not slow"` | ✅ W0 | ⬜ pending |
| 01-07 Task 2 (run probe + report verdict) — PHASE GATE | 07 | 5 | DATA-03 | T-07-TM | slow probe RUNS (errors if Parquet absent, never skips); report records `Leakage probe: PASS`, no FAIL | slow integration | `pytest -q tests/test_leakage_probe.py -m slow && grep -qiE "leakage probe[: ].*pass" reports/corpus_report.md && ! grep -qiE "leakage probe[: ].*fail" reports/corpus_report.md` | ✅ W0 | ⬜ pending |

*All target test files are created as stubs in Wave 0 (plan 01-01), so every row's File Exists = ✅ W0. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Infra/source/AST rows have a non-pytest automated gate (version+gitignore, AST/source assertions, source-order check); every task has an automated verify or a Wave 0 dependency. The two manual rows are the Kaggle-token and download-confirmation checkpoints (DATA-01 needs live creds — see Manual-Only Verifications); they are not three-consecutive (each is bracketed by automated tasks).*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` — install + configure pytest + register the `slow` marker (no framework detected)
- [ ] `tests/conftest.py` — shared fixtures (tiny in-repo Bangla + English + code-mixed + URL + CAPS sample corpora; `sample_corpus` DataFrame)
- [ ] `tests/test_preprocess.py` — implemented for DATA-05 (idempotence; preserves URLs/CAPS/punctuation)
- [ ] `tests/test_schema.py` — stubs for DATA-02 (label map + provenance completeness; Parquet Bangla round-trip)
- [ ] `tests/test_leakage_strip.py` — stubs for DATA-03 (boilerplate stripped; language tagging)
- [ ] `tests/test_dedup.py` — stub for DATA-04 (near-dup removed)
- [ ] `tests/test_split_disjoint.py` — stubs for DATA-04 (source-disjoint; all classes present; ratios)
- [ ] `tests/test_build_corpus.py` — stubs for DATA-02/04 (Parquet written; provenance schema on disk)
- [ ] `tests/test_leakage_probe.py` — stub for DATA-03 (source-stripped near-chance, `@pytest.mark.slow`)
- [ ] `tests/test_acquire.py` — stub for DATA-01 (sources present; skips if creds absent)

*See `01-RESEARCH.md` for the concrete assertions behind each.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Source dataset download | DATA-01 | Requires Kaggle token + network; not run in CI | Run download script with `kaggle.json` present; confirm raw files land under gitignored `data/raw/` |

*Acquisition is the only step needing live credentials/network; everything downstream is deterministically testable on the cached raw data. The slow leakage probe (01-07 Task 2) is NOT manual — it runs automatically against the built Parquet and gates the phase.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (all test files created as stubs in plan 01-01)
- [x] No watch-mode flags
- [ ] Feedback latency < {N}s *(set after Wave 0 measures `pytest -q` runtime)*
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
</content>
