# Corpus Report -- Phase 01 Data Foundation

Deterministic, re-runnable build of the de-leaked, source-disjoint 3-class corpus (real / fake / malicious). **Counts only -- no raw article text** is recorded here (SYS-02). The processed Parquet is gitignored (A5); rebuild with the command below.

## Build command (deterministic, seed=42)

```bash
python -m src.data.build_corpus
```

Writes `data/processed/{train,val,test}.parquet` with the full D-13 provenance schema (`text, label, source_dataset, original_label, language, split, dedup_cluster_id`).

## Policy

- Class distribution is **reported, NOT physically rebalanced** (D-03/D-04). Balancing (class weights / in-fold resampling) is a train-time concern applied in Phase 2/3.
- **Everything surviving dedup + leakage removal is kept** -- no hard row/size cap (D-05).
- Splits are **source-disjoint** (D-10) 70/15/15; near-dupes removed **before** splitting (D-11) so no near-duplicate can straddle train/test.

## Overall class distribution

| label | count |
|---|---|
| fake | 33308 |
| malicious | 33449 |
| real | 70412 |
| **total** | **137169** |

## Class distribution per split

| split \ label | fake | malicious | real | total |
|---|---|---|---|---|
| test | 4908 | 4892 | 5208 | 15008 |
| train | 23246 | 23584 | 56971 | 103801 |
| val | 5154 | 4973 | 8233 | 18360 |

## Split ratios

| split | rows | ratio |
|---|---|---|
| train | 103801 | 0.7567 |
| val | 18360 | 0.1338 |
| test | 15008 | 0.1094 |
| **total** | **137169** | **1.0000** |

## Per-language coverage

Bengali-ratio tag (D-02). The `malicious` class is assembled from SMS/phishing and is **English-only** (D-01) -- a documented limitation, surfaced in the language x label table.

| language | count |
|---|---|
| bn | 54720 |
| code-mixed | 107 |
| en | 82340 |
| unknown | 2 |
| **total** | **137169** |

### Language x class

| language \ label | fake | malicious | real | total |
|---|---|---|---|---|
| bn | 9736 | 0 | 44984 | 54720 |
| code-mixed | 83 | 0 | 24 | 107 |
| en | 23489 | 33447 | 25404 | 82340 |
| unknown | 0 | 2 | 0 | 2 |

## Per-source row counts

| source_dataset | count |
|---|---|
| banfakenews | 46195 |
| banfakenews2 | 8646 |
| isot | 38741 |
| liar | 10138 |
| phishing | 32875 |
| smsspam | 574 |
| **total** | **137169** |

### Source x class

| source_dataset \ label | fake | malicious | real | total |
|---|---|---|---|---|
| banfakenews | 1178 | 0 | 45017 | 46195 |
| banfakenews2 | 8645 | 0 | 1 | 8646 |
| isot | 17849 | 0 | 20892 | 38741 |
| liar | 5636 | 0 | 4502 | 10138 |
| phishing | 0 | 32875 | 0 | 32875 |
| smsspam | 0 | 574 | 0 | 574 |

## Deduplication

- Near-dup + exact removal rate: **0.3700** (37.00% of pre-dedup rows removed).
- Mixed-label clusters kept (Pitfall 4, one survivor per cluster x label): **2918**.
- Operating point (A4): SHA1(preprocess) exact pre-pass + MinHashLSH (num_perm=128, Jaccard=0.85, char-5-gram UTF-8).

## Dataset licenses (provenance only -- raw + processed are gitignored, not redistributed)

| source | license |
|---|---|
| BanFakeNews v1 (cryptexcode/banfakenews) | CC BY-NC-SA 4.0 (LREC-2020) |
| BanFakeNews-2.0 (hrithikmajumdar/bangla-fake-news) | Apache 2.0 |
| ISOT Fake News | academic/research use (cite ISOT, Univ. of Victoria) |
| LIAR | public research (PolitiFact / UCSB) |
| SMS Spam Collection (UCI id 228) | public, free for research |
| Phishing (naserabdullahalam/phishing-email-dataset) | CC BY-SA 4.0 |

Redistribution risk is nil: `data/raw/` and `data/processed/` are gitignored; only these counts are committed.
