# Transformer Selection Report -- Phase 03

Selection of the better fine-tuned transformer backbone for the two-stage cascade
(gate → real/fake). **Counts / metrics only -- no raw input text** is recorded here
(SYS-02). The exported `models/transformer/<backbone>/` artifacts are gitignored; rebuild
via the Colab notebook + `python -m src.models.select_transformer`.

> **Status:** the wiring (loader, gate-threshold sweep, Bangla-priority selection, SC-3
> score-smell, report writer) is implemented and unit-proven on the tiny fixture. The
> populated per-language numbers below are produced by the **manual Colab fine-tune run**
> (`notebooks/03_train_transformer_colab.ipynb`) followed by
> `python -m src.models.select_transformer` on the Mac Mini M4. This committed report
> documents the locked structure + selection rules + caveats; the metric cells are filled
> in by that run (they read `(pending Colab run)` until then).

## Build command (deterministic, seed=42)

```bash
# 1. Fine-tune both backbones x {gate, realfake} on a free Colab T4 (notebook):
#    notebooks/03_train_transformer_colab.ipynb  ->  models/transformer/<backbone>/
# 2. Download models/transformer/ locally, then run the selection on the M4:
python -m src.models.select_transformer
```

Writes `reports/transformer_selection_report.md` (this file). **Macro-F1 is the headline
metric** -- accuracy is never the headline; per-language macro-F1 with Bangla priority is
the selection criterion.

## Chosen model

**(pending Colab run)** -- selected by **per-language macro-F1 with Bangla priority**
(D-01): the stronger-Bangla backbone wins; near-ties (within `1e-4` on the Bangla number)
break toward the **lighter** backbone (`banglishbert` ~110M over `xlmr` ~270M, D-03). A
backbone whose `fake`/`malicious` recall has collapsed is disqualified (minority guard).

Candidates ranked: `banglishbert` (csebuetnlp/banglishbert, ~110M, **code-mixed primary**),
`xlmr` (xlm-roberta-base, ~270M, robust fallback). BanglaBERT is the optional-stretch third
backbone (D-04).

## Per-language macro-F1 (test)

### banglishbert

| language | macro-F1 |
|---|---|
| bn | _(pending Colab run)_ |
| en | _(pending Colab run)_ |
| code-mixed | _(pending Colab run)_ |

### xlmr

| language | macro-F1 |
|---|---|
| bn | _(pending Colab run)_ |
| en | _(pending Colab run)_ |
| code-mixed | _(pending Colab run)_ |

## Gate threshold (val sweep, D-09)

| metric | value |
|---|---|
| chosen threshold | _(pending Colab run)_ |
| val cascade macro-F1 | _(pending Colab run)_ |
| malicious precision | _(pending Colab run)_ |
| malicious recall | _(pending Colab run)_ |

Chosen on the **validation** split by `choose_gate_threshold`, which sweeps the gate
threshold grid (0.3..0.9) and maximizes the full 3-class cascade macro-F1 -- never on test
(Pitfall 3 / T-03-12). The 0.5 sentinel persisted at export time is overwritten by this
val-argmax value.

## Leakage re-check (SC-3, score-smell only — D-15)

`transformer_leakage_smell(test_macro_f1)` flags **investigate** when the test macro-F1 is
**>= 0.98** (the reused `INVESTIGATE_THRESHOLD`). Transformers expose no per-token feature
weights, so the token-tell half of the classical re-check has **no analog** here -- this is
the SCORE threshold only (D-15). Result: _(pending Colab run)_.

## Code-mixed probe (qualitative, small-N — D-02)

A hand-curated code-mixed Bangla+English probe lives at `data/probe/code_mixed_probe.csv`
(59 rows: 18 real / 18 fake / 23 malicious; authentic-style social posts, viral
misinformation, and scam SMS). Its macro-F1 is **qualitative and small-N** -- indicative
only, it does **not** gate selection. Code-mixed macro-F1: _(pending Colab run)_.

## Caveats

- **Bangla gate abstains (D-07):** the malicious class is English-only, so the gate has zero
  Bangla malicious rows; on Bangla input the gate effectively abstains and routing falls to
  the real/fake head. Documented limitation.
- **Malicious class is English-only (Phase-1 D-01):** visible in the per-language tables
  above (no Bangla / code-mixed malicious coverage in the training corpus).
- **Code-mixed is small-N (D-02):** the code-mixed probe is qualitative; its macro-F1 is
  indicative only and does not gate selection.
- **No timing metric (D-11):** inference is interactive on the target hardware (Mac Mini M4
  for development, Colab for the submission demo); wall-clock speed is deliberately not a
  reported metric (counts/metrics only, SYS-02).
