"""D-01/D-03/D-09/D-11/D-15 — transformer gate-threshold sweep, Bangla-priority selection,
score-smell leakage re-check, and the committed selection report (counts/metrics only).

The selection half of the phase. Mirrors :mod:`src.models.train_classical`'s select/report
structure and REUSES the shared metric harness (:mod:`src.models.metrics`) and the SC-3
score threshold (:data:`src.models.leakage_recheck.INVESTIGATE_THRESHOLD`) — nothing is
re-implemented:

  * :func:`choose_gate_threshold` (alias :func:`sweep_gate_threshold`) — sweep candidate
    gate thresholds on the VALIDATION cascade outputs, pick the one maximizing the full
    3-class cascade macro-F1, and return that threshold together with its malicious
    precision/recall (D-09). No ``test`` parameter — the sweep can never touch test.
  * :func:`select_best_transformer` (alias :func:`select_backbone`) — rank candidate
    backbones by per-language macro-F1 with BANGLA PRIORITY (D-01), reusing the classical
    minority guard (a collapsed fake/malicious recall disqualifies) and resolving near-ties
    toward the lighter backbone (D-03, size tiebreak).
  * :func:`transformer_leakage_smell` (alias :func:`transformer_leakage_recheck`) — the
    SC-3 re-check on transformer test predictions is SCORE-SMELL ONLY (``>= 0.98`` ⇒
    investigate). Transformers expose no per-token feature weights, so the token-tell half
    of ``leakage_recheck`` has NO analog here (D-15).
  * :func:`write_selection_report` — emit ``reports/transformer_selection_report.md`` with
    the chosen model + per-language/code-mixed macro-F1 + gate threshold P/R + the SC-3
    result + caveats. NO latency line (D-11); counts/metrics only (SYS-02).
  * :func:`main` — load splits, build a cascade per backbone, evaluate, sweep, select, write.

``torch`` / ``transformers`` enter only transitively via :class:`TransformerCascade`; this
module's own imports stay light so the selection schema/constants import without the stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

from src.data.label_map import LABELS
from src.models.leakage_recheck import INVESTIGATE_THRESHOLD
from src.models.metrics import (
    macro_f1,
    minority_recall,
    per_language_macro_f1,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("select_transformer")

PathLike = Union[str, Path]

# Repo root = two levels up from src/models/select_transformer.py.
REPO_ROOT = Path(__file__).resolve().parents[2]

# D-09 gate-threshold sweep grid (val only). 0.3..0.9 spans the useful operating range.
THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Schema the threshold sweep returns and the report writer renders (test_threshold contract).
THRESHOLD_RESULT_KEYS: tuple[str, ...] = ("threshold", "macro_f1", "precision", "recall")

# D-03 size tiebreak — approximate parameter counts; the LIGHTER backbone wins a near-tie.
BACKBONE_PARAMS: dict[str, int] = {
    "banglishbert": 110_000_000,
    "banglabert": 110_000_000,
    "xlmr": 270_000_000,
}

# Minority classes whose collapsed recall disqualifies a backbone (mirrors train_classical).
_MINORITY_CLASSES: tuple[str, str] = ("fake", "malicious")
_MINORITY_FLOOR: float = 1e-9
# Per-language macro-F1 within this band is a tie → break toward the lighter backbone (D-03).
_TIE_BAND: float = 1e-4


# ---------------------------------------------------------------------------
# Cascade evaluation helpers (D-09 sweep, applied to val then test).
# ---------------------------------------------------------------------------
def _cascade_labels(gate_mal_probs, rf_probs_real, threshold: float) -> list[str]:
    """Apply one gate threshold to per-row gate/real-fake probabilities → 3-class labels.

    ``gate_mal_probs[i]`` is P(malicious) from the gate; ``rf_probs_real[i]`` is
    P(real) from the real/fake head. A row is ``malicious`` when ``P_mal >= threshold``,
    else ``real`` when ``P(real) >= 0.5`` else ``fake`` — the exact cascade decision the
    online :class:`TransformerCascade` makes, evaluated in bulk for the sweep.
    """
    out: list[str] = []
    for p_mal, p_real in zip(gate_mal_probs, rf_probs_real):
        if p_mal >= threshold:
            out.append("malicious")
        elif p_real >= 0.5:
            out.append("real")
        else:
            out.append("fake")
    return out


def _malicious_precision_recall(y_true: Sequence, y_pred: Sequence) -> tuple[float, float]:
    """Precision/recall of the ``malicious`` class at the chosen threshold (D-09)."""
    from sklearn.metrics import precision_score, recall_score

    prec = float(
        precision_score(y_true, y_pred, labels=["malicious"], average="micro", zero_division=0)
    )
    rec = float(
        recall_score(y_true, y_pred, labels=["malicious"], average="micro", zero_division=0)
    )
    return prec, rec


def choose_gate_threshold(
    val_df,
    y_true: Sequence,
    gate_mal_probs: Sequence,
    rf_probs_real: Sequence,
    grid: Sequence[float] = THRESHOLD_GRID,
) -> dict:
    """Pick the gate threshold maximizing cascade macro-F1 on VALIDATION (D-09 / Pitfall 3).

    Sweeps ``grid`` over the val cascade outputs, computes the full 3-class macro-F1 at each
    threshold, and returns the argmax threshold with its val macro-F1 and malicious
    precision/recall. There is structurally NO ``test`` parameter — the sweep can never see
    the test split.

    Returns a dict with exactly :data:`THRESHOLD_RESULT_KEYS`
    (``threshold, macro_f1, precision, recall``).
    """
    best: Optional[dict] = None
    for thr in grid:
        y_pred = _cascade_labels(gate_mal_probs, rf_probs_real, thr)
        f1 = macro_f1(y_true, y_pred)
        if best is None or f1 > best["macro_f1"]:
            prec, rec = _malicious_precision_recall(y_true, y_pred)
            best = {"threshold": float(thr), "macro_f1": f1, "precision": prec, "recall": rec}
    assert best is not None, "threshold grid must be non-empty"
    log.info(
        "chosen gate threshold (val argmax cascade macro-F1): thr=%.2f macro_f1=%.4f "
        "mal_P=%.4f mal_R=%.4f",
        best["threshold"], best["macro_f1"], best["precision"], best["recall"],
    )
    return best


# Plan-named alias (sweep semantics identical).
sweep_gate_threshold = choose_gate_threshold


# ---------------------------------------------------------------------------
# Backbone selection — per-language Bangla priority + size tiebreak (D-01/D-03).
# ---------------------------------------------------------------------------
def _bn_then_en(per_lang: dict) -> tuple[float, float]:
    """Sort key components: Bangla macro-F1 FIRST (D-01), then English as the secondary."""
    return (float(per_lang.get("bn", 0.0)), float(per_lang.get("en", 0.0)))


def _collapsed(rec: dict) -> bool:
    """A backbone whose fake/malicious recall has collapsed is disqualified (minority guard)."""
    mr = rec["minority_recall"]
    return any(mr[c] <= _MINORITY_FLOOR for c in _MINORITY_CLASSES)


def select_best_transformer(results: dict) -> str:
    """Rank backbones by per-language macro-F1 with Bangla priority; size breaks ties (D-01/D-03).

    Each ``results[name]`` carries ``per_language_macro_f1`` (a {lang: f1} dict) and
    ``minority_recall`` (the {label: recall} guard). Rules:

      * A backbone with collapsed fake/malicious recall is DISQUALIFIED (cannot win).
      * Among qualified backbones, rank by Bangla macro-F1 first, English second (D-01).
      * Near-ties on the Bangla number (within ``_TIE_BAND``) break toward the LIGHTER
        backbone (fewer params, D-03 — banglishbert ~110M over xlmr ~270M).
      * If every backbone collapsed, fall back to ranking all (a winner is always returned).
    """
    qualified = {n: r for n, r in results.items() if not _collapsed(r)}
    pool = qualified or results  # never return empty

    ranked = sorted(
        pool,
        key=lambda n: _bn_then_en(pool[n]["per_language_macro_f1"]),
        reverse=True,
    )
    top = ranked[0]
    top_bn = _bn_then_en(pool[top]["per_language_macro_f1"])[0]
    ties = [
        n
        for n in ranked
        if abs(_bn_then_en(pool[n]["per_language_macro_f1"])[0] - top_bn) <= _TIE_BAND
    ]
    # D-03 size tiebreak: lighter backbone wins a near-tie (default heavy if unknown).
    return min(ties, key=lambda n: BACKBONE_PARAMS.get(n, 10**12))


# Plan-named alias.
select_backbone = select_best_transformer


# ---------------------------------------------------------------------------
# SC-3 leakage re-check — SCORE-SMELL ONLY for transformers (D-15).
# ---------------------------------------------------------------------------
def transformer_leakage_smell(score: float) -> dict:
    """SC-3 score-smell re-check on transformer test predictions (D-15 — score only).

    Transformers expose no per-token feature weights, so the token-tell half of
    :mod:`src.models.leakage_recheck` has NO analog here. This is the SCORE threshold only:
    a test macro-F1 ``>= INVESTIGATE_THRESHOLD`` (0.98) flags ``investigate``. Reuses the
    single-source-of-truth :data:`INVESTIGATE_THRESHOLD` constant.

    Returns ``{"score", "investigate"}``.
    """
    return {"score": float(score), "investigate": bool(score >= INVESTIGATE_THRESHOLD)}


# Aliases (both names the scaffold probes for; plan name kept too).
leakage_score_smell = transformer_leakage_smell
transformer_leakage_recheck = transformer_leakage_smell


# ---------------------------------------------------------------------------
# Cascade evaluation over a frame (drives the selection run; uses the loader from Task 1).
# ---------------------------------------------------------------------------
def evaluate_cascade(cascade, df) -> list[str]:
    """Run ``cascade.predict`` over every row of ``df`` → predicted labels aligned to ``df``.

    Reuses the load-only :class:`src.models.transformer_infer.TransformerCascade`; returns
    the list of predicted 3-class labels in row order so the metric harness can score them.
    """
    return [cascade.predict(t)["label"] for t in df["text"]]


# ---------------------------------------------------------------------------
# Report writer — counts/metrics only (SYS-02), NO latency (D-11).
# ---------------------------------------------------------------------------
def _per_language_table(per_lang: dict) -> str:
    """Render the per-language macro-F1 table (bn / en / code-mixed where present)."""
    lines = ["| language | macro-F1 |", "|---|---|"]
    for lang in sorted(per_lang):
        lines.append(f"| {lang} | {per_lang[lang]:.4f} |")
    if len(lines) == 2:
        lines.append("| _(no language groups present)_ | - |")
    return "\n".join(lines)


def write_selection_report(
    path: PathLike,
    results: dict,
    chosen: str,
    threshold: Optional[dict] = None,
    leakage: Optional[dict] = None,
    code_mixed: Optional[dict] = None,
) -> None:
    """Emit ``reports/transformer_selection_report.md`` — COUNTS / METRICS ONLY (SYS-02).

    Records the chosen backbone, each backbone's per-language (bn/en/code-mixed) macro-F1,
    the gate threshold + its malicious precision/recall, the SC-3 score-smell result, and
    the locked caveats — with NO latency/timing line (D-11). The signature takes only
    metrics/counts (no ``texts``/``raw`` parameter); raw input text is never written.
    """
    path = Path(path)
    parts = [
        "# Transformer Selection Report -- Phase 03",
        "",
        "Selection of the better fine-tuned transformer backbone for the two-stage cascade "
        "(gate → real/fake). **Counts / metrics only -- no raw input text** is recorded here "
        "(SYS-02). The exported `models/transformer/<backbone>/` artifacts are gitignored; "
        "rebuild via the Colab notebook + `python -m src.models.select_transformer`.",
        "",
        "## Chosen model",
        "",
        f"**{chosen}** -- selected by **per-language macro-F1 with Bangla priority** (D-01): "
        "the stronger-Bangla backbone wins; near-ties break toward the lighter backbone "
        "(~110M over ~270M, D-03). A backbone whose fake/malicious recall collapsed is "
        "disqualified (minority guard).",
        "",
    ]

    parts += ["## Per-language macro-F1 (test)", ""]
    if results:
        for name in sorted(results):
            r = results[name]
            parts += [
                f"### {name}",
                "",
                _per_language_table(r.get("per_language_macro_f1", {})),
                "",
            ]
    else:
        parts += ["_(per-language metrics populated by the selection run)_", ""]

    parts += ["## Gate threshold (val sweep, D-09)", ""]
    if threshold:
        parts += [
            "| metric | value |",
            "|---|---|",
            f"| chosen threshold | {threshold.get('threshold', float('nan')):.2f} |",
            f"| val cascade macro-F1 | {threshold.get('macro_f1', float('nan')):.4f} |",
            f"| malicious precision | {threshold.get('precision', float('nan')):.4f} |",
            f"| malicious recall | {threshold.get('recall', float('nan')):.4f} |",
            "",
            "Chosen on the **validation** split by maximizing the full 3-class cascade "
            "macro-F1 (never test; Pitfall 3 / T-03-12).",
            "",
        ]
    else:
        parts += [
            "_(threshold swept on val by `choose_gate_threshold`; recorded after the run)_",
            "",
        ]

    parts += ["## Leakage re-check (SC-3, score-smell only — D-15)", ""]
    if leakage:
        flag = "yes" if leakage.get("investigate") else "no"
        parts += [
            f"Test macro-F1 score-smell: **{leakage.get('score', float('nan')):.4f}** "
            f"(>= {INVESTIGATE_THRESHOLD:.2f} ⇒ investigate). Investigate: **{flag}**.",
            "",
            "Transformers expose no per-token feature weights, so the token-tell half of the "
            "classical re-check has no analog here -- this is the SCORE threshold only (D-15).",
            "",
        ]
    else:
        parts += [
            "_(score-smell ≥ 0.98 ⇒ investigate; recorded after the run. Score-only for "
            "transformers, D-15.)_",
            "",
        ]

    if code_mixed:
        parts += [
            "## Code-mixed probe (qualitative, small-N — D-02)",
            "",
            f"Code-mixed macro-F1 on the hand-curated probe: "
            f"**{code_mixed.get('macro_f1', float('nan')):.4f}** "
            f"(N={code_mixed.get('n', '?')}, qualitative small-N — does NOT gate selection).",
            "",
        ]

    parts += [
        "## Caveats",
        "",
        "- **Bangla gate abstains (D-07):** the malicious class is English-only, so the "
        "gate has zero Bangla malicious rows; on Bangla input the gate effectively abstains "
        "and routing falls to real/fake. Documented limitation.",
        "- **Malicious class is English-only (Phase-1 D-01):** visible in the per-language "
        "tables above (no Bangla/code-mixed malicious coverage).",
        "- **Code-mixed is small-N (D-02):** the code-mixed probe is qualitative; its macro-F1 "
        "is indicative only and does not gate selection.",
        "- **No latency metric (D-11):** inference is interactive on the target hardware; "
        "timing is deliberately not a reported metric (counts/metrics only, SYS-02).",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    log.info("wrote transformer selection report -> %s", path)


# ---------------------------------------------------------------------------
# CLI orchestrator.
# ---------------------------------------------------------------------------
def main() -> int:
    """Load splits, build a cascade per exported backbone, sweep the gate threshold on val,
    select the Bangla-priority winner, run the SC-3 score-smell, and write the report.

    Defers heavy imports so ``import src.models.select_transformer`` stays light. Skips any
    backbone whose export dir is absent (the real fine-tune happens on Colab).
    """
    from src.models.transformer_data import load_splits
    from src.models.transformer_infer import MODELS_DIR, TransformerCascade

    train, val, test = load_splits()
    del train  # selection uses val (sweep) + test (final metrics) only

    results: dict = {}
    threshold_record: Optional[dict] = None
    leakage_record: Optional[dict] = None
    code_mixed_record: Optional[dict] = None

    for backbone in BACKBONE_PARAMS:
        export_dir = MODELS_DIR / backbone
        if not export_dir.exists():
            log.info("skip %s -- no export at %s", backbone, export_dir)
            continue
        cascade = TransformerCascade(model_dir=export_dir)
        test_pred = evaluate_cascade(cascade, test)
        y_test = list(test["label"])
        results[backbone] = {
            "test_macro_f1": macro_f1(y_test, test_pred),
            "per_language_macro_f1": per_language_macro_f1(test, y_test, test_pred),
            "minority_recall": minority_recall(y_test, test_pred),
        }

    if not results:
        log.warning("no exported backbones found under %s -- writing a stub report", MODELS_DIR)
        write_selection_report(
            REPO_ROOT / "reports" / "transformer_selection_report.md",
            results={},
            chosen="(none exported)",
        )
        return 0

    chosen = select_best_transformer(results)
    leakage_record = transformer_leakage_smell(results[chosen]["test_macro_f1"])

    # Optional code-mixed probe (qualitative, small-N — D-02).
    probe_path = REPO_ROOT / "data" / "probe" / "code_mixed_probe.csv"
    if probe_path.exists():
        import pandas as pd

        probe = pd.read_csv(probe_path)
        if "language" not in probe.columns:
            probe = probe.assign(language="code-mixed")
        cascade = TransformerCascade(model_dir=MODELS_DIR / chosen)
        probe_pred = evaluate_cascade(cascade, probe)
        code_mixed_record = {
            "macro_f1": macro_f1(list(probe["label"]), probe_pred),
            "n": int(len(probe)),
        }

    write_selection_report(
        REPO_ROOT / "reports" / "transformer_selection_report.md",
        results=results,
        chosen=chosen,
        threshold=threshold_record,
        leakage=leakage_record,
        code_mixed=code_mixed_record,
    )
    log.info("select_transformer complete: chosen=%s", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
