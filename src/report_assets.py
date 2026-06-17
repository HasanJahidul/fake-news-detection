"""Generate report-ready assets for Chapters 3–5.

Consumes the metrics JSON written by training (classical_metrics.json,
xlmr_metrics.json, banglabert_metrics.json) plus dataset_stats.csv, and emits:
  reports/model_comparison.md / .csv     consolidated metrics table
  reports/confusion_<model>.png          confusion matrix per model
  reports/architecture.png               system block diagram

Run:  python -m src.report_assets
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .config import LABELS, path as cfg_path

REPORTS = cfg_path("reports_dir")


def _load(name: str) -> dict | None:
    p = REPORTS / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _classical_rows() -> List[dict]:
    data = _load("classical_metrics.json")
    rows = []
    if not data:
        return rows
    for model, d in data.items():
        rep = d["report"]
        rows.append({
            "model": model, "type": "classical",
            "accuracy": round(rep["accuracy"], 4),
            "macro_f1": round(rep["macro avg"]["f1-score"], 4),
            "precision": round(rep["macro avg"]["precision"], 4),
            "recall": round(rep["macro avg"]["recall"], 4),
        })
    return rows


def _transformer_rows() -> List[dict]:
    rows = []
    for fname, model in (("xlmr_metrics.json", "xlm-roberta"),
                         ("banglabert_metrics.json", "banglabert")):
        d = _load(fname)
        if not d:
            continue
        rep = d["report"]
        rows.append({
            "model": model, "type": "transformer",
            "accuracy": round(rep["accuracy"], 4),
            "macro_f1": round(rep["macro avg"]["f1-score"], 4),
            "precision": round(rep["macro avg"]["precision"], 4),
            "recall": round(rep["macro avg"]["recall"], 4),
        })
    return rows


def comparison_table() -> List[dict]:
    rows = _classical_rows() + _transformer_rows()
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    if not rows:
        print("No metrics JSON found yet — train models first.")
        return rows
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "model_comparison.md").write_text(
        "# Model Comparison\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
    print("Wrote reports/model_comparison.{csv,md}")
    print(df.to_string(index=False))
    return rows


def confusion_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sources = {"classical": ("classical_metrics.json", None),
               "xlm-roberta": ("xlmr_metrics.json", "confusion_matrix"),
               "banglabert": ("banglabert_metrics.json", "confusion_matrix")}

    def _draw(cm, title, fname):
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); fig.savefig(REPORTS / fname, dpi=150); plt.close(fig)
        print(f"Wrote reports/{fname}")

    cls = _load("classical_metrics.json")
    if cls:
        best = max(cls.items(), key=lambda kv: kv[1]["macro_f1"])
        _draw(best[1]["confusion_matrix"], f"Confusion — {best[0]}",
              "confusion_classical.png")
    for model, (fname, _) in sources.items():
        if model == "classical":
            continue
        d = _load(fname)
        if d:
            _draw(d["confusion_matrix"], f"Confusion — {model}",
                  f"confusion_{model}.png")


def architecture_diagram() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                     fc=color, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14, color="#555", lw=1.3))

    box(2, 78, 22, 14, "User Input\n(Text / URL)", "#dbeafe")
    box(2, 54, 22, 14, "Preprocess\nclean · normalize\nlang-detect (en/bn)\ntokenize", "#e0f2fe")
    arrow(13, 78, 13, 68)

    box(30, 70, 26, 22, "Text Classifier\nTF-IDF + LR/NB/RF\nXLM-RoBERTa (primary)\nBanglaBERT (bn)",
        "#dcfce7")
    box(30, 42, 26, 12, "Source Credibility\ndomain lists + heuristics", "#fef9c3")
    box(30, 26, 26, 12, "Writing-Style\nclickbait/scam lexicon\ncaps · !! · links", "#fee2e2")
    box(30, 10, 26, 12, "External Verify\nNewsAPI trusted sources", "#ede9fe")
    arrow(24, 60, 30, 78); arrow(24, 58, 30, 48)
    arrow(24, 56, 30, 32); arrow(24, 54, 30, 16)

    box(64, 40, 22, 22, "Decision Fusion\nweighted combine\n→ label + confidence\n+ explainability",
        "#fde68a")
    for y in (81, 48, 32, 16):
        arrow(56, y, 64, 51)

    box(90, 40, 9, 22, "Result\nUI", "#bbf7d0")
    arrow(86, 51, 90, 51)

    ax.set_title("AI-Driven Real-Time Detection — System Architecture", fontsize=12)
    fig.tight_layout(); fig.savefig(REPORTS / "architecture.png", dpi=150); plt.close(fig)
    print("Wrote reports/architecture.png")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    comparison_table()
    try:
        confusion_plots()
    except Exception as e:  # noqa: BLE001
        print(f"[confusion] skipped: {e}")
    architecture_diagram()


if __name__ == "__main__":
    main()
