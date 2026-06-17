"""Assemble standardized raw CSVs into a single cleaned, split 3-class corpus.

Steps: load data/raw/*.csv -> clean text -> drop empties/dupes -> per-class caps
(balance) -> stratified train/val/test split -> write parquet + dataset stats.

Run:  python -m src.data.build_corpus
If data/raw/ is empty, it auto-runs the synthetic downloader so the pipeline is
always runnable end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import LABELS, load_config, path as cfg_path
from .preprocess import clean_text, detect_lang

RAW_DIR = cfg_path("raw_dir")
PROC_DIR = cfg_path("processed_dir")


def _load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print("No raw CSVs found — generating synthetic fallback ...")
        from .download import main as dl_main
        dl_main(synthetic_only=True)
        files = sorted(RAW_DIR.glob("*.csv"))
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin(LABELS)].copy()
    return df


def _balance(df: pd.DataFrame, caps: dict, seed: int) -> pd.DataFrame:
    parts = []
    for lab, grp in df.groupby("label"):
        cap = caps.get(lab)
        if cap and len(grp) > cap:
            grp = grp.sample(n=cap, random_state=seed)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def build() -> None:
    cfg = load_config()
    dcfg = cfg["data"]
    seed = dcfg["random_state"]
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_raw()
    print(f"Loaded {len(df)} raw rows from {RAW_DIR}")

    # Clean + re-detect language on cleaned text.
    df["text"] = df["text"].map(lambda t: clean_text(t, max_chars=dcfg["max_chars"]))
    df = df[df["text"].str.split().str.len() >= 3]          # need >=3 tokens
    df["lang"] = df["text"].map(detect_lang)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"After clean/dedup: {len(df)} rows")

    df = _balance(df, dcfg.get("caps", {}) or {}, seed)
    print(f"After per-class caps: {len(df)} rows")

    # Stratified split: train / temp, then temp -> val / test.
    test_size = dcfg["test_size"]
    val_size = dcfg["val_size"]
    train, temp = train_test_split(
        df, test_size=test_size + val_size, stratify=df["label"], random_state=seed)
    rel_val = val_size / (test_size + val_size)
    val, test = train_test_split(
        temp, test_size=1 - rel_val, stratify=temp["label"], random_state=seed)

    corpus = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")],
        ignore_index=True)
    corpus.to_parquet(cfg_path("corpus"), index=False)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.reset_index(drop=True).to_parquet(PROC_DIR / f"{name}.parquet", index=False)

    _write_stats(corpus, cfg)
    print(f"\nWrote corpus -> {cfg_path('corpus')}")
    print(f"Splits: train={len(train)} val={len(val)} test={len(test)}")


def _write_stats(corpus: pd.DataFrame, cfg) -> None:
    reports = cfg_path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    pivot = (corpus.groupby(["label", "lang"]).size()
             .unstack(fill_value=0))
    by_split = corpus.groupby(["split", "label"]).size().unstack(fill_value=0)
    by_source = corpus.groupby(["source", "label"]).size().unstack(fill_value=0)

    md = ["# Dataset Statistics\n",
          "## Class x Language\n", pivot.to_markdown(), "\n",
          "## Split x Class\n", by_split.to_markdown(), "\n",
          "## Source x Class\n", by_source.to_markdown(), "\n"]
    (reports / "dataset_stats.md").write_text("\n".join(md), encoding="utf-8")
    pivot.to_csv(reports / "dataset_stats.csv")
    print("\n=== Dataset Statistics (class x lang) ===")
    print(pivot)


if __name__ == "__main__":
    try:
        build()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        raise
