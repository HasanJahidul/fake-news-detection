"""DATA-02 + DATA-04 — the corpus orchestrator.

This is the Phase-1 integration point: it composes every data-layer module into the
single committed-report + ``data/processed/*.parquet`` contract that Phases 2/3 consume.
It turns the gitignored raw data (plan 01-02) into the de-leaked, source-disjoint,
fully-provenanced 3-class corpus.

Pipeline ORDER is the leak-free guarantee (RESEARCH System Architecture steps 1->8;
Anti-Pattern "splitting before dedup"):

    1. load each raw source + ``label_map.map_dataframe``   (01-03; drops half-true/ham/legit,
                                                              retains original_label)
    2. ``leakage_strip.strip_dataframe``                     (01-04; ISOT Reuters dateline +
                                                              drop the ISOT date year-leak column)
    3. ``src.preprocess.preprocess`` on the text column      (01-01; the SAME shared fn used at
                                                              inference -- no fork)
    4. ``language.detect_language`` -> ``language`` column   (01-04; D-02 Bengali-ratio tag)
    5. ``dedup.dedup_dataframe`` -> ``dedup_cluster_id``     (01-05; BEFORE splitting -- the
                                                              Anti-Pattern guard)
    6. ``split.derive_group_key``                            (01-05; per-source group keys, A2/A6/A7)
    7. ``split.make_splits(seed=42)`` -> ``split`` column    (01-05; source-disjoint 70/15/15) +
       ``split.assert_disjoint``                             (D-10 gate)
    8. ``schema.write_parquet`` each split                   (01-03; D-13 provenance, Unicode-safe)

Policy (CONTEXT decisions):
  * D-03/D-04 -- store the NATURAL class distribution; do NOT physically rebalance. Balancing
    is a train-time concern (class weights / in-fold resampling), reported here, applied later.
  * D-05     -- keep EVERYTHING that survives dedup + leakage removal; no hard row/size cap.
  * A5       -- the build is deterministic (fixed seed) and re-runnable, so the processed
    Parquet stays gitignored; the committed corpus report carries COUNTS ONLY, never raw text.

CLI: ``python -m src.data.build_corpus``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.data import label_map, leakage_strip, language, dedup, split, schema
from src.preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_corpus")

# Repo root = two levels up from src/data/build_corpus.py.
REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

# Unified intermediate columns produced by the loaders (before mapping/strip/split).
# `text`, `source_dataset`, `original_label`, `raw_group` are required by the downstream
# modules; `date` is loaded ONLY for ISOT so leakage_strip can drop the year-leak column.
_LOAD_COLUMNS = ["text", "source_dataset", "original_label", "raw_group"]


# ---------------------------------------------------------------------------
# Per-source loaders -> a unified intermediate frame
# ---------------------------------------------------------------------------
def _frame(records: list[dict]) -> pd.DataFrame:
    """Build a normalized intermediate frame; missing columns default to None."""
    df = pd.DataFrame(records)
    for col in _LOAD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)


def load_banfakenews(raw_dir: Path) -> pd.DataFrame:
    """BanFakeNews v1 (Bangla). Authentic-48K + Fake-1K carry a `label` column; the
    Labeled* files do not -- their class is the filename (Authentic->real, Fake->fake)."""
    src = raw_dir / "banfakenews"
    frames: List[pd.DataFrame] = []
    if not src.exists():
        return _frame([])
    file_label = {
        "Authentic-48K.csv": None,        # has its own `label` column
        "Fake-1K.csv": None,              # has its own `label` column
        "LabeledAuthentic-7K.csv": "authentic",
        "LabeledFake-1K.csv": "fake",
    }
    for fname, forced in file_label.items():
        p = src / fname
        if not p.exists():
            continue
        df = _read_csv(p)
        text = (df.get("headline", "").fillna("") + " " + df.get("content", "").fillna("")).str.strip()
        orig = df["label"] if (forced is None and "label" in df.columns) else forced
        frames.append(pd.DataFrame({
            "text": text,
            "source_dataset": "banfakenews",
            "original_label": orig,
            "raw_group": df.get("domain"),
        }))
    return _frame([]) if not frames else pd.concat(frames, ignore_index=True)


def load_banfakenews2(raw_dir: Path) -> pd.DataFrame:
    """BanFakeNews-2.0 (Bangla, better balanced). Columns Headline/Content/Label (numeric 1/0)."""
    src = raw_dir / "banfakenews2"
    if not src.exists():
        return _frame([])
    frames = []
    for fname in ("train_cleaned.csv", "val_cleaned.csv", "test_cleaned.csv"):
        p = src / fname
        if not p.exists():
            continue
        df = _read_csv(p)
        text = (df.get("Headline", "").fillna("") + " " + df.get("Content", "").fillna("")).str.strip()
        frames.append(pd.DataFrame({
            "text": text,
            "source_dataset": "banfakenews2",
            "original_label": df.get("Label"),
            "raw_group": None,  # no publisher column -> synthetic per-row group
        }))
    return _frame([]) if not frames else pd.concat(frames, ignore_index=True)


def load_isot(raw_dir: Path) -> pd.DataFrame:
    """ISOT (English). Class is the source file (True.csv->true, Fake.csv->fake). Carries
    a `date` (year-leak, dropped at strip) and `subject` (group key when class-mixed, A7)."""
    src = raw_dir / "isot"
    if not src.exists():
        return _frame([])
    # Files may be nested (e.g. isot/News_Dataset/True.csv).
    frames = []
    for label, pattern in (("true", "True.csv"), ("fake", "Fake.csv")):
        matches = list(src.rglob(pattern))
        if not matches:
            continue
        df = _read_csv(matches[0])
        text = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")).str.strip()
        frames.append(pd.DataFrame({
            "text": text,
            "source_dataset": "isot",
            "original_label": label,
            "raw_group": df.get("subject"),
            "date": df.get("date"),  # ISOT-only; leakage_strip.strip_dataframe drops it
        }))
    if not frames:
        return _frame([])
    out = pd.concat(frames, ignore_index=True)
    for col in _LOAD_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out


def load_liar(raw_dir: Path) -> pd.DataFrame:
    """LIAR (English short statements). 14-col TSV, no header: col1=label, col2=statement,
    col4=speaker (Pitfall 2 -- raw TSV, never the deprecated loader)."""
    src = raw_dir / "liar"
    if not src.exists():
        return _frame([])
    frames = []
    for fname in ("train.tsv", "valid.tsv", "test.tsv"):
        p = src / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t", header=None, dtype=str, keep_default_na=False)
        frames.append(pd.DataFrame({
            "text": df[2].fillna("").str.strip(),
            "source_dataset": "liar",
            "original_label": df[1],
            "raw_group": df[4],  # speaker
        }))
    return _frame([]) if not frames else pd.concat(frames, ignore_index=True)


def load_smsspam(raw_dir: Path) -> pd.DataFrame:
    """SMS Spam (English). v1=label (ham/spam), v2=text; latin-1 encoded. No publisher
    -> always synthetic per-row group (handled in split._SYNTHETIC_SOURCES)."""
    src = raw_dir / "smsspam"
    if not src.exists():
        return _frame([])
    frames = []
    for p in sorted(src.glob("*.csv")):
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="latin-1")
        except Exception:  # pragma: no cover - encoding fallback
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8", on_bad_lines="skip")
        if "v1" in df.columns and "v2" in df.columns:        # Kaggle uciml shape
            text, orig = df["v2"], df["v1"]
        elif "label" in df.columns and "text" in df.columns:  # UCI ucimlrepo fallback
            text, orig = df["text"], df["label"]
        elif "Class" in df.columns and "Text" in df.columns:
            text, orig = df["Text"], df["Class"]
        else:
            continue
        frames.append(pd.DataFrame({
            "text": text.fillna("").str.strip(),
            "source_dataset": "smsspam",
            "original_label": orig,
            "raw_group": None,
        }))
    return _frame([]) if not frames else pd.concat(frames, ignore_index=True)


def load_phishing(raw_dir: Path) -> pd.DataFrame:
    """Phishing emails (English, intentionally-malicious text). Multiple sub-CSVs; shapes
    vary (subject/body/label or text_combined/label). Sender domain -> group key when
    present, else synthetic (A6)."""
    src = raw_dir / "phishing"
    if not src.exists():
        return _frame([])
    frames = []
    for p in sorted(src.glob("*.csv")):
        if p.name == "phishing_email.csv":
            # Consolidated set -- avoid double-counting the per-corpus files it aggregates.
            continue
        df = _read_csv(p)
        if "text_combined" in df.columns:
            text = df["text_combined"].fillna("")
        else:
            text = (df.get("subject", "").fillna("") + " " + df.get("body", "").fillna("")).str.strip()
        if "label" not in df.columns:
            continue
        sender = df.get("sender")
        raw_group = None
        if sender is not None:
            # sender domain (after the @) as the natural group; None where absent.
            raw_group = sender.fillna("").str.extract(r"@([^>\s]+)", expand=False)
            raw_group = raw_group.where(raw_group.astype(bool), None)
        frames.append(pd.DataFrame({
            "text": text,
            "source_dataset": "phishing",
            "original_label": df["label"],
            "raw_group": raw_group,
        }))
    return _frame([]) if not frames else pd.concat(frames, ignore_index=True)


_LOADERS = (
    load_banfakenews,
    load_banfakenews2,
    load_isot,
    load_liar,
    load_smsspam,
    load_phishing,
)


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Load every present source into one intermediate frame (text + provenance + raw_group)."""
    frames = []
    for loader in _LOADERS:
        df = loader(raw_dir)
        if len(df):
            log.info("loaded %d rows from %s", len(df), loader.__name__)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"no source data found under {raw_dir} -- run `python -m src.data.acquire` first"
        )
    # `date` only exists on ISOT; align columns across frames.
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_corpus(
    raw_dir: str | Path = None,
    out_dir: str | Path = None,
    seed: int = SEED,
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compose the full leak-safe pipeline and write ``data/processed/{train,val,test}.parquet``.

    Steps run in the EXACT order the leak-free guarantee depends on (dedup + strip BEFORE
    split). Returns the assembled, validated, fully-split corpus DataFrame (D-13 schema +
    build-time helper columns). ``raw_df`` lets callers/tests inject a synthetic raw frame.
    """
    raw_dir = Path(raw_dir) if raw_dir else REPO_ROOT / "data" / "raw"
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "data" / "processed"

    # 1. load + label-map (drops None-mapped rows; retains original_label).
    raw = raw_df if raw_df is not None else load_raw(raw_dir)
    df = label_map.map_dataframe(raw)
    log.info("after label_map: %d rows", len(df))

    # 2. offline source-leakage strip (ISOT Reuters dateline + drop ISOT date column).
    df = leakage_strip.strip_dataframe(df)

    # 3. shared preprocess() on the text column -- the SAME fn used at inference (D-08).
    df["text"] = [preprocess(t) for t in df["text"]]
    # Drop rows that preprocess emptied (e.g. whitespace-only) -- no null/empty text in corpus.
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    log.info("after preprocess (non-empty): %d rows", len(df))

    # 4. language tag -> D-13 `language` column (D-02 Bengali ratio).
    df["language"] = [language.detect_language(t) for t in df["text"]]

    # 5. dedup BEFORE split (Anti-Pattern guard) -> dedup_cluster_id + removal_rate attr.
    df = dedup.dedup_dataframe(df)
    removal_rate = df.attrs.get("removal_rate", 0.0)
    mixed_clusters = df.attrs.get("mixed_label_clusters", set())
    log.info("after dedup: %d rows (removal_rate=%.4f, mixed_label_clusters=%d)",
             len(df), removal_rate, len(mixed_clusters))

    # 6. per-source group key (A2/A6/A7).
    df = split.derive_group_key(df)

    # 7. source-disjoint 70/15/15 split (deterministic seed) + disjointness gate.
    df = split.make_splits(df, seed=seed)
    split.assert_disjoint(df)

    # D-03/D-04: distribution is REPORTED, not rebalanced. D-05: no size cap -- everything
    # surviving dedup + leakage removal is kept. (No down/up-sampling happens anywhere above.)

    # Re-attach the dedup attrs the split copies dropped (for the report).
    df.attrs["removal_rate"] = removal_rate
    df.attrs["mixed_label_clusters"] = mixed_clusters

    # 8. validate provenance, then write each split to Parquet (D-13, Unicode-safe).
    schema.validate_provenance(df)
    for s in ("train", "val", "test"):
        part = df[df["split"] == s].reset_index(drop=True)
        schema.write_parquet(part, out_dir / f"{s}.parquet")
        log.info("wrote %s.parquet: %d rows", s, len(part))

    return df


# ---------------------------------------------------------------------------
# Reporting (counts only -- SYS-02 / A5: NO raw article text in the committed report)
# ---------------------------------------------------------------------------
def _counts_table(series: pd.Series, header: str) -> str:
    vc = series.value_counts().sort_index()
    lines = [f"| {header} | count |", "|---|---|"]
    for k, v in vc.items():
        lines.append(f"| {k} | {int(v)} |")
    lines.append(f"| **total** | **{int(vc.sum())}** |")
    return "\n".join(lines)


def _crosstab(df: pd.DataFrame, index: str, columns: str) -> str:
    ct = pd.crosstab(df[index], df[columns])
    cols = list(ct.columns)
    header = f"| {index} \\ {columns} | " + " | ".join(map(str, cols)) + " | total |"
    sep = "|---|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = [header, sep]
    for idx, row in ct.iterrows():
        vals = " | ".join(str(int(row[c])) for c in cols)
        lines.append(f"| {idx} | {vals} | {int(row.sum())} |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, report_path: str | Path) -> None:
    """Emit reports/corpus_report.md -- COUNTS ONLY, no raw text (SYS-02 / A5)."""
    report_path = Path(report_path)
    removal_rate = df.attrs.get("removal_rate", 0.0)
    mixed = df.attrs.get("mixed_label_clusters", set())

    parts = [
        "# Corpus Report -- Phase 01 Data Foundation",
        "",
        "Deterministic, re-runnable build of the de-leaked, source-disjoint 3-class corpus "
        "(real / fake / malicious). **Counts only -- no raw article text** is recorded here "
        "(SYS-02). The processed Parquet is gitignored (A5); rebuild with the command below.",
        "",
        "## Build command (deterministic, seed=42)",
        "",
        "```bash",
        "python -m src.data.build_corpus",
        "```",
        "",
        "Writes `data/processed/{train,val,test}.parquet` with the full D-13 provenance schema "
        "(`text, label, source_dataset, original_label, language, split, dedup_cluster_id`).",
        "",
        "## Policy",
        "",
        "- Class distribution is **reported, NOT physically rebalanced** (D-03/D-04). Balancing "
        "(class weights / in-fold resampling) is a train-time concern applied in Phase 2/3.",
        "- **Everything surviving dedup + leakage removal is kept** -- no hard row/size cap (D-05).",
        "- Splits are **source-disjoint** (D-10) 70/15/15; near-dupes removed **before** splitting "
        "(D-11) so no near-duplicate can straddle train/test.",
        "",
        "## Overall class distribution",
        "",
        _counts_table(df["label"], "label"),
        "",
        "## Class distribution per split",
        "",
        _crosstab(df, "split", "label"),
        "",
        "## Split ratios",
        "",
    ]
    total = len(df)
    ratio_lines = ["| split | rows | ratio |", "|---|---|---|"]
    for s in ("train", "val", "test"):
        n = int((df["split"] == s).sum())
        ratio_lines.append(f"| {s} | {n} | {n / total:.4f} |")
    ratio_lines.append(f"| **total** | **{total}** | **1.0000** |")
    parts += ["\n".join(ratio_lines), ""]

    parts += [
        "## Per-language coverage",
        "",
        "Bengali-ratio tag (D-02). The `malicious` class is assembled from SMS/phishing and is "
        "**English-only** (D-01) -- a documented limitation, surfaced in the language x label table.",
        "",
        _counts_table(df["language"], "language"),
        "",
        "### Language x class",
        "",
        _crosstab(df, "language", "label"),
        "",
        "## Per-source row counts",
        "",
        _counts_table(df["source_dataset"], "source_dataset"),
        "",
        "### Source x class",
        "",
        _crosstab(df, "source_dataset", "label"),
        "",
        "## Deduplication",
        "",
        f"- Near-dup + exact removal rate: **{removal_rate:.4f}** "
        f"({removal_rate * 100:.2f}% of pre-dedup rows removed).",
        f"- Mixed-label clusters kept (Pitfall 4, one survivor per cluster x label): "
        f"**{len(mixed)}**.",
        f"- Operating point (A4): SHA1(preprocess) exact pre-pass + MinHashLSH "
        f"(num_perm=128, Jaccard=0.85, char-5-gram UTF-8).",
        "",
        "## Dataset licenses (provenance only -- raw + processed are gitignored, not redistributed)",
        "",
        "| source | license |",
        "|---|---|",
        "| BanFakeNews v1 (cryptexcode/banfakenews) | CC BY-NC-SA 4.0 (LREC-2020) |",
        "| BanFakeNews-2.0 (hrithikmajumdar/bangla-fake-news) | Apache 2.0 |",
        "| ISOT Fake News | academic/research use (cite ISOT, Univ. of Victoria) |",
        "| LIAR | public research (PolitiFact / UCSB) |",
        "| SMS Spam Collection (UCI id 228) | public, free for research |",
        "| Phishing (naserabdullahalam/phishing-email-dataset) | CC BY-SA 4.0 |",
        "",
        "Redistribution risk is nil: `data/raw/` and `data/processed/` are gitignored; only "
        "these counts are committed.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(parts), encoding="utf-8")
    log.info("wrote report -> %s", report_path)


def main() -> int:
    """Build the corpus on data/raw/ and write the committed corpus report."""
    df = build_corpus()
    write_report(df, REPO_ROOT / "reports" / "corpus_report.md")
    log.info("build_corpus complete: %d total rows across train/val/test", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
