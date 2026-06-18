"""DATA-02 — D-13 provenance schema + Unicode-safe Parquet I/O.

Every corpus row carries the full provenance schema (D-13), with no nulls:

    text, label, source_dataset, original_label, language, split, dedup_cluster_id

Storage is Parquet via pyarrow (D-12) — a typed columnar store that round-trips Bangla
Unicode byte-identically. ``write_parquet`` enforces :data:`SCHEMA` (so a stray column
type or an out-of-vocabulary ``label`` is caught at write time); ``read_parquet`` returns
a pandas DataFrame. ``validate_provenance`` is the completeness gate (T-03-TM, V5 input
validation): missing column, null in any required column, or a label outside the locked
3-class set all raise.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.label_map import LABELS

# D-13 provenance schema — field order + types are LOCKED for the orchestrator (plan 01-06).
SCHEMA: pa.Schema = pa.schema(
    [
        ("text", pa.string()),            # UTF-8, preserves Bangla
        ("label", pa.string()),           # real | fake | malicious
        ("source_dataset", pa.string()),  # banfakenews|banfakenews2|isot|liar|smsspam|phishing
        ("original_label", pa.string()),  # source's own label, retained (provenance)
        ("language", pa.string()),        # bn | en | code-mixed | unknown
        ("split", pa.string()),           # train | val | test
        ("dedup_cluster_id", pa.int64()), # near-dup cluster audit id
    ]
)

# Required columns, in canonical order.
PROVENANCE_COLUMNS: list[str] = [f.name for f in SCHEMA]

PathLike = Union[str, os.PathLike]


def validate_provenance(df: pd.DataFrame) -> None:
    """Assert ``df`` satisfies the D-13 provenance contract; raise with a clear message otherwise.

    Checks:
      1. every required column is present (raises ``KeyError``),
      2. no null in any required column (raises ``ValueError``),
      3. every ``label`` is in :data:`LABELS` (raises ``ValueError``).
    """
    missing = [c for c in PROVENANCE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"provenance schema missing required column(s): {missing}")

    null_cols = [c for c in PROVENANCE_COLUMNS if df[c].isnull().any()]
    if null_cols:
        raise ValueError(f"provenance has null values in required column(s): {null_cols}")

    bad = set(df["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"label column has values outside {LABELS}: {sorted(bad)}")


def write_parquet(df: pd.DataFrame, path: PathLike) -> None:
    """Write ``df`` to Parquet under :data:`SCHEMA`, creating parent dirs.

    Validates provenance first (fail closed before persisting). Coerces the frame to the
    canonical column order so ``pa.Table.from_pandas`` binds cleanly to the explicit schema.
    """
    validate_provenance(df)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ordered = df[PROVENANCE_COLUMNS]
    table = pa.Table.from_pandas(ordered, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, p)


def read_parquet(path: PathLike) -> pd.DataFrame:
    """Read a provenance Parquet file back into a pandas DataFrame (UTF-8 preserved)."""
    return pq.read_table(Path(path)).to_pandas()
