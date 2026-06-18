"""DATA-02 — unified source_label -> 3-class label mapping with provenance retention.

The whole project is locked to three classes (CONTEXT D-06/D-07, RESEARCH §user_constraints):

    LABELS = ("real", "fake", "malicious")

- ``real`` / ``fake`` come from the news corpora (ISOT, BanFakeNews v1/v2) and from LIAR
  (collapsed per D-06).
- ``malicious`` is assembled from the spam / phishing corpora ONLY (SMS Spam, phishing email).
  Per D-01 this class is English-only and that is a documented limitation — no translation,
  no synthetic Bangla malicious data.

Two source labels are deliberately DROPPED (mapped to ``None``):

- LIAR ``half-true`` — genuinely mixed; folding it into either class injects label noise (D-06).
- SMS Spam ``ham`` — a non-spam SMS is not "real news"; it does not belong to the ``real`` class.
  (Likewise phishing-corpus ``legit`` rows.)

``original_label`` is ALWAYS retained on every surviving row (D-13 provenance / audit trail).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Locked, project-wide 3-class label set.
LABELS: tuple[str, str, str] = ("real", "fake", "malicious")

# ---------------------------------------------------------------------------
# LIAR collapse (D-06) — the single source of truth for the 6-way -> 3-class fold.
#   fake = pants-fire + false + barely-true
#   real = mostly-true + true
#   half-true -> None (DROPPED)
# ---------------------------------------------------------------------------
LIAR_COLLAPSE: dict[str, Optional[str]] = {
    "pants-fire": "fake",
    "false": "fake",
    "barely-true": "fake",
    "mostly-true": "real",
    "true": "real",
    "half-true": None,  # DROPPED — genuinely mixed (D-06)
}

# ---------------------------------------------------------------------------
# Per-source mapping: source_dataset -> {normalized original_label: 3-class | None}.
#
# Keys are matched case-insensitively against a stripped original_label (see _normalize).
# A value of ``None`` means "drop this row" (e.g. SMS ham, phishing legit, LIAR half-true).
# original_label values reflect what plan 01-02 actually wrote to data/raw (see 01-02 SUMMARY):
#   - banfakenews v1: lowercase semantic labels ("authentic"/"fake")
#   - banfakenews2:   numeric Label column (1 = authentic/real, 0 = fake)
#   - isot:           class is the source file (True.csv -> "true", Fake.csv -> "fake")
#   - liar:           6-way string label at TSV col 1 (see LIAR_COLLAPSE)
#   - smsspam:        v1 column = "ham" | "spam" (latin-1 source)
#   - phishing:       numeric label column (1 = phishing/malicious, 0 = legit)
# ---------------------------------------------------------------------------
SOURCE_MAP: dict[str, dict[str, Optional[str]]] = {
    "banfakenews": {
        "authentic": "real",
        "real": "real",
        "1": "real",
        "fake": "fake",
        "0": "fake",
    },
    "banfakenews2": {
        # BanFakeNews-2.0 (hrithikmajumdar/bangla-fake-news) ships a 4-CLASS numeric Label
        # (observed in data/raw at build time: 3=~80% majority, then 2,1,0), NOT the binary
        # 1/0 the 01-03 contract assumed. Per the dataset's published taxonomy, Label 3 is the
        # AUTHENTIC majority class and 0/1/2 are non-authentic fake-news subtypes (satire /
        # clickbait-fake / false-context). Collapsed to the project's 3-class scheme the same
        # way ISOT/LIAR collapse: authentic -> real, every non-authentic subtype -> fake.
        # NOTE (flagged for human review in 01-06 SUMMARY): the 0/1/2 subtype semantics are
        # taken from the dataset card, not an in-file codebook; if a future audit shows label 0
        # is "satire" that should be excluded rather than treated as fake, drop it here.
        "3": "real",
        "2": "fake",
        "1": "fake",
        "0": "fake",
        "authentic": "real",
        "real": "real",
        "fake": "fake",
    },
    "isot": {
        "true": "real",
        "real": "real",
        "fake": "fake",
        "false": "fake",
    },
    "liar": LIAR_COLLAPSE,
    "smsspam": {
        "spam": "malicious",
        "ham": None,  # DROPPED — a non-spam SMS is not "real news"
    },
    "phishing": {
        "1": "malicious",
        "phishing": "malicious",
        "phish": "malicious",
        "spam": "malicious",
        "0": None,  # DROPPED — legitimate email is not "real news"
        "legit": None,
        "ham": None,
    },
}


def _normalize(original_label: object) -> str:
    """Canonicalize a raw source label for dict lookup (lowercase, stripped, no .0 suffix)."""
    s = str(original_label).strip().lower()
    # Numeric labels can arrive as "1.0"/"0.0" from float-typed columns.
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        s = s[:-2]
    return s


def map_label(source_dataset: str, original_label: object) -> Optional[str]:
    """Map one source's native label to a 3-class label, or ``None`` if the row is dropped.

    Raises ``KeyError`` for an unknown ``source_dataset`` (fail closed — a typo in the
    source tag must never silently swallow rows). Raises ``ValueError`` for a known source
    but an unrecognized ``original_label`` (a new/unexpected source label must not pass silently).
    """
    src = str(source_dataset).strip().lower()
    if src not in SOURCE_MAP:
        raise KeyError(
            f"unknown source_dataset {source_dataset!r}; "
            f"known sources: {sorted(SOURCE_MAP)}"
        )
    table = SOURCE_MAP[src]
    key = _normalize(original_label)
    if key not in table:
        raise ValueError(
            f"unrecognized original_label {original_label!r} for source {source_dataset!r}; "
            f"known labels: {sorted(table)}"
        )
    mapped = table[key]
    # Defensive: a non-None mapping must always be a locked label (T-03-TM).
    if mapped is not None and mapped not in LABELS:
        raise ValueError(f"mapping produced invalid label {mapped!r} (not in {LABELS})")
    return mapped


def map_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`map_label` row-wise, drop ``None``-mapped rows, retain ``original_label``.

    Requires ``source_dataset`` and ``original_label`` columns. Adds/overwrites a ``label``
    column with the 3-class value. ``original_label`` is preserved on every surviving row
    (D-13 provenance). Asserts every surviving ``label`` is in :data:`LABELS` (T-03-TM —
    no mislabeled row can silently enter the corpus).
    """
    for col in ("source_dataset", "original_label"):
        if col not in df.columns:
            raise KeyError(f"map_dataframe requires a {col!r} column; got {list(df.columns)}")

    out = df.copy()
    out["label"] = [
        map_label(src, orig)
        for src, orig in zip(out["source_dataset"], out["original_label"])
    ]
    # Drop rows whose mapping is None (LIAR half-true, SMS ham, phishing legit).
    out = out[out["label"].notna()].reset_index(drop=True)

    # original_label must still be present (provenance, D-13).
    assert "original_label" in out.columns, "original_label was lost during mapping"

    surviving = set(out["label"])
    assert surviving <= set(LABELS), (
        f"map_dataframe produced labels outside {LABELS}: {surviving - set(LABELS)}"
    )
    return out
