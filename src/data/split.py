"""DATA-04 / D-10 — per-source group-key derivation + source-disjoint splits.

Two operations make the held-out test set leak-free and realistic:

  * ``derive_group_key(df)`` builds a ``group_key`` per source (RESEARCH Pattern 3)
    so ``GroupShuffleSplit`` keeps any one source out of two splits. Sources with
    no natural publisher (SMS spam; phishing without a sender) get UNIQUE synthetic
    per-row keys so they distribute freely across splits but never collide.
  * ``make_splits(df, seed=42)`` carves a source-disjoint 70/15/15 train/val/test
    split (first split off 30%, then halve it) and writes a ``split`` column.
  * ``assert_disjoint(df)`` raises if any ``group_key`` spans two splits (the D-10
    source-disjoint gate; T-05-TM mitigation — the build fails if leakage breaks).

Class-presence guard (Pitfall 3): ``GroupShuffleSplit`` does NOT stratify (whole
groups move). After splitting we assert every split holds all 3 classes with
non-trivial minority counts; if violated, ``make_splits`` re-seeds and, as a last
resort, falls back to ``StratifiedGroupKFold`` and picks the best-balanced fold.

Per-source group-key choices (RESEARCH Pattern 3, A2/A6/A7):
  | source        | group key                                              |
  |---------------|--------------------------------------------------------|
  | isot          | `subject` IF class-mixed; class-pure subject leaks ->   |
  |               | synthetic per-row group instead (A7)                    |
  | banfakenews   | domain/source column (A2; synthetic fallback if absent) |
  | liar          | `speaker` (A2)                                          |
  | smsspam       | synthetic `smsspam_{rownum}` (no publisher)            |
  | phishing      | sender domain if present, else synthetic (A6)          |

The per-source natural group value is read from a ``raw_group`` column (populated
upstream from each source's confirmed column per the 01-03 locked schema). When it
is null/absent for a source, a synthetic key is generated.
"""

from __future__ import annotations

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

LABELS = ("real", "fake", "malicious")

# Sources that NEVER use a natural publisher group — always synthetic per-row.
_SYNTHETIC_SOURCES = {"smsspam"}


def _is_class_pure(df, source: str, key_col: str) -> set:
    """Return the set of group values (for ``source``) that map to exactly one
    label — these would LEAK the label through the group and must be synthesized
    (A7, ISOT-subject class-purity check)."""
    sub = df[df["source_dataset"] == source]
    pure = set()
    for gval, grp in sub.groupby(key_col):
        if grp["label"].nunique() <= 1:
            pure.add(gval)
    return pure


def derive_group_key(df, raw_col: str = "raw_group") -> "object":
    """Build a ``group_key`` column per source (Pattern 3). Non-mutating.

    ``raw_col`` carries each row's natural per-source group value (subject for
    ISOT, domain for BanFakeNews, speaker for LIAR, sender domain for phishing).
    Null values, the SMS source, and class-pure ISOT subjects (A7) get unique
    synthetic per-row keys (``{source}_{rownum}``) so they never collide and never
    leak the label.
    """
    out = df.reset_index(drop=True).copy()
    has_raw = raw_col in out.columns

    # A7: ISOT subjects that are class-pure leak the label -> synthesize them.
    isot_pure = set()
    if has_raw and "label" in out.columns:
        isot_pure = _is_class_pure(out, "isot", raw_col)

    keys = []
    for pos in range(len(out)):
        row = out.iloc[pos]
        source = row["source_dataset"]
        raw = row[raw_col] if has_raw else None
        raw_missing = raw is None or (isinstance(raw, float) and raw != raw) or raw == ""

        use_synthetic = (
            source in _SYNTHETIC_SOURCES
            or raw_missing
            or (source == "isot" and raw in isot_pure)
        )
        if use_synthetic:
            keys.append(f"{source}_{pos}")
        else:
            # Natural publisher/speaker/subject group, namespaced by source so two
            # sources sharing a literal value cannot accidentally merge.
            keys.append(f"{source}:{raw}")

    out["group_key"] = keys
    return out


def _all_classes_ok(df, min_minority: int = 1) -> bool:
    """True if every split has all 3 classes with at least ``min_minority`` rows."""
    for s in ("train", "val", "test"):
        counts = df[df["split"] == s]["label"].value_counts()
        for lab in LABELS:
            if counts.get(lab, 0) < min_minority:
                return False
    return True


def _grouped_split(df, group_col: str, seed: int):
    """Single 70/15/15 GroupShuffleSplit attempt -> writes ``split``. Non-mutating."""
    out = df.reset_index(drop=True).copy()
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, hold_idx = next(gss1.split(out, out["label"], out[group_col]))
    hold = out.iloc[hold_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(hold, hold["label"], hold[group_col]))

    out["split"] = "train"
    out.loc[hold.index[val_rel], "split"] = "val"
    out.loc[hold.index[test_rel], "split"] = "test"
    return out


def _stratified_group_fallback(df, group_col: str, seed: int):
    """Fallback (Pitfall 3): StratifiedGroupKFold, pick the best class-balanced
    fold as the test split, then split the remainder into train/val similarly."""
    out = df.reset_index(drop=True).copy()
    # ~15% test -> n_splits ~ 7; clamp to a sane range for tiny frames. StratifiedGroupKFold
    # ALSO requires n_splits <= the smallest per-class member count, else it raises -- clamp
    # by min class count too so the fallback degrades gracefully on small/skewed frames.
    n_groups = out[group_col].nunique()
    min_class = int(out["label"].value_counts().min())
    n_splits = max(2, min(7, n_groups, min_class))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best = None
    for _, test_rel in sgkf.split(out, out["label"], out[group_col]):
        test_labels = set(out.iloc[test_rel]["label"])
        # prefer a fold whose test set carries the most classes
        score = len(test_labels & set(LABELS))
        if best is None or score > best[0]:
            best = (score, test_rel)
    test_rel = best[1]

    out["split"] = "train"
    out.loc[out.index[test_rel], "split"] = "test"

    # Carve val from the remaining train rows with a second SGKF pass.
    rem = out[out["split"] == "train"]
    n_rem_groups = rem[group_col].nunique()
    min_class_rem = int(rem["label"].value_counts().min())
    n_splits2 = max(2, min(6, n_rem_groups, min_class_rem))
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits2, shuffle=True, random_state=seed)
    best2 = None
    for _, val_rel in sgkf2.split(rem, rem["label"], rem[group_col]):
        score = len(set(rem.iloc[val_rel]["label"]) & set(LABELS))
        if best2 is None or score > best2[0]:
            best2 = (score, val_rel)
    out.loc[rem.index[best2[1]], "split"] = "val"
    return out


def make_splits(df, group_col: str = "group_key", seed: int = 42, max_retries: int = 20):
    """Source-disjoint, class-aware 70/15/15 split (D-10). Writes a ``split`` column.

    Tries ``GroupShuffleSplit`` across several seeds until every split carries all
    3 classes (Pitfall 3); if none qualifies, falls back to a class-balanced
    ``StratifiedGroupKFold`` fold. Always source-disjoint (asserts before return).
    """
    if group_col not in df.columns:
        raise KeyError(
            f"make_splits: '{group_col}' missing — call derive_group_key first"
        )

    best_attempt = None
    for k in range(max_retries):
        attempt = _grouped_split(df, group_col, seed + k)
        if _all_classes_ok(attempt):
            assert_disjoint(attempt)
            return attempt
        if best_attempt is None:
            best_attempt = attempt

    # Fallback to stratified-group split (Pitfall 3) when no seed balanced classes.
    fallback = _stratified_group_fallback(df, group_col, seed)
    if _all_classes_ok(fallback):
        assert_disjoint(fallback)
        return fallback

    # Last resort: return the best GroupShuffleSplit attempt (still disjoint) and
    # let the caller's class-presence assertion surface the imbalance.
    assert_disjoint(best_attempt)
    return best_attempt


def assert_disjoint(df, group_col: str = "group_key") -> None:
    """Raise ``AssertionError`` if any ``group_key`` appears in more than one split
    (D-10 source-disjoint gate; T-05-TM mitigation)."""
    by_split = {s: set(g[group_col]) for s, g in df.groupby("split")}
    splits = list(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise AssertionError(
                    f"source-disjoint violation: group(s) {sorted(overlap)} appear "
                    f"in both '{a}' and '{b}' splits (D-10)"
                )
