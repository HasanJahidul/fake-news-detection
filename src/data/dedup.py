"""DATA-04 / D-11 — two-stage corpus deduplication.

Exact-duplicate pre-pass (SHA1 of the normalized text) followed by fuzzy
near-duplicate clustering with ``datasketch`` MinHashLSH over char-level 5-gram
shingles (UTF-8 encoded, so Bangla codepoints cluster correctly). Every cluster
is assigned one ``dedup_cluster_id``; ``dedup_dataframe`` keeps ONE survivor per
cluster — or one per (cluster x label) when a cluster spans more than one label —
and always RETAINS ``dedup_cluster_id`` on the survivor (D-13).

Operating point (Assumption A4 — char 5-gram / 128 perms / Jaccard 0.85):
the de-facto MinHashLSH near-dup recipe (SlimPajama / bigcode). This runs BEFORE
the split (Anti-Pattern: splitting before dedup lets a near-dup land in both
train and test = silent leakage). Sub-quadratic via LSH banding, not all-pairs
(T-05-DoS): bounded runtime at ~100k rows on local CPU.

Pitfall 4: a near-dup pair carrying CONTRADICTORY labels is NOT merged away —
both survive (one per label) so the contrast is preserved — and the cluster is
flagged in the returned mixed-label set.

Cross-references the shared text contract: the exact-hash key is computed on
``src.preprocess.preprocess(text)`` so dedup is consistent with train/inference.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Tuple

from datasketch import MinHash, MinHashLSH

from src.preprocess import preprocess

# --- Operating point (A4) -------------------------------------------------
NUM_PERM = 128
THRESHOLD = 0.85  # Jaccard; D-11 ~0.9 intent, 0.85 a slightly looser practical default
SHINGLE = 5  # char-level 5-grams: language-agnostic, works on Bangla codepoints

_WS = re.compile(r"\s+")


def _shingles(text: str) -> set:
    """Char-level 5-gram shingle set (whitespace-collapsed)."""
    t = _WS.sub(" ", (text or "").strip())
    if not t:
        return set()
    return {t[i : i + SHINGLE] for i in range(max(len(t) - SHINGLE + 1, 1))}


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))  # UTF-8 so Bangla codepoints hash correctly
    return m


def _exact_key(text: str) -> str:
    """SHA1 over the normalized (preprocess) form — the exact-dup pre-pass key."""
    return hashlib.sha1(preprocess(text or "").encode("utf-8")).hexdigest()


# --- Union-Find -----------------------------------------------------------
class _UF:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the smaller index as the canonical root (stable survivors)
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


def assign_clusters(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    """Assign a ``dedup_cluster_id`` to every row.

    ``rows``: list of dicts, each with at least a ``text`` key.
    Returns ``{row_index: dedup_cluster_id}`` (one id per cluster; near-dupes and
    exact-dupes share an id). Ids are small contiguous integers (stable per call).
    """
    n = len(rows)
    if n == 0:
        return {}

    # 1) Exact pre-pass: group by normalized-text SHA1.
    exact: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        exact.setdefault(_exact_key(r.get("text", "")), []).append(i)

    uf = _UF(range(n))
    # collapse exact dups into the same component up-front
    for idxs in exact.values():
        first = idxs[0]
        for j in idxs[1:]:
            uf.union(first, j)

    # 2) Fuzzy near-dups via MinHashLSH over the exact-representatives only.
    reps = [idxs[0] for idxs in exact.values()]
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    mh: Dict[int, MinHash] = {}
    for i in reps:
        m = _minhash(rows[i].get("text", ""))
        mh[i] = m
        lsh.insert(str(i), m)
    for i in reps:
        for cand in lsh.query(mh[i]):
            j = int(cand)
            if j != i:
                uf.union(i, j)

    # 3) Materialize contiguous cluster ids from union-find roots.
    root_to_cid: Dict[int, int] = {}
    result: Dict[int, int] = {}
    next_cid = 0
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_cid:
            root_to_cid[root] = next_cid
            next_cid += 1
        result[i] = root_to_cid[root]
    return result


def dedup_dataframe(df, text_col: str = "text", label_col: str = "label") -> Tuple:
    """Cluster ``df`` and keep one survivor per cluster (per cluster x label when
    a cluster spans >1 label, Pitfall 4), RETAINING ``dedup_cluster_id``.

    Returns the deduplicated DataFrame. The set of mixed-label cluster ids is
    available via the attribute ``result.attrs['mixed_label_clusters']`` (and is
    also logged). The input frame is not mutated.
    """
    if len(df) == 0:
        out = df.copy()
        out["dedup_cluster_id"] = []
        out.attrs["mixed_label_clusters"] = set()
        return out

    work = df.reset_index(drop=True).copy()
    cids = assign_clusters(work.to_dict("records"))
    work["dedup_cluster_id"] = [cids[i] for i in range(len(work))]

    has_labels = label_col in work.columns

    # Flag mixed-label clusters (clusters with >1 distinct label).
    mixed: set = set()
    if has_labels:
        for cid, grp in work.groupby("dedup_cluster_id"):
            if grp[label_col].nunique() > 1:
                mixed.add(int(cid))

    # Survivor selection: first row per cluster, or per (cluster, label) when the
    # cluster spans multiple labels so contrasting labels are not merged away.
    keys = ["dedup_cluster_id", label_col] if has_labels else ["dedup_cluster_id"]
    survivors = work.drop_duplicates(subset=keys, keep="first")

    out = survivors.reset_index(drop=True)
    out["dedup_cluster_id"] = out["dedup_cluster_id"].astype("int64")
    out.attrs["mixed_label_clusters"] = mixed
    removed = len(work) - len(out)
    rate = removed / len(work) if len(work) else 0.0
    out.attrs["removal_rate"] = rate
    if mixed:
        # Surface contrasting-label clusters (Pitfall 4) without failing the build.
        import warnings

        warnings.warn(
            f"dedup: {len(mixed)} mixed-label cluster(s) kept contrasting labels: "
            f"{sorted(mixed)}",
            stacklevel=2,
        )
    return out
