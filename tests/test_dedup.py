"""DATA-04 / D-11 — exact + fuzzy near-dup clustering (implemented in plan 01-05).

Survivors retain ``dedup_cluster_id``; mixed-label clusters keep one survivor
per (cluster x label) so contrasting labels are not silently merged (Pitfall 4).
"""

import pandas as pd

from src.data.dedup import assign_clusters, dedup_dataframe


def _make_df(rows):
    """rows: list of (text, label) -> small corpus DataFrame."""
    return pd.DataFrame(rows, columns=["text", "label"])


def test_near_dup_removed():
    """Two near-identical strings cluster together; one survives with a cluster_id."""
    a = "The central bank raised interest rates by half a point this morning in a surprise move."
    # Reworded slightly — same content, high Jaccard on char 5-grams.
    b = "The central bank raised interest rates by half a point this morning in a surprising move."
    df = _make_df([(a, "real"), (b, "real")])

    out = dedup_dataframe(df)

    assert len(out) == 1, "near-dupes must collapse to a single survivor"
    assert "dedup_cluster_id" in out.columns
    assert out["dedup_cluster_id"].notna().all(), "survivor must retain a cluster id"


def test_exact_dup_collapses():
    """Identical text collapses via the SHA1 exact pre-pass."""
    s = "সরকার আজ নতুন একটি নীতি ঘোষণা করেছে যা অর্থনীতিতে প্রভাব ফেলবে।"
    df = _make_df([(s, "real"), (s, "real"), (s, "real")])

    out = dedup_dataframe(df)

    assert len(out) == 1
    # all three shared one cluster
    ids = assign_clusters(df.to_dict("records"))
    assert len(set(ids.values())) == 1


def test_distinct_rows_not_merged():
    """Clearly different texts stay in separate clusters."""
    rows = [
        ("The senate passed the climate bill after a long debate in Washington.", "real"),
        ("চাঁদে গোপন শহর আবিষ্কারের ভাইরাল দাবি সম্পূর্ণ মিথ্যা প্রমাণিত হয়েছে।", "fake"),
        ("WIN a FREE prize now click this link to claim your reward immediately!!!", "malicious"),
    ]
    df = _make_df(rows)
    out = dedup_dataframe(df)
    assert len(out) == 3
    ids = assign_clusters(df.to_dict("records"))
    assert len(set(ids.values())) == 3


def test_mixed_label_cluster_keeps_contrasting_labels():
    """A cluster spanning >1 label keeps one survivor PER (cluster x label) (Pitfall 4)."""
    a = "The president announced a new economic stimulus package during the press briefing today."
    b = "The president announced a new economic stimulus package during the press briefing today!"
    # Same near-dup content but contradictory labels -> both must survive.
    df = _make_df([(a, "real"), (b, "fake")])

    out = dedup_dataframe(df)

    assert len(out) == 2, "contrasting labels in one cluster must both survive"
    # They share the SAME cluster id (they are near-dupes), but differ by label.
    assert out["dedup_cluster_id"].nunique() == 1
    assert set(out["label"]) == {"real", "fake"}


def test_bangla_codepoints_cluster():
    """Char 5-gram shingles (UTF-8) cluster Bangla near-dupes correctly."""
    a = "বাংলাদেশের অর্থনীতি গত বছরে উল্লেখযোগ্য হারে প্রবৃদ্ধি অর্জন করেছে বলে জানা গেছে।"
    b = "বাংলাদেশের অর্থনীতি গত বছরে উল্লেখযোগ্য হারে প্রবৃদ্ধি অর্জন করেছে বলে জানানো হয়েছে।"
    df = _make_df([(a, "real"), (b, "real")])
    out = dedup_dataframe(df)
    assert len(out) == 1, "Bangla near-dupes must cluster via UTF-8 char shingles"


def test_survivor_cluster_ids_unique_per_cluster():
    """Each distinct cluster gets a distinct dedup_cluster_id."""
    rows = [
        ("Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima.", "real"),
        ("Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo limo.", "real"),  # near-dup of #1
        ("Completely unrelated sentence about quantum computing and superposition states.", "fake"),
    ]
    df = _make_df(rows)
    out = dedup_dataframe(df)
    assert len(out) == 2
    assert out["dedup_cluster_id"].nunique() == 2
