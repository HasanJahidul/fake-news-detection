"""DATA-02 — label mapping (D-06/D-07) + provenance schema / Bangla Parquet round-trip (D-12/D-13).

Implemented in plan 01-03 (src/data/label_map.py + src/data/schema.py).
"""

import pytest

from src.data.label_map import (
    LABELS,
    LIAR_COLLAPSE,
    map_dataframe,
    map_label,
)


# ---------------------------------------------------------------------------
# Task 1 — label mapping (D-06 LIAR collapse, D-07 fold, SMS ham drop)
# ---------------------------------------------------------------------------


class TestLabelMap:
    def test_labels_locked(self):
        assert LABELS == ("real", "fake", "malicious")

    def test_liar_fake_collapse(self):
        # D-06: pants-fire + false + barely-true -> fake
        for lbl in ("pants-fire", "false", "barely-true"):
            assert map_label("liar", lbl) == "fake", lbl

    def test_liar_real_collapse(self):
        # D-06: mostly-true + true -> real
        for lbl in ("mostly-true", "true"):
            assert map_label("liar", lbl) == "real", lbl

    def test_liar_half_true_dropped(self):
        # D-06: half-true is genuinely mixed -> DROPPED (None)
        assert map_label("liar", "half-true") is None

    def test_liar_collapse_table_matches(self):
        # The explicit collapse table encodes exactly D-06.
        assert LIAR_COLLAPSE["pants-fire"] == "fake"
        assert LIAR_COLLAPSE["false"] == "fake"
        assert LIAR_COLLAPSE["barely-true"] == "fake"
        assert LIAR_COLLAPSE["mostly-true"] == "real"
        assert LIAR_COLLAPSE["true"] == "real"
        assert LIAR_COLLAPSE["half-true"] is None

    def test_isot_mapping(self):
        assert map_label("isot", "true") == "real"
        assert map_label("isot", "fake") == "fake"

    def test_banfakenews_mapping(self):
        assert map_label("banfakenews", "authentic") == "real"
        assert map_label("banfakenews", "fake") == "fake"

    def test_banfakenews2_mapping(self):
        # v2 ships a numeric Label (1 = authentic/real, 0 = fake)
        assert map_label("banfakenews2", "1") == "real"
        assert map_label("banfakenews2", "0") == "fake"

    def test_smsspam_mapping(self):
        # spam -> malicious; ham is NOT "real news" -> DROPPED
        assert map_label("smsspam", "spam") == "malicious"
        assert map_label("smsspam", "ham") is None

    def test_phishing_mapping(self):
        # phishing corpora label 1 = phishing/malicious; 0 = legit -> DROPPED
        assert map_label("phishing", "1") == "malicious"
        assert map_label("phishing", "0") is None

    def test_unknown_source_raises(self):
        with pytest.raises((KeyError, ValueError)):
            map_label("nonexistent_source", "whatever")

    def test_map_dataframe_drops_and_retains_provenance(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            [
                ("a", "liar", "half-true"),   # dropped
                ("b", "liar", "true"),        # -> real
                ("c", "liar", "false"),       # -> fake
                ("d", "smsspam", "ham"),      # dropped
                ("e", "smsspam", "spam"),     # -> malicious
                ("f", "isot", "true"),        # -> real
            ],
            columns=["text", "source_dataset", "original_label"],
        )
        out = map_dataframe(df)
        # half-true + ham rows dropped
        assert set(out["text"]) == {"b", "c", "e", "f"}
        # original_label retained on every surviving row (D-13 provenance)
        assert "original_label" in out.columns
        assert out.loc[out["text"] == "b", "original_label"].iloc[0] == "true"
        # every surviving label is a valid 3-class label
        assert set(out["label"]) <= set(LABELS)
        assert set(out["label"]) == {"real", "fake", "malicious"}


# ---------------------------------------------------------------------------
# Task 2 — D-13 provenance schema + Unicode-safe Parquet round-trip
# ---------------------------------------------------------------------------


def _full_corpus(sample_corpus):
    """Augment the conftest fixture (lacks split + dedup_cluster_id) into the full D-13 shape."""
    df = sample_corpus.copy()
    df["split"] = ["train", "train", "val", "val", "test", "test"]
    df["dedup_cluster_id"] = list(range(len(df)))
    return df


def test_provenance_complete(sample_corpus):
    """Every row has the full D-13 schema, no nulls, label in {real,fake,malicious}."""
    from src.data.schema import PROVENANCE_COLUMNS, validate_provenance

    required = {
        "text",
        "label",
        "source_dataset",
        "original_label",
        "language",
        "split",
        "dedup_cluster_id",
    }
    assert required == set(PROVENANCE_COLUMNS)

    df = _full_corpus(sample_corpus)
    assert required <= set(df.columns)
    assert set(df["label"]) <= {"real", "fake", "malicious"}

    # validator passes on a complete frame ...
    validate_provenance(df)

    # ... and raises on a missing column
    with pytest.raises((ValueError, KeyError)):
        validate_provenance(df.drop(columns=["language"]))

    # ... and raises on a null in a required column
    nulled = df.copy()
    nulled.loc[0, "text"] = None
    with pytest.raises((ValueError, AssertionError)):
        validate_provenance(nulled)

    # ... and raises on an out-of-vocabulary label
    badlabel = df.copy()
    badlabel.loc[0, "label"] = "satire"
    with pytest.raises((ValueError, AssertionError)):
        validate_provenance(badlabel)


def test_bangla_roundtrip(tmp_path, sample_bn, sample_corpus):
    """A Bangla string survives a Parquet write -> read byte-identical (D-12)."""
    from src.data.schema import read_parquet, write_parquet

    df = _full_corpus(sample_corpus)
    # force a known Bangla string into a row so we can assert exact equality
    df.loc[0, "text"] = sample_bn

    path = tmp_path / "sub" / "roundtrip.parquet"  # parent dir does not exist yet
    write_parquet(df, path)
    back = read_parquet(path)

    got = back.loc[0, "text"]
    assert got == sample_bn
    # byte-identical UTF-8
    assert got.encode("utf-8") == sample_bn.encode("utf-8")
    # full frame survives (column-set + row-count)
    assert set(back.columns) >= set(df.columns)
    assert len(back) == len(df)
