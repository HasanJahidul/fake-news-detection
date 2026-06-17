"""DATA-02 stubs — implemented in plan 01-03.

Provenance completeness + Bangla Parquet round-trip. Turned green by 01-03.
"""

import pytest


@pytest.mark.skip(reason="implemented in plan 01-03 (label_map.py + schema.py)")
def test_provenance_complete(sample_corpus):
    """Every row has the full D-13 schema, no nulls, label in {real,fake,malicious}."""
    required = {
        "text",
        "label",
        "source_dataset",
        "original_label",
        "language",
        "split",
        "dedup_cluster_id",
    }
    assert required <= set(sample_corpus.columns)
    assert set(sample_corpus["label"]) <= {"real", "fake", "malicious"}


@pytest.mark.skip(reason="implemented in plan 01-03 (schema.py Parquet round-trip)")
def test_bangla_roundtrip():
    """A Bangla string survives a Parquet write -> read byte-identical."""
    raise NotImplementedError("01-03: pyarrow Parquet Unicode round-trip")
