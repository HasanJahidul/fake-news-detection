"""DATA-02 / DATA-04 stubs — implemented in plan 01-06.

build_corpus.py orchestrates load -> strip -> preprocess -> dedup -> split -> Parquet.
"""

import pytest


@pytest.mark.skip(reason="implemented in plan 01-06 (build_corpus.py)")
def test_parquet_written():
    """data/processed/{train,val,test}.parquet are produced by the build."""
    raise NotImplementedError("01-06: assert the three Parquet files exist after build")


@pytest.mark.skip(reason="implemented in plan 01-06 (build_corpus.py)")
def test_provenance_schema_on_disk():
    """On-disk Parquet carries the full D-13 provenance schema with no nulls."""
    raise NotImplementedError("01-06: read Parquet, assert schema columns + non-null")
