"""DATA-02 / DATA-04 build-integration tests (plan 01-06).

`build_corpus.py` orchestrates load -> label-map -> strip -> preprocess -> language ->
dedup -> group-key -> split -> Parquet. These tests exercise the integration path on a
tiny synthetic raw frame (always), and additionally assert the real on-disk Parquet when
a build has been run against the populated data/raw/ (skipped cleanly otherwise / in CI).
"""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from src.data import schema, build_corpus
from src.data.schema import PROVENANCE_COLUMNS
from src.data.label_map import LABELS

REPO_ROOT = Path(build_corpus.__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"


@pytest.fixture
def synthetic_raw():
    """A multi-source raw intermediate frame (text + provenance + raw_group) that exercises
    every branch: all 3 classes with enough distinct groups per class to satisfy the
    source-disjoint 70/15/15 GroupShuffleSplit, an ISOT Reuters dateline to strip, a dropped
    label (LIAR half-true), and a near-dup pair."""
    rows = []
    # ISOT real (Reuters-datelined, distinct subjects = distinct groups) + fake.
    for i in range(8):
        rows.append((f"WASHINGTON (Reuters) - The senate passed budget bill number {i} after a long debate this week.",
                     "isot", "true", f"politicsNews{i}"))
    for i in range(8):
        rows.append((f"The senate secretly approved hidden bill {i} that nobody ever read, anonymous insiders allege.",
                     "isot", "fake", f"left-news{i}"))
    # BanFakeNews (Bangla) real + fake across distinct publisher groups.
    for i in range(6):
        rows.append((f"সরকার আজ একটি নতুন শিক্ষা নীতি {i} ঘোষণা করেছে যা আগামী বছর কার্যকর হবে বলে জানা গেছে।",
                     "banfakenews", "authentic", f"prothomalo{i}"))
    for i in range(6):
        rows.append((f"চাঁদে গোপন এলিয়েন শহর {i} আবিষ্কার করেছেন বিজ্ঞানীরা এমন ভিত্তিহীন দাবি ভাইরাল হয়েছে ফেসবুকে।",
                     "banfakenews", "fake", f"viralpost{i}"))
    # banfakenews2 near-dup of a banfakenews authentic row (tests dedup retention).
    rows.append(("সরকার আজ একটি নতুন শিক্ষা নীতি 0 ঘোষণা করেছে যা আগামী বছর কার্যকর হবে বলে জানা গেছে।",
                 "banfakenews2", "1", None))
    # LIAR fake/real + a dropped half-true.
    for i in range(4):
        rows.append((f"Claims the new policy {i} will double taxes for every working family across the whole state.",
                     "liar", "false", f"speaker-f{i}"))
    for i in range(4):
        rows.append((f"The mayor cut the ribbon at new public library {i} downtown earlier this calm morning.",
                     "liar", "true", f"speaker-t{i}"))
    rows.append(("This statement is genuinely mixed and should be dropped from the corpus entirely.",
                 "liar", "half-true", "speaker-mixed"))  # DROPPED by label_map
    # malicious: SMS spam + phishing (synthetic per-row groups).
    for i in range(8):
        rows.append((f"WIN a FREE prize number {i} now!!! Click http://spam{i}.example to claim it today.",
                     "smsspam", "spam", None))
    for i in range(8):
        rows.append((f"Your bank account {i} is locked. Verify immediately at http://phish{i}.example/login now.",
                     "phishing", "1", None))
    return pd.DataFrame(rows, columns=["text", "source_dataset", "original_label", "raw_group"])


def test_build_corpus_integration(synthetic_raw, tmp_path):
    """The full pipeline runs end-to-end on a synthetic raw frame and writes three Parquet
    splits with the complete D-13 schema, no nulls, source-disjoint, all labels in vocab."""
    df = build_corpus.build_corpus(out_dir=tmp_path, seed=42, raw_df=synthetic_raw)

    # LIAR half-true was dropped.
    assert "half-true" not in set(df["original_label"]) or (df["original_label"] == "half-true").sum() == 0
    # Reuters dateline stripped (T-06-TM): no '(Reuters)' survives in any text.
    assert not any("(Reuters)" in t for t in df["text"])

    # Three Parquet files written.
    for s in ("train", "val", "test"):
        p = tmp_path / f"{s}.parquet"
        assert p.exists(), f"missing {s}.parquet"
        back = schema.read_parquet(p)
        assert list(back.columns) == PROVENANCE_COLUMNS, f"{s} schema mismatch"
        assert not back.isnull().any().any(), f"{s} has nulls"
        assert set(back["label"]) <= set(LABELS), f"{s} labels outside {LABELS}"

    # Source-disjoint: no group_key spans two splits (gate ran inside build_corpus already).
    by_split = {s: set(g["group_key"]) for s, g in df.groupby("split")}
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["train"] & by_split["val"])
    assert not (by_split["val"] & by_split["test"])

    # No physical rebalancing / no size cap: every surviving row landed in exactly one split.
    assert (df["split"].isin(["train", "val", "test"])).all()


_HAVE_BUILD = all((PROCESSED / f"{s}.parquet").exists() for s in ("train", "val", "test"))


@pytest.mark.skipif(not _HAVE_BUILD, reason="no on-disk build (run `python -m src.data.build_corpus`)")
def test_parquet_written():
    """data/processed/{train,val,test}.parquet are produced by the build."""
    for s in ("train", "val", "test"):
        assert (PROCESSED / f"{s}.parquet").exists()


@pytest.mark.skipif(not _HAVE_BUILD, reason="no on-disk build (run `python -m src.data.build_corpus`)")
def test_provenance_schema_on_disk():
    """On-disk Parquet carries the full D-13 provenance schema with no nulls and in-vocab labels."""
    for s in ("train", "val", "test"):
        back = schema.read_parquet(PROCESSED / f"{s}.parquet")
        assert list(back.columns) == PROVENANCE_COLUMNS
        assert not back.isnull().any().any()
        assert set(back["label"]) <= set(LABELS)
