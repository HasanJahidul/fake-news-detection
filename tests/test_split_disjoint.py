"""DATA-04 — source-disjoint grouped 70/15/15 split (D-10), implemented in 01-05.

Guarantees: no group_key in two splits, all 3 classes present per split with
non-trivial minority counts, ratios approximately 70/15/15.
"""

import pandas as pd
import pytest

from src.data.split import assert_disjoint, derive_group_key, make_splits

LABELS = ("real", "fake", "malicious")


@pytest.fixture
def split_corpus():
    """Multi-source labeled frame large enough for a 70/15/15 grouped split.

    Each class spans several distinct source groups so a source-disjoint split
    can still land all 3 classes in every split (the conftest 6-row sample is too
    small for that). Mirrors the D-13 provenance shape used downstream.
    Columns: text, label, source_dataset, original_label, language.
    """
    rows = []

    # ISOT real — multiple subjects (group via `subject`); fake — multiple subjects.
    for i in range(8):
        rows.append((f"Reuters real story number {i} about policy.", "real", "isot",
                     "true", "en", f"politicsNews_{i}"))
    for i in range(8):
        rows.append((f"Fabricated isot fake story number {i}.", "fake", "isot",
                     "fake", "en", f"left-news_{i}"))

    # BanFakeNews — multiple domains.
    for i in range(6):
        rows.append((f"বাংলা সত্য সংবাদ {i} টি প্রকাশিত হয়েছে।", "real", "banfakenews",
                     "authentic", "bn", f"domain_{i}"))
    for i in range(6):
        rows.append((f"বাংলা ভুয়া সংবাদ {i} ভাইরাল হয়েছে।", "fake", "banfakenews",
                     "fake", "bn", f"fakedomain_{i}"))

    # LIAR — multiple speakers.
    for i in range(6):
        rows.append((f"A politician statement {i} rated true.", "real", "liar",
                     "true", "en", f"speaker_{i}"))

    # SMS spam — synthetic per-row groups (malicious).
    for i in range(10):
        rows.append((f"WIN free prize {i} click now!!!", "malicious", "smsspam",
                     "spam", "en", None))

    # Phishing — synthetic / sender-domain groups (malicious).
    for i in range(10):
        rows.append((f"Verify your account {i} at http://phish.example", "malicious",
                     "phishing", "phish", "en", None))

    df = pd.DataFrame(rows, columns=[
        "text", "label", "source_dataset", "original_label", "language", "raw_group",
    ])
    return df


def test_groups_disjoint(split_corpus):
    """No source group_key appears in more than one of {train,val,test}."""
    df = make_splits(derive_group_key(split_corpus), seed=42)
    by_split = {s: set(g["group_key"]) for s, g in df.groupby("split")}
    train, val, test = by_split["train"], by_split["val"], by_split["test"]
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # assert_disjoint must agree (raises on violation)
    assert_disjoint(df)


def test_all_classes_present(split_corpus):
    """Each split contains all 3 classes with non-trivial minority counts (Pitfall 3)."""
    df = make_splits(derive_group_key(split_corpus), seed=42)
    for s in ("train", "val", "test"):
        present = set(df[df["split"] == s]["label"])
        assert present == set(LABELS), f"split {s} missing classes: {set(LABELS) - present}"


def test_split_ratios(split_corpus):
    """Split sizes are approximately 70/15/15 (tolerance band)."""
    df = make_splits(derive_group_key(split_corpus), seed=42)
    n = len(df)
    frac = {s: len(df[df["split"] == s]) / n for s in ("train", "val", "test")}
    assert 0.55 <= frac["train"] <= 0.82, frac
    assert 0.05 <= frac["val"] <= 0.27, frac
    assert 0.05 <= frac["test"] <= 0.27, frac


def test_synthetic_groups_for_sms_phishing(split_corpus):
    """SMS/phishing rows (no publisher) get unique synthetic per-row group keys."""
    df = derive_group_key(split_corpus)
    sms = df[df["source_dataset"] == "smsspam"]["group_key"]
    phish = df[df["source_dataset"] == "phishing"]["group_key"]
    # All unique (never collide) and prefixed by source.
    assert sms.is_unique
    assert phish.is_unique
    assert sms.str.startswith("smsspam_").all()
    assert phish.str.startswith("phishing_").all()


def test_assert_disjoint_raises_on_leak(split_corpus):
    """assert_disjoint raises when a group spans two splits."""
    df = make_splits(derive_group_key(split_corpus), seed=42)
    # Force a leak: copy one train group's key onto a val row.
    train_key = df[df["split"] == "train"]["group_key"].iloc[0]
    val_idx = df[df["split"] == "val"].index[0]
    df.loc[val_idx, "group_key"] = train_key
    with pytest.raises(Exception):
        assert_disjoint(df)
