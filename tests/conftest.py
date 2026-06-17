"""Shared pytest fixtures for the Phase 01 data-foundation suite.

Tiny in-repo Bangla / English / code-mixed / URL / ALL-CAPS samples plus a small
labeled ``sample_corpus`` DataFrame consumed by the downstream split + schema tests.
No network, no large files — everything here is byte-literal and deterministic.
"""

import pytest

# ---------------------------------------------------------------------------
# Text fixtures (preprocess + language tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bn():
    """Pure Bangla string."""
    return "এই খবরটি সত্য নাকি মিথ্যা তা যাচাই করা প্রয়োজন।"


@pytest.fixture
def sample_en():
    """Pure English string."""
    return "The government announced a new policy today."


@pytest.fixture
def sample_codemixed():
    """Code-mixed Bangla + English string."""
    return "Govt আজ একটি new policy ঘোষণা করেছে today."


@pytest.fixture
def sample_url():
    """String containing a URL that must survive preprocessing."""
    return "Read more at http://example.com/path?q=1 about this story."


@pytest.fixture
def sample_caps():
    """ALL-CAPS + punctuation that must be preserved verbatim (Phase 4 signal)."""
    return "BREAKING!!!"


@pytest.fixture
def sample_texts(sample_bn, sample_en, sample_codemixed, sample_url, sample_caps):
    """All single-string fixtures as a list — handy for idempotence loops."""
    return [sample_bn, sample_en, sample_codemixed, sample_url, sample_caps]


# ---------------------------------------------------------------------------
# Labeled corpus fixture (schema + split tests downstream)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_corpus():
    """Tiny labeled DataFrame mirroring the D-13 provenance shape.

    Columns: text, label, source_dataset, original_label, language, group_key.
    Covers all three classes and multiple source groups so downstream
    source-disjoint split / schema tests have something realistic to assert on.
    Skips cleanly if pandas is not installed in the current environment.
    """
    pd = pytest.importorskip("pandas")
    rows = [
        # text, label, source_dataset, original_label, language, group_key
        ("The senate passed the bill today.", "real", "isot", "true", "en", "politicsNews"),
        ("Aliens secretly run the central bank, sources say.", "fake", "isot", "fake", "en", "left-news"),
        ("সরকার আজ নতুন নীতি ঘোষণা করেছে।", "real", "banfakenews", "authentic", "bn", "prothomalo"),
        ("চাঁদে গোপন শহর আবিষ্কার, দাবি ভাইরাল পোস্টের।", "fake", "banfakenews", "fake", "bn", "viralpost"),
        ("WIN a FREE prize now!!! Click http://spam.example", "malicious", "smsspam", "spam", "en", "smsspam_0"),
        ("Your account is locked, verify at http://phish.example", "malicious", "phishing", "phish", "en", "phish_0"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "text",
            "label",
            "source_dataset",
            "original_label",
            "language",
            "group_key",
        ],
    )
