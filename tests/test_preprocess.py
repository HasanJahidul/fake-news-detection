"""DATA-05 / D-08 contract: the shared preprocess() entry point.

Lossless, lightly-normalizing only (csebuetnlp/normalizer + whitespace collapse).
Idempotent. Preserves URLs, ALL-CAPS, and punctuation (Phase 4 signals depend on these).
"""

from src.preprocess import preprocess


def test_idempotent(sample_texts):
    """preprocess(preprocess(x)) == preprocess(x) for every text class."""
    for x in sample_texts:
        once = preprocess(x)
        assert preprocess(once) == once


def test_preserves_url_caps_punct(sample_url, sample_caps):
    """URLs survive; ALL-CAPS + punctuation stay byte-intact."""
    assert "http" in preprocess(sample_url)
    assert preprocess("BREAKING!!!") == "BREAKING!!!"
    assert preprocess(sample_caps) == "BREAKING!!!"


def test_codemixed(sample_codemixed):
    """Code-mixed Bangla+English returns non-empty and is byte-stable across calls."""
    first = preprocess(sample_codemixed)
    second = preprocess(sample_codemixed)
    assert first != ""
    assert first == second


def test_none_and_empty():
    """None and whitespace-only collapse to the empty string."""
    assert preprocess(None) == ""
    assert preprocess("   ") == ""


def test_whitespace_collapse():
    """Internal whitespace runs collapse to single spaces; ends stripped."""
    assert preprocess("  hello   world  ") == "hello world"
    assert preprocess("line\n\nbreak\there") == "line break here"
