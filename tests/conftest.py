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


# ---------------------------------------------------------------------------
# Phase 03 — transformer suite helpers + tiny GPU-free fixture model
# ---------------------------------------------------------------------------
#
# NOTE: torch / transformers are imported LAZILY *inside* the fixture body, never
# at this module's top level, so ``import tests.conftest`` (and therefore the whole
# fast suite collection) works even when the transformer stack is not installed.
# The transformer test files themselves guard with ``pytest.importorskip("torch")``.


def _make_lang_frame(rows):
    """Build a transformer-suite DataFrame including the ``language`` column.

    Mirrors ``tests/test_train_classical._make_frame`` but is the shared helper for
    the Phase-3 transformer scaffolds, whose per-language macro-F1 + Bangla-priority
    selection assertions need the ``language`` provenance column. Columns:
    ``text, label, source_dataset, original_label, language``.
    """
    import pandas as pd  # local import keeps conftest import light

    return pd.DataFrame(
        rows,
        columns=["text", "label", "source_dataset", "original_label", "language"],
    )


@pytest.fixture(scope="session")
def tiny_seqcls_model(tmp_path_factory):
    """A 2-label toy ``AutoModelForSequenceClassification`` saved to a temp dir.

    Built locally via ``BertConfig`` + ``from_config`` (RANDOM weights — no remote
    download, T-03-02), with a minimal matching ``BertTokenizerFast`` written from a
    tiny temp vocab. Both model and tokenizer are ``save_pretrained`` into the same
    temp directory, whose ``Path`` is returned. Consumers ``from_pretrained(dir)`` and
    run a CPU forward pass (2-logit output) — no GPU, no network.

    All torch/transformers imports happen here (lazily) so conftest itself imports
    without the transformer stack. Skips the dependent test if torch is absent.
    """
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import (
        AutoModelForSequenceClassification,
        BertConfig,
        BertTokenizerFast,
    )

    out_dir = tmp_path_factory.mktemp("tiny_seqcls")

    # Minimal WordPiece vocab: required specials + a handful of usable tokens so the
    # tokenizer can encode the toy text. vocab_size MUST match the config below.
    vocab_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    ]
    # Pad out to a fixed small vocab_size with deterministic dummy tokens.
    vocab_size = 64
    vocab_tokens += [f"tok{i}" for i in range(vocab_size - len(vocab_tokens))]
    assert len(vocab_tokens) == vocab_size

    vocab_file = out_dir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))

    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=2,
    )
    model = AutoModelForSequenceClassification.from_config(cfg)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    return out_dir
