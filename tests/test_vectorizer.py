"""Plan 02-01 Task 1 — tests for the shared hybrid word+char TF-IDF vectorizer (D-02).

The vectorizer is a sklearn ``FeatureUnion`` of a word (1,2)-gram TfidfVectorizer and a
char_wb (3,5)-gram TfidfVectorizer over text fed strictly through the shared lossless
``preprocess()`` (D-02 / D-08). Tests assert the locked structure, the bilingual
fit_transform shape, and that ``texts_from_frame`` routes every row through preprocess().
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.preprocess import preprocess
from src.models.vectorizer import build_vectorizer, texts_from_frame


def _make_frame(rows):
    return pd.DataFrame(
        rows, columns=["text", "label", "source_dataset", "original_label", "language"]
    )


# A small bilingual fixture: Bangla + English + code-mixed rows.
_ROWS = [
    ("the government announced a new economic policy today", "real", "isot", "true", "en"),
    ("breaking shocking news you will not believe this", "fake", "isot", "fake", "en"),
    ("সরকার আজ একটি নতুন অর্থনৈতিক নীতি ঘোষণা করেছে", "real", "banfakenews", "authentic", "bn"),
    ("চাঞ্চল্যকর খবর আপনি বিশ্বাস করবেন না", "fake", "banfakenews", "fake", "bn"),
    ("click here to claim your free prize now urgent", "malicious", "smsspam", "spam", "en"),
    ("আজকের breaking news এখনই দেখুন link এখানে", "fake", "banfakenews", "fake", "bn"),
]


def test_build_vectorizer_is_featureunion_with_word_and_char():
    vec = build_vectorizer()
    assert isinstance(vec, FeatureUnion)
    names = [name for name, _ in vec.transformer_list]
    assert names == ["word", "char"]


def test_word_transformer_config():
    vec = build_vectorizer()
    word = dict(vec.transformer_list)["word"]
    assert isinstance(word, TfidfVectorizer)
    assert word.analyzer == "word"
    assert word.ngram_range == (1, 2)
    assert word.min_df == 2
    assert word.max_features == 50000
    assert word.lowercase is True


def test_char_transformer_config():
    vec = build_vectorizer()
    char = dict(vec.transformer_list)["char"]
    assert isinstance(char, TfidfVectorizer)
    assert char.analyzer == "char_wb"
    assert char.ngram_range == (3, 5)
    assert char.min_df == 2
    assert char.max_features == 50000
    assert char.lowercase is True


def test_fit_transform_bilingual_shape():
    df = _make_frame(_ROWS)
    vec = build_vectorizer()
    texts = texts_from_frame(df)
    matrix = vec.fit_transform(texts)
    # one row per input, at least one feature column from the union
    assert matrix.shape[0] == len(_ROWS)
    assert matrix.shape[1] > 0


def test_texts_from_frame_routes_through_preprocess():
    df = _make_frame(_ROWS)
    out = texts_from_frame(df)
    expected = [preprocess(t) for t in df["text"]]
    assert out == expected


def test_texts_from_frame_tolerates_none_and_empty():
    df = _make_frame(
        [
            (None, "real", "isot", "true", "en"),
            ("   ", "fake", "isot", "fake", "en"),
        ]
    )
    out = texts_from_frame(df)
    assert out == ["", ""]
