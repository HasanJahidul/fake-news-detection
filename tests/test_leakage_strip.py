"""DATA-03 — offline source-leakage stripping + language tagging (plan 01-04).

Per-source boilerplate stripping (the make-or-break ISOT Reuters dateline etc.),
offline-only (D-09 — must NOT live inside preprocess()). Plus the deterministic
Bengali-ratio language tagger (D-02, D-13).
"""

import inspect
import re

import pytest

from src.data.leakage_strip import (
    strip_boilerplate,
    strip_dataframe,
    strip_isot_dateline,
)
from src.data.language import detect_language

# A synthetic ISOT-style "real" body — ISOT real is 100% Reuters-datelined.
_ISOT_REAL = (
    "WASHINGTON (Reuters) - The senate passed the bill on Tuesday, "
    "according to (Reuters) officials familiar with the matter."
)
_DATELINE_RE = re.compile(r"\(Reuters\)\s*[-–—]")


# ---------------------------------------------------------------------------
# Task 1: ISOT Reuters dateline + per-source boilerplate stripping (D-09)
# ---------------------------------------------------------------------------


def test_no_reuters_dateline():
    """No '(Reuters) -' dateline survives in ISOT real bodies after stripping."""
    out = strip_isot_dateline(_ISOT_REAL)
    assert _DATELINE_RE.search(out) is None


def test_strip_removes_leading_dateline_prefix():
    """The leading 'CITY (Reuters) -' prefix is removed; real content stays."""
    out = strip_isot_dateline(_ISOT_REAL)
    assert out.startswith("The senate passed the bill")
    assert "WASHINGTON" not in out


def test_strip_removes_residual_reuters_mentions():
    """Residual '(Reuters)' mentions inside the body are also removed."""
    out = strip_isot_dateline(_ISOT_REAL)
    assert "Reuters" not in out


def test_strip_isot_dateline_idempotent():
    """Stripping is idempotent — running it twice changes nothing further."""
    once = strip_isot_dateline(_ISOT_REAL)
    twice = strip_isot_dateline(once)
    assert once == twice


def test_strip_boilerplate_dispatches_per_source():
    """strip_boilerplate only applies the Reuters rule to isot rows."""
    isot_out = strip_boilerplate(_ISOT_REAL, "isot")
    assert _DATELINE_RE.search(isot_out) is None
    # A non-ISOT row containing the literal token is left untouched by the
    # Reuters rule (per-source dispatch, not a global strip).
    other = "A blog discussing (Reuters) - style datelines in journalism."
    other_out = strip_boilerplate(other, "banfakenews")
    assert other_out == other


def test_strip_boilerplate_handles_none_and_empty():
    """None / empty input returns empty string, never raises."""
    assert strip_boilerplate(None, "isot") == ""
    assert strip_boilerplate("", "isot") == ""


def test_strip_dataframe_per_row_and_drops_isot_date():
    """strip_dataframe applies the right rule per source_dataset row and DROPS
    the ISOT date column (year leak — Pitfall 1)."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        [
            (_ISOT_REAL, "isot", "2017-12-31"),
            ("চাঁদে গোপন শহর আবিষ্কার।", "banfakenews", None),
        ],
        columns=["text", "source_dataset", "date"],
    )
    out = strip_dataframe(df)
    # ISOT row de-leaked.
    isot_text = out.loc[out["source_dataset"] == "isot", "text"].iloc[0]
    assert _DATELINE_RE.search(isot_text) is None
    assert "Reuters" not in isot_text
    # Non-ISOT row unaffected by the Reuters rule.
    bfn_text = out.loc[out["source_dataset"] == "banfakenews", "text"].iloc[0]
    assert bfn_text == "চাঁদে গোপন শহর আবিষ্কার।"
    # The ISOT date column (year leak) is dropped from the corpus.
    assert "date" not in out.columns


def test_strip_module_not_in_preprocess():
    """D-09 boundary: preprocess() must NOT import / reference leakage_strip."""
    import src.preprocess as p

    assert "leakage_strip" not in inspect.getsource(p)


# ---------------------------------------------------------------------------
# Task 2: Bengali-ratio language tagger (D-02, D-13)
# ---------------------------------------------------------------------------


class TestLanguage:
    def test_pure_bangla_is_bn(self, sample_bn):
        assert detect_language(sample_bn) == "bn"

    def test_pure_english_is_en(self, sample_en):
        assert detect_language(sample_en) == "en"

    def test_codemixed_is_code_mixed(self, sample_codemixed):
        assert detect_language(sample_codemixed) == "code-mixed"

    def test_empty_is_unknown(self):
        assert detect_language("") == "unknown"

    def test_no_letter_is_unknown(self):
        # ASCII digits + punctuation only — neither Latin letters nor Bengali-block chars.
        assert detect_language("12345 !!! ??? --- @#$%") == "unknown"

    def test_thresholds_are_documented_constants(self):
        """A3: mix_lo / mix_hi are exposed module constants, not magic numbers."""
        import src.data.language as lang

        assert hasattr(lang, "MIX_LO")
        assert hasattr(lang, "MIX_HI")
        assert lang.MIX_LO == 0.15
        assert lang.MIX_HI == 0.85
