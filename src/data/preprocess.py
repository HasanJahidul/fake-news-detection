"""Text cleaning, normalization, language detection, tokenization.

Shared by the data pipeline (corpus build) and inference (pipeline/app) so that
training-time and serving-time text go through exactly the same transform.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

import regex  # better Unicode property support than stdlib re

# Bangla Unicode block: U+0980–U+09FF
_BENGALI = regex.compile(r"[ঀ-৿]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_MULTISPACE = re.compile(r"\s+")
# Keep letters (any script), digits, and basic punctuation that carries style signal.
_KEEP = regex.compile(r"[^\p{L}\p{N}\s!?.$%@#'\"-]")

# Optional Bangla normalizer (bnlp). Imported lazily; falls back to NFC.
try:  # pragma: no cover - optional dep
    from bnlp import CleanText  # type: ignore

    _bn_cleaner = CleanText(
        fix_unicode=True,
        unicode_norm=True,
        unicode_norm_form="NFKC",
        remove_url=False,
        remove_email=False,
        remove_emoji=False,
        remove_number=False,
        remove_digits=False,
        remove_punct=False,
    )
except Exception:  # noqa: BLE001
    _bn_cleaner = None


def detect_lang(text: str) -> str:
    """Crude but fast: 'bn' if >=15% of letters are Bengali, else 'en'."""
    letters = regex.findall(r"\p{L}", text)
    if not letters:
        return "en"
    bn = sum(1 for ch in letters if _BENGALI.match(ch))
    return "bn" if bn / len(letters) >= 0.15 else "en"


def normalize_unicode(text: str, lang: str) -> str:
    if lang == "bn" and _bn_cleaner is not None:
        try:
            return _bn_cleaner(text)
        except Exception:  # noqa: BLE001
            pass
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str, *, max_chars: int = 5000, drop_urls: bool = True) -> str:
    """Full cleaning used for model input. Lowercasing left to the vectorizer/tokenizer."""
    if not isinstance(text, str):
        return ""
    lang = detect_lang(text)
    text = normalize_unicode(text, lang)
    text = _HTML.sub(" ", text)
    if drop_urls:
        text = _URL.sub(" ", text)
    text = _KEEP.sub(" ", text)
    text = _MULTISPACE.sub(" ", text).strip()
    return text[:max_chars]


def simple_tokens(text: str) -> List[str]:
    """Whitespace + punctuation tokenizer (Unicode aware). For style/feature stats."""
    return regex.findall(r"\p{L}+|\p{N}+|[!?.]", text.lower())
