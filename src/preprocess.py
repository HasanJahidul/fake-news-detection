"""The ONE shared text preprocessing contract (DATA-05, D-08).

Imported as ``from src.preprocess import preprocess`` by the offline corpus build
(Phase 1), classical + transformer training (Phases 2/3), and live inference
(Phase 7). Train and inference call the exact same function — no fork.

Contract (D-08, lossless / lightly normalizing only):
  * csebuetnlp/normalizer Unicode normalization (MANDATORY before BanglaBERT —
    the model was pretrained on normalized text; skipping it silently degrades
    accuracy, see CLAUDE.md "BanglaBERT without the normalizer").
  * Whitespace collapse to single spaces, then strip.
  * PRESERVES URLs, ALL-CAPS, and punctuation — Phase 4 style / malicious-content
    signals depend on these. Boilerplate / dateline stripping is offline-only
    (D-09, src/data/leakage_strip.py) and deliberately NOT done here.
  * Idempotent: preprocess(preprocess(x)) == preprocess(x).
  * None / empty / whitespace-only -> "".

Assumption A1 (resolved): the installed csebuetnlp normalizer exposes
``normalize(text, unicode_norm='NFKC', punct_replacement=None, url_replacement=None,
emoji_replacement=None, apply_unicode_norm_last=True)``. The defaults
(punct_replacement=None, url_replacement=None, no case-folding parameter) already
PRESERVE URLs, ALL-CAPS, and punctuation, so we call ``normalize(text)`` with the
defaults and add only a whitespace pass. No flag strips the preserved tokens.
"""

import re

from normalizer import normalize  # csebuetnlp/normalizer — git-installed, MANDATORY

# Collapse any run of Unicode whitespace (incl. newlines/tabs) to a single space.
_WS = re.compile(r"\s+")


def preprocess(text: str) -> str:
    """Lossless/light normalization shared by train AND inference (DATA-05, D-08).

    PRESERVES URLs, ALL-CAPS, and punctuation. Idempotent. Returns "" for
    None / empty / whitespace-only input.
    """
    if text is None:
        return ""
    # Bangla Unicode normalization (NFKC). Defaults preserve URLs/case/punct (A1).
    text = normalize(text)
    # Whitespace collapse only — keep everything else verbatim.
    text = _WS.sub(" ", text).strip()
    return text
