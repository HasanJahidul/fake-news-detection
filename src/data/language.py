"""DATA-03 / D-02 — deterministic bn / en / code-mixed language tagger.

Fills the D-13 ``language`` provenance field so later phases can report per-language
macro-F1. This is the ONE part of the pipeline worth hand-rolling: the bn-vs-en
distinction is a deterministic Unicode-block question, not a probabilistic one, and an
off-the-shelf model (fasttext / lingua) is actually *less* accurate on short code-mixed
Bangla text (RESEARCH "Don't Hand-Roll" + Assumption A3; modelpredict.com survey,
arXiv 2407.09187). No ML model, no extra dependency.

Method: count Bengali-block letters (U+0980–U+09FF) vs Latin letters, compute the
Bengali ratio = bengali / (bengali + latin), and bucket it against two thresholds:

    ratio >= MIX_HI  -> "bn"
    ratio <= MIX_LO  -> "en"
    otherwise        -> "code-mixed"
    no letters       -> "unknown"

A3: the thresholds are tunable heuristics; the chosen defaults (MIX_LO=0.15,
MIX_HI=0.85) are documented module constants below.
"""

import re

# Bengali Unicode block U+0980–U+09FF (matches the RESEARCH char class [ঀ-৿]).
_BENGALI = re.compile(r"[ঀ-৿]")
_LATIN = re.compile(r"[A-Za-z]")

# A3 — documented, tunable thresholds on the Bengali letter ratio.
MIX_LO = 0.15  # ratio <= MIX_LO  -> predominantly English
MIX_HI = 0.85  # ratio >= MIX_HI  -> predominantly Bangla


def detect_language(text: str, mix_lo: float = MIX_LO, mix_hi: float = MIX_HI) -> str:
    """Tag ``text`` as ``"bn" | "en" | "code-mixed" | "unknown"`` by Bengali ratio.

    Deterministic; returns ``"unknown"`` when the string contains no Bengali-block or
    Latin letters (empty, digits/punctuation only, etc.).
    """
    if not text:
        return "unknown"
    bn = len(_BENGALI.findall(text))
    en = len(_LATIN.findall(text))
    total = bn + en
    if total == 0:
        return "unknown"
    ratio = bn / total  # fraction of letters that are Bengali
    if ratio >= mix_hi:
        return "bn"
    if ratio <= mix_lo:
        return "en"
    return "code-mixed"
