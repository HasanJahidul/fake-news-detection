"""Writing-style & behavioral-pattern module.

Rule-based detector for clickbait / sensational / phishing language plus surface
signals (ALL-CAPS, exclamation spam, link count). Returns a risk score in [0,1]
and the matched spans so the UI can highlight them (explainability).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from ..config import REPO_ROOT, load_config

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EXCLAIM_RE = re.compile(r"!{2,}")
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{4,}\b")


def _load_lexicon() -> dict:
    rel = load_config()["modules"]["style"]["lexicon_file"]
    with open(REPO_ROOT / rel, "r", encoding="utf-8") as f:
        return json.load(f)


_LEX = _load_lexicon()
# Pre-compile phrase patterns once (word-boundary, case-insensitive).
_PATTERNS: Dict[str, List[re.Pattern]] = {
    cat: [re.compile(re.escape(term), re.IGNORECASE) for term in terms]
    for cat, terms in _LEX.items()
}
# Weight per category toward overall suspicion.
_CAT_WEIGHT = {"clickbait": 0.25, "sensational": 0.20, "scam_phishing": 0.40,
               "clickbait_bn": 0.25}


def analyze(text: str) -> dict:
    """Return risk + matched spans + feature breakdown for explainability."""
    if not text:
        return {"risk": 0.0, "matches": [], "features": {}, "reasons": []}

    matches: List[dict] = []
    cat_hits: Dict[str, int] = {}
    for cat, pats in _PATTERNS.items():
        for pat in pats:
            for m in pat.finditer(text):
                matches.append({"category": cat, "term": m.group(0),
                                "start": m.start(), "end": m.end()})
                cat_hits[cat] = cat_hits.get(cat, 0) + 1

    n_words = max(1, len(text.split()))
    caps = _CAPS_WORD_RE.findall(text)
    exclaim = _EXCLAIM_RE.findall(text)
    links = _URL_RE.findall(text)
    caps_ratio = len(caps) / n_words

    features = {
        "lexicon_hits": int(sum(cat_hits.values())),
        "caps_words": len(caps),
        "caps_ratio": round(caps_ratio, 3),
        "exclaim_runs": len(exclaim),
        "links": len(links),
    }

    # Aggregate risk: lexicon categories (saturating) + surface signals.
    risk = 0.0
    reasons: List[str] = []
    for cat, n in cat_hits.items():
        contrib = _CAT_WEIGHT.get(cat, 0.2) * (1 - 0.6 ** n)  # diminishing returns
        risk += contrib
        if n:
            reasons.append(f"{n} {cat.replace('_', '/')} phrase(s).")
    if caps_ratio > 0.08:
        risk += 0.15
        reasons.append(f"High ALL-CAPS ratio ({caps_ratio:.0%}).")
    if exclaim:
        risk += 0.10
        reasons.append(f"Exclamation spam ({len(exclaim)} run(s) of '!!').")
    if len(links) >= 2:
        risk += 0.10
        reasons.append(f"Multiple links ({len(links)}).")

    risk = max(0.0, min(1.0, risk))
    if not reasons:
        reasons.append("No notable clickbait/sensational/phishing style signals.")
    return {"risk": risk, "matches": matches, "features": features, "reasons": reasons}
