"""External verification module.

Extracts salient keywords from the input and queries NewsAPI for coverage by
trusted outlets. Corroboration by reputable sources lowers risk; absence is
treated as neutral (not proof of falsity). Network/key failures degrade
gracefully to 'unavailable' so the pipeline never blocks.

Needs an API key in the env var named by config.modules.external_verification.newsapi_key_env.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from functools import lru_cache
from typing import List, Optional

from ..config import load_config

_STOP = set("""a an the of and or but if then so to in on at for with from by as is are was were be been being this that these those it its they them their he she his her you your we our i not no yes do does did has have had will would can could should may might just about over under more most very into out up down new news said say says report reports""".split())

_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def _keywords(text: str, k: int = 6) -> List[str]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    words = [w for w in words if w not in _STOP]
    common = [w for w, _ in Counter(words).most_common(k * 2)]
    # Prefer capitalized proper-noun-ish tokens from the original text first.
    proper = [m.group(0) for m in re.finditer(r"\b[A-Z][a-z]{3,}\b", text)]
    seen, out = set(), []
    for w in proper + common:
        wl = w.lower()
        if wl not in seen and wl not in _STOP:
            seen.add(wl)
            out.append(w)
        if len(out) >= k:
            break
    return out


def _enabled() -> bool:
    return bool(load_config()["modules"]["external_verification"]["enabled"])


def _api_key() -> Optional[str]:
    env = load_config()["modules"]["external_verification"]["newsapi_key_env"]
    return os.environ.get(env)


@lru_cache(maxsize=256)
def _query_newsapi(query: str, key: str) -> int:
    """Return number of articles from trusted sources matching the query (cached)."""
    import requests

    cfg = load_config()["modules"]["external_verification"]
    params = {
        "q": query,
        "sources": ",".join(cfg["trusted_sources"]),
        "pageSize": cfg["max_articles"],
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": key,
    }
    r = requests.get("https://newsapi.org/v2/everything", params=params,
                     timeout=cfg["timeout_seconds"])
    r.raise_for_status()
    return int(r.json().get("totalResults", 0))


def verify(text: str) -> dict:
    """Return {'risk', 'status', 'matches', 'query', 'reasons'}."""
    if not _enabled():
        return {"risk": 0.5, "status": "disabled", "matches": 0, "query": None,
                "reasons": ["External verification disabled in config."]}
    key = _api_key()
    if not key:
        return {"risk": 0.5, "status": "no_api_key", "matches": 0, "query": None,
                "reasons": ["No NewsAPI key set; external verification skipped."]}

    kws = _keywords(text)
    query = " ".join(kws[:4])
    if not query:
        return {"risk": 0.5, "status": "no_keywords", "matches": 0, "query": None,
                "reasons": ["Could not extract keywords to verify."]}
    try:
        n = _query_newsapi(query, key)
    except Exception as e:  # noqa: BLE001
        return {"risk": 0.5, "status": "error", "matches": 0, "query": query,
                "reasons": [f"Verification request failed: {type(e).__name__}."]}

    # More corroboration from trusted outlets -> lower risk. Saturates at ~5 hits.
    risk = max(0.1, 1.0 - min(n, 5) / 5.0 * 0.8)
    if n == 0:
        reasons = [f"No trusted-source coverage found for: '{query}'."]
    else:
        reasons = [f"{n} trusted-source article(s) match: '{query}'."]
    return {"risk": risk, "status": "ok", "matches": n, "query": query, "reasons": reasons}
