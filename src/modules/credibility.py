"""Source credibility module.

Given a URL (or bare domain), returns a risk score in [0,1] where higher = less
credible, plus a human-readable rationale for explainability. Pure heuristic +
curated lists — no network calls, so it is fast and offline-safe.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from ..config import REPO_ROOT, load_config

_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _load_lists() -> dict:
    rel = load_config()["modules"]["credibility"]["domains_file"]
    with open(REPO_ROOT / rel, "r", encoding="utf-8") as f:
        return json.load(f)


_LISTS = _load_lists()


def _domain(url_or_domain: str) -> str:
    s = url_or_domain.strip()
    if "://" not in s:
        s = "http://" + s
    host = (urlparse(s).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def score(url: Optional[str]) -> dict:
    """Return {'risk': float, 'verdict': str, 'domain': str, 'reasons': [...]}."""
    if not url or not url.strip():
        return {"risk": 0.5, "verdict": "unknown", "domain": None,
                "reasons": ["No URL provided; source credibility unavailable."]}

    domain = _domain(url)
    reasons, risk = [], 0.5  # neutral prior for unknown domains

    if not domain:
        return {"risk": 0.5, "verdict": "unknown", "domain": None,
                "reasons": ["Could not parse a domain from the input."]}

    if any(domain == d or domain.endswith("." + d) for d in _LISTS["trusted"]):
        risk, verdict = 0.1, "trusted"
        reasons.append(f"'{domain}' is on the curated trusted-source list.")
        return {"risk": risk, "verdict": verdict, "domain": domain, "reasons": reasons}

    if any(domain == d or domain.endswith("." + d) for d in _LISTS["untrusted"]):
        risk, verdict = 0.9, "untrusted"
        reasons.append(f"'{domain}' is on the curated untrusted/satire list.")
        return {"risk": risk, "verdict": verdict, "domain": domain, "reasons": reasons}

    # Heuristics for unknown domains.
    if _IP_RE.match(domain):
        risk += 0.3
        reasons.append("URL uses a raw IP address instead of a domain name.")
    for tld in _LISTS.get("suspicious_tlds", []):
        if domain.endswith(tld):
            risk += 0.2
            reasons.append(f"Domain uses a frequently-abused TLD '{tld}'.")
            break
    if domain.count("-") >= 3:
        risk += 0.15
        reasons.append("Domain contains many hyphens (common in lookalike/phishing).")
    if len(domain) > 30:
        risk += 0.1
        reasons.append("Unusually long domain name.")
    if any(tok in domain for tok in ("secure", "login", "verify", "update", "account")):
        risk += 0.15
        reasons.append("Domain mimics security/login wording (phishing pattern).")

    risk = max(0.0, min(1.0, risk))
    if not reasons:
        reasons.append(f"'{domain}' is unknown; using neutral credibility prior.")
    verdict = "low" if risk >= 0.65 else ("high" if risk <= 0.35 else "neutral")
    return {"risk": risk, "verdict": verdict, "domain": domain, "reasons": reasons}
