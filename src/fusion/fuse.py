"""Decision-fusion layer.

Combines the text classifier's class probabilities with the three auxiliary
module risk scores into a single label + confidence, and returns a full
per-module contribution breakdown for explainability.

Convention: every auxiliary module emits `risk` in [0,1] where higher = more
likely fake/malicious (less credible). The classifier contributes
risk = 1 - P(real). Module risks are *gating* signals on real-vs-not; the
fake-vs-malicious split is decided by the classifier (nudged by phishing-style).
"""
from __future__ import annotations

from typing import Dict, Optional

from ..config import load_config


def _redistribute(weights: Dict[str, float], drop: str) -> Dict[str, float]:
    """Remove an unavailable module and renormalize remaining weights."""
    w = {k: v for k, v in weights.items() if k != drop}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def fuse(classifier_proba: Dict[str, float],
         credibility: dict,
         style: dict,
         external: dict,
         *,
         classifier_name: str = "classifier") -> dict:
    cfg = load_config()["fusion"]
    weights = dict(cfg["weights"])

    p_real = classifier_proba.get("real", 0.0)
    p_fake = classifier_proba.get("fake", 0.0)
    p_mal = classifier_proba.get("malicious", 0.0)
    classifier_risk = 1.0 - p_real

    risks = {
        "classifier": classifier_risk,
        "credibility": credibility.get("risk", 0.5),
        "style": style.get("risk", 0.0),
        "external": external.get("risk", 0.5),
    }

    # Drop modules that are explicitly unavailable so they don't dilute the signal.
    if external.get("status") in {"disabled", "no_api_key", "no_keywords", "error"}:
        weights = _redistribute(weights, "external")
        risks.pop("external", None)
    if credibility.get("verdict") == "unknown" and credibility.get("domain") is None:
        weights = _redistribute(weights, "credibility")
        risks.pop("credibility", None)

    fused_risk = sum(weights[k] * risks[k] for k in risks)

    # Label decision.
    fake_thr = cfg["fake_threshold"]
    mal_thr = cfg["malicious_threshold"]
    style_scam = sum(1 for m in style.get("matches", []) if m["category"] == "scam_phishing")

    if fused_risk < fake_thr:
        label = "real"
        confidence = 1.0 - fused_risk
    else:
        lean_malicious = (p_mal >= p_fake) or style_scam >= 2
        if lean_malicious and fused_risk >= mal_thr:
            label = "malicious"
        else:
            label = "fake"
        confidence = fused_risk

    # Sharpen confidence with classifier margin (how decisive the top class is).
    margin = abs(max(p_fake, p_mal) - p_real)
    confidence = round(min(0.99, 0.5 * confidence + 0.5 * (0.5 + 0.5 * margin)), 4)

    contributions = {
        k: {"risk": round(risks[k], 4), "weight": round(weights[k], 4),
            "weighted": round(weights[k] * risks[k], 4)}
        for k in risks
    }

    return {
        "label": label,
        "confidence": confidence,
        "fused_risk": round(fused_risk, 4),
        "classifier": {"name": classifier_name, "proba": {k: round(v, 4)
                                                          for k, v in classifier_proba.items()}},
        "contributions": contributions,
        "explanations": {
            "credibility": credibility.get("reasons", []),
            "style": style.get("reasons", []),
            "external": external.get("reasons", []),
        },
        "style_matches": style.get("matches", []),
    }
