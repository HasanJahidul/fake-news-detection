"""End-to-end inference pipeline: text or URL -> full prediction object.

Orchestrates: (optional URL fetch) -> preprocess -> classifier (transformer if
available, else classical) -> credibility + style + external modules -> fusion.

CLI:
    python -m src.pipeline --text "BREAKING: shocking miracle cure ..."
    python -m src.pipeline --url  "https://example.com/some-article"
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from typing import Optional

from .config import LABELS
from .data.preprocess import clean_text, detect_lang
from .fusion.fuse import fuse
from .modules import credibility, style, verify


# ── Lazy singletons so the app loads models once ────────────────────────────
@lru_cache(maxsize=1)
def _classifier():
    """Prefer the fine-tuned transformer; fall back to the classical model."""
    try:
        from .models.transformer import TransformerClassifier
        if TransformerClassifier.available():
            return ("xlm-roberta", TransformerClassifier())
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] transformer unavailable ({type(e).__name__}); using classical.")
    from .models.classical import ClassicalClassifier
    clf = ClassicalClassifier()
    return (clf.name, clf)


def fetch_url(url: str) -> str:
    """Extract main article text from a URL (returns '' on failure)."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False,
                                       include_tables=False)
            return text or ""
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] URL fetch failed: {type(e).__name__}: {e}")
    return ""


def analyze(text: Optional[str] = None, url: Optional[str] = None) -> dict:
    """Run the full pipeline. Provide `text`, `url`, or both."""
    raw = text or ""
    if url and not raw:
        raw = fetch_url(url)
    if not raw.strip():
        return {"error": "No text to analyze (empty input or URL extraction failed)."}

    cleaned = clean_text(raw)
    lang = detect_lang(cleaned)

    name, clf = _classifier()
    proba = clf.predict_proba(cleaned)

    cred = credibility.score(url)
    sty = style.analyze(raw)               # style on RAW text (caps/!! preserved)
    ext = verify.verify(cleaned)

    result = fuse(proba, cred, sty, ext, classifier_name=name)
    result.update({
        "input": {"chars": len(raw), "lang": lang, "url": url,
                  "preview": raw[:240]},
        "source_credibility": cred,
        "style": {k: sty[k] for k in ("risk", "features", "matches")},
        "external": ext,
    })
    return result


def _print(result: dict) -> None:
    if "error" in result:
        print(result["error"])
        return
    print(f"\n=== Prediction ===")
    print(f"Label       : {result['label'].upper()}")
    print(f"Confidence  : {result['confidence']:.2%}")
    print(f"Fused risk  : {result['fused_risk']:.3f}")
    print(f"Classifier  : {result['classifier']['name']}  {result['classifier']['proba']}")
    print(f"Language     : {result['input']['lang']}")
    print("\n--- Module contributions ---")
    for mod, c in result["contributions"].items():
        print(f"  {mod:12s} risk={c['risk']:.3f}  w={c['weight']:.2f}  -> {c['weighted']:.3f}")
    print("\n--- Why ---")
    for mod, reasons in result["explanations"].items():
        for r in reasons:
            print(f"  [{mod}] {r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fake-news / malicious-content detector")
    ap.add_argument("--text", type=str, help="Raw text to analyze")
    ap.add_argument("--url", type=str, help="URL of an article to analyze")
    ap.add_argument("--json", action="store_true", help="Print full JSON result")
    args = ap.parse_args()
    if not args.text and not args.url:
        ap.error("provide --text or --url")
    res = analyze(text=args.text, url=args.url)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        _print(res)
