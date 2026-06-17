"""Download / assemble raw source datasets into a standardized schema.

Output: one CSV per source under data/raw/, each with columns:
    text   : str   raw document/message text
    label  : str   one of real | fake | malicious
    lang   : str   en | bn  (auto-detected if absent)
    source : str   dataset id (provenance)

Datasets are pulled from the HuggingFace Hub where licensing allows. Network or
hub failures are non-fatal: each source is independent and a synthetic fallback
keeps the rest of the pipeline (and the demo UI) runnable fully offline.

Run:  python -m src.data.download           # all sources
      python -m src.data.download --synthetic-only
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from ..config import path as cfg_path
from .preprocess import detect_lang

RAW_DIR = cfg_path("raw_dir")

STD_COLS = ["text", "label", "lang", "source"]


def _standardize(df: pd.DataFrame, label: str, source: str,
                 text_col: str, lang: Optional[str] = None) -> pd.DataFrame:
    out = pd.DataFrame({"text": df[text_col].astype(str)})
    out["label"] = label
    out["source"] = source
    out["lang"] = lang if lang else out["text"].map(detect_lang)
    out = out[out["text"].str.strip().str.len() > 0]
    return out[STD_COLS].reset_index(drop=True)


def _try_hf(loader: Callable[[], pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
    try:
        df = loader()
        print(f"  [ok] {name}: {len(df)} rows")
        return df
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] {name}: {type(e).__name__}: {e}")
        return None


# ── English fake / real news ────────────────────────────────────────────────
def load_english_fakenews() -> List[pd.DataFrame]:
    from datasets import load_dataset

    frames: List[pd.DataFrame] = []
    # GonzaloA/fake_news: columns 'text', 'label' (0=fake,1=real per card)
    def _gonzalo() -> pd.DataFrame:
        ds = load_dataset("GonzaloA/fake_news", split="train")
        df = ds.to_pandas()
        df["label"] = df["label"].map({0: "fake", 1: "real"})
        return df.dropna(subset=["text", "label"])

    d = _try_hf(_gonzalo, "GonzaloA/fake_news (en)")
    if d is not None:
        for lab in ("real", "fake"):
            sub = d[d["label"] == lab]
            frames.append(_standardize(sub, lab, "gonzaloa_fake_news", "text", "en"))
    return frames


# ── Bangla fake / real news (BanFakeNews) ───────────────────────────────────
def load_bangla_fakenews() -> List[pd.DataFrame]:
    from datasets import load_dataset

    frames: List[pd.DataFrame] = []

    def _banfake() -> pd.DataFrame:
        ds = load_dataset("Karim-Ashraf/BanFakeNews", split="train")
        df = ds.to_pandas()
        # label: 1 = authentic/real, 0 = fake (per dataset card variants)
        col = "label" if "label" in df.columns else "F-type"
        txt = "content" if "content" in df.columns else (
            "headline" if "headline" in df.columns else df.columns[0])
        df = df.rename(columns={txt: "text"})
        df["label"] = df[col].map({1: "real", 0: "fake"}).fillna("real")
        return df.dropna(subset=["text"])

    d = _try_hf(_banfake, "BanFakeNews (bn)")
    if d is not None:
        for lab in ("real", "fake"):
            sub = d[d["label"] == lab]
            if len(sub):
                frames.append(_standardize(sub, lab, "banfakenews", "text", "bn"))
    return frames


# ── Malicious content (phishing + spam) ─────────────────────────────────────
def load_malicious() -> List[pd.DataFrame]:
    from datasets import load_dataset

    frames: List[pd.DataFrame] = []

    def _sms_spam() -> pd.DataFrame:
        ds = load_dataset("sms_spam", split="train")
        df = ds.to_pandas()
        df = df[df["label"] == 1]  # 1 = spam
        return df.rename(columns={"sms": "text"})

    d = _try_hf(_sms_spam, "sms_spam (malicious)")
    if d is not None:
        frames.append(_standardize(d, "malicious", "sms_spam", "text", "en"))

    def _phishing() -> pd.DataFrame:
        ds = load_dataset("ealvaradob/phishing-dataset", "texts",
                          split="train", trust_remote_code=True)
        df = ds.to_pandas()
        df = df[df["label"] == 1]  # 1 = phishing
        return df
    d = _try_hf(_phishing, "ealvaradob/phishing-dataset (malicious)")
    if d is not None:
        frames.append(_standardize(d, "malicious", "phishing_dataset", "text", "en"))
    return frames


# ── Synthetic fallback (offline-safe demo data) ─────────────────────────────
_SYNTH = {
    "real": [
        "The central bank announced a 0.25 percent cut in the benchmark interest rate after its quarterly meeting.",
        "Researchers published a peer-reviewed study on coastal erosion trends over the past decade.",
        "The national cricket team won the series after a closely contested final match on Sunday.",
        "Officials confirmed that the new metro line will open to the public next month after safety checks.",
        "সরকার আগামী অর্থবছরের জন্য নতুন বাজেট ঘোষণা করেছে যেখানে শিক্ষা খাতে বরাদ্দ বৃদ্ধি করা হয়েছে।",
        "আবহাওয়া অধিদপ্তর জানিয়েছে আগামী দুই দিন সারাদেশে হালকা থেকে মাঝারি বৃষ্টিপাত হতে পারে।",
    ],
    "fake": [
        "BREAKING: All banks will be permanently closed tomorrow, government secretly confirms!!!",
        "SHOCKING miracle cure discovered that doctors do not want you to know about, share now!",
        "Scientists reveal the moon will disappear next week causing worldwide blackout.",
        "Famous actor secretly replaced by a robot, leaked documents prove the conspiracy.",
        "চাঞ্চল্যকর খবর! আগামীকাল থেকে সব মোবাইল নেটওয়ার্ক চিরতরে বন্ধ হয়ে যাবে, দ্রুত শেয়ার করুন!",
        "অবিশ্বাস্য! এই পাতা খেলে এক রাতেই সব রোগ ভালো হয়ে যায়, ডাক্তাররা লুকিয়ে রাখছে।",
    ],
    "malicious": [
        "Your account will be blocked. Verify your details now at http://secure-login-update.example to avoid suspension.",
        "Congratulations! You have won a lottery of $1,000,000. Send your bank details to claim the prize immediately.",
        "URGENT: Your package is held. Pay a small customs fee here to release it: http://parcel-pay.example",
        "Invest today and earn double profit within one week, guaranteed returns, limited slots, act fast!",
        "Download this file to check your exam result: http://result-check.example/result.exe",
        "Dear customer, your card is temporarily locked. Click http://card-reactivate.example and enter your PIN.",
    ],
}


# Filler clauses to synthesize unique variants (keeps dedup from collapsing the toy set).
_EN_CTX = [
    "according to reports", "officials said today", "as shared on social media",
    "in a recent update", "many users claim", "sources mention", "it is reported",
    "people are saying", "the post reads", "circulating online",
]
_BN_CTX = [
    "প্রতিবেদনে বলা হয়েছে", "সূত্র জানিয়েছে", "অনলাইনে ছড়িয়ে পড়েছে",
    "অনেকে দাবি করছেন", "সম্প্রতি জানা গেছে", "পোস্টে বলা হয়",
]


def synthetic(variants_per_text: int = 22) -> List[pd.DataFrame]:
    from .preprocess import detect_lang as _dl
    frames = []
    for lab, texts in _SYNTH.items():
        rows = []
        for base in texts:
            ctxs = _BN_CTX if _dl(base) == "bn" else _EN_CTX
            for i in range(variants_per_text):
                ctx = ctxs[i % len(ctxs)]
                rows.append(f"{ctx}: {base}" if i else base)
        df = pd.DataFrame({"text": rows})
        frames.append(_standardize(df, lab, "synthetic", "text"))
    print(f"  [ok] synthetic: {sum(len(f) for f in frames)} rows")
    return frames


def main(synthetic_only: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    if not synthetic_only:
        print("Loading sources from HuggingFace Hub ...")
        frames += load_english_fakenews()
        frames += load_bangla_fakenews()
        frames += load_malicious()

    have_labels = {l for f in frames for l in f["label"].unique()}
    missing = set(["real", "fake", "malicious"]) - have_labels
    if synthetic_only or missing:
        if missing:
            print(f"Missing labels {missing} from hub — adding synthetic fallback.")
        frames += synthetic()

    combined = pd.concat(frames, ignore_index=True)
    for source, grp in combined.groupby("source"):
        out = RAW_DIR / f"{source}.csv"
        grp.to_csv(out, index=False)
        print(f"  wrote {out}  ({len(grp)} rows)")
    print(f"Done. Total raw rows: {len(combined)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-only", action="store_true",
                    help="Skip hub downloads; generate offline toy corpus only.")
    args = ap.parse_args()
    main(synthetic_only=args.synthetic_only)
