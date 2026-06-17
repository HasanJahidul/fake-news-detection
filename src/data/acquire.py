"""DATA-01 — dataset acquisition into a gitignored ``data/raw/`` tree.

Per-source download entry points plus an :func:`acquire_all` orchestrator and a
``python -m src.data.acquire [--source <name>]`` CLI.

Sources (all written under ``data/raw/<source>/``):

  * ``banfakenews``  — Kaggle ``cryptexcode/banfakenews``           (Bangla)
  * ``banfakenews2`` — Kaggle ``hrithikmajumdar/bangla-fake-news``  (Bangla, better balanced)
  * ``isot``         — Kaggle ``rahulogoel/isot-fake-news-dataset`` (English)
  * ``liar``         — raw TSVs (no deprecated ``load_dataset`` loader; Pitfall 2)
  * ``smsspam``      — Kaggle ``uciml/sms-spam-collection-dataset``; UCI id 228 fallback
  * ``phishing``     — Kaggle ``naserabdullahalam/phishing-email-dataset``

Security (threat model plan 01-02):
  * T-02-ID  — credentials come ONLY from ``python-dotenv`` (.env) or the standard
               ``~/.kaggle/kaggle.json``; never hardcoded.
  * T-02-TM  — every archive is extracted through :func:`_safe_extract`, which asserts
               each member resolves *inside* the destination (zip-slip guard) BEFORE
               writing. All downloaded bytes are treated strictly as DATA — never
               ``eval``/``exec``/rendered (the phishing corpus is malicious text).
  * Files are decoded as explicit UTF-8 with error handling.
"""

from __future__ import annotations

import argparse
import logging
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()  # T-02-ID: pull KAGGLE_USERNAME/KAGGLE_KEY from .env if present
except Exception:  # pragma: no cover - dotenv optional
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("acquire")

# Repo root = two levels up from this file (src/data/acquire.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw"

SOURCES = ("banfakenews", "banfakenews2", "isot", "liar", "smsspam", "phishing")

# LIAR original TSVs (14-col schema: label=col1, statement=col2, speaker=col4).
# Raw files — NOT the deprecated load_dataset("ucsbnlp/liar") script loader (Pitfall 2).
LIAR_BASE = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master"
LIAR_FILES = ("train.tsv", "valid.tsv", "test.tsv")


def _raw_dir(source: str) -> Path:
    """Return (and create) ``data/raw/<source>/`` — gitignored at runtime."""
    d = RAW_ROOT / source
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract ``zip_path`` into ``dest`` with a zip-slip / path-traversal guard.

    For EACH member the joined destination is resolved and asserted to remain within
    ``dest`` BEFORE any bytes are written (threat T-02-TM). Any member that escapes
    raises ``RuntimeError`` and aborts the whole extraction.
    """
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        # Validate ALL members first — fail closed before writing anything.
        for member in members:
            resolved = (dest / member).resolve()
            # commonpath containment check: resolved must live under dest.
            if os.path.commonpath([str(dest), str(resolved)]) != str(dest):
                raise RuntimeError(
                    f"zip-slip blocked: member {member!r} escapes {dest} (T-02-TM)"
                )
        for member in members:
            zf.extract(member, dest)
    log.info("extracted %d members from %s -> %s", len(members), zip_path.name, dest)


def _kaggle_download(dataset: str, dest: Path) -> bool:
    """Download a Kaggle dataset zip into ``dest`` and zip-slip-safe extract it.

    Returns True on success, False (logged) if creds/network/source unavailable so
    partial runs work. Auth comes from ``~/.kaggle/kaggle.json`` or KAGGLE_* env vars
    (T-02-ID) — never hardcoded here.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover - kaggle optional at import time
        log.warning("kaggle package unavailable (%s) — skipping %s", exc, dataset)
        return False

    try:
        api = KaggleApi()
        api.authenticate()  # reads kaggle.json / KAGGLE_USERNAME / KAGGLE_KEY
    except Exception as exc:
        log.warning("Kaggle auth failed (%s) — skipping %s", exc, dataset)
        return False

    try:
        # unzip=False so WE control extraction through the zip-slip guard.
        api.dataset_download_files(dataset, path=str(dest), unzip=False, quiet=False)
    except Exception as exc:
        log.warning("Kaggle download failed for %s (%s) — skipping", dataset, exc)
        return False

    zips = sorted(dest.glob("*.zip"))
    for z in zips:
        try:
            _safe_extract(z, dest)
            z.unlink()  # drop the archive once safely extracted
        except Exception as exc:
            log.error("extraction failed for %s (%s)", z, exc)
            return False
    log.info("acquired Kaggle dataset %s -> %s", dataset, dest)
    return True


def acquire_banfakenews() -> bool:
    """BanFakeNews v1 (Bangla) — Kaggle ``cryptexcode/banfakenews``."""
    return _kaggle_download("cryptexcode/banfakenews", _raw_dir("banfakenews"))


def acquire_banfakenews2() -> bool:
    """BanFakeNews-2.0 (Bangla, better balanced) — Kaggle ``hrithikmajumdar/bangla-fake-news``.

    Preferred over v1 per CONTEXT. Verify redistribution license on the Kaggle page.
    """
    return _kaggle_download("hrithikmajumdar/bangla-fake-news", _raw_dir("banfakenews2"))


def acquire_isot() -> bool:
    """ISOT (English, balanced) — Kaggle ``rahulogoel/isot-fake-news-dataset``."""
    return _kaggle_download("rahulogoel/isot-fake-news-dataset", _raw_dir("isot"))


def acquire_liar() -> bool:
    """LIAR (English short statements) — raw TSVs (Pitfall 2: no deprecated loader).

    Fetches ``train/valid/test.tsv`` (14-col schema) and writes them as-is. Content is
    treated strictly as DATA — decoded UTF-8, never executed.
    """
    dest = _raw_dir("liar")
    ok_any = False
    for fname in LIAR_FILES:
        url = f"{LIAR_BASE}/{fname}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read()
            # explicit UTF-8 decode w/ error handling, then re-encode for the on-disk file.
            text = raw.decode("utf-8", errors="replace")
            (dest / fname).write_text(text, encoding="utf-8")
            ok_any = True
            log.info("acquired LIAR %s (%d bytes) -> %s", fname, len(raw), dest)
        except (urllib.error.URLError, OSError) as exc:
            log.warning("LIAR fetch failed for %s (%s) — skipping", url, exc)
    return ok_any


def acquire_smsspam() -> bool:
    """SMS Spam (English) — Kaggle ``uciml/sms-spam-collection-dataset``; UCI id 228 fallback.

    Tries Kaggle first; on any failure falls back to ``ucimlrepo`` (no token needed).
    """
    dest = _raw_dir("smsspam")
    if _kaggle_download("uciml/sms-spam-collection-dataset", dest):
        return True

    log.info("SMS Spam: Kaggle path unavailable — trying UCI ucimlrepo id 228 fallback")
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=228)  # SMS Spam Collection
        df = ds.data.original if ds.data.original is not None else ds.data.features
        # treat as DATA only; explicit UTF-8 csv write
        df.to_csv(dest / "sms_spam_uci.csv", index=False, encoding="utf-8")
        log.info("acquired SMS Spam via UCI ucimlrepo (%d rows) -> %s", len(df), dest)
        return True
    except Exception as exc:
        log.warning("SMS Spam UCI fallback failed (%s) — skipping", exc)
        return False


def acquire_phishing() -> bool:
    """Phishing emails (English, intentionally-malicious text) — Kaggle.

    ``naserabdullahalam/phishing-email-dataset``. Content is DATA only: decoded UTF-8,
    never eval/exec/rendered (T-02-TM). License recorded in SUMMARY.
    """
    return _kaggle_download("naserabdullahalam/phishing-email-dataset", _raw_dir("phishing"))


_ACQUIRERS = {
    "banfakenews": acquire_banfakenews,
    "banfakenews2": acquire_banfakenews2,
    "isot": acquire_isot,
    "liar": acquire_liar,
    "smsspam": acquire_smsspam,
    "phishing": acquire_phishing,
}


def acquire_all() -> dict[str, bool]:
    """Run every per-source acquirer. Returns a ``{source: success}`` map.

    Each source skips+logs gracefully on missing creds/network so partial runs work.
    """
    results: dict[str, bool] = {}
    for name, fn in _ACQUIRERS.items():
        log.info("=== acquiring %s ===", name)
        try:
            results[name] = bool(fn())
        except Exception as exc:  # defensive: one bad source never aborts the rest
            log.error("acquire %s raised (%s) — marking failed", name, exc)
            results[name] = False
    ok = [k for k, v in results.items() if v]
    skipped = [k for k, v in results.items() if not v]
    log.info("acquire_all done — ok=%s skipped/failed=%s", ok, skipped)
    return results


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download source datasets into data/raw/")
    parser.add_argument(
        "--source",
        choices=SOURCES,
        help="download a single source (default: all)",
    )
    args = parser.parse_args(argv)

    if args.source:
        ok = _ACQUIRERS[args.source]()
        return 0 if ok else 1

    results = acquire_all()
    # exit 0 if at least one source succeeded (partial runs are valid)
    return 0 if any(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
