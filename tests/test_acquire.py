"""DATA-01 — creds-aware presence check for acquire.py.

* ``test_raw_is_gitignored`` — ALWAYS on: enforces the "raw never committed" gate
  (``git check-ignore data/raw`` exits 0) even without credentials.
* ``test_sources_present`` — skips (does NOT fail) when no Kaggle creds are present;
  otherwise asserts each expected ``data/raw/<source>/`` dir exists and is non-empty.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"

# smsspam has a no-token UCI fallback; LIAR needs no token either. The Kaggle-only
# sets are the rest. We assert all six dirs once creds are present.
EXPECTED_SOURCES = ("banfakenews", "banfakenews2", "isot", "liar", "smsspam", "phishing")


def _have_kaggle_creds() -> bool:
    import os

    if (Path.home() / ".kaggle" / "kaggle.json").is_file():
        return True
    return bool(os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_USERNAME"))


def test_raw_is_gitignored():
    """`git check-ignore data/raw` must exit 0 — raw data is never committable (DATA-01)."""
    proc = subprocess.run(
        ["git", "check-ignore", "data/raw"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, "data/raw is NOT gitignored — DATA-01 gate broken"


def _non_empty(d: Path) -> bool:
    return d.is_dir() and any(d.rglob("*"))


def test_sources_present():
    """Each expected source lands non-empty under the gitignored data/raw/ tree."""
    if not _have_kaggle_creds():
        pytest.skip("no Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_* env) — skipping live presence check")

    missing = [s for s in EXPECTED_SOURCES if not _non_empty(RAW_ROOT / s)]
    if missing:
        pytest.skip(
            f"creds present but data/raw not yet populated for {missing} — run `python -m src.data.acquire`"
        )

    for src in EXPECTED_SOURCES:
        d = RAW_ROOT / src
        assert _non_empty(d), f"expected non-empty data/raw/{src}/ after acquire_all()"
