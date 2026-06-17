"""DATA-01 stub — implemented in plan 01-02.

Source datasets fetch from Kaggle/HF/UCI into gitignored data/raw/.
Skips when credentials / downloaded data are absent (no network in CI).
"""

import pytest


@pytest.mark.skip(reason="implemented in plan 01-02 (acquire.py — skips without Kaggle creds)")
def test_sources_present():
    """Each expected source lands under the gitignored data/raw/ tree."""
    raise NotImplementedError("01-02: assert raw source files present (skip if creds absent)")
