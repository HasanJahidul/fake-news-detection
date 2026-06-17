"""DATA-03 stub — implemented in plan 01-04.

Per-source boilerplate stripping (Reuters dateline etc.), offline-only (D-09).
Language tagging (D-02) is also turned on by 01-04.
"""

import pytest


@pytest.mark.skip(reason="implemented in plan 01-04 (leakage_strip.py)")
def test_no_reuters_dateline():
    """No '(Reuters) -' dateline survives in ISOT real bodies after stripping."""
    raise NotImplementedError("01-04: assert re.search(r'\\(Reuters\\)\\s*[-–—]', body) is None")
