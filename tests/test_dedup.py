"""DATA-04 / D-11 stub — implemented in plan 01-05.

Exact + fuzzy near-dup clustering; survivors retain dedup_cluster_id.
"""

import pytest


@pytest.mark.skip(reason="implemented in plan 01-05 (dedup.py MinHashLSH)")
def test_near_dup_removed():
    """A near-duplicate row is collapsed; the survivor keeps a cluster_id."""
    raise NotImplementedError("01-05: MinHashLSH near-dup removal + cluster_id retention")
