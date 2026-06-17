"""DATA-04 stubs — implemented in plan 01-05.

Source-disjoint grouped 70/15/15 split (D-10): no group in two splits,
all 3 classes present per split, ratios approximately 70/15/15.
"""

import pytest


@pytest.mark.skip(reason="implemented in plan 01-05 (split.py GroupShuffleSplit)")
def test_groups_disjoint():
    """No source group_key appears in more than one split."""
    raise NotImplementedError("01-05: assert train/val/test group sets are pairwise disjoint")


@pytest.mark.skip(reason="implemented in plan 01-05 (split.py)")
def test_all_classes_present():
    """Each split contains all 3 classes with non-trivial minority counts."""
    raise NotImplementedError("01-05: assert {real,fake,malicious} in every split")


@pytest.mark.skip(reason="implemented in plan 01-05 (split.py)")
def test_split_ratios():
    """Split sizes are approximately 70/15/15."""
    raise NotImplementedError("01-05: assert split fractions ~ 0.70 / 0.15 / 0.15")
