"""
Define a suite of tests for the CoreSegmenter class.
"""
import pytest

from corebreakout import CoreSegmenter


def test_expected_tops_bases():
    """Test the computation of tops/bases from `depth_range`s and `col_height`s."""

    depth_range, col_height = [1.0, 2.5], 1.0
    tops, bases = CoreSegmenter.expected_tops_bases(depth_range, col_height)
    assert tops == [1.0, 2.0] and bases == [2.0, 3.0], "Simple case."

    depth_range = [1.0, 2.0]
    tops, bases = CoreSegmenter.expected_tops_bases(depth_range, col_height)
    assert len(tops) == 1, "Should find one expected column."


# Since construction/segmentation requires saved weights,
# we will not use these tests for now:
def test_segmenter_construction():
    pass


def test_segmenter_segmentation():
    pass
