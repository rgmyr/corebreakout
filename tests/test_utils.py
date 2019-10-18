"""
Define a suite of tests for functions in the `corebreakout.utils` module.
"""
import pytest

import numpy as np
from skimage import measure

from corebreakout import utils


example_masks = np.load('tests/data/example_masks.npy')
example_labels = np.load('tests/data/example_labels.npy')

example_regions = measure.regionprops(example_labels)
flipped_regions = measure.regionprops(example_labels[:,::-1])

def region_labels(regions):
    return [r.label for r in regions]


def test_masks_to_labels():
    _labels = utils.masks_to_labels(example_masks)
    assert np.array_equal(_labels, example_labels), 'Masks conversion.'


def test_sort_regions():
    vertical = utils.sort_regions(example_regions, 't2b')
    assert region_labels(vertical) == [1, 3, 2], 't2b sort order'

    horizontal = utils.sort_regions(flipped_regions, 'l2r')
    assert region_labels(horizontal) == [2, 3, 1], 'l2r sort order'


def test_max_extent():
    assert utils.maximum_extent(example_regions, 0) == (2, 66), 'width extent'

    assert utils.maximum_extent(example_regions, 1) == (14, 45), 'height extent'
