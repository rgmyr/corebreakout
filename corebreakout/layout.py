"""
Utilities related to column layout and ordering.
"""
import numpy as np


# Could add 'b2t' and 'r2l', although those are very unusual
ORIENTATIONS = ['t2b', 'l2r']


def sort_regions(regions, order):
    """
    Sort `skimage` regions (core columns), given the column `order` orientation
    """
    assert order in ORIENTATIONS, f'order {order} must be one of {ORIENTATIONS}'
    idx = 0 if order is 't2b' else 1
    regions.sort(key=lambda x: x.bbox[idx])
    return regions


def transform_region(region, orientation):
    """
    Transform individual cropped `region` (core column), given the depth orientation
    """
    assert orientation in ORIENTATIONS, f'orientation {orientation} must be one of {ORIENTATIONS}'
    if orientation is 't2b':
        return region
    elif orientation is 'l2r':
        return np.rot90(region, k=-1)
