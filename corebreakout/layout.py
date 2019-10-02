"""
Utilities related to column layout, order, and cropping.
"""
import numpy as np


# Could add 'b2t' and 'r2l', although those are very unusual
ORIENTATIONS = ['t2b', 'l2r']


def sort_regions(regions, order):
    """Sort skimage `regions` (core columns), given the column `order`."""
    assert order in ORIENTATIONS, f'order {order} must be one of {ORIENTATIONS}'

    idx = 0 if order is 't2b' else 1
    regions.sort(key=lambda x: x.bbox[idx])

    return regions


def transform_region(region_img, orientation):
    """Transform cropped `region_img` (core column), given the depth `orientation`."""
    assert orientation in ORIENTATIONS, f'orientation {orientation} must be one of {ORIENTATIONS}'

    if orientation is 't2b':
        return region_img
    elif orientation is 'l2r':
        return np.rot90(region_img, k=-1)


def crop_region(img, labels, region, axis=0, endpts=(815, 6775)):
    """Adjust region bbox and return cropped region * mask.

    Parameters
    ----------
    img : array
        The image to crop
    labels : array
        Mask of integer labels, same height and width as `img`
    region : skimage.RegionProperties instance
        Region object corresponding to column to crop around
    axis : int, optional
        Which axis to change `endpts` along, default=0 (y-coordinates)
    endpts : tuple(int)
        Least extreme endpoint coordinates allowed along `axis`

    Returns
    -------
    region : array
        Masked image region, cropped in (adjusted) bounding box
    """
    r0, c0, r1, c1 = region.bbox

    if axis is 0:
        c0, c1 = min(c0, endpts[0]), max(c1, endpts[1])
    elif axis is 1:
        r0, r1 = min(r0, endpts[0]), max(r1, endpts[1])

    region_img = img * np.expand_dims(labels==region.label, -1)

    return region_img[r0:r1,c0:c1,:]
