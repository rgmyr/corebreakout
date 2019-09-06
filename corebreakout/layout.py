
import numpy as np

# Should `Layout` be a class?
DEFAULT_LAYOUT = {
    'sort_axis' : 1,        # columns laid out vertically, ordered horizontally (0 for the inverse)
    'sort_order' : +1,      # +1 for left-to-right or top-to-bottom (-1 for right-to-left or bottom to top)
    'col_height' : 1.0,
    'endpts' : (815, 6775)  # can also be name of a class for object-based column endpoints
}

ORIENTATIONS = ['t2b', 'l2r'] # 'b2t'? 'r2l'?

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
