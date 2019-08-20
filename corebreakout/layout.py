

# Should `Layout` be a class?
DEFAULT_LAYOUT = {
    'sort_axis' : 1,        # columns laid out vertically, ordered horizontally (0 for the inverse)
    'sort_order' : +1,      # +1 for left-to-right or top-to-bottom (-1 for right-to-left or bottom to top)
    'col_height' : 1.0
    'endpts' : (815, 6775)  # can also be name of a class for object-based column endpoints
}


class ImageLayout:
    """
    Parameters necessary for consistent cropping and sorting of column regions.
    """
    # The axis to sort columns along
    # 0 :
    # 1
    SORT_AXIS = 1

    #
    SORT_ORDER = 1

    # Height of a single (full) column in depth units
    COL_HEIGHT = 1.0

    # Name of class to use for setting column endpoints (e.g., 'tray', 'scale')
    # or hardcoded endpoints in pixel coordinates (lower_idx, upper_idx)
    ENDPTS = 'tray'

    def __init__(self, **params):
        for k, v in params:
            if hasattr(self, k.upper()):
                setattr()
