"""
CoreColumn object representing single column images of core material.

TODO:
    - do we really need 'slice_depth', or can we do something more clever?
"""
import numpy as np
import pandas as pd

from corebreakout import utils


class CoreColumn:
    """
    Container for depth-registered, single-column images of core material.

    `CoreColumn`s can be stacked via the __add__ operator, using `add_mode` from LHS instance,
    and padding the width of the narrower of (LHS.img, RHS.img) with zeros if necessary.

    Either `depths` array or scalar `top` and `base` values must be provided to constructor.

    Parameters
    ----------
    img : array
        2D (grayscale) or 3D (RGB) image array representing a single column of core material
    depths : array, optional
        1D array of depths with `size=img.shape[0]` (one value for each row).
        If not provided, depths are computed as `np.linspace(top, base, num=img.shape[0])`.
    top : array, optional
        The top depth. If not given and `depths` is, assumed to be depth of first row.
    base : float, optional
        The base depth. If not given and `depths` is, assumed to be depth of last row.
    add_tol : float, optional
        Maximum allowed depth gap between columns when adding. Default is to use `2*dd`,
        where `@property dd` is the median difference in depth between adjacent `img` rows.
    add_mode : one of {'fill', 'collapse'}, optional
        How to add to this column. Both methods enforce depth ordering (LHS.base <= RHS.top):
            - 'fill' is the default. Fills any depth gap with zero image and interpolated depths.
            - 'collapse' will simply concatenate `img` and `depths` arrays.
        Default is 'fill'.
    """
    def __init__(self, img, depths=None, top=None, base=None, add_tol=None, add_mode='fill'):

        self.img = img    # img.setter called

        assert depths or (top and base), 'Must specify either `depths` or `top` and `base`'

        if not depths:
            self.top, self.base = top, base
            eps = (base - top) / (2 * self.height)
            self.depths = np.linspace(top+eps, base-eps, num=self.height)

        elif not (top and base):
            self.depths = depths
            eps = (depths[-1] - depths[0]) / (2 * self.height)
            self.top = depths[0] - eps
            self.base = depths[-1] + eps

        else:
            self.top, self.base, self.depths = top, base, depths

        assert self.base > self.top, '`top` and `base` must be depth ordered'
        assert np.all(self.depths >= self.top), 'no `depths` can be above `top`'
        assert np.all(self.depths <= self.base), 'no `depths` can be below `base`'

        # settings for column addition
        self.add_mode = add_mode
        self.add_tol = add_tol or 2*self.dd


    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, arr):
        if arr.ndim == 2:
            self._img = arr[:,:,np.newaxis]
        elif arr.ndim == 3:
            self._img = arr
        else:
            raise ValueError('`img` array must have 2 or 3 dimensions.')

        self.height, self.width, self.channels = self._img.shape

    @property
    def depths(self):
        return self._depths

    @depths.setter
    def depths(self, arr):
        assert type(arr) is np.ndarray and arr.ndim == 1, '`depths` must be 1D array'
        assert arr.size == self.height, 'length of `depths` must match image height'
        self._depths = arr

    @property
    def depth_range(self):
        return (self.top, self.base)

    @property
    def dd(self):
        """
        An approximate value for the size of each row in depth units.
        """
        return np.median(np.diff(self.depths))

    @property
    def add_tol(self):
        return self._add_tol

    @add_tol.setter
    def add_tol(self, value):
        assert value >= 0.0, '`add_tol` cannot be negative'
        self._add_tol = value


    @property
    def add_mode(self):
        return self._add_mode

    @add_mode.setter
    def add_mode(self, mode):
        assert mode in ('collapse', 'fill'), f'{mode} not a valid `add_mode`'
        self._add_mode = mode


    def slice_depth(self, top=None, base=None):
        """
        Slice the CoreColumn between `top` and `base`, if it would have an effect and is possible to do so.
        """
        top = top or self.top
        base = base or self.base
        assert base > top, 'Slice boundaries must maintain depth order.'

        # check that there's a difference, and that it isn't a superset of current range
        if [top, base] != [self.top, self.base] and (top > self.top or base < self.base):
            idxs = np.logical_and(self.depths >= top, self.depths <= base)
            self.img = self.img[idxs]
            self.depths = self.depths[idxs]
            self.top, self.base = top, base

        return self


    def __repr__(self):
        return (
            f'CoreColumn instance with:\n'
            f'\t img.shape: {self.img.shape}\n'
            f'\t depth_range: ({self.top}, {self.base})\n'
            f'\t add_tol & mode: {self.add_tol} , {self.add_mode}\n'
        )


    ###++++++++++++++++++++###
    ### Column Combination ###
    ###++++++++++++++++++++###

    def __add__(self, other):
        """
        Adding two CoreColumn objects appends RHS below LHS and returns new CoreColumn instance.
            - `img` and `depths` are concatenated:
                - If `add_mode` is 'collapse', arrays are naively stacked.
                - If `add_mode` is 'fill', a zero image + interpolated depths are added b/t LHS & RHS.
            - `LHS.top`, `RHS.base` and `LHS.add_mode` are passed through to new instance
            - `add_tol` is propagated by `max(LHS.add_tol, RHS.add_tol)`.
        """
        depth_diff = other.top - self.base
        print(self.depth_range, other.depth_range, depth_diff)

        if depth_diff < 0:
            raise UserWarning(f'Cant add shallower {other} below deeper {self}!')

        elif depth_diff > self.add_tol:
            raise UserWarning(f'Gap of {depth_diff} greater than `LHS.add_tol`: {self.add_tol}!')

        # if 'fill', extend `self.img` and `self.depths` to fill the gap
        if self.add_mode is 'fill':
            fill_dd = (self.dd + other.dd) / 2
            fill_rows = depth_diff // fill_dd

            fill_depths = np.linspace(self.depths[-1]+fill_dd, other.depths[0]-fill_dd, num=fill_rows)
            fill_img = np.zeros((fill_rows, self.width, self.channels), dtype=self.img.dtype)

            self.img = utils.vstack_images(self.img, fill_img)
            self.depths = np.concatenate(self.depths, fill_depths)


        return CoreColumn(utils.vstack_images(self.img, other.img),
                         depths = np.concatenate([self.depths, other.depths]),
                         top = self.top, base = other.base,
                         add_tol = max(self.add_tol, other.add_tol),
                         add_mode = self.add_mode)
