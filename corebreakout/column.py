"""
CoreColumn object representing single column images of core material.

TODO:
    - do we really need 'slice_depth', or can we do something more clever?
"""
import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt

from corebreakout import utils
from corebreakout.defaults import MAJOR_TICK_PARAMS, MINOR_TICK_PARAMS


class CoreColumn:
    """Container for depth-registered, single-column images of core material.

    `CoreColumn`s can be stacked via the `__add__` operator, using `add_mode` from LHS instance,
    and padding the width of the narrower of (LHS.img, RHS.img) with zeros if necessary.

    Either `depths` array or scalar `top` and `base` values must be provided to constructor.
    """
    def __init__(self, img, depths=None, top=None, base=None, add_tol=None, add_mode='fill'):
        """
        Parameters
        ----------
        img : array
            2D (grayscale) or 3D (RGB) image array representing a single column of core material
        depths : array, optional
            1D array of depths with `size=img.shape[0]` (one value for each row).
            If not provided, depths will be evenly spaced between `top` and `base`.
        top : array, optional
            The top depth. If not given, assumed to be just above first row of `depths`.
        base : float, optional
            The base depth. If not given, assumed to be just below last row of `depths`.
        add_tol : float, optional
            Maximum allowed depth gap between columns when adding. Default is to use `2*dd`,
            where `@property dd` is the median difference in depth between adjacent `img` rows.
        add_mode : one of {'fill', 'collapse'}, optional
            How to add to this column. Both methods enforce depth ordering (LHS.base <= RHS.top):
                - 'fill' is the default. Fills any depth gap with zero image and interpolated depths.
                - 'collapse' will simply concatenate `img` and `depths` arrays.
            Default is 'fill'.
        """
        self.img = img    # img.setter called

        depths_given, top_given, base_given = depths is not None, top is not None, base is not None

        assert depths_given or (top_given and base_given), 'Must specify either `depths`, or `top` and `base`'

        if not depths_given:
            self.top, self.base = top, base
            eps = (base - top) / (2 * self.height)
            self.depths = np.linspace(top+eps, base-eps, num=self.height)

        elif not (top_given and base_given):
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
        assert np.all(np.diff(arr) > 0), '`depths` must be strictly increasing'
        self._depths = arr

    @property
    def depth_range(self):
        return (self.top, self.base)

    @property
    def dd(self):
        """An approximate value for the size of each row in depth units."""
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
        """Get a sliced CoreColumn between `top` and `base`,
        if it would have an effect and it is possible to do so.
        """
        top = top or self.top
        base = base or self.base
        assert base > top, 'Slice boundaries must maintain depth order.'

        assert top < self.base, f'Cannot slice to top {top} with base {self.base}'
        assert base > self.top, f'Cannot slice to base {base} with top {self.top}'

        # check that there's a difference, and that it isn't a superset of current range
        if [top, base] != [self.top, self.base] and (top > self.top or base < self.base):
            idxs = np.logical_and(self.depths >= top, self.depths <= base)
            #self.img = self.img[idxs]
            #self.depths = self.depths[idxs]
            #self.top, self.base = top, base
            return CoreColumn(self.img[idxs,...], depths=self.depths[idxs],
                             top=top, base=base,
                             add_mode=self.add_mode, add_tol=self.add_tol)
        else:
            return self


    def __repr__(self):
        return (
            f'CoreColumn instance with:\n'
            f'\t img.shape: {self.img.shape}\n'
            f'\t (top, base): ({self.top}, {self.base})\n'
            f'\t add_tol & add_mode: {self.add_tol} , {self.add_mode}\n'
        )


    ###++++++++++++++++++++###
    ### Column Combination ###
    ###++++++++++++++++++++###

    def __add__(self, other):
        """Adding two CoreColumn objects appends RHS below LHS and returns new CoreColumn instance.
            - `img` and `depths` are concatenated:
                - If `add_mode` is 'collapse', arrays are naively stacked.
                - If `add_mode` is 'fill', a zero image + interpolated depths are added b/t LHS & RHS.
            - `LHS.top`, `RHS.base` and `LHS.add_mode` are passed through to new instance
            - `add_tol` is propagated by `max(LHS.add_tol, RHS.add_tol)`.
        """
        depth_diff = other.top - self.base
        print(self.depth_range, ' + ', other.depth_range, ' gap: ', depth_diff)

        if depth_diff < 0:
            raise UserWarning(f'Cant add shallower {other} below deeper {self}!')

        elif depth_diff > self.add_tol:
            raise UserWarning(f'Gap of {depth_diff} greater than `LHS.add_tol`: {self.add_tol}!')

        # If 'fill' mode, extend `self.img` and `self.depths` to fill any gap
        if self.add_mode is 'fill':
            fill_dd = (self.dd + other.dd) / 2
            fill_rows = int(depth_diff // fill_dd)  # have to call int() for cases of 0.0

            if fill_rows > 0:
                fill_depths = np.linspace(self.depths[-1]+fill_dd, other.depths[0]-fill_dd, num=fill_rows)
                fill_img = np.zeros((fill_rows, self.width, self.channels), dtype=self.img.dtype)

                self.img = utils.vstack_images(self.img, fill_img)
                self.depths = np.concatenate([self.depths, fill_depths])

        return CoreColumn(utils.vstack_images(self.img, other.img),
                         depths = np.concatenate([self.depths, other.depths]),
                         top = self.top, base = other.base,
                         add_tol = max(self.add_tol, other.add_tol),
                         add_mode = self.add_mode)


    ###+++++++++++++++###
    ### Save and Load ###
    ###+++++++++++++++###

    def save(self, path, name=None):
        """Save the CoreColumn to directory `path`.

        Parameters
        ----------
        path : str or Path
            Location to save to (must exist and be a directory).
        name : str, optional
            Stem to save files as, default=`CoreColumn_<top>_<base>`.

        This will save three files in `path`:
            <name>_img.npy : image array
            <name>_depths.npy : depths array
            <name>.pkl : the rest of the instance
        """

        pass

    @classmethod
    def load(cls, path, name):
        """Return a new CoreColumn instance from directory `path`. The three required
        files (<name>_img.npy, <name>_depths.npy, <name>.pkl) must exist.
        """

        pass


    ###+++++++++++++++++++###
    ###  Column Plotting  ###
    ###+++++++++++++++++++###

    def plot(self, figsize=(15, 650), **kwargs):
        """Make an image figure with major and minor depth ticks.

        Parameters
        ----------
        figsize : tuple(int)
            Size of matplotlib figure to plot on.  Note: at default DPI of 100, 650 is
            about as large as common image formats will support saving (~2^16 pixels).
        **kwargs:
            Parameters for tick creation and appearance: 'major' and 'minor' options for
            `*_precision`, `*_format_str`, `*_tick_size`. See `_make_image_ticks()` docs.

        Returns
        -------
        fig, ax
            Matplotlib figure and axis with image + ticks plotted.
        """
        fig, ax = plt.subplots(figsize=figsize)

        major_ticks, major_locs, minor_ticks, minor_locs = self._make_image_ticks(**kwargs)

        ax.yaxis.set_major_formatter(ticker.FixedFormatter((major_ticks)))
        ax.yaxis.set_major_locator(ticker.FixedLocator((major_locs)))

        ax.yaxis.set_minor_formatter(ticker.FixedFormatter((minor_ticks)))
        ax.yaxis.set_minor_locator(ticker.FixedLocator((minor_locs)))

        ax.tick_params(which='major', labelsize=kwargs.get('major_tick_size', 32), color='black')
        ax.tick_params(which='minor', labelsize=kwargs.get('minor_tick_size', 12), color='gray')

        ax.set_xticks([], [])
        ax.grid(False)

        ax.imshow(self.img)

        return fig, ax


    def _make_image_ticks(self, major_precision=0.1,
                          major_format_str='{:.1f}',
                          minor_precision=0.01,
                          minor_format_str='{:.2f}', **kwargs):
        """Generate major & minor (ticks, locs) for image axis.

        Parameters
        ----------
        *_precision : float, optional
            Major, minor tick spacing (in depth units), defaults=0.1, 0.01.
        *_format_str : str, optional
            Format strings to coerce depths -> tick strings, defaults='{:.1f}', '{:.2f}'.

        Returns
        -------
        major_ticks, major_locs, minor_ticks, minor_locs

        *_ticks : lists of tick strings
        *_locs : lists of tick locations in image coordinates (fractional row indices)
        """
        # lambdas to convert values --> strs
        major_fmt_fn = lambda x: major_format_str.format(x)
        minor_fmt_fn = lambda x: minor_format_str.format(x)

        major_ticks, major_locs = [], []
        minor_ticks, minor_locs = [], []

        # remainders w.r.t. precision, round close numbers
        major_rmndr = np.insert(self.depths % major_precision, (0, self.height), np.inf)
        minor_rmndr = np.insert(self.depths % minor_precision, (0, self.height), np.inf)

        for i in np.arange(1, self.height+1):

            if np.argmin(major_rmndr[i-1:i+2]) == 1:
                major_ticks.append(major_fmt_fn(self.depths[i-1]))
                major_locs.append(i)

            elif np.argmin(minor_rmndr[i-1:i+2]) == 1:
                #if major_ticks[-1]+'0' == minor_fmt_fn(self.depths[i-1]):
                if major_ticks[-1] == minor_fmt_fn(self.depths[i-1]):
                    # fixes some overlapping ticks, BUT not robust
                    # enough for all possible precision combos
                    continue
                minor_ticks.append(minor_fmt_fn(self.depths[i-1]))
                minor_locs.append(i)

        # get last tick if needed, doesn't work above for some reason
        last_depth = np.round(self.depths[-1], decimals=1)
        if (last_depth % 1.0) == 0.0:
            major_ticks.append(major_fmt_fn(last_depth))
            major_locs.append(self.height-1)

        return major_ticks, major_locs, minor_ticks, minor_locs
