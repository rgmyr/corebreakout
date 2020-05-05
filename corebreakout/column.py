"""
CoreColumn abstraction representing depth-registered single-column images of core material.
"""

from pathlib import Path

import dill
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt

from corebreakout import utils, defaults
from corebreakout.viz import make_depth_ticks


class CoreColumn:
    """Container for depth-registered, single-column images of core material.

    These can be stacked with the ``__add__`` operator, using ``add_mode`` from LHS instance,
    and padding the width of the narrower of ``(LHS.img, RHS.img)`` with zeros if necessary.

    Either ``depths`` array or scalar ``top`` and ``base`` values must be provided to constructor.
    """

    def __init__(
        self, img, depths=None, top=None, base=None, add_tol=None, add_mode="fill"
    ):
        """
        Parameters
        ----------
        img : array
            2D (grayscale) or 3D (RGB) image array representing a single column of core material
        depths : array, optional
            1D array of depths with `size=img.shape[0]` (one value for each row).
            If not provided, depths will be evenly spaced between `top` and `base`.
        top : array, optional
            The top depth. If not given, assumed to be first row of `depths`.
        base : float, optional
            The base depth. If not given, assumed to be last row of `depths`.
        add_tol : float, optional
            Maximum allowed depth gap between columns when adding. Default is to use `2*dd`,
            where `@property dd` is the median difference in depth between adjacent `img` rows.
        add_mode : one of {'fill', 'collapse'}, optional
            How to add to this column. Both methods enforce depth ordering (LHS.base <= RHS.top):
                - 'fill' is the default. Fills any depth gap with zero image and interpolated depths.
                - 'collapse' will simply concatenate `img` and `depths` arrays.
            Default is 'fill'.
        """
        self.img = img  # img.setter called

        depths_given, top_given, base_given = (
            depths is not None,
            top is not None,
            base is not None,
        )

        assert depths_given or (
            top_given and base_given
        ), "Must specify either `depths`, or `top` and `base`"

        if not depths_given:
            self.top, self.base = top, base
            #eps = (base - top) / (2 * self.height)
            self.depths = np.linspace(top, base, num=self.height)

        elif not (top_given and base_given):
            self.depths = depths
            #eps = (depths[-1] - depths[0]) / (2 * self.height)
            self.top, self.base = depths[0], depths[-1]

        else:
            self.top, self.base, self.depths = top, base, depths

        assert self.base > self.top, "`top` and `base` must be depth ordered"
        assert np.all(self.depths >= self.top), "no `depths` can be above `top`"
        assert np.all(self.depths <= self.base), "no `depths` can be below `base`"

        # settings for column addition
        self.add_mode = add_mode
        self.add_tol = add_tol or 2 * self.dd

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, arr):
        if arr.ndim == 2:
            self._img = arr[:, :, np.newaxis]
        elif arr.ndim == 3:
            self._img = arr
        else:
            raise ValueError("`img` array must have 2 or 3 dimensions.")

        self.height, self.width, self.channels = self._img.shape

    @property
    def depths(self):
        return self._depths

    @depths.setter
    def depths(self, arr):
        assert type(arr) is np.ndarray and arr.ndim == 1, "`depths` must be 1D array"
        assert arr.size == self.height, "length of `depths` must match image height"
        assert np.all(np.diff(arr) >= 0), "`depths` must be monotonic"
        self._depths = arr

    @property
    def depth_range(self):
        """``(self.top, self.base)``"""
        return (self.top, self.base)

    @property
    def dd(self):
        """Median gap between adjacent ``depths``."""
        return np.median(np.diff(self.depths))

    @property
    def add_tol(self):
        """Maximum allowed depth gap between columns when adding. Default is ``2*dd``."""
        return self._add_tol

    @add_tol.setter
    def add_tol(self, value):
        assert value >= 0.0, "`add_tol` cannot be negative"
        self._add_tol = value

    @property
    def add_mode(self):
        return self._add_mode

    @add_mode.setter
    def add_mode(self, mode):
        assert mode in ("collapse", "fill"), f"{mode} not a valid `add_mode`"
        self._add_mode = mode


    def slice_depth(self, top=None, base=None):
        """Get a sliced CoreColumn between `top` and `base`,
        if it would have an effect and it is possible to do so.
        """
        top = top or self.top
        base = base or self.base
        assert base > top, "Slice boundaries must maintain depth order."

        assert top < self.base, f"Cannot slice to top {top} with base {self.base}"
        assert base > self.top, f"Cannot slice to base {base} with top {self.top}"

        # check that there's a difference, and that it isn't a superset of current range
        if [top, base] != [self.top, self.base] and (
            top > self.top or base < self.base
        ):
            idxs = np.logical_and(self.depths >= top, self.depths <= base)
            # self.img = self.img[idxs]
            # self.depths = self.depths[idxs]
            # self.top, self.base = top, base
            return CoreColumn(
                self.img[idxs, ...],
                depths=self.depths[idxs],
                top=top,
                base=base,
                add_mode=self.add_mode,
                add_tol=self.add_tol,
            )
        else:
            return self


    def __repr__(self):
        return (
            f"CoreColumn instance with:\n"
            f"\t img.shape: {self.img.shape}\n"
            f"\t (top, base): ({self.top:.3f}, {self.base:.3f})\n"
            f"\t add_tol & add_mode: {self.add_tol:.4f} , {self.add_mode}\n"
        )


    def __eq__(self, other):
        """Equivalence testing. Includes add options.

        Uses np.isclose/allclose because of floating point errors.
        """
        if self.add_mode != other.add_mode:
            return False
        if not np.isclose(self.add_tol, other.add_tol):
            return False

        if not np.isclose(self.top, other.top):
            return False
        if not np.isclose(self.base, other.base):
            return False

        if (self.height != other.height):
            return False

        if not np.allclose(self.depths, other.depths):
            return False
        if not np.allclose(self.img, other.img):
            return False

        return True


    def iter_chunks(self, chunk_size, depths=True, step_size=None):
        """Generate data in `chunk_size` pieces, starting `step_size` apart.

        If `depths`, yields `(img, depths)` of each chunk, else just `img`

        TODO: make this able to fill partial last chunks.
        """
        step_size = step_size or chunk_size

        i = 0
        while i  < self.height:
            if depths:
                yield self.img[i:i+chunk_size], self.depths[i:i+chunk_size]
            else:
                yield self.img[i:i+chunk_size]
            i += step_size

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
        print(self.depth_range, " + ", other.depth_range, " gap: ", depth_diff)

        if depth_diff < 0:
            raise UserWarning(f"Cant add shallower {other} below deeper {self}!")

        elif depth_diff > self.add_tol:
            raise UserWarning(
                f"Gap of {depth_diff} greater than `LHS.add_tol`: {self.add_tol}!"
            )

        # If 'fill' mode, extend `self.img` and `self.depths` to fill any gap
        if self.add_mode is "fill":
            fill_dd = (self.dd + other.dd) / 2
            # Note: have to call int() for cases of 0.0
            fill_rows = int(depth_diff // fill_dd)

            if fill_rows > 0:
                fill_depths = np.linspace(
                    self.depths[-1] + fill_dd, other.depths[0] - fill_dd, num=fill_rows
                )
                fill_img = np.zeros(
                    (fill_rows, self.width, self.channels), dtype=self.img.dtype
                )

                # NOTE: could use temp var here to avoid modifying `self`
                # Doesn't seem too important + obvious ways of doing that
                # result in writing these funcs multiple (3?) times
                self.img = utils.vstack_images(self.img, fill_img)
                self.depths = np.concatenate([self.depths, fill_depths])

        return CoreColumn(
            utils.vstack_images(self.img, other.img),
            depths=np.concatenate([self.depths, other.depths]),
            top=self.top,
            base=other.base,
            add_tol=max(self.add_tol, other.add_tol),
            add_mode=self.add_mode,
        )

    ###+++++++++++++++###
    ### Save and Load ###
    ###+++++++++++++++###

    def save(self, path, name=None, pickle=True, image=False, depths=False):
        """Save the CoreColumn (or parts of it) to directory `path`.

        Parameters
        ----------
        path : str or Path
            Location to save to (must exist and be a directory).
        name : str, optional
            Stem to save files as, default=`CoreColumn_<top>_<base>`.
        pickle : bool, optional
            Whether to pickle the entire object with `dill`, default=True.
        image : bool, optional
            Whether to save the image as '.npy' file, default=False.
        depths : bool, optional
            Whether to save the depths as a '.npy' file, default=False
        """
        assert pickle or image or depths, "Must save something."

        path = Path(path)
        assert path.exists() and path.is_dir(), f"Save location {path} doesnt exist."

        if name:
            assert type(name) is str, "Name must be a string"
        else:
            name = f"CoreColumn_{self.top:.2f}_{self.base:.2f}"

        if pickle:
            with open(path / (name + ".pkl"), "wb") as pfile:
                dill.dump(self, pfile)
        if image:
            np.save((path / (name + "_image.npy")), self.img)
        if depths:
            np.save((path / (name + "_depths.npy")), self.depths)


    @classmethod
    def load(cls, path, name, **kwargs):
        """Load a CoreColumn instance from directory `path`.

        If '<name>.pkl' exists, will just load from that file.

        Otherwise, at least '<name>_image.npy' must exist. If '<name>_depths.npy'
        also exists, those will be read as `depths`. If not, the user must pass
        either `depths` or `top` & `base` as **kwargs.
        """
        path = Path(path)
        assert path.exists() and path.is_dir(), f"Load location {path} doesnt exist."

        pickle_path = path / (name + ".pkl")
        image_path = path / (name + "_image.npy")
        depths_path = path / (name + "_depths.npy")

        if pickle_path.is_file():
            with open(pickle_path, 'rb') as pickle_file:
                return dill.load(pickle_file)

        assert image_path.is_file(), "_image.npy file must exist if pickle doesnt."
        img = np.load(image_path)

        if depths_path.is_file():
            kwargs["depths"] = np.load(depths_path)
        else:
            assert (
                "top" in kwargs.keys() and "base" in kwargs.keys()
            ), "Depth info needed."

        return cls(img, **kwargs)

    ###+++++++++++++++++++###
    ###  Column Plotting  ###
    ###+++++++++++++++++++###

    def plot(self, figsize=(15, 50), tick_kwargs={}, major_kwargs={}, minor_kwargs={}):
        """Make an image figure with major and minor depth ticks.

        Parameters
        ----------
        figsize : tuple(int)
            Size of matplotlib figure to plot on.  Note: at default DPI of 100, 650 is
            about as large as common image formats will support saving (~2^16 pixels).
        tick_kwargs:
            Parameters for tick creation: 'major' and 'minor' options for
            `*_precision` and `*_format_str`. See `viz.make_depth_ticks()`.
        major/minor_kwargs:
            Parameters for tick size and appearance. Passed to `ax.tick_params`.

        Returns
        -------
        fig, ax
            Matplotlib figure and axis with image + ticks plotted.
        """
        # Update any new tick kwargs
        tick_kwargs = utils.strict_update(defaults.DEPTH_TICK_ARGS, tick_kwargs)
        major_kwargs = utils.strict_update(defaults.MAJOR_TICK_PARAMS, major_kwargs)
        minor_kwargs = utils.strict_update(defaults.MINOR_TICK_PARAMS, minor_kwargs)

        fig, ax = plt.subplots(figsize=figsize)

        major_ticks, major_locs, minor_ticks, minor_locs = make_depth_ticks(
            self.depths, **tick_kwargs
        )

        ax.yaxis.set_major_formatter(ticker.FixedFormatter((major_ticks)))
        ax.yaxis.set_major_locator(ticker.FixedLocator((major_locs)))

        ax.yaxis.set_minor_formatter(ticker.FixedFormatter((minor_ticks)))
        ax.yaxis.set_minor_locator(ticker.FixedLocator((minor_locs)))

        ax.tick_params(which="major", **major_kwargs)
        ax.tick_params(which="minor", **minor_kwargs)

        ax.set_xticks([], [])
        ax.grid(False)

        ax.imshow(self.img)

        return fig, ax
