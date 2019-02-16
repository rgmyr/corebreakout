"""
CoreColumn object for manipulating and combining single columns of core.

TODO:
    - do we really need 'slice_depth', or can we do something more clever?
"""
import numpy as np
import pandas as pd


def stack_images(imgA, imgB)
    """
    Stack `imgA` and `imgB` vertically after zero-padding the narrower one.
    """
    dw = imgA.shape[1] - imgB.shape[1]

    if dw == 0:
        return np.concatenate([self.img, other.img])
    else:
        pads = ((0,0), (0, abs(dw)), (0,0))
    
    if dw < 0:
        paddedA = np.pad(imgA, pads, 'constant')
        return np.concatenate([paddedA, imgB])
    else:
        paddedB = np.pad(imgB, pads, 'constant')
        return np.concatenate([imgA, paddedB])



class CoreColumn:
    """
    Handles combining of depth labeled columns via the __add__ operator. 
    
    Either `depths` or `top` and `base` must be given.
    
    Parameters
    ----------
    img : array
        2D (grayscale) or 3D (color) image array representing a single column of core material
    depths : array, optional
        An array of depths (one value for each row in `img`). 
        If not given, will be computed with np.linspace from `top` and `base`.
    top : array, optional
        The top depth. If not given and `depths` is, will be taken as depth of first row.
    base : float, optional
        The base depth. If not given and `depths` is, will be taken as depth of last row.
    add_tol : float, optional
        Maximum allowed depth gap between columns when adding. Default use `2*dd`,
        where `dd` is the median difference in depth between adjacent column row depths.
    add_mode : one of {'fill', 'collapse'}, optional
        How to add to this column. Both methods enforce depth ordering:
            - 'fill' will fill in "missing" depth gap with a black image.
            - 'collapse' will simply concatenate `img` and `depths` arrays.
        Default is 'fill'. 
    """
    def __init__(self, img, depths=None, top=None, base=None, add_tol=None, add_mode='safe'):

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
        assert np.all(self.depths >= self.top), 'no `depths` allowed above `top`'
        assert np.all(self.depths <= self.base), 'no `depths` allowed below `base`'

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
    def dd(self):
        """An approximate value for the size of each row in depth units."""
        return np.median(np.diff(self.depths)) 

    @property
    def depth_range(self):
        return (self.top, self.base)


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
        assert mode in ('safe', 'collapse', 'fill'), f'{mode} not a valid `add_mode`'
        self._add_mode = mode


    def slice_depth(self, top=None, base=None):
        """
        Slice the core column between top and base, if possible to do so.
        """
        top = top or self.top
        base = base or self.base
        assert base > top, 'Slice boundaries must maintain depth order.'

        # check that there's a difference, and that it isn't a superset of current range
        if [top, base] != [self.top, self.base] and (top > self.top or base < self.base):

            in_new_range = np.logical_and(self.depths >= top, self.depths <= new_bottom)
            self.img = self.img[in_new_range]
            self.depths = self.depths[in_new_range]

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
        Adding two CoreSampleColumn objects should append RHS below LHS.
        `img` and `depths` are concatenated, `top` and `base` passed through.

        If `add_mode` is 'collapse', things are just naively stacked. 
        If `add_mode` is 'fill', an black image appended to self to fill the gap.
        """
        if self.channels != other.channels:
            raise UserWarning(f'Cant add columns ({self} \n\t+\n {other})!')

        depth_diff = other.top - self.base
        print(self.depth_range, other.depth_range, depth_diff)
            
        if depth_diff < 0:
            raise UserWarning(f'Cant add shallower {other} below deeper {self}!')

        elif depth_diff > self.add_tol:
            raise UserWarning(f'Gap of {depth_diff} greater than `add_tol`: {self.add_tol}!')


        if self.add_mode is 'fill':

            fill_dd = (self.dd + other.dd) / 2
            fill_rows = depth_diff // fill_dd
            fill_depths = np.linspace(self.depths[-1]+fill_dd, other.depths[0]-fill_dd, num=fill_rows)
            fill_img = np.zeros((fill_rows, self.width, self.channels), dtype=self.img.dtype)

            self.img = stack_images(self.img, fill_img)
            self.depths = np.concatenate(self.depths, fill_depths)


        return CoreColumn(stack_images(self.img, other.img),
                         depths = np.concatenate([self.depths, other.depths]),
                         top = self.top, base = other.base,
                         add_tol = max(self.add_tol, other.add_tol),
                         add_mode = self.add_mode)

