"""
Define a suite of tests for the CoreColumn class.
"""
import pytest

import numpy as np
from skimage import io, color
from corebreakout import CoreColumn

# Example (unmasked) single-column images
img1 = io.imread('data/column1.jpeg')       # shape = (6070, 782, 3)
img2 = io.imread('data/column2.jpeg')       # shape = (5917, 779, 3)
img3 = io.imread('data/column3.jpeg')       # shape = (4469, 803, 3)


def test_construction():

    with pytest.raises(ValueError):
        _ = CoreColumn(np.random.random(100))       # 1D array should fail
    with pytest.raises(ValueError):
        _ = CoreColumn(np.expand_dims(img1, -1))    # 4D array should fail



def test_addition():
    pass


def test_slicing():
    pass
