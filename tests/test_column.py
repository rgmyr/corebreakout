"""
Define a suite of tests for the `corebreakout.CoreColumn` class.
"""
import tempfile

import pytest
import numpy as np
from skimage import io, color

from corebreakout import CoreColumn


# Example (unmasked) single-column images
img1 = io.imread("tests/data/column1.jpeg")  # shape = (6070, 782, 3)
img2 = io.imread("tests/data/column2.jpeg")  # shape = (5917, 779, 3)
img3 = io.imread("tests/data/column3.jpeg")  # shape = (4469, 803, 3)

height1, height2, height3 = (img.shape[0] for img in [img1, img2, img3])


def test_construction():
    """Try various construction arguments that should fail or succeed."""

    # 1D array should fail
    with pytest.raises(ValueError):
        _ = CoreColumn(np.random.random(100), top=1.0, base=2.0)

    # 4D array should fail
    with pytest.raises(ValueError):
        _ = CoreColumn(np.expand_dims(img1, -1), top=1.0, base=2.0)

    # No depth info should fail
    with pytest.raises(AssertionError):
        _ = CoreColumn(img1)

    # Just depths should be fine
    _ = CoreColumn(img1, depths=np.linspace(1.0, 2.0, num=img1.shape[0]))

    # Depth and image size mismatch should fail
    with pytest.raises(AssertionError):
        _ = CoreColumn(img1, depths=np.linspace(1.0, 2.0, num=100))

    # Grayscale should be allowed
    gray_column = CoreColumn(color.rgb2gray(img1), top=1.0, base=2.0)
    assert gray_column.channels == 1, "Grayscale image should have 1 channel"


def test_addition():
    """Test various column combination possibilities."""

    column1 = CoreColumn(img1, top=1.0, base=2.0)
    column2 = CoreColumn(img2, top=2.0, base=3.0)
    column3 = CoreColumn(img3, top=3.0, base=4.0)

    # Adjacent images -> should not fill between
    one_plus_two = column1 + column2
    assert one_plus_two.height == (height1 + height2)

    # Gap is bigger than default `add_tol`
    with pytest.raises(UserWarning):
        _ = column1 + column3

    # Should only be able to add in depth order
    with pytest.raises(UserWarning):
        _ = column2 + column1

    # Change `add_tol`
    column1.add_tol = 1.0
    column1.add_mode = "collapse"
    one_plus_three = column1 + column3
    assert one_plus_three.height == (height1 + height3), "collapse == naive vstack"

    # Make sure 'fill' ends up filling
    column1.add_mode = "fill"
    one_plus_three = column1 + column3
    assert one_plus_three.height > (
        height1 + height3
    ), "`fill` should have filled something"


def test_slicing():
    """Test the `slice_depth` method."""

    column = CoreColumn(img1, top=1.0, base=2.0)
    height = column.height

    # Superset should have no effect
    superset_column = column.slice_depth(top=0.0, base=3.0)
    assert superset_column == column

    # These should reduce data
    slice1 = column.slice_depth(base=1.5)
    assert slice1.height < column.height

    slice2 = column.slice_depth(top=1.5)
    assert slice2.height < column.height

    # These should not be allowed
    with pytest.raises(AssertionError):
        _ = column.slice_depth(top=1.75, base=1.25)

    with pytest.raises(AssertionError):
        _ = column.slice_depth(top=2.5)

    with pytest.raises(AssertionError):
        _ = column.slice_depth(base=0.5)


def test_pickle_save_load():
    """Test saving as a single pickle file."""

    save_column = CoreColumn(img1, top=1.0, base=2.0)

    with tempfile.TemporaryDirectory() as TEMP_PATH:

        save_column.save(TEMP_PATH, name='testcol',
                        pickle=True, image=False, depths=False)

        load_column = CoreColumn.load(TEMP_PATH, 'testcol')

    assert load_column == save_column, 'Loaded should match saved.'


def test_numpy_save_load():
    """Test numpy image + depths save."""

    save_column = CoreColumn(img1, top=1.0, base=2.0)

    with tempfile.TemporaryDirectory() as TEMP_PATH:

        save_column.save(TEMP_PATH, name='testcol',
                        pickle=False, image=True, depths=True)

        load_column = CoreColumn.load(TEMP_PATH, 'testcol')

    assert load_column == save_column, 'Loaded should match saved.'


def test_image_only_save_load():
    """Test numpy image-only save."""

    save_column = CoreColumn(img1, top=1.0, base=2.0)

    with tempfile.TemporaryDirectory() as TEMP_PATH:

        save_column.save(TEMP_PATH, name='testcol',
                        pickle=False, image=True, depths=False)

        load_column = CoreColumn.load(TEMP_PATH, 'testcol', top=1.0, base=2.0)

    assert load_column == save_column, 'Loaded should match saved.'
