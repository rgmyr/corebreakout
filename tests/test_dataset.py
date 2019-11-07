"""
Define a suite of tests for the PolygonDataset class.
"""
from pathlib import Path

import pytest

from corebreakout.datasets import PolygonDataset
from corebreakout import __file__ as PKG_FILE

DATA_DIR = Path(PKG_FILE).parent.parent / "tests/data"
DATA_SUBSET = "two_image_dataset"

good_classes = ["col", "tray"]
not_good_classes = ["col", "col_tray"]


def test_class_check():
    """Test the substring restriction on classes."""

    assert PolygonDataset.check_classes(good_classes), "These should work fine."

    assert not PolygonDataset.check_classes(not_good_classes), "Substrings not allowed."


def test_dataset_construction():
    """Make sure dataset directory can be read."""

    dataset = PolygonDataset(classes=good_classes)
    assert len(dataset.class_info) == 3, "Class check, BG added."

    dataset.collect_annotated_images(DATA_DIR, DATA_SUBSET)
    dataset.prepare()
    assert len(dataset.image_ids) == 2, "Data check."
