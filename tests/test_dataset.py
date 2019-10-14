
import pytest

from corebreakout.datasets import PolygonDataset


def test_class_check():
    """Test the restriction on classes."""
    good_classes = ['col', 'tray']
    assert PolygonDataset.check_classes(good_classes), 'These should work fine.'

    not_good_classes = ['col', 'col_tray']
    assert not PolygonDataset.check_classes(not_good_classes), 'Substrings not allowed.'
