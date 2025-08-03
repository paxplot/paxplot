# pylint: disable=C0116, W0212
"""Tests for the CategoricalNormalizedArray"""

import numpy as np

from paxplot.data_managers.categorical_normalized_array import CategoricalNormalizedArray


def test_basic_construction():
    arr = CategoricalNormalizedArray(array=["apple", "banana", "apple", "cherry"])

    np.testing.assert_array_equal(
        arr.array,
        np.array(["apple", "banana", "apple", "cherry"], dtype=np.str_)
    )
    expected_label_to_index = {"apple": 0, "banana": 1, "cherry": 2}
    assert arr._label_to_index == expected_label_to_index
    np.testing.assert_array_equal(arr.array_indeces, np.array([0, 1, 0, 2], dtype=np.float64))
    np.testing.assert_array_equal(arr.array_normalized, np.array([-1.0, 0, -1.0, 1.0]))
