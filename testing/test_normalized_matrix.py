# pylint: disable=C0116, W0212
"""Tests for the NormalizedMatrix"""

import numpy as np
from paxplot.data_managers.normalized_matrix import NormalizedMatrix
from paxplot.data_managers.numeric_normalized_array import NumericNormalizedArray
from paxplot.data_managers.categorical_normalized_array import CategoricalNormalizedArray


def test_normalized_matrix_basic():
    data = [
        [1.0, "apple"],
        [2.0, "banana"],
        [3.0, "apple"]
    ]

    matrix = NormalizedMatrix(data=data)

    # Test dimensions
    assert matrix.num_columns == 2
    assert matrix.num_rows == 3

    # Test column types
    col0 = matrix._columns[0]
    col1 = matrix._columns[1]
    assert isinstance(col0, NumericNormalizedArray)
    assert isinstance(col1, CategoricalNormalizedArray)

    # Test normalized arrays
    np.testing.assert_array_equal(matrix.get_normalized_array(0), np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(matrix.get_normalized_array(1), np.array([-1.0, 1.0, -1.0]))
