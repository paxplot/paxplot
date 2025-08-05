# pylint: disable=C0116, W0212
"""Tests for the NormalizedMatrix"""

import pytest
import numpy as np

from paxplot.data_managers.normalized_matrix import NormalizedMatrix, ColumnType
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
    assert matrix.get_column_type(0) == ColumnType.NUMERIC
    assert matrix.get_column_type(1) == ColumnType.CATEGORICAL

    # Test normalized arrays
    np.testing.assert_array_equal(matrix.get_normalized_array(0), np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(matrix.get_normalized_array(1), np.array([-1.0, 1.0, -1.0]))

    # Testing original arrays
    assert matrix.get_numeric_array(0) == [1.0, 2.0, 3.0]
    assert matrix.get_categorical_array(1) == ["apple", "banana", "apple"]


def test_normalized_matrix_raises_on_none():
    """Test that initializing with None values raises a ValueError."""
    data_with_none = [
        [1.0, "apple"],
        [None, "banana"],  # Invalid None value in numeric column
        [3.0, "apple"]
    ]

    with pytest.raises(ValueError, match="None values.*not allowed"):
        NormalizedMatrix(data=data_with_none)