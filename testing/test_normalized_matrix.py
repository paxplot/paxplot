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

def test_append_data_success():
    data = [
        [1.0, "apple"],
        [2.0, "banana"]
    ]
    matrix = NormalizedMatrix(data=data)

    # Append compatible new data
    matrix.append_data([
        [3.0, "apple"],
        [4.0, "cherry"]
    ])

    assert matrix.num_rows == 4
    assert matrix.get_numeric_array(0) == [1.0, 2.0, 3.0, 4.0]
    assert matrix.get_categorical_array(1) == ["apple", "banana", "apple", "cherry"]

    np.testing.assert_allclose(
        matrix.get_normalized_array(0), np.array([-1.0, -0.33333333, 0.33333333, 1.0])
    )


def test_append_data_type_mismatch():
    data = [
        [1.0, "apple"],
        [2.0, "banana"]
    ]
    matrix = NormalizedMatrix(data=data)

    # Try appending wrong types (string into numeric column)
    with pytest.raises(TypeError, match="expects numeric values"):
        matrix.append_data([
            ["oops", "cherry"]
        ])

    # Try appending wrong types (number into categorical column)
    with pytest.raises(TypeError, match="expects string values"):
        matrix.append_data([
            [3.0, 123]
        ])


def test_append_data_dimension_mismatch():
    data = [
        [1.0, "apple"]
    ]
    matrix = NormalizedMatrix(data=data)

    with pytest.raises(ValueError, match="Expected 2 columns, got 1"):
        matrix.append_data([
            [2.0]  # Too few columns
        ])

    with pytest.raises(ValueError, match="Expected 2 columns, got 3"):
        matrix.append_data([
            [2.0, "banana", "extra"]
        ])


def test_get_column_type_invalid_index():
    data = [
        [1.0, "x"]
    ]
    matrix = NormalizedMatrix(data=data)

    with pytest.raises(IndexError):
        _ = matrix.get_column_type(2)  # out of range


def test_empty_input_raises():
    with pytest.raises(ValueError, match="Input must be a 2D array-like structure"):
        NormalizedMatrix(data=[])


def test_append_empty_list_does_nothing():
    data = [[1, "a"]]
    matrix = NormalizedMatrix(data=data)
    matrix.append_data([])
    assert matrix.num_rows == 1
