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
    col0 = matrix[0]
    col1 = matrix[1]
    assert isinstance(col0, NumericNormalizedArray)
    assert isinstance(col1, CategoricalNormalizedArray)
    assert matrix.get_column_type(0) == ColumnType.NUMERIC
    assert matrix.get_column_type(1) == ColumnType.CATEGORICAL

    # Test normalized arrays
    np.testing.assert_array_equal(col0.values_normalized, np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(col1.values_normalized, np.array([-1.0, 1.0, -1.0]))

    # Testing original arrays
    assert col0.values == [1.0, 2.0, 3.0]
    assert col1.values == ["apple", "banana", "apple"]

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
    col0 = matrix[0]
    col1 = matrix[1]
    assert col0.values == [1.0, 2.0, 3.0, 4.0]
    assert col1.values == ["apple", "banana", "apple", "cherry"]

    np.testing.assert_allclose(
        col0.values_normalized, np.array([-1.0, -0.33333333, 0.33333333, 1.0])
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

def test_remove_rows_success():
    data = [
        [10.0, "apple"],
        [20.0, "banana"],
        [30.0, "cherry"],
        [40.0, "banana"]
    ]
    matrix = NormalizedMatrix(data=data)

    # Remove second and fourth rows (index 1 and 3)
    matrix.remove_rows([1, 3])

    # Check resulting size
    assert matrix.num_rows == 2
    col0 = matrix[0]
    col1 = matrix[1]
    assert col0.values == [10.0, 30.0]
    assert col1.values == ["apple", "cherry"]

    # Check normalized values
    np.testing.assert_array_equal(col0.values_normalized, np.array([-1.0, 1.0]))
    np.testing.assert_array_equal(col1.values_normalized, np.array([-1.0, 1.0]))

def test_remove_rows_invalid_index():
    data = [
        [1.0, "a"],
        [2.0, "b"]
    ]
    matrix = NormalizedMatrix(data=data)

    # Index out of range
    with pytest.raises(IndexError, match="out of bounds"):
        matrix.remove_rows([2])

    # Non-integer input
    with pytest.raises(TypeError, match="sequence of integers"):
        matrix.remove_rows(["not-an-index"]) # type: ignore

def test_remove_rows_empty_list_does_nothing():
    data = [
        [1.0, "x"],
        [2.0, "y"]
    ]
    matrix = NormalizedMatrix(data=data)

    matrix.remove_rows([])

    assert matrix.num_rows == 2
    assert matrix[0].values == [1.0, 2.0]
    assert matrix[1].values == ["x", "y"]
