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

def test_to_dict_and_from_dict():
    # Sample mixed data: 2 numeric columns, 1 categorical
    data = [
        [1.0, 10, "cat"],
        [2.0, 20, "dog"],
        [3.0, 15, "mouse"],
    ]
    nm = NormalizedMatrix(data=data)

    # Set custom bounds on first numeric column only
    nm.set_custom_bounds(0, min_val=0.0, max_val=5.0)
    # Second numeric column no custom bounds
    # Third column categorical, so no custom bounds

    # Serialize to dict
    serialized = nm.to_dict()

    # Check serialized keys
    assert "data" in serialized
    assert "column_types" in serialized
    assert "custom_bounds" in serialized
    assert "_schema_version" in serialized

    # Check data is preserved
    assert serialized["data"] == data

    # Check column types are correct
    assert serialized["column_types"] == ["numeric", "numeric", "categorical"]

    # Check custom bounds present only for first column
    assert 0 in serialized["custom_bounds"]
    assert serialized["custom_bounds"][0]["custom_min_val"] == 0.0
    assert serialized["custom_bounds"][0]["custom_max_val"] == 5.0
    # No custom bounds for second numeric column
    assert 1 not in serialized["custom_bounds"] or all(
        v is None for v in serialized["custom_bounds"][1].values()
    )

    # Deserialize from dict
    nm2 = NormalizedMatrix.from_dict(serialized)

    # Data roundtrip
    assert nm2.data == data

    # Confirm types of columns
    assert isinstance(nm2._columns[0], NumericNormalizedArray)
    assert isinstance(nm2._columns[1], NumericNormalizedArray)
    assert isinstance(nm2._columns[2], CategoricalNormalizedArray)

    # Confirm custom bounds preserved on first numeric column
    min_val, max_val = nm2.get_custom_bounds(0)
    assert min_val == 0.0
    assert max_val == 5.0

    # Confirm no custom bounds on second numeric column
    with pytest.raises(TypeError):
        nm2.get_custom_bounds(2)
