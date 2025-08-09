# pylint: disable=C0116, W0212
"""Tests for the PlotModel"""

import pytest
from paxplot.plot_model import PlotModel

# Sample initial data: mixed numeric and categorical columns
INITIAL_DATA = [
    [1.0, "cat", 10],
    [2.0, "dog", 20],
    [3.0, "cat", 30],
]

def test_initialization():
    model = PlotModel(initial_data=INITIAL_DATA)
    assert model.num_rows == 3
    assert model.num_columns == 3

def test_append_rows():
    model = PlotModel(initial_data=INITIAL_DATA)
    new_rows = [
        [4.0, "dog", 40],
        [5.0, "mouse", 50],
    ]
    model.append_rows(new_rows)
    assert model.num_rows == 5

    # Confirm raw data updated
    raw_col0 = model.get_raw_column(0)
    assert raw_col0[-2:] == [4.0, 5.0]

def test_remove_rows():
    model = PlotModel(initial_data=INITIAL_DATA)
    model.remove_rows([0])  # Remove first row
    assert model.num_rows == 2

    # Confirm that remaining raw data does not include removed row
    raw_col1 = model.get_raw_column(1)
    assert "cat" in raw_col1
    assert "dog" in raw_col1
    assert len(raw_col1) == 2

def test_get_normalized_column_returns_list_of_floats():
    model = PlotModel(initial_data=INITIAL_DATA)
    normalized = model.get_normalized_column(0)
    assert isinstance(normalized, list)
    assert all(isinstance(x, float) for x in normalized)

def test_get_raw_column_types():
    model = PlotModel(initial_data=INITIAL_DATA)
    # Numeric column
    raw_numeric = model.get_raw_column(0)
    assert all(isinstance(x, (int, float)) for x in raw_numeric)
    # Categorical column
    raw_categorical = model.get_raw_column(1)
    assert all(isinstance(x, str) for x in raw_categorical)

def test_set_and_get_custom_bounds():
    model = PlotModel(initial_data=INITIAL_DATA)
    # Set custom bounds on numeric column 0
    model.set_custom_bounds(0, min_val=0.0, max_val=10.0)
    bounds = model.get_custom_bounds(0)
    assert bounds == (0.0, 10.0)

    # Setting bounds on a categorical column should raise TypeError
    with pytest.raises(TypeError):
        model.set_custom_bounds(1, min_val=0.0)

    with pytest.raises(TypeError):
        model.get_custom_bounds(1)

def test_append_invalid_data_raises():
    model = PlotModel(initial_data=INITIAL_DATA)
    # Append row with wrong number of columns
    with pytest.raises(ValueError):
        model.append_rows([[1.0, "cat"]])  # only 2 columns, should be 3

    # Append row with wrong type in numeric column
    with pytest.raises(TypeError):
        model.append_rows([["not-a-number", "cat", 10]])

def test_remove_rows_invalid_index_raises():
    model = PlotModel(initial_data=INITIAL_DATA)
    with pytest.raises(IndexError):
        model.remove_rows([100])  # out of bounds

    with pytest.raises(TypeError):
        model.remove_rows(["not-an-int"]) # type: ignore
