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
    model = PlotModel(INITIAL_DATA)
    assert model.num_rows == 3
    assert model.num_columns == 3

def test_append_rows():
    model = PlotModel(INITIAL_DATA)
    new_rows = [
        [4.0, "dog", 40],
        [5.0, "mouse", 50],
    ]
    model.append_rows(new_rows)
    assert model.num_rows == 5
    raw_col0 = model.get_column_values(0)
    assert raw_col0[-2:] == [4.0, 5.0]

def test_remove_rows():
    model = PlotModel(INITIAL_DATA)
    model.remove_rows([0])
    assert model.num_rows == 2
    raw_col1 = model.get_column_values(1)
    assert "cat" in raw_col1 and "dog" in raw_col1
    assert len(raw_col1) == 2

def test_get_normalized_column_returns_list_of_floats():
    model = PlotModel(INITIAL_DATA)
    normalized = model.get_column_values_normalized(0)
    assert isinstance(normalized, list) or hasattr(normalized, "__array__")
    # Accept both list and numpy array for normalized output
    assert all(isinstance(x, float) for x in normalized)

def test_get_raw_column_types():
    model = PlotModel(INITIAL_DATA)
    raw_numeric = model.get_column_values(0)
    assert all(isinstance(x, (int, float)) for x in raw_numeric)
    raw_categorical = model.get_column_values(1)
    assert all(isinstance(x, str) for x in raw_categorical)

def test_set_and_get_custom_bounds():
    model = PlotModel(INITIAL_DATA)
    model.set_custom_bounds(0, min_val=0.0, max_val=10.0)
    # The PlotModel does not have get_custom_bounds, so access via the matrix directly
    effective_column_min = model.get_column_effective_min(0)
    effective_column_max = model.get_column_effective_max(0)
    assert (effective_column_min, effective_column_max) == (0.0, 10.0)

    with pytest.raises(TypeError):
        model.set_custom_bounds(1, min_val=0.0)

def test_append_invalid_data_raises():
    model = PlotModel(INITIAL_DATA)
    with pytest.raises(ValueError):
        model.append_rows([[1.0, "cat"]])
    with pytest.raises(TypeError):
        model.append_rows([["not-a-number", "cat", 10]])

def test_remove_rows_invalid_index_raises():
    model = PlotModel(INITIAL_DATA)
    with pytest.raises(IndexError):
        model.remove_rows([100])
    with pytest.raises(TypeError):
        model.remove_rows(["not-an-int"])  # type: ignore

def test_set_and_get_column_names():
    model = PlotModel(INITIAL_DATA)
    names = ["num1", "animal", "num2"]
    model.set_column_names(names)
    assert model.get_column_names() == names
    assert model.get_column_name(1) == "animal"

def test_set_column_name_and_get_it():
    model = PlotModel(INITIAL_DATA)
    names = ["num1", "animal", "num2"]
    model.set_column_names(names)
    model.set_column_name(2, "count")
    assert model.get_column_name(2) == "count"

def test_get_column_names_before_set_raises():
    model = PlotModel(INITIAL_DATA)
    with pytest.raises(RuntimeError):
        _ = model.get_column_names()

def test_get_column_name_before_set_raises():
    model = PlotModel(INITIAL_DATA)
    with pytest.raises(RuntimeError):
        _ = model.get_column_name(0)

def test_set_column_name_before_set_raises():
    model = PlotModel(INITIAL_DATA)
    with pytest.raises(RuntimeError):
        model.set_column_name(0, "newname")

def test_get_normalized_column_by_name():
    model = PlotModel(INITIAL_DATA)
    names = ["num1", "animal", "num2"]
    model.set_column_names(names)
    normalized = model.get_column_values_normalized_by_name("num1")
    assert isinstance(normalized, list) or hasattr(normalized, "__array__")
    assert all(isinstance(x, float) for x in normalized)

def test_get_raw_column_by_name():
    model = PlotModel(INITIAL_DATA)
    names = ["num1", "animal", "num2"]
    model.set_column_names(names)
    raw_numeric = model.get_column_values_by_name("num1")
    assert all(isinstance(x, (int, float)) for x in raw_numeric)
    raw_categorical = model.get_column_values_by_name("animal")
    assert all(isinstance(x, str) for x in raw_categorical)

def test_get_columns_by_name_before_set_raises():
    model = PlotModel(INITIAL_DATA)
    with pytest.raises(RuntimeError):
        model.get_column_values_normalized_by_name("num1")
    with pytest.raises(RuntimeError):
        model.get_column_values_by_name("animal")

def test_set_and_get_custom_bounds_by_name():
    model = PlotModel(INITIAL_DATA)
    names = ["num1", "animal", "num2"]
    model.set_column_names(names)
    model.set_custom_bounds_by_name("num1", min_val=1.0, max_val=100.0)
    # Ensure _named_view is initialized and not None before accessing its methods
    assert model._named_view is not None
    effective_column_min = model.get_column_effective_min_by_name("num1")
    effective_column_max = model.get_column_effective_max_by_name("num1")
    assert (effective_column_min, effective_column_max) == (1.0, 100.0)

def test_column_name_not_found_raises():
    model = PlotModel(INITIAL_DATA)
    names = ["num1", "animal", "num2"]
    model.set_column_names(names)
    with pytest.raises(KeyError):
        model.get_column_values_normalized_by_name("nonexistent")

    with pytest.raises(KeyError):
        model.get_column_values_by_name("nonexistent")

    with pytest.raises(KeyError):
        model.set_custom_bounds_by_name("nonexistent", min_val=0.0)

    # Removed test for non-existent get_custom_bounds_by_name method
