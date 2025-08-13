# pylint: disable=C0116, W0212, W0621
"""Tests for NamedNormalizedMatrixView"""

import pytest
from paxplot.data_managers.normalized_matrix import NormalizedMatrix, ColumnType
from paxplot.data_managers.named_normalized_matrix_view import NamedNormalizedMatrixView

@pytest.fixture
def sample_normalized_matrix():
    # Create a matrix with 3 columns: numeric, numeric, categorical
    data = [
        [1.0, 10, "a"],
        [2.0, 20, "b"],
        [3.0, 30, "c"],
    ]
    matrix = NormalizedMatrix(data=data)
    return matrix

def test_init_valid(sample_normalized_matrix):
    names = ["col1", "col2", "col3"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)
    assert view.column_names == names
    assert view.matrix is sample_normalized_matrix

def test_init_errors(sample_normalized_matrix):
    # Non-string column names
    with pytest.raises(TypeError):
        NamedNormalizedMatrixView(sample_normalized_matrix, [1, 2, 3]) # type: ignore

    # Mismatched number of names
    with pytest.raises(ValueError):
        NamedNormalizedMatrixView(sample_normalized_matrix, ["col1", "col2"])

    # Duplicate names
    with pytest.raises(ValueError):
        NamedNormalizedMatrixView(sample_normalized_matrix, ["col1", "col2", "col1"])

def test_get_index_and_keyerror(sample_normalized_matrix):
    names = ["a", "b", "c"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)
    assert view._get_index("b") == 1
    with pytest.raises(KeyError):
        view._get_index("nonexistent")

def test_column_name_get_set(sample_normalized_matrix):
    names = ["x", "y", "z"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)
    assert view.get_column_name(2) == "z"
    view.set_column_name(2, "w")
    assert view.get_column_name(2) == "w"
    assert view._get_index("w") == 2
    with pytest.raises(IndexError):
        view.get_column_name(10)
    with pytest.raises(IndexError):
        view.set_column_name(10, "bad")
    with pytest.raises(ValueError):
        view.set_column_name(1, "w")  # duplicate name

def test_set_column_names(sample_normalized_matrix):
    names = ["a", "b", "c"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)
    new_names = ["x", "y", "z"]
    view.set_column_names(new_names)
    assert view.column_names == new_names
    with pytest.raises(TypeError):
        view.set_column_names([1, 2, 3])  # type: ignore
    with pytest.raises(ValueError):
        view.set_column_names(["a", "b"])
    with pytest.raises(ValueError):
        view.set_column_names(["a", "a", "b"])

def test_getitem_and_column_types(sample_normalized_matrix):
    names = ["num1", "num2", "cat"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)
    # __getitem__ returns the correct array
    arr = view["num1"]
    assert arr is sample_normalized_matrix[0]
    # get_numeric_column returns correct type
    num_col = view.get_numeric_column("num2")
    assert num_col is sample_normalized_matrix.get_numeric_column(1)
    # get_categorical_column returns correct type
    cat_col = view.get_categorical_column("cat")
    assert cat_col is sample_normalized_matrix.get_categorical_column(2)
    # get_column_type matches underlying matrix
    assert view.get_column_type("cat") == ColumnType.CATEGORICAL

def test_num_columns_and_rows(sample_normalized_matrix):
    names = ["a", "b", "c"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)
    assert view.num_columns == 3
    assert view.num_rows == 3

def test_append_data_valid(sample_normalized_matrix):
    names = ["x", "y", "z"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)

    new_rows = [
        [4.0, 40, "d"],
        [5.0, 50, "e"],
    ]
    view.append_data(new_rows)

    # The underlying matrix row count increases
    assert sample_normalized_matrix.num_rows == 5

    # Check last row matches appended data
    assert sample_normalized_matrix[0][-1] == 5.0
    assert sample_normalized_matrix[2][-1] == "e"

def test_append_data_invalid_column_count(sample_normalized_matrix):
    names = ["x", "y", "z"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)

    # New row with wrong number of columns
    with pytest.raises(ValueError):
        view.append_data([[1.0, 2.0]])  # only 2 columns, expected 3

def test_append_data_invalid_type(sample_normalized_matrix):
    names = ["x", "y", "z"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)

    # Numeric column gets string
    with pytest.raises(TypeError):
        view.append_data([[1.0, "not numeric", "a"]])

    # Categorical column gets numeric
    with pytest.raises(TypeError):
        view.append_data([[1.0, 20, 123]])

def test_append_empty_data_no_change(sample_normalized_matrix):
    names = ["x", "y", "z"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)

    original_rows = sample_normalized_matrix.num_rows
    view.append_data([])
    assert sample_normalized_matrix.num_rows == original_rows

def test_remove_rows(sample_normalized_matrix):
    names = ["a", "b", "c"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)

    # Remove second row
    view.remove_rows([1])

    assert sample_normalized_matrix.num_rows == 2
    # Check that the removed row is gone
    assert sample_normalized_matrix[0].values == [1.0, 3.0]
    assert sample_normalized_matrix[2].values == ["a", "c"]

def test_remove_rows_invalid_indices(sample_normalized_matrix):
    names = ["a", "b", "c"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)

    # Non-integer indices
    with pytest.raises(TypeError):
        view.remove_rows(["1"]) # type: ignore

    # Out-of-bounds index
    with pytest.raises(IndexError):
        view.remove_rows([100])

def test_remove_rows_empty_no_change(sample_normalized_matrix):
    names = ["a", "b", "c"]
    view = NamedNormalizedMatrixView(sample_normalized_matrix, names)

    original_rows = sample_normalized_matrix.num_rows
    view.remove_rows([])
    assert sample_normalized_matrix.num_rows == original_rows
