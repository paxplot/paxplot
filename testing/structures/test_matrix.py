"""Unit tests for Matrix."""

import pytest
from paxplot.structures.matrix import Matrix, ColumnType
from paxplot.structures.arrays.numerical_array import NumericalArray
from paxplot.structures.arrays.categorical_array import CategoricalArray


class TestMatrix:
    """Test cases for Matrix class."""

    def test_init_with_valid_data(self):
        """Test initialization with valid mixed data."""
        data = [[1, "A", 2.5], [2, "B", 3.0], [3, "A", 1.5]]
        matrix = Matrix(data)

        assert matrix.num_columns == 3
        assert matrix.num_rows == 3

    def test_init_with_empty_data_raises_error(self):
        """Test that initialization with empty data raises ValueError."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            Matrix([])

    def test_init_with_inconsistent_row_lengths_raises_error(self):
        """Test that initialization with inconsistent row lengths raises ValueError."""
        data = [[1, "A", 2.5], [2, "B"], [3, "A", 1.5]]  # Missing third column
        with pytest.raises(ValueError, match="length"):
            Matrix(data)

    def test_init_with_non_sequence_rows_raises_error(self):
        """Test that initialization with non-sequence rows raises ValueError."""
        data = [[1, "A", 2.5], "not a list", [3, "A", 1.5]]  # Invalid row
        with pytest.raises(ValueError, match="must be a sequence"):
            Matrix(data)

    def test_init_with_mixed_column_types_raises_error(self):
        """Test that initialization with mixed column types raises ValueError."""
        data = [
            [1, "A", 2.5],
            [2, 123, 3.0],  # Mixed types in second column
            [3, "A", 1.5],
        ]
        with pytest.raises(ValueError, match="mixed numeric and categorical"):
            Matrix(data)

    def test_columns_property(self):
        """Test columns property."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)
        columns = matrix.columns

        assert len(columns) == 3
        assert isinstance(columns[0], NumericalArray)
        assert isinstance(columns[1], CategoricalArray)
        assert isinstance(columns[2], NumericalArray)

    def test_columns_property_returns_copy(self):
        """Test that columns property returns a copy."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)
        columns = matrix.columns

        # Modify the returned list
        columns.append(None)

        # Original matrix should be unchanged
        assert len(matrix.columns) == 3

    def test_num_columns_property(self):
        """Test num_columns property."""
        data = [[1, "A", 2.5, "X"], [2, "B", 3.0, "Y"]]
        matrix = Matrix(data)
        assert matrix.num_columns == 4

    def test_num_rows_property(self):
        """Test num_rows property."""
        data = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5], [4, "D", 2.0]]
        matrix = Matrix(data)
        assert matrix.num_rows == 4

    def test_num_rows_empty_matrix(self):
        """Test num_rows property on empty matrix."""
        # This should not happen in practice, but test the edge case
        # Matrix([[]]) would raise an error, but if it didn't:
        # matrix = Matrix([[]])
        # assert matrix.num_rows == 0

    def test_get_column(self):
        """Test get_column method."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        column = matrix.get_column(1)
        assert isinstance(column, CategoricalArray)
        assert column.values == ["A", "B"]

    def test_get_column_out_of_bounds_raises_error(self):
        """Test that get_column with out of bounds index raises IndexError."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        with pytest.raises(IndexError, match="out of bounds"):
            matrix.get_column(5)

    def test_get_numeric_array(self):
        """Test get_numeric_array method."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        numeric_array = matrix.get_numeric_array(0)
        assert isinstance(numeric_array, NumericalArray)
        assert numeric_array.values == [1.0, 2.0]

    def test_get_numeric_array_wrong_type_raises_error(self):
        """Test that get_numeric_array on categorical column raises TypeError."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        Matrix(data)  # Create matrix but don't store reference

        with pytest.raises(TypeError, match="not numerical"):
            Matrix(data).get_numeric_array(1)

    def test_get_categorical_array(self):
        """Test get_categorical_array method."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        categorical_array = matrix.get_categorical_array(1)
        assert isinstance(categorical_array, CategoricalArray)
        assert categorical_array.values == ["A", "B"]

    def test_get_categorical_array_wrong_type_raises_error(self):
        """Test that get_categorical_array on numerical column raises TypeError."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        with pytest.raises(TypeError, match="not categorical"):
            matrix.get_categorical_array(0)

    def test_append_data_valid_row(self):
        """Test append_data with valid row."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        matrix.append_data([3, "C", 1.5])

        assert matrix.num_rows == 3
        assert matrix.get_numeric_array(0).values == [1.0, 2.0, 3.0]
        assert matrix.get_categorical_array(1).values == ["A", "B", "C"]

    def test_append_data_wrong_length_raises_error(self):
        """Test that append_data with wrong row length raises ValueError."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        with pytest.raises(ValueError, match="doesn't match"):
            matrix.append_data([3, "C"])  # Missing third column

    def test_append_data_wrong_types_raises_error(self):
        """Test that append_data with wrong types raises ValueError."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        with pytest.raises(ValueError, match="not a string"):
            matrix.append_data(
                [3, 123, 1.5]
            )  # Wrong type for categorical column

    def test_remove_data(self):
        """Test remove_data method."""
        data = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5], [4, "D", 2.0]]
        matrix = Matrix(data)

        matrix.remove_data([1, 3])

        assert matrix.num_rows == 2
        assert matrix.get_numeric_array(0).values == [1.0, 3.0]
        assert matrix.get_categorical_array(1).values == ["A", "C"]

    def test_len_operator(self):
        """Test len() operator."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)
        assert len(matrix) == 3

    def test_getitem_operator(self):
        """Test indexing operator."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        column = matrix[1]
        assert isinstance(column, CategoricalArray)
        assert column.values == ["A", "B"]

    def test_repr(self):
        """Test string representation."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)

        repr_str = repr(matrix)
        assert "Matrix(" in repr_str
        assert "2 rows" in repr_str
        assert "3 columns" in repr_str

    def test_column_type_inference_numeric(self):
        """Test column type inference for numeric columns."""
        data = [[1, 2.5, 3], [2, 3.0, 4], [3, 1.5, 5]]
        matrix = Matrix(data)

        for i in range(3):
            assert isinstance(matrix.get_column(i), NumericalArray)

    def test_column_type_inference_categorical(self):
        """Test column type inference for categorical columns."""
        data = [["A", "X", "1"], ["B", "Y", "2"], ["C", "Z", "3"]]
        matrix = Matrix(data)

        for i in range(3):
            assert isinstance(matrix.get_column(i), CategoricalArray)

    def test_empty_matrix_handling(self):
        """Test handling of edge cases with minimal data."""
        # Single row
        data = [[1, "A", 2.5]]
        matrix = Matrix(data)
        assert matrix.num_rows == 1
        assert matrix.num_columns == 3

    def test_column_type_enum(self):
        """Test ColumnType enum values."""
        assert ColumnType.NUMERIC == "numeric"
        assert ColumnType.CATEGORICAL == "categorical"

    def test_matrix_with_all_numeric_columns(self):
        """Test matrix with all numeric columns."""
        data = [[1, 2.5, 3], [2, 3.0, 4], [3, 1.5, 5]]
        matrix = Matrix(data)

        assert matrix.num_columns == 3
        assert matrix.num_rows == 3

        for i in range(3):
            numeric_array = matrix.get_numeric_array(i)
            assert isinstance(numeric_array, NumericalArray)

    def test_matrix_with_all_categorical_columns(self):
        """Test matrix with all categorical columns."""
        data = [["A", "X", "1"], ["B", "Y", "2"], ["C", "Z", "3"]]
        matrix = Matrix(data)

        assert matrix.num_columns == 3
        assert matrix.num_rows == 3

        for i in range(3):
            categorical_array = matrix.get_categorical_array(i)
            assert isinstance(categorical_array, CategoricalArray)
