"""Tests for Matrix."""

import pytest
from paxplot.structures.matrix import Matrix, ColumnType
from paxplot.structures.arrays.numerical_array import NumericalArray
from paxplot.structures.arrays.categorical_array import CategoricalArray


class TestMatrix:
    """Test cases for Matrix."""

    def test_init_with_valid_data(self):
        """Test initialization with valid data."""
        data = [[1, "A", 2.5], [2, "B", 3.0], [3, "A", 1.5]]
        matrix = Matrix(data)
        assert matrix.num_columns == 3
        assert matrix.num_rows == 3

    def test_init_with_empty_data_raises_error(self):
        """Test initialization with empty data raises error."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            Matrix([])

    def test_init_with_invalid_data_raises_error(self):
        """Test initialization with invalid data raises error."""
        with pytest.raises(ValueError, match="Data must be a 2D sequence"):
            Matrix([1, 2, 3])  # type: ignore

    def test_init_with_uneven_rows_raises_error(self):
        """Test initialization with uneven rows raises error."""
        data = [[1, "A", 2.5], [2, "B"], [3, "A", 1.5]]  # Missing third column
        with pytest.raises(ValueError, match="has length 2 but expected 3"):
            Matrix(data)

    def test_columns_property_returns_copy(self):
        """Test that columns property returns a copy."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        columns = matrix.columns
        assert len(columns) == 2
        # Modifying the returned list shouldn't affect the matrix
        columns.append(None)  # type: ignore
        assert len(matrix.columns) == 2

    def test_num_columns_property(self):
        """Test num_columns property."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        matrix = Matrix(data)
        assert matrix.num_columns == 3

    def test_num_rows_property(self):
        """Test num_rows property."""
        data = [[1, "A"], [2, "B"], [3, "C"]]
        matrix = Matrix(data)
        assert matrix.num_rows == 3

    def test_num_rows_empty_matrix(self):
        """Test num_rows property with empty matrix."""
        # Matrix doesn't allow empty data, so we need to test with minimal data
        data = [[1.0, "A"]]
        matrix = Matrix(data)
        matrix.remove_data([0])  # Remove the only row
        assert matrix.num_rows == 0

    def test_get_column(self):
        """Test get_column method."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        column = matrix.get_column(0)
        assert isinstance(column, NumericalArray)
        assert column.values == [1.0, 2.0]

    def test_get_column_out_of_bounds_raises_error(self):
        """Test get_column with out of bounds index raises error."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        with pytest.raises(IndexError, match="out of bounds"):
            matrix.get_column(5)

    def test_get_numeric_array(self):
        """Test get_numeric_array method."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        numeric_array = matrix.get_numeric_array(0)
        assert isinstance(numeric_array, NumericalArray)
        assert numeric_array.values == [1.0, 2.0]

    def test_get_numeric_array_wrong_type_raises_error(self):
        """Test get_numeric_array with wrong column type raises error."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        with pytest.raises(TypeError, match="not numerical"):
            matrix.get_numeric_array(1)

    def test_get_categorical_array(self):
        """Test get_categorical_array method."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        categorical_array = matrix.get_categorical_array(1)
        assert isinstance(categorical_array, CategoricalArray)
        assert categorical_array.values == ["A", "B"]

    def test_get_categorical_array_wrong_type_raises_error(self):
        """Test get_categorical_array with wrong column type raises error."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        with pytest.raises(TypeError, match="not categorical"):
            matrix.get_categorical_array(0)

    def test_get_column_type(self):
        """Test get_column_type method."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        assert matrix.get_column_type(0) == ColumnType.NUMERIC
        assert matrix.get_column_type(1) == ColumnType.CATEGORICAL

    def test_get_column_type_out_of_bounds_raises_error(self):
        """Test get_column_type with out of bounds index raises error."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        with pytest.raises(IndexError, match="out of bounds"):
            matrix.get_column_type(5)

    def test_append_data(self):
        """Test append_data method."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        matrix.append_data([3, "C"])
        assert matrix.num_rows == 3
        assert matrix.get_column(0).values == [1.0, 2.0, 3.0]
        assert matrix.get_column(1).values == ["A", "B", "C"]

    def test_append_data_wrong_length_raises_error(self):
        """Test append_data with wrong row length raises error."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        with pytest.raises(ValueError, match="doesn't match"):
            matrix.append_data([3])  # Missing second column

    def test_remove_data(self):
        """Test remove_data method."""
        data = [[1, "A"], [2, "B"], [3, "C"]]
        matrix = Matrix(data)
        matrix.remove_data([1])  # Remove second row
        assert matrix.num_rows == 2
        assert matrix.get_column(0).values == [1.0, 3.0]
        assert matrix.get_column(1).values == ["A", "C"]

    def test_remove_data_multiple_indices(self):
        """Test remove_data with multiple indices."""
        data = [[1, "A"], [2, "B"], [3, "C"], [4, "D"]]
        matrix = Matrix(data)
        matrix.remove_data([1, 3])  # Remove second and fourth rows
        assert matrix.num_rows == 2
        assert matrix.get_column(0).values == [1.0, 3.0]
        assert matrix.get_column(1).values == ["A", "C"]

    def test_len_operator(self):
        """Test len() operator."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        assert len(matrix) == 2

    def test_getitem_operator(self):
        """Test indexing operator."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        column = matrix[0]
        assert isinstance(column, NumericalArray)
        assert column.values == [1.0, 2.0]

    def test_repr(self):
        """Test string representation."""
        data = [[1, "A"], [2, "B"]]
        matrix = Matrix(data)
        assert repr(matrix) == "Matrix(2 rows, 2 columns)"

    def test_column_type_inference_numeric(self):
        """Test column type inference for numeric data."""
        data = [[1, "A"], [2.5, "B"], [3, "C"]]
        matrix = Matrix(data)
        assert matrix.get_column_type(0) == ColumnType.NUMERIC
        assert matrix.get_column_type(1) == ColumnType.CATEGORICAL

    def test_column_type_inference_categorical(self):
        """Test column type inference for categorical data."""
        data = [["A", 1], ["B", 2], ["C", 3]]
        matrix = Matrix(data)
        assert matrix.get_column_type(0) == ColumnType.CATEGORICAL
        assert matrix.get_column_type(1) == ColumnType.NUMERIC

    def test_column_type_inference_mixed_types_raises_error(self):
        """Test column type inference with mixed types raises error."""
        data = [[1, "A"], ["B", 2], [3, "C"]]  # First column has mixed types
        with pytest.raises(
            ValueError, match="mixed numeric and categorical data"
        ):
            Matrix(data)

    def test_column_type_inference_all_nan_numeric(self):
        """Test column type inference with all NaN values in numeric column."""
        data = [[None, "A"], [float("nan"), "B"], [None, "C"]]
        matrix = Matrix(data)
        assert (
            matrix.get_column_type(0) == ColumnType.CATEGORICAL
        )  # All NaN defaults to categorical
        assert matrix.get_column_type(1) == ColumnType.CATEGORICAL

    def test_column_type_inference_all_nan_categorical(self):
        """Test column type inference with all NaN values in categorical column."""
        data = [[1.0, None], [2.0, float("nan")], [3.0, None]]
        matrix = Matrix(data)
        assert matrix.get_column_type(0) == ColumnType.NUMERIC
        assert matrix.get_column_type(1) == ColumnType.CATEGORICAL

    def test_append_data_with_nan(self):
        """Test append_data with NaN values."""
        data = [[1.0, "A"], [2.0, "B"]]
        matrix = Matrix(data)
        matrix.append_data([float("nan"), None])  # type: ignore
        assert matrix.num_rows == 3
        # The arrays handle NaN validation, so we just check the row count

    def test_remove_data_with_nan(self):
        """Test remove_data with NaN values."""
        data = [[1.0, "A"], [float("nan"), "B"], [3.0, None]]
        matrix = Matrix(data)
        matrix.remove_data([1])  # Remove row with NaN
        assert matrix.num_rows == 2
        # The arrays handle NaN operations, so we just check the row count

    def test_infer_column_type_with_nan(self):
        """Test infer_column_type with NaN values."""
        data = [[1.0, "A"]]
        matrix = Matrix(data)

        # Column with numeric and NaN values should be numeric
        column_data = [1.0, float("nan"), 3.0, None]
        column_type = matrix.infer_column_type(column_data)
        assert column_type == ColumnType.NUMERIC

        # Column with string and NaN values should be categorical
        column_data = ["A", None, "B", float("nan")]
        column_type = matrix.infer_column_type(column_data)
        assert column_type == ColumnType.CATEGORICAL

    def test_infer_column_type_mixed_types_raises_error(self):
        """Test infer_column_type with mixed types raises error."""
        data = [[1.0, "A"]]
        matrix = Matrix(data)

        # Column with both numeric and string values (excluding NaN) should raise error
        column_data = [
            1.0,
            "A",
            3.0,
            None,
        ]  # NaN is ignored, but 1.0 and 'A' conflict
        with pytest.raises(
            ValueError, match="mixed numeric and categorical data"
        ):
            matrix.infer_column_type(column_data)

    def test_comprehensive_operations(self):
        """Test comprehensive operations."""
        # Create matrix with mixed values including NaN
        data = [
            [1.0, "A", 2.5],
            [float("nan"), "B", 3.0],
            [3.0, None, 1.5],
            [4.0, "C", float("nan")],
        ]
        matrix = Matrix(data)

        # Check basic properties
        assert matrix.num_columns == 3
        assert matrix.num_rows == 4

        # Test column operations
        numeric_col = matrix.get_numeric_array(0)
        assert numeric_col.has_nan is True
        assert numeric_col.nan_count == 1
        assert numeric_col.min == 1.0  # Ignores NaN
        assert numeric_col.max == 4.0  # Ignores NaN

        categorical_col = matrix.get_categorical_array(1)
        assert categorical_col.has_nan is True
        assert categorical_col.nan_count == 1
        # The order of unique values is based on appearance: A, B, <NaN>, C
        assert categorical_col.unique_values == ["A", "B", "<NaN>", "C"]

        # Remove rows with NaN
        matrix.remove_data([1, 2, 3])  # Remove all rows except first
        assert matrix.num_rows == 1
        assert numeric_col.nan_count == 0
        assert categorical_col.nan_count == 0

        # Add new data with NaN
        matrix.append_data([float("nan"), None, 5.0])  # type: ignore
        assert matrix.num_rows == 2


class TestMatrixNaNSupport:
    """Test cases for Matrix NaN support - verifying arrays handle NaN correctly."""

    def test_matrix_delegates_nan_handling_to_arrays(self):
        """Test that Matrix properly delegates NaN handling to underlying arrays."""
        data = [[1.0, "A", 2.5], [float("nan"), "B", 3.0], [3.0, None, 1.5]]
        matrix = Matrix(data)

        # Verify numerical array handles NaN correctly
        numeric_col = matrix.get_numeric_array(0)
        assert numeric_col.has_nan is True
        assert numeric_col.nan_count == 1
        assert numeric_col.nan_indices == [1]
        assert numeric_col.min == 1.0  # Ignores NaN
        assert numeric_col.max == 3.0  # Ignores NaN
        assert numeric_col.non_nan_values == [1.0, 3.0]

        # Verify categorical array handles NaN correctly
        categorical_col = matrix.get_categorical_array(1)
        assert categorical_col.has_nan is True
        assert categorical_col.nan_count == 1
        assert categorical_col.nan_indices == [2]
        assert categorical_col.unique_values == ["A", "B", "<NaN>"]
        assert categorical_col.non_nan_values == ["A", "B"]

    def test_matrix_append_preserves_nan_handling(self):
        """Test that Matrix append operations preserve array NaN handling."""
        data = [[1.0, "A"], [2.0, "B"]]
        matrix = Matrix(data)

        # Append NaN values
        matrix.append_data([float("nan"), None])  # type: ignore

        # Verify arrays still handle NaN correctly after append
        numeric_col = matrix.get_numeric_array(0)
        assert numeric_col.has_nan is True
        assert numeric_col.nan_count == 1
        assert numeric_col.nan_indices == [2]
        assert numeric_col.min == 1.0
        assert numeric_col.max == 2.0

        categorical_col = matrix.get_categorical_array(1)
        assert categorical_col.has_nan is True
        assert categorical_col.nan_count == 1
        assert categorical_col.nan_indices == [2]
        assert categorical_col.unique_values == ["A", "B", "<NaN>"]

    def test_matrix_remove_preserves_nan_handling(self):
        """Test that Matrix remove operations preserve array NaN handling."""
        data = [[1.0, "A"], [float("nan"), "B"], [3.0, None]]
        matrix = Matrix(data)

        # Remove row with NaN
        matrix.remove_data([1])

        # Verify arrays still handle NaN correctly after removal
        numeric_col = matrix.get_numeric_array(0)
        assert numeric_col.has_nan is False  # NaN was removed
        assert numeric_col.nan_count == 0
        assert numeric_col.min == 1.0
        assert numeric_col.max == 3.0

        categorical_col = matrix.get_categorical_array(1)
        assert categorical_col.has_nan is True  # Still has NaN from third row
        assert categorical_col.nan_count == 1
        assert categorical_col.nan_indices == [
            1
        ]  # Index shifted after removal
        assert categorical_col.unique_values == ["A", "<NaN>"]

    def test_matrix_handles_mixed_nan_types(self):
        """Test that Matrix handles different NaN representations correctly."""
        data = [
            [1.0, "A"],
            [None, "B"],  # None
            [float("nan"), "C"],  # float('nan')
        ]
        matrix = Matrix(data)

        # Verify all NaN types are handled consistently
        numeric_col = matrix.get_numeric_array(0)
        assert numeric_col.has_nan is True
        assert numeric_col.nan_count == 2
        assert numeric_col.nan_indices == [1, 2]
        assert numeric_col.min == 1.0
        assert numeric_col.max == 1.0

        categorical_col = matrix.get_categorical_array(1)
        assert categorical_col.has_nan is False  # No NaN in categorical column
        assert categorical_col.nan_count == 0
        assert categorical_col.unique_values == ["A", "B", "C"]

    def test_matrix_preserves_array_nan_state_cache(self):
        """Test that Matrix operations preserve the array NaN state caching."""
        data = [[1.0, "A"], [2.0, "B"]]
        matrix = Matrix(data)

        # Initially no NaN
        numeric_col = matrix.get_numeric_array(0)
        assert numeric_col.has_nan is False

        # Add NaN
        matrix.append_data([float("nan"), "C"])  # type: ignore
        assert numeric_col.has_nan is True

        # Remove NaN
        matrix.remove_data([2])
        assert numeric_col.has_nan is False

        # Add NaN again
        matrix.append_data([None, "D"])  # type: ignore
        assert numeric_col.has_nan is True
