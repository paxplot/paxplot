"""Tests for ArrayManager."""

import pytest
from paxplot.structures.array_manager import ArrayManager, ArrayType
from paxplot.structures.arrays.numerical_array import NumericalArray
from paxplot.structures.arrays.categorical_array import CategoricalArray


class TestArrayManager:
    """Test cases for ArrayManager."""

    def test_init_with_valid_arrays(self):
        """Test initialization with valid arrays."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(["A", "B", "A"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        assert array_manager.num_arrays == 2

    def test_init_with_empty_arrays(self):
        """Test initialization with empty arrays."""
        array_manager = ArrayManager([])
        assert array_manager.num_arrays == 0

    def test_init_with_none_arrays(self):
        """Test initialization with None arrays."""
        array_manager = ArrayManager(None)
        assert array_manager.num_arrays == 0

    def test_init_with_invalid_array_raises_error(self):
        """Test initialization with invalid array raises error."""
        with pytest.raises(
            TypeError,
            match="Array at index 0 must be NumericalArray or CategoricalArray",
        ):
            ArrayManager(["invalid"])  # type: ignore

    def test_arrays_property_returns_copy(self):
        """Test that arrays property returns a copy."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        arrays = array_manager.arrays
        assert len(arrays) == 2
        # Modifying the returned list shouldn't affect the array manager
        arrays.append(None)  # type: ignore
        assert len(array_manager.arrays) == 2

    def test_num_arrays_property(self):
        """Test num_arrays property."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(["A", "B", "C"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        assert array_manager.num_arrays == 2

    def test_get_array(self):
        """Test get_array method."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        array = array_manager.get_array(0)
        assert isinstance(array, NumericalArray)
        assert array.get_values() == [1.0, 2.0]

    def test_get_array_out_of_bounds_raises_error(self):
        """Test get_array with out of bounds index raises error."""
        numeric_array = NumericalArray([1, 2])
        array_manager = ArrayManager([numeric_array])
        with pytest.raises(IndexError, match="Array index 5 out of bounds"):
            array_manager.get_array(5)

    def test_get_numeric_array(self):
        """Test get_numeric_array method."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        numeric_array_result = array_manager.get_numeric_array(0)
        assert isinstance(numeric_array_result, NumericalArray)
        assert numeric_array_result.get_values() == [1.0, 2.0]

    def test_get_numeric_array_wrong_type_raises_error(self):
        """Test get_numeric_array with wrong array type raises error."""
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([categorical_array])
        with pytest.raises(TypeError, match="Array 0 is not numerical"):
            array_manager.get_numeric_array(0)

    def test_get_categorical_array(self):
        """Test get_categorical_array method."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        categorical_array_result = array_manager.get_categorical_array(1)
        assert isinstance(categorical_array_result, CategoricalArray)
        assert categorical_array_result.get_values() == ["A", "B"]

    def test_get_categorical_array_wrong_type_raises_error(self):
        """Test get_categorical_array with wrong array type raises error."""
        numeric_array = NumericalArray([1, 2])
        array_manager = ArrayManager([numeric_array])
        with pytest.raises(TypeError, match="Array 0 is not categorical"):
            array_manager.get_categorical_array(0)

    def test_get_array_type(self):
        """Test get_array_type method."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        assert array_manager.get_array_type(0) == ArrayType.NUMERIC
        assert array_manager.get_array_type(1) == ArrayType.CATEGORICAL

    def test_get_array_type_out_of_bounds_raises_error(self):
        """Test get_array_type with out of bounds index raises error."""
        numeric_array = NumericalArray([1, 2])
        array_manager = ArrayManager([numeric_array])
        with pytest.raises(IndexError, match="Array index 5 out of bounds"):
            array_manager.get_array_type(5)

    def test_append_values(self):
        """Test append_values method."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        array_manager.append_values([3, "C"])
        assert array_manager.get_array(0).get_values() == [1.0, 2.0, 3.0]
        assert array_manager.get_array(1).get_values() == ["A", "B", "C"]

    def test_append_values_wrong_length_raises_error(self):
        """Test append_values with wrong row length raises error."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        with pytest.raises(
            ValueError, match="Row length 1 doesn't match number of arrays 2"
        ):
            array_manager.append_values([3])  # Missing second value

    def test_remove_values(self):
        """Test remove_values method."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(["A", "B", "C"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        array_manager.remove_values([1])  # Remove second row
        assert array_manager.get_array(0).get_values() == [1.0, 3.0]
        assert array_manager.get_array(1).get_values() == ["A", "C"]

    def test_remove_values_multiple_indices(self):
        """Test remove_values with multiple indices."""
        numeric_array = NumericalArray([1, 2, 3, 4])
        categorical_array = CategoricalArray(["A", "B", "C", "D"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        array_manager.remove_values([1, 3])  # Remove second and fourth rows
        assert array_manager.get_array(0).get_values() == [1.0, 3.0]
        assert array_manager.get_array(1).get_values() == ["A", "C"]

    def test_len_operator(self):
        """Test len() operator."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        assert len(array_manager) == 2

    def test_getitem_operator(self):
        """Test indexing operator."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        array = array_manager[0]
        assert isinstance(array, NumericalArray)
        assert array.get_values() == [1.0, 2.0]

    def test_repr(self):
        """Test string representation."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        assert repr(array_manager) == "ArrayManager(2 arrays)"

    def test_infer_array_type_numeric(self):
        """Test array type inference for numeric data."""
        array_manager = ArrayManager()
        array_type = array_manager.infer_array_type([1, 2.5, 3])
        assert array_type == ArrayType.NUMERIC

    def test_infer_array_type_categorical(self):
        """Test array type inference for categorical data."""
        array_manager = ArrayManager()
        array_type = array_manager.infer_array_type(["A", "B", "C"])
        assert array_type == ArrayType.CATEGORICAL

    def test_infer_array_type_mixed_types_raises_error(self):
        """Test array type inference with mixed types raises error."""
        array_manager = ArrayManager()
        with pytest.raises(
            ValueError, match="Array contains mixed numeric and categorical values"
        ):
            array_manager.infer_array_type([1, "A", 3])

    def test_infer_array_type_all_nan_numeric(self):
        """Test array type inference with all NaN values in numeric column."""
        array_manager = ArrayManager()
        # Use proper typing for None values
        array_type = array_manager.infer_array_type([None, float("nan"), None])  # type: ignore
        assert array_type == ArrayType.CATEGORICAL  # All NaN defaults to categorical

    def test_infer_array_type_all_nan_categorical(self):
        """Test array type inference with all NaN values in categorical column."""
        array_manager = ArrayManager()
        # Use proper typing for None values
        array_type = array_manager.infer_array_type([1.0, None, 3.0])  # type: ignore
        assert array_type == ArrayType.NUMERIC

    def test_append_values_with_nan(self):
        """Test append_values with NaN values."""
        numeric_array = NumericalArray([1.0, 2.0])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        array_manager.append_values([float("nan"), None])  # type: ignore
        # The arrays handle NaN validation, so we just check the values were added
        assert len(array_manager.get_array(0)) == 3
        assert len(array_manager.get_array(1)) == 3

    def test_remove_values_with_nan(self):
        """Test remove_values with NaN values."""
        numeric_array = NumericalArray([1.0, float("nan"), 3.0])
        categorical_array = CategoricalArray(["A", "B", None])  # type: ignore
        array_manager = ArrayManager([numeric_array, categorical_array])
        array_manager.remove_values([1])  # Remove row with NaN
        # The arrays handle NaN operations, so we just check the values were removed
        assert len(array_manager.get_array(0)) == 2
        assert len(array_manager.get_array(1)) == 2

    def test_infer_array_type_with_nan(self):
        """Test infer_array_type with NaN values."""
        array_manager = ArrayManager()

        # Array with numeric and NaN values should be numeric
        array_data = [1.0, float("nan"), 3.0, None]
        array_type = array_manager.infer_array_type(array_data)
        assert array_type == ArrayType.NUMERIC

        # Array with string and NaN values should be categorical
        array_data = ["A", None, "B", float("nan")]  # type: ignore
        array_type = array_manager.infer_array_type(array_data)
        assert array_type == ArrayType.CATEGORICAL

    def test_comprehensive_operations(self):
        """Test comprehensive operations."""
        # Create arrays with mixed values including NaN
        numeric_array = NumericalArray([1.0, float("nan"), 3.0, 4.0])
        categorical_array = CategoricalArray(["A", "B", None, "C"])  # type: ignore
        numeric_array2 = NumericalArray([2.5, 3.0, 1.5, float("nan")])
        array_manager = ArrayManager([numeric_array, categorical_array, numeric_array2])

        # Check basic properties
        assert array_manager.num_arrays == 3

        # Test array operations
        numeric_col = array_manager.get_numeric_array(0)
        assert numeric_col.has_nan is True
        assert numeric_col.nan_count == 1
        assert numeric_col.min == 1.0  # Ignores NaN
        assert numeric_col.max == 4.0  # Ignores NaN

        categorical_col = array_manager.get_categorical_array(1)
        assert categorical_col.has_nan is True
        assert categorical_col.nan_count == 1
        # The order of unique values is based on appearance: A, B, <NaN>, C
        assert categorical_col.unique_values == ["A", "B", "<NaN>", "C"]

        # Remove rows with NaN
        array_manager.remove_values([1, 2, 3])  # Remove all rows except first
        assert len(array_manager.get_array(0)) == 1
        assert numeric_col.nan_count == 0
        assert categorical_col.nan_count == 0

        # Add new data with NaN
        array_manager.append_values([float("nan"), None, 5.0])  # type: ignore
        assert len(array_manager.get_array(0)) == 2


class TestArrayManagerNaNSupport:
    """Test cases for ArrayManager NaN support - verifying arrays handle NaN correctly."""

    def test_array_manager_delegates_nan_handling_to_arrays(self):
        """Test that ArrayManager properly delegates NaN handling to underlying arrays."""
        numeric_array = NumericalArray([1.0, float("nan"), 3.0])
        categorical_array = CategoricalArray(["A", "B", None])  # type: ignore
        numeric_array2 = NumericalArray([2.5, 3.0, 1.5])
        array_manager = ArrayManager([numeric_array, categorical_array, numeric_array2])

        # Verify numerical array handles NaN correctly
        numeric_col = array_manager.get_numeric_array(0)
        assert numeric_col.has_nan is True
        assert numeric_col.nan_count == 1
        assert numeric_col.nan_indices == [1]
        assert numeric_col.min == 1.0  # Ignores NaN
        assert numeric_col.max == 3.0  # Ignores NaN
        assert numeric_col.non_nan_values == [1.0, 3.0]

        # Verify categorical array handles NaN correctly
        categorical_col = array_manager.get_categorical_array(1)
        assert categorical_col.has_nan is True
        assert categorical_col.nan_count == 1
        assert categorical_col.nan_indices == [2]
        assert categorical_col.unique_values == ["A", "B", "<NaN>"]
        assert categorical_col.non_nan_values == ["A", "B"]

    def test_array_manager_append_preserves_nan_handling(self):
        """Test that ArrayManager append operations preserve array NaN handling."""
        numeric_array = NumericalArray([1.0, 2.0])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])

        # Append NaN values
        array_manager.append_values([float("nan"), None])  # type: ignore

        # Verify arrays still handle NaN correctly after append
        numeric_col = array_manager.get_numeric_array(0)
        assert numeric_col.has_nan is True
        assert numeric_col.nan_count == 1
        assert numeric_col.nan_indices == [2]
        assert numeric_col.min == 1.0
        assert numeric_col.max == 2.0

        categorical_col = array_manager.get_categorical_array(1)
        assert categorical_col.has_nan is True
        assert categorical_col.nan_count == 1
        assert categorical_col.nan_indices == [2]
        assert categorical_col.unique_values == ["A", "B", "<NaN>"]

    def test_array_manager_remove_preserves_nan_handling(self):
        """Test that ArrayManager remove operations preserve array NaN handling."""
        numeric_array = NumericalArray([1.0, float("nan"), 3.0])
        categorical_array = CategoricalArray(["A", "B", None])  # type: ignore
        array_manager = ArrayManager([numeric_array, categorical_array])

        # Remove row with NaN
        array_manager.remove_values([1])

        # Verify arrays still handle NaN correctly after removal
        numeric_col = array_manager.get_numeric_array(0)
        assert numeric_col.has_nan is False  # NaN was removed
        assert numeric_col.nan_count == 0
        assert numeric_col.min == 1.0
        assert numeric_col.max == 3.0

        categorical_col = array_manager.get_categorical_array(1)
        assert categorical_col.has_nan is True  # Still has NaN from third row
        assert categorical_col.nan_count == 1
        assert categorical_col.nan_indices == [1]  # Index shifted after removal
        assert categorical_col.unique_values == ["A", "<NaN>"]

    def test_array_manager_handles_mixed_nan_types(self):
        """Test that ArrayManager handles different NaN representations correctly."""
        numeric_array = NumericalArray([1.0, None, float("nan")])
        categorical_array = CategoricalArray(["A", "B", "C"])
        array_manager = ArrayManager([numeric_array, categorical_array])

        # Verify all NaN types are handled consistently
        numeric_col = array_manager.get_numeric_array(0)
        assert numeric_col.has_nan is True
        assert numeric_col.nan_count == 2
        assert numeric_col.nan_indices == [1, 2]
        assert numeric_col.min == 1.0
        assert numeric_col.max == 1.0

        categorical_col = array_manager.get_categorical_array(1)
        assert categorical_col.has_nan is False  # No NaN in categorical column
        assert categorical_col.nan_count == 0
        assert categorical_col.unique_values == ["A", "B", "C"]

    def test_array_manager_preserves_array_nan_state_cache(self):
        """Test that ArrayManager operations preserve the array NaN state caching."""
        numeric_array = NumericalArray([1.0, 2.0])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])

        # Initially no NaN
        numeric_col = array_manager.get_numeric_array(0)
        assert numeric_col.has_nan is False

        # Add NaN
        array_manager.append_values([float("nan"), "C"])  # type: ignore
        assert numeric_col.has_nan is True

        # Remove NaN
        array_manager.remove_values([2])
        assert numeric_col.has_nan is False

        # Add NaN again
        array_manager.append_values([None, "D"])  # type: ignore
        assert numeric_col.has_nan is True

    def test_set_arrays_from_types(self):
        """Test setting arrays from types."""
        array_manager = ArrayManager()
        array_manager.set_arrays_from_types([ArrayType.NUMERIC, ArrayType.CATEGORICAL])

        assert array_manager.num_arrays == 2
        assert isinstance(array_manager.get_array(0), NumericalArray)
        assert isinstance(array_manager.get_array(1), CategoricalArray)

    def test_set_arrays_from_types_empty_raises_error(self):
        """Test setting arrays from empty types raises error."""
        array_manager = ArrayManager()
        with pytest.raises(ValueError, match="array_types cannot be empty"):
            array_manager.set_arrays_from_types([])

    def test_set_arrays_from_types_invalid_type_raises_error(self):
        """Test setting arrays from invalid type raises error."""
        array_manager = ArrayManager()
        with pytest.raises(
            TypeError, match="array_type at index 0 must be an ArrayType"
        ):
            array_manager.set_arrays_from_types(["invalid"])  # type: ignore

    def test_append_arrays(self):
        """Test appending arrays."""
        numeric_array = NumericalArray([1, 2])
        array_manager = ArrayManager([numeric_array])
        assert array_manager.num_arrays == 1

        categorical_array = CategoricalArray(["A", "B"])
        array_manager.append_arrays(categorical_array)
        assert array_manager.num_arrays == 2
        assert isinstance(array_manager.get_array(1), CategoricalArray)

    def test_append_arrays_invalid_type_raises_error(self):
        """Test appending invalid array type raises error."""
        array_manager = ArrayManager()
        with pytest.raises(
            TypeError, match="Array must be NumericalArray or CategoricalArray"
        ):
            array_manager.append_arrays("invalid")  # type: ignore

    def test_remove_arrays(self):
        """Test removing arrays."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        assert array_manager.num_arrays == 2

        array_manager.remove_arrays([1])
        assert array_manager.num_arrays == 1
        assert isinstance(array_manager.get_array(0), NumericalArray)

    def test_clear_arrays(self):
        """Test clearing all arrays."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])
        assert array_manager.num_arrays == 2

        array_manager.clear_arrays()
        assert array_manager.num_arrays == 0

    def test_set_values(self):
        """Test setting values from 2D data."""
        array_manager = ArrayManager()
        values = [[1, "A"], [2, "B"], [3, "C"]]
        array_manager.set_values(values)

        assert array_manager.num_arrays == 2
        assert isinstance(array_manager.get_array(0), NumericalArray)
        assert isinstance(array_manager.get_array(1), CategoricalArray)
        assert array_manager.get_array(0).get_values() == [1.0, 2.0, 3.0]
        assert array_manager.get_array(1).get_values() == ["A", "B", "C"]

    def test_set_values_empty_raises_error(self):
        """Test setting empty values raises error."""
        array_manager = ArrayManager()
        with pytest.raises(ValueError, match="Values cannot be empty"):
            array_manager.set_values([])

    def test_set_values_invalid_structure_raises_error(self):
        """Test setting invalid values structure raises error."""
        array_manager = ArrayManager()
        with pytest.raises(ValueError, match="Values must be a 2D sequence"):
            array_manager.set_values([1, 2, 3])  # type: ignore

    def test_set_values_uneven_rows_raises_error(self):
        """Test setting values with uneven rows raises error."""
        array_manager = ArrayManager()
        with pytest.raises(ValueError, match="Row 1 has length 1 but expected 2"):
            array_manager.set_values([[1, "A"], [2]])  # Missing second column

    def test_clear_values(self):
        """Test clearing values from arrays."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])
        array_manager = ArrayManager([numeric_array, categorical_array])

        array_manager.clear_values()
        assert len(array_manager.get_array(0)) == 0
        assert len(array_manager.get_array(1)) == 0
