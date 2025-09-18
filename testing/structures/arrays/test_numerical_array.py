"""Unit tests for NumericalArray."""

import pytest
from paxplot.structures.arrays.numerical_array import NumericalArray


class TestNumericalArray:
    """Test cases for NumericalArray class."""

    def test_init_with_valid_values(self):
        """Test initialization with valid numerical values."""
        values = [1, 2.5, 3, 4.0, 5]
        array = NumericalArray(values)

        assert array.get_values() == [1.0, 2.5, 3.0, 4.0, 5.0]
        assert array.length == 5

    def test_init_with_empty_list(self):
        """Test initialization with empty list."""
        array = NumericalArray([])

        assert not array.get_values()
        assert array.length == 0

    def test_init_with_none_creates_empty_array(self):
        """Test that initialization with None creates an empty array."""
        array = NumericalArray(None)  # type: ignore
        assert not array.get_values()
        assert array.length == 0

    def test_init_with_mixed_types_raises_error(self):
        """Test that initialization with mixed types raises ValueError."""
        with pytest.raises(ValueError, match="must be numerical"):
            NumericalArray([1, "string", 3])

    def test_init_with_none_values_converts_to_nan(self):
        """Test that initialization with None values converts them to NaN."""
        array = NumericalArray([1, None, 3])  # type: ignore
        assert array.get_values()[0] == 1.0
        assert array.get_values()[2] == 3.0
        assert str(array.get_values()[1]) == "nan"  # NaN doesn't equal itself
        assert array.has_nan is True
        assert array.nan_count == 1
        assert array.nan_indices == [1]

    def test_min_property(self):
        """Test min property."""
        array = NumericalArray([5, 2, 8, 1, 9])
        assert array.min == 1.0

    def test_max_property(self):
        """Test max property."""
        array = NumericalArray([5, 2, 8, 1, 9])
        assert array.max == 9.0

    def test_min_empty_array_raises_error(self):
        """Test that min on empty array raises ValueError."""
        array = NumericalArray([])
        with pytest.raises(
            ValueError, match="Cannot compute min of empty array"
        ):
            _ = array.min

    def test_max_empty_array_raises_error(self):
        """Test that max on empty array raises ValueError."""
        array = NumericalArray([])
        with pytest.raises(
            ValueError, match="Cannot compute max of empty array"
        ):
            _ = array.max

    def test_append_valid_values(self):
        """Test appending valid numerical values."""
        array = NumericalArray([1, 2, 3])
        array.append_values([4, 5])

        assert array.get_values() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert array.length == 5

    def test_append_empty_list(self):
        """Test appending empty list."""
        array = NumericalArray([1, 2, 3])
        array.append_values([])

        assert array.get_values() == [1.0, 2.0, 3.0]
        assert array.length == 3

    def test_append_invalid_values_raises_error(self):
        """Test that appending invalid values raises ValueError."""
        array = NumericalArray([1, 2, 3])
        with pytest.raises(ValueError, match="must be numerical"):
            array.append_values([4, "string", 6])

    def test_remove_valid_indices(self):
        """Test removing values at valid indices."""
        array = NumericalArray([1, 2, 3, 4, 5])
        array.remove_values([1, 3])

        assert array.get_values() == [1.0, 3.0, 5.0]
        assert array.length == 3

    def test_remove_empty_indices(self):
        """Test removing with empty indices list."""
        array = NumericalArray([1, 2, 3])
        array.remove_values([])

        assert array.get_values() == [1.0, 2.0, 3.0]
        assert array.length == 3

    def test_remove_out_of_bounds_index_raises_error(self):
        """Test that removing out of bounds index raises IndexError."""
        array = NumericalArray([1, 2, 3])
        with pytest.raises(IndexError, match="out of bounds"):
            array.remove_values([5])

    def test_remove_negative_index_raises_error(self):
        """Test that removing negative index raises IndexError."""
        array = NumericalArray([1, 2, 3])
        with pytest.raises(IndexError, match="out of bounds"):
            array.remove_values([-1])

    def test_remove_invalid_index_type_raises_error(self):
        """Test that removing with invalid index type raises ValueError."""
        array = NumericalArray([1, 2, 3])
        with pytest.raises(ValueError, match="must be an integer"):
            array.remove_values(["invalid"])  # type: ignore

    def test_len_operator(self):
        """Test len() operator."""
        array = NumericalArray([1, 2, 3, 4, 5])
        assert len(array) == 5

    def test_getitem_operator(self):
        """Test indexing operator."""
        array = NumericalArray([1, 2, 3, 4, 5])
        assert array[0] == 1.0
        assert array[2] == 3.0
        assert array[-1] == 5.0

    def test_getitem_out_of_bounds_raises_error(self):
        """Test that indexing out of bounds raises IndexError."""
        array = NumericalArray([1, 2, 3])
        with pytest.raises(IndexError):
            _ = array[5]

    def test_repr_small_array(self):
        """Test string representation for small array."""
        array = NumericalArray([1, 2, 3])
        assert repr(array) == "NumericalArray([1.0, 2.0, 3.0])"

    def test_repr_large_array(self):
        """Test string representation for large array."""
        array = NumericalArray(list(range(15)))
        repr_str = repr(array)
        assert "NumericalArray(" in repr_str
        assert "..." in repr_str
        assert repr_str.endswith("])")

    def test_values_property_returns_copy(self):
        """Test that values property returns a copy, not the original list."""
        original_values = [1, 2, 3]
        array = NumericalArray(original_values)
        returned_values = array.get_values()

        # Modify the returned list
        returned_values.append(4)

        # Original array should be unchanged
        assert array.get_values() == [1.0, 2.0, 3.0]
        assert array.length == 3

    def test_float_conversion(self):
        """Test that integers are properly converted to floats."""
        array = NumericalArray([1, 2, 3])
        assert all(isinstance(x, float) for x in array.get_values())
        assert array.get_values() == [1.0, 2.0, 3.0]

    def test_remove_multiple_indices_in_reverse_order(self):
        """Test that removing multiple indices works correctly in reverse order."""
        array = NumericalArray([0, 1, 2, 3, 4, 5])
        array.remove_values([1, 3, 5])  # Remove indices 1, 3, 5

        # Should remove in reverse order to avoid index shifting
        assert array.get_values() == [0.0, 2.0, 4.0]
        assert array.length == 3

    def test_nan_indices_property(self):
        """Test nan_indices property."""
        # Array without NaN
        array_no_nan = NumericalArray([1.0, 2.0, 3.0])
        assert array_no_nan.nan_indices == []

        # Array with NaN at specific indices
        array_with_nan = NumericalArray([1.0, float("nan"), 3.0, None, 5.0])
        assert array_with_nan.nan_indices == [1, 3]

    def test_non_nan_values_property(self):
        """Test non_nan_values property."""
        # Array without NaN
        array_no_nan = NumericalArray([1.0, 2.0, 3.0])
        assert array_no_nan.non_nan_values == [1.0, 2.0, 3.0]

        # Array with NaN values
        array_with_nan = NumericalArray([1.0, float("nan"), 3.0, None, 5.0])
        assert array_with_nan.non_nan_values == [1.0, 3.0, 5.0]

        # Array with all NaN values
        array_all_nan = NumericalArray([float("nan"), None, float("nan")])
        assert array_all_nan.non_nan_values == []

    def test_min_with_nan_values(self):
        """Test that min on array with NaN values returns the correct value."""
        array = NumericalArray([1.0, float("nan"), 3.0])
        assert array.min == 1.0
