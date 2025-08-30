"""Unit tests for CategoricalArray."""

import pytest
from paxplot.structures.arrays.categorical_array import CategoricalArray


class TestCategoricalArray:
    """Test cases for CategoricalArray class."""

    def test_init_with_valid_values(self):
        """Test initialization with valid string values."""
        values = ["A", "B", "A", "C"]
        array = CategoricalArray(values)

        assert array.values == ["A", "B", "A", "C"]
        assert array.unique_values == ["A", "B", "C"]
        assert array.length == 4

    def test_init_with_empty_list(self):
        """Test initialization with empty list."""
        array = CategoricalArray([])

        assert not array.values
        assert not array.unique_values
        assert array.length == 0

    def test_init_with_none_raises_error(self):
        """Test that initialization with None raises ValueError."""
        with pytest.raises(ValueError, match="Values cannot be None"):
            CategoricalArray(None)

    def test_init_with_non_string_values_raises_error(self):
        """Test that initialization with non-string values raises ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            CategoricalArray(["A", 123, "C"])

    def test_init_with_none_values_converts_to_nan(self):
        """Test that initialization with None values converts them to NaN."""
        array = CategoricalArray(["A", None, "C"])  # type: ignore
        assert array.values == ["A", "<NaN>", "C"]
        assert array.has_nan is True
        assert array.nan_count == 1
        assert array.nan_indices == [1]

    def test_unique_values_property(self):
        """Test unique_values property."""
        array = CategoricalArray(["A", "B", "A", "C", "B"])
        assert array.unique_values == ["A", "B", "C"]

    def test_unique_values_order_preserved(self):
        """Test that unique values preserve order of appearance."""
        array = CategoricalArray(["C", "A", "B", "A", "C"])
        assert array.unique_values == ["C", "A", "B"]

    def test_append_valid_values(self):
        """Test appending valid string values."""
        array = CategoricalArray(["A", "B"])
        array.append(["C", "A"])

        assert array.values == ["A", "B", "C", "A"]
        assert array.unique_values == ["A", "B", "C"]
        assert array.length == 4

    def test_append_empty_list(self):
        """Test appending empty list."""
        array = CategoricalArray(["A", "B"])
        array.append([])

        assert array.values == ["A", "B"]
        assert array.unique_values == ["A", "B"]
        assert array.length == 2

    def test_append_invalid_values_raises_error(self):
        """Test that appending invalid values raises ValueError."""
        array = CategoricalArray(["A", "B"])
        with pytest.raises(ValueError, match="must be a string"):
            array.append(["C", 123])

    def test_append_new_unique_values(self):
        """Test that appending new unique values updates unique_values."""
        array = CategoricalArray(["A", "B"])
        array.append(["C", "D"])

        assert array.unique_values == ["A", "B", "C", "D"]

    def test_remove_valid_indices(self):
        """Test removing values at valid indices."""
        array = CategoricalArray(["A", "B", "C", "D"])
        array.remove([1, 3])

        assert array.values == ["A", "C"]
        assert array.unique_values == ["A", "C"]
        assert array.length == 2

    def test_remove_empty_indices(self):
        """Test removing with empty indices list."""
        array = CategoricalArray(["A", "B", "C"])
        array.remove([])

        assert array.values == ["A", "B", "C"]
        assert array.unique_values == ["A", "B", "C"]
        assert array.length == 3

    def test_remove_out_of_bounds_index_raises_error(self):
        """Test that removing out of bounds index raises IndexError."""
        array = CategoricalArray(["A", "B", "C"])
        with pytest.raises(IndexError, match="out of bounds"):
            array.remove([5])

    def test_remove_negative_index_raises_error(self):
        """Test that removing negative index raises IndexError."""
        array = CategoricalArray(["A", "B", "C"])
        with pytest.raises(IndexError, match="out of bounds"):
            array.remove([-1])

    def test_remove_invalid_index_type_raises_error(self):
        """Test that removing with invalid index type raises ValueError."""
        array = CategoricalArray(["A", "B", "C"])
        with pytest.raises(ValueError, match="must be an integer"):
            array.remove(["invalid"])

    def test_get_category_indices(self):
        """Test get_category_indices method."""
        array = CategoricalArray(["A", "B", "A", "C"])
        indices = array.get_category_indices()

        # A=0, B=1, C=2
        assert indices == [0, 1, 0, 2]

    def test_get_category_indices_empty_array(self):
        """Test get_category_indices on empty array."""
        array = CategoricalArray([])
        indices = array.get_category_indices()

        assert indices == []

    def test_get_category_indices_single_value(self):
        """Test get_category_indices with single value."""
        array = CategoricalArray(["A"])
        indices = array.get_category_indices()

        assert indices == [0]

    def test_len_operator(self):
        """Test len() operator."""
        array = CategoricalArray(["A", "B", "C", "D", "E"])
        assert len(array) == 5

    def test_getitem_operator(self):
        """Test indexing operator."""
        array = CategoricalArray(["A", "B", "C", "D", "E"])
        assert array[0] == "A"
        assert array[2] == "C"
        assert array[-1] == "E"

    def test_getitem_out_of_bounds_raises_error(self):
        """Test that indexing out of bounds raises IndexError."""
        array = CategoricalArray(["A", "B", "C"])
        with pytest.raises(IndexError):
            _ = array[5]

    def test_repr_small_array(self):
        """Test string representation for small array."""
        array = CategoricalArray(["A", "B", "C"])
        assert repr(array) == "CategoricalArray(['A', 'B', 'C'])"

    def test_repr_large_array(self):
        """Test string representation for large array."""
        array = CategoricalArray(
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        )
        repr_str = repr(array)
        assert "CategoricalArray(" in repr_str
        assert "..." in repr_str
        assert repr_str.endswith("])")

    def test_values_property_returns_copy(self):
        """Test that values property returns a copy, not the original list."""
        original_values = ["A", "B", "C"]
        array = CategoricalArray(original_values)
        returned_values = array.values

        # Modify the returned list
        returned_values.append("D")

        # Original array should be unchanged
        assert array.values == ["A", "B", "C"]
        assert array.length == 3

    def test_unique_values_property_returns_copy(self):
        """Test that unique_values property returns a copy."""
        array = CategoricalArray(["A", "B", "A", "C"])
        returned_unique = array.unique_values

        # Modify the returned list
        returned_unique.append("D")

        # Original array should be unchanged
        assert array.unique_values == ["A", "B", "C"]

    def test_remove_updates_unique_values(self):
        """Test that removing values updates unique_values correctly."""
        array = CategoricalArray(["A", "B", "A", "C", "B"])
        array.remove([1, 3])  # Remove 'B' at index 1 and 'C' at index 3

        assert array.values == ["A", "A", "B"]
        assert array.unique_values == ["A", "B"]

    def test_remove_all_instances_of_category(self):
        """Test removing all instances of a category."""
        array = CategoricalArray(["A", "B", "A", "C", "A"])
        array.remove([0, 2, 4])  # Remove all 'A's

        assert array.values == ["B", "C"]
        assert array.unique_values == ["B", "C"]

    def test_remove_multiple_indices_in_reverse_order(self):
        """Test that removing multiple indices works correctly in reverse order."""
        array = CategoricalArray(["A", "B", "C", "D", "E", "F"])
        array.remove([1, 3, 5])  # Remove indices 1, 3, 5

        # Should remove in reverse order to avoid index shifting
        assert array.values == ["A", "C", "E"]
        assert array.length == 3

    def test_case_sensitive_categories(self):
        """Test that categories are case sensitive."""
        array = CategoricalArray(["A", "a", "A", "a"])
        assert array.unique_values == ["A", "a"]
        assert array.get_category_indices() == [0, 1, 0, 1]

    def test_empty_string_categories(self):
        """Test handling of empty string categories."""
        array = CategoricalArray(["A", "", "B", ""])
        assert array.unique_values == ["A", "", "B"]
        assert array.get_category_indices() == [0, 1, 2, 1]

    def test_nan_indices_property(self):
        """Test nan_indices property."""
        # Array without NaN
        array_no_nan = CategoricalArray(['A', 'B', 'C'])
        assert array_no_nan.nan_indices == []

        # Array with NaN at specific indices
        array_with_nan = CategoricalArray(['A', None, 'B', float('nan'), 'C'])  # type: ignore
        assert array_with_nan.nan_indices == [1, 3]

    def test_non_nan_values_property(self):
        """Test non_nan_values property."""
        # Array without NaN
        array_no_nan = CategoricalArray(['A', 'B', 'C'])
        assert array_no_nan.non_nan_values == ['A', 'B', 'C']

        # Array with NaN values
        array_with_nan = CategoricalArray(['A', None, 'B', float('nan'), 'C'])  # type: ignore
        assert array_with_nan.non_nan_values == ['A', 'B', 'C']

        # Array with all NaN values
        array_all_nan = CategoricalArray([None, float('nan'), None])  # type: ignore
        assert array_all_nan.non_nan_values == []

    def test_unique_values_with_nan(self):
        """Test that unique_values property handles NaN values correctly."""
        array = CategoricalArray(["A", None, "B", float('nan'), "A"])  # type: ignore
        assert array.unique_values == ["A", "<NaN>", "B"]  # NaN is included in unique values
        assert array.get_category_indices() == [0, 1, 2, 1, 0]
