"""Tests for BaseArray."""

import math
import pytest
from paxplot.structures.arrays.base_array import BaseArray
from paxplot.structures.arrays.numerical_array import NumericalArray
from paxplot.structures.arrays.categorical_array import CategoricalArray


class TestBaseArray:
    """Test cases for BaseArray functionality through concrete implementations."""

    def test_base_array_cannot_be_instantiated(self):
        """Test that BaseArray cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseArray([1, 2, 3])

    def test_numerical_array_inherits_base_functionality(self):
        """Test that NumericalArray inherits base functionality correctly."""
        array = NumericalArray([1, 2, 3])
        assert hasattr(array, "has_nan")
        assert hasattr(array, "nan_count")
        assert hasattr(array, "nan_indices")
        assert hasattr(array, "non_nan_values")
        assert hasattr(array, "length")
        assert hasattr(array, "append")
        assert hasattr(array, "remove")
        assert hasattr(array, "set_values")
        assert hasattr(array, "reset_nan_state")

    def test_categorical_array_inherits_base_functionality(self):
        """Test that CategoricalArray inherits base functionality correctly."""
        array = CategoricalArray(["A", "B", "C"])
        assert hasattr(array, "has_nan")
        assert hasattr(array, "nan_count")
        assert hasattr(array, "nan_indices")
        assert hasattr(array, "non_nan_values")
        assert hasattr(array, "length")
        assert hasattr(array, "append")
        assert hasattr(array, "remove")
        assert hasattr(array, "set_values")
        assert hasattr(array, "reset_nan_state")

    def test_shared_nan_handling_works_consistently(self):
        """Test that NaN handling works consistently across array types."""
        numeric_array = NumericalArray([1.0, None, 3.0])
        categorical_array = CategoricalArray(["A", None, "B"])

        # Both should have NaN
        assert numeric_array.has_nan is True
        assert categorical_array.has_nan is True

        # Both should have same NaN count
        assert numeric_array.nan_count == 1
        assert categorical_array.nan_count == 1

        # Both should have same NaN indices
        assert numeric_array.nan_indices == [1]
        assert categorical_array.nan_indices == [1]

    def test_shared_append_functionality(self):
        """Test that append functionality works consistently across array types."""
        numeric_array = NumericalArray([1, 2])
        categorical_array = CategoricalArray(["A", "B"])

        # Append values
        numeric_array.append([3, 4])
        categorical_array.append(["C", "D"])

        # Check lengths
        assert numeric_array.length == 4
        assert categorical_array.length == 4

        # Check values
        assert numeric_array.values == [1.0, 2.0, 3.0, 4.0]
        assert categorical_array.values == ["A", "B", "C", "D"]

    def test_shared_remove_functionality(self):
        """Test that remove functionality works consistently across array types."""
        numeric_array = NumericalArray([1, 2, 3, 4])
        categorical_array = CategoricalArray(["A", "B", "C", "D"])

        # Remove values
        numeric_array.remove([1, 3])
        categorical_array.remove([1, 3])

        # Check lengths
        assert numeric_array.length == 2
        assert categorical_array.length == 2

        # Check values (removed in reverse order)
        assert numeric_array.values == [1.0, 3.0]
        assert categorical_array.values == ["A", "C"]

    def test_shared_length_property(self):
        """Test that length property works consistently across array types."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(["A", "B", "C"])

        assert numeric_array.length == 3
        assert categorical_array.length == 3
        assert len(numeric_array) == 3
        assert len(categorical_array) == 3

    def test_shared_indexing_functionality(self):
        """Test that indexing functionality works consistently across array types."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(["A", "B", "C"])

        assert numeric_array[0] == 1.0
        assert numeric_array[1] == 2.0
        assert categorical_array[0] == "A"
        assert categorical_array[1] == "B"

    def test_shared_values_property_returns_copy(self):
        """Test that values property returns a copy consistently across array types."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(["A", "B", "C"])

        # Get values and modify them
        numeric_values = numeric_array.values
        categorical_values = categorical_array.values

        numeric_values.append(4)
        categorical_values.append("D")

        # Original arrays should be unchanged
        assert numeric_array.values == [1.0, 2.0, 3.0]
        assert categorical_array.values == ["A", "B", "C"]

    def test_shared_nan_handling_with_different_nan_types(self):
        """Test that different NaN representations are handled consistently."""
        numeric_array = NumericalArray([1.0, None, 3.0, float("nan")])
        categorical_array = CategoricalArray(["A", None, "B", float("nan")])

        # Both should handle None and float('nan') as NaN
        assert numeric_array.has_nan is True
        assert categorical_array.has_nan is True
        assert numeric_array.nan_count == 2
        assert categorical_array.nan_count == 2
        assert numeric_array.nan_indices == [1, 3]
        assert categorical_array.nan_indices == [1, 3]

    def test_shared_reset_nan_state_functionality(self):
        """Test that reset_nan_state functionality works consistently."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(["A", "B", "C"])

        # Initially no NaN
        assert numeric_array.has_nan is False
        assert categorical_array.has_nan is False

        # Manually add NaN and reset state
        numeric_array._values.append(float("nan"))
        categorical_array._values.append("<NaN>")

        # State should be stale
        assert numeric_array.has_nan is False
        assert categorical_array.has_nan is False

        # Reset state
        numeric_array.reset_nan_state()
        categorical_array.reset_nan_state()

        # State should be updated
        assert numeric_array.has_nan is True
        assert categorical_array.has_nan is True

    def test_set_values_numerical_array(self):
        """Test set_values method for NumericalArray."""
        array = NumericalArray([1, 2, 3])
        assert array.values == [1.0, 2.0, 3.0]
        assert array.length == 3
        
        # Set new values
        array.set_values([10, 20, 30, 40])
        assert array.values == [10.0, 20.0, 30.0, 40.0]
        assert array.length == 4
        
        # Set values with NaN
        array.set_values([1, None, 3])
        values = array.values
        assert values[0] == 1.0
        assert math.isnan(values[1])
        assert values[2] == 3.0
        assert array.has_nan is True
        assert array.nan_count == 1

    def test_set_values_categorical_array(self):
        """Test set_values method for CategoricalArray."""
        array = CategoricalArray(['A', 'B', 'C'])
        assert array.values == ['A', 'B', 'C']
        assert array.unique_values == ['A', 'B', 'C']
        assert array.length == 3
        
        # Set new values
        array.set_values(['X', 'Y', 'Z', 'W'])
        assert array.values == ['X', 'Y', 'Z', 'W']
        assert array.unique_values == ['X', 'Y', 'Z', 'W']
        assert array.length == 4
        
        # Set values with duplicates
        array.set_values(['A', 'B', 'A', 'C', 'B'])
        assert array.values == ['A', 'B', 'A', 'C', 'B']
        assert array.unique_values == ['A', 'B', 'C']
        assert array.length == 5
        
        # Set values with NaN
        array.set_values(['A', None, 'B'])
        assert array.values == ['A', '<NaN>', 'B']
        assert array.unique_values == ['A', '<NaN>', 'B']
        assert array.has_nan is True
        assert array.nan_count == 1

    def test_set_values_validation(self):
        """Test that set_values validates input correctly."""
        # Test NumericalArray validation
        numeric_array = NumericalArray([1, 2, 3])
        
        with pytest.raises(ValueError, match="Value at index 0 must be numerical"):
            numeric_array.set_values(['invalid', 2, 3])
        
        # Test CategoricalArray validation
        categorical_array = CategoricalArray(['A', 'B', 'C'])
        
        with pytest.raises(ValueError, match="Value at index 0 must be a string"):
            categorical_array.set_values([123, 'B', 'C'])

    def test_set_values_empty_sequence(self):
        """Test set_values with empty sequence."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(['A', 'B', 'C'])
        
        # Set empty values
        numeric_array.set_values([])
        categorical_array.set_values([])
        
        assert numeric_array.values == []
        assert numeric_array.length == 0
        assert numeric_array.has_nan is False
        
        assert categorical_array.values == []
        assert categorical_array.unique_values == []
        assert categorical_array.length == 0
        assert categorical_array.has_nan is False

    def test_set_values_replaces_all_values(self):
        """Test that set_values completely replaces existing values."""
        numeric_array = NumericalArray([1, 2, 3, 4, 5])
        categorical_array = CategoricalArray(['A', 'B', 'C', 'D', 'E'])
        
        # Set fewer values
        numeric_array.set_values([10, 20])
        categorical_array.set_values(['X', 'Y'])
        
        assert numeric_array.values == [10.0, 20.0]
        assert numeric_array.length == 2
        
        assert categorical_array.values == ['X', 'Y']
        assert categorical_array.unique_values == ['X', 'Y']
        assert categorical_array.length == 2
        
        # Set more values
        numeric_array.set_values([100, 200, 300, 400, 500, 600])
        categorical_array.set_values(['P', 'Q', 'R', 'S', 'T', 'U', 'V'])
        
        assert numeric_array.values == [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        assert numeric_array.length == 6
        
        assert categorical_array.values == ['P', 'Q', 'R', 'S', 'T', 'U', 'V']
        assert categorical_array.unique_values == ['P', 'Q', 'R', 'S', 'T', 'U', 'V']
        assert categorical_array.length == 7
