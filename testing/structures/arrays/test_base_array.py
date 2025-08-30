"""Tests for BaseArray."""

import pytest
from src.paxplot.structures.arrays.base_array import BaseArray
from src.paxplot.structures.arrays.numerical_array import NumericalArray
from src.paxplot.structures.arrays.categorical_array import CategoricalArray


class TestBaseArray:
    """Test cases for BaseArray functionality through concrete implementations."""
    
    def test_base_array_cannot_be_instantiated(self):
        """Test that BaseArray cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseArray([1, 2, 3])
    
    def test_numerical_array_inherits_base_functionality(self):
        """Test that NumericalArray inherits base functionality correctly."""
        array = NumericalArray([1, 2, 3])
        assert hasattr(array, 'has_nan')
        assert hasattr(array, 'nan_count')
        assert hasattr(array, 'nan_indices')
        assert hasattr(array, 'non_nan_values')
        assert hasattr(array, 'length')
        assert hasattr(array, 'append')
        assert hasattr(array, 'remove')
        assert hasattr(array, 'reset_nan_state')
    
    def test_categorical_array_inherits_base_functionality(self):
        """Test that CategoricalArray inherits base functionality correctly."""
        array = CategoricalArray(['A', 'B', 'C'])
        assert hasattr(array, 'has_nan')
        assert hasattr(array, 'nan_count')
        assert hasattr(array, 'nan_indices')
        assert hasattr(array, 'non_nan_values')
        assert hasattr(array, 'length')
        assert hasattr(array, 'append')
        assert hasattr(array, 'remove')
        assert hasattr(array, 'reset_nan_state')
    
    def test_shared_nan_handling_works_consistently(self):
        """Test that NaN handling works consistently across array types."""
        numeric_array = NumericalArray([1.0, None, 3.0])
        categorical_array = CategoricalArray(['A', None, 'B'])
        
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
        categorical_array = CategoricalArray(['A', 'B'])
        
        # Append values
        numeric_array.append([3, 4])
        categorical_array.append(['C', 'D'])
        
        # Check lengths
        assert numeric_array.length == 4
        assert categorical_array.length == 4
        
        # Check values
        assert numeric_array.values == [1.0, 2.0, 3.0, 4.0]
        assert categorical_array.values == ['A', 'B', 'C', 'D']
    
    def test_shared_remove_functionality(self):
        """Test that remove functionality works consistently across array types."""
        numeric_array = NumericalArray([1, 2, 3, 4])
        categorical_array = CategoricalArray(['A', 'B', 'C', 'D'])
        
        # Remove values
        numeric_array.remove([1, 3])
        categorical_array.remove([1, 3])
        
        # Check lengths
        assert numeric_array.length == 2
        assert categorical_array.length == 2
        
        # Check values (removed in reverse order)
        assert numeric_array.values == [1.0, 3.0]
        assert categorical_array.values == ['A', 'C']
    
    def test_shared_length_property(self):
        """Test that length property works consistently across array types."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(['A', 'B', 'C'])
        
        assert numeric_array.length == 3
        assert categorical_array.length == 3
        assert len(numeric_array) == 3
        assert len(categorical_array) == 3
    
    def test_shared_indexing_functionality(self):
        """Test that indexing functionality works consistently across array types."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(['A', 'B', 'C'])
        
        assert numeric_array[0] == 1.0
        assert numeric_array[1] == 2.0
        assert categorical_array[0] == 'A'
        assert categorical_array[1] == 'B'
    
    def test_shared_values_property_returns_copy(self):
        """Test that values property returns a copy consistently across array types."""
        numeric_array = NumericalArray([1, 2, 3])
        categorical_array = CategoricalArray(['A', 'B', 'C'])
        
        # Get values and modify them
        numeric_values = numeric_array.values
        categorical_values = categorical_array.values
        
        numeric_values.append(4)
        categorical_values.append('D')
        
        # Original arrays should be unchanged
        assert numeric_array.values == [1.0, 2.0, 3.0]
        assert categorical_array.values == ['A', 'B', 'C']
    
    def test_shared_nan_handling_with_different_nan_types(self):
        """Test that different NaN representations are handled consistently."""
        numeric_array = NumericalArray([1.0, None, 3.0, float('nan')])
        categorical_array = CategoricalArray(['A', None, 'B', float('nan')])
        
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
        categorical_array = CategoricalArray(['A', 'B', 'C'])
        
        # Initially no NaN
        assert numeric_array.has_nan is False
        assert categorical_array.has_nan is False
        
        # Manually add NaN and reset state
        numeric_array._values.append(float('nan'))
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
