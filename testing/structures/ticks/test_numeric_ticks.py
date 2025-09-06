"""Tests for NumericTicks class."""

import pytest
import math

from paxplot.structures.ticks.numeric_ticks import NumericTicks


class TestNumericTicks:
    """Test cases for NumericTicks class."""

    def test_init_empty(self):
        """Test initialization with empty arrays."""
        ticks = NumericTicks()
        assert len(ticks.labels) == 0
        assert len(ticks.locations) == 0

    def test_generate_ticks_from_range_basic(self):
        """Test basic tick generation from range."""
        ticks = NumericTicks()
        ticks.set_ticks_from_range(0, 100)
        
        assert len(ticks.labels) == 5
        assert len(ticks.locations) == 5
        
        # Check that locations are in ascending order
        locations = ticks.locations.values
        assert locations == sorted(locations)
        
        # Check that first and last locations are close to min/max
        assert abs(locations[0] - 0) < 1e-10
        # MaxNLocator may not include the exact max value, so check it's close
        assert locations[-1] >= 80  # Should be reasonably close to 100

    def test_generate_ticks_from_range_custom_max_ticks(self):
        """Test tick generation with custom max_ticks parameter."""
        ticks = NumericTicks()
        ticks.set_ticks_from_range(0, 100, max_ticks=3)
        
        assert len(ticks.labels) == 3
        assert len(ticks.locations) == 3

    def test_generate_ticks_from_range_custom_precision(self):
        """Test tick generation with custom precision."""
        ticks = NumericTicks()
        ticks.set_ticks_from_range(0, 1, max_ticks=3, precision=3)
        
        # Check that labels have 3 decimal places
        for label in ticks.labels.values:
            if '.' in label:
                decimal_places = len(label.split('.')[1])
                assert decimal_places <= 3

    def test_generate_ticks_from_range_invalid_range(self):
        """Test tick generation with invalid range."""
        ticks = NumericTicks()
        
        with pytest.raises(ValueError, match="min_value .* must be less than max_value"):
            ticks.set_ticks_from_range(100, 0)
        
        with pytest.raises(ValueError, match="min_value .* must be less than max_value"):
            ticks.set_ticks_from_range(50, 50)

    def test_generate_ticks_from_range_invalid_max_ticks(self):
        """Test tick generation with invalid max_ticks."""
        ticks = NumericTicks()
        
        with pytest.raises(ValueError, match="max_ticks must be positive"):
            ticks.set_ticks_from_range(0, 100, max_ticks=0)
        
        with pytest.raises(ValueError, match="max_ticks must be positive"):
            ticks.set_ticks_from_range(0, 100, max_ticks=-1)

    def test_generate_ticks_from_range_small_range(self):
        """Test tick generation with very small range."""
        ticks = NumericTicks()
        ticks.set_ticks_from_range(0, 0.1)
        
        assert len(ticks.labels) > 0
        assert len(ticks.locations) > 0
        
        # Locations should be within the range
        for location in ticks.locations.values:
            assert 0 <= location <= 0.1

    def test_generate_ticks_from_range_large_range(self):
        """Test tick generation with large range."""
        ticks = NumericTicks()
        ticks.set_ticks_from_range(0, 1000000)
        
        assert len(ticks.labels) == 5
        assert len(ticks.locations) == 5
        
        # Check that locations are reasonable
        locations = ticks.locations.values
        assert locations[0] >= 0
        assert locations[-1] <= 1000000

    def test_generate_ticks_from_range_negative_values(self):
        """Test tick generation with negative values."""
        ticks = NumericTicks()
        ticks.set_ticks_from_range(-100, 100)
        
        assert len(ticks.labels) == 5
        assert len(ticks.locations) == 5
        
        # Check that locations span the range
        locations = ticks.locations.values
        assert locations[0] >= -100
        assert locations[-1] <= 100

    def test_generate_ticks_from_range_regeneration(self):
        """Test regenerating ticks with different parameters."""
        ticks = NumericTicks()
        
        # First generation
        ticks.set_ticks_from_range(0, 100)
        first_labels = ticks.labels.values.copy()
        first_locations = ticks.locations.values.copy()
        
        # Second generation with different range
        ticks.set_ticks_from_range(0, 200, max_ticks=3)
        second_labels = ticks.labels.values
        second_locations = ticks.locations.values
        
        # Should be different
        assert first_labels != second_labels
        assert first_locations != second_locations
        assert len(second_labels) == 3


    def test_repr_empty(self):
        """Test string representation of empty ticks."""
        ticks = NumericTicks()
        repr_str = repr(ticks)
        assert "NumericTicks(empty)" in repr_str

    def test_repr_with_data(self):
        """Test string representation with data."""
        ticks = NumericTicks()
        ticks.set_ticks_from_range(0, 100, max_ticks=3)
        repr_str = repr(ticks)
        
        assert "NumericTicks" in repr_str
        assert "labels=" in repr_str
        assert "locations=" in repr_str

    def test_maxnlocator_integration(self):
        """Test that MaxNLocator produces reasonable tick positions."""
        ticks = NumericTicks()
        ticks.set_ticks_from_range(0, 100, max_ticks=6)
        
        locations = ticks.locations.values
        
        # MaxNLocator should produce "nice" numbers
        # Check that spacing is reasonable
        if len(locations) > 1:
            spacing = locations[1] - locations[0]
            # Spacing should be a "nice" number (divisible by common factors)
            assert spacing > 0
            # All spacings should be the same (for linear scales)
            for i in range(1, len(locations)):
                assert abs((locations[i] - locations[i-1]) - spacing) < 1e-10

    def test_inheritance_from_base_ticks(self):
        """Test that NumericTicks properly inherits from BaseTicks."""
        ticks = NumericTicks()
        
        # Should have all BaseTicks methods
        assert hasattr(ticks, 'labels')
        assert hasattr(ticks, 'locations')
        assert hasattr(ticks, 'validate')
        assert hasattr(ticks, 'append')
        assert hasattr(ticks, 'remove')
        
        # Should be able to use BaseTicks methods after generating ticks
        ticks.set_ticks_from_range(0, 100, max_ticks=3)
        
        # Test adding more ticks
        result = ticks.append(['150.0'], [150.0])
        assert result is True
        assert len(ticks.labels) == 4
        
        # Test removing ticks
        result = ticks.remove([0])
        assert result is True
        assert len(ticks.labels) == 3

