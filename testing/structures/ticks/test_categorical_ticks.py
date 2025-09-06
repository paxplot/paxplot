"""Tests for CategoricalTicks class."""

import pytest

from paxplot.structures.ticks.categorical_ticks import CategoricalTicks


class TestCategoricalTicks:
    """Test cases for CategoricalTicks class."""

    def test_init_empty(self):
        """Test initialization with empty arrays."""
        ticks = CategoricalTicks()
        assert len(ticks.labels) == 0
        assert len(ticks.locations) == 0

    def test_generate_ticks_from_categories_basic(self):
        """Test basic tick generation from categories."""
        categories = ['Red', 'Blue', 'Green', 'Yellow']
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        
        assert len(ticks.labels) == 4
        assert len(ticks.locations) == 4
        
        # Check that labels match input categories
        assert ticks.labels.values == ['Red', 'Blue', 'Green', 'Yellow']
        
        # Check that locations are sequential indices
        assert ticks.locations.values == [0, 1, 2, 3]

    def test_generate_ticks_from_categories_with_duplicates(self):
        """Test tick generation with duplicate categories."""
        categories = ['Blue', 'Red', 'Blue', 'Green', 'Red']
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        
        # CategoricalArray should handle deduplication
        assert len(ticks.labels) == 3
        assert ticks.labels.values == ['Blue', 'Red', 'Green']
        assert ticks.locations.values == [0, 1, 2]

    def test_generate_ticks_from_categories_empty_input(self):
        """Test tick generation with empty categories."""
        ticks = CategoricalTicks()
        
        with pytest.raises(ValueError, match="Categories cannot be empty"):
            ticks.generate_ticks_from_categories([])

    def test_generate_ticks_from_categories_invalid_categories(self):
        """Test tick generation with invalid categories."""
        ticks = CategoricalTicks()
        
        # Test with non-string categories
        with pytest.raises(ValueError, match="Category at index 0 must be a string"):
            ticks.generate_ticks_from_categories([123, 'Blue'])
        
        # Test with empty string categories
        with pytest.raises(ValueError, match="Category at index 1 cannot be empty"):
            ticks.generate_ticks_from_categories(['Red', '', 'Blue'])

    def test_generate_ticks_from_categories_regeneration(self):
        """Test regenerating ticks with different categories."""
        ticks = CategoricalTicks()
        
        # First generation
        ticks.generate_ticks_from_categories(['A', 'B', 'C'])
        first_labels = ticks.labels.values.copy()
        first_locations = ticks.locations.values.copy()
        
        # Second generation with different categories
        ticks.generate_ticks_from_categories(['X', 'Y', 'Z', 'W'])
        second_labels = ticks.labels.values
        second_locations = ticks.locations.values
        
        # Should be different
        assert first_labels != second_labels
        assert first_locations != second_locations
        assert len(second_labels) == 4
        assert second_locations == [0, 1, 2, 3]

    def test_generate_ticks_from_categories_single_category(self):
        """Test tick generation with single category."""
        categories = ['Single']
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        
        assert len(ticks.labels) == 1
        assert ticks.labels.values == ['Single']
        assert ticks.locations.values == [0]

    def test_generate_ticks_from_categories_many_categories(self):
        """Test tick generation with many categories."""
        categories = [f'Category_{i}' for i in range(20)]
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        
        assert len(ticks.labels) == 20
        assert len(ticks.locations) == 20
        
        # Check that locations are sequential
        expected_locations = list(range(20))
        assert ticks.locations.values == expected_locations

    def test_generate_ticks_from_categories_special_characters(self):
        """Test tick generation with special characters in categories."""
        categories = ['Category-1', 'Category_2', 'Category 3', 'Category@4']
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        
        assert len(ticks.labels) == 4
        assert ticks.labels.values == categories
        assert ticks.locations.values == [0, 1, 2, 3]

    def test_generate_ticks_from_categories_unicode(self):
        """Test tick generation with unicode characters."""
        categories = ['红色', '蓝色', '绿色', '黄色']  # Chinese colors
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        
        assert len(ticks.labels) == 4
        assert ticks.labels.values == categories
        assert ticks.locations.values == [0, 1, 2, 3]


    def test_repr_empty(self):
        """Test string representation of empty ticks."""
        ticks = CategoricalTicks()
        repr_str = repr(ticks)
        assert "CategoricalTicks(empty)" in repr_str

    def test_repr_with_data(self):
        """Test string representation with data."""
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(['Red', 'Blue', 'Green'])
        repr_str = repr(ticks)
        
        assert "CategoricalTicks" in repr_str
        assert "labels=" in repr_str
        assert "locations=" in repr_str

    def test_repr_with_many_categories(self):
        """Test string representation with many categories."""
        categories = [f'Category_{i}' for i in range(10)]
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        repr_str = repr(ticks)
        
        assert "CategoricalTicks" in repr_str
        assert "labels=['Category_0', 'Category_1', 'Category_2']..." in repr_str
        assert "locations=[0.0, 1.0, 2.0]..." in repr_str

    def test_categorical_array_integration(self):
        """Test that CategoricalArray integration works correctly."""
        categories = ['A', 'B', 'A', 'C', 'B']
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        
        # CategoricalArray should handle deduplication and ordering
        assert ticks.labels.values == ['A', 'B', 'C']
        assert ticks.locations.values == [0, 1, 2]
        
        # Test that we can access the underlying CategoricalArray methods
        assert hasattr(ticks.labels, 'unique_values')
        assert ticks.labels.unique_values == ['A', 'B', 'C']

    def test_inheritance_from_base_ticks(self):
        """Test that CategoricalTicks properly inherits from BaseTicks."""
        ticks = CategoricalTicks()
        
        # Should have all BaseTicks methods
        assert hasattr(ticks, 'labels')
        assert hasattr(ticks, 'locations')
        assert hasattr(ticks, 'validate')
        assert hasattr(ticks, 'add_ticks')
        assert hasattr(ticks, 'remove_ticks')
        
        # Should be able to use BaseTicks methods after generating ticks
        ticks.generate_ticks_from_categories(['Red', 'Blue', 'Green'])
        
        # Test adding more ticks
        result = ticks.add_ticks(['Yellow'], [3])
        assert result is True
        assert len(ticks.labels) == 4
        
        # Test removing ticks
        result = ticks.remove_ticks([0])
        assert result is True
        assert len(ticks.labels) == 3

    def test_direct_array_access(self):
        """Test direct access to underlying arrays for lookups."""
        categories = ['Red', 'Blue', 'Green', 'Yellow']
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        
        # Test direct access for lookups
        blue_index = ticks.locations.values[ticks.labels.values.index('Blue')]
        assert blue_index == 1
        
        green_category = ticks.labels.values[2]
        assert green_category == 'Green'
        
        # Test that arrays are properly synchronized
        assert len(ticks.labels.values) == len(ticks.locations.values)
        for i, (label, location) in enumerate(zip(ticks.labels.values, ticks.locations.values)):
            assert location == i

    def test_validation_after_generation(self):
        """Test that validation works correctly after tick generation."""
        categories = ['Red', 'Blue', 'Green']
        ticks = CategoricalTicks()
        ticks.generate_ticks_from_categories(categories)
        
        # Should not raise any exception
        ticks.validate()
        
        # Check that arrays are properly synchronized
        assert len(ticks.labels) == len(ticks.locations)
        assert ticks.labels.values == ['Red', 'Blue', 'Green']
        assert ticks.locations.values == [0, 1, 2]

