"""Tests for BaseTicks abstract class."""

import pytest

from paxplot.structures.ticks.base_ticks import BaseTicks


class ConcreteTicks(BaseTicks):
    """Concrete implementation of BaseTicks for testing."""

    def generate_ticks(self, data):  # pylint: disable=unused-argument
        """Generate ticks for testing."""
        return None


class TestBaseTicks:
    """Test cases for BaseTicks abstract class."""

    def test_init_with_valid_data(self):
        """Test initialization with valid labels and locations."""
        labels = ["A", "B", "C"]
        locations = [0.0, 1.0, 2.0]
        ticks = ConcreteTicks(labels, locations)

        assert len(ticks.labels) == 3
        assert len(ticks.locations) == 3
        assert ticks.labels.values == ["A", "B", "C"]
        assert ticks.locations.values == [0.0, 1.0, 2.0]

    def test_init_with_mismatched_lengths(self):
        """Test initialization with mismatched label and location lengths."""
        labels = ["A", "B", "C"]
        locations = [0.0, 1.0]  # Different length

        with pytest.raises(
            ValueError, match="Labels and locations must have the same length"
        ):
            ConcreteTicks(labels, locations)

    def test_init_with_empty_labels(self):
        """Test initialization with empty labels."""
        labels = ["A", "", "C"]
        locations = [0.0, 1.0, 2.0]

        with pytest.raises(
            ValueError, match="Label at index 1 cannot be empty"
        ):
            ConcreteTicks(labels, locations)

    def test_init_with_invalid_locations(self):
        """Test initialization with invalid locations."""
        labels = ["A", "B", "C"]
        locations = [0.0, float("inf"), 2.0]

        with pytest.raises(
            ValueError, match="Location at index 1 must be finite"
        ):
            ConcreteTicks(labels, locations)

    def test_init_with_nan_locations(self):
        """Test initialization with NaN locations."""
        labels = ["A", "B", "C"]
        locations = [0.0, float("nan"), 2.0]

        with pytest.raises(
            ValueError, match="Location at index 1 must be finite"
        ):
            ConcreteTicks(labels, locations)

    def test_validate_success(self):
        """Test successful validation."""
        labels = ["A", "B", "C"]
        locations = [0.0, 1.0, 2.0]
        ticks = ConcreteTicks(labels, locations)

        # Should not raise any exception
        ticks.validate()

    def test_append_success(self):
        """Test successful appending of ticks."""
        labels = ["A", "B"]
        locations = [0.0, 1.0]
        ticks = ConcreteTicks(labels, locations)

        new_labels = ["C", "D"]
        new_locations = [2.0, 3.0]

        result = ticks.append(new_labels, new_locations)
        assert result is True
        assert len(ticks.labels) == 4
        assert ticks.labels.values == ["A", "B", "C", "D"]
        assert ticks.locations.values == [0.0, 1.0, 2.0, 3.0]

    def test_append_with_invalid_data(self):
        """Test appending ticks with invalid data."""
        labels = ["A", "B"]
        locations = [0.0, 1.0]
        ticks = ConcreteTicks(labels, locations)

        new_labels = ["C", ""]  # Empty label
        new_locations = [2.0, 3.0]

        result = ticks.append(new_labels, new_locations)
        assert result is False
        # Original data should be unchanged
        assert len(ticks.labels) == 2
        assert ticks.labels.values == ["A", "B"]

    def test_append_with_mismatched_lengths(self):
        """Test appending ticks with mismatched lengths."""
        labels = ["A", "B"]
        locations = [0.0, 1.0]
        ticks = ConcreteTicks(labels, locations)

        new_labels = ["C", "D", "E"]  # Different length
        new_locations = [2.0, 3.0]

        result = ticks.append(new_labels, new_locations)
        assert result is False
        # Original data should be unchanged
        assert len(ticks.labels) == 2

    def test_remove_success(self):
        """Test successful removal of ticks."""
        labels = ["A", "B", "C", "D"]
        locations = [0.0, 1.0, 2.0, 3.0]
        ticks = ConcreteTicks(labels, locations)

        result = ticks.remove([1, 3])  # Remove B and D
        assert result is True
        assert len(ticks.labels) == 2
        assert ticks.labels.values == ["A", "C"]
        assert ticks.locations.values == [0.0, 2.0]

    def test_remove_with_invalid_indices(self):
        """Test removing ticks with invalid indices."""
        labels = ["A", "B", "C"]
        locations = [0.0, 1.0, 2.0]
        ticks = ConcreteTicks(labels, locations)

        result = ticks.remove([5])  # Out of bounds
        assert result is False
        # Original data should be unchanged
        assert len(ticks.labels) == 3

    def test_remove_with_negative_indices(self):
        """Test removing ticks with negative indices."""
        labels = ["A", "B", "C"]
        locations = [0.0, 1.0, 2.0]
        ticks = ConcreteTicks(labels, locations)

        result = ticks.remove([-1])  # Negative index
        assert result is False
        # Original data should be unchanged
        assert len(ticks.labels) == 3

    def test_len(self):
        """Test length property."""
        labels = ["A", "B", "C"]
        locations = [0.0, 1.0, 2.0]
        ticks = ConcreteTicks(labels, locations)

        assert len(ticks) == 3

    def test_repr_empty(self):
        """Test string representation of empty ticks."""
        ticks = ConcreteTicks([], [])
        repr_str = repr(ticks)
        assert "ConcreteTicks(empty)" in repr_str

    def test_repr_with_data(self):
        """Test string representation with data."""
        labels = ["A", "B", "C"]
        locations = [0.0, 1.0, 2.0]
        ticks = ConcreteTicks(labels, locations)
        repr_str = repr(ticks)

        assert "ConcreteTicks" in repr_str
        assert "labels=['A', 'B', 'C']" in repr_str
        assert "locations=[0.0, 1.0, 2.0]" in repr_str

    def test_repr_with_many_ticks(self):
        """Test string representation with many ticks."""
        labels = ["A", "B", "C", "D", "E", "F"]
        locations = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ticks = ConcreteTicks(labels, locations)
        repr_str = repr(ticks)

        assert "ConcreteTicks" in repr_str
        assert "labels=['A', 'B', 'C']..." in repr_str
        assert "locations=[0.0, 1.0, 2.0]..." in repr_str

    def test_concrete_implementation(self):
        """Test that concrete classes can be instantiated."""
        # This should work since ConcreteTicks is a concrete implementation
        ticks = ConcreteTicks(["A"], [0.0])
        assert len(ticks) == 1
