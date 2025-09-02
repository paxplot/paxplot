"""Tests for CustomAxisLimit dataclass."""

import pytest
from paxplot.structures.limits import CustomAxisLimit


class TestCustomAxisLimit:
    """Test cases for CustomAxisLimit dataclass."""

    def test_valid_numeric_inputs(self):
        """Test that valid numeric inputs work correctly."""
        # Test with floats
        limits = CustomAxisLimit(min_val=0.0, max_val=10.0)
        assert limits.min_val == 0.0
        assert limits.max_val == 10.0

        # Test with integers
        limits = CustomAxisLimit(min_val=5, max_val=15)
        assert limits.min_val == 5
        assert limits.max_val == 15

        # Test with mixed types
        limits = CustomAxisLimit(min_val=1.5, max_val=20)
        assert limits.min_val == 1.5
        assert limits.max_val == 20

    def test_none_values(self):
        """Test that None values are valid."""
        # Test with no limits
        limits = CustomAxisLimit()
        assert limits.min_val is None
        assert limits.max_val is None

        # Test with only min set
        limits = CustomAxisLimit(min_val=0.0)
        assert limits.min_val == 0.0
        assert limits.max_val is None

        # Test with only max set
        limits = CustomAxisLimit(max_val=10.0)
        assert limits.min_val is None
        assert limits.max_val == 10.0

    def test_invalid_type_inputs(self):
        """Test that invalid types raise TypeError."""
        # Test string input
        with pytest.raises(TypeError, match="min_val must be numeric or None"):
            CustomAxisLimit(min_val="hello")

        with pytest.raises(TypeError, match="max_val must be numeric or None"):
            CustomAxisLimit(max_val="world")

        # Test list input
        with pytest.raises(TypeError, match="min_val must be numeric or None"):
            CustomAxisLimit(min_val=[1, 2, 3])

        # Test dict input
        with pytest.raises(TypeError, match="max_val must be numeric or None"):
            CustomAxisLimit(max_val={"key": "value"})

    def test_invalid_range(self):
        """Test that invalid ranges raise ValueError."""
        # Test min >= max
        with pytest.raises(
            ValueError,
            match="min_val \\(10.0\\) must be less than max_val \\(5.0\\)",
        ):
            CustomAxisLimit(min_val=10.0, max_val=5.0)

        # Test min == max
        with pytest.raises(
            ValueError,
            match="min_val \\(5.0\\) must be less than max_val \\(5.0\\)",
        ):
            CustomAxisLimit(min_val=5.0, max_val=5.0)

    def test_nan_rejection(self):
        """Test that NaN values are rejected."""
        with pytest.raises(ValueError, match="min_val cannot be NaN"):
            CustomAxisLimit(min_val=float("nan"))

        with pytest.raises(ValueError, match="max_val cannot be NaN"):
            CustomAxisLimit(max_val=float("nan"))

    def test_infinity_rejection(self):
        """Test that infinity values are rejected."""
        with pytest.raises(ValueError, match="min_val cannot be infinity"):
            CustomAxisLimit(min_val=float("inf"))

        with pytest.raises(ValueError, match="max_val cannot be infinity"):
            CustomAxisLimit(max_val=float("inf"))

        with pytest.raises(ValueError, match="min_val cannot be infinity"):
            CustomAxisLimit(min_val=float("-inf"))

        with pytest.raises(ValueError, match="max_val cannot be infinity"):
            CustomAxisLimit(max_val=float("-inf"))

    def test_is_set_method(self):
        """Test the is_set method."""
        # No limits set
        limits = CustomAxisLimit()
        assert not limits.is_set()

        # Only min set
        limits = CustomAxisLimit(min_val=0.0)
        assert limits.is_set()

        # Only max set
        limits = CustomAxisLimit(max_val=10.0)
        assert limits.is_set()

        # Both set
        limits = CustomAxisLimit(min_val=0.0, max_val=10.0)
        assert limits.is_set()

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very large numbers
        limits = CustomAxisLimit(min_val=-1e308, max_val=1e308)
        assert limits.min_val == -1e308
        assert limits.max_val == 1e308

        # Very small numbers
        limits = CustomAxisLimit(min_val=-1e-308, max_val=1e-308)
        assert limits.min_val == -1e-308
        assert limits.max_val == 1e-308

        # Zero values
        limits = CustomAxisLimit(min_val=0.0, max_val=1.0)
        assert limits.min_val == 0.0
        assert limits.max_val == 1.0

        # Negative values
        limits = CustomAxisLimit(min_val=-10.0, max_val=-5.0)
        assert limits.min_val == -10.0
        assert limits.max_val == -5.0
