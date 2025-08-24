# pylint: disable=C0116, W0212
"""
Tests for TickCollection.

This module contains unit tests for the TickCollection class,
which manages multiple tick collections for a NormalizedMatrix.
"""

import pytest
from paxplot.plot_component_managers.tick_collection import (
    TickCollection,
)
from paxplot.plot_component_managers.numeric_ticks import (
    NumericTicks,
)
from paxplot.plot_component_managers.categorical_ticks import (
    CategoricalTicks,
)
from paxplot.data_managers.normalized_matrix import NormalizedMatrix


class TestTickCollection:
    """Test cases for TickCollection."""

    def test_init_with_mixed_data(self):
        """Test initialization with mixed numeric and categorical data."""
        # Create data with numeric and categorical columns
        data = [[1.0, "A", 2.5], [2.0, "B", 3.0], [3.0, "A", 1.5]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        # Should have 3 tick collections
        assert len(coordinator._tick_managers) == 3

        # Check that appropriate tick collections were created
        assert isinstance(
            coordinator._tick_managers[0], NumericTicks
        )
        assert isinstance(
            coordinator._tick_managers[1], CategoricalTicks
        )
        assert isinstance(
            coordinator._tick_managers[2], NumericTicks
        )

    def test_init_with_numeric_only(self):
        """Test initialization with numeric data only."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        assert len(coordinator._tick_managers) == 3
        for tick_manager in coordinator._tick_managers:
            assert isinstance(tick_manager, NumericTicks)

    def test_init_with_categorical_only(self):
        """Test initialization with categorical data only."""
        data = [["A", "B", "C"], ["B", "A", "D"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        assert len(coordinator._tick_managers) == 3
        for tick_manager in coordinator._tick_managers:
            assert isinstance(tick_manager, CategoricalTicks)

    def test_get_tick_manager_valid_index(self):
        """Test getting ticks with valid index."""
        data = [[1.0, "A"], [2.0, "B"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        # Get first column (numeric)
        tick_manager = coordinator.get_tick_manager(0)
        assert isinstance(tick_manager, NumericTicks)

        # Get second column (categorical)
        tick_manager = coordinator.get_tick_manager(1)
        assert isinstance(tick_manager, CategoricalTicks)

    def test_get_tick_manager_invalid_index(self):
        """Test getting ticks with invalid index."""
        data = [[1.0, "A"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        # Index out of bounds
        assert coordinator.get_tick_manager(2) is None
        assert coordinator.get_tick_manager(-1) is None

    def test_get_numeric_tick_manager_success(self):
        """Test getting numeric ticks successfully."""
        data = [[1.0, "A"], [2.0, "B"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        tick_manager = coordinator.get_numeric_tick_manager(0)
        assert isinstance(tick_manager, NumericTicks)

    def test_get_numeric_tick_manager_wrong_type(self):
        """Test getting numeric ticks for categorical column."""
        data = [[1.0, "A"], [2.0, "B"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        with pytest.raises(TypeError, match="Column 1 is not numeric"):
            coordinator.get_numeric_tick_manager(1)

    def test_get_numeric_tick_manager_invalid_index(self):
        """Test getting numeric ticks with invalid index."""
        data = [[1.0]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        with pytest.raises(TypeError, match="Column 1 is not numeric"):
            coordinator.get_numeric_tick_manager(1)

    def test_get_categorical_tick_manager_success(self):
        """Test getting categorical ticks successfully."""
        data = [[1.0, "A"], [2.0, "B"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        tick_manager = coordinator.get_categorical_tick_manager(1)
        assert isinstance(tick_manager, CategoricalTicks)

    def test_get_categorical_tick_manager_wrong_type(self):
        """Test getting categorical ticks for numeric column."""
        data = [[1.0, "A"], [2.0, "B"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        with pytest.raises(TypeError, match="Column 0 is not categorical"):
            coordinator.get_categorical_tick_manager(0)

    def test_get_categorical_tick_manager_invalid_index(self):
        """Test getting categorical ticks with invalid index."""
        data = [["A"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        with pytest.raises(TypeError, match="Column 1 is not categorical"):
            coordinator.get_categorical_tick_manager(1)

    def test_automatic_tick_generation_numeric(self):
        """Test that numeric ticks automatically generate tick values."""
        data = [
            [1.0, 5.0, 10.0],
            [2.0, 6.0, 11.0],
        ]  # Need at least 2 rows for proper column detection
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        tick_manager = coordinator.get_numeric_tick_manager(0)

        # Should have generated ticks based on data range
        raw_ticks = tick_manager.get_raw_values()
        assert raw_ticks is not None
        assert len(raw_ticks) > 0

        # Check that ticks cover the data range for the first column (1.0 to 2.0)
        assert min(raw_ticks) <= 1.0
        assert max(raw_ticks) >= 2.0

    def test_automatic_tick_generation_categorical(self):
        """Test that categorical ticks automatically set tick values."""
        data = [
            ["A", "B", "C"],
            ["B", "A", "D"],
        ]  # Need at least 2 rows for proper column detection
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        tick_manager = coordinator.get_categorical_tick_manager(0)

        # Should have set ticks based on categories from the first column (A, B)
        raw_ticks = tick_manager.get_raw_values()
        assert raw_ticks is not None
        assert set(raw_ticks) == {"A", "B"}

    def test_empty_column_handling(self):
        """Test handling of empty columns."""
        data = [[]]  # Empty data
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        # Should handle empty matrix gracefully
        assert len(coordinator._tick_managers) == 0

    def test_single_value_column(self):
        """Test handling of columns with single values."""
        data = [
            [1.0, "A"],
            [1.0, "A"],
        ]  # Need at least 2 rows for proper column detection
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        # Should create tick collections even for single values
        assert len(coordinator._tick_managers) == 2
        assert isinstance(
            coordinator._tick_managers[0], NumericTicks
        )
        assert isinstance(
            coordinator._tick_managers[1], CategoricalTicks
        )

    def test_matrix_reference(self):
        """Test that collection maintains reference to the matrix."""
        data = [[1.0, "A"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        assert coordinator._matrix is matrix

    def test_tick_manager_observer_registration(self):
        """Test that tick collections are properly observing their arrays."""
        data = [[1.0, 2.0, 3.0]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        tick_manager = coordinator.get_numeric_tick_manager(0)
        numeric_column = matrix.get_numeric_column(0)

        # The tick collection should be registered as an observer
        # We can verify this by checking that the tick collection has the column reference
        assert tick_manager._axis_data is numeric_column

    def test_categorical_tick_manager_observer_registration(self):
        """Test that categorical tick collections are properly observing their arrays."""
        data = [["A", "B", "C"]]
        matrix = NormalizedMatrix(data=data)
        coordinator = TickCollection(matrix)

        tick_manager = coordinator.get_categorical_tick_manager(0)
        categorical_column = matrix.get_categorical_column(0)

        # The tick collection should be registered as an observer
        assert tick_manager._axis_data is categorical_column
