"""Tests for TickManager."""

import pytest
from paxplot.structures.tick_manger import TickManager, TickType
from paxplot.structures.ticks.numeric_ticks import NumericTicks
from paxplot.structures.ticks.categorical_ticks import CategoricalTicks


class TestTickManager:
    """Test cases for TickManager."""

    def test_init_with_valid_ticks(self):
        """Test initialization with valid ticks."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])
        assert manager.num_ticks == 2
        assert len(manager) == 2

    def test_init_with_single_tick(self):
        """Test initialization with single tick."""
        numeric_ticks = NumericTicks()
        manager = TickManager([numeric_ticks])
        assert manager.num_ticks == 1
        assert manager.get_tick_type(0) == TickType.NUMERIC

    def test_init_with_empty_ticks(self):
        """Test initialization with empty ticks."""
        manager = TickManager([])
        assert manager.num_ticks == 0
        assert len(manager) == 0

    def test_init_with_none_ticks(self):
        """Test initialization with None ticks."""
        manager = TickManager(None)
        assert manager.num_ticks == 0
        assert len(manager) == 0

    def test_init_with_invalid_tick_raises_error(self):
        """Test initialization with invalid tick raises error."""
        with pytest.raises(
            TypeError, match="Tick at index 0 must be NumericTicks or CategoricalTicks"
        ):
            TickManager(["invalid"])  # type: ignore

    def test_init_creates_correct_tick_objects(self):
        """Test that initialization creates correct tick objects."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])

        # Check that the objects are of the correct types
        assert isinstance(manager.get_ticks(0), NumericTicks)
        assert isinstance(manager.get_ticks(1), CategoricalTicks)

    def test_get_ticks_valid_index(self):
        """Test getting ticks with valid index."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])

        ticks_0 = manager.get_ticks(0)
        ticks_1 = manager.get_ticks(1)

        assert isinstance(ticks_0, NumericTicks)
        assert isinstance(ticks_1, CategoricalTicks)

    def test_get_ticks_invalid_index_raises_error(self):
        """Test getting ticks with invalid index raises error."""
        numeric_ticks = NumericTicks()
        manager = TickManager([numeric_ticks])

        with pytest.raises(IndexError, match="Index 1 out of bounds"):
            manager.get_ticks(1)

        with pytest.raises(IndexError, match="Index -1 out of bounds"):
            manager.get_ticks(-1)

    def test_get_numeric_ticks_valid_index(self):
        """Test getting numeric ticks with valid index."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])

        numeric_ticks_result = manager.get_numeric_ticks(0)
        assert isinstance(numeric_ticks_result, NumericTicks)

    def test_get_numeric_ticks_wrong_type_raises_error(self):
        """Test getting numeric ticks with wrong type raises error."""
        categorical_ticks = CategoricalTicks()
        manager = TickManager([categorical_ticks])

        with pytest.raises(TypeError, match="Tick 0 is not numeric"):
            manager.get_numeric_ticks(0)

    def test_get_categorical_ticks_valid_index(self):
        """Test getting categorical ticks with valid index."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])

        categorical_ticks_result = manager.get_categorical_ticks(1)
        assert isinstance(categorical_ticks_result, CategoricalTicks)

    def test_get_categorical_ticks_wrong_type_raises_error(self):
        """Test getting categorical ticks with wrong type raises error."""
        numeric_ticks = NumericTicks()
        manager = TickManager([numeric_ticks])

        with pytest.raises(TypeError, match="Tick 0 is not categorical"):
            manager.get_categorical_ticks(0)

    def test_get_tick_type_valid_index(self):
        """Test getting tick type with valid index."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])

        assert manager.get_tick_type(0) == TickType.NUMERIC
        assert manager.get_tick_type(1) == TickType.CATEGORICAL

    def test_get_tick_type_invalid_index_raises_error(self):
        """Test getting tick type with invalid index raises error."""
        numeric_ticks = NumericTicks()
        manager = TickManager([numeric_ticks])

        with pytest.raises(IndexError, match="Index 1 out of bounds"):
            manager.get_tick_type(1)

    def test_remove_ticks_valid_index(self):
        """Test removing ticks with valid index."""
        numeric_ticks1 = NumericTicks()
        categorical_ticks = CategoricalTicks()
        numeric_ticks2 = NumericTicks()
        manager = TickManager([numeric_ticks1, categorical_ticks, numeric_ticks2])
        assert manager.num_ticks == 3

        manager.remove_ticks([1])
        assert manager.num_ticks == 2
        assert manager.get_tick_type(0) == TickType.NUMERIC
        assert manager.get_tick_type(1) == TickType.NUMERIC

    def test_remove_ticks_invalid_index_raises_error(self):
        """Test removing ticks with invalid index raises error."""
        numeric_ticks = NumericTicks()
        manager = TickManager([numeric_ticks])

        with pytest.raises(IndexError, match="Index 1 out of bounds"):
            manager.remove_ticks([1])

    def test_len_magic_method(self):
        """Test __len__ magic method."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])
        assert len(manager) == 2

    def test_getitem_magic_method(self):
        """Test __getitem__ magic method."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])

        assert isinstance(manager[0], NumericTicks)
        assert isinstance(manager[1], CategoricalTicks)

    def test_getitem_invalid_index_raises_error(self):
        """Test __getitem__ with invalid index raises error."""
        numeric_ticks = NumericTicks()
        manager = TickManager([numeric_ticks])

        with pytest.raises(IndexError, match="Index 1 out of bounds"):
            _ = manager[1]

    def test_repr_empty_manager(self):
        """Test __repr__ with empty manager."""
        manager = TickManager([])
        repr_str = repr(manager)
        assert "TickManager(empty)" in repr_str

    def test_repr_single_tick(self):
        """Test __repr__ with single tick."""
        numeric_ticks = NumericTicks()
        manager = TickManager([numeric_ticks])
        repr_str = repr(manager)
        assert "TickManager" in repr_str
        assert "1 ticks" in repr_str
        assert "0:NUMERIC" in repr_str

    def test_repr_multiple_ticks(self):
        """Test __repr__ with multiple ticks."""
        numeric_ticks1 = NumericTicks()
        categorical_ticks = CategoricalTicks()
        numeric_ticks2 = NumericTicks()
        manager = TickManager([numeric_ticks1, categorical_ticks, numeric_ticks2])
        repr_str = repr(manager)
        assert "TickManager" in repr_str
        assert "3 ticks" in repr_str
        assert "0:NUMERIC" in repr_str
        assert "1:CATEGORICAL" in repr_str
        assert "2:NUMERIC" in repr_str

    def test_ticks_property_returns_copy(self):
        """Test that ticks property returns a copy."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])
        ticks = manager.ticks

        assert len(ticks) == 2
        # Modifying the returned list shouldn't affect the manager
        ticks.append(None)  # type: ignore
        assert len(manager.ticks) == 2

    def test_num_ticks_property(self):
        """Test num_ticks property."""
        numeric_ticks1 = NumericTicks()
        categorical_ticks = CategoricalTicks()
        numeric_ticks2 = NumericTicks()
        manager = TickManager([numeric_ticks1, categorical_ticks, numeric_ticks2])
        assert manager.num_ticks == 3

    def test_ticks_property_types(self):
        """Test that ticks property returns correct types."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])
        ticks = manager.ticks

        assert isinstance(ticks[0], NumericTicks)
        assert isinstance(ticks[1], CategoricalTicks)

    def test_integration_with_numeric_ticks(self):
        """Test integration with NumericTicks functionality."""
        numeric_ticks = NumericTicks()
        manager = TickManager([numeric_ticks])
        numeric_ticks_result = manager.get_numeric_ticks(0)

        # Configure the ticks
        numeric_ticks_result.set_ticks_from_range(0, 100, max_ticks=5)

        # Verify the configuration worked
        assert len(numeric_ticks_result) == 5
        assert numeric_ticks_result.locations.get_values()[0] == 0.0
        # The last tick might not be exactly 100.0 due to optimal tick generation
        assert numeric_ticks_result.locations.get_values()[-1] >= 80.0

    def test_integration_with_categorical_ticks(self):
        """Test integration with CategoricalTicks functionality."""
        categorical_ticks = CategoricalTicks()
        manager = TickManager([categorical_ticks])
        categorical_ticks_result = manager.get_categorical_ticks(0)

        # Configure the ticks
        categories = ["Low", "Medium", "High"]
        categorical_ticks_result.set_ticks_from_categories(categories)

        # Verify the configuration worked
        assert len(categorical_ticks_result) == 3
        assert categorical_ticks_result.labels.get_values() == categories
        assert categorical_ticks_result.locations.get_values() == [0, 1, 2]

    def test_mixed_tick_types_management(self):
        """Test managing mixed tick types."""
        numeric_ticks1 = NumericTicks()
        categorical_ticks1 = CategoricalTicks()
        numeric_ticks2 = NumericTicks()
        categorical_ticks2 = CategoricalTicks()
        manager = TickManager(
            [numeric_ticks1, categorical_ticks1, numeric_ticks2, categorical_ticks2]
        )

        # Configure numeric ticks
        manager.get_numeric_ticks(0).set_ticks_from_range(0, 50)
        manager.get_numeric_ticks(2).set_ticks_from_range(100, 200)

        # Configure categorical ticks
        manager.get_categorical_ticks(1).set_ticks_from_categories(["A", "B"])
        manager.get_categorical_ticks(3).set_ticks_from_categories(["X", "Y", "Z"])

        # Verify all configurations
        assert len(manager.get_numeric_ticks(0)) > 0
        assert len(manager.get_categorical_ticks(1)) == 2
        assert len(manager.get_numeric_ticks(2)) > 0
        assert len(manager.get_categorical_ticks(3)) == 3

    def test_remove_ticks_affects_indices(self):
        """Test that removing ticks affects subsequent indices."""
        numeric_ticks1 = NumericTicks()
        categorical_ticks1 = CategoricalTicks()
        numeric_ticks2 = NumericTicks()
        categorical_ticks2 = CategoricalTicks()
        manager = TickManager(
            [numeric_ticks1, categorical_ticks1, numeric_ticks2, categorical_ticks2]
        )

        # Remove the second tick (index 1)
        manager.remove_ticks([1])

        # Verify indices shifted
        assert manager.get_tick_type(0) == TickType.NUMERIC
        assert manager.get_tick_type(1) == TickType.NUMERIC  # Was index 2
        assert manager.get_tick_type(2) == TickType.CATEGORICAL  # Was index 3

        # Verify we can't access the old index 3
        with pytest.raises(IndexError):
            manager.get_tick_type(3)

    def test_tick_type_enum_values(self):
        """Test TickType enum values."""
        assert TickType.NUMERIC == "numeric"
        assert TickType.CATEGORICAL == "categorical"
        assert TickType.NUMERIC.value == "numeric"
        assert TickType.CATEGORICAL.value == "categorical"

    def test_set_ticks_from_types(self):
        """Test setting ticks from types."""
        manager = TickManager()
        manager.set_ticks_from_types([TickType.NUMERIC, TickType.CATEGORICAL])

        assert manager.num_ticks == 2
        assert isinstance(manager.get_ticks(0), NumericTicks)
        assert isinstance(manager.get_ticks(1), CategoricalTicks)

    def test_set_ticks_from_types_empty_raises_error(self):
        """Test setting ticks from empty types raises error."""
        manager = TickManager()
        with pytest.raises(ValueError, match="tick_types cannot be empty"):
            manager.set_ticks_from_types([])

    def test_set_ticks_from_types_invalid_type_raises_error(self):
        """Test setting ticks from invalid type raises error."""
        manager = TickManager()
        with pytest.raises(TypeError, match="tick_type at index 0 must be a TickType"):
            manager.set_ticks_from_types(["invalid"])  # type: ignore

    def test_append_ticks(self):
        """Test appending ticks."""
        numeric_ticks = NumericTicks()
        manager = TickManager([numeric_ticks])
        assert manager.num_ticks == 1

        categorical_ticks = CategoricalTicks()
        manager.append_ticks(categorical_ticks)
        assert manager.num_ticks == 2
        assert isinstance(manager.get_ticks(1), CategoricalTicks)

    def test_append_ticks_invalid_type_raises_error(self):
        """Test appending invalid tick type raises error."""
        manager = TickManager()
        with pytest.raises(
            TypeError, match="Tick must be NumericTicks or CategoricalTicks"
        ):
            manager.append_ticks("invalid")  # type: ignore

    def test_clear_ticks(self):
        """Test clearing all ticks."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager([numeric_ticks, categorical_ticks])
        assert manager.num_ticks == 2

        manager.clear_ticks()
        assert manager.num_ticks == 0

    def test_infer_tick_type(self):
        """Test inferring tick type."""
        numeric_ticks = NumericTicks()
        categorical_ticks = CategoricalTicks()
        manager = TickManager()

        assert manager.infer_tick_type(numeric_ticks) == TickType.NUMERIC
        assert manager.infer_tick_type(categorical_ticks) == TickType.CATEGORICAL

    def test_infer_tick_type_invalid_raises_error(self):
        """Test inferring tick type with invalid tick raises error."""
        manager = TickManager()
        with pytest.raises(
            TypeError, match="Tick must be NumericTicks or CategoricalTicks"
        ):
            manager.infer_tick_type("invalid")  # type: ignore
