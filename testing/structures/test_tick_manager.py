"""Tests for TickManager."""

import pytest
from paxplot.structures.tick_manger import TickManager, TickType
from paxplot.structures.ticks.numeric_ticks import NumericTicks
from paxplot.structures.ticks.categorical_ticks import CategoricalTicks


class TestTickManager:
    """Test cases for TickManager."""

    def test_init_with_valid_tick_types(self):
        """Test initialization with valid tick types."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])
        assert manager.num_collections == 2
        assert len(manager) == 2

    def test_init_with_single_tick_type(self):
        """Test initialization with single tick type."""
        manager = TickManager([TickType.NUMERIC])
        assert manager.num_collections == 1
        assert manager.get_tick_type(0) == TickType.NUMERIC

    def test_init_with_empty_tick_types_raises_error(self):
        """Test initialization with empty tick types raises error."""
        with pytest.raises(ValueError, match="tick_types cannot be empty"):
            TickManager([])

    def test_init_with_invalid_tick_type_raises_error(self):
        """Test initialization with invalid tick type raises error."""
        with pytest.raises(
            TypeError, match="tick_type at index 0 must be a TickType"
        ):
            TickManager(["invalid"])  # type: ignore

    def test_init_creates_correct_tick_objects(self):
        """Test that initialization creates correct tick objects."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])

        # Check that the objects are of the correct types
        assert isinstance(manager.get_tick_collection(0), NumericTicks)
        assert isinstance(manager.get_tick_collection(1), CategoricalTicks)

    def test_get_tick_collection_valid_index(self):
        """Test getting tick collection with valid index."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])

        collection_0 = manager.get_tick_collection(0)
        collection_1 = manager.get_tick_collection(1)

        assert isinstance(collection_0, NumericTicks)
        assert isinstance(collection_1, CategoricalTicks)

    def test_get_tick_collection_invalid_index_raises_error(self):
        """Test getting tick collection with invalid index raises error."""
        manager = TickManager([TickType.NUMERIC])

        with pytest.raises(IndexError, match="Index 1 out of bounds"):
            manager.get_tick_collection(1)

        with pytest.raises(IndexError, match="Index -1 out of bounds"):
            manager.get_tick_collection(-1)

    def test_get_numeric_ticks_valid_index(self):
        """Test getting numeric ticks with valid index."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])

        numeric_ticks = manager.get_numeric_ticks(0)
        assert isinstance(numeric_ticks, NumericTicks)

    def test_get_numeric_ticks_wrong_type_raises_error(self):
        """Test getting numeric ticks with wrong type raises error."""
        manager = TickManager([TickType.CATEGORICAL])

        with pytest.raises(TypeError, match="Collection 0 is not numeric"):
            manager.get_numeric_ticks(0)

    def test_get_categorical_ticks_valid_index(self):
        """Test getting categorical ticks with valid index."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])

        categorical_ticks = manager.get_categorical_ticks(1)
        assert isinstance(categorical_ticks, CategoricalTicks)

    def test_get_categorical_ticks_wrong_type_raises_error(self):
        """Test getting categorical ticks with wrong type raises error."""
        manager = TickManager([TickType.NUMERIC])

        with pytest.raises(TypeError, match="Collection 0 is not categorical"):
            manager.get_categorical_ticks(0)

    def test_get_tick_type_valid_index(self):
        """Test getting tick type with valid index."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])

        assert manager.get_tick_type(0) == TickType.NUMERIC
        assert manager.get_tick_type(1) == TickType.CATEGORICAL

    def test_get_tick_type_invalid_index_raises_error(self):
        """Test getting tick type with invalid index raises error."""
        manager = TickManager([TickType.NUMERIC])

        with pytest.raises(IndexError, match="Index 1 out of bounds"):
            manager.get_tick_type(1)

    def test_remove_tick_collection_valid_index(self):
        """Test removing tick collection with valid index."""
        manager = TickManager(
            [TickType.NUMERIC, TickType.CATEGORICAL, TickType.NUMERIC]
        )
        assert manager.num_collections == 3

        manager.remove_tick_collection(1)
        assert manager.num_collections == 2
        assert manager.get_tick_type(0) == TickType.NUMERIC
        assert manager.get_tick_type(1) == TickType.NUMERIC

    def test_remove_tick_collection_invalid_index_raises_error(self):
        """Test removing tick collection with invalid index raises error."""
        manager = TickManager([TickType.NUMERIC])

        with pytest.raises(IndexError, match="Index 1 out of bounds"):
            manager.remove_tick_collection(1)

    def test_len_magic_method(self):
        """Test __len__ magic method."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])
        assert len(manager) == 2

    def test_getitem_magic_method(self):
        """Test __getitem__ magic method."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])

        assert isinstance(manager[0], NumericTicks)
        assert isinstance(manager[1], CategoricalTicks)

    def test_getitem_invalid_index_raises_error(self):
        """Test __getitem__ with invalid index raises error."""
        manager = TickManager([TickType.NUMERIC])

        with pytest.raises(IndexError, match="Index 1 out of bounds"):
            _ = manager[1]

    def test_repr_empty_manager(self):
        """Test __repr__ with empty manager."""
        # This test would require a way to create an empty manager
        # For now, we'll test with a single collection
        manager = TickManager([TickType.NUMERIC])
        repr_str = repr(manager)
        assert "TickManager" in repr_str
        assert "1 collections" in repr_str
        assert "0:NUMERIC" in repr_str

    def test_repr_multiple_collections(self):
        """Test __repr__ with multiple collections."""
        manager = TickManager(
            [TickType.NUMERIC, TickType.CATEGORICAL, TickType.NUMERIC]
        )
        repr_str = repr(manager)
        assert "TickManager" in repr_str
        assert "3 collections" in repr_str
        assert "0:NUMERIC" in repr_str
        assert "1:CATEGORICAL" in repr_str
        assert "2:NUMERIC" in repr_str

    def test_tick_collections_property_returns_copy(self):
        """Test that tick_collections property returns a copy."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])
        collections = manager.tick_collections

        assert len(collections) == 2
        # Modifying the returned list shouldn't affect the manager
        collections.append(None)  # type: ignore
        assert len(manager.tick_collections) == 2

    def test_num_collections_property(self):
        """Test num_collections property."""
        manager = TickManager(
            [TickType.NUMERIC, TickType.CATEGORICAL, TickType.NUMERIC]
        )
        assert manager.num_collections == 3

    def test_tick_collections_property_types(self):
        """Test that tick_collections property returns correct types."""
        manager = TickManager([TickType.NUMERIC, TickType.CATEGORICAL])
        collections = manager.tick_collections

        assert isinstance(collections[0], NumericTicks)
        assert isinstance(collections[1], CategoricalTicks)

    def test_integration_with_numeric_ticks(self):
        """Test integration with NumericTicks functionality."""
        manager = TickManager([TickType.NUMERIC])
        numeric_ticks = manager.get_numeric_ticks(0)

        # Configure the ticks
        numeric_ticks.set_ticks_from_range(0, 100, max_ticks=5)

        # Verify the configuration worked
        assert len(numeric_ticks) == 5
        assert numeric_ticks.locations.get_values()[0] == 0.0
        # The last tick might not be exactly 100.0 due to optimal tick generation
        assert numeric_ticks.locations.get_values()[-1] >= 80.0

    def test_integration_with_categorical_ticks(self):
        """Test integration with CategoricalTicks functionality."""
        manager = TickManager([TickType.CATEGORICAL])
        categorical_ticks = manager.get_categorical_ticks(0)

        # Configure the ticks
        categories = ["Low", "Medium", "High"]
        categorical_ticks.set_ticks_from_categories(categories)

        # Verify the configuration worked
        assert len(categorical_ticks) == 3
        assert categorical_ticks.labels.get_values() == categories
        assert categorical_ticks.locations.get_values() == [0, 1, 2]

    def test_mixed_tick_types_management(self):
        """Test managing mixed tick types."""
        manager = TickManager(
            [
                TickType.NUMERIC,
                TickType.CATEGORICAL,
                TickType.NUMERIC,
                TickType.CATEGORICAL,
            ]
        )

        # Configure numeric ticks
        manager.get_numeric_ticks(0).set_ticks_from_range(0, 50)
        manager.get_numeric_ticks(2).set_ticks_from_range(100, 200)

        # Configure categorical ticks
        manager.get_categorical_ticks(1).set_ticks_from_categories(["A", "B"])
        manager.get_categorical_ticks(3).set_ticks_from_categories(
            ["X", "Y", "Z"]
        )

        # Verify all configurations
        assert len(manager.get_numeric_ticks(0)) > 0
        assert len(manager.get_categorical_ticks(1)) == 2
        assert len(manager.get_numeric_ticks(2)) > 0
        assert len(manager.get_categorical_ticks(3)) == 3

    def test_remove_collection_affects_indices(self):
        """Test that removing a collection affects subsequent indices."""
        manager = TickManager(
            [
                TickType.NUMERIC,  # index 0
                TickType.CATEGORICAL,  # index 1
                TickType.NUMERIC,  # index 2
                TickType.CATEGORICAL,  # index 3
            ]
        )

        # Remove the second collection (index 1)
        manager.remove_tick_collection(1)

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
