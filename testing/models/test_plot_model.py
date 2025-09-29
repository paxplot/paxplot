"""Tests for PlotModel."""

import pytest
from paxplot.models.plot_model import PlotModel


class TestPlotModel:
    """Test cases for PlotModel."""

    def test_init_with_valid_values(self):
        """Test initialization with valid values."""
        values = [[1, "A", 2.5], [2, "B", 3.0], [3, "A", 1.5]]
        plot_model = PlotModel(values)

        assert len(plot_model) == 3
        assert plot_model.array_manager.num_arrays == 3

    def test_init_with_empty_values_raises_error(self):
        """Test initialization with empty values raises error."""
        with pytest.raises(ValueError):
            PlotModel([])

    def test_init_with_invalid_values_raises_error(self):
        """Test initialization with invalid values raises error."""
        with pytest.raises(ValueError):
            PlotModel([1, 2, 3])  # type: ignore

    def test_init_with_uneven_rows_raises_error(self):
        """Test initialization with uneven rows raises error."""
        values = [[1, "A", 2.5], [2, "B"], [3, "A", 1.5]]  # Missing third column
        with pytest.raises(ValueError):
            PlotModel(values)

    def test_init_with_no_values_creates_empty_model(self):
        """Test initialization with no values creates empty model."""
        plot_model = PlotModel()
        
        assert len(plot_model) == 0
        assert plot_model.array_manager.num_arrays == 0
        assert len(plot_model.axis_labels) == 0
        assert len(plot_model.custom_limits) == 0

    def test_append_values_valid(self):
        """Test appending valid values."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        initial_rows = len(plot_model.array_manager.get_array(0))
        plot_model.append_values([3, "C", 1.5])

        assert len(plot_model.array_manager.get_array(0)) == initial_rows + 1
        assert plot_model.array_manager.num_arrays == 3

    def test_append_values_invalid_length_raises_error(self):
        """Test appending values with invalid length raises error."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        with pytest.raises(ValueError):
            plot_model.append_values([1, "A"])  # Wrong length

    def test_remove_values_valid(self):
        """Test removing valid values."""
        values = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5]]
        plot_model = PlotModel(values)

        initial_rows = len(plot_model.array_manager.get_array(0))
        plot_model.remove_values([0, 2])

        assert len(plot_model.array_manager.get_array(0)) == initial_rows - 2
        assert plot_model.array_manager.num_arrays == 3

    def test_remove_values_invalid_index_raises_error(self):
        """Test removing values with invalid index raises error."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        with pytest.raises(IndexError):
            plot_model.remove_values([10])  # Out of bounds

    def test_set_values_valid(self):
        """Test setting valid values."""
        initial_values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(initial_values)

        new_values = [[10, "X", 1.1], [20, "Y", 2.2], [30, "Z", 3.3]]
        plot_model.set_values(new_values)

        assert len(plot_model.array_manager.get_array(0)) == 3
        assert plot_model.array_manager.num_arrays == 3

    def test_set_values_invalid_raises_error(self):
        """Test setting invalid values raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(ValueError):
            plot_model.set_values([])

    def test_set_axis_label_valid(self):
        """Test setting valid axis label."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        plot_model.set_axis_label(0, "Index")
        plot_model.set_axis_label(1, "Category")

        assert plot_model.get_axis_label(0) == "Index"
        assert plot_model.get_axis_label(1) == "Category"

    def test_set_axis_label_invalid_index_raises_error(self):
        """Test setting axis label with invalid index raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(IndexError):
            plot_model.set_axis_label(10, "Invalid")

    def test_set_axis_label_empty_raises_error(self):
        """Test setting empty axis label raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(ValueError):
            plot_model.set_axis_label(0, "")

    def test_set_axis_label_whitespace_only_raises_error(self):
        """Test setting whitespace-only axis label raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(ValueError):
            plot_model.set_axis_label(0, "   ")

    def test_clear_axis_label_valid(self):
        """Test clearing valid axis label."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        plot_model.set_axis_label(0, "Index")
        assert plot_model.get_axis_label(0) == "Index"

        plot_model.clear_axis_label(0)
        assert plot_model.get_axis_label(0) is None

    def test_clear_axis_label_invalid_index_raises_error(self):
        """Test clearing axis label with invalid index raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(IndexError):
            plot_model.clear_axis_label(10)

    def test_set_custom_limit_valid(self):
        """Test setting valid custom limits."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)
        plot_model.set_custom_limit(2, min_val=1.0, max_val=5.0)

        min_val, max_val = plot_model.get_custom_limit(0)
        assert min_val == 0.0
        assert max_val == 10.0
        min_val, max_val = plot_model.get_custom_limit(2)
        assert min_val == 1.0
        assert max_val == 5.0

    def test_set_custom_limit_partial(self):
        """Test setting partial custom limits."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        plot_model.set_custom_limit(0, min_val=0.0)
        plot_model.set_custom_limit(2, max_val=5.0)

        min_val, max_val = plot_model.get_custom_limit(0)
        assert min_val == 0.0
        assert max_val is None
        min_val, max_val = plot_model.get_custom_limit(2)
        assert min_val is None
        assert max_val == 5.0

    def test_set_custom_limit_invalid_index_raises_error(self):
        """Test setting custom limit with invalid index raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(IndexError):
            plot_model.set_custom_limit(10, min_val=0.0)

    def test_set_custom_limit_invalid_values_raises_error(self):
        """Test setting custom limit with invalid values raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(ValueError):
            plot_model.set_custom_limit(0, min_val=10.0, max_val=5.0)

    def test_clear_custom_limit_valid(self):
        """Test clearing valid custom limit."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)
        min_val, max_val = plot_model.get_custom_limit(0)
        assert min_val is not None or max_val is not None

        plot_model.clear_custom_limit(0)
        min_val, max_val = plot_model.get_custom_limit(0)
        assert min_val is None and max_val is None

    def test_clear_custom_limit_invalid_index_raises_error(self):
        """Test clearing custom limit with invalid index raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(IndexError):
            plot_model.clear_custom_limit(10)

    # Property Tests
    def test_array_manager_property(self):
        """Test array_manager property access."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        array_manager = plot_model.array_manager
        assert array_manager.num_arrays == 3
        assert len(array_manager.get_array(0)) == 2

    def test_tick_manager_property(self):
        """Test tick_manager property access."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        tick_manager = plot_model.tick_manager
        assert tick_manager is not None
        # Should have ticks for each column
        assert len(tick_manager.get_ticks(0)) is not None

    def test_axis_labels_property(self):
        """Test axis_labels property access."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        axis_labels = plot_model.axis_labels
        assert len(axis_labels) == 3
        # Initially all labels should be None
        assert all(label.label is None for label in axis_labels)

    def test_custom_limits_property(self):
        """Test custom_limits property access."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        custom_limits = plot_model.custom_limits
        assert len(custom_limits) == 3
        # Initially all limits should be None
        assert all(limit.min_val is None and limit.max_val is None for limit in custom_limits)

    def test_get_axis_label_valid(self):
        """Test get_axis_label method with valid index."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        # Initially no label
        assert plot_model.get_axis_label(0) is None

        # Set a label and get it
        plot_model.set_axis_label(0, "Index")
        assert plot_model.get_axis_label(0) == "Index"

    def test_get_axis_label_invalid_index_raises_error(self):
        """Test get_axis_label with invalid index raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(IndexError):
            plot_model.get_axis_label(10)

    def test_get_custom_limit_valid(self):
        """Test get_custom_limit method with valid index."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        # Initially no limits
        min_val, max_val = plot_model.get_custom_limit(0)
        assert min_val is None
        assert max_val is None

        # Set limits and get them
        plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)
        min_val, max_val = plot_model.get_custom_limit(0)
        assert min_val == 0.0
        assert max_val == 10.0

    def test_get_custom_limit_invalid_index_raises_error(self):
        """Test get_custom_limit with invalid index raises error."""
        values = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(values)

        with pytest.raises(IndexError):
            plot_model.get_custom_limit(10)

    def test_len_dunder(self):
        """Test __len__ method."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        assert len(plot_model) == 3

    def test_repr_dunder(self):
        """Test __repr__ method."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        repr_str = repr(plot_model)
        assert "PlotModel" in repr_str
        assert "2 rows" in repr_str
        assert "3 columns" in repr_str

    def test_structures_updated_after_values_modification(self):
        """Test that structures are updated after values modification."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        # Set some labels and limits
        plot_model.set_axis_label(0, "Index")
        plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)

        # Modify values
        plot_model.append_values([3, "C", 1.5])

        # Structures should be updated
        assert len(plot_model.axis_labels) == 3
        assert len(plot_model.custom_limits) == 3
        # Labels and limits should be preserved
        assert plot_model.get_axis_label(0) == "Index"
        min_val, max_val = plot_model.get_custom_limit(0)
        assert min_val == 0.0
        assert max_val == 10.0

    def test_structures_updated_after_append(self):
        """Test that structures are updated after append."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        initial_axis_labels = len(plot_model.axis_labels)
        initial_custom_limits = len(plot_model.custom_limits)

        plot_model.append_values([3, "C", 1.5])

        # Structures should be updated
        assert len(plot_model.axis_labels) == initial_axis_labels
        assert len(plot_model.custom_limits) == initial_custom_limits

    def test_structures_updated_after_remove(self):
        """Test that structures are updated after remove."""
        values = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5]]
        plot_model = PlotModel(values)

        initial_axis_labels = len(plot_model.axis_labels)
        initial_custom_limits = len(plot_model.custom_limits)

        plot_model.remove_values([0])

        # Structures should be updated
        assert len(plot_model.axis_labels) == initial_axis_labels
        assert len(plot_model.custom_limits) == initial_custom_limits

    def test_axis_labels_initialized_with_defaults(self):
        """Test that axis labels are initialized with defaults."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        axis_labels = plot_model.axis_labels
        assert len(axis_labels) == 3
        # Initially all labels should be None
        assert all(label.label is None for label in axis_labels)

    def test_custom_limits_initialized_with_defaults(self):
        """Test that custom limits are initialized with defaults."""
        values = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(values)

        custom_limits = plot_model.custom_limits
        assert len(custom_limits) == 3
        # Initially all limits should be None
        assert all(limit.min_val is None and limit.max_val is None for limit in custom_limits)