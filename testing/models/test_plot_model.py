"""Tests for PlotModel."""

import pytest
from paxplot.models.plot_model import PlotModel
from paxplot.structures.matrix import Matrix, ColumnType
from paxplot.structures.tick_manger import TickManager, TickType
from paxplot.structures.labels.axis_label import AxisLabel
from paxplot.structures.limits.custom_axis_limit import CustomAxisLimit


class TestPlotModel:
    """Test cases for PlotModel."""

    def test_init_with_valid_data(self):
        """Test initialization with valid data."""
        data = [[1, "A", 2.5], [2, "B", 3.0], [3, "A", 1.5]]
        plot_model = PlotModel(data)
        
        assert plot_model.get_column_count() == 3
        assert plot_model.get_row_count() == 3
        assert isinstance(plot_model.matrix, Matrix)
        assert isinstance(plot_model.tick_manager, TickManager)
        assert len(plot_model.axis_labels) == 3
        assert len(plot_model.custom_limits) == 3

    def test_init_with_empty_data_raises_error(self):
        """Test initialization with empty data raises error."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            PlotModel([])

    def test_init_with_invalid_data_raises_error(self):
        """Test initialization with invalid data raises error."""
        with pytest.raises(ValueError, match="Data must be a 2D sequence"):
            PlotModel([1, 2, 3])  # type: ignore

    def test_init_with_uneven_rows_raises_error(self):
        """Test initialization with uneven rows raises error."""
        data = [[1, "A", 2.5], [2, "B"], [3, "A", 1.5]]  # Missing third column
        with pytest.raises(ValueError, match="has length 2 but expected 3"):
            PlotModel(data)

    def test_matrix_property(self):
        """Test matrix property returns correct Matrix instance."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        matrix = plot_model.matrix
        assert isinstance(matrix, Matrix)
        assert matrix.num_columns == 2
        assert matrix.num_rows == 2

    def test_tick_manager_property(self):
        """Test tick_manager property returns correct TickManager instance."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        tick_manager = plot_model.tick_manager
        assert isinstance(tick_manager, TickManager)
        assert tick_manager.num_collections == 2

    def test_axis_labels_property_returns_copy(self):
        """Test that axis_labels property returns a copy."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        axis_labels = plot_model.axis_labels
        assert len(axis_labels) == 2
        assert all(isinstance(label, AxisLabel) for label in axis_labels)
        
        # Modifying the returned list shouldn't affect the model
        axis_labels.append(AxisLabel("test"))
        assert len(plot_model.axis_labels) == 2

    def test_custom_limits_property_returns_copy(self):
        """Test that custom_limits property returns a copy."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        custom_limits = plot_model.custom_limits
        assert len(custom_limits) == 2
        assert all(isinstance(limit, CustomAxisLimit) for limit in custom_limits)
        
        # Modifying the returned list shouldn't affect the model
        custom_limits.append(CustomAxisLimit(0.0, 1.0))
        assert len(plot_model.custom_limits) == 2

    def test_append_data_valid(self):
        """Test appending valid data."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(data)
        
        initial_rows = plot_model.get_row_count()
        plot_model.append_data([3, "C", 1.5])
        
        assert plot_model.get_row_count() == initial_rows + 1
        assert plot_model.get_column_count() == 3

    def test_append_data_invalid_length_raises_error(self):
        """Test appending data with invalid length raises error."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(data)
        
        with pytest.raises(ValueError, match="Row length 2 doesn't match number of columns 3"):
            plot_model.append_data([1, "A"])  # Wrong length

    def test_remove_data_valid(self):
        """Test removing valid data."""
        data = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5]]
        plot_model = PlotModel(data)
        
        initial_rows = plot_model.get_row_count()
        plot_model.remove_data([0, 2])
        
        assert plot_model.get_row_count() == initial_rows - 2
        assert plot_model.get_column_count() == 3

    def test_remove_data_invalid_index_raises_error(self):
        """Test removing data with invalid index raises error."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(data)
        
        with pytest.raises(IndexError):
            plot_model.remove_data([10])  # Out of bounds

    def test_set_data_valid(self):
        """Test setting valid data."""
        initial_data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(initial_data)
        
        new_data = [[10, "X", 1.1], [20, "Y", 2.2], [30, "Z", 3.3]]
        plot_model.set_data(new_data)
        
        assert plot_model.get_row_count() == 3
        assert plot_model.get_column_count() == 3

    def test_set_data_invalid_raises_error(self):
        """Test setting invalid data raises error."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            plot_model.set_data([])

    def test_set_axis_label_valid(self):
        """Test setting valid axis label."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        plot_model.set_axis_label(0, "Index")
        plot_model.set_axis_label(1, "Category")
        
        assert plot_model.axis_labels[0].label == "Index"
        assert plot_model.axis_labels[1].label == "Category"

    def test_set_axis_label_invalid_index_raises_error(self):
        """Test setting axis label with invalid index raises error."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        with pytest.raises(IndexError, match="Index 10 out of bounds"):
            plot_model.set_axis_label(10, "Invalid")

    def test_set_axis_label_empty_raises_error(self):
        """Test setting empty axis label raises error."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        with pytest.raises(ValueError, match="Label cannot be empty or whitespace only"):
            plot_model.set_axis_label(0, "")

    def test_set_axis_label_whitespace_only_raises_error(self):
        """Test setting whitespace-only axis label raises error."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        with pytest.raises(ValueError, match="Label cannot be empty or whitespace only"):
            plot_model.set_axis_label(0, "   ")

    def test_clear_axis_label_valid(self):
        """Test clearing valid axis label."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        plot_model.set_axis_label(0, "Index")
        assert plot_model.axis_labels[0].label == "Index"
        
        plot_model.clear_axis_label(0)
        assert plot_model.axis_labels[0].label is None

    def test_clear_axis_label_invalid_index_raises_error(self):
        """Test clearing axis label with invalid index raises error."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        with pytest.raises(IndexError, match="Index 10 out of bounds"):
            plot_model.clear_axis_label(10)

    def test_set_custom_limit_valid(self):
        """Test setting valid custom limits."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(data)
        
        plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)
        plot_model.set_custom_limit(2, min_val=1.0, max_val=5.0)
        
        assert plot_model.custom_limits[0].min_val == 0.0
        assert plot_model.custom_limits[0].max_val == 10.0
        assert plot_model.custom_limits[2].min_val == 1.0
        assert plot_model.custom_limits[2].max_val == 5.0

    def test_set_custom_limit_partial(self):
        """Test setting partial custom limits."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(data)
        
        plot_model.set_custom_limit(0, min_val=0.0)
        plot_model.set_custom_limit(2, max_val=5.0)
        
        assert plot_model.custom_limits[0].min_val == 0.0
        assert plot_model.custom_limits[0].max_val is None
        assert plot_model.custom_limits[2].min_val is None
        assert plot_model.custom_limits[2].max_val == 5.0

    def test_set_custom_limit_invalid_index_raises_error(self):
        """Test setting custom limit with invalid index raises error."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        with pytest.raises(IndexError, match="Index 10 out of bounds"):
            plot_model.set_custom_limit(10, min_val=0.0)

    def test_set_custom_limit_invalid_values_raises_error(self):
        """Test setting custom limit with invalid values raises error."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        with pytest.raises(ValueError, match="min_val \\(10.0\\) must be less than max_val \\(5.0\\)"):
            plot_model.set_custom_limit(0, min_val=10.0, max_val=5.0)

    def test_clear_custom_limit_valid(self):
        """Test clearing valid custom limit."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)
        assert plot_model.custom_limits[0].is_set()
        
        plot_model.clear_custom_limit(0)
        assert not plot_model.custom_limits[0].is_set()

    def test_clear_custom_limit_invalid_index_raises_error(self):
        """Test clearing custom limit with invalid index raises error."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        with pytest.raises(IndexError, match="Index 10 out of bounds"):
            plot_model.clear_custom_limit(10)

    def test_get_column_count(self):
        """Test get_column_count method."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(data)
        
        assert plot_model.get_column_count() == 3

    def test_get_row_count(self):
        """Test get_row_count method."""
        data = [[1, "A"], [2, "B"], [3, "C"]]
        plot_model = PlotModel(data)
        
        assert plot_model.get_row_count() == 3

    def test_len_dunder(self):
        """Test __len__ method."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(data)
        
        assert len(plot_model) == 3

    def test_getitem_dunder(self):
        """Test __getitem__ method."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]
        plot_model = PlotModel(data)
        
        column = plot_model[0]
        assert hasattr(column, 'values')
        assert column.values == [1.0, 2.0]

    def test_repr_dunder(self):
        """Test __repr__ method."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        repr_str = repr(plot_model)
        assert "PlotModel" in repr_str
        assert "2 rows" in repr_str
        assert "2 columns" in repr_str
        assert "2 axis labels" in repr_str
        assert "2 custom limits" in repr_str

    def test_structures_updated_after_data_modification(self):
        """Test that structures are updated after data modification."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        # Store initial structure counts
        initial_axis_labels = len(plot_model.axis_labels)
        initial_custom_limits = len(plot_model.custom_limits)
        initial_tick_collections = plot_model.tick_manager.num_collections
        
        # Modify data
        new_data = [[10, "X", 1.1], [20, "Y", 2.2]]
        plot_model.set_data(new_data)
        
        # Check that structures are updated
        assert len(plot_model.axis_labels) == 3  # New column count
        assert len(plot_model.custom_limits) == 3  # New column count
        assert plot_model.tick_manager.num_collections == 3  # New column count

    def test_structures_updated_after_append(self):
        """Test that structures are maintained after append."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        # Store initial structure counts
        initial_axis_labels = len(plot_model.axis_labels)
        initial_custom_limits = len(plot_model.custom_limits)
        initial_tick_collections = plot_model.tick_manager.num_collections
        
        # Append data
        plot_model.append_data([3, "C"])
        
        # Check that structures are maintained
        assert len(plot_model.axis_labels) == initial_axis_labels
        assert len(plot_model.custom_limits) == initial_custom_limits
        assert plot_model.tick_manager.num_collections == initial_tick_collections

    def test_structures_updated_after_remove(self):
        """Test that structures are maintained after remove."""
        data = [[1, "A"], [2, "B"], [3, "C"]]
        plot_model = PlotModel(data)
        
        # Store initial structure counts
        initial_axis_labels = len(plot_model.axis_labels)
        initial_custom_limits = len(plot_model.custom_limits)
        initial_tick_collections = plot_model.tick_manager.num_collections
        
        # Remove data
        plot_model.remove_data([0])
        
        # Check that structures are maintained
        assert len(plot_model.axis_labels) == initial_axis_labels
        assert len(plot_model.custom_limits) == initial_custom_limits
        assert plot_model.tick_manager.num_collections == initial_tick_collections

    def test_tick_types_correctly_mapped(self):
        """Test that tick types are correctly mapped from matrix column types."""
        data = [[1, "A", 2.5], [2, "B", 3.0]]  # numeric, categorical, numeric
        plot_model = PlotModel(data)
        
        # Check that tick types match column types
        assert plot_model.tick_manager.get_tick_type(0) == TickType.NUMERIC
        assert plot_model.tick_manager.get_tick_type(1) == TickType.CATEGORICAL
        assert plot_model.tick_manager.get_tick_type(2) == TickType.NUMERIC

    def test_axis_labels_initialized_with_defaults(self):
        """Test that axis labels are initialized with default values."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        for label in plot_model.axis_labels:
            assert isinstance(label, AxisLabel)
            assert label.label is None

    def test_custom_limits_initialized_with_defaults(self):
        """Test that custom limits are initialized with default values."""
        data = [[1, "A"], [2, "B"]]
        plot_model = PlotModel(data)
        
        for limit in plot_model.custom_limits:
            assert isinstance(limit, CustomAxisLimit)
            assert not limit.is_set()
