# """Tests for PlotModel."""

# import pytest
# from paxplot.models.plot_model import PlotModel


# class TestPlotModel:
#     """Test cases for PlotModel."""

#     def test_init_with_valid_data(self):
#         """Test initialization with valid data."""
#         data = [[1, "A", 2.5], [2, "B", 3.0], [3, "A", 1.5]]
#         plot_model = PlotModel(data)

#         assert plot_model.get_column_count() == 3
#         assert plot_model.get_row_count() == 3

#     def test_init_with_empty_data_raises_error(self):
#         """Test initialization with empty data raises error."""
#         with pytest.raises(ValueError, match="Data cannot be empty"):
#             PlotModel([])

#     def test_init_with_invalid_data_raises_error(self):
#         """Test initialization with invalid data raises error."""
#         with pytest.raises(ValueError, match="Data must be a 2D sequence"):
#             PlotModel([1, 2, 3])  # type: ignore

#     def test_init_with_uneven_rows_raises_error(self):
#         """Test initialization with uneven rows raises error."""
#         data = [[1, "A", 2.5], [2, "B"], [3, "A", 1.5]]  # Missing third column
#         with pytest.raises(ValueError, match="has length 2 but expected 3"):
#             PlotModel(data)

#     def test_append_data_valid(self):
#         """Test appending valid data."""
#         data = [[1, "A", 2.5], [2, "B", 3.0]]
#         plot_model = PlotModel(data)

#         initial_rows = plot_model.get_row_count()
#         plot_model.append_data([3, "C", 1.5])

#         assert plot_model.get_row_count() == initial_rows + 1
#         assert plot_model.get_column_count() == 3

#     def test_append_data_invalid_length_raises_error(self):
#         """Test appending data with invalid length raises error."""
#         data = [[1, "A", 2.5], [2, "B", 3.0]]
#         plot_model = PlotModel(data)

#         with pytest.raises(
#             ValueError, match="Row length 2 doesn't match number of columns 3"
#         ):
#             plot_model.append_data([1, "A"])  # Wrong length

#     def test_remove_data_valid(self):
#         """Test removing valid data."""
#         data = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5]]
#         plot_model = PlotModel(data)

#         initial_rows = plot_model.get_row_count()
#         plot_model.remove_data([0, 2])

#         assert plot_model.get_row_count() == initial_rows - 2
#         assert plot_model.get_column_count() == 3

#     def test_remove_data_invalid_index_raises_error(self):
#         """Test removing data with invalid index raises error."""
#         data = [[1, "A", 2.5], [2, "B", 3.0]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError):
#             plot_model.remove_data([10])  # Out of bounds

#     def test_set_data_valid(self):
#         """Test setting valid data."""
#         initial_data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(initial_data)

#         new_data = [[10, "X", 1.1], [20, "Y", 2.2], [30, "Z", 3.3]]
#         plot_model.set_data(new_data)

#         assert plot_model.get_row_count() == 3
#         assert plot_model.get_column_count() == 3

#     def test_set_data_invalid_raises_error(self):
#         """Test setting invalid data raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(ValueError, match="Data cannot be empty"):
#             plot_model.set_data([])

#     def test_set_axis_label_valid(self):
#         """Test setting valid axis label."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         plot_model.set_axis_label(0, "Index")
#         plot_model.set_axis_label(1, "Category")

#         assert plot_model.get_axis_label(0) == "Index"
#         assert plot_model.get_axis_label(1) == "Category"

#     def test_set_axis_label_invalid_index_raises_error(self):
#         """Test setting axis label with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.set_axis_label(10, "Invalid")

#     def test_set_axis_label_empty_raises_error(self):
#         """Test setting empty axis label raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(
#             ValueError, match="Label cannot be empty or whitespace only"
#         ):
#             plot_model.set_axis_label(0, "")

#     def test_set_axis_label_whitespace_only_raises_error(self):
#         """Test setting whitespace-only axis label raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(
#             ValueError, match="Label cannot be empty or whitespace only"
#         ):
#             plot_model.set_axis_label(0, "   ")

#     def test_clear_axis_label_valid(self):
#         """Test clearing valid axis label."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         plot_model.set_axis_label(0, "Index")
#         assert plot_model.get_axis_label(0) == "Index"

#         plot_model.clear_axis_label(0)
#         assert plot_model.get_axis_label(0) is None

#     def test_clear_axis_label_invalid_index_raises_error(self):
#         """Test clearing axis label with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.clear_axis_label(10)

#     def test_set_custom_limit_valid(self):
#         """Test setting valid custom limits."""
#         data = [[1, "A", 2.5], [2, "B", 3.0]]
#         plot_model = PlotModel(data)

#         plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)
#         plot_model.set_custom_limit(2, min_val=1.0, max_val=5.0)

#         min_val, max_val = plot_model.get_custom_limit(0)
#         assert min_val == 0.0
#         assert max_val == 10.0
#         min_val, max_val = plot_model.get_custom_limit(2)
#         assert min_val == 1.0
#         assert max_val == 5.0

#     def test_set_custom_limit_partial(self):
#         """Test setting partial custom limits."""
#         data = [[1, "A", 2.5], [2, "B", 3.0]]
#         plot_model = PlotModel(data)

#         plot_model.set_custom_limit(0, min_val=0.0)
#         plot_model.set_custom_limit(2, max_val=5.0)

#         min_val, max_val = plot_model.get_custom_limit(0)
#         assert min_val == 0.0
#         assert max_val is None
#         min_val, max_val = plot_model.get_custom_limit(2)
#         assert min_val is None
#         assert max_val == 5.0

#     def test_set_custom_limit_invalid_index_raises_error(self):
#         """Test setting custom limit with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.set_custom_limit(10, min_val=0.0)

#     def test_set_custom_limit_invalid_values_raises_error(self):
#         """Test setting custom limit with invalid values raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(
#             ValueError,
#             match="min_val \\(10.0\\) must be less than max_val \\(5.0\\)",
#         ):
#             plot_model.set_custom_limit(0, min_val=10.0, max_val=5.0)

#     def test_clear_custom_limit_valid(self):
#         """Test clearing valid custom limit."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)
#         min_val, max_val = plot_model.get_custom_limit(0)
#         assert min_val is not None or max_val is not None

#         plot_model.clear_custom_limit(0)
#         min_val, max_val = plot_model.get_custom_limit(0)
#         assert min_val is None and max_val is None

#     def test_clear_custom_limit_invalid_index_raises_error(self):
#         """Test clearing custom limit with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.clear_custom_limit(10)

#     def test_get_column_count(self):
#         """Test get_column_count method."""
#         data = [[1, "A", 2.5], [2, "B", 3.0]]
#         plot_model = PlotModel(data)

#         assert plot_model.get_column_count() == 3

#     def test_get_row_count(self):
#         """Test get_row_count method."""
#         data = [[1, "A"], [2, "B"], [3, "C"]]
#         plot_model = PlotModel(data)

#         assert plot_model.get_row_count() == 3

#     def test_get_axis_label_valid(self):
#         """Test get_axis_label method with valid index."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         # Initially no label
#         assert plot_model.get_axis_label(0) is None

#         # Set a label and get it
#         plot_model.set_axis_label(0, "Index")
#         assert plot_model.get_axis_label(0) == "Index"

#     def test_get_axis_label_invalid_index_raises_error(self):
#         """Test get_axis_label with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.get_axis_label(10)

#     def test_get_custom_limit_valid(self):
#         """Test get_custom_limit method with valid index."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         # Initially no limits
#         min_val, max_val = plot_model.get_custom_limit(0)
#         assert min_val is None
#         assert max_val is None

#         # Set limits and get them
#         plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)
#         min_val, max_val = plot_model.get_custom_limit(0)
#         assert min_val == 0.0
#         assert max_val == 10.0

#     def test_get_custom_limit_invalid_index_raises_error(self):
#         """Test get_custom_limit with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.get_custom_limit(10)

#     def test_get_column_type_valid(self):
#         """Test get_column_type method with valid index."""
#         data = [[1, "A", 2.5], [2, "B", 3.0]]
#         plot_model = PlotModel(data)

#         assert plot_model.get_column_type(0) == "numeric"
#         assert plot_model.get_column_type(1) == "categorical"
#         assert plot_model.get_column_type(2) == "numeric"

#     def test_get_column_type_invalid_index_raises_error(self):
#         """Test get_column_type with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.get_column_type(10)

#     def test_get_numeric_values_valid(self):
#         """Test get_numeric_values method with valid numeric column."""
#         data = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5]]
#         plot_model = PlotModel(data)

#         values = plot_model.get_numeric_values(0)
#         assert values == [1.0, 2.0, 3.0]
#         assert isinstance(values, list)
#         assert all(isinstance(v, float) for v in values)

#         values = plot_model.get_numeric_values(2)
#         assert values == [2.5, 3.0, 1.5]

#     def test_get_numeric_values_categorical_column_raises_error(self):
#         """Test get_numeric_values with categorical column raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(
#             TypeError, match="Column 1 is not numeric, it is categorical"
#         ):
#             plot_model.get_numeric_values(1)

#     def test_get_numeric_values_invalid_index_raises_error(self):
#         """Test get_numeric_values with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.get_numeric_values(10)

#     def test_get_categorical_values_valid(self):
#         """Test get_categorical_values method with valid categorical column."""
#         data = [[1, "A", 2.5], [2, "B", 3.0], [3, "A", 1.5]]
#         plot_model = PlotModel(data)

#         values = plot_model.get_categorical_values(1)
#         assert values == ["A", "B", "A"]
#         assert isinstance(values, list)
#         assert all(isinstance(v, str) for v in values)

#     def test_get_categorical_values_numeric_column_raises_error(self):
#         """Test get_categorical_values with numeric column raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(
#             TypeError, match="Column 0 is not categorical, it is numeric"
#         ):
#             plot_model.get_categorical_values(0)

#     def test_get_categorical_values_invalid_index_raises_error(self):
#         """Test get_categorical_values with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.get_categorical_values(10)

#     def test_get_unique_values_valid(self):
#         """Test get_unique_values method with valid categorical column."""
#         data = [[1, "A", 2.5], [2, "B", 3.0], [3, "A", 1.5], [4, "C", 2.0]]
#         plot_model = PlotModel(data)

#         unique_vals = plot_model.get_unique_values(1)
#         assert unique_vals == ["A", "B", "C"]
#         assert isinstance(unique_vals, list)
#         assert all(isinstance(v, str) for v in unique_vals)

#     def test_get_unique_values_numeric_column_raises_error(self):
#         """Test get_unique_values with numeric column raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(
#             TypeError, match="Column 0 is not categorical, it is numeric"
#         ):
#             plot_model.get_unique_values(0)

#     def test_get_unique_values_invalid_index_raises_error(self):
#         """Test get_unique_values with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.get_unique_values(10)

#     def test_data_access_with_nan_values(self):
#         """Test data access methods with NaN values."""
#         data = [[1, "A", 2.5], [None, "B", None], [3, None, 1.5]]
#         plot_model = PlotModel(data)

#         # Test numeric values with NaN
#         values = plot_model.get_numeric_values(0)
#         assert len(values) == 3
#         assert values[0] == 1.0
#         assert str(values[1]) == "nan"  # NaN value
#         assert values[2] == 3.0

#         # Test categorical values with NaN
#         values = plot_model.get_categorical_values(1)
#         assert len(values) == 3
#         assert values[0] == "A"
#         assert values[1] == "B"
#         assert values[2] == "<NaN>"  # NaN representation

#         # Test that numeric values can be used to compute range (ignoring NaN values)
#         numeric_values = plot_model.get_numeric_values(0)
#         non_nan_values = [v for v in numeric_values if str(v) != "nan"]
#         assert min(non_nan_values) == 1.0
#         assert max(non_nan_values) == 3.0

#         # Test unique values with NaN
#         unique_vals = plot_model.get_unique_values(1)
#         assert "<NaN>" in unique_vals
#         assert "A" in unique_vals
#         assert "B" in unique_vals

#     def test_data_access_after_data_modification(self):
#         """Test data access methods after data modification."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         # Initial values
#         assert plot_model.get_numeric_values(0) == [1.0, 2.0]
#         assert plot_model.get_categorical_values(1) == ["A", "B"]

#         # Append data
#         plot_model.append_data([3, "C"])
#         assert plot_model.get_numeric_values(0) == [1.0, 2.0, 3.0]
#         assert plot_model.get_categorical_values(1) == ["A", "B", "C"]

#         # Remove data
#         plot_model.remove_data([0])
#         assert plot_model.get_numeric_values(0) == [2.0, 3.0]
#         assert plot_model.get_categorical_values(1) == ["B", "C"]

#         # Set new data
#         new_data = [[10, "X"], [20, "Y"], [30, "Z"]]
#         plot_model.set_data(new_data)
#         assert plot_model.get_numeric_values(0) == [10.0, 20.0, 30.0]
#         assert plot_model.get_categorical_values(1) == ["X", "Y", "Z"]

#     def test_get_tick_labels_valid(self):
#         """Test get_tick_labels method with valid index."""
#         data = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5]]
#         plot_model = PlotModel(data)

#         # Test numeric column tick labels
#         tick_labels = plot_model.get_tick_labels(0)
#         assert isinstance(tick_labels, list)
#         assert all(isinstance(label, str) for label in tick_labels)
#         assert len(tick_labels) > 0  # Should have some ticks generated

#         # Test categorical column tick labels
#         tick_labels = plot_model.get_tick_labels(1)
#         assert isinstance(tick_labels, list)
#         assert all(isinstance(label, str) for label in tick_labels)
#         assert len(tick_labels) > 0  # Should have some ticks generated

#     def test_get_tick_labels_invalid_index_raises_error(self):
#         """Test get_tick_labels with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.get_tick_labels(10)

#     def test_get_tick_locations_valid(self):
#         """Test get_tick_locations method with valid index."""
#         data = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5]]
#         plot_model = PlotModel(data)

#         # Test numeric column tick locations
#         tick_locations = plot_model.get_tick_locations(0)
#         assert isinstance(tick_locations, list)
#         assert all(isinstance(loc, float) for loc in tick_locations)
#         assert len(tick_locations) > 0  # Should have some ticks generated

#         # Test categorical column tick locations
#         tick_locations = plot_model.get_tick_locations(1)
#         assert isinstance(tick_locations, list)
#         assert all(isinstance(loc, float) for loc in tick_locations)
#         assert len(tick_locations) > 0  # Should have some ticks generated

#     def test_get_tick_locations_invalid_index_raises_error(self):
#         """Test get_tick_locations with invalid index raises error."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         with pytest.raises(IndexError, match="Index 10 out of bounds"):
#             plot_model.get_tick_locations(10)

#     def test_tick_labels_and_locations_consistency(self):
#         """Test that tick labels and locations have consistent lengths."""
#         data = [[1, "A", 2.5], [2, "B", 3.0], [3, "C", 1.5]]
#         plot_model = PlotModel(data)

#         for i in range(plot_model.get_column_count()):
#             tick_labels = plot_model.get_tick_labels(i)
#             tick_locations = plot_model.get_tick_locations(i)

#             # Labels and locations should have the same length
#             assert len(tick_labels) == len(tick_locations)
#             assert len(tick_labels) > 0  # Should have at least one tick

#     def test_tick_access_after_data_modification(self):
#         """Test tick access methods after data modification."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         # Store initial tick info
#         _ = plot_model.get_tick_labels(0)
#         _ = plot_model.get_tick_locations(0)

#         # Append data
#         plot_model.append_data([3, "C"])

#         # Ticks should be updated (may have different values due to range changes)
#         new_labels_0 = plot_model.get_tick_labels(0)
#         new_locations_0 = plot_model.get_tick_locations(0)

#         # Should still have valid tick data
#         assert len(new_labels_0) > 0
#         assert len(new_locations_0) > 0
#         assert len(new_labels_0) == len(new_locations_0)

#         # Set new data
#         new_data = [[10, "X"], [20, "Y"], [30, "Z"]]
#         plot_model.set_data(new_data)

#         # Ticks should be updated for new data
#         final_labels_0 = plot_model.get_tick_labels(0)
#         final_locations_0 = plot_model.get_tick_locations(0)

#         assert len(final_labels_0) > 0
#         assert len(final_locations_0) > 0
#         assert len(final_labels_0) == len(final_locations_0)

#     def test_tick_access_with_nan_values(self):
#         """Test tick access methods with NaN values in data."""
#         data = [[1, "A", 2.5], [None, "B", None], [3, None, 1.5]]
#         plot_model = PlotModel(data)

#         # Should still be able to get tick information even with NaN values
#         for i in range(plot_model.get_column_count()):
#             tick_labels = plot_model.get_tick_labels(i)
#             tick_locations = plot_model.get_tick_locations(i)

#             assert len(tick_labels) > 0
#             assert len(tick_locations) > 0
#             assert len(tick_labels) == len(tick_locations)

#     def test_len_dunder(self):
#         """Test __len__ method."""
#         data = [[1, "A", 2.5], [2, "B", 3.0]]
#         plot_model = PlotModel(data)

#         assert len(plot_model) == 3

#     def test_getitem_dunder(self):
#         """Test __getitem__ method."""
#         data = [[1, "A", 2.5], [2, "B", 3.0]]
#         plot_model = PlotModel(data)

#         column = plot_model[0]
#         assert hasattr(column, "get_values")
#         assert column.get_values() == [1.0, 2.0]

#     def test_repr_dunder(self):
#         """Test __repr__ method."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         repr_str = repr(plot_model)
#         assert "PlotModel" in repr_str
#         assert "2 rows" in repr_str
#         assert "2 columns" in repr_str
#         assert "2 axis labels" in repr_str
#         assert "2 custom limits" in repr_str

#     def test_structures_updated_after_data_modification(self):
#         """Test that structures are updated after data modification."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         # Modify data
#         new_data = [[10, "X", 1.1], [20, "Y", 2.2]]
#         plot_model.set_data(new_data)

#         # Check that column count is updated
#         assert plot_model.get_column_count() == 3  # New column count

#     def test_structures_updated_after_append(self):
#         """Test that structures are maintained after append."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         # Store initial structure counts
#         initial_columns = plot_model.get_column_count()

#         # Append data
#         plot_model.append_data([3, "C"])

#         # Check that column count is maintained
#         assert plot_model.get_column_count() == initial_columns

#     def test_structures_updated_after_remove(self):
#         """Test that structures are maintained after remove."""
#         data = [[1, "A"], [2, "B"], [3, "C"]]
#         plot_model = PlotModel(data)

#         # Store initial structure counts
#         initial_columns = plot_model.get_column_count()

#         # Remove data
#         plot_model.remove_data([0])

#         # Check that column count is maintained
#         assert plot_model.get_column_count() == initial_columns

#     def test_tick_types_correctly_mapped(self):
#         """Test that tick types are correctly mapped from matrix column types."""
#         # This test is no longer relevant since we don't expose tick_manager
#         # The tick manager is still created internally but not exposed
#         # Test passes by default since the functionality is tested elsewhere

#     def test_axis_labels_initialized_with_defaults(self):
#         """Test that axis labels are initialized with default values."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         # Test that axis labels are initialized with default values
#         for i in range(plot_model.get_column_count()):
#             assert plot_model.get_axis_label(i) is None

#     def test_custom_limits_initialized_with_defaults(self):
#         """Test that custom limits are initialized with default values."""
#         data = [[1, "A"], [2, "B"]]
#         plot_model = PlotModel(data)

#         # Test that custom limits are initialized with default values
#         for i in range(plot_model.get_column_count()):
#             min_val, max_val = plot_model.get_custom_limit(i)
#             assert min_val is None and max_val is None
