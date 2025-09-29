"""Plot model for PaxPlot.

This module defines the PlotModel class that serves as the main interface
for creating and managing plot structures in PaxPlot.
"""

from typing import List, Optional, Sequence, Union

from ..structures.array_manager import ArrayManager, ArrayType
from ..structures.tick_manger import TickManager, TickType
from ..structures.labels.axis_label import AxisLabel
from ..structures.limits.custom_axis_limit import CustomAxisLimit


class PlotModel:
    """
    Main interface for creating and managing plot structures.

    The PlotModel serves as the primary interface for users to interact with
    PaxPlot values. It manages an ArrayManager for values storage and automatically
    creates and maintains associated structures including TickManager,
    AxisLabels, and CustomAxisLimits.

    Parameters
    ----------
    values : Sequence[Sequence[Union[str, int, float]]], optional
        Initial values as a 2D sequence where each row is a sequence of values
        and each column should be consistently typed. If None, creates an
        empty model that can be populated later.

    Examples
    --------
    >>> # Initialize with values
    >>> values = [
    ...     [1, 'A', 2.5],
    ...     [2, 'B', 3.0],
    ...     [3, 'A', 1.5]
    ... ]
    >>> plot_model = PlotModel(values)
    >>> print(plot_model.array_manager.num_arrays)  # 3
    >>> print(plot_model.array_manager.get_array(0).length)  # 3
    >>>
    >>> # Append new values
    >>> plot_model.append_values([4, 'C', 2.0])
    >>> print(plot_model.array_manager.get_array(0).length)  # 4
    >>>
    >>> # Remove values by indices
    >>> plot_model.remove_values([0, 2])
    >>> print(plot_model.array_manager.get_array(0).length)  # 2
    >>>
    >>> # Access structures
    >>> array_manager = plot_model.array_manager
    >>> tick_manager = plot_model.tick_manager
    >>> axis_labels = plot_model.axis_labels
    >>> custom_limits = plot_model.custom_limits
    >>>
    >>> # Manage axis labels
    >>> plot_model.set_axis_label(0, "X Values")
    >>> print(plot_model.get_axis_label(0))   # "X Values"
    >>> plot_model.clear_axis_label(0)
    >>> print(plot_model.get_axis_label(0))   # None
    >>>
    >>> # Manage custom limits
    >>> plot_model.set_custom_limit(0, min_val=0.0, max_val=10.0)
    >>> min_val, max_val = plot_model.get_custom_limit(0)
    >>> print(f"Limits: {min_val} to {max_val}")  # Limits: 0.0 to 10.0
    >>> plot_model.clear_custom_limit(0)
    >>> min_val, max_val = plot_model.get_custom_limit(0)
    >>> print(f"Limits: {min_val} to {max_val}")  # Limits: None to None
    """

    def __init__(
        self, values: Optional[Sequence[Sequence[Union[str, int, float]]]] = None
    ) -> None:
        """
        Initialize PlotModel with optional initial values.

        Parameters
        ----------
        values : Sequence[Sequence[Union[str, int, float]]], optional
            Initial values as a 2D sequence. If None, creates an empty model.

        Raises
        ------
        ValueError
            If the values structure is invalid (empty or inconsistent row lengths).
        """
        # Initialize with empty structures first, then use set_values method
        self._array_manager = ArrayManager([])
        self._tick_manager = TickManager([])
        self._axis_labels = []
        self._custom_limits = []
        
        if values is not None:
            self.set_values(values)

    # Properties
    @property
    def array_manager(self) -> ArrayManager:
        """
        Get the underlying array manager.

        Returns
        -------
        ArrayManager
            The array manager containing all values columns.
        """
        return self._array_manager

    @property
    def tick_manager(self) -> TickManager:
        """
        Get the tick manager for all columns.

        Returns
        -------
        TickManager
            The tick manager containing tick collections for each column.
        """
        return self._tick_manager

    @property
    def axis_labels(self) -> List[AxisLabel]:
        """
        Get the list of axis labels.

        Returns
        -------
        List[AxisLabel]
            List of axis labels, one for each column. Initially all labels are None.
        """
        return self._axis_labels

    @property
    def custom_limits(self) -> List[CustomAxisLimit]:
        """
        Get the list of custom axis limits.

        Returns
        -------
        List[CustomAxisLimit]
            List of custom axis limits, one for each column. Initially all limits are None.
        """
        return self._custom_limits

    # Data Management Methods
    def append_values(self, row: Sequence[Union[str, int, float]]) -> None:
        """
        Append a new row of values to the array manager and update all structures.

        This method adds a new row to the underlying array manager and automatically
        updates the tick manager, axis labels, and custom limits to maintain
        consistency. The tick manager will regenerate ticks based on the
        updated values.

        Parameters
        ----------
        row : Sequence[Union[str, int, float]]
            The row of values to append. Must have the same length as existing columns.

        Raises
        ------
        ValueError
            If the row length doesn't match the number of columns.
        """
        self._array_manager.append_values(row)
        self._initialize_structures()

    def remove_values(self, indices: Sequence[int]) -> None:
        """
        Remove rows at the specified indices and update all structures.

        This method removes the specified rows from the underlying array manager
        and automatically updates the tick manager, axis labels, and custom
        limits to maintain consistency. The tick manager will regenerate
        ticks based on the updated values.

        Parameters
        ----------
        indices : Sequence[int]
            The indices of rows to remove.

        Raises
        ------
        IndexError
            If any index is out of bounds.
        ValueError
            If indices are not valid integers.
        """
        self._array_manager.remove_values(indices)
        self._initialize_structures()

    def set_values(
        self, values: Sequence[Sequence[Union[str, int, float]]]
    ) -> None:
        """
        Set new values for the plot model, replacing all existing values and updating all structures.

        This method replaces all existing values in the underlying array manager
        and automatically updates the tick manager, axis labels, and custom
        limits to maintain consistency. The tick manager will regenerate
        ticks based on the new values, and all associated structures will
        be reset to their initial state.

        Parameters
        ----------
        values : Sequence[Sequence[Union[str, int, float]]]
            The new values as a 2D sequence where each row is a sequence
            of values and each column should be consistently typed.

        Raises
        ------
        ValueError
            If the values structure is invalid (empty or inconsistent row lengths).
        """
        self._array_manager.set_values(values)
        self._initialize_structures()

    # Axis Label Methods
    def get_axis_label(self, index: int) -> Optional[str]:
        """
        Get the axis label for a specific column.

        Parameters
        ----------
        index : int
            The index of the column to get the label for.

        Returns
        -------
        Optional[str]
            The axis label text, or None if no label is set.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._axis_labels):
            raise IndexError(
                f"Index {index} out of bounds for model with {len(self._axis_labels)} columns"
            )

        return self._axis_labels[index].label

    def set_axis_label(self, index: int, label: str) -> None:
        """
        Set the axis label for a specific column.

        Parameters
        ----------
        index : int
            The index of the column to set the label for.
        label : str
            The label text to set.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        ValueError
            If the label is empty or invalid.
        """
        if index < 0 or index >= len(self._axis_labels):
            raise IndexError(
                f"Index {index} out of bounds for model with {len(self._axis_labels)} columns"
            )

        if not label or not label.strip():
            raise ValueError("Label cannot be empty or whitespace only")

        self._axis_labels[index].label = label.strip()

    def clear_axis_label(self, index: int) -> None:
        """
        Clear the axis label for a specific column.

        Parameters
        ----------
        index : int
            The index of the column to clear the label for.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._axis_labels):
            raise IndexError(
                f"Index {index} out of bounds for model with {len(self._axis_labels)} columns"
            )

        self._axis_labels[index].label = None

    # Custom Limit Methods
    def get_custom_limit(
        self, index: int
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Get the custom axis limits for a specific column.

        Parameters
        ----------
        index : int
            The index of the column to get limits for.

        Returns
        -------
        tuple[Optional[float], Optional[float]]
            A tuple of (min_value, max_value), where either can be None if not set.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._custom_limits):
            raise IndexError(
                f"Index {index} out of bounds for model with {len(self._custom_limits)} columns"
            )

        limit = self._custom_limits[index]
        return limit.min_val, limit.max_val

    def set_custom_limit(
        self,
        index: int,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> None:
        """
        Set custom axis limits for a specific column.

        Parameters
        ----------
        index : int
            The index of the column to set limits for.
        min_val : float, optional
            Custom minimum value for the axis.
        max_val : float, optional
            Custom maximum value for the axis.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If min_val or max_val are not numeric or None.
        ValueError
            If both min_val and max_val are set and min_val >= max_val.
        """
        if index < 0 or index >= len(self._custom_limits):
            raise IndexError(
                f"Index {index} out of bounds for model with {len(self._custom_limits)} columns"
            )

        self._custom_limits[index] = CustomAxisLimit(
            min_val=min_val, max_val=max_val
        )

    def clear_custom_limit(self, index: int) -> None:
        """
        Clear custom axis limits for a specific column.

        Parameters
        ----------
        index : int
            The index of the column to clear limits for.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._custom_limits):
            raise IndexError(
                f"Index {index} out of bounds for model with {len(self._custom_limits)} columns"
            )

        self._custom_limits[index] = CustomAxisLimit()

    # Special Methods
    def __len__(self) -> int:
        """
        Get the number of columns in the model.

        Returns
        -------
        int
            The number of columns.
        """
        return self._array_manager.num_arrays


    def __repr__(self) -> str:
        """
        Get a string representation of the plot model.

        Returns
        -------
        str
            A string representation showing the model dimensions and structure counts.
        """
        if self._array_manager.num_arrays == 0:
            row_count = 0
        else:
            row_count = len(self._array_manager.get_array(0))
        return (
            f"PlotModel({row_count} rows, {self._array_manager.num_arrays} columns, "
            f"{len(self._axis_labels)} axis labels, {len(self._custom_limits)} custom limits)"
        )

    # Private Methods
    def _initialize_structures(self) -> None:
        """
        Update all associated structures when values change.

        This method is called whenever the array manager values are modified to ensure
        all associated structures (TickManager, AxisLabels, CustomAxisLimits)
        are kept in sync with the current values.
        """
        # Determine tick types from array manager array types
        tick_types = []
        for i in range(self._array_manager.num_arrays):
            array_type = self._array_manager.get_array_type(i)
            if array_type == ArrayType.NUMERIC:
                tick_types.append(TickType.NUMERIC)
            else:  # CATEGORICAL
                tick_types.append(TickType.CATEGORICAL)

        # Create new tick manager with appropriate types
        self._tick_manager = TickManager()
        self._tick_manager.set_ticks_from_types(tick_types)

        # Generate ticks for each column based on values
        self._generate_ticks_from_values()

        # Update axis labels list to match column count
        old_axis_labels = self._axis_labels.copy()
        self._axis_labels = []
        for i in range(self._array_manager.num_arrays):
            if i < len(old_axis_labels):
                # Preserve existing label
                self._axis_labels.append(old_axis_labels[i])
            else:
                # Create new label for new column
                self._axis_labels.append(AxisLabel())

        # Update custom limits list to match column count
        old_custom_limits = self._custom_limits.copy()
        self._custom_limits = []
        for i in range(self._array_manager.num_arrays):
            if i < len(old_custom_limits):
                # Preserve existing limit
                self._custom_limits.append(old_custom_limits[i])
            else:
                # Create new limit for new column
                self._custom_limits.append(CustomAxisLimit())

    def _generate_ticks_from_values(self) -> None:
        """
        Generate ticks for each column based on the current values.

        This method populates the tick collections with appropriate tick data
        based on the array types and values.
        """
        for i in range(self._array_manager.num_arrays):
            array_type = self._array_manager.get_array_type(i)

            if array_type == ArrayType.NUMERIC:
                # Generate numeric ticks
                numeric_array = self._array_manager.get_numeric_array(i)
                if len(numeric_array) > 0:
                    # Get non-NaN values for range calculation
                    non_nan_values = numeric_array.non_nan_values
                    if non_nan_values:
                        min_val = min(non_nan_values)
                        max_val = max(non_nan_values)

                        # Only generate ticks if we have a valid range
                        if min_val != max_val:
                            numeric_ticks = (
                                self._tick_manager.get_numeric_ticks(i)
                            )
                            numeric_ticks.set_ticks_from_range(
                                min_val, max_val
                            )
                        else:
                            # Single value case - create a simple tick
                            numeric_ticks = (
                                self._tick_manager.get_numeric_ticks(i)
                            )
                            numeric_ticks.set_ticks(
                                [f"{min_val:.2f}"], [min_val]
                            )

            elif array_type == ArrayType.CATEGORICAL:
                # Generate categorical ticks
                categorical_array = self._array_manager.get_categorical_array(i)
                if len(categorical_array) > 0:
                    # Get unique categories (excluding NaN)
                    unique_categories = [
                        cat
                        for cat in categorical_array.unique_values
                        if cat != "<NaN>"
                    ]
                    if unique_categories:
                        categorical_ticks = (
                            self._tick_manager.get_categorical_ticks(i)
                        )
                        categorical_ticks.set_ticks_from_categories(
                            unique_categories
                        )
