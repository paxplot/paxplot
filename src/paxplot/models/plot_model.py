"""Plot model for PaxPlot.

This module defines the PlotModel class that serves as the main interface
for creating and managing plot structures in PaxPlot.
"""

from typing import List, Optional, Sequence, Union

from ..structures.matrix import Matrix, ColumnType
from ..structures.tick_manger import TickManager, TickType
from ..structures.labels.axis_label import AxisLabel
from ..structures.limits.custom_axis_limit import CustomAxisLimit


class PlotModel:
    """
    Main interface for creating and managing plot structures.

    The PlotModel serves as the primary interface for users to interact with
    PaxPlot data. It provides a clean API for managing data, axis labels,
    and custom limits without exposing internal implementation details.

    Parameters
    ----------
    data : Sequence[Sequence[Union[str, int, float]]]
        Initial data as a 2D sequence where each row is a sequence of values
        and each column should be consistently typed.

    Examples
    --------
    >>> # Initialize with data
    >>> data = [
    ...     [1, 'A', 2.5],
    ...     [2, 'B', 3.0],
    ...     [3, 'A', 1.5]
    ... ]
    >>> plot_model = PlotModel(data)
    >>> print(plot_model.get_column_count())  # 3
    >>> print(plot_model.get_row_count())     # 3
    >>>
    >>> # Append new data
    >>> plot_model.append_data([4, 'C', 2.0])
    >>> print(plot_model.get_row_count())     # 4
    >>>
    >>> # Remove data by indices
    >>> plot_model.remove_data([0, 2])
    >>> print(plot_model.get_row_count())     # 2
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
    >>>
    >>> # Access data values
    >>> column_type = plot_model.get_column_type(0)  # "numeric"
    >>> numeric_values = plot_model.get_numeric_values(0)  # [1.0, 2.0, 3.0]
    >>> categorical_values = plot_model.get_categorical_values(1)  # ['A', 'B', 'A']
    >>> unique_vals = plot_model.get_unique_values(1)  # ['A', 'B']
    """

    def __init__(
        self, data: Sequence[Sequence[Union[str, int, float]]]
    ) -> None:
        """
        Initialize PlotModel with initial data.

        Parameters
        ----------
        data : Sequence[Sequence[Union[str, int, float]]]
            Initial data as a 2D sequence where each row is a sequence
            of values and each column should be consistently typed.

        Raises
        ------
        ValueError
            If the data structure is invalid (empty or inconsistent row lengths).
        """
        # Initialize matrix with data directly
        self._matrix = Matrix(data)
        
        # Initialize other structures based on the matrix
        self._initialize_structures()

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

    def get_custom_limit(self, index: int) -> tuple[Optional[float], Optional[float]]:
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

    def get_column_type(self, index: int) -> str:
        """
        Get the type of a column.

        Parameters
        ----------
        index : int
            The index of the column to get the type for.

        Returns
        -------
        str
            The column type: "numeric" or "categorical".

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= self._matrix.num_columns:
            raise IndexError(
                f"Index {index} out of bounds for model with {self._matrix.num_columns} columns"
            )
        
        return self._matrix.get_column_type(index).value

    def get_numeric_values(self, index: int) -> List[float]:
        """
        Get numeric values from a column.

        Parameters
        ----------
        index : int
            The index of the column to get values from.

        Returns
        -------
        List[float]
            The numeric values from the column.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the column is not numeric.
        """
        if index < 0 or index >= self._matrix.num_columns:
            raise IndexError(
                f"Index {index} out of bounds for model with {self._matrix.num_columns} columns"
            )
        
        if self._matrix.get_column_type(index) != ColumnType.NUMERIC:
            raise TypeError(f"Column {index} is not numeric, it is {self._matrix.get_column_type(index).value}")
        
        return self._matrix.get_numeric_array(index).values

    def get_categorical_values(self, index: int) -> List[str]:
        """
        Get categorical values from a column.

        Parameters
        ----------
        index : int
            The index of the column to get values from.

        Returns
        -------
        List[str]
            The categorical values from the column.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the column is not categorical.
        """
        if index < 0 or index >= self._matrix.num_columns:
            raise IndexError(
                f"Index {index} out of bounds for model with {self._matrix.num_columns} columns"
            )
        
        if self._matrix.get_column_type(index) != ColumnType.CATEGORICAL:
            raise TypeError(f"Column {index} is not categorical, it is {self._matrix.get_column_type(index).value}")
        
        return self._matrix.get_categorical_array(index).values

    def get_unique_values(self, index: int) -> Optional[List[str]]:
        """
        Get the unique values for a categorical column.

        Parameters
        ----------
        index : int
            The index of the column to get unique values for.

        Returns
        -------
        Optional[List[str]]
            The unique values for categorical columns, None for numeric columns.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the column is not categorical.
        """
        if index < 0 or index >= self._matrix.num_columns:
            raise IndexError(
                f"Index {index} out of bounds for model with {self._matrix.num_columns} columns"
            )
        
        if self._matrix.get_column_type(index) != ColumnType.CATEGORICAL:
            raise TypeError(f"Column {index} is not categorical, it is {self._matrix.get_column_type(index).value}")
        
        categorical_array = self._matrix.get_categorical_array(index)
        return categorical_array.unique_values

    def append_data(self, row: Sequence[Union[str, int, float]]) -> None:
        """
        Append a new row of data to the matrix and update all structures.

        This method adds a new row to the underlying matrix and automatically
        updates the tick manager, axis labels, and custom limits to maintain
        consistency. The tick manager will regenerate ticks based on the
        updated data.

        Parameters
        ----------
        row : Sequence[Union[str, int, float]]
            The row of data to append. Must have the same length as existing columns.

        Raises
        ------
        ValueError
            If the row length doesn't match the number of columns.
        """
        self._matrix.append_data(row)
        self._initialize_structures()

    def remove_data(self, indices: Sequence[int]) -> None:
        """
        Remove rows at the specified indices and update all structures.

        This method removes the specified rows from the underlying matrix
        and automatically updates the tick manager, axis labels, and custom
        limits to maintain consistency. The tick manager will regenerate
        ticks based on the updated data.

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
        self._matrix.remove_data(indices)
        self._initialize_structures()

    def set_data(self, data: Sequence[Sequence[Union[str, int, float]]]) -> None:
        """
        Set new data for the plot model, replacing all existing data and updating all structures.

        This method replaces all existing data in the underlying matrix
        and automatically updates the tick manager, axis labels, and custom
        limits to maintain consistency. The tick manager will regenerate
        ticks based on the new data, and all associated structures will
        be reset to their initial state.

        Parameters
        ----------
        data : Sequence[Sequence[Union[str, int, float]]]
            The new data as a 2D sequence where each row is a sequence
            of values and each column should be consistently typed.

        Raises
        ------
        ValueError
            If the data structure is invalid (empty or inconsistent row lengths).
        """
        self._matrix.set_data(data)
        self._initialize_structures()

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

    def set_custom_limit(
        self, 
        index: int, 
        min_val: Optional[float] = None, 
        max_val: Optional[float] = None
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
        
        self._custom_limits[index] = CustomAxisLimit(min_val=min_val, max_val=max_val)

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

    def get_column_count(self) -> int:
        """
        Get the number of columns in the model.

        Returns
        -------
        int
            The number of columns.
        """
        return self._matrix.num_columns

    def get_row_count(self) -> int:
        """
        Get the number of rows in the model.

        Returns
        -------
        int
            The number of rows.
        """
        return self._matrix.num_rows

    def __len__(self) -> int:
        """
        Get the number of columns in the model.

        Returns
        -------
        int
            The number of columns.
        """
        return self._matrix.num_columns

    def __getitem__(self, index: int):
        """
        Get a column at the specified index.

        Parameters
        ----------
        index : int
            The index of the column to get.

        Returns
        -------
        Union[NumericalArray, CategoricalArray]
            The column at the specified index.
        """
        return self._matrix[index]

    def __repr__(self) -> str:
        """
        Get a string representation of the plot model.

        Returns
        -------
        str
            A string representation showing the model dimensions and structure counts.
        """
        return (
            f"PlotModel({self._matrix.num_rows} rows, {self._matrix.num_columns} columns, "
            f"{len(self._axis_labels)} axis labels, {len(self._custom_limits)} custom limits)"
        )

    def _initialize_structures(self) -> None:
        """
        Update all associated structures when data changes.
        
        This method is called whenever the matrix data is modified to ensure
        all associated structures (TickManager, AxisLabels, CustomAxisLimits)
        are kept in sync with the current data.
        """
        # Determine tick types from matrix column types
        tick_types = []
        for i in range(self._matrix.num_columns):
            column_type = self._matrix.get_column_type(i)
            if column_type == ColumnType.NUMERIC:
                tick_types.append(TickType.NUMERIC)
            else:  # CATEGORICAL
                tick_types.append(TickType.CATEGORICAL)
        
        # Create new tick manager with appropriate types
        self._tick_manager = TickManager(tick_types)
        
        # Update axis labels list to match column count
        self._axis_labels = [AxisLabel() for _ in range(self._matrix.num_columns)]
        
        # Update custom limits list to match column count
        self._custom_limits = [CustomAxisLimit() for _ in range(self._matrix.num_columns)]
