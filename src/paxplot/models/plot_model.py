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
    PaxPlot data. It manages a Matrix for data storage and automatically
    creates and maintains associated structures including TickManager,
    AxisLabels, and CustomAxisLimits.

    Parameters
    ----------
    data : Sequence[Sequence[Union[str, int, float]]]
        Initial data as a 2D sequence where each row is a sequence of values
        and each column should be consistently typed.

    Attributes
    ----------
    matrix : Matrix
        The underlying data matrix managing all columns of data.
    tick_manager : TickManager
        Manager for tick collections corresponding to each column.
    axis_labels : List[AxisLabel]
        List of axis labels, one for each column (initially with default values).
    custom_limits : List[CustomAxisLimit]
        List of custom axis limits, one for each column (initially with default values).

    Examples
    --------
    >>> # Initialize with data
    >>> data = [
    ...     [1, 'A', 2.5],
    ...     [2, 'B', 3.0],
    ...     [3, 'A', 1.5]
    ... ]
    >>> plot_model = PlotModel(data)
    >>> print(plot_model.matrix.num_columns)  # 3
    >>> print(plot_model.matrix.num_rows)     # 3
    >>>
    >>> # Append new data
    >>> plot_model.append_data([4, 'C', 2.0])
    >>> print(plot_model.matrix.num_rows)     # 4
    >>>
    >>> # Remove data by indices
    >>> plot_model.remove_data([0, 2])
    >>> print(plot_model.matrix.num_rows)     # 2
    >>>
    >>> # Access structures
    >>> matrix = plot_model.matrix
    >>> numeric_ticks = plot_model.tick_manager.get_numeric_ticks(0)
    >>> axis_label = plot_model.axis_labels[0]
    >>> custom_limit = plot_model.custom_limits[0]
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

    @property
    def matrix(self) -> Matrix:
        """
        Get the underlying data matrix.

        Returns
        -------
        Matrix
            The matrix containing all data columns.
        """
        return self._matrix

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
            List of axis labels, one for each column. Initially all labels have default values.
        """
        return self._axis_labels.copy()

    @property
    def custom_limits(self) -> List[CustomAxisLimit]:
        """
        Get the list of custom axis limits.

        Returns
        -------
        List[CustomAxisLimit]
            List of custom axis limits, one for each column. Initially all limits have default values.
        """
        return self._custom_limits.copy()

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
