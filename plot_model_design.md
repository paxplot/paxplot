# PlotModel Design Document

## Overview

The `PlotModel` class serves as the main interface for creating and managing plot structures in PaxPlot. It coordinates data management through a Matrix and automatically creates associated structures including TickManager, AxisLabels, and CustomAxisLimits.

## Core Concept

Every time data is modified (initialization, append, or remove operations), the PlotModel automatically updates all associated structures to maintain consistency across the plot components.

## Class Design

```python
class PlotModel:
    """
    Main interface for creating and managing plot structures.

    The PlotModel serves as the primary interface for users to interact with
    PaxPlot data. It manages a Matrix for data storage and automatically
    creates and maintains associated structures including TickManager,
    AxisLabels, and CustomAxisLimits.

    Parameters
    ----------
    data : Sequence[Sequence[Union[str, int, float]]], optional
        Initial data as a 2D sequence where each row is a sequence of values
        and each column should be consistently typed. If None, creates an
        empty model that can be populated later.

    Attributes
    ----------
    matrix : Matrix
        The underlying data matrix managing all columns of data.
    tick_manager : TickManager
        Manager for tick collections corresponding to each column.
    axis_labels : List[AxisLabel]
        List of axis labels, one for each column (initially None).
    custom_limits : List[CustomAxisLimit]
        List of custom axis limits, one for each column (initially None).

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
    >>> plot_model.append([4, 'C', 2.0])
    >>> print(plot_model.matrix.num_rows)     # 4
    >>>
    >>> # Remove data by indices
    >>> plot_model.remove([0, 2])
    >>> print(plot_model.matrix.num_rows)     # 2
    >>>
    >>> # Access structures
    >>> matrix = plot_model.matrix
    >>> numeric_ticks = plot_model.tick_manager.get_numeric_ticks(0)
    >>> axis_label = plot_model.axis_labels[0]
    >>> custom_limit = plot_model.custom_limits[0]
    """

    def __init__(
        self, data: Optional[Sequence[Sequence[Union[str, int, float]]]] = None
    ) -> None:
        """
        Initialize PlotModel with optional initial data.

        Parameters
        ----------
        data : Sequence[Sequence[Union[str, int, float]]], optional
            Initial data as a 2D sequence. If None, creates an empty model.

        Raises
        ------
        ValueError
            If the data structure is invalid (empty or inconsistent row lengths).
        """

    @property
    def matrix(self) -> Matrix:
        """
        Get the underlying data matrix.

        Returns
        -------
        Matrix
            The matrix containing all data columns.
        """

    @property
    def tick_manager(self) -> TickManager:
        """
        Get the tick manager for all columns.

        Returns
        -------
        TickManager
            The tick manager containing tick collections for each column.
        """

    @property
    def axis_labels(self) -> List[AxisLabel]:
        """
        Get the list of axis labels.

        Returns
        -------
        List[AxisLabel]
            List of axis labels, one for each column. Initially all labels are None.
        """

    @property
    def custom_limits(self) -> List[CustomAxisLimit]:
        """
        Get the list of custom axis limits.

        Returns
        -------
        List[CustomAxisLimit]
            List of custom axis limits, one for each column. Initially all limits are None.
        """

    def append(self, row: Sequence[Union[str, int, float]]) -> None:
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

    def remove(self, indices: Sequence[int]) -> None:
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

    def get_column_count(self) -> int:
        """
        Get the number of columns in the model.

        Returns
        -------
        int
            The number of columns.
        """

    def get_row_count(self) -> int:
        """
        Get the number of rows in the model.

        Returns
        -------
        int
            The number of rows.
        """

    def is_empty(self) -> bool:
        """
        Check if the model contains any data.

        Returns
        -------
        bool
            True if the model has no columns or no rows, False otherwise.
        """

    def __len__(self) -> int:
        """
        Get the number of columns in the model.

        Returns
        -------
        int
            The number of columns.
        """

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

    def __repr__(self) -> str:
        """
        Get a string representation of the plot model.

        Returns
        -------
        str
            A string representation showing the model dimensions and structure counts.
        """
```

## Key Design Principles

1. **Automatic Structure Management**: Every data modification triggers updates to all associated structures
2. **Type Safety**: Full type hints and proper error handling
3. **Consistency**: Maintains consistency between Matrix, TickManager, AxisLabels, and CustomAxisLimits
4. **User-Friendly Interface**: Simple methods like `append()` and `remove()` for data manipulation
5. **Extensibility**: Designed to be easily extended with new features

## Usage Pattern

```python
# Create model with data
plot_model = PlotModel(data)

# Modify data
plot_model.append(more_data)
plot_model.remove(indices)

# Access structures (automatically maintained)
matrix = plot_model.matrix  # Direct access to underlying Matrix
ticks = plot_model.tick_manager
labels = plot_model.axis_labels
limits = plot_model.custom_limits
```

## Implementation Notes

- The underlying data management happens through the Matrix class
- Every time data is modified, the PlotModel calls internal methods to update:
  - TickManager (regenerates ticks based on current data)
  - AxisLabels (maintains list structure)
  - CustomAxisLimits (maintains list structure)
- All structures are initially created as None/empty and populated as needed
