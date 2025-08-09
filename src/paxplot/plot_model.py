"""
Module providing the PlotModel class for managing and interacting with
normalized tabular data.

PlotModel wraps a NormalizedMatrix to enable operations such as adding,
removing, and accessing rows of mixed-type data with numeric and categorical
columns. It supports normalization, raw data access, and custom bounds setting.

Additionally, PlotModel can maintain column names through a NamedNormalizedMatrixView,
allowing intuitive access to columns by name as well as by index.

Typical usage:
- Initialize with initial data.
- Append or remove rows dynamically.
- Access normalized or raw data columns by index or name.
- Set and get custom normalization bounds for numeric columns.
- Define and manage unique column names for improved readability.
"""

from typing import Sequence, Union, Tuple, List, Optional
from paxplot.data_managers.normalized_matrix import NormalizedMatrix, ColumnType
from paxplot.data_managers.named_normalized_matrix_view import NamedNormalizedMatrixView

class PlotModel:
    """
    A plot model that manages data via a NormalizedMatrix.

    Focuses on data operations: adding, appending, and removing rows,
    as well as accessing normalized and raw column data, and setting custom bounds.
    """

    def __init__(self, data: Sequence[Sequence[Union[str, int, float]]]):
        """
        Initialize PlotModel with initial data.

        Parameters
        ----------
        data : Sequence of rows with mixed-type values
            Initial data to populate the normalized matrix.
        """
        self._matrix = NormalizedMatrix(data=data)
        self._named_view: Optional[NamedNormalizedMatrixView] = None

    @property
    def num_rows(self) -> int:
        """Number of rows in the data."""
        return self._matrix.num_rows

    @property
    def num_columns(self) -> int:
        """Number of columns in the data."""
        return self._matrix.num_columns

    def append_rows(self, new_rows: Sequence[Sequence[Union[str, int, float]]]) -> None:
        """
        Append new rows to the existing data.

        Parameters
        ----------
        new_rows : Sequence of rows with mixed-type values
        """
        self._matrix.append_data(new_rows)

    def remove_rows(self, indices: Sequence[int]) -> None:
        """
        Remove rows by indices.

        Parameters
        ----------
        indices : Sequence of row indices to remove
        """
        self._matrix.remove_rows(indices)

    def get_normalized_column(self, column_index: int) -> Sequence[float]:
        """
        Get the normalized data for the specified column.

        Parameters
        ----------
        column_index : int

        Returns
        -------
        Sequence[float]
            The normalized values for the column.
        """
        arr = self._matrix.get_normalized_array(column_index)
        return arr.tolist()

    def get_raw_column(self, column_index: int) -> Sequence[Union[str, int, float]]:
        """
        Get the original raw (unnormalized) data for the specified column.

        Parameters
        ----------
        column_index : int

        Returns
        -------
        Sequence[Union[str, int, float]]
            The raw data values for the column.
        """
        col_type = self._matrix.get_column_type(column_index)
        if col_type == ColumnType.NUMERIC:
            return self._matrix.get_numeric_array(column_index)
        elif col_type == ColumnType.CATEGORICAL:
            return self._matrix.get_categorical_array(column_index)
        else:
            raise ValueError(f"Unknown column type for index {column_index}")

    def set_custom_bounds(
        self,
        column_index: int,
        min_val: float | None = None,
        max_val: float | None = None
    ) -> None:
        """
        Set custom min and/or max bounds for the numeric column.

        Parameters
        ----------
        column_index : int
        min_val : float | None
            Custom minimum bound for normalization, or None to leave unchanged.
        max_val : float | None
            Custom maximum bound for normalization, or None to leave unchanged.

        Raises
        ------
        TypeError
            If the specified column is not numeric.
        """
        self._matrix.set_custom_bounds(column_index, min_val=min_val, max_val=max_val)

    def get_custom_bounds(self, column_index: int) -> Tuple[float | None, float | None]:
        """
        Get custom min and max bounds for the numeric column.

        Parameters
        ----------
        column_index : int

        Returns
        -------
        Tuple[float | None, float | None]
            (min_val, max_val) custom bounds, or (None, None) if none set.

        Raises
        ------
        TypeError
            If the specified column is not numeric.
        """
        return self._matrix.get_custom_bounds(column_index)

    def set_column_names(self, column_names: Sequence[str]) -> None:
        """
        Set all column names at once.

        Parameters
        ----------
        column_names : Sequence[str]
            New list of column names, must match matrix column count and be unique.

        Raises
        ------
        ValueError
            If length mismatch or duplicate names found.
        TypeError
            If any name is not a string.
        """
        if self._named_view is None:
            # create NamedNormalizedMatrixView with names
            self._named_view = NamedNormalizedMatrixView(self._matrix, column_names)
        else:
            self._named_view.set_column_names(column_names)

    def get_column_names(self) -> List[str]:
        """
        Get the list of column names.

        Returns
        -------
        List[str]
            List of column names.

        Raises
        ------
        RuntimeError
            If column names have not been set.
        """
        if self._named_view is None:
            raise RuntimeError("Column names have not been set.")
        return self._named_view.column_names

    def get_column_name(self, column_index: int) -> str:
        """
        Get the name of a column by index.

        Raises
        ------
        RuntimeError
            If column names have not been set.
        """
        if self._named_view is None:
            raise RuntimeError("Column names have not been set.")
        return self._named_view.get_column_name(column_index)

    def set_column_name(self, column_index: int, new_name: str) -> None:
        """
        Set the name of a column by index.

        Raises
        ------
        RuntimeError
            If column names have not been set.
        """
        if self._named_view is None:
            raise RuntimeError("Column names have not been set.")
        self._named_view.set_column_name(column_index, new_name)

    def get_normalized_column_by_name(self, column_name: str) -> Sequence[float]:
        """
        Get the normalized data for the specified column name.

        Raises
        ------
        RuntimeError
            If column names have not been set.
        """
        if self._named_view is None:
            raise RuntimeError("Column names have not been set.")
        arr = self._named_view.get_normalized_array(column_name)
        return arr.tolist()

    def get_raw_column_by_name(self, column_name: str) -> Sequence[Union[str, int, float]]:
        """
        Get the original raw (unnormalized) data for the specified column name.

        Raises
        ------
        RuntimeError
            If column names have not been set.
        """
        if self._named_view is None:
            raise RuntimeError("Column names have not been set.")

        col_type = self._named_view.get_column_type(column_name)
        if col_type == ColumnType.NUMERIC:
            return self._named_view.get_numeric_array(column_name)
        elif col_type == ColumnType.CATEGORICAL:
            return self._named_view.get_categorical_array(column_name)
        else:
            raise ValueError(f"Unknown column type for name '{column_name}'")

    def set_custom_bounds_by_name(
        self,
        column_name: str,
        min_val: float | None = None,
        max_val: float | None = None
    ) -> None:
        """
        Set custom min and/or max bounds for the numeric column name.

        Raises
        ------
        RuntimeError
            If column names have not been set.
        """
        if self._named_view is None:
            raise RuntimeError("Column names have not been set.")
        self._named_view.set_custom_bounds(column_name, min_val=min_val, max_val=max_val)

    def get_custom_bounds_by_name(self, column_name: str) -> Tuple[float | None, float | None]:
        """
        Get custom min and max bounds for the numeric column name.

        Raises
        ------
        RuntimeError
            If column names have not been set.
        """
        if self._named_view is None:
            raise RuntimeError("Column names have not been set.")
        return self._named_view.get_custom_bounds(column_name)
