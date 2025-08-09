from typing import Sequence, Union, Tuple
from paxplot.data_managers.normalized_matrix import NormalizedMatrix, ColumnType

class PlotModel:
    """
    A plot model that manages data via a NormalizedMatrix.

    Focuses on data operations: adding, appending, and removing rows,
    as well as accessing normalized and raw column data, and setting custom bounds.
    """

    def __init__(self, initial_data: Sequence[Sequence[Union[str, int, float]]]):
        """
        Initialize PlotModel with initial data.

        Parameters
        ----------
        initial_data : Sequence of rows with mixed-type values
            Initial data to populate the normalized matrix.
        """
        self._matrix = NormalizedMatrix(data=initial_data)

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
