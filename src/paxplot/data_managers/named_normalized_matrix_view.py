"""
NamedNormalizedMatrixView module.

Provides a lightweight named view over an existing NormalizedMatrix instance,
allowing users to access matrix columns by meaningful string names instead of
integer indices.

This view maintains a one-to-one mapping between unique column names and
columns of the underlying matrix, delegating most operations back to the
NormalizedMatrix. It enables intuitive name-based interaction while preserving
direct access to the original matrix data.

Typical usage includes:
- Initializing with a NormalizedMatrix and corresponding unique column names.
- Retrieving normalized or original column data by name.
- Appending new rows or removing existing rows by delegating to the matrix.
- Setting and getting normalization bounds on numeric columns by name.

Raises
------
ValueError
    If the number of column names does not match the number of matrix columns
    or if column names are not unique.
"""

from typing import Sequence, Union, List
from numpy.typing import NDArray
import numpy as np

from paxplot.data_managers.normalized_matrix import (
    NormalizedMatrix,
    ColumnType,
)


class NamedNormalizedMatrixView:
    """
    A lightweight named view over an existing NormalizedMatrix.

    Parameters
    ----------
    matrix : NormalizedMatrix
        The underlying matrix instance.
    column_names : Sequence[str]
        Unique names corresponding to columns in the matrix.

    Raises
    ------
    ValueError
        If number of names doesn't match number of columns or if names aren't unique.
    """

    def __init__(self, matrix: NormalizedMatrix, column_names: Sequence[str]):
        if not isinstance(column_names, Sequence) or not all(
            isinstance(n, str) for n in column_names
        ):
            raise TypeError("column_names must be a sequence of strings.")
        if matrix.num_columns != len(column_names):
            raise ValueError("Number of names must match matrix columns.")
        if len(set(column_names)) != len(column_names):
            raise ValueError("Column names must be unique.")

        self._matrix = matrix
        self._name_to_index = {
            name: idx for idx, name in enumerate(column_names)
        }
        self._index_to_name = list(column_names)

    @property
    def column_names(self) -> List[str]:
        """
        Return the list of column names in order.

        Returns
        -------
        List[str]
            A copy of the list of column names corresponding to matrix columns.
        """
        return self._index_to_name.copy()


    def set_column_names(self, new_names: Sequence[str]) -> None:
        """
        Set all column names at once.

        Parameters
        ----------
        new_names : Sequence[str]
            New list of column names, must match matrix column count and be unique.

        Raises
        ------
        ValueError
            If length mismatch or duplicate names found.
        TypeError
            If any name is not a string.
        """
        if not isinstance(new_names, Sequence) or not all(isinstance(n, str) for n in new_names):
            raise TypeError("new_names must be a sequence of strings.")
        if len(new_names) != self._matrix.num_columns:
            raise ValueError(f"Number of new names ({len(new_names)}) must match matrix columns ({self._matrix.num_columns}).")
        if len(set(new_names)) != len(new_names):
            raise ValueError("All column names must be unique.")

        self._index_to_name = list(new_names)
        self._name_to_index = {name: idx for idx, name in enumerate(self._index_to_name)}

    @property
    def matrix(self) -> NormalizedMatrix:
        """
        Return the underlying NormalizedMatrix instance.

        Returns
        -------
        NormalizedMatrix
            The wrapped normalized matrix.
        """
        return self._matrix

    def _get_index(self, column_name: str) -> int:
        """
        Map a column name to its integer index.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        int
            The integer index of the column.

        Raises
        ------
        KeyError
            If the column name does not exist.
        """
        if column_name not in self._name_to_index:
            raise KeyError(f"Column '{column_name}' not found.")
        return self._name_to_index[column_name]

    def get_column_name(self, column_index: int) -> str:
        """
        Get the name of a column by index.

        Parameters
        ----------
        column_index : int

        Returns
        -------
        str
            The column name.

        Raises
        ------
        IndexError
            If the column_index is out of range.
        """
        if column_index < 0 or column_index >= len(self._index_to_name):
            raise IndexError("Column index out of range.")
        return self._index_to_name[column_index]


    def set_column_name(self, column_index: int, new_name: str) -> None:
        """
        Set the name of a column by index.

        Parameters
        ----------
        column_index : int
        new_name : str
            The new column name. Must be unique.

        Raises
        ------
        IndexError
            If the column_index is out of range.
        ValueError
            If new_name is not a string or not unique.
        """
        if not isinstance(new_name, str):
            raise ValueError("Column name must be a string.")
        if column_index < 0 or column_index >= len(self._index_to_name):
            raise IndexError("Column index out of range.")
        if new_name in self._name_to_index and self._name_to_index[new_name] != column_index:
            raise ValueError(f"Column name '{new_name}' already exists.")

        old_name = self._index_to_name[column_index]
        # Update mappings
        self._index_to_name[column_index] = new_name
        del self._name_to_index[old_name]
        self._name_to_index[new_name] = column_index

    def get_normalized_array(self, column_name: str) -> NDArray[np.float64]:
        """
        Get the normalized data array for the given column name.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        NDArray[np.float64]
            The normalized numeric array for the specified column.
        """
        return self._matrix.get_normalized_array(self._get_index(column_name))

    def get_column_type(self, column_name: str) -> ColumnType:
        """
        Get the type of the specified column.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        ColumnType
            The type of the column (numeric or categorical).
        """
        return self._matrix.get_column_type(self._get_index(column_name))

    def get_numeric_array(
        self, column_name: str
    ) -> Sequence[Union[int, float]]:
        """
        Get the original numeric data array for the specified column.

        Parameters
        ----------
        column_name : str
            The name of the numeric column.

        Returns
        -------
        Sequence[Union[int, float]]
            The original numeric values in the column.

        Raises
        ------
        TypeError
            If the column is not numeric.
        """
        return self._matrix.get_numeric_array(self._get_index(column_name))

    def get_categorical_array(self, column_name: str) -> Sequence[str]:
        """
        Get the original categorical data array for the specified column.

        Parameters
        ----------
        column_name : str
            The name of the categorical column.

        Returns
        -------
        Sequence[str]
            The original categorical values in the column.

        Raises
        ------
        TypeError
            If the column is not categorical.
        """
        return self._matrix.get_categorical_array(self._get_index(column_name))

    def set_custom_bounds(
        self,
        column_name: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        """
        Set custom minimum and/or maximum normalization bounds for a numeric column.

        Parameters
        ----------
        column_name : str
            The name of the numeric column.
        min_val : float or None, optional
            The custom minimum bound. If None, the bound is unchanged. (default is None)
        max_val : float or None, optional
            The custom maximum bound. If None, the bound is unchanged. (default is None)

        Raises
        ------
        TypeError
            If the specified column is not numeric.
        """
        self._matrix.set_custom_bounds(
            self._get_index(column_name), min_val, max_val
        )

    def get_custom_bounds(
        self, column_name: str
    ) -> tuple[float | None, float | None]:
        """
        Retrieve the custom minimum and maximum normalization bounds for a numeric column.

        Parameters
        ----------
        column_name : str
            The name of the numeric column.

        Returns
        -------
        tuple of (float or None, float or None)
            The custom (min_val, max_val) bounds.

        Raises
        ------
        TypeError
            If the specified column is not numeric.
        """
        return self._matrix.get_custom_bounds(self._get_index(column_name))

    def append_data(
        self, new_rows: Sequence[Sequence[Union[str, int, float]]]
    ) -> None:
        """
        Append new rows to the matrix.

        Parameters
        ----------
        new_rows : Sequence of Sequences of (str, int, float)
            Rows to append, each row must have exactly as many columns as the matrix.

        Raises
        ------
        ValueError
            If any row does not have the exact number of columns.
        """
        if not new_rows:
            return

        if any(len(row) != self._matrix.num_columns for row in new_rows):
            raise ValueError(
                f"All rows must have exactly {self._matrix.num_columns} columns."
            )

        self._matrix.append_data(new_rows)

    def remove_rows(self, indices: Sequence[int]) -> None:
        """
        Remove rows at the specified indices from the matrix.

        Parameters
        ----------
        indices : Sequence[int]
            Indices of rows to remove.
        """
        self._matrix.remove_rows(indices)
