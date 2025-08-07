from typing import Sequence, Union, List
from numpy.typing import NDArray
import numpy as np

from paxplot.data_managers.normalized_matrix import NormalizedMatrix, ColumnType

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
        if not isinstance(column_names, Sequence) or not all(isinstance(n, str) for n in column_names):
            raise TypeError("column_names must be a sequence of strings.")
        if matrix.num_columns != len(column_names):
            raise ValueError("Number of names must match matrix columns.")
        if len(set(column_names)) != len(column_names):
            raise ValueError("Column names must be unique.")

        self._matrix = matrix
        self._name_to_index = {name: idx for idx, name in enumerate(column_names)}
        self._index_to_name = list(column_names)

    @property
    def column_names(self) -> List[str]:
        return self._index_to_name.copy()

    @property
    def matrix(self) -> NormalizedMatrix:
        return self._matrix

    def _get_index(self, column_name: str) -> int:
        if column_name not in self._name_to_index:
            raise KeyError(f"Column '{column_name}' not found.")
        return self._name_to_index[column_name]

    def get_normalized_array(self, column_name: str) -> NDArray[np.float64]:
        return self._matrix.get_normalized_array(self._get_index(column_name))

    def get_column_type(self, column_name: str) -> ColumnType:
        return self._matrix.get_column_type(self._get_index(column_name))

    def get_numeric_array(self, column_name: str) -> Sequence[Union[int, float]]:
        return self._matrix.get_numeric_array(self._get_index(column_name))

    def get_categorical_array(self, column_name: str) -> Sequence[str]:
        return self._matrix.get_categorical_array(self._get_index(column_name))

    def set_custom_bounds(self, column_name: str, min_val: float | None = None, max_val: float | None = None) -> None:
        self._matrix.set_custom_bounds(self._get_index(column_name), min_val, max_val)

    def get_custom_bounds(self, column_name: str) -> tuple[float | None, float | None]:
        return self._matrix.get_custom_bounds(self._get_index(column_name))

    def append_data(self, new_rows: Sequence[Sequence[Union[str, int, float]]]) -> None:
        if not new_rows:
            return

        if any(len(row) != self._matrix.num_columns for row in new_rows):
            raise ValueError(f"All rows must have exactly {self._matrix.num_columns} columns.")

        self._matrix.append_data(new_rows)

    def remove_rows(self, indices: Sequence[int]) -> None:
        self._matrix.remove_rows(indices)
