"""
A matrix class that handles normalization of mixed numeric and categorical data.

This class provides functionality to store and manage a 2D data structure where each column
can be either numeric or categorical. Each column is independently normalized according to
its type. Numeric columns are scaled to a [0,1] range, while categorical columns are
encoded as normalized indicators.

Notes
-----
- Each column must contain consistently typed data (either all numeric or all categorical).
- Null values (None) are not allowed in either numeric or categorical columns.
- Column access is zero-indexed.
"""

from typing import Sequence, Union, List
from enum import Enum
from pydantic import BaseModel, field_validator
import numpy as np
from numpy.typing import NDArray

from paxplot.data_managers.numeric_normalized_array import NumericNormalizedArray
from paxplot.data_managers.categorical_normalized_array import CategoricalNormalizedArray
from paxplot.data_managers.base_normalized_array import BaseNormalizedArray


class ColumnType(str, Enum):
    """
    Enumeration of possible column types in the NormalizedMatrix.

    Attributes
    ----------
    NUMERIC : str
        Indicates a column with numeric data (int or float), to be normalized using min-max scaling.
    CATEGORICAL : str
        Indicates a column with string (categorical) data, to be encoded as numeric indices.
    """
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class NormalizedMatrix(BaseModel):
    """
    A typed, column-oriented matrix that normalizes each column independently.

    Each column is either numeric or categorical and stored as a normalized array.
    Columns are accessed by integer index.

    Notes
    -----
    None values are not allowed in the matrix.
    """

    data: Sequence[Sequence[Union[str, int, float]]]
    _columns: List[BaseNormalizedArray] = []

    class Config:
        """
        Pydantic configuration for the NormalizedMatrix class.
        """
        arbitrary_types_allowed = True
        extra = "forbid"

    @field_validator("data", mode="before")
    @classmethod
    def validate_input(
        cls,
        v: Sequence[Sequence[Union[str, int, float, None]]]
    ) -> Sequence[Sequence[Union[str, int, float]]]:
        """
        Validate the input data structure.

        Parameters
        ----------
        v : Sequence[Sequence[Union[str, int, float, None]]]
            The input data to validate.

        Returns
        -------
        Sequence[Sequence[Union[str, int, float]]]
            The validated input data.

        Raises
        ------
        ValueError
            If the input is not a 2D array-like structure or contains None values.
        """
        arr = np.array(v, dtype=object)
        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array-like structure")
        if any(x is None for row in arr for x in row):  # pylint: disable=singleton-comparison
            raise ValueError("Input contains None values, which are not allowed")
        return v # type: ignore

    def __init__(self, **data):
        super().__init__(**data)
        arr = NormalizedMatrix.validate_column_shape_and_nulls(self.data)
        col_types = NormalizedMatrix.infer_column_types(arr)

        self._columns = []
        for i, col_type in enumerate(col_types):
            col_data = arr[:, i].tolist()
            if col_type == ColumnType.NUMERIC:
                self._columns.append(NumericNormalizedArray(array=col_data))
            elif col_type == ColumnType.CATEGORICAL:
                self._columns.append(CategoricalNormalizedArray(array=col_data))

    @staticmethod
    def validate_column_shape_and_nulls(
        rows: Sequence[Sequence[Union[str, int, float]]]
    ) -> np.ndarray:
        """
        Validates that the input is a 2D array-like structure and contains no None values.

        Returns
        -------
        np.ndarray
            The validated array with dtype=object.

        Raises
        ------
        ValueError
            If input is not 2D or contains None values.
        """
        arr = np.array(rows, dtype=object)
        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array-like structure")
        if np.any(arr == None):  # pylint: disable=singleton-comparison
            raise ValueError("Input contains None values, which are not allowed")
        return arr

    @staticmethod
    def infer_column_types(arr: np.ndarray) -> List[ColumnType]:
        """
        Infers column types from a validated 2D ndarray.

        Parameters
        ----------
        arr : np.ndarray
            A validated 2D NumPy array with dtype=object.

        Returns
        -------
        List[ColumnType]
            The inferred column types.

        Raises
        ------
        ValueError
            If a column contains mixed or unsupported types.
        """
        n_cols = arr.shape[1]
        column_types: List[ColumnType] = []

        for col_index in range(n_cols):
            col_data = arr[:, col_index]
            types = {type(x) for x in col_data}

            if types.issubset({int, float}):
                column_types.append(ColumnType.NUMERIC)
            elif types.issubset({str}):
                column_types.append(ColumnType.CATEGORICAL)
            else:
                raise ValueError(
                    f"Column {col_index} contains mixed or unsupported types: {types}"
                )

        return column_types

    @property
    def num_columns(self) -> int:
        """
        Returns the number of columns in the matrix.

        Returns
        -------
        int
            The number of columns.
        """
        return len(self._columns)

    @property
    def num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        Returns
        -------
        int
            The number of rows.
        """
        if not self._columns:
            return 0
        return len(self._columns[0].array)

    def get_normalized_array(self, column_index: int) -> NDArray[np.float64]:
        """
        Get the normalized array for a specific column.

        Parameters
        ----------
        column_index : int
            The index of the column to retrieve.

        Returns
        -------
        NDArray[np.float64]
            The normalized array for the specified column.
        """
        return self._columns[column_index]._normalizer.array_normalized  # pylint: disable=protected-access

    def get_column_type(self, column_index: int) -> ColumnType:
        """
        Retrieve the column type for a specified column index.

        Parameters
        ----------
        column_index : int
            The index of the column whose type is to be retrieved.

        Returns
        -------
        ColumnType
            The type of the specified column (numeric or categorical).
        """
        column_instance = self._columns[column_index]
        if isinstance(column_instance, NumericNormalizedArray):
            return ColumnType.NUMERIC
        elif isinstance(column_instance, CategoricalNormalizedArray):
            return ColumnType.CATEGORICAL
        else:
            raise ValueError("Unknown column type")

    def get_numeric_array(self, column_index: int) -> Sequence[Union[int, float]]:
        """
        Get the original numeric array (before normalization) for the specified column.

        Parameters
        ----------
        column_index : int
            The index of the column to retrieve.

        Returns
        -------
        Sequence[Union[int, float]]
            The original numeric values from the column.

        Raises
        ------
        TypeError
            If the column is not of numeric type.
        """
        column_instance = self._columns[column_index]
        if not isinstance(column_instance, NumericNormalizedArray):
            raise TypeError(f"Column {column_index} is not of numeric type")
        return column_instance.array

    def get_categorical_array(self, column_index: int) -> Sequence[str]:
        """
        Get the original categorical array (before normalization) for the specified column.

        Parameters
        ----------
        column_index : int
            The index of the column to retrieve.

        Returns
        -------
        Sequence[str]
            The original categorical values from the column.

        Raises
        ------
        TypeError
            If the column is not of categorical type.
        """
        column_instance = self._columns[column_index]
        if not isinstance(column_instance, CategoricalNormalizedArray):
            raise TypeError(f"Column {column_index} is not of categorical type")
        return column_instance.array

    def append_data(self, new_rows: Sequence[Sequence[Union[str, int, float]]]) -> None:
        """
        Append one or more new rows to the matrix.

        Raises
        ------
        ValueError
            If dimensions are invalid or None values are present.
        TypeError
            If any value doesn't match expected column type.
        """
        if not new_rows:
            return
        arr = self.validate_column_shape_and_nulls(new_rows)

        if arr.shape[1] != self.num_columns:
            raise ValueError(
                f"Expected {self.num_columns} columns, got {arr.shape[1]}"
            )

        for col_index, col_data in enumerate(arr.T):
            column = self._columns[col_index]
            if isinstance(column, NumericNormalizedArray):
                if not all(isinstance(x, (int, float)) for x in col_data):
                    raise TypeError(f"Column {col_index} expects numeric values")
            elif isinstance(column, CategoricalNormalizedArray):
                if not all(isinstance(x, str) for x in col_data):
                    raise TypeError(f"Column {col_index} expects string values")
            else:
                raise TypeError(f"Unknown column type at index {col_index}")

        for i, column in enumerate(self._columns):
            column.append_array(arr[:, i].tolist())
