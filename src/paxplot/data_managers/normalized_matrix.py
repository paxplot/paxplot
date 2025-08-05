"""
A matrix class that handles normalization of mixed numeric and categorical data.

This class provides functionality to store and manage a 2D data structure where each column
can be either numeric or categorical. Each column is independently normalized according to
its type. Numeric columns are scaled to a [0,1] range, while categorical columns are
encoded as normalized indicators.

Notes
-----
- Each column must contain consistently typed data (either all numeric or all categorical)
- Null values (None) are allowed in both numeric and categorical columns
- Column access is zero-indexed
"""

from typing import Sequence, Union, Tuple, List
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
    """

    data: Sequence[Sequence[Union[str, int, float, None]]]
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
    ) -> Sequence[Sequence[Union[str, int, float, None]]]:
        """
        Validate the input data structure.

        Parameters
        ----------
        v : Sequence[Sequence[Union[str, int, float, None]]]
            The input data to validate.

        Returns
        -------
        Sequence[Sequence[Union[str, int, float, None]]]
            The validated input data.

        Raises
        ------
        ValueError
            If the input is not a 2D array-like structure.
        """
        arr = np.array(v, dtype=object)
        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array-like structure")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        arr, col_types = self.validate_and_infer_column_types(self.data)

        self._columns = []
        for i, col_type in enumerate(col_types):
            col_data = arr[:, i].tolist()
            if col_type == ColumnType.NUMERIC:
                self._columns.append(NumericNormalizedArray(array=col_data))
            elif col_type == ColumnType.CATEGORICAL:
                self._columns.append(CategoricalNormalizedArray(array=col_data))

    @staticmethod
    def validate_and_infer_column_types(
        rows: Sequence[Sequence[Union[str, int, float, None]]]
    ) -> Tuple[np.ndarray, List[ColumnType]]:
        """
        Validates the input rows and infers column types.

        Parameters
        ----------
        rows : Sequence[Sequence[Union[str, int, float, None]]]
            The input tabular data in row-major form.

        Returns
        -------
        tuple
            - A NumPy ndarray (2D) with dtype=object.
            - A list of inferred column types: 'numeric' or 'categorical'.

        Raises
        ------
        ValueError
            If the input is not 2D or contains unsupported/mixed column types.
        """
        arr = np.array(rows, dtype=object)

        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array-like structure")

        n_cols = arr.shape[1]
        column_types: List[ColumnType] = []

        for col_index in range(n_cols):
            col_data = arr[:, col_index]
            types = {type(x) for x in col_data if x is not None}

            if types.issubset({int, float}):
                column_types.append(ColumnType.NUMERIC)
            elif types.issubset({str}):
                column_types.append(ColumnType.CATEGORICAL)
            else:
                raise ValueError(
                    f"Column {col_index} contains mixed or unsupported types: {types}"
                )

        return arr, column_types

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
        """Get the normalized array for a specific column.

        Parameters
        ----------
        column_index : int
            The index of the column to retrieve.

        Returns
        -------
        NDArray[np.float64]
            The normalized array for the specified column.
        """
        return self._columns[column_index]._normalizer.array_normalized # pylint: disable=protected-access

    def get_column_type(self, column_index: int) -> ColumnType:
        """Retrieve the column type for a specified column index.

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

    def get_numeric_array(self, column_index: int) -> Sequence[Union[int, float, None]]:
        """
        Get the original numeric array (before normalization) for the specified column.

        Parameters
        ----------
        column_index : int
            The index of the column to retrieve.

        Returns
        -------
        Sequence[Union[int, float, None]]
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

    def get_categorical_array(self, column_index: int) -> Sequence[Union[str, None]]:
        """
        Get the original categorical array (before normalization) for the specified column.

        Parameters
        ----------
        column_index : int
            The index of the column to retrieve.

        Returns
        -------
        Sequence[Union[str, None]]
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
