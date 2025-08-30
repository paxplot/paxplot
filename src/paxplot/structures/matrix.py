"""Matrix data structure for PaxPlot.

This module defines the Matrix class for managing collections of
NumericalArray and CategoricalArray objects.
"""

from typing import List, Sequence, Union, cast
from enum import Enum

from .arrays.numerical_array import NumericalArray
from .arrays.categorical_array import CategoricalArray


class ColumnType(str, Enum):
    """Represents the type of a column in the matrix.

    Parameters
    ----------
    NUMERIC : str
        Column contains numerical data.
    CATEGORICAL : str
        Column contains categorical (string) data.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class Matrix:
    """
    A matrix for managing collections of numerical and categorical arrays.

    This class provides a unified interface for managing multiple columns
    of data, where each column can be either numerical or categorical.
    The matrix ensures that all columns have the same number of rows.

    Parameters
    ----------
    data : Sequence[Sequence[Union[str, int, float]]]
        The initial data as a 2D sequence where each row is a sequence
        of values and each column should be consistently typed.

    Attributes
    ----------
    columns : List[Union[NumericalArray, CategoricalArray]]
        The stored arrays, one for each column.
    num_columns : int
        The number of columns in the matrix.
    num_rows : int
        The number of rows in the matrix.

    Examples
    --------
    >>> data = [
    ...     [1, 'A', 2.5],
    ...     [2, 'B', 3.0],
    ...     [3, 'A', 1.5]
    ... ]
    >>> matrix = Matrix(data)
    >>> matrix.num_columns
    3
    >>> matrix.num_rows
    3
    >>> matrix.get_numeric_array(0).values
    [1.0, 2.0, 3.0]
    >>> matrix.get_categorical_array(1).values
    ['A', 'B', 'A']
    """

    def __init__(self, data: Sequence[Sequence[Union[str, int, float]]]):
        self._validate_data(data)
        self._columns = self._create_columns(data)

    @property
    def columns(self) -> List[Union[NumericalArray, CategoricalArray]]:
        """Get the columns as a list of arrays.

        Returns
        -------
        List[Union[NumericalArray, CategoricalArray]]
            The columns as a list of arrays.
        """
        return self._columns.copy()

    @property
    def num_columns(self) -> int:
        """Get the number of columns in the matrix.

        Returns
        -------
        int
            The number of columns.
        """
        return len(self._columns)

    @property
    def num_rows(self) -> int:
        """Get the number of rows in the matrix.

        Returns
        -------
        int
            The number of rows.
        """
        if not self._columns:
            return 0
        return self._columns[0].length

    def get_column(
        self, index: int
    ) -> Union[NumericalArray, CategoricalArray]:
        """Get a column at the specified index.

        Parameters
        ----------
        index : int
            The index of the column to get.

        Returns
        -------
        Union[NumericalArray, CategoricalArray]
            The column at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._columns):
            raise IndexError(
                f"Column index {index} out of bounds for matrix with {len(self._columns)} columns"
            )
        return self._columns[index]

    def get_numeric_array(self, index: int) -> NumericalArray:
        """Get a numerical array at the specified index.

        Parameters
        ----------
        index : int
            The index of the column to get.

        Returns
        -------
        NumericalArray
            The numerical array at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the column at the specified index is not numerical.
        """
        if self.get_column_type(index) != ColumnType.NUMERIC:
            raise TypeError(
                f"Column {index} is not numerical, it is {type(self._columns[index]).__name__}"
            )
        column = self.get_column(index)
        assert isinstance(column, NumericalArray)
        return column

    def get_column_type(self, index: int) -> ColumnType:
        """Get the type of a column at the specified index.

        Parameters
        ----------
        index : int
            The index of the column to get.

        Returns
        -------
        ColumnType
            The type of the column.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._columns):
            raise IndexError(
                f"Column index {index} out of bounds for matrix with {len(self._columns)} columns"
            )

        if isinstance(self._columns[index], NumericalArray):
            return ColumnType.NUMERIC
        return ColumnType.CATEGORICAL

    def get_categorical_array(self, index: int) -> CategoricalArray:
        """Get a categorical array at the specified index.

        Parameters
        ----------
        index : int
            The index of the column to get.

        Returns
        -------
        CategoricalArray
            The categorical array at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the column at the specified index is not categorical.
        """
        if self.get_column_type(index) != ColumnType.CATEGORICAL:
            raise TypeError(
                f"Column {index} is not categorical, it is {type(self._columns[index]).__name__}"
            )
        column = self.get_column(index)
        assert isinstance(column, CategoricalArray)
        return column

    def append_data(self, row: Sequence[Union[str, int, float]]) -> None:
        """Append a new row of data to the matrix.

        Parameters
        ----------
        row : Sequence[Union[str, int, float]]
            The row of data to append.

        Raises
        ------
        ValueError
            If the row length doesn't match the number of columns.
        """
        if len(row) != self.num_columns:
            raise ValueError(
                f"Row length {len(row)} doesn't match number of columns {self.num_columns}"
            )

        for i, (column, value) in enumerate(zip(self._columns, row)):
            if isinstance(column, NumericalArray):
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Column {i} is numerical but value {value} is not numerical"
                    )
                column.append([value])
            if isinstance(column, CategoricalArray):
                if not isinstance(value, str):
                    raise ValueError(
                        f"Column {i} is categorical but value {value} is not a string"
                    )
                column.append([value])

    def remove_data(self, indices: Sequence[int]) -> None:
        """Remove rows at the specified indices.

        Parameters
        ----------
        indices : Sequence[int]
            The indices of rows to remove.

        Raises
        ------
        IndexError
            If any index is out of bounds.
        """
        for column in self._columns:
            column.remove(indices)

    def _validate_data(
        self, data: Sequence[Sequence[Union[str, int, float]]]
    ) -> None:
        """Validate the input data.

        Parameters
        ----------
        data : Sequence[Sequence[Union[str, int, float]]]
            The data to validate.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if not data:
            raise ValueError("Data cannot be empty")

        if not isinstance(data[0], (list, tuple)):
            raise ValueError("Data must be a 2D sequence")

        row_length = len(data[0])
        for i, row in enumerate(data):
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"Row {i} must be a sequence")
            if len(row) != row_length:
                raise ValueError(
                    f"Row {i} has length {len(row)} but expected {row_length}"
                )

    def _create_columns(
        self, data: Sequence[Sequence[Union[str, int, float]]]
    ) -> List[Union[NumericalArray, CategoricalArray]]:
        """Create columns from the input data.

        Parameters
        ----------
        data : Sequence[Sequence[Union[str, int, float]]]
            The input data.

        Returns
        -------
        List[Union[NumericalArray, CategoricalArray]]
            The created columns.
        """
        if not data:
            return []

        num_columns = len(data[0])
        columns = []

        for col_idx in range(num_columns):
            column_data = [row[col_idx] for row in data]
            column_type = self._infer_column_type(column_data)

            if column_type == ColumnType.NUMERIC:
                # Type checker doesn't know column_data is all numeric, so we cast it
                numeric_data = cast(List[Union[int, float]], column_data)
                columns.append(NumericalArray(numeric_data))
            else:
                # Type checker doesn't know column_data is all strings, so we cast it
                categorical_data = cast(List[str], column_data)
                columns.append(CategoricalArray(categorical_data))

        return columns

    def _infer_column_type(
        self, column_data: List[Union[str, int, float]]
    ) -> ColumnType:
        """Infer the type of a column based on its data.

        Parameters
        ----------
        column_data : List[Union[str, int, float]]
            The data in the column.

        Returns
        -------
        ColumnType
            The inferred column type.

        Raises
        ------
        ValueError
            If the column contains mixed types.
        """
        has_numeric = False
        has_categorical = False

        for value in column_data:
            if isinstance(value, (int, float)):
                has_numeric = True
            elif isinstance(value, str):
                has_categorical = True
            else:
                raise ValueError(
                    f"Column contains unsupported type: {type(value)}"
                )

        if has_numeric and has_categorical:
            raise ValueError(
                "Column contains mixed numeric and categorical data"
            )
        if has_numeric:
            return ColumnType.NUMERIC
        return ColumnType.CATEGORICAL

    def __len__(self) -> int:
        """Get the number of columns in the matrix.

        Returns
        -------
        int
            The number of columns.
        """
        return len(self._columns)

    def __getitem__(
        self, index: int
    ) -> Union[NumericalArray, CategoricalArray]:
        """Get a column at the specified index.

        Parameters
        ----------
        index : int
            The index of the column to get.

        Returns
        -------
        Union[NumericalArray, CategoricalArray]
            The column at the specified index.
        """
        return self.get_column(index)

    def __repr__(self) -> str:
        """Get a string representation of the matrix.

        Returns
        -------
        str
            A string representation showing the matrix dimensions.
        """
        return f"Matrix({self.num_rows} rows, {self.num_columns} columns)"
