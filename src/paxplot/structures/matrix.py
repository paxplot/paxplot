"""Matrix data structure for PaxPlot.

This module defines the Matrix class for managing collections of
NumericalArray and CategoricalArray objects.
"""

from typing import List, Sequence, Union
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

    This class provides a unified interface for managing multiple arrays
    of data, where each array can be either numerical or categorical.
    The matrix ensures that all arrays have the same number of rows.

    Parameters
    ----------
    arrays : Sequence[Union[NumericalArray, CategoricalArray]], optional
        Pre-existing arrays to use. If None, creates empty matrix.

    Attributes
    ----------
    arrays : List[Union[NumericalArray, CategoricalArray]]
        The stored arrays.
    num_arrays : int
        The number of arrays in the matrix.
    num_rows : int
        The number of rows in the matrix.

    Examples
    --------
    >>> # Initialize with existing arrays
    >>> numeric_array = NumericalArray([1, 2, 3])
    >>> categorical_array = CategoricalArray(['A', 'B', 'A'])
    >>> matrix = Matrix([numeric_array, categorical_array])
    >>> matrix.num_arrays
    2
    >>> matrix.num_rows
    3
    >>> matrix.get_numeric_array(0).get_values()
    [1.0, 2.0, 3.0]
    >>> matrix.get_categorical_array(1).get_values()
    ['A', 'B', 'A']
    >>>
    >>> # Or initialize empty and set from types
    >>> matrix = Matrix()  # Empty
    >>> matrix.set_arrays_from_types([ColumnType.NUMERIC, ColumnType.CATEGORICAL])
    """

    def __init__(
        self, 
        arrays: Union[Sequence[Union[NumericalArray, CategoricalArray]], None] = None
    ):
        # Initialize with empty arrays first, then use set_arrays method
        self._arrays = []
        self.set_arrays(arrays)

    @property
    def arrays(self) -> List[Union[NumericalArray, CategoricalArray]]:
        """Get the arrays as a list.

        Returns
        -------
        List[Union[NumericalArray, CategoricalArray]]
            The arrays as a list.
        """
        return self._arrays.copy()

    @property
    def num_arrays(self) -> int:
        """Get the number of arrays in the matrix.

        Returns
        -------
        int
            The number of arrays.
        """
        return len(self._arrays)


    def get_array(
        self, index: int
    ) -> Union[NumericalArray, CategoricalArray]:
        """Get an array at the specified index.

        Parameters
        ----------
        index : int
            The index of the array to get.

        Returns
        -------
        Union[NumericalArray, CategoricalArray]
            The array at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._arrays):
            raise IndexError(
                f"Array index {index} out of bounds for matrix with {len(self._arrays)} arrays"
            )
        return self._arrays[index]

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
        if self.get_array_type(index) != ColumnType.NUMERIC:
            raise TypeError(
                f"Array {index} is not numerical, it is {type(self._arrays[index]).__name__}"
            )
        array = self.get_array(index)
        assert isinstance(array, NumericalArray)
        return array

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
        if self.get_array_type(index) != ColumnType.CATEGORICAL:
            raise TypeError(
                f"Array {index} is not categorical, it is {type(self._arrays[index]).__name__}"
            )
        array = self.get_array(index)
        assert isinstance(array, CategoricalArray)
        return array

    def get_array_type(self, index: int) -> ColumnType:
        """Get the type of an array at the specified index.

        Parameters
        ----------
        index : int
            The index of the array to get.

        Returns
        -------
        ColumnType
            The type of the array.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._arrays):
            raise IndexError(
                f"Array index {index} out of bounds for matrix with {len(self._arrays)} arrays"
            )

        if isinstance(self._arrays[index], NumericalArray):
            return ColumnType.NUMERIC
        return ColumnType.CATEGORICAL

    def get_arrays(self) -> List[Union[NumericalArray, CategoricalArray]]:
        """Get all arrays in the matrix.

        Returns
        -------
        List[Union[NumericalArray, CategoricalArray]]
            A copy of all arrays.
        """
        return self._arrays.copy()

    def set_arrays(
        self, 
        arrays: Union[Sequence[Union[NumericalArray, CategoricalArray]], None] = None
    ) -> None:
        """Set new arrays, replacing all existing arrays.

        Parameters
        ----------
        arrays : Sequence[Union[NumericalArray, CategoricalArray]], optional
            The new arrays to set. If None, creates empty matrix.

        Raises
        ------
        TypeError
            If any array is not a valid array type.
        """
        if arrays is None:
            arrays = []
        
        # Validate arrays
        for i, array in enumerate(arrays):
            if not isinstance(array, (NumericalArray, CategoricalArray)):
                raise TypeError(
                    f"Array at index {i} must be NumericalArray or CategoricalArray, "
                    f"got {type(array)}"
                )
        
        self._arrays = list(arrays)

    def append_arrays(
        self, 
        array: Union[NumericalArray, CategoricalArray]
    ) -> None:
        """Append a new array to the matrix.

        Parameters
        ----------
        array : Union[NumericalArray, CategoricalArray]
            The array to append.

        Raises
        ------
        TypeError
            If the array is not a valid array type.
        """
        if not isinstance(array, (NumericalArray, CategoricalArray)):
            raise TypeError(
                f"Array must be NumericalArray or CategoricalArray, "
                f"got {type(array)}"
            )
        
        self._arrays.append(array)

    def remove_arrays(self, indices: Sequence[int]) -> None:
        """Remove arrays at the specified indices.

        Parameters
        ----------
        indices : Sequence[int]
            The indices of arrays to remove.

        Raises
        ------
        IndexError
            If any index is out of bounds.
        ValueError
            If indices are not valid integers.
        """
        # Convert to list and sort in reverse order to avoid index shifting
        indices_list = sorted(indices, reverse=True)
        
        for index in indices_list:
            if not isinstance(index, int):
                raise ValueError(
                    f"Index must be an integer, got {type(index)}"
                )
            if index < 0 or index >= len(self._arrays):
                raise IndexError(
                    f"Index {index} out of bounds for matrix with "
                    f"{len(self._arrays)} arrays"
                )
            del self._arrays[index]

    def clear_arrays(self) -> None:
        """Clear all arrays from the matrix."""
        self.set_arrays([])

    def set_arrays_from_types(
        self, 
        column_types: Sequence[ColumnType]
    ) -> None:
        """Set new arrays from column types, creating empty arrays.

        Parameters
        ----------
        column_types : Sequence[ColumnType]
            The types of arrays to create.

        Raises
        ------
        ValueError
            If column_types is empty.
        TypeError
            If any element in column_types is not a ColumnType.
        """
        if not column_types:
            raise ValueError("column_types cannot be empty")
        
        # Validate column types
        for i, column_type in enumerate(column_types):
            if not isinstance(column_type, ColumnType):
                raise TypeError(
                    f"column_type at index {i} must be a ColumnType, got {type(column_type)}"
                )
        
        # Create empty arrays based on types
        new_arrays = []
        for column_type in column_types:
            if column_type == ColumnType.NUMERIC:
                new_arrays.append(NumericalArray())
            elif column_type == ColumnType.CATEGORICAL:
                new_arrays.append(CategoricalArray())
            else:
                raise ValueError(f"Unknown column type: {column_type}")
        
        self._arrays = new_arrays

    def append_data(self, row: Sequence[Union[str, int, float]]) -> None:
        """Append a new row of data to the matrix.

        Parameters
        ----------
        row : Sequence[Union[str, int, float]]
            The row of data to append.

        Raises
        ------
        ValueError
            If the row length doesn't match the number of arrays.
        """
        if len(row) != self.num_arrays:
            raise ValueError(
                f"Row length {len(row)} doesn't match number of arrays {self.num_arrays}"
            )

        # Let the arrays handle all validation and NaN checking
        for array, value in zip(self._arrays, row):
            array.append_values([value])  # type: ignore

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
        for array in self._arrays:
            array.remove_values(indices)

    def set_data(self, data: Sequence[Sequence[Union[str, int, float]]]) -> None:
        """Set new data for the matrix, replacing all existing data.

        Parameters
        ----------
        data : Sequence[Sequence[Union[str, int, float]]]
            The new data as a 2D sequence where each row is a sequence
            of values and each column should be consistently typed.

        Raises
        ------
        ValueError
            If the data structure is invalid.
        """
        self._validate_data(data)
        self._arrays = self._create_arrays(data)


    def clear_data(self) -> None:
        """Clear all data from the arrays but keep the array objects."""
        for array in self._arrays:
            array.clear_values()

    def _validate_data(
        self, data: Sequence[Sequence[Union[str, int, float]]]
    ) -> None:
        """Validate the input data structure.

        Parameters
        ----------
        data : Sequence[Sequence[Union[str, int, float]]]
            The data to validate.

        Raises
        ------
        ValueError
            If the data structure is invalid.
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

    def _create_arrays(
        self, data: Sequence[Sequence[Union[str, int, float]]]
    ) -> List[Union[NumericalArray, CategoricalArray]]:
        """Create arrays from the input data.

        Parameters
        ----------
        data : Sequence[Sequence[Union[str, int, float]]]
            The input data.

        Returns
        -------
        List[Union[NumericalArray, CategoricalArray]]
            The created arrays.
        """
        if not data:
            return []

        num_arrays = len(data[0])
        arrays = []

        for array_idx in range(num_arrays):
            array_data = [row[array_idx] for row in data]
            array_type = self.infer_column_type(array_data)

            if array_type == ColumnType.NUMERIC:
                arrays.append(NumericalArray(array_data))  # type: ignore
            else:
                arrays.append(CategoricalArray(array_data))  # type: ignore

        return arrays

    def infer_column_type(
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
            if value is None or (
                isinstance(value, float) and str(value) == "nan"
            ):
                # Skip NaN values in type inference
                continue
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
        """Get the number of arrays in the matrix.

        Returns
        -------
        int
            The number of arrays.
        """
        return len(self._arrays)

    def __getitem__(
        self, index: int
    ) -> Union[NumericalArray, CategoricalArray]:
        """Get an array at the specified index.

        Parameters
        ----------
        index : int
            The index of the array to get.

        Returns
        -------
        Union[NumericalArray, CategoricalArray]
            The array at the specified index.
        """
        return self.get_array(index)

    def __repr__(self) -> str:
        """Get a string representation of the matrix.

        Returns
        -------
        str
            A string representation showing the matrix dimensions.
        """
        return f"Matrix({self.num_arrays} arrays)"
