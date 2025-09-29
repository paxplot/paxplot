"""Array manager values structure for PaxPlot.

This module defines the ArrayManager class for managing collections of
NumericalArray and CategoricalArray objects.
"""

from typing import List, Sequence, Union
from enum import Enum

from .arrays.numerical_array import NumericalArray
from .arrays.categorical_array import CategoricalArray


class ArrayType(str, Enum):
    """Represents the type of an array in the array manager.

    Parameters
    ----------
    NUMERIC : str
        Array contains numerical values.
    CATEGORICAL : str
        Array contains categorical (string) values.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class ArrayManager:
    """
    An array manager for managing collections of numerical and categorical arrays.

    This class provides a unified interface for managing multiple arrays
    of values, where each array can be either numerical or categorical.
    The array manager ensures that all arrays have the same number of rows.

    Parameters
    ----------
    arrays : Sequence[Union[NumericalArray, CategoricalArray]], optional
        Pre-existing arrays to use. If None, creates empty array manager.

    Attributes
    ----------
    arrays : List[Union[NumericalArray, CategoricalArray]]
        The stored arrays.
    num_arrays : int
        The number of arrays in the array manager.
    num_rows : int
        The number of rows in the array manager.

    Examples
    --------
    >>> # Initialize with existing arrays
    >>> numeric_array = NumericalArray([1, 2, 3])
    >>> categorical_array = CategoricalArray(['A', 'B', 'A'])
    >>> array_manager = ArrayManager([numeric_array, categorical_array])
    >>> array_manager.num_arrays
    2
    >>> array_manager.num_rows
    3
    >>> array_manager.get_numeric_array(0).get_values()
    [1.0, 2.0, 3.0]
    >>> array_manager.get_categorical_array(1).get_values()
    ['A', 'B', 'A']
    >>>
    >>> # Or initialize empty and set from types
    >>> array_manager = ArrayManager()  # Empty
    >>> array_manager.set_arrays_from_types([ArrayType.NUMERIC, ArrayType.CATEGORICAL])
    """

    def __init__(
        self,
        arrays: Union[Sequence[Union[NumericalArray, CategoricalArray]], None] = None,
    ):
        # Initialize with empty arrays first, then use set_arrays method
        self._arrays = []
        self.set_arrays(arrays)

    def get_array(self, index: int) -> Union[NumericalArray, CategoricalArray]:
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
                f"Array index {index} out of bounds for array manager with "
                f"{len(self._arrays)} arrays"
            )
        return self._arrays[index]

    def get_numeric_array(self, index: int) -> NumericalArray:
        """Get a numerical array at the specified index.

        Parameters
        ----------
        index : int
            The index of the array to get.

        Returns
        -------
        NumericalArray
            The numerical array at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the array at the specified index is not numerical.
        """
        if self.get_array_type(index) != ArrayType.NUMERIC:
            raise TypeError(
                f"Array {index} is not numerical, it is "
                f"{type(self._arrays[index]).__name__}"
            )
        array = self.get_array(index)
        assert isinstance(array, NumericalArray)
        return array

    def get_categorical_array(self, index: int) -> CategoricalArray:
        """Get a categorical array at the specified index.

        Parameters
        ----------
        index : int
            The index of the array to get.

        Returns
        -------
        CategoricalArray
            The categorical array at the specified index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        TypeError
            If the array at the specified index is not categorical.
        """
        if self.get_array_type(index) != ArrayType.CATEGORICAL:
            raise TypeError(
                f"Array {index} is not categorical, it is "
                f"{type(self._arrays[index]).__name__}"
            )
        array = self.get_array(index)
        assert isinstance(array, CategoricalArray)
        return array

    def get_array_type(self, index: int) -> ArrayType:
        """Get the type of an array at the specified index.

        Parameters
        ----------
        index : int
            The index of the array to get.

        Returns
        -------
        ArrayType
            The type of the array.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self._arrays):
            raise IndexError(
                f"Array index {index} out of bounds for array manager with "
                f"{len(self._arrays)} arrays"
            )

        if isinstance(self._arrays[index], NumericalArray):
            return ArrayType.NUMERIC
        return ArrayType.CATEGORICAL

    def set_arrays(
        self,
        arrays: Union[Sequence[Union[NumericalArray, CategoricalArray]], None] = None,
    ) -> None:
        """Set new arrays, replacing all existing arrays.

        Parameters
        ----------
        arrays : Sequence[Union[NumericalArray, CategoricalArray]], optional
            The new arrays to set. If None, creates empty array manager.

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

    def append_arrays(self, array: Union[NumericalArray, CategoricalArray]) -> None:
        """Append a new array to the array manager.

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
                raise ValueError(f"Index must be an integer, got {type(index)}")
            if index < 0 or index >= len(self._arrays):
                raise IndexError(
                    f"Index {index} out of bounds for array manager with "
                    f"{len(self._arrays)} arrays"
                )
            del self._arrays[index]

    def clear_arrays(self) -> None:
        """Clear all arrays from the array manager."""
        self.set_arrays([])

    def set_arrays_from_types(self, array_types: Sequence[ArrayType]) -> None:
        """Set new arrays from array types, creating empty arrays.

        Parameters
        ----------
        array_types : Sequence[ArrayType]
            The types of arrays to create.

        Raises
        ------
        ValueError
            If array_types is empty.
        TypeError
            If any element in array_types is not an ArrayType.
        """
        if not array_types:
            raise ValueError("array_types cannot be empty")

        # Validate array types
        for i, array_type in enumerate(array_types):
            if not isinstance(array_type, ArrayType):
                raise TypeError(
                    f"array_type at index {i} must be an ArrayType, "
                    f"got {type(array_type)}"
                )

        # Create empty arrays based on types
        new_arrays = []
        for array_type in array_types:
            if array_type == ArrayType.NUMERIC:
                new_arrays.append(NumericalArray())
            elif array_type == ArrayType.CATEGORICAL:
                new_arrays.append(CategoricalArray())
            else:
                raise ValueError(f"Unknown array type: {array_type}")

        self._arrays = new_arrays

    def append_values(self, row: Sequence[Union[str, int, float]]) -> None:
        """Append a new row of values to the array manager.

        Parameters
        ----------
        row : Sequence[Union[str, int, float]]
            The row of values to append.

        Raises
        ------
        ValueError
            If the row length doesn't match the number of arrays.
        """
        if len(row) != self.num_arrays:
            raise ValueError(
                f"Row length {len(row)} doesn't match number of arrays "
                f"{self.num_arrays}"
            )

        # Let the arrays handle all validation and NaN checking
        for array, value in zip(self._arrays, row):
            array.append_values([value])  # type: ignore

    def remove_values(self, indices: Sequence[int]) -> None:
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

    def set_values(self, values: Sequence[Sequence[Union[str, int, float]]]) -> None:
        """Set new values for the array manager, replacing all existing values.

        Parameters
        ----------
        values : Sequence[Sequence[Union[str, int, float]]]
            The new values as a 2D sequence where each row is a sequence
            of values and each array should be consistently typed.

        Raises
        ------
        ValueError
            If the values structure is invalid.
        """
        self._validate_values(values)
        self._arrays = self._create_arrays(values)

    def clear_values(self) -> None:
        """Clear all values from the arrays but keep the array objects."""
        for array in self._arrays:
            array.clear_values()

    def _validate_values(
        self, values: Sequence[Sequence[Union[str, int, float]]]
    ) -> None:
        """Validate the input values structure.

        Parameters
        ----------
        values : Sequence[Sequence[Union[str, int, float]]]
            The values to validate.

        Raises
        ------
        ValueError
            If the values structure is invalid.
        """
        if not values:
            raise ValueError("Values cannot be empty")

        if not isinstance(values[0], (list, tuple)):
            raise ValueError("Values must be a 2D sequence")

        row_length = len(values[0])
        for i, row in enumerate(values):
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"Row {i} must be a sequence")
            if len(row) != row_length:
                raise ValueError(
                    f"Row {i} has length {len(row)} but expected {row_length}"
                )

    def _create_arrays(
        self, values: Sequence[Sequence[Union[str, int, float]]]
    ) -> List[Union[NumericalArray, CategoricalArray]]:
        """Create arrays from the input values.

        Parameters
        ----------
        values : Sequence[Sequence[Union[str, int, float]]]
            The input values.

        Returns
        -------
        List[Union[NumericalArray, CategoricalArray]]
            The created arrays.
        """
        if not values:
            return []

        num_arrays = len(values[0])
        arrays = []

        for array_idx in range(num_arrays):
            array_values = [row[array_idx] for row in values]
            array_type = self.infer_array_type(array_values)

            if array_type == ArrayType.NUMERIC:
                arrays.append(NumericalArray(array_values))  # type: ignore
            else:
                arrays.append(CategoricalArray(array_values))  # type: ignore

        return arrays

    def infer_array_type(self, array_values: List[Union[str, int, float]]) -> ArrayType:
        """Infer the type of an array based on its values.

        Parameters
        ----------
        array_values : List[Union[str, int, float]]
            The values in the array.

        Returns
        -------
        ArrayType
            The inferred array type.

        Raises
        ------
        ValueError
            If the array contains mixed types.
        """
        has_numeric = False
        has_categorical = False

        for value in array_values:
            if value is None or (isinstance(value, float) and str(value) == "nan"):
                # Skip NaN values in type inference
                continue
            if isinstance(value, (int, float)):
                has_numeric = True
            elif isinstance(value, str):
                has_categorical = True
            else:
                raise ValueError(f"Array contains unsupported type: {type(value)}")

        if has_numeric and has_categorical:
            raise ValueError("Array contains mixed numeric and categorical values")
        if has_numeric:
            return ArrayType.NUMERIC
        return ArrayType.CATEGORICAL

    def __len__(self) -> int:
        """Get the number of arrays in the array manager.

        Returns
        -------
        int
            The number of arrays.
        """
        return len(self._arrays)

    def __getitem__(self, index: int) -> Union[NumericalArray, CategoricalArray]:
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
        """Get a string representation of the array manager.

        Returns
        -------
        str
            A string representation showing the array manager dimensions.
        """
        return f"ArrayManager({self.num_arrays} arrays)"

    @property
    def num_arrays(self) -> int:
        """Get the number of arrays in the array manager.

        Returns
        -------
        int
            The number of arrays.
        """
        return len(self._arrays)

    @property
    def arrays(self) -> List[Union[NumericalArray, CategoricalArray]]:
        """_summary_

        Returns
        -------
        List[Union[NumericalArray, CategoricalArray]]
            _description_
        """
        return self._arrays.copy()
