"""Numerical array data structure for PaxPlot.

This module defines the NumericalArray class for storing and managing
sequences of numerical values (floats and integers).
"""

from typing import List, Sequence, Union


class NumericalArray:
    """
    A simple array for storing numerical data.

    This class provides basic operations for storing and managing
    sequences of numerical values without any normalization logic.

    Parameters
    ----------
    values : Sequence[Union[float, int]]
        The initial numerical values to store.

    Attributes
    ----------
    values : List[float]
        The stored numerical values as a list of floats.

    Examples
    --------
    >>> array = NumericalArray([1, 2, 3, 4, 5])
    >>> array.values
    [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> array.append([6, 7])
    >>> array.values
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    """

    def __init__(self, values: Sequence[Union[float, int]]):
        self._values = self._validate_and_convert(values)

    @property
    def values(self) -> List[float]:
        """Get the stored numerical values.

        Returns
        -------
        List[float]
            The numerical values as a list of floats.
        """
        return self._values.copy()

    @property
    def min(self) -> float:
        """Get the minimum value in the array.

        Returns
        -------
        float
            The minimum value.

        Raises
        ------
        ValueError
            If the array is empty.
        """
        if not self._values:
            raise ValueError("Cannot compute min of empty array")
        return min(self._values)

    @property
    def max(self) -> float:
        """Get the maximum value in the array.

        Returns
        -------
        float
            The maximum value.

        Raises
        ------
        ValueError
            If the array is empty.
        """
        if not self._values:
            raise ValueError("Cannot compute max of empty array")
        return max(self._values)

    @property
    def length(self) -> int:
        """Get the number of values in the array.

        Returns
        -------
        int
            The number of values.
        """
        return len(self._values)

    def append(self, values: Sequence[Union[float, int]]) -> None:
        """Append new numerical values to the array.

        Parameters
        ----------
        values : Sequence[Union[float, int]]
            The numerical values to append.

        Raises
        ------
        ValueError
            If any value is not numerical or is None.
        """
        new_values = self._validate_and_convert(values)
        self._values.extend(new_values)

    def remove(self, indices: Sequence[int]) -> None:
        """Remove values at the specified indices.

        Parameters
        ----------
        indices : Sequence[int]
            The indices of values to remove.

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
            if index < 0 or index >= len(self._values):
                raise IndexError(
                    f"Index {index} out of bounds for array of length {len(self._values)}"
                )
            del self._values[index]

    def _validate_and_convert(
        self, values: Sequence[Union[float, int]]
    ) -> List[float]:
        """Validate and convert values to a list of floats.

        Parameters
        ----------
        values : Sequence[Union[float, int]]
            The values to validate and convert.

        Returns
        -------
        List[float]
            The validated values as a list of floats.

        Raises
        ------
        ValueError
            If any value is not numerical or is None.
        """
        if values is None:
            raise ValueError("Values cannot be None")

        converted_values = []
        for i, value in enumerate(values):
            if value is None:
                raise ValueError(f"Value at index {i} cannot be None")

            try:
                converted_values.append(float(value))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Value at index {i} must be numerical, got {type(value)}: {value}"
                ) from e

        return converted_values

    def __len__(self) -> int:
        """Get the number of values in the array.

        Returns
        -------
        int
            The number of values.
        """
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        """Get a value at the specified index.

        Parameters
        ----------
        index : int
            The index of the value to get.

        Returns
        -------
        float
            The value at the specified index.
        """
        return self._values[index]

    def __repr__(self) -> str:
        """Get a string representation of the array.

        Returns
        -------
        str
            A string representation showing the array length and first few values.
        """
        if len(self._values) <= 10:
            return f"NumericalArray({self._values})"
        return f"NumericalArray({self._values[:5]}...{self._values[-5:]})"
